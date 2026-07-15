"""Deep track: multimodal ViT + Bi-LSTM hybrid fusion, from scratch (CPU).

Two causal views of the audio before each pause:
  * GAF image (Gramian Angular Field of the Hilbert envelope) -> a tiny Vision
    Transformer (patch-embed + 1 self-attention block).
  * MFCC sequence over the last ~0.9 s -> a Bi-LSTM.
Their embeddings are concatenated and fused by an MLP head (late fusion).

Trained from scratch on the provided data only (no pretrained weights), on CPU.
With ~500 pauses this is regularised hard (small dims, dropout, weight decay,
noise augmentation) and validated with the SAME honest OOF-by-turn protocol so
we can see if it earns a place in the ensemble. Expectation: a useful *view*,
not necessarily a winner alone -- its value is in the fusion.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from features import load_wav
from dsp import hilbert_envelope, gramian_angular_field
import librosa

torch.manual_seed(0)
np.random.seed(0)

WIN_S = 0.9
SEQ_LEN = 64          # MFCC frames
N_MFCC = 13
GAF = 32
LANGS = ["english", "hindi"]
DATA_ROOT = "eot_handout_extracted/eot_handout/eot_data"


# --------------------------- input extraction ------------------------------ #
def deep_inputs(x, sr, pause_start):
    end = int(round(pause_start * sr))
    seg = x[max(0, end - int(WIN_S * sr)):end].astype(np.float32)
    if len(seg) < sr // 20:
        return np.zeros((SEQ_LEN, N_MFCC), np.float32), np.zeros((GAF, GAF), np.float32)
    m = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=160)
    m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)
    m = m.T                                        # (frames, n_mfcc)
    if len(m) >= SEQ_LEN:
        m = m[-SEQ_LEN:]
    else:
        m = np.vstack([np.zeros((SEQ_LEN - len(m), N_MFCC), np.float32), m])
    env = hilbert_envelope(seg)
    g = gramian_angular_field(env, size=GAF, kind="summation")
    return m.astype(np.float32), g.astype(np.float32)


def build_deep_cache(cache="deep_cache.npz", rebuild=False):
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return z
    import csv
    store = {}
    for l in LANGS:
        d = os.path.join(DATA_ROOT, l)
        rows = list(csv.DictReader(open(os.path.join(d, "labels.csv"))))
        wcache, M, G, y, keys, groups = {}, [], [], [], [], []
        for r in rows:
            p = os.path.join(d, r["audio_file"])
            if p not in wcache:
                wcache[p] = load_wav(p)
            x, sr = wcache[p]
            mm, gg = deep_inputs(x, sr, float(r["pause_start"]))
            M.append(mm); G.append(gg)
            y.append(1 if r["label"] == "eot" else 0)
            keys.append((r["turn_id"], r["pause_index"])); groups.append(r["turn_id"])
        store[f"{l}_M"] = np.array(M); store[f"{l}_G"] = np.array(G)
        store[f"{l}_y"] = np.array(y)
        store[f"{l}_keys"] = np.array(keys, object); store[f"{l}_groups"] = np.array(groups, object)
    np.savez(cache, **store); print("built + cached", cache)
    return np.load(cache, allow_pickle=True)


# ------------------------------- the model --------------------------------- #
class TinyViT(nn.Module):
    """Patch-embed + 1 transformer block over a GAF image -> embedding."""
    def __init__(self, img=GAF, patch=8, dim=24, heads=2, emb=24):
        super().__init__()
        self.patch = patch
        self.np_ = (img // patch) ** 2
        self.proj = nn.Linear(patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.np_ + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.3,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(dim, emb)

    def forward(self, g):                            # g: (B, H, W)
        B, H, W = g.shape
        p = self.patch
        x = g.unfold(1, p, p).unfold(2, p, p).contiguous().view(B, -1, p * p)
        x = self.proj(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.enc(x)
        return self.head(x[:, 0])


class BiLSTMBranch(nn.Module):
    def __init__(self, n_in=N_MFCC, hid=16, emb=24):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(2 * hid, emb)

    def forward(self, m):                            # m: (B, T, n_in)
        out, _ = self.lstm(m)
        return self.head(self.drop(out[:, -1]))


class HybridFusion(nn.Module):
    def __init__(self, emb=24):
        super().__init__()
        self.vit = TinyViT(emb=emb)
        self.lstm = BiLSTMBranch(emb=emb)
        self.fuse = nn.Sequential(nn.Linear(2 * emb, 24), nn.GELU(), nn.Dropout(0.4),
                                  nn.Linear(24, 1))

    def forward(self, m, g):
        return self.fuse(torch.cat([self.lstm(m), self.vit(g)], dim=1)).squeeze(1)


# ------------------------------- training ---------------------------------- #
def _auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.5


def train_fold(Mtr, Gtr, ytr, Mva, Gva, yva, epochs=120, patience=18, aug=0.05):
    net = HybridFusion()
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3)
    Mtr_t, Gtr_t = torch.tensor(Mtr), torch.tensor(Gtr)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Mva_t, Gva_t = torch.tensor(Mva), torch.tensor(Gva)
    n = len(ytr); best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            mb = Mtr_t[idx] + aug * torch.randn_like(Mtr_t[idx])
            gb = Gtr_t[idx] + aug * torch.randn_like(Gtr_t[idx])
            opt.zero_grad()
            lossf(net(mb, gb), ytr_t[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Mva_t, Gva_t)).numpy()
        a = _auc(yva, pv)
        if a > best_auc:
            best_auc, bad = a, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Mva_t, Gva_t)).numpy()


def train_fold_all(M, G, y, epochs=55, aug=0.05):
    """Train on ALL data (no val split) for a fixed budget; return the net."""
    net = HybridFusion()
    pos_w = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3)
    Mt, Gt = torch.tensor(M), torch.tensor(G)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(y)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            mb = Mt[idx] + aug * torch.randn_like(Mt[idx])
            gb = Gt[idx] + aug * torch.randn_like(Gt[idx])
            opt.zero_grad()
            lossf(net(mb, gb), yt[idx]).backward(); opt.step()
    net.eval()
    return net


def oof_deep(cache="deep_cache.npz", rebuild=False, n_splits=5):
    """Honest OOF-by-turn deep-net p_eot per language, scored officially."""
    from sklearn.model_selection import GroupKFold
    import score as official
    import tempfile
    z = build_deep_cache(cache, rebuild=rebuild)
    results, preds = {}, {}
    for lang in LANGS:
        other = LANGS[1 - LANGS.index(lang)]
        M, G, y = z[f"{lang}_M"], z[f"{lang}_G"], z[f"{lang}_y"]
        groups = np.array(z[f"{lang}_groups"]); keys = [tuple(k) for k in z[f"{lang}_keys"]]
        Mo, Go, yo = z[f"{other}_M"], z[f"{other}_G"], z[f"{other}_y"]
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits).split(M, y, groups):
            Mtr = np.concatenate([M[tr], Mo]); Gtr = np.concatenate([G[tr], Go])
            ytr = np.concatenate([y[tr], yo])
            oof[te] = train_fold(Mtr, Gtr, ytr, M[te], G[te], y[te])
        preds[lang] = (keys, oof)
        # score officially
        fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
        with open(path, "w", newline="") as f:
            f.write("turn_id,pause_index,p_eot\n")
            for (tid, pi), pv in zip(keys, oof):
                f.write(f"{tid},{pi},{pv:.6f}\n")
        results[lang] = official.score(os.path.join(DATA_ROOT, lang, "labels.csv"), path)
        os.remove(path)
    return results, preds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--save_oof", default="deep_oof.npz")
    args = ap.parse_args()
    res, preds = oof_deep(rebuild=args.rebuild)
    print("\n=== Deep track (ViT+Bi-LSTM fusion) OOF-by-turn ===")
    for l in LANGS:
        r = res[l]
        print(f"  {l:8s}: delay={r['latency']*1000:5.0f}ms  cut={r['cutoff']*100:4.1f}%  AUC={r['auc']:.3f}")
    # persist OOF preds for stacking
    store = {}
    for l in LANGS:
        keys, oof = preds[l]
        store[f"{l}_keys"] = np.array(keys, object); store[f"{l}_p"] = oof
    np.savez(args.save_oof, **store)
    print("saved OOF preds ->", args.save_oof)
