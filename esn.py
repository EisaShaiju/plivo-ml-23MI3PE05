"""Track 5: Reservoir Computing / Echo State Network (from P6).

A fixed, random recurrent 'reservoir' is driven left-to-right by the MFCC frame
sequence of the audio BEFORE the pause. Its state at the pause onset (plus the
mean state) is a rich temporal embedding -- and because the reservoir is fixed,
only a cheap LINEAR readout is trained. This is:
  * a genuinely different modelling paradigm from our GBDT / ViT+Bi-LSTM stack,
  * causal by construction (state read exactly at pause onset),
  * from scratch in numpy, no pretrained weights, trivially CPU-friendly.

Reuses the cached MFCC sequences from deep_cache.npz (no new extraction).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from train_model import _score_preds, LANGS
DATA_ROOT = "eot_handout_extracted/eot_handout/eot_data"


class Reservoir:
    """Leaky-integrator Echo State Network (fixed random weights)."""
    def __init__(self, n_in, n_res=220, spectral_radius=0.9, leak=0.3,
                 input_scale=0.6, density=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.leak = leak
        self.W_in = (rng.standard_normal((n_res, n_in)) * input_scale).astype(np.float32)
        W = rng.standard_normal((n_res, n_res)).astype(np.float32)
        mask = rng.random((n_res, n_res)) < density          # sparse recurrent
        W *= mask
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = (W * (spectral_radius / (eig + 1e-9))).astype(np.float32)
        self.n_res = n_res

    def states(self, M):
        """M: (B, T, n_in) -> concat(final_state, mean_state): (B, 2*n_res)."""
        B, T, _ = M.shape
        H = np.zeros((B, self.n_res), np.float32)
        acc = np.zeros((B, self.n_res), np.float32)
        for t in range(T):
            pre = M[:, t, :] @ self.W_in.T + H @ self.W.T
            H = (1 - self.leak) * H + self.leak * np.tanh(pre)
            acc += H
        return np.hstack([H, acc / T])


def make_readout():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, C=0.5,
                                               class_weight="balanced"))])


def esn_states_all(cache="deep_cache.npz", **kw):
    z = np.load(cache, allow_pickle=True)
    res = Reservoir(n_in=z["english_M"].shape[2], **kw)
    out = {}
    for l in LANGS:
        out[l] = dict(S=res.states(z[f"{l}_M"]), y=z[f"{l}_y"],
                      keys=[tuple(k) for k in z[f"{l}_keys"]],
                      groups=np.array(z[f"{l}_groups"]),
                      dir=f"{DATA_ROOT}/{l}")
    return out


def oof_esn(cache="deep_cache.npz"):
    data = esn_states_all(cache)
    results, preds = {}, {}
    for lang in LANGS:
        d = data[lang]; other = LANGS[1 - LANGS.index(lang)]
        So, yo = data[other]["S"], data[other]["y"]
        oof = np.zeros(len(d["y"]))
        for tr, te in GroupKFold(5).split(d["S"], d["y"], d["groups"]):
            Xtr = np.vstack([d["S"][tr], So]); ytr = np.concatenate([d["y"][tr], yo])
            m = make_readout(); m.fit(Xtr, ytr)
            oof[te] = m.predict_proba(d["S"][te])[:, 1]
        results[lang] = _score_preds(d["dir"], d["keys"], oof)
        preds[lang] = (d["keys"], oof)
    return results, preds


if __name__ == "__main__":
    res, preds = oof_esn()
    print("=== Reservoir Computing (Echo State Network) OOF-by-turn ===")
    for l in LANGS:
        r = res[l]
        print(f"  {l:8s}: delay={r['latency']*1000:5.0f}ms  cut={r['cutoff']*100:4.1f}%  AUC={r['auc']:.3f}")
    store = {}
    for l in LANGS:
        keys, oof = preds[l]
        store[f"{l}_keys"] = np.array(keys, object); store[f"{l}_p"] = oof
    np.savez("esn_oof.npz", **store)
    print("saved OOF -> esn_oof.npz")
