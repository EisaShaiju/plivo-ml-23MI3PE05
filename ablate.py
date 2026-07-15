"""Ablation harness: does a feature block earn its place on the OOF scorer?

Combines cached feature blocks (Track 1 prosodic, Track 2 DSP, Track 3 EMD, ...)
and reports honest out-of-fold-by-turn delay + AUC for chosen combinations.
"""
import argparse, os, numpy as np
from sklearn.model_selection import GroupKFold
from train_model import make_model, _score_preds, LANGS

DATA_ROOT = "eot_handout_extracted/eot_handout/eot_data"


def load_track1(cache="feat_cache.npz"):
    z = np.load(cache, allow_pickle=True)
    d = {}
    for l in LANGS:
        d[l] = dict(dir=os.path.join(DATA_ROOT, l), X=z[f"{l}_X"], y=z[f"{l}_y"],
                    keys=[tuple(k) for k in z[f"{l}_keys"]],
                    groups=np.array(z[f"{l}_groups"]))
    return d


def build_track2(track1, cache="dsp_cache.npz", rebuild=False):
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return {l: z[f"{l}_X"] for l in LANGS}
    from track2 import build_dsp_matrix
    store, out = {}, {}
    for l in LANGS:
        X, keys = build_dsp_matrix(os.path.join(DATA_ROOT, l))
        assert keys == track1[l]["keys"], f"key misalignment in {l}"
        out[l] = X; store[f"{l}_X"] = X
    np.savez(cache, **store)
    print("built + cached", cache)
    return out


def build_track3(track1, cache="emd_cache.npz", rebuild=False):
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return {l: z[f"{l}_X"] for l in LANGS}
    from track3 import build_emd_matrix
    store, out = {}, {}
    for l in LANGS:
        X, keys = build_emd_matrix(os.path.join(DATA_ROOT, l))
        assert keys == track1[l]["keys"], f"key misalignment in {l}"
        out[l] = X; store[f"{l}_X"] = X
    np.savez(cache, **store)
    print("built + cached", cache)
    return out


def oof(track1, blocks, kind, blend=False):
    """blocks: dict lang->X to hstack with track1 features (or None for track1 only)."""
    res = {}
    for lang in LANGS:
        d = track1[lang]; other = LANGS[1 - LANGS.index(lang)]
        def feat(l):
            base = track1[l]["X"]
            return np.hstack([base] + [b[l] for b in blocks]) if blocks else base
        X, Xo, yo = feat(lang), feat(other), track1[other]["y"]
        oofp = np.zeros(len(d["y"]))
        for tr, te in GroupKFold(5).split(X, d["y"], d["groups"]):
            Xtr = np.vstack([X[tr], Xo]); ytr = np.concatenate([d["y"][tr], yo])
            if blend:
                p = np.zeros(len(te))
                for k in ("logreg", "hgb"):
                    m = make_model(k); m.fit(Xtr, ytr); p += m.predict_proba(X[te])[:, 1]
                oofp[te] = p / 2
            else:
                m = make_model(kind); m.fit(Xtr, ytr); oofp[te] = m.predict_proba(X[te])[:, 1]
        res[lang] = _score_preds(d["dir"], d["keys"], oofp)
    return res


def show(tag, r):
    print(f"{tag:26s}  EN {r['english']['latency']*1000:5.0f}ms (AUC {r['english']['auc']:.3f})   "
          f"HI {r['hindi']['latency']*1000:5.0f}ms (AUC {r['hindi']['auc']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--kind", default="hgb")
    args = ap.parse_args()
    t1 = load_track1()
    t2 = build_track2(t1, rebuild=args.rebuild)
    # Track 2 alone: replace track1 X with t2 X
    t2only = {l: dict(t1[l], X=t2[l]) for l in LANGS}
    print(f"\n=== OOF ablation (model={args.kind}) ===")
    show("Track1 (prosody) only", oof(t1, None, args.kind))
    show("Track2 (VMD/SK/ent) only", oof(t2only, None, args.kind))
    show("Track1 + Track2", oof(t1, [t2], args.kind))
    print("--- with logreg+hgb blend ---")
    show("Track1 only [blend]", oof(t1, None, args.kind, blend=True))
    show("Track1 + Track2 [blend]", oof(t1, [t2], args.kind, blend=True))


if __name__ == "__main__":
    main()
