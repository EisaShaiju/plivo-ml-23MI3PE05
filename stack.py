"""Stack the classical hybrid (T1+T2+T3) with the deep ViT+Bi-LSTM track.

Both prediction sets are honest OOF-by-turn, so blending their OOF probabilities
and scoring is leakage-free. We sweep the blend weight and report the official
delay for classical-only, deep-only and the best blend."""
import os, numpy as np
from sklearn.model_selection import GroupKFold
from ablate import load_track1, build_track2, build_track3
from track2 import FEATURE_NAMES as T2N
from train_model import make_model, _score_preds, LANGS

T2_KEEP = ["sk_mean", "sk_lowband", "sk_max", "sk_centroid_hz",
           "env_skew", "env_mod_depth", "vmd_low_energy_frac", "vmd_residual_frac"]


def classical_oof(t1, t2s, t3):
    preds = {}
    for lang in LANGS:
        d = t1[lang]; other = LANGS[1 - LANGS.index(lang)]
        def feat(l):
            return np.hstack([t1[l]["X"], t2s[l], t3[l]])
        X, Xo, yo = feat(lang), feat(other), t1[other]["y"]
        oof = np.zeros(len(d["y"]))
        for tr, te in GroupKFold(5).split(X, d["y"], d["groups"]):
            Xtr = np.vstack([X[tr], Xo]); ytr = np.concatenate([d["y"][tr], yo])
            p = np.zeros(len(te))
            for k in ("logreg", "hgb"):
                m = make_model(k); m.fit(Xtr, ytr); p += m.predict_proba(X[te])[:, 1]
            oof[te] = p / 2
        preds[lang] = (d["keys"], oof, d["dir"])
    return preds


def main():
    t1 = load_track1()
    t2 = build_track2(t1)
    t3 = build_track3(t1)
    idx = [T2N.index(c) for c in T2_KEEP]
    t2s = {l: t2[l][:, idx] for l in LANGS}

    clf = classical_oof(t1, t2s, t3)
    deep = np.load("deep_oof.npz", allow_pickle=True)

    out_store = {}
    print("=== Stacking classical hybrid (T1+T2+T3) + deep (ViT+Bi-LSTM) ===")
    for lang in LANGS:
        keys, pc, ddir = clf[lang]
        dkeys = [tuple(k) for k in deep[f"{lang}_keys"]]
        # align deep preds to classical key order
        dmap = {k: v for k, v in zip(dkeys, deep[f"{lang}_p"])}
        pd_ = np.array([dmap[k] for k in keys])
        rc = _score_preds(ddir, keys, pc)
        rd = _score_preds(ddir, keys, pd_)
        best = None
        for w in np.linspace(0, 1, 21):            # w on deep
            pb = (1 - w) * pc + w * pd_
            r = _score_preds(ddir, keys, pb)
            if best is None or r["latency"] < best[0]["latency"]:
                best = (r, w, pb)
        rb, wb, pb = best
        out_store[f"{lang}_keys"] = np.array(keys, object); out_store[f"{lang}_p"] = pb
        print(f"\n{lang.upper()}")
        print(f"  classical : delay={rc['latency']*1000:5.0f}ms  AUC={rc['auc']:.3f}")
        print(f"  deep      : delay={rd['latency']*1000:5.0f}ms  AUC={rd['auc']:.3f}")
        print(f"  BLEND w*={wb:.2f} deep : delay={rb['latency']*1000:5.0f}ms  AUC={rb['auc']:.3f}  <== best")
    np.savez("stack_oof.npz", **out_store)
    print("\nsaved blended OOF -> stack_oof.npz")


if __name__ == "__main__":
    main()
