"""3-way stack: classical hybrid (T1+T2+T3) + deep (ViT+Bi-LSTM) + ESN (reservoir).

All three prediction sets are honest OOF-by-turn -> blending their OOF probs with
fixed weights is leakage-free. Grid-search the simplex for the best per-language
blend, and also report a single global weight (for deployment)."""
import numpy as np
from ablate import load_track1, build_track2, build_track3
from track2 import FEATURE_NAMES as T2N
from train_model import _score_preds, LANGS
from stack import classical_oof, T2_KEEP


def aligned(deep, esn, keys, lang):
    dm = {tuple(k): v for k, v in zip([tuple(k) for k in deep[f"{lang}_keys"]], deep[f"{lang}_p"])}
    em = {tuple(k): v for k, v in zip([tuple(k) for k in esn[f"{lang}_keys"]], esn[f"{lang}_p"])}
    return np.array([dm[k] for k in keys]), np.array([em[k] for k in keys])


def main():
    t1 = load_track1(); t2 = build_track2(t1); t3 = build_track3(t1)
    idx = [T2N.index(c) for c in T2_KEEP]
    t2s = {l: t2[l][:, idx] for l in LANGS}
    clf = classical_oof(t1, t2s, t3)
    deep = np.load("deep_oof.npz", allow_pickle=True)
    esn = np.load("esn_oof.npz", allow_pickle=True)

    grid = np.round(np.arange(0, 1.01, 0.1), 2)
    P = {}
    for lang in LANGS:
        keys, pc, ddir = clf[lang]
        pdp, pes = aligned(deep, esn, keys, lang)
        P[lang] = (keys, ddir, pc, pdp, pes)

    print("=== per-language best simplex blend (c=classical,d=deep,e=esn) ===")
    for lang in LANGS:
        keys, ddir, pc, pdp, pes = P[lang]
        best = None
        for wd in grid:
            for we in grid:
                if wd + we > 1.0 + 1e-9:
                    continue
                wc = 1 - wd - we
                p = wc * pc + wd * pdp + we * pes
                r = _score_preds(ddir, keys, p)
                if best is None or r["latency"] < best[0]["latency"]:
                    best = (r, wc, wd, we)
        r, wc, wd, we = best
        print(f"  {lang:8s}: delay={r['latency']*1000:5.0f}ms AUC={r['auc']:.3f}  "
              f"[c={wc:.1f} d={wd:.1f} e={we:.1f}]")

    print("\n=== single GLOBAL weight (same for both langs, for deployment) ===")
    best = None
    for wd in grid:
        for we in grid:
            if wd + we > 1.0 + 1e-9:
                continue
            wc = 1 - wd - we
            lat = []
            for lang in LANGS:
                keys, ddir, pc, pdp, pes = P[lang]
                r = _score_preds(ddir, keys, wc * pc + wd * pdp + we * pes)
                lat.append(r["latency"])
            avg = np.mean(lat)
            if best is None or avg < best[0]:
                best = (avg, wc, wd, we, lat)
    avg, wc, wd, we, lat = best
    print(f"  best global [c={wc:.1f} d={wd:.1f} e={we:.1f}]: "
          f"EN={lat[0]*1000:.0f}ms HI={lat[1]*1000:.0f}ms (avg {avg*1000:.0f}ms)")


if __name__ == "__main__":
    main()
