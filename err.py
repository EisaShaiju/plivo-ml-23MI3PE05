"""Error analysis: where does the ranking break? Uses OOF-by-turn p_eot."""
import argparse, os, numpy as np, pandas as pd
from train_model import load_all, oof_eval, make_model, LANGS
from sklearn.model_selection import GroupKFold


def oof_preds(data, kind, n_splits=5):
    P = {}
    for lang in LANGS:
        d = data[lang]; other = LANGS[1 - LANGS.index(lang)]
        Xo, yo = data[other]["X"], data[other]["y"]
        oof = np.zeros(len(d["y"]))
        for tr, te in GroupKFold(n_splits).split(d["X"], d["y"], d["groups"]):
            m = make_model(kind)
            m.fit(np.vstack([d["X"][tr], Xo]), np.concatenate([d["y"][tr], yo]))
            oof[te] = m.predict_proba(d["X"][te])[:, 1]
        P[lang] = oof
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="eot_handout_extracted/eot_handout/eot_data")
    ap.add_argument("--cache", default="feat_cache.npz")
    ap.add_argument("--kind", default="logreg")
    args = ap.parse_args()
    data = load_all(args.data_root, cache=args.cache)
    res = oof_eval(data, args.kind)
    P = oof_preds(data, args.kind)
    for lang in LANGS:
        d = data[lang]
        lab = pd.read_csv(os.path.join(d["dir"], "labels.csv"))
        lab["dur"] = lab.pause_end - lab.pause_start
        lab["p"] = P[lang]
        r = res[lang]; thr, delay = r["threshold"], r["delay"]
        print(f"\n===== {lang.upper()}  (score delay={r['latency']*1000:.0f}ms "
              f"AUC={r['auc']:.3f} op: thr={thr} delay={delay*1000:.0f}ms) =====")
        holds = lab[lab.label == "hold"]; eots = lab[lab.label == "eot"]
        # false cutoffs at the operating point: fires AND delay<dur
        fc = holds[(holds.p >= thr) & (delay < holds.dur)]
        miss = eots[eots.p < thr]
        print(f"holds firing at op-point (dur>delay -> interrupt): {len(fc)}/{len(holds)}")
        print(f"eots missed at op-point (-> 1.6s timeout):         {len(miss)}/{len(eots)}")
        print(" TOP-8 holds by p_eot (dur, p):  <- dangerous if dur large")
        for _, r2 in holds.sort_values("p", ascending=False).head(8).iterrows():
            flag = "  <-- LONG (cut risk)" if r2.dur > 0.6 else ""
            print(f"    {r2.turn_id} pi{r2.pause_index}  dur={r2.dur:.2f}s  p={r2.p:.3f}{flag}")
        print(" BOTTOM-8 eots by p_eot (dur, p):  <- missed ends")
        for _, r2 in eots.sort_values("p").head(8).iterrows():
            print(f"    {r2.turn_id} pi{r2.pause_index}  dur={r2.dur:.2f}s  p={r2.p:.3f}")


if __name__ == "__main__":
    main()
