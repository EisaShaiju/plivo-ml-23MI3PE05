"""Why is Hindi pinned at 850ms? Show the achievable frontier + blocking pauses."""
import os, numpy as np, pandas as pd
from train_model import load_all, make_model, LANGS
from sklearn.model_selection import GroupKFold

data = load_all("eot_handout_extracted/eot_handout/eot_data", cache="feat_cache.npz")


def with_pos(d):
    pidx = np.array([float(k[1]) for k in d["keys"]]).reshape(-1, 1)
    return np.hstack([d["X"], pidx, np.log1p(pidx)])


def blend_oof(lang):
    d = data[lang]; other = LANGS[1 - LANGS.index(lang)]
    X = with_pos(d); Xo = with_pos(data[other]); yo = data[other]["y"]
    oof = np.zeros(len(d["y"]))
    for tr, te in GroupKFold(5).split(X, d["y"], d["groups"]):
        Xtr = np.vstack([X[tr], Xo]); ytr = np.concatenate([d["y"][tr], yo])
        p = np.zeros(len(te))
        for k in ("logreg", "hgb"):
            m = make_model(k); m.fit(Xtr, ytr); p += m.predict_proba(X[te])[:, 1]
        oof[te] = p / 2
    return oof


lang = "hindi"
d = data[lang]
lab = pd.read_csv(os.path.join(d["dir"], "labels.csv"))
lab["dur"] = lab.pause_end - lab.pause_start
lab["p"] = blend_oof(lang)
n_turns = lab.turn_id.nunique()
THR = np.round(np.arange(0.05, 1.0, 0.05), 3)
DEL = np.round(np.arange(0.10, 1.65, 0.05), 3)

print("achievable (min mean-delay) per max-delay cap, cutoff<=5%:")
best_overall = None
for cap in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
    best = None
    for t in THR:
        for dl in DEL:
            if dl > cap: continue
            cut = set(); lat = []
            for _, r in lab.iterrows():
                fires = r.p >= t
                if r.label == "hold":
                    if fires and dl < r.dur: cut.add(r.turn_id)
                else:
                    lat.append(dl if fires else 1.6)
            cr = len(cut)/n_turns; ml = np.mean(lat)
            if cr <= 0.05 and (best is None or ml < best[0]):
                best = (ml, t, dl, cr)
    if best: print(f"  cap {cap:.2f}s: mean_delay={best[0]*1000:5.0f}ms thr={best[1]:.2f} delay={best[2]*1000:.0f}ms cut={best[3]*100:.1f}%")

# blocking holds at delay 0.60 (want to use small delay): holds dur>0.6 sorted by p
print("\nHolds with dur>0.60s (block small delays), by p_eot desc:")
blk = lab[(lab.label=='hold') & (lab.dur>0.60)].sort_values('p', ascending=False)
for _, r in blk.head(12).iterrows():
    print(f"  {r.turn_id} pi{r.pause_index}  dur={r.dur:.2f}  p={r.p:.3f}")
print(f"  (total holds dur>0.60: {len(blk)}; >0.80: {(lab[(lab.label=='hold')].dur>0.80).sum()})")
print("\nEots ranked below p=0.5 (would be missed if thr raised):")
for _, r in lab[(lab.label=='eot')&(lab.p<0.5)].sort_values('p').head(10).iterrows():
    print(f"  {r.turn_id} pi{r.pause_index}  dur={r.dur:.2f}  p={r.p:.3f}")
