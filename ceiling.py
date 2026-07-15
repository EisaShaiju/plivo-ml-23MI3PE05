"""Oracle ceiling + English-only vs mixed model check."""
import os, numpy as np
from train_model import load_all, make_model, _score_preds, LANGS
from sklearn.model_selection import GroupKFold

data = load_all("eot_handout_extracted/eot_handout/eot_data", cache="feat_cache.npz")

print("=== ORACLE ceiling (p_eot = true label) ===")
for lang in LANGS:
    d = data[lang]
    r = _score_preds(d["dir"], d["keys"], (d["y"] == 1).astype(float))
    print(f"  {lang:8s}: delay={r['latency']*1000:.0f}ms  cut={r['cutoff']*100:.1f}%  AUC={r['auc']:.3f}")

print("\n=== English OOF: mixed (has Hindi) vs English-only training ===")
for use_other in (True, False):
    for lang in LANGS:
        d = data[lang]; other = LANGS[1 - LANGS.index(lang)]
        oof = np.zeros(len(d["y"]))
        for tr, te in GroupKFold(5).split(d["X"], d["y"], d["groups"]):
            Xtr, ytr = d["X"][tr], d["y"][tr]
            if use_other:
                Xtr = np.vstack([Xtr, data[other]["X"]]); ytr = np.concatenate([ytr, data[other]["y"]])
            m = make_model("logreg"); m.fit(Xtr, ytr)
            oof[te] = m.predict_proba(d["X"][te])[:, 1]
        r = _score_preds(d["dir"], d["keys"], oof)
        tag = "mixed" if use_other else "own-lang-only"
        print(f"  {lang:8s} [{tag:14s}]: delay={r['latency']*1000:.0f}ms  AUC={r['auc']:.3f}")
