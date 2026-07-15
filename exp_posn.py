"""Quick experiment: does turn-position metadata (pause_index) help OOF?
Also tries a logreg+hgb blend. No re-extraction (appends to cached X)."""
import numpy as np
from train_model import load_all, make_model, _score_preds, LANGS
from sklearn.model_selection import GroupKFold

data = load_all("eot_handout_extracted/eot_handout/eot_data", cache="feat_cache.npz")


def with_pos(d):
    pidx = np.array([float(k[1]) for k in d["keys"]]).reshape(-1, 1)
    return np.hstack([d["X"], pidx, np.log1p(pidx)])


def oof(data, kind, add_pos=False, blend=False):
    out = {}
    for lang in LANGS:
        d = data[lang]; other = LANGS[1 - LANGS.index(lang)]
        X = with_pos(d) if add_pos else d["X"]
        Xo = with_pos(data[other]) if add_pos else data[other]["X"]
        yo = data[other]["y"]
        oofp = np.zeros(len(d["y"]))
        for tr, te in GroupKFold(5).split(X, d["y"], d["groups"]):
            Xtr = np.vstack([X[tr], Xo]); ytr = np.concatenate([d["y"][tr], yo])
            if blend:
                p = np.zeros(len(te))
                for k in ("logreg", "hgb"):
                    m = make_model(k); m.fit(Xtr, ytr); p += m.predict_proba(X[te])[:, 1]
                oofp[te] = p / 2
            else:
                m = make_model(kind); m.fit(Xtr, ytr)
                oofp[te] = m.predict_proba(X[te])[:, 1]
        out[lang] = _score_preds(d["dir"], d["keys"], oofp)
    return out


for tag, kw in [("hgb base", dict(kind="hgb")),
                ("hgb +pos", dict(kind="hgb", add_pos=True)),
                ("logreg +pos", dict(kind="logreg", add_pos=True)),
                ("blend base", dict(kind="x", blend=True)),
                ("blend +pos", dict(kind="x", blend=True, add_pos=True))]:
    r = oof(data, **kw)
    print(f"{tag:14s}  EN {r['english']['latency']*1000:5.0f}ms (AUC {r['english']['auc']:.3f})   "
          f"HI {r['hindi']['latency']*1000:5.0f}ms (AUC {r['hindi']['auc']:.3f})")
