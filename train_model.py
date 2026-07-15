"""Train + honestly evaluate the EOT model, then save it for predict.py.

Evaluation protocol (matches how we are graded: a model trained on the given
data, tested on UNSEEN turns, mostly Hindi):

  1. OOF-by-turn: GroupKFold over each language's turns. For a held-out fold of
     language L we train on the whole OTHER language plus L's remaining turns,
     then predict the fold. Concatenated out-of-fold p_eot are scored with the
     OFFICIAL score.py -> an honest "held-out turn" delay per language.
  2. Cross-lingual stress test: train on all English, predict all Hindi (and
     vice-versa) -- pure transfer, no target-language turns seen.

Then we refit on EVERYTHING and persist the pipeline to model.joblib.

    python train_model.py --data_root <dir with english/ hindi/>
"""
import argparse
import os
import tempfile

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

from featurelib import build_matrix, FEATURE_NAMES
import score as official  # official scorer, imported directly

LANGS = ["english", "hindi"]


def make_model(kind):
    if kind == "logreg":
        return Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0,
                                      class_weight="balanced")),
        ])
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=20,
            class_weight="balanced", random_state=0)
    raise ValueError(kind)


def _score_preds(data_dir, keys, p):
    """Write a temp predictions.csv and run the official scorer on it."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="") as f:
        f.write("turn_id,pause_index,p_eot\n")
        for (tid, pi), pv in zip(keys, p):
            f.write(f"{tid},{pi},{pv:.6f}\n")
    r = official.score(os.path.join(data_dir, "labels.csv"), path)
    os.remove(path)
    return r


def load_all(data_root, cache=None):
    data = {}
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        for lang in LANGS:
            data[lang] = dict(
                dir=os.path.join(data_root, lang),
                X=z[f"{lang}_X"], y=z[f"{lang}_y"],
                keys=[tuple(k) for k in z[f"{lang}_keys"]],
                groups=np.array(z[f"{lang}_groups"]))
        print("loaded features from cache", cache)
        return data
    for lang in LANGS:
        d = os.path.join(data_root, lang)
        X, y, keys, groups = build_matrix(d)
        data[lang] = dict(dir=d, X=X, y=y, keys=keys, groups=np.array(groups))
    return data


def oof_eval(data, kind, n_splits=5):
    """Out-of-fold-by-turn p_eot per language, scored officially."""
    results = {}
    for lang in LANGS:
        d = data[lang]
        other = LANGS[1 - LANGS.index(lang)]
        Xo, yo = data[other]["X"], data[other]["y"]
        oof = np.zeros(len(d["y"]))
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(d["X"], d["y"], d["groups"]):
            Xtr = np.vstack([d["X"][tr], Xo])
            ytr = np.concatenate([d["y"][tr], yo])
            m = make_model(kind)
            m.fit(Xtr, ytr)
            oof[te] = m.predict_proba(d["X"][te])[:, 1]
        results[lang] = _score_preds(d["dir"], d["keys"], oof)
    return results


def crosslingual(data, kind):
    """Train on one language, test on the other (pure transfer)."""
    out = {}
    for train_lang in LANGS:
        test_lang = LANGS[1 - LANGS.index(train_lang)]
        m = make_model(kind)
        m.fit(data[train_lang]["X"], data[train_lang]["y"])
        p = m.predict_proba(data[test_lang]["X"])[:, 1]
        out[f"{train_lang[:2]}->{test_lang[:2]}"] = _score_preds(
            data[test_lang]["dir"], data[test_lang]["keys"], p)
    return out


def _fmt(r):
    return (f"delay={r['latency']*1000:5.0f}ms  cut={r['cutoff']*100:4.1f}%  "
            f"AUC={r['auc']:.3f}  thr={r['threshold']}  d={r['delay']*1000:.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="eot_handout_extracted/eot_handout/eot_data")
    ap.add_argument("--kinds", default="logreg,hgb")
    ap.add_argument("--save", default="model.joblib")
    ap.add_argument("--save_kind", default="logreg")
    ap.add_argument("--cache", default="feat_cache.npz")
    args = ap.parse_args()

    data = load_all(args.data_root, cache=args.cache)
    print(f"features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    for lang in LANGS:
        y = data[lang]["y"]
        print(f"  {lang}: {len(y)} pauses  eot={int((y==1).sum())} hold={int((y==0).sum())}")

    for kind in [k for k in args.kinds.split(",") if k]:
        print(f"\n########## MODEL = {kind} ##########")
        print("--- OOF by turn (honest held-out; trains on other lang + rest) ---")
        oof = oof_eval(data, kind)
        for lang in LANGS:
            print(f"  {lang:8s}: {_fmt(oof[lang])}")
        print("--- cross-lingual transfer (no target-lang turns seen) ---")
        xl = crosslingual(data, kind)
        for k, r in xl.items():
            print(f"  {k}   : {_fmt(r)}")

    # refit chosen model(s) on EVERYTHING and persist as an ensemble list
    Xall = np.vstack([data[l]["X"] for l in LANGS])
    yall = np.concatenate([data[l]["y"] for l in LANGS])
    members = ["logreg", "hgb"] if args.save_kind == "blend" else [args.save_kind]
    models = []
    for k in members:
        m = make_model(k); m.fit(Xall, yall); models.append(m)
    joblib.dump({"models": models, "features": FEATURE_NAMES,
                 "kind": args.save_kind}, args.save)
    print(f"\nsaved '{args.save_kind}' ({len(models)} member(s)) "
          f"trained on {len(yall)} pauses -> {args.save}")


if __name__ == "__main__":
    main()
