"""Cache features to disk and report univariate separation per feature."""
import argparse, os, numpy as np
from featurelib import build_matrix, FEATURE_NAMES

LANGS = ["english", "hindi"]


def auc(y, s):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    n1 = y.sum(); n0 = len(y)-n1
    if n1 == 0 or n0 == 0: return float("nan")
    return (ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="eot_handout_extracted/eot_handout/eot_data")
    ap.add_argument("--cache", default="feat_cache.npz")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.cache) and not args.rebuild:
        z = np.load(args.cache, allow_pickle=True)
        data = {l: dict(X=z[f"{l}_X"], y=z[f"{l}_y"]) for l in LANGS}
        print("loaded cache", args.cache)
    else:
        data = {}
        store = {}
        for l in LANGS:
            X, y, keys, groups = build_matrix(os.path.join(args.data_root, l))
            data[l] = dict(X=X, y=y)
            store[f"{l}_X"] = X; store[f"{l}_y"] = y
            store[f"{l}_keys"] = np.array(keys, object)
            store[f"{l}_groups"] = np.array(groups, object)
        np.savez(args.cache, **store)
        print("built + cached", args.cache)

    print(f"\n{'feature':22s} {'EN auc':>7s} {'HI auc':>7s} {'|EN-.5|+|HI-.5|':>14s}")
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        ae = auc(data["english"]["y"], data["english"]["X"][:, i])
        ah = auc(data["hindi"]["y"], data["hindi"]["X"][:, i])
        rows.append((name, ae, ah, abs(ae-.5)+abs(ah-.5)))
    for name, ae, ah, strength in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:22s} {ae:7.3f} {ah:7.3f} {strength:14.3f}")


if __name__ == "__main__":
    main()
