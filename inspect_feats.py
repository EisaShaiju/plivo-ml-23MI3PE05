import numpy as np
from featurelib import FEATURE_NAMES
z = np.load("feat_cache.npz", allow_pickle=True)
for lang in ["english", "hindi"]:
    X = z[f"{lang}_X"]; y = z[f"{lang}_y"]
    eot = X[y == 1]; hold = X[y == 0]
    sd = X.std(0) + 1e-9
    diff = (eot.mean(0) - hold.mean(0)) / sd
    order = np.argsort(-np.abs(diff))
    print(f"\n===== {lang.upper()}: eot vs hold (norm diff, sorted) =====")
    print(f"{'feature':20s} {'eot':>8s} {'hold':>8s} {'ndiff':>7s}")
    for i in order:
        print(f"{FEATURE_NAMES[i]:20s} {eot.mean(0)[i]:8.3f} {hold.mean(0)[i]:8.3f} {diff[i]:7.3f}")
