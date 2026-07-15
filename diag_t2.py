import numpy as np
from track2 import FEATURE_NAMES
z1 = np.load("feat_cache.npz", allow_pickle=True)
z2 = np.load("dsp_cache.npz", allow_pickle=True)


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else float("nan")


print(f"{'track2 feature':22s} {'EN':>6s} {'HI':>6s} {'strength':>8s}")
rows = []
for i, nm in enumerate(FEATURE_NAMES):
    ae = auc(z1["english_y"], z2["english_X"][:, i])
    ah = auc(z1["hindi_y"], z2["hindi_X"][:, i])
    rows.append((nm, ae, ah, abs(ae-.5)+abs(ah-.5)))
for nm, ae, ah, s in sorted(rows, key=lambda r: -r[3]):
    print(f"{nm:22s} {ae:6.3f} {ah:6.3f} {s:8.3f}")
