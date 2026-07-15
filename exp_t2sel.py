"""Select the Track-2 features that actually help, re-ablate."""
import numpy as np
from ablate import load_track1, build_track2, oof, show
from track2 import FEATURE_NAMES

t1 = load_track1()
t2 = build_track2(t1)

# strong subset (from univariate AUC): spectral kurtosis + envelope + harmonic frac
STRONG = ["sk_mean", "sk_lowband", "sk_max", "sk_centroid_hz",
          "env_skew", "env_mod_depth", "vmd_low_energy_frac", "vmd_residual_frac"]
LEAN = ["sk_mean", "sk_lowband", "sk_max", "sk_centroid_hz", "env_skew", "env_mod_depth"]


def subset(cols):
    idx = [FEATURE_NAMES.index(c) for c in cols]
    return {l: t2[l][:, idx] for l in ["english", "hindi"]}


print("=== Track-2 feature selection (model=blend) ===")
show("T1 only", oof(t1, None, "hgb", blend=True))
show("T1 + T2(all 15)", oof(t1, [t2], "hgb", blend=True))
show("T1 + T2(strong 8)", oof(t1, [subset(STRONG)], "hgb", blend=True))
show("T1 + T2(lean 6=SK+env)", oof(t1, [subset(LEAN)], "hgb", blend=True))
