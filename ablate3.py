"""Full-stack ablation across Track 1 (prosody), Track 2 (VMD/SK/Hilbert),
Track 3 (EMD/HHT). Reports honest OOF-by-turn with the logreg+HGB blend."""
import numpy as np
from ablate import load_track1, build_track2, build_track3, oof, show
from track2 import FEATURE_NAMES as T2N

t1 = load_track1()
t2 = build_track2(t1)
t3 = build_track3(t1, rebuild=True)

# Track-2 winning subset (from exp_t2sel): spectral kurtosis + envelope + 2 VMD
T2_STRONG = ["sk_mean", "sk_lowband", "sk_max", "sk_centroid_hz",
             "env_skew", "env_mod_depth", "vmd_low_energy_frac", "vmd_residual_frac"]
idx = [T2N.index(c) for c in T2_STRONG]
t2s = {l: t2[l][:, idx] for l in ["english", "hindi"]}

print("=== Full-stack ablation (blend logreg+hgb, OOF-by-turn) ===")
show("T1", oof(t1, None, "hgb", blend=True))
show("T1 + T2(strong)", oof(t1, [t2s], "hgb", blend=True))
show("T1 + T3(EMD)", oof(t1, [t3], "hgb", blend=True))
show("T1 + T2 + T3", oof(t1, [t2s, t3], "hgb", blend=True))
