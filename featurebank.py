"""Production feature assembly for the classical hybrid model.

One place that concatenates the causal feature blocks used by the final model:
  Track 1 (featurelib)  : prosody  (pitch/energy/timing/spectral)  -- 29 feats
  Track 2 (track2)      : VMD + Spectral Kurtosis + Hilbert (winning subset)
  Track 3 (track3)      : EMD / Hilbert-Huang                       -- 11 feats

Used by both train_final.py and predict.py so training and inference always see
the exact same vector. Everything remains causal (audio before pause_start only).
"""
import csv
import os

import numpy as np

from features import load_wav
from featurelib import extract_features as t1_extract, FEATURE_NAMES as T1_NAMES
from track2 import dsp_features as t2_extract, FEATURE_NAMES as T2_NAMES
from track3 import emd_features as t3_extract, FEATURE_NAMES as T3_NAMES

# Track-2 subset that survived OOF ablation (spectral kurtosis + envelope + VMD)
T2_KEEP = ["sk_mean", "sk_lowband", "sk_max", "sk_centroid_hz",
           "env_skew", "env_mod_depth", "vmd_low_energy_frac", "vmd_residual_frac"]
_T2_IDX = [T2_NAMES.index(c) for c in T2_KEEP]

FEATURE_NAMES = (list(T1_NAMES)
                 + [f"t2_{c}" for c in T2_KEEP]
                 + [f"t3_{c}" for c in T3_NAMES])
N_FEATURES = len(FEATURE_NAMES)


def assemble_one(x, sr, pause_start, pause_index):
    t1 = t1_extract(x, sr, pause_start, pause_index=pause_index)
    t2 = t2_extract(x, sr, pause_start)[_T2_IDX]
    t3 = t3_extract(x, sr, pause_start)
    return np.concatenate([t1, t2, t3]).astype(np.float32)


def assemble(data_dir, labels_rows=None):
    """Return X, y, keys, groups for every pause in `data_dir`.
    y = 1 (eot) / 0 (hold) / -1 (label absent)."""
    if labels_rows is None:
        with open(os.path.join(data_dir, "labels.csv")) as f:
            labels_rows = list(csv.DictReader(f))
    cache, X, y, keys, groups = {}, [], [], [], []
    for r in labels_rows:
        path = os.path.join(data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(assemble_one(x, sr, float(r["pause_start"]), int(r["pause_index"])))
        lab = r.get("label", "")
        y.append(1 if lab == "eot" else (0 if lab == "hold" else -1))
        keys.append((r["turn_id"], r["pause_index"]))
        groups.append(r["turn_id"])
    return np.asarray(X, np.float32), np.asarray(y), keys, groups
