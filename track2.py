"""Track 2 features: gear-fault DSP ideas adapted to end-of-turn detection.

Reuses the from-scratch primitives in `dsp.py` (VMD, spectral kurtosis, sample
entropy, Hilbert envelope). All features are causal -- computed on the last
~0.75 s of audio strictly BEFORE the pause -- and language-agnostic.

Hypotheses (why a gear-signal toolkit transfers to a turn ending):
  * VMD harmonicity: a resolved turn ending is often creaky/breathy -> energy
    leaves the low quasi-periodic modes into the noise residual.
  * Spectral kurtosis: a final plosive/fricative release is IMPULSIVE (high SK);
    a sustained continuation is not.
  * Sample entropy: a settling voice is more REGULAR (low SampEn) than a voice
    that is mid-articulation.
  * Hilbert envelope: true ends TRAIL OFF (negative envelope slope, decaying
    modulation); mid-turn holds are abrupt cut-offs.
"""
import numpy as np
from scipy.signal import resample_poly

from features import load_wav
from dsp import vmd, spectral_kurtosis, sample_entropy, hilbert_envelope

WIN_S = 0.75            # causal analysis window before the pause
VMD_SR = 8000          # downsample target for VMD (pitch/formant band is <4 kHz)
VMD_K = 4

FEATURE_NAMES = [
    "vmd_cf_low_hz",       # centre freq of the lowest VMD mode (pitch band)
    "vmd_low_energy_frac", # energy fraction in the 2 lowest modes (harmonic)
    "vmd_top_mode_frac",   # energy fraction of the single strongest mode (tonality)
    "vmd_hi_energy_frac",  # energy fraction in the highest mode (aperiodic/fricative)
    "vmd_low_decay",       # low-mode energy: last-half / first-half (voicing decay)
    "vmd_residual_frac",   # unmodelled energy fraction (noise/breath)
    "sk_max",              # max spectral kurtosis (impulsiveness of the ending)
    "sk_mean",             # mean spectral kurtosis
    "sk_lowband",          # mean SK below 1 kHz
    "sk_centroid_hz",      # SK-weighted frequency centroid
    "sampen_env",          # sample entropy of the Hilbert envelope (regularity)
    "env_decay_slope",     # Hilbert-envelope linear slope (normalised; <0 = trail off)
    "env_mod_depth",       # envelope std / mean (modulation depth)
    "env_last_first",      # envelope last-third mean / first-third mean
    "env_skew",            # envelope skewness
]
N_FEATURES = len(FEATURE_NAMES)


def _slope(y):
    t = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0
    t -= t.mean()
    d = float((t * t).sum())
    return float((t * (y - y.mean())).sum() / d) if d > 1e-9 else 0.0


def dsp_features(x, sr, pause_start):
    """Fixed-length Track-2 feature vector from audio before `pause_start`."""
    end = int(round(pause_start * sr))
    seg = x[max(0, end - int(WIN_S * sr)):end].astype(float)
    if len(seg) < sr // 20:                       # <50 ms
        return np.zeros(N_FEATURES, dtype=np.float32)

    # ---- VMD on a downsampled window ----
    up, down = VMD_SR, sr
    g = np.gcd(up, down)
    seg_ds = resample_poly(seg, up // g, down // g) if sr != VMD_SR else seg
    seg_ds = seg_ds - seg_ds.mean()
    vmd_cf_low_hz = vmd_low_energy_frac = vmd_top_mode_frac = 0.0
    vmd_hi_energy_frac = vmd_low_decay = vmd_residual_frac = 0.0
    if len(seg_ds) >= 32:
        try:
            u, omega = vmd(seg_ds, K=VMD_K, alpha=2000.0, max_iter=120)
            e = (u ** 2).sum(axis=1)               # energy per mode (sorted by freq)
            etot = e.sum() + 1e-12
            vmd_cf_low_hz = float(omega[0] * VMD_SR)
            vmd_low_energy_frac = float(e[:2].sum() / etot)
            vmd_top_mode_frac = float(e.max() / etot)
            vmd_hi_energy_frac = float(e[-1] / etot)
            half = u.shape[1] // 2
            low = u[0]
            e1 = (low[:half] ** 2).sum() + 1e-12
            e2 = (low[half:] ** 2).sum()
            vmd_low_decay = float(e2 / e1)
            sig_e = (seg_ds ** 2).sum() + 1e-12
            vmd_residual_frac = float(max(0.0, 1.0 - etot / sig_e))
        except Exception:
            pass

    # ---- spectral kurtosis ----
    freqs, sk = spectral_kurtosis(seg, sr, nperseg=256, hop=128)
    sk = np.nan_to_num(sk)
    sk_max = float(sk.max()) if len(sk) else 0.0
    sk_mean = float(sk.mean()) if len(sk) else 0.0
    sk_lowband = float(sk[freqs < 1000].mean()) if (freqs < 1000).any() else 0.0
    skc = np.clip(sk, 0, None)
    sk_centroid_hz = float((freqs * skc).sum() / (skc.sum() + 1e-12))

    # ---- Hilbert envelope shape ----
    env = hilbert_envelope(seg)
    # smooth + downsample envelope for entropy/shape
    if len(env) > 200:
        k = len(env) // 100
        env_s = env[:len(env) // k * k].reshape(-1, k).mean(axis=1)
    else:
        env_s = env
    env_s = env_s / (env_s.max() + 1e-12)
    sampen_env = sample_entropy(env_s, m=2, r=0.2)
    env_decay_slope = _slope(env_s) * len(env_s)   # normalised total change
    env_mod_depth = float(env_s.std() / (env_s.mean() + 1e-12))
    third = max(1, len(env_s) // 3)
    env_last_first = float(env_s[-third:].mean() / (env_s[:third].mean() + 1e-12))
    m = env_s.mean(); s = env_s.std() + 1e-12
    env_skew = float(np.mean(((env_s - m) / s) ** 3))

    feats = np.array([
        vmd_cf_low_hz, vmd_low_energy_frac, vmd_top_mode_frac, vmd_hi_energy_frac,
        vmd_low_decay, vmd_residual_frac,
        sk_max, sk_mean, sk_lowband, sk_centroid_hz,
        sampen_env, env_decay_slope, env_mod_depth, env_last_first, env_skew,
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_dsp_matrix(data_dir, labels_rows=None):
    import csv, os
    if labels_rows is None:
        with open(os.path.join(data_dir, "labels.csv")) as f:
            labels_rows = list(csv.DictReader(f))
    cache, X, keys = {}, [], []
    for r in labels_rows:
        path = os.path.join(data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(dsp_features(x, sr, float(r["pause_start"])))
        keys.append((r["turn_id"], r["pause_index"]))
    return np.asarray(X, np.float32), keys
