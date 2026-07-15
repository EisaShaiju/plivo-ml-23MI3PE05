"""Track 3 features: EMD / Hilbert-Huang Transform for end-of-turn detection.

VMD (Track 2) imposes a bandwidth prior; EMD is fully DATA-ADAPTIVE -- it lets
the signal define its own intrinsic modes. We decompose the last ~0.75 s before
the pause into IMFs, then use the Hilbert-Huang instantaneous frequency /
amplitude of the dominant mode as a decomposition-independent view of how the
voice behaves right before the silence.

Hypotheses:
  * The dominant IMF's instantaneous frequency ~ a pitch-like track; its slope
    into the pause captures terminal pitch movement without a pitch tracker.
  * Instantaneous-amplitude decay of the dominant IMF = trailing off.
  * Energy spreading into high-frequency IMFs / residual = creak / breath / a
    final fricative rather than a sustained continuation.
"""
import numpy as np
from scipy.signal import resample_poly

from features import load_wav
from dsp import emd, hilbert_huang

WIN_S = 0.75
EMD_SR = 8000

FEATURE_NAMES = [
    "emd_n_imf",           # number of intrinsic mode functions
    "emd_imf0_frac",       # energy fraction of the highest-freq IMF
    "emd_imf1_frac",       # energy fraction of the 2nd IMF
    "emd_resid_frac",      # energy fraction of the monotonic residual (trend)
    "emd_dom_if_hz",       # median instantaneous freq of the dominant IMF
    "emd_dom_if_slope",    # IF slope of dominant IMF into the pause (Hz/s)
    "emd_dom_if_final",    # final IF / median IF of the dominant IMF
    "emd_dom_amp_decay",   # dominant IMF inst-amplitude: last-half / first-half
    "emd_hht_centroid_hz", # amplitude-weighted mean IF over all IMFs (HHT centroid)
    "emd_hht_cent_decay",  # HHT centroid last-half / first-half
    "emd_dom_if_std",      # instantaneous-freq stability of the dominant IMF
]
N_FEATURES = len(FEATURE_NAMES)


def _slope(y, dt):
    n = len(y)
    if n < 2:
        return 0.0
    t = np.arange(n) * dt
    t -= t.mean()
    d = float((t * t).sum())
    return float((t * (y - y.mean())).sum() / d) if d > 1e-9 else 0.0


def emd_features(x, sr, pause_start):
    end = int(round(pause_start * sr))
    seg = x[max(0, end - int(WIN_S * sr)):end].astype(float)
    if len(seg) < sr // 20:
        return np.zeros(N_FEATURES, dtype=np.float32)
    g = np.gcd(EMD_SR, sr)
    seg = resample_poly(seg, EMD_SR // g, sr // g) if sr != EMD_SR else seg
    seg = seg - seg.mean()
    dt = 1.0 / EMD_SR

    imfs, resid = emd(seg, max_imf=6)
    if len(imfs) == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)
    e = (imfs ** 2).sum(axis=1)
    e_res = float((resid ** 2).sum())
    etot = float(e.sum()) + e_res + 1e-12

    n_imf = float(len(imfs))
    imf0_frac = float(e[0] / etot)
    imf1_frac = float(e[1] / etot) if len(e) > 1 else 0.0
    resid_frac = float(e_res / etot)

    dom = int(np.argmax(e))                          # dominant (highest-energy) IMF
    amp, ifr = hilbert_huang(imfs[dom], EMD_SR)
    ifr = np.clip(ifr, 0, EMD_SR / 2)
    # trim Hilbert edge transients
    m = max(1, len(ifr) // 20)
    ifr_c = ifr[m:-m] if len(ifr) > 2 * m else ifr
    amp_c = amp[m:-m] if len(amp) > 2 * m else amp
    dom_if_hz = float(np.median(ifr_c)) if len(ifr_c) else 0.0
    dom_if_slope = _slope(ifr_c, dt)
    dom_if_final = float(np.median(ifr_c[-max(1, len(ifr_c) // 5):]) / (dom_if_hz + 1e-6)) if len(ifr_c) else 0.0
    dom_if_std = float(np.std(ifr_c)) if len(ifr_c) else 0.0
    h = len(amp_c) // 2
    dom_amp_decay = float(amp_c[h:].mean() / (amp_c[:h].mean() + 1e-9)) if h > 0 else 0.0

    # Hilbert-Huang spectral centroid over all IMFs (amplitude-weighted IF)
    num = np.zeros(len(seg) - 1); den = np.zeros(len(seg) - 1)
    for k in range(len(imfs)):
        ak, fk = hilbert_huang(imfs[k], EMD_SR)
        L = min(len(num), len(fk))
        num[:L] += ak[:L] * np.clip(fk[:L], 0, EMD_SR / 2)
        den[:L] += ak[:L]
    cent = num / (den + 1e-9)
    hht_centroid_hz = float(np.median(cent)) if len(cent) else 0.0
    hc = len(cent) // 2
    hht_cent_decay = float(cent[hc:].mean() / (cent[:hc].mean() + 1e-9)) if hc > 0 else 0.0

    feats = np.array([
        n_imf, imf0_frac, imf1_frac, resid_frac,
        dom_if_hz, dom_if_slope, dom_if_final, dom_amp_decay,
        hht_centroid_hz, hht_cent_decay, dom_if_std,
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_emd_matrix(data_dir, labels_rows=None):
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
        X.append(emd_features(x, sr, float(r["pause_start"])))
        keys.append((r["turn_id"], r["pause_index"]))
    return np.asarray(X, np.float32), keys
