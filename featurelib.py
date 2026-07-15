"""Causal prosodic features for end-of-turn (EOT) detection.

CAUSALITY (hard rule): every feature for a pause at `pause_start` is computed
ONLY from audio[0 : pause_start]. We never read a single sample at or after the
pause, and we never use `pause_end`/`pause_index` (that is FUTURE info: a live
agent does not know a pause's duration, nor whether more pauses follow).

We chase turn-final PROSODY, made language- and speaker-robust by normalising
every "final" measurement against statistics of the SAME turn's speech so far
(also causal):

  * terminal F0 fall            -- statements resolve; pitch settles / drops
  * final-syllable lengthening  -- speakers stretch the last unit before ending
  * energy decay into the pause -- trailing off vs. an abrupt mid-thought stop
  * final voicing / spectral tilt -- creak & devoicing cluster at turn ends

Pitch is tracked with librosa's pYIN (a pure DSP algorithm -- autocorrelation /
YIN with an HMM smoother, NO trained weights), which is far cleaner than the
starter's frame-autocorrelation tracker. Dividing F0 by the turn's own
median/floor removes gender/language pitch offsets so an English-trained model
transfers to Hindi.
"""
import warnings

import numpy as np
import librosa

from features import load_wav, frames, frame_energy_db  # official utils

warnings.filterwarnings("ignore")

SR_DEFAULT = 16000
F0_WIN_S = 2.5           # window (before the pause) used for pitch + normalisation
F0_HOP = 160             # 10 ms at 16 kHz
F0_FRAME = 1024          # 64 ms analysis frame
FMIN, FMAX = 65.0, 400.0
E_HOP_S = 0.010          # energy hop (official frame_energy_db)

FEATURE_NAMES = [
    "ctx_speech_s",        # speech context available (log seconds)
    "elapsed_s",           # turn clock at the pause (causal)
    "energy_slope_300",    # energy dB slope over last 300 ms (decay<0)
    "energy_slope_600",    # energy dB slope over last 600 ms
    "energy_drop_db",      # final 150 ms level minus turn speech level
    "energy_tail_frac",    # final level vs turn peak (0..1 linear)
    "voiced_frac_tail",    # voicing fraction in last 400 ms
    "ends_voiced",         # is the last ~120 ms voiced (vowel- vs consonant-final)
    "f0_final_st",         # last voiced F0 vs turn median (semitones; fall<0)
    "f0_slope_st_300",     # F0 slope over last 300 ms voiced (st/s)
    "f0_slope_st_600",     # F0 slope over last 600 ms voiced (st/s)
    "f0_final_vs_floor",   # final F0 above the turn's own floor (semitones)
    "f0_frac_falling",     # fraction of final voiced frames that are descending
    "last_vseg_s",         # final voiced-run duration (final lengthening, abs)
    "last_vseg_ratio",     # final voiced-run / mean voiced-run (lengthening, rel)
    "vseg_rate_hz",        # voiced segments per second (speaking rate)
    "tilt_final_db",       # spectral tilt of last 300 ms (high/low band, dB)
    "centroid_final_khz",  # spectral centroid of last 300 ms (kHz)
    "n_voiced_ctx",        # voiced-frame count (confidence; log)
    "f0_range_tail_st",    # pitch range over last 600 ms (semitones; compresses at end)
    "f0_reset_st",         # final-run end vs its start (within-word contour; fall<0)
    "spectral_flux_tail",  # mean spectral change over last 300 ms (settling = low)
    "zcr_tail",            # zero-crossing rate over last 150 ms
    "energy_release_db",   # last 150 ms level minus the 150 ms before it (dB)
    "f0_slope_st_1000",    # F0 slope over last 1.0 s voiced (st/s; slow declination)
    "f0_decl_residual",    # final F0 vs the turn's own declination line (st; <0 = resolved)
    "eng_nucleus_final_s", # duration of the final loud (near-peak) region (s)
    "pos_index",           # pause ordinal so far (causal metadata, not audio)
    "pos_log",             # log1p(pause ordinal)
]
N_FEATURES = len(FEATURE_NAMES)


def _lin_slope(t, y):
    if len(t) < 2:
        return 0.0
    t = t - t.mean()
    d = float((t * t).sum())
    if d < 1e-9:
        return 0.0
    return float((t * (y - y.mean())).sum() / d)


def _runs(mask):
    """(start, end_excl) of maximal True runs in a boolean array."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def _spectral(seg, sr):
    fr = frames(seg, sr)
    if len(fr) == 0:
        return 0.0, 0.0
    win = np.hanning(fr.shape[1])[None, :]
    mag = np.abs(np.fft.rfft(fr * win, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(fr.shape[1], 1.0 / sr)
    centroid = float((mag * freqs).sum() / mag.sum())
    hi = mag[:, freqs >= 2000].sum()
    lo = mag[:, freqs < 2000].sum() + 1e-9
    tilt = float(20.0 * np.log10((hi + 1e-9) / lo))
    return centroid, tilt


def extract_features(x, sr, pause_start, pause_index=0):
    """Fixed-length feature vector from audio strictly before `pause_start`.

    `pause_index` is present-time metadata (how many times the user has paused
    so far) -- causal, not audio, and known to a live agent. It is NOT the
    forbidden `pause_end`."""
    end = int(round(pause_start * sr))
    prefix = x[:end]
    if len(prefix) < sr // 5:                        # <200 ms
        return np.zeros(N_FEATURES, dtype=np.float32)

    # ---- energy over the whole prefix (cheap) ----
    e_db = frame_energy_db(prefix, sr)
    if len(e_db) == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)
    e_db = np.maximum(e_db, -80.0)                   # clip silence floor (de-noise slopes)
    peak = float(np.max(e_db))
    speech_mask = e_db > (peak - 25.0)
    speech_lvl = float(np.median(e_db[speech_mask])) if speech_mask.any() else peak
    ctx_speech_s = float(speech_mask.sum()) * E_HOP_S

    def e_frames(ms):
        return e_db[-max(2, int(ms / 1000.0 / E_HOP_S)):]
    energy_slope_300 = _lin_slope(np.arange(len(e_frames(300))) * E_HOP_S, e_frames(300))
    energy_slope_600 = _lin_slope(np.arange(len(e_frames(600))) * E_HOP_S, e_frames(600))
    energy_tail = float(np.mean(e_frames(150)))
    energy_drop_db = energy_tail - speech_lvl
    energy_tail_frac = float(10 ** ((energy_tail - peak) / 20.0))

    # ---- pitch via pYIN on the last F0_WIN_S seconds ----
    fseg = prefix[-int(F0_WIN_S * sr):].astype(np.float32)
    try:
        f0, vflag, _ = librosa.pyin(fseg, fmin=FMIN, fmax=FMAX, sr=sr,
                                    frame_length=F0_FRAME, hop_length=F0_HOP,
                                    center=True)
    except Exception:
        f0 = np.full(1, np.nan); vflag = np.zeros(1, bool)
    voiced = np.nan_to_num(f0, nan=0.0)
    vmask = np.isfinite(f0) & (f0 > 0)
    fps = sr / F0_HOP                                # frames per second

    # voicing near the end
    def last_frames(sec):
        return max(1, int(sec * fps))
    voiced_frac_tail = float(vmask[-last_frames(0.4):].mean()) if len(vmask) else 0.0
    ends_voiced = float(vmask[-last_frames(0.12):].any()) if len(vmask) else 0.0

    # speaker-normalising F0 stats (this turn's own voice, causal)
    vox = f0[vmask]
    if len(vox) >= 3:
        f0_med = float(np.median(vox))
        f0_floor = float(np.percentile(vox, 10))
    else:
        f0_med = f0_floor = 0.0

    f0_final_st = f0_slope_st_300 = f0_slope_st_600 = 0.0
    f0_final_vs_floor = f0_frac_falling = 0.0
    last_vseg_s = last_vseg_ratio = 0.0
    vruns = _runs(vmask)
    vseg_rate_hz = len(vruns) / (len(prefix) / sr + 1e-6)
    if vruns and f0_med > 0:
        s, ee = vruns[-1]
        run = f0[s:ee]
        last_vseg_s = (ee - s) / fps
        lens = np.array([(b - a) for a, b in vruns], float) / fps
        last_vseg_ratio = last_vseg_s / (lens.mean() + 1e-6)
        st = 12.0 * np.log2(np.clip(voiced, 1e-6, None) / f0_med)     # semitones vs median
        f0_final_st = float(np.mean(st[s:ee][-int(max(1, 0.15 * fps)):]))
        f0_final_vs_floor = 12.0 * np.log2((np.mean(run[-3:]) + 1e-6) / (f0_floor + 1e-6))
        # slopes over the final voiced frames (concatenate tail voiced values)
        vt = np.where(vmask)[0]
        for span, out_name in ((0.3, "300"), (0.6, "600")):
            k = vt[vt >= len(vmask) - int(span * fps)]
            if len(k) >= 3:
                sl = _lin_slope(k / fps, st[k])
                if out_name == "300":
                    f0_slope_st_300 = sl
                else:
                    f0_slope_st_600 = sl
        # fraction of descending steps in the final ~0.4 s of voiced frames
        kf = vt[vt >= len(vmask) - int(0.4 * fps)]
        if len(kf) >= 3:
            d = np.diff(st[kf])
            f0_frac_falling = float((d < 0).mean())

    # extra pitch shape: range compression + within-word reset
    f0_range_tail_st = f0_reset_st = 0.0
    if f0_med > 0:
        kt = np.where(vmask)[0]
        kt = kt[kt >= len(vmask) - int(0.6 * fps)]
        if len(kt) >= 3:
            stv = 12.0 * np.log2(np.clip(f0[kt], 1e-6, None) / f0_med)
            f0_range_tail_st = float(np.percentile(stv, 90) - np.percentile(stv, 10))
        if vruns:
            s, ee = vruns[-1]
            run = f0[s:ee]
            if len(run) >= 4:
                f0_reset_st = 12.0 * np.log2((np.mean(run[-2:]) + 1e-6) /
                                             (np.mean(run[:2]) + 1e-6))

    tilt_final_db = centroid = 0.0
    fin = prefix[-int(0.3 * sr):]
    spectral_flux_tail = 0.0
    if len(fin) > 0:
        centroid, tilt_final_db = _spectral(fin, sr)
        ffr = frames(fin, sr)
        if len(ffr) >= 2:
            win = np.hanning(ffr.shape[1])[None, :]
            mag = np.abs(np.fft.rfft(ffr * win, axis=1))
            mag = mag / (mag.sum(axis=1, keepdims=True) + 1e-9)
            spectral_flux_tail = float(np.mean(np.sqrt(((np.diff(mag, axis=0)) ** 2).sum(1))))

    # zero-crossing rate of the last 150 ms (fricative/creak texture)
    tail150 = prefix[-int(0.15 * sr):]
    zcr_tail = float(np.mean(np.abs(np.diff(np.sign(tail150)))) / 2.0) if len(tail150) > 1 else 0.0

    # energy release shape: last 150 ms vs the 150 ms before it
    n150 = max(1, int(0.15 / E_HOP_S))
    if len(e_db) >= 2 * n150:
        energy_release_db = float(np.mean(e_db[-n150:]) - np.mean(e_db[-2 * n150:-n150]))
    else:
        energy_release_db = 0.0

    # slow declination: 1.0 s pitch slope + residual vs the turn's own F0 trend line
    f0_slope_st_1000 = f0_decl_residual = 0.0
    if f0_med > 0:
        vt = np.where(vmask)[0]
        stall = 12.0 * np.log2(np.clip(f0[vt], 1e-6, None) / f0_med) if len(vt) else np.array([])
        k1 = vt[vt >= len(vmask) - int(1.0 * fps)]
        if len(k1) >= 3:
            f0_slope_st_1000 = _lin_slope(k1 / fps, 12.0 * np.log2(np.clip(f0[k1], 1e-6, None) / f0_med))
        if len(vt) >= 5:
            tt = vt / fps
            A = np.vstack([tt, np.ones_like(tt)]).T
            coef, *_ = np.linalg.lstsq(A, stall, rcond=None)   # declination line
            pred_last = coef[0] * tt[-1] + coef[1]
            f0_decl_residual = float(stall[-1] - pred_last)     # <0 = final undershoots trend

    # final lengthening via energy: duration of the last near-peak "loud" region
    loud = e_db > (speech_lvl - 6.0)
    eng_nucleus_final_s = 0.0
    if loud.any() and loud[-1]:
        j = len(loud) - 1
        while j >= 0 and loud[j]:
            j -= 1
        eng_nucleus_final_s = (len(loud) - 1 - j) * E_HOP_S

    feats = np.array([
        np.log1p(ctx_speech_s),
        float(pause_start),
        energy_slope_300,
        energy_slope_600,
        energy_drop_db,
        energy_tail_frac,
        voiced_frac_tail,
        ends_voiced,
        f0_final_st,
        f0_slope_st_300,
        f0_slope_st_600,
        f0_final_vs_floor,
        f0_frac_falling,
        last_vseg_s,
        last_vseg_ratio,
        vseg_rate_hz,
        tilt_final_db,
        centroid / 1000.0,
        np.log1p(float(vmask.sum())),
        f0_range_tail_st,
        f0_reset_st,
        spectral_flux_tail,
        zcr_tail,
        energy_release_db,
        f0_slope_st_1000,
        f0_decl_residual,
        eng_nucleus_final_s,
        float(pause_index),
        np.log1p(float(pause_index)),
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_matrix(data_dir, labels_rows=None):
    """Extract features for every pause in `data_dir`. Returns X, y, keys, groups.

    y = 1 (eot) / 0 (hold) / -1 (label missing, e.g. unseen test set)."""
    import csv, os
    if labels_rows is None:
        with open(os.path.join(data_dir, "labels.csv")) as f:
            labels_rows = list(csv.DictReader(f))
    cache, X, y, keys, groups = {}, [], [], [], []
    for r in labels_rows:
        path = os.path.join(data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"]),
                                  pause_index=int(r["pause_index"])))
        lab = r.get("label", "")
        y.append(1 if lab == "eot" else (0 if lab == "hold" else -1))
        keys.append((r["turn_id"], r["pause_index"]))
        groups.append(r["turn_id"])
    return np.asarray(X, np.float32), np.asarray(y), keys, groups
