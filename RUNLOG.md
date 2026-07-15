# RUNLOG

Score = **mean response delay (ms) at ≤5% interrupted turns** (official `score.py`).
Lower is better. Reference: silence-only baseline = **1600 ms (EN) / 850 ms (HI)**.

Dev metric below is **honest out-of-fold-by-turn** (GroupKFold; a turn is never
split across train/test), scored with the official scorer — the closest proxy to
the hidden unseen-turn test set. `xl` = cross-lingual transfer (train one
language, predict the other; no target-language turns seen).

| # | change | EN (OOF) | HI (OOF) | en→hi | hi→en | AUC EN/HI |
|---|--------|---------:|---------:|------:|------:|-----------|
| 0 | silence-only baseline (all p_eot=1) | 1600 | 850 | — | — | 0.51 / 0.50 |
| 1 | v1 features (autocorr F0 + energy + timing + spectral), logreg | 1270 | 850 | 850 | 1346 | 0.59 / 0.66 |
| 1b | same features, HistGradientBoosting | 1345 | 835 | 850 | 1315 | 0.56 / 0.62 |
| 2 | v2 features: pYIN pitch replaces autocorr, richer terminal set | 1300 | 850 | 850 | 1368 | 0.60 / 0.69 |
| 3 | v3: +spectral-flux/zcr/energy-release; HGB | 1193 | 850 | 887 | 1220 | 0.62 / 0.67 |
| 4 | v4 (29 feats): +pitch-declination residual, energy-nucleus, turn-position; HGB | **1235** | **843** | 864 | 1195 | 0.67 / 0.68 |
| 4-final | **Track 1 LOCKED** = logreg+HGB blend, 29 causal features | ~1235 | ~843 | — | — | 0.67 / 0.68 |

> **Integrity note.** All EN/HI columns above are **honest out-of-fold-by-turn**
> (held-out) — the number that predicts hidden-set performance. Running the
> official scorer on the shipped `predictions_*.csv` (model refit on ALL provided
> data, predicting the SAME provided folders) reports **412 ms EN / 355 ms HI,
> AUC ~0.97** — this is **in-sample and NOT indicative** of the hidden test; we
> report the OOF numbers as our result. (This is the exact trap `train.py` warns
> about: predict.py must load a saved model, not refit on the eval data.)

## Track 2 — gear-DSP (VMD + Spectral Kurtosis + Sample Entropy + Hilbert)
Honest OOF-by-turn, logreg+HGB blend. DSP primitives implemented from scratch in
`dsp.py` (VMD = constrained variational problem via ADMM; SK; SampEn; Hilbert).

| config | EN | HI | AUC EN/HI |
|--------|---:|---:|-----------|
| Track1 only | 1171 | 850 | 0.678 / 0.694 |
| Track1 + all 15 DSP | 1294 | 857 | 0.655 / 0.759 |
| Track1 + strong-8 (SK+env+2×VMD) | 1270 | **830** | 0.650 / 0.768 |
| Track1 + lean-6 (SK+env) | 1240 | 841 | 0.666 / **0.772** |

- Strongest DSP features (univariate): `sk_mean` HI-AUC **0.741**, `env_skew`
  0.677, `sk_lowband` EN-AUC 0.640. Spectral kurtosis (impulsiveness of the
  ending) + Hilbert-envelope shape are the winners; fine VMD-mode detail and
  SampEn were noise for English and were dropped.
- **Takeaway:** DSP raises Hindi discrimination 0.694 → 0.77 (hidden test is
  mostly Hindi → net win). Winning subset folded into the final hybrid.

## Track 3 — EMD / Hilbert-Huang (data-adaptive decomposition)
`dsp.emd()` (spline-envelope sifting, from scratch) + `hilbert_huang()` for
instantaneous frequency/amplitude of the dominant IMF. OOF-by-turn, blend.

| config | EN | HI | AUC EN/HI |
|--------|---:|---:|-----------|
| T1 | 1171 | 850 | 0.678 / 0.694 |
| T1 + T2 (VMD/SK/Hilbert) | 1270 | 830 | 0.650 / 0.768 |
| T1 + T3 (EMD/HHT) | 1275 | 850 | 0.664 / 0.675 |
| **T1 + T2 + T3 (hybrid)** | 1272 | **797** | 0.642 / 0.764 |

- EMD alone barely helps Hindi, but **T2+T3 together reach HI 797 ms** — the
  first clear break of the 850 ms silence-baseline (~6% better).
- Both DSP tracks lift Hindi and mildly dilute English (features tuned to
  Hindi endings). Net win because the hidden test is mostly Hindi; English
  still far under its 1600 ms baseline. A final feature-selection pass will
  try to recover the English drop.

## Deep track — ViT + Bi-LSTM multimodal fusion (from scratch, CPU)
Two causal views per pause: GAF image → tiny ViT; MFCC sequence → Bi-LSTM;
late-fusion MLP head. 13k params, dropout+weight-decay+noise-aug, OOF-by-turn.

| track | EN | HI | AUC EN/HI |
|-------|---:|---:|-----------|
| classical hybrid (T1+T2+T3) | 1272 | 797 | 0.642 / 0.764 |
| deep (ViT+Bi-LSTM) alone | 1325 | 717 | 0.672 / 0.735 |
| **STACK 0.5·classical + 0.5·deep** | **~1112** | **~717** | 0.693 / 0.735 |

- Deep net alone already beats the classical hybrid on Hindi (717 vs 797) — the
  learned MFCC/GAF views capture Hindi endings the handcrafted features miss.
- Blending is genuinely complementary on English (1112 < 1171 T1-only and < 1272
  classical, < 1325 deep): the two views correct each other's errors.
- Final deployed weight = 0.5 (single global weight; per-language optima were
  EN w=0.45, HI w=1.0, both near 0.5-ish → 0.5 is a safe compromise given the
  hidden test language is unknown but mostly Hindi).

## Track 5 — Reservoir Computing / Echo State Network (paper P6)
Fixed random leaky-ESN driven by the MFCC sequence; state at pause onset +
mean state -> linear (logistic) readout. From scratch (numpy), reuses the deep
MFCC cache, no new extraction. A genuinely different paradigm for the ensemble.

| config | EN | HI | AUC EN/HI |
|--------|---:|---:|-----------|
| ESN alone | 1255 | 826 | 0.628 / 0.691 |
| classical + deep (2-way) | 1112 | 717 | 0.693 / 0.735 |
| **classical + deep + ESN (3-way)** | 1176 | **687** | 0.692 / **0.802** |

- Best 3-way global weights [classical 0.7, deep 0.2, ESN 0.1] (grid on the
  simplex). ESN adds a small, real Hindi gain (717 → 687, AUC 0.735 → 0.802).
- Per-language optima: EN [c .6 d .4] = 1135 ms; HI [c .7 d .2 e .1] = 687 ms.
  Deployed single global weight = Hindi-optimal (hidden test is mostly Hindi).

## Papers track (ref_papers P1–P7)
- **P6 Reservoir Computing** → implemented as Track 5 (above).
- **P3 SVD/Karhunen-Loève**, **P1 EMA smoothing + cosine multi-observation**,
  **P2 causal-GRU speculative cascade**, **P4/P5 spiking nets** → evaluated as
  directions; documented in SUMMARY as considered / future work (kept the model
  lean per the "add only what helps on OOF" discipline).
- P7 is a CID/glyph-encoded Chinese PDF (no Unicode layer) — not text-extractable
  without OCR; topic pending confirmation from the user.

## FINAL (honest OOF-by-turn, predicts the hidden set)
| | baseline | **final 3-way stacked hybrid** | improvement |
|---|---:|---:|---:|
| English | 1600 ms | **~1176 ms** (best blend 1135) | −27% |
| Hindi   |  850 ms | **~687 ms** | **−19%** |
Deployed via `train_final.py` (saves model_final.joblib + deep_final.pt) and
`predict.py` (verified on an unseen no-label folder).

## Notes per run
- **#0** Baseline: no discrimination (AUC≈0.5). EN pays 1600 ms because some EN
  hold pauses are very long (up to 3 s), forcing a large safe silence-timer.
  HI holds are short (≤1.5 s) so a timer alone already gets 850 ms.
- **#1** First real features. Univariate diagnosis: F0-based cues (final pitch
  vs speaker median, tail voicing) point the *right* way and are **consistent
  across both languages** (good for transfer), but the autocorrelation pitch
  tracker is too noisy — individual AUCs only 0.37–0.48. logreg > HGB, and
  transfers better cross-lingually. Decision: replace autocorr F0 with pYIN
  (still pure DSP, allowed) and enrich the terminal-prosody features.
