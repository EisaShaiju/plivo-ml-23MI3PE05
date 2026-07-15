# EOT Project — Living Index

The single place to understand the whole project. (`RUNLOG.md` = score history;
`SUMMARY.html`/`NOTES.md` = final polished deliverables, written at the end.)

## What we're building
For every silence pause inside a user's turn, output `p_eot` = probability the
turn is over. Scored by the official `score.py` as **mean response delay (ms) at
≤5% interrupted turns** (lower is better). Baseline (silence timer): **1600 ms
English / 850 ms Hindi**. Hidden test = unseen turns, **mostly Hindi**.

Hard rules we obey: CPU-only, no cloud; libs = numpy/scipy/sklearn/pandas/
librosa/torch; **no pretrained weights**; **causal** features (audio `[0:pause_start]`
only — never `pause_end`).

## How to read / run everything
Environment: WSL Ubuntu venv `~/speedrun/env` (Python 3.12). Data lives in
`eot_handout_extracted/eot_handout/eot_data/{english,hindi}`.

```bash
# honest held-out evaluation (out-of-fold by turn + cross-lingual)
python train_model.py --cache feat_cache.npz          # trains + saves model.joblib
# rebuild feature cache after changing features (~2 min, runs pYIN)
python diag.py --rebuild                               # also prints per-feature AUC
# ship predictions for a folder (loads saved model; unseen-safe)
python predict.py --data_dir <folder> --out predictions.csv
# official score
python score.py  --data_dir <folder> --pred predictions.csv
```

## File map
| file | role |
|---|---|
| `featurelib.py` | **Track 1** causal prosodic features (pYIN pitch, energy, timing, spectral) |
| `dsp.py` | **Track 2** DSP core, from scratch: VMD, spectral kurtosis, sample entropy, Hilbert, GAF |
| `track2.py` | Track 2 feature extractor built on `dsp.py` *(in progress)* |
| `train_model.py` | honest OOF-by-turn + cross-lingual eval; trains & saves `model.joblib` |
| `predict.py` | **deliverable** — loads saved model, writes `turn_id,pause_index,p_eot` |
| `score.py` | official scorer (unchanged) |
| `diag.py` | builds feature cache, prints per-feature univariate AUC |
| `err.py` / `hindi_margin.py` | error analysis — which pauses break the score |
| `ceiling.py` | oracle ceiling (100 ms) + mixed-vs-own-language check |
| `model.joblib` | saved trained ensemble (our own weights) |
| `predictions_english.csv`, `predictions_hindi.csv` | shipped predictions |
| `RUNLOG.md` | every scoring run + what changed (graded deliverable) |
| `SUMMARY.html`, `NOTES.md` | final write-ups *(pending)* |

## Current honest results (out-of-fold by turn = predicts hidden set)
| | Baseline | **FINAL (3-way stack)** | Ceiling (oracle) |
|---|---|---|---|
| English | 1600 ms | **~1176 ms** (best blend 1135) | 100 ms |
| Hindi | 850 ms | **~687 ms** (AUC 0.80) | 100 ms |

⚠️ Running `score.py` on the shipped `predictions_*.csv` shows ~115/100 ms — that
is **in-sample** (model predicting data it trained on) and **not indicative** of
the hidden test. Always quote the OOF numbers above.

## Track status — ALL COMPLETE
- ✅ **Track 1 — prosody**: 29 causal features + logreg/HGB blend.
- ✅ **Track 2 — VMD DSP**: VMD + Spectral Kurtosis + Hilbert (HI AUC 0.69→0.77).
- ✅ **Track 3 — EMD / Hilbert-Huang**: data-adaptive decomposition (hybrid HI 797).
- ✅ **Track 4 — Deep**: ViT(GAF)+Bi-LSTM(MFCC) fusion from scratch (HI 717).
- ✅ **Track 5 — Reservoir/ESN** (paper P6): fixed reservoir + linear readout.
- ✅ **Final**: 3-way stack [classical .7 / deep .2 / esn .1] → EN 1176 / HI 687.
  Deployed: `train_final.py` saves `model_final.joblib` + `deep_final.pt`;
  `predict.py` runs the 3-way blend (verified on an unseen no-label folder).

## Deliverables (all 5 present)
`SUMMARY.html` · `predict.py` · `predictions_english.csv` + `predictions_hindi.csv`
· `RUNLOG.md` · `NOTES.md`.  New files this phase: `dsp.py`, `track2.py`,
`track3.py`, `deepnet.py`, `esn.py`, `featurebank.py`, `train_final.py`,
`stack.py`/`stack3.py`, `ablate*.py`.

## Key insights so far
1. Ceiling is 100 ms → the whole game is feature discrimination (AUC).
2. Two failure modes: **English** = true ends missed (weak pitch fall);
   **Hindi** = a few long holds ranked high (pins the 850 ms wall).
3. Pitch resolving to the speaker's own floor is the strongest single cue;
   speaker-relative normalisation is what makes English→Hindi transfer work.
4. Mixed-language training helps English and doesn't hurt Hindi → one shared model.
