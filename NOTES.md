# NOTES

The model reads **turn-final prosody** from the audio strictly before each pause:
terminal pitch falling to the speaker's own floor (pYIN F0, normalised per-turn
so it transfers English→Hindi), final-syllable and energy decay, final voicing
and spectral tilt. On top of that it adds **DSP views** — VMD harmonicity,
**spectral kurtosis** (impulsiveness of a final consonant), Hilbert-envelope
shape, and EMD/Hilbert-Huang instantaneous frequency — plus two learned temporal
views, a small **ViT+Bi-LSTM** on GAF/MFCC inputs and a **reservoir (Echo State
Network)**, all fused by a stacked ensemble. Everything is causal (never uses
`pause_end` or audio after the pause) and trained from scratch on the provided
data only. Honest held-out (out-of-fold by turn) delay drops from the 1600 ms /
850 ms silence baseline to **~1176 ms English / ~687 ms Hindi** at ≤5% cutoffs.
It still fails on genuinely ambiguous cases — a few long hold pauses that carry
full falling-pitch "completion" prosody (mid-turn phrase boundaries that sound
final), and short true ends with little prosodic resolution — which pin the last
~100 ms of delay. Pitch works far better for Hindi than English, whose endings
resolve less prosodically, so English keeps more missed-end latency. With one
more day I would add a **causal GRU/streaming endpoint** view (per P2) and
**training-time augmentation** (noise, pitch/tempo jitter) to harden Hindi
generalisation, add per-utterance speaker embeddings for stronger normalisation,
and calibrate a light EMA smoothing of p_eot across a turn's successive pauses.
