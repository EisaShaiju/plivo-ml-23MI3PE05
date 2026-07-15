"""EOT predictor -- ships p_eot for every pause in a data folder.

    python predict.py --data_dir <folder> --out predictions.csv

Runs the FINAL 3-way stacked hybrid we trained ourselves on the provided data
(no pretrained/downloaded weights):
    p_eot = wc*classical + wd*deep + we*esn
      classical = logreg+HGB blend over the 48-dim causal feature bank
                  (Track1 prosody + Track2 VMD/Spectral-Kurtosis/Hilbert
                   + Track3 EMD/Hilbert-Huang)
      deep      = ViT(GAF image) + Bi-LSTM(MFCC seq) late-fusion net
      esn       = Echo State Network reservoir + linear readout (MFCC seq)
All features are causal (audio before each pause_start only). Works on any folder
with the task's layout; the `label` column is not required (unseen test set).
Output columns: turn_id, pause_index, p_eot.
"""
import argparse
import csv
import os

import numpy as np
import joblib

from features import load_wav
from featurebank import assemble_one, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=os.path.join(HERE, "model_final.joblib"))
    ap.add_argument("--deep", default=os.path.join(HERE, "deep_final.pt"))
    args = ap.parse_args()

    art = joblib.load(args.model)
    if art.get("features") != FEATURE_NAMES:
        raise SystemExit("feature/model mismatch -- re-run train_final.py")
    wts = art.get("weights", {"classical": 1.0, "deep": 0.0, "esn": 0.0})
    wc, wd, we = wts["classical"], wts["deep"], wts["esn"]

    # deep branch (graceful fallback: reweight onto classical if unavailable)
    net = None
    if wd > 0 and os.path.exists(args.deep):
        try:
            import torch
            from deepnet import HybridFusion
            net = HybridFusion(); net.load_state_dict(torch.load(args.deep)); net.eval()
        except Exception as e:
            print(f"[warn] deep branch unavailable ({e}); folding weight into classical")
            wc += wd; wd = 0.0; net = None
    need_seq = (net is not None) or (we > 0)   # ESN also needs the MFCC sequence

    with open(os.path.join(args.data_dir, "labels.csv")) as f:
        rows = list(csv.DictReader(f))

    wav = {}
    X, Ms, Gs, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in wav:
            wav[path] = load_wav(path)
        x, sr = wav[path]
        ps, pi = float(r["pause_start"]), int(r["pause_index"])
        X.append(assemble_one(x, sr, ps, pi))
        if need_seq:
            from deepnet import deep_inputs
            m, g = deep_inputs(x, sr, ps)
            Ms.append(m); Gs.append(g)
        keys.append((r["turn_id"], r["pause_index"]))
    X = np.asarray(X, np.float32)

    p = wc * np.mean([m.predict_proba(X)[:, 1] for m in art["models"]], axis=0)
    if net is not None:
        import torch
        with torch.no_grad():
            deep = torch.sigmoid(net(torch.tensor(np.asarray(Ms, np.float32)),
                                     torch.tensor(np.asarray(Gs, np.float32)))).numpy()
        p = p + wd * deep
    if we > 0 and "esn_readout" in art:
        S = art["esn_reservoir"].states(np.asarray(Ms, np.float32))
        p = p + we * art["esn_readout"].predict_proba(S)[:, 1]
    p = np.clip(p, 0.0, 1.0)

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pidx), pv in zip(keys, p):
            wr.writerow([tid, pidx, f"{pv:.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}  "
          f"(3-way stack c={wc:.2f} d={wd:.2f} e={we:.2f}, "
          f"deep={'on' if net is not None else 'off'})")


if __name__ == "__main__":
    main()
