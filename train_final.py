"""Train the FINAL deployable stacked hybrid on ALL provided data, and save it.

Artifacts:
  model_final.joblib : classical members (logreg+hgb on the 48-dim feature bank),
                       feature names, and the deep-blend weight.
  deep_final.pt      : the ViT+Bi-LSTM state_dict (trained on all data).

predict.py loads both and outputs p = (1-w)*classical + w*deep.
"""
import numpy as np
import torch
import joblib

from featurebank import FEATURE_NAMES, T2_KEEP
from track2 import FEATURE_NAMES as T2N
from train_model import make_model, LANGS
from deepnet import HybridFusion, train_fold_all
from esn import Reservoir, make_readout

# 3-way stack weights (classical, deep, esn) from stack3.py global grid search
W_CLASSICAL, W_DEEP, W_ESN = 0.7, 0.2, 0.1


def assemble_from_caches():
    """Build the 48-dim feature-bank matrix for all data from the track caches."""
    z1 = np.load("feat_cache.npz", allow_pickle=True)
    z2 = np.load("dsp_cache.npz", allow_pickle=True)
    z3 = np.load("emd_cache.npz", allow_pickle=True)
    idx = [T2N.index(c) for c in T2_KEEP]
    X, y = [], []
    for l in LANGS:
        Xl = np.hstack([z1[f"{l}_X"], z2[f"{l}_X"][:, idx], z3[f"{l}_X"]])
        X.append(Xl); y.append(z1[f"{l}_y"])
    return np.vstack(X), np.concatenate(y)


def main():
    # ---- classical blend on the full feature bank ----
    X, y = assemble_from_caches()
    assert X.shape[1] == len(FEATURE_NAMES), (X.shape, len(FEATURE_NAMES))
    models = []
    for k in ("logreg", "hgb"):
        m = make_model(k); m.fit(X, y); models.append(m)

    # ---- ESN reservoir + linear readout on all data (reuse MFCC seqs) ----
    z = np.load("deep_cache.npz", allow_pickle=True)
    M = np.concatenate([z[f"{l}_M"] for l in LANGS])
    G = np.concatenate([z[f"{l}_G"] for l in LANGS])
    yd = np.concatenate([z[f"{l}_y"] for l in LANGS])
    reservoir = Reservoir(n_in=M.shape[2])
    readout = make_readout(); readout.fit(reservoir.states(M), yd)

    joblib.dump({"models": models, "features": FEATURE_NAMES,
                 "esn_reservoir": reservoir, "esn_readout": readout,
                 "weights": {"classical": W_CLASSICAL, "deep": W_DEEP, "esn": W_ESN},
                 "kind": "3-way stack (classical + ViT/BiLSTM + ESN)"},
                "model_final.joblib")
    print(f"classical blend + ESN readout trained on {len(y)} pauses -> model_final.joblib")

    # ---- deep ViT+Bi-LSTM on all data ----
    net = train_fold_all(M, G, yd, epochs=55)
    torch.save(net.state_dict(), "deep_final.pt")
    print(f"deep net trained on {len(yd)} pauses -> deep_final.pt")


if __name__ == "__main__":
    main()
