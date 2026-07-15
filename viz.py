"""Plot energy(dB) + pYIN F0 over the 2.5 s before a pause, for chosen cases."""
import argparse, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
from features import load_wav, frame_energy_db

WIN = 2.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cases", required=True, help="tid:pi,tid:pi,...")
    ap.add_argument("--out", default="cases.png")
    args = ap.parse_args()
    lab = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    cases = [(c.split(":")[0], int(c.split(":")[1])) for c in args.cases.split(",")]
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5), squeeze=False)
    for j, (tid, pi) in enumerate(cases):
        row = lab[(lab.turn_id == tid) & (lab.pause_index == pi)].iloc[0]
        ps = float(row.pause_start); dur = float(row.pause_end - ps)
        x, sr = load_wav(os.path.join(args.data_dir, row.audio_file))
        a = max(0, int((ps - WIN) * sr)); b = int(ps * sr)
        seg = x[a:b].astype(np.float32)
        t0 = ps - (b - a) / sr
        e = frame_energy_db(seg, sr); te = t0 + np.arange(len(e)) * 0.010
        f0, vf, _ = librosa.pyin(seg, fmin=65, fmax=400, sr=sr,
                                 frame_length=1024, hop_length=160, center=True)
        tf = t0 + np.arange(len(f0)) * (160 / sr)
        axes[0][j].plot(te, e, lw=0.8)
        axes[0][j].set_title(f"{tid} pi{pi}\n{row.label} dur={dur:.2f}s", fontsize=9,
                             color=("green" if row.label == "eot" else "firebrick"))
        axes[0][j].axvline(ps, color="k", ls="--", lw=0.8)
        axes[0][j].set_ylabel("energy dB" if j == 0 else "")
        axes[1][j].plot(tf, f0, ".", ms=3)
        axes[1][j].axvline(ps, color="k", ls="--", lw=0.8)
        axes[1][j].set_ylim(60, 400); axes[1][j].set_ylabel("F0 Hz" if j == 0 else "")
        axes[1][j].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(args.out, dpi=90)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
