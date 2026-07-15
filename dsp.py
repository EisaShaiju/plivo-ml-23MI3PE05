"""From-scratch signal-processing toolkit for EOT (Track 2).

Everything here is implemented directly in numpy/scipy -- NO external DSP
packages (no vmdpy / PyEMD / antropy) and NO pretrained weights -- so it is
rules-legal, and every routine runs on the pre-pause window only (causal).

Contents:
  * vmd()               -- Variational Mode Decomposition (Dragomiretskiy &
                           Zosso 2014): the constrained variational problem
                           min_{u_k,w_k} Σ_k ||∂_t[(δ+j/πt)*u_k] e^{-jw_k t}||²
                           s.t. Σ_k u_k = f, solved by ADMM in the Fourier domain.
  * spectral_kurtosis() -- STFT-based kurtosis per frequency bin (impulsiveness).
  * sample_entropy()    -- SampEn(m, r): signal regularity / predictability.
  * hilbert_envelope()  -- analytic-signal amplitude envelope (AM / decay shape).
"""
import numpy as np
from scipy.signal import hilbert, get_window
from scipy.interpolate import CubicSpline


# --------------------------------------------------------------------------- #
# Variational Mode Decomposition (constrained variational problem, ADMM)       #
# --------------------------------------------------------------------------- #
def vmd(f, alpha=2000.0, tau=0.0, K=4, DC=False, tol=1e-6, max_iter=150):
    """Decompose 1-D signal `f` into K narrow-band modes.

    Parameters
    ----------
    f        : real signal (1-D)
    alpha    : bandwidth constraint (higher = narrower modes)
    tau      : dual ascent step (0 = noise-tolerant, no strict Σu=f)
    K        : number of modes
    DC       : if True, force mode 0 to 0 Hz (a DC term)
    Returns
    -------
    u        : (K, T) reconstructed modes (time domain)
    omega    : (K,) final normalised centre frequencies (cycles/sample)
    """
    f = np.asarray(f, dtype=float)
    T = len(f)
    if T < 8:
        return np.zeros((K, T)), np.zeros(K)
    # mirror-extend to reduce boundary effects, then work on the analytic spectrum
    fmir = np.concatenate([f[T // 2:0:-1], f, f[-1:-T // 2 - 1:-1]])
    Tm = len(fmir)
    freqs = np.arange(Tm) / Tm - 0.5
    f_hat = np.fft.fftshift(np.fft.fft(fmir))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:Tm // 2] = 0                     # positive-frequency (analytic) half

    u_hat_plus = np.zeros((K, Tm), dtype=complex)
    # init centre frequencies spread over [0, 0.5]
    omega = np.zeros(K)
    for k in range(K):
        omega[k] = (0.5 / K) * (k + 0.5)
    if DC:
        omega[0] = 0.0
    lamb = np.zeros(Tm, dtype=complex)           # dual variable
    u_prev = u_hat_plus.copy()

    for _ in range(max_iter):
        sum_uk = np.zeros(Tm, dtype=complex)
        for k in range(K):
            # residual excluding mode k
            resid = f_hat_plus - (u_hat_plus.sum(axis=0) - u_hat_plus[k]) - lamb / 2.0
            u_hat_plus[k] = resid / (1.0 + alpha * (freqs - omega[k]) ** 2)
            if not (DC and k == 0):
                pw = np.abs(u_hat_plus[k, Tm // 2:]) ** 2
                denom = pw.sum()
                if denom > 1e-12:
                    omega[k] = (freqs[Tm // 2:] @ pw) / denom
            sum_uk += u_hat_plus[k]
        lamb = lamb + tau * (sum_uk - f_hat_plus)
        if np.sum(np.abs(u_hat_plus - u_prev) ** 2) / (np.sum(np.abs(u_prev) ** 2) + 1e-12) < tol:
            u_prev = u_hat_plus.copy()
            break
        u_prev = u_hat_plus.copy()

    # reconstruct modes in time
    u_hat = np.zeros((K, Tm), dtype=complex)
    u_hat[:, Tm // 2:] = u_hat_plus[:, Tm // 2:]
    u_hat[:, 1:Tm // 2 + 1] = np.conj(u_hat_plus[:, -1:Tm // 2 - 1:-1])
    u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat, axes=1), axis=1))
    u = u[:, Tm // 4: Tm // 4 + T]               # crop the mirror padding
    order = np.argsort(omega)
    return u[order], omega[order]


# --------------------------------------------------------------------------- #
# Empirical Mode Decomposition + Hilbert-Huang (Track 3, data-adaptive)        #
# --------------------------------------------------------------------------- #
def _extrema(x):
    d = np.diff(x)
    maxima = np.where((np.hstack([d, 0]) < 0) & (np.hstack([0, d]) > 0))[0]
    minima = np.where((np.hstack([d, 0]) > 0) & (np.hstack([0, d]) < 0))[0]
    return maxima, minima


def _envelope(idx, val, n):
    # anchor the ends to tame spline boundary swing
    idx = np.concatenate([[0], idx, [n - 1]])
    val = np.concatenate([[val[0]], val, [val[-1]]])
    idx, uniq = np.unique(idx, return_index=True)
    return CubicSpline(idx, val[uniq])(np.arange(n))


def emd(x, max_imf=6, max_sift=30, sd_thresh=0.2):
    """Empirical Mode Decomposition -> list of IMFs + residual (Huang 1998).

    Sifts local-extrema spline envelopes to peel off intrinsic mode functions,
    high frequency first. No basis is assumed -- it is fully data-adaptive."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return np.zeros((0, n)), x.copy()
    imfs, r = [], x.copy()
    for _ in range(max_imf):
        h = r.copy()
        for _ in range(max_sift):
            mx, mn = _extrema(h)
            if len(mx) < 2 or len(mn) < 2:
                break
            up = _envelope(mx, h[mx], n)
            lo = _envelope(mn, h[mn], n)
            mean = 0.5 * (up + lo)
            h_new = h - mean
            sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h_new
            if sd < sd_thresh:
                break
        imfs.append(h)
        r = r - h
        mx, mn = _extrema(r)
        if len(mx) + len(mn) < 3:                    # residual is monotonic
            break
    return np.array(imfs), r


def hilbert_huang(imf, sr):
    """Instantaneous amplitude and frequency (Hz) of one IMF via Hilbert."""
    a = hilbert(imf)
    amp = np.abs(a)
    phase = np.unwrap(np.angle(a))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * sr
    return amp, inst_freq


# --------------------------------------------------------------------------- #
# Spectral kurtosis (impulsiveness per frequency bin)                          #
# --------------------------------------------------------------------------- #
def spectral_kurtosis(x, sr, nperseg=256, hop=128):
    """SK(f) = <|X|^4>/<|X|^2>^2 - 2 over time frames. Returns (freqs, SK)."""
    x = np.asarray(x, float)
    if len(x) < nperseg + hop:
        return np.zeros(nperseg // 2 + 1), np.zeros(nperseg // 2 + 1)
    win = get_window("hann", nperseg)
    n = 1 + (len(x) - nperseg) // hop
    frames = np.stack([x[i * hop:i * hop + nperseg] * win for i in range(n)])
    X = np.fft.rfft(frames, axis=1)
    p = np.abs(X) ** 2
    m2 = p.mean(axis=0)
    m4 = (p ** 2).mean(axis=0)
    sk = m4 / (m2 ** 2 + 1e-12) - 2.0
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sr)
    return freqs, sk


# --------------------------------------------------------------------------- #
# Sample entropy (regularity of a short series)                                #
# --------------------------------------------------------------------------- #
def sample_entropy(x, m=2, r=0.2):
    """SampEn(m, r*std). Small = regular/predictable, large = complex."""
    x = np.asarray(x, float)
    N = len(x)
    if N < m + 2:
        return 0.0
    r = r * (np.std(x) + 1e-12)

    def _phi(mm):
        # count template matches within Chebyshev distance r (excluding self)
        templates = np.stack([x[i:i + mm] for i in range(N - mm + 1)])
        C = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            C += np.sum(d <= r) - 1
        return C

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


# --------------------------------------------------------------------------- #
# Hilbert analytic envelope                                                    #
# --------------------------------------------------------------------------- #
def hilbert_envelope(x):
    """Amplitude envelope via the analytic signal."""
    x = np.asarray(x, float)
    if len(x) < 4:
        return np.abs(x)
    return np.abs(hilbert(x))


# --------------------------------------------------------------------------- #
# Gramian Angular Field (1-D series -> 2-D image, for the deep track)          #
# --------------------------------------------------------------------------- #
def gramian_angular_field(x, size=48, kind="summation"):
    """GASF/GADF encoding of a 1-D series into a (size x size) image in [-1,1]."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.zeros((size, size), np.float32)
    # min-max scale to [-1, 1]
    mn, mx = x.min(), x.max()
    xs = (2 * (x - mn) / (mx - mn + 1e-12)) - 1.0
    # resample to `size` via linear interpolation
    xi = np.interp(np.linspace(0, len(xs) - 1, size), np.arange(len(xs)), xs)
    phi = np.arccos(np.clip(xi, -1, 1))
    if kind == "summation":
        g = np.cos(phi[:, None] + phi[None, :])
    else:  # difference
        g = np.sin(phi[:, None] - phi[None, :])
    return g.astype(np.float32)


if __name__ == "__main__":
    # self-test on a synthetic 2-tone + impulse signal
    sr = 8000
    t = np.arange(sr) / sr
    sig = np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
    sig[4000] += 5
    u, w = vmd(sig, K=3, alpha=2000)
    print("VMD modes:", u.shape, "centre freqs (Hz):", np.round(w * sr, 1))
    fr, sk = spectral_kurtosis(sig, sr)
    print("SK peak at", round(fr[np.argmax(sk)], 1), "Hz  max SK", round(float(sk.max()), 2))
    print("SampEn:", round(sample_entropy(sig[:200]), 3))
    print("GAF:", gramian_angular_field(sig[:400]).shape)
