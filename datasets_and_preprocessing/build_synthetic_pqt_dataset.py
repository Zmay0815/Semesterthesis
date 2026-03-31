"""Build a synthetic ECG dataset with dense P/QRS/T labels.

This script generates synthetic one-lead ECG records composed of Gaussian
P, QRS, and T components with realistic variability in:

- heart rate
- PR interval
- QRS width
- T-wave polarity
- T-wave sharpness
- PVC-like morphology

It then cuts the signals into fixed windows and stores:

- X: (N, 1, L) signal windows
- Y: (N, L) dense class masks
- fs: sampling rate
- win_sec, hop_sec
- meta: object array with record id, lead name, start sample, and morphology type
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

CLS_BG = 0
CLS_P = 1
CLS_QRS = 2
CLS_T = 3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
NPZ_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = NPZ_DIR / "synth_bestlead_pqt_windows.npz"


def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return a Gaussian pulse.

    Args:
        t: Time axis.
        mu: Pulse center.
        sigma: Pulse width parameter.

    Returns:
        Gaussian pulse values.
    """
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def synth_one_beat(
    t: np.ndarray,
    p_mu: float,
    p_w: float,
    qrs_mu: float,
    qrs_w: float,
    t_mu: float,
    t_w: float,
    p_amp: float = 0.12,
    r_amp: float = 1.0,
    t_amp: float = 0.35,
    invert_t: bool = False,
    peaked_t: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize one ECG beat and its dense sample-wise labels.

    The beat is modeled as a sum of Gaussian components for P, Q, R, S, and T.

    Args:
        t: Local time axis.
        p_mu: P-wave center.
        p_w: P-wave width.
        qrs_mu: QRS center.
        qrs_w: QRS width.
        t_mu: T-wave center.
        t_w: T-wave width.
        p_amp: P-wave amplitude.
        r_amp: R-wave amplitude scale.
        t_amp: T-wave amplitude.
        invert_t: Whether the T-wave should be inverted.
        peaked_t: Whether the T-wave should be sharper and taller.

    Returns:
        Tuple containing:
            - synthesized signal
            - dense label vector
    """
    p = p_amp * _gauss(t, p_mu, p_w)

    q = (-0.15 * r_amp) * _gauss(t, qrs_mu - 0.012, qrs_w * 0.7)
    r = (1.00 * r_amp) * _gauss(t, qrs_mu, qrs_w)
    s = (-0.25 * r_amp) * _gauss(t, qrs_mu + 0.014, qrs_w * 0.8)
    qrs = q + r + s

    amp = t_amp
    t_width = t_w

    if peaked_t:
        amp *= 1.6
        t_width *= 0.7

    if invert_t:
        amp *= -1.0

    tw = amp * _gauss(t, t_mu, t_width)

    x = p + qrs + tw
    y = np.zeros_like(t, dtype=np.int64)

    p_region = (t >= (p_mu - 2.5 * p_w)) & (t <= (p_mu + 2.5 * p_w))
    y[p_region] = CLS_P

    qrs_region = (t >= (qrs_mu - 2.8 * qrs_w)) & (t <= (qrs_mu + 2.8 * qrs_w))
    y[qrs_region] = CLS_QRS

    t_region = (t >= (t_mu - 2.8 * t_width)) & (t <= (t_mu + 2.8 * t_width))
    y[t_region] = CLS_T

    return x, y


def synth_record(
    fs: int,
    duration_s: float,
    hr: float,
    kind: str = "normal",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic ECG record and its dense labels.

    Supported morphology types:
        - normal
        - long_pr
        - wide_qrs
        - t_inversion
        - peaked_t
        - pvc_like

    Args:
        fs: Sampling rate in Hz.
        duration_s: Record length in seconds.
        hr: Heart rate in beats per minute.
        kind: Morphology type.
        seed: Random seed.

    Returns:
        Tuple containing:
            - synthetic ECG signal
            - dense sample-wise label vector
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples) / fs

    rr = 60.0 / float(hr)
    beat_times = np.arange(0.5, duration_s - 0.5, rr)

    x = np.zeros(n_samples, dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    drift = 0.06 * np.sin(2 * np.pi * 0.33 * t + rng.uniform(0, 2 * np.pi))
    white = 0.02 * rng.standard_normal(n_samples)
    x += drift.astype(np.float32) + white.astype(np.float32)

    for beat_time in beat_times:
        pr = rng.normal(0.16, 0.02)
        qrs_dur = rng.normal(0.09, 0.01)
        qt = rng.normal(0.36, 0.03)

        invert_t = False
        peaked_t = False
        pvc = False

        if kind == "long_pr":
            pr = rng.normal(0.24, 0.02)
        elif kind == "wide_qrs":
            qrs_dur = rng.normal(0.14, 0.015)
        elif kind == "t_inversion":
            invert_t = True
        elif kind == "peaked_t":
            peaked_t = True
        elif kind == "pvc_like":
            pvc = True

        qrs_mu = beat_time
        p_mu = beat_time - pr
        t_mu = beat_time + (qt - 0.09)

        p_w = rng.uniform(0.018, 0.030)
        qrs_w = max(0.010, qrs_dur / 6.0)
        t_w = rng.uniform(0.045, 0.070)

        p_amp = rng.uniform(0.06, 0.14)
        r_amp = rng.uniform(0.8, 1.3)
        t_amp = rng.uniform(0.20, 0.45)

        if pvc:
            qrs_mu = beat_time - rng.uniform(0.12, 0.18)
            qrs_w *= 1.6
            r_amp *= 1.2
            p_amp = 0.0

        w0 = int(max(0, np.floor((qrs_mu - 0.6) * fs)))
        w1 = int(min(n_samples, np.ceil((qrs_mu + 0.8) * fs)))
        tt = t[w0:w1]

        xb, yb = synth_one_beat(
            tt,
            p_mu=p_mu,
            p_w=p_w,
            qrs_mu=qrs_mu,
            qrs_w=qrs_w,
            t_mu=t_mu,
            t_w=t_w,
            p_amp=p_amp,
            r_amp=r_amp,
            t_amp=t_amp,
            invert_t=invert_t,
            peaked_t=peaked_t,
        )

        x[w0:w1] += xb.astype(np.float32)

        mask = yb != CLS_BG
        y[w0:w1][mask] = yb[mask]

    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x.astype(np.float32), y.astype(np.int64)


def cut_windows(
    x: np.ndarray,
    y: np.ndarray,
    fs: int,
    win_sec: float,
    hop_sec: float,
    rec_id: str,
    lead: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut a full record into fixed sliding windows.

    Args:
        x: Input ECG signal.
        y: Dense label vector.
        fs: Sampling rate in Hz.
        win_sec: Window length in seconds.
        hop_sec: Hop size in seconds.
        rec_id: Record identifier.
        lead: Lead identifier.

    Returns:
        Tuple containing:
            - Xw: window tensor of shape (N, 1, L)
            - Yw: dense label tensor of shape (N, L)
            - meta: object array with record id, lead, and start sample
    """
    window_len = int(round(win_sec * fs))
    hop_len = int(round(hop_sec * fs))

    Xw = []
    Yw = []
    meta = []

    for start in range(0, len(x) - window_len + 1, hop_len):
        Xw.append(x[start : start + window_len][None, :])
        Yw.append(y[start : start + window_len])
        meta.append([str(rec_id), str(lead), int(start)])

    Xw_arr = np.stack(Xw, axis=0).astype(np.float32)
    Yw_arr = np.stack(Yw, axis=0).astype(np.int64)
    meta_arr = np.array(meta, dtype=object)

    return Xw_arr, Yw_arr, meta_arr


def main() -> None:
    """Build the full synthetic ECG dataset and save it as NPZ."""
    fs = 500
    win_sec = 1.2
    hop_sec = 0.2

    n_records = 200
    duration_s = 30.0
    lead = "best"

    kinds = ["normal", "long_pr", "wide_qrs", "t_inversion", "peaked_t", "pvc_like"]
    probs = np.array([0.55, 0.10, 0.12, 0.10, 0.08, 0.05], dtype=float)
    probs = probs / probs.sum()

    X_all = []
    Y_all = []
    meta_all = []

    for rid in range(1, n_records + 1):
        rng = np.random.default_rng(rid)
        kind = str(rng.choice(kinds, p=probs))
        hr = float(rng.uniform(55, 110))

        x, y = synth_record(
            fs=fs,
            duration_s=duration_s,
            hr=hr,
            kind=kind,
            seed=rid,
        )

        Xw, Yw, meta = cut_windows(
            x=x,
            y=y,
            fs=fs,
            win_sec=win_sec,
            hop_sec=hop_sec,
            rec_id=str(rid),
            lead=lead,
        )

        X_all.append(Xw)
        Y_all.append(Yw)

        kind_col = np.array([[kind] for _ in range(meta.shape[0])], dtype=object)
        meta_with_kind = np.concatenate([meta, kind_col], axis=1)
        meta_all.append(meta_with_kind)

    X_all_arr = np.concatenate(X_all, axis=0)
    Y_all_arr = np.concatenate(Y_all, axis=0)
    meta_all_arr = np.concatenate(meta_all, axis=0)

    np.savez_compressed(
        OUT_NPZ,
        X=X_all_arr,
        Y=Y_all_arr,
        fs=np.array([fs], dtype=np.int64),
        win_sec=np.array([win_sec], dtype=np.float32),
        hop_sec=np.array([hop_sec], dtype=np.float32),
        meta=meta_all_arr,
    )

    print("[OK] saved:", OUT_NPZ)
    print("X:", X_all_arr.shape, "Y:", Y_all_arr.shape, "meta:", meta_all_arr.shape)
    print("meta example:", meta_all_arr[0])


if __name__ == "__main__":
    main()