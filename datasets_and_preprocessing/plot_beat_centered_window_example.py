"""Plot one beat-centered ECG window and its segmentation mask.

This script loads one QT record, selects the best lead automatically, extracts
explicit P, R, and T peaks, chooses one central beat, and plots:

1. A beat-centered ECG window
2. A dense segmentation mask with:
   - 0: background
   - 1: P wave
   - 2: QRS complex
   - 3: T wave

The figure is saved to the thesis figure output folder.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import resample

RECORD = "sel100"
WINDOW_LENGTH = 600
TARGET_FS = 500
ANN_EXT = "q1c"

CLS_BG = 0
CLS_P = 1
CLS_QRS = 2
CLS_T = 3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "figure3_beat_centered_window_example.png"


def resolve_qt_database_dir() -> Path:
    """Find the QT Database folder.

    Returns:
        Path to the QT Database directory.

    Raises:
        FileNotFoundError: If no valid QT Database directory was found.
    """
    env_path = os.environ.get("QT_DATABASE_DIR")

    candidates = [
        Path(env_path) if env_path else None,
        SCRIPT_DIR / "qt-database-1.0.0" / "qt-database-1.0.0",
        SCRIPT_DIR / "qt-database-1.0.0",
        PROJECT_ROOT / "qt-database-1.0.0" / "qt-database-1.0.0",
        PROJECT_ROOT / "qt-database-1.0.0",
        PROJECT_ROOT / "datasets" / "qt-database-1.0.0" / "qt-database-1.0.0",
        PROJECT_ROOT / "datasets" / "qt-database-1.0.0",
        PROJECT_ROOT / "datasets_and_preprocessing" / "qt-database-1.0.0" / "qt-database-1.0.0",
        PROJECT_ROOT / "datasets_and_preprocessing" / "qt-database-1.0.0",
    ]

    valid_candidates = [c for c in candidates if c is not None]

    for candidate in valid_candidates:
        if (candidate / "sel100.hea").exists():
            return candidate

    searched = "\n".join(str(c) for c in valid_candidates)
    raise FileNotFoundError(
        "Could not find the QT Database folder.\n"
        "Set QT_DATABASE_DIR or place the database in one of these locations:\n"
        f"{searched}"
    )


QT_PATH = resolve_qt_database_dir()


def try_rdann(record_path: Path) -> Tuple[Optional[str], Optional[wfdb.Annotation]]:
    """Try multiple annotation extensions for a QT record.

    Args:
        record_path: Path without extension to the QT record.

    Returns:
        Tuple containing:
            - annotation extension if found
            - loaded annotation object if found
    """
    for ext in ["q1c", "pu", "qt1", "atr"]:
        try:
            ann = wfdb.rdann(str(record_path), ext)
            return ext, ann
        except Exception:
            continue
    return None, None


def mad(x: np.ndarray) -> float:
    """Compute the median absolute deviation."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def choose_best_lead(signal: np.ndarray) -> Tuple[int, np.ndarray]:
    """Choose the best ECG lead using a robust derivative-based score.

    Args:
        signal: Multi-lead ECG signal.

    Returns:
        Tuple containing:
            - best lead index
            - selected lead signal
    """
    if signal.ndim == 1:
        return 0, signal.astype(np.float64)

    scores = []
    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        d = np.diff(x)
        qrs_strength = np.percentile(np.abs(d), 99)
        noise = mad(d)
        scores.append(qrs_strength / (noise + 1e-12))

    best_idx = int(np.argmax(scores))
    return best_idx, signal[:, best_idx].astype(np.float64)


def load_qt_record_bestlead_500hz(
    record_name: str,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Load one QT record, choose the best lead, and resample to 500 Hz.

    Args:
        record_name: QT record name.

    Returns:
        Tuple containing:
            - best-lead ECG signal at 500 Hz
            - output sampling rate
            - annotation sample indices at 500 Hz
            - annotation symbols
    """
    record_path = QT_PATH / record_name
    record = wfdb.rdrecord(str(record_path))
    ann_ext, ann = try_rdann(record_path)

    if ann is None or ann_ext is None:
        raise RuntimeError(f"No usable annotation found for {record_name}")

    fs_in = int(record.fs)
    signal = record.p_signal
    _, best_lead = choose_best_lead(signal)

    samples = ann.sample.copy()
    symbols = np.array(ann.symbol)

    if fs_in != TARGET_FS:
        new_len = int(len(best_lead) * TARGET_FS / fs_in)
        best_lead = resample(best_lead, new_len).astype(np.float64)
        samples = (samples * TARGET_FS / fs_in).astype(int)
        fs_out = TARGET_FS
    else:
        fs_out = fs_in
        best_lead = best_lead.astype(np.float64)

    return best_lead, fs_out, samples, symbols


def extract_peaks(
    samples: np.ndarray,
    symbols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract explicit P, R, and T peaks.

    Args:
        samples: Annotation sample indices.
        symbols: Annotation symbols.

    Returns:
        Tuple containing:
            - R peaks
            - P peaks
            - T peaks
    """
    r_peaks = []
    p_peaks = []
    t_peaks = []

    for sample, symbol in zip(samples, symbols):
        if symbol == "N":
            r_peaks.append(int(sample))
        if symbol == "p":
            p_peaks.append(int(sample))
        if symbol == "t":
            t_peaks.append(int(sample))

    return (
        np.array(r_peaks, dtype=int),
        np.array(p_peaks, dtype=int),
        np.array(t_peaks, dtype=int),
    )


def put_region(mask: np.ndarray, left: int, right: int, cls_id: int) -> None:
    """Write one class region into the segmentation mask."""
    left = max(0, left)
    right = min(len(mask), right)
    if right > left:
        mask[left:right] = cls_id


def build_segmentation_mask(
    start: int,
    end: int,
    fs: int,
    p_peaks: np.ndarray,
    r_peaks: np.ndarray,
    t_peaks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a region-based segmentation mask inside one window.

    Approximate region widths:
        - P:   100 ms
        - QRS: 100 ms
        - T:   200 ms

    Args:
        start: Window start sample.
        end: Window end sample.
        fs: Sampling rate.
        p_peaks: P peak indices.
        r_peaks: R peak indices.
        t_peaks: T peak indices.

    Returns:
        Tuple containing:
            - segmentation mask
            - local P peak indices
            - local R peak indices
            - local T peak indices
    """
    window_length = end - start
    mask = np.zeros(window_length, dtype=np.int64)

    p_width = int(0.10 * fs)
    qrs_width = int(0.10 * fs)
    t_width = int(0.20 * fs)

    p_local = []
    r_local = []
    t_local = []

    for p in p_peaks:
        if start <= p < end:
            idx = p - start
            p_local.append(idx)
            put_region(mask, idx - p_width // 2, idx + p_width // 2, CLS_P)

    for r in r_peaks:
        if start <= r < end:
            idx = r - start
            r_local.append(idx)
            put_region(mask, idx - qrs_width // 2, idx + qrs_width // 2, CLS_QRS)

    for t in t_peaks:
        if start <= t < end:
            idx = t - start
            t_local.append(idx)
            put_region(mask, idx - t_width // 2, idx + t_width // 2, CLS_T)

    return (
        mask,
        np.array(p_local, dtype=int),
        np.array(r_local, dtype=int),
        np.array(t_local, dtype=int),
    )


def main() -> None:
    """Create and save the beat-centered window example figure."""
    signal, fs, samples, symbols = load_qt_record_bestlead_500hz(RECORD)
    r_peaks, p_peaks, t_peaks = extract_peaks(samples, symbols)

    if len(r_peaks) == 0:
        raise RuntimeError("No annotated R peaks found.")

    n_r = int(r_peaks[len(r_peaks) // 2])

    half = WINDOW_LENGTH // 2
    start = n_r - half
    end = n_r + half

    if start < 0 or end > len(signal):
        raise RuntimeError("Selected beat-centered window exceeds signal boundaries.")

    window = signal[start:end]
    t_axis = np.arange(-half, half) / fs

    mask, p_local, r_local, t_local = build_segmentation_mask(
        start=start,
        end=end,
        fs=fs,
        p_peaks=p_peaks,
        r_peaks=r_peaks,
        t_peaks=t_peaks,
    )

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax[0].plot(t_axis, window, color="black")

    if len(p_local) > 0:
        ax[0].scatter(t_axis[p_local], window[p_local], color="blue", label="P peak")
    if len(r_local) > 0:
        ax[0].scatter(t_axis[r_local], window[r_local], color="red", label="R peak")
    if len(t_local) > 0:
        ax[0].scatter(t_axis[t_local], window[t_local], color="green", label="T peak")

    ax[0].set_ylabel("Amplitude (mV)")
    ax[0].set_title("Beat-centered ECG window")
    ax[0].legend()

    ax[1].step(t_axis, mask, where="mid")
    ax[1].set_xlabel("Time relative to R peak (s)")
    ax[1].set_ylabel("Class label")
    ax[1].set_yticks([0, 1, 2, 3])
    ax[1].set_yticklabels(["Background", "P", "QRS", "T"])
    ax[1].set_title("Segmentation mask $Y[n]$")

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure to:", OUT_FILE)


if __name__ == "__main__":
    main()