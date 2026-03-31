"""Plot pointwise P- and T-template search windows on one QT beat.

This script loads one QT record, extracts explicit P, R, and T annotations,
selects one central beat, and visualizes the physiological search windows used
for pointwise P- and T-template detection.

The figure shows:
- the ECG segment around the selected beat
- the selected R peak
- annotated P peaks in the segment
- annotated T peaks in the segment
- the P search window
- the T search window
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

FS_TARGET = 500

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "pt_template_detection.png"

RECORD = "sel100"
CAND_EXTS = ["q1c", "pu", "qt1", "atr"]

P_WINDOW = (-0.25, -0.08)
T_WINDOW = (0.10, 0.45)
DISPLAY_WINDOW_SEC = 0.60

BEAT_SYMBOLS = {
    "N",
    "L",
    "R",
    "A",
    "a",
    "J",
    "S",
    "V",
    "E",
    "F",
    "e",
    "j",
    "/",
    "f",
    "Q",
}


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


def list_qt_records(base_path: Path) -> List[str]:
    """List QT record names from a QT Database directory.

    Args:
        base_path: QT Database folder.

    Returns:
        Sorted list of record names without extension.
    """
    records = [
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(str(base_path / "*.hea"))
    ]
    return sorted(records)


def try_rdann(record_path: Path) -> Tuple[Optional[str], Optional[wfdb.Annotation]]:
    """Try multiple annotation extensions for a QT record.

    Args:
        record_path: Path without extension to the QT record.

    Returns:
        Tuple containing:
            - annotation extension if found
            - loaded annotation object if found
    """
    for ext in CAND_EXTS:
        try:
            ann = wfdb.rdann(str(record_path), ext)
            return ext, ann
        except Exception:
            continue
    return None, None


def load_qt_record(
    record_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Load one QT record and resample it to 500 Hz if needed.

    Args:
        record_name: QT record name.

    Returns:
        Tuple containing:
            - ECG signal
            - annotation sample indices
            - annotation symbols
            - sampling rate
            - annotation extension

    Raises:
        RuntimeError: If no usable annotation was found.
    """
    record_path = QT_PATH / record_name
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, 0].astype(np.float64)
    fs_orig = int(record.fs)

    ann_ext, ann = try_rdann(record_path)
    if ann is None or ann_ext is None:
        raise RuntimeError(f"No usable annotation found for {record_name}")

    samples = ann.sample.copy()
    symbols = np.array(ann.symbol)

    if fs_orig != FS_TARGET:
        new_len = int(len(signal) * FS_TARGET / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)
        samples = (samples * FS_TARGET / fs_orig).astype(int)
        fs = FS_TARGET
    else:
        fs = fs_orig

    return signal, samples, symbols, fs, ann_ext


def extract_prt_peaks(
    samples: np.ndarray,
    symbols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract explicit P, R, and T peak indices.

    Args:
        samples: Annotation sample indices.
        symbols: Annotation symbols.

    Returns:
        Tuple containing:
            - R peak indices
            - P peak indices
            - T peak indices
    """
    r_peaks = np.array(
        [samples[i] for i, symbol in enumerate(symbols) if symbol in BEAT_SYMBOLS],
        dtype=int,
    )
    p_peaks = np.array(
        [samples[i] for i, symbol in enumerate(symbols) if symbol.lower() == "p"],
        dtype=int,
    )
    t_peaks = np.array(
        [samples[i] for i, symbol in enumerate(symbols) if symbol.lower() == "t"],
        dtype=int,
    )

    return r_peaks, p_peaks, t_peaks


def choose_central_r_peak(r_peaks: np.ndarray) -> int:
    """Choose the central annotated R peak.

    Args:
        r_peaks: Annotated R peak indices.

    Returns:
        Selected central R peak index.

    Raises:
        RuntimeError: If no R peaks were found.
    """
    if len(r_peaks) == 0:
        raise RuntimeError("No R peaks found in the selected record.")
    return int(r_peaks[len(r_peaks) // 2])


def plot_detection_windows(
    signal: np.ndarray,
    fs: int,
    r_peak: int,
    p_peaks: np.ndarray,
    t_peaks: np.ndarray,
) -> None:
    """Plot the ECG segment and the P/T pointwise search windows.

    Args:
        signal: ECG signal.
        fs: Sampling rate.
        r_peak: Selected R peak index.
        p_peaks: Annotated P peak indices.
        t_peaks: Annotated T peak indices.
    """
    p_start = int(r_peak + P_WINDOW[0] * fs)
    p_end = int(r_peak + P_WINDOW[1] * fs)

    t_start = int(r_peak + T_WINDOW[0] * fs)
    t_end = int(r_peak + T_WINDOW[1] * fs)

    display_window = int(DISPLAY_WINDOW_SEC * fs)
    start = max(0, r_peak - display_window)
    end = min(len(signal), r_peak + display_window)

    time_axis = np.arange(start, end) / fs
    segment = signal[start:end]

    p_in = p_peaks[(p_peaks >= start) & (p_peaks <= end)]
    t_in = t_peaks[(t_peaks >= start) & (t_peaks <= end)]

    fig, ax = plt.subplots(figsize=(10, 3.2))

    ax.plot(time_axis, segment, color="black", linewidth=1.2, label="ECG")
    ax.scatter(r_peak / fs, signal[r_peak], color="red", s=80, label="R peak", zorder=5)

    if len(p_in) > 0:
        ax.scatter(p_in / fs, signal[p_in], color="blue", s=60, label="P peak", zorder=5)
    if len(t_in) > 0:
        ax.scatter(t_in / fs, signal[t_in], color="green", s=60, label="T peak", zorder=5)

    ax.axvspan(p_start / fs, p_end / fs, color="blue", alpha=0.15, label="P search window")
    ax.axvspan(t_start / fs, t_end / fs, color="green", alpha=0.15, label="T search window")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("Pointwise template detection for P and T waves")
    ax.legend(fontsize=8, frameon=True)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUT_FILE}")


def main() -> None:
    """Run the pointwise P/T template detection window plot."""
    available_records = list_qt_records(QT_PATH)
    if RECORD not in available_records:
        raise RuntimeError(f"Record {RECORD} was not found in the QT database.")

    signal, samples, symbols, fs, ann_ext = load_qt_record(RECORD)
    print(f"Using annotation extension: {ann_ext}")

    r_peaks, p_peaks, t_peaks = extract_prt_peaks(samples, symbols)
    r_peak = choose_central_r_peak(r_peaks)

    plot_detection_windows(
        signal=signal,
        fs=fs,
        r_peak=r_peak,
        p_peaks=p_peaks,
        t_peaks=t_peaks,
    )


if __name__ == "__main__":
    main()