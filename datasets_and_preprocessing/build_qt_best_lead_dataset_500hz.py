"""Build a QT best-lead P/QRS/T window dataset at 500 Hz.

This script scans the QT Database, selects the best lead for each record,
rescales the ECG to 500 Hz, parses wave annotations, and builds beat-centered
windows with dense sample-wise labels for:

    0. background
    1. P wave
    2. QRS complex
    3. T wave

The final dataset is saved as one NPZ file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import wfdb
from scipy.signal import resample_poly
from tqdm import tqdm

TARGET_FS = 500
ANN_EXT = "q1c"

WIN_SEC = 1.2
HALF_WIN = WIN_SEC / 2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
NPZ_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = NPZ_DIR / "qt_bestlead_pqt_windows_500hz.npz"

CLS_BG = 0
CLS_P = 1
CLS_QRS = 2
CLS_T = 3


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


def mad(x: np.ndarray) -> float:
    """Compute the median absolute deviation.

    Args:
        x: Input array.

    Returns:
        Median absolute deviation.
    """
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def choose_best_lead(signal: np.ndarray) -> Tuple[int, np.ndarray]:
    """Choose the best ECG lead using a robust derivative-based score.

    The score is:
        percentile_99(|dx|) / MAD(dx)

    Args:
        signal: Multi-lead ECG signal of shape (n_samples, n_leads).

    Returns:
        Tuple containing:
            - best lead index
            - selected lead signal
    """
    scores = []

    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        d = np.diff(x)
        qrs_strength = np.percentile(np.abs(d), 99)
        noise = mad(d)
        score = qrs_strength / (noise + 1e-12)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return best_idx, signal[:, best_idx]


def parse_triplets(ann: wfdb.Annotation) -> Dict[str, List[Tuple[int, int, int]]]:
    """Parse QT annotation triplets into P, QRS, and T intervals.

    Expected pattern:
        "(", wave_symbol, ")"

    The sample positions are interpreted as:
        onset, peak, offset

    Args:
        ann: WFDB annotation object.

    Returns:
        Dictionary with keys ``P``, ``QRS``, and ``T``.
    """
    waves: Dict[str, List[Tuple[int, int, int]]] = {"P": [], "QRS": [], "T": []}

    symbols = ann.symbol
    samples = ann.sample

    i = 0
    while i < len(symbols) - 2:
        if symbols[i] == "(" and symbols[i + 2] == ")":
            wave_type = symbols[i + 1]
            onset = int(samples[i])
            peak = int(samples[i + 1])
            offset = int(samples[i + 2])

            if wave_type == "p":
                waves["P"].append((onset, peak, offset))
            elif wave_type == "N":
                waves["QRS"].append((onset, peak, offset))
            elif wave_type == "t":
                waves["T"].append((onset, peak, offset))

            i += 3
        else:
            i += 1

    return waves


def main() -> None:
    """Build the full QT best-lead dataset and save it as NPZ."""
    records = sorted(
        {
            os.path.splitext(filename)[0]
            for filename in os.listdir(QT_PATH)
            if filename.endswith(".hea")
        }
    )

    print("Found records:", len(records))

    X_list = []
    Y_list = []
    meta_list = []

    for rec in tqdm(records):
        try:
            record = wfdb.rdrecord(str(QT_PATH / rec))
            ann = wfdb.rdann(str(QT_PATH / rec), ANN_EXT)
        except Exception:
            continue

        fs_in = int(record.fs)
        signal = record.p_signal

        best_idx, ecg = choose_best_lead(signal)

        if fs_in != TARGET_FS:
            ecg = resample_poly(ecg, up=TARGET_FS, down=fs_in)

        fs = TARGET_FS
        waves = parse_triplets(ann)

        scale = fs / fs_in
        waves_500: Dict[str, List[Tuple[int, int, int]]] = {"P": [], "QRS": [], "T": []}

        for key in waves:
            for onset, peak, offset in waves[key]:
                waves_500[key].append(
                    (
                        int(round(onset * scale)),
                        int(round(peak * scale)),
                        int(round(offset * scale)),
                    )
                )

        if len(waves_500["QRS"]) == 0:
            continue

        win_samples = int(WIN_SEC * fs)
        half_win = win_samples // 2

        for onset, peak, offset in waves_500["QRS"]:
            _ = onset, offset

            start = peak - half_win
            end = peak + half_win

            if start < 0 or end >= len(ecg):
                continue

            x_win = ecg[start:end].astype(np.float32)
            y_mask = np.zeros(win_samples, dtype=np.int32)

            for p_on, p_peak, p_off in waves_500["P"]:
                _ = p_peak
                if start <= p_on and p_off <= end:
                    y_mask[p_on - start : p_off - start] = CLS_P

            for q_on, q_peak, q_off in waves_500["QRS"]:
                _ = q_peak
                if start <= q_on and q_off <= end:
                    y_mask[q_on - start : q_off - start] = CLS_QRS

            for t_on, t_peak, t_off in waves_500["T"]:
                _ = t_peak
                if start <= t_on and t_off <= end:
                    y_mask[t_on - start : t_off - start] = CLS_T

            X_list.append(x_win)
            Y_list.append(y_mask)
            meta_list.append([rec, best_idx, peak, "QT"])

    if len(X_list) == 0:
        raise RuntimeError("No windows were built. Check annotations and paths.")

    X = np.array(X_list, dtype=np.float32)[:, None, :]
    Y = np.array(Y_list, dtype=np.int32)
    meta = np.array(meta_list, dtype=object)

    print("Final dataset shape:", X.shape)

    np.savez(
        OUT_FILE,
        X=X,
        Y=Y,
        fs=np.array([TARGET_FS], dtype=np.int64),
        win_sec=np.array([WIN_SEC], dtype=np.float32),
        meta=meta,
    )

    print("Saved:", OUT_FILE)


if __name__ == "__main__":
    main()