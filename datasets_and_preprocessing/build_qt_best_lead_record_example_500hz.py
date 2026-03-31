"""Prepare one QT best-lead example record at 500 Hz.

This script:
1. Loads one QT record
2. Selects the best lead automatically
3. Resamples the signal to 500 Hz
4. Maps wave annotations to 500 Hz
5. Saves the prepared example as one NPZ file

The saved output can be used later for plotting or debugging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import wfdb
from scipy.signal import resample_poly

RECORD = "sel301"
ANN_EXT = "q1c"
TARGET_FS = 500

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
NPZ_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = NPZ_DIR / f"qt_bestlead_example_{RECORD}_500hz.npz"


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


def choose_best_lead(signal: np.ndarray) -> Tuple[int, np.ndarray, List[float]]:
    """Choose the best ECG lead using a robust derivative-based score.

    The score is:
        percentile_99(|dx|) / MAD(dx)

    Args:
        signal: Multi-lead ECG signal of shape (n_samples, n_leads).

    Returns:
        Tuple containing:
            - best lead index
            - selected lead signal
            - list of all lead scores
    """
    scores: List[float] = []

    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        d = np.diff(x)

        qrs_strength = float(np.percentile(np.abs(d), 99))
        noise = mad(d)
        score = qrs_strength / (noise + 1e-12)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return best_idx, signal[:, best_idx], scores


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
    """Prepare one QT best-lead example record and save it."""
    print("Loading record...")
    record_path = QT_PATH / RECORD

    record = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), ANN_EXT)

    fs_in = int(record.fs)
    signal = record.p_signal

    print("Original fs:", fs_in)
    print("Signal shape:", signal.shape)

    best_idx, ecg_best, scores = choose_best_lead(signal)
    print("Best lead index:", best_idx)
    print("Lead scores:", scores)

    if fs_in != TARGET_FS:
        gcd = int(np.gcd(TARGET_FS, fs_in))
        up = TARGET_FS // gcd
        down = fs_in // gcd
        ecg_500 = resample_poly(ecg_best, up=up, down=down).astype(np.float32)
        fs_out = TARGET_FS
    else:
        ecg_500 = ecg_best.astype(np.float32)
        fs_out = fs_in

    print("Resampled length:", len(ecg_500))

    waves_in = parse_triplets(ann)

    waves_500: Dict[str, List[Tuple[int, int, int]]] = {"P": [], "QRS": [], "T": []}
    scale = fs_out / fs_in

    for key in waves_in:
        for onset, peak, offset in waves_in[key]:
            waves_500[key].append(
                (
                    int(round(onset * scale)),
                    int(round(peak * scale)),
                    int(round(offset * scale)),
                )
            )

    print("Number of QRS:", len(waves_500["QRS"]))
    print("Number of P:", len(waves_500["P"]))
    print("Number of T:", len(waves_500["T"]))

    np.savez(
        OUT_FILE,
        record=np.array([RECORD], dtype=object),
        ann_ext=np.array([ANN_EXT], dtype=object),
        fs_in=np.array([fs_in], dtype=np.int64),
        fs_out=np.array([fs_out], dtype=np.int64),
        best_lead_index=np.array([best_idx], dtype=np.int64),
        lead_scores=np.array(scores, dtype=np.float32),
        ecg_500=ecg_500,
        P=np.array(waves_500["P"], dtype=np.int64),
        QRS=np.array(waves_500["QRS"], dtype=np.int64),
        T=np.array(waves_500["T"], dtype=np.int64),
    )

    print("Saved:", OUT_FILE)
    print("Done.")


if __name__ == "__main__":
    main()