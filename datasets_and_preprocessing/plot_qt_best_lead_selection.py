"""Plot the QT best-lead selection score for two leads.

This script loads one QT record, extracts the first two leads, and visualizes
the derivative-based score used for best-lead selection:

    score = percentile_99(|dx|) / MAD(dx)

The figure shows:
1. Lead 1 ECG segment
2. Lead 1 derivative with Q99 and MAD reference lines
3. Lead 2 ECG segment
4. Lead 2 derivative with Q99 and MAD reference lines
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import wfdb

RECORD = "sel100"
START_SEC = 60.0
END_SEC = 63.0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "best_lead_score_visualized.png"


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


def score_terms(d: np.ndarray) -> Tuple[float, float, float]:
    """Compute the derivative-based lead score components.

    Args:
        d: First derivative of one ECG lead.

    Returns:
        Tuple containing:
            - 99th percentile of absolute derivative
            - median absolute deviation of derivative
            - score = Q99 / MAD
    """
    q99 = float(np.percentile(np.abs(d), 99))
    mad = float(np.median(np.abs(d - np.median(d))))
    score = q99 / (mad + 1e-12)
    return q99, mad, score


def main() -> None:
    """Create and save the best-lead score visualization figure."""
    record = wfdb.rdrecord(str(QT_PATH / RECORD))
    signal = record.p_signal
    fs = int(record.fs)

    if signal.ndim != 2 or signal.shape[1] < 2:
        raise RuntimeError("This figure requires a QT record with at least two leads.")

    lead1 = signal[:, 0].astype(np.float64)
    lead2 = signal[:, 1].astype(np.float64)

    d1 = np.diff(lead1)
    d2 = np.diff(lead2)

    q1, m1, s1 = score_terms(d1)
    q2, m2, s2 = score_terms(d2)

    start = int(START_SEC * fs)
    end = int(END_SEC * fs)

    if start < 0 or end > len(lead1) or end <= start:
        raise RuntimeError("Selected time segment is outside the record range.")

    t = np.arange(start, end) / fs

    seg1 = lead1[start:end]
    seg2 = lead2[start:end]

    # Derivative length is one sample shorter than the signal.
    dseg1 = d1[start : min(end, len(d1))]
    dseg2 = d2[start : min(end, len(d2))]
    td = np.arange(start, start + len(dseg1)) / fs

    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=False)

    ax[0].plot(t, seg1, color="black")
    ax[0].set_title("Lead 1 ECG signal")
    ax[0].set_ylabel("Amplitude (mV)")

    ax[1].plot(td, dseg1, color="black")
    ax[1].axhline(q1, color="red", linestyle="--", label="99th percentile")
    ax[1].axhline(m1, color="blue", linestyle="--", label="MAD")
    ax[1].set_title(f"Lead 1 derivative  |  Score = {s1:.2f}")
    ax[1].set_ylabel("Derivative amplitude (mV)")
    ax[1].legend()

    ax[2].plot(t, seg2, color="black")
    ax[2].set_title("Lead 2 ECG signal")
    ax[2].set_ylabel("Amplitude (mV)")

    ax[3].plot(td, dseg2, color="black")
    ax[3].axhline(q2, color="red", linestyle="--", label="99th percentile")
    ax[3].axhline(m2, color="blue", linestyle="--", label="MAD")
    ax[3].set_title(f"Lead 2 derivative  |  Score = {s2:.2f}")
    ax[3].set_ylabel("Derivative amplitude (mV)")
    ax[3].set_xlabel("Time (s)")
    ax[3].legend()

    for axis in ax:
        axis.tick_params(labelsize=9)
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure to:", OUT_FILE)


if __name__ == "__main__":
    main()