"""Plot a QT resampling example from 250 Hz to 500 Hz.

This script:
1. Loads one QT record
2. Parses manual P, QRS, and T triplet annotations
3. Plots the original signal at its native sampling rate
4. Resamples the signal to 500 Hz
5. Maps the annotation sample indices to 500 Hz
6. Plots the resampled signal with mapped annotations
7. Saves both figures to the thesis figure output folder

This figure pair is intended to illustrate the preprocessing step before
template building or segmentation dataset generation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import resample_poly

REC = "sel301"
ANN_EXT = "q1c"
TARGET_FS = 500

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE_250 = OUTPUT_DIR / "figure1_qt_250hz_annotations.png"
OUT_FILE_500 = OUTPUT_DIR / "figure2_qt_500hz_annotations.png"


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


def load_qt_record(qt_path: Path, rec: str, ann_ext: str) -> Tuple[wfdb.Record, wfdb.Annotation]:
    """Load one QT record and its annotation file.

    Args:
        qt_path: QT Database directory.
        rec: Record name.
        ann_ext: Annotation extension.

    Returns:
        Tuple containing the WFDB record and annotation objects.
    """
    rec_path = qt_path / rec
    record = wfdb.rdrecord(str(rec_path))
    ann = wfdb.rdann(str(rec_path), ann_ext)
    return record, ann


def parse_triplets(ann: wfdb.Annotation) -> Dict[str, List[Tuple[int, int, int]]]:
    """Extract P, QRS, and T annotation triplets.

    Expected annotation pattern:
        "(", wave_symbol, ")"

    Each triplet is interpreted as:
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


def map_samples(samples: np.ndarray | List[int], fs_in: int, fs_out: int) -> np.ndarray:
    """Map sample indices from one sampling rate to another.

    Args:
        samples: Input sample indices.
        fs_in: Original sampling rate.
        fs_out: Target sampling rate.

    Returns:
        Sample indices mapped to the target sampling rate.
    """
    factor = fs_out / fs_in
    return np.array(np.round(np.asarray(samples) * factor), dtype=int)


def map_triplet_waves(
    waves_in: Dict[str, List[Tuple[int, int, int]]],
    fs_in: int,
    fs_out: int,
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Map a wave-triplet dictionary from one sampling rate to another.

    Args:
        waves_in: Input triplet dictionary.
        fs_in: Original sampling rate.
        fs_out: Target sampling rate.

    Returns:
        Mapped triplet dictionary.
    """
    waves_out: Dict[str, List[Tuple[int, int, int]]] = {"P": [], "QRS": [], "T": []}

    for key in ["P", "QRS", "T"]:
        for onset, peak, offset in waves_in[key]:
            onset2 = int(round(onset * (fs_out / fs_in)))
            peak2 = int(round(peak * (fs_out / fs_in)))
            offset2 = int(round(offset * (fs_out / fs_in)))
            waves_out[key].append((onset2, peak2, offset2))

    return waves_out


def plot_with_waves(
    ecg: np.ndarray,
    fs: int,
    waves: Dict[str, List[Tuple[int, int, int]]],
    title: str,
    t0_s: float,
    t1_s: float,
    out_file: Path,
) -> None:
    """Plot one ECG segment with wave onset, peak, and offset lines.

    Args:
        ecg: ECG signal.
        fs: Sampling rate.
        waves: Wave triplet dictionary.
        title: Figure title.
        t0_s: Start time in seconds.
        t1_s: End time in seconds.
        out_file: Output PNG path.
    """
    n0 = int(round(t0_s * fs))
    n1 = int(round(t1_s * fs))
    n0 = max(0, n0)
    n1 = min(len(ecg), n1)

    t = np.arange(n0, n1) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ecg[n0:n1], lw=1.0, color="black")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    def draw_triplets(triplets: List[Tuple[int, int, int]], color: str, label_prefix: str) -> None:
        for onset, peak, offset in triplets:
            if n0 <= onset <= n1:
                ax.axvline(onset / fs, color=color, alpha=0.35)
            if n0 <= peak <= n1:
                ax.axvline(peak / fs, color=color, alpha=0.75)
            if n0 <= offset <= n1:
                ax.axvline(offset / fs, color=color, alpha=0.35)

        ax.plot([], [], color=color, label=label_prefix)

    draw_triplets(waves["P"], "tab:blue", "P (onset/peak/offset)")
    draw_triplets(waves["QRS"], "tab:red", "QRS (onset/peak/offset)")
    draw_triplets(waves["T"], "tab:green", "T (onset/peak/offset)")

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the QT resampling example and save both figures."""
    record, ann = load_qt_record(QT_PATH, REC, ANN_EXT)

    fs_in = int(record.fs)
    sig = record.p_signal.astype(np.float32)
    ecg0 = sig[:, 0]

    waves_250 = parse_triplets(ann)

    if len(waves_250["QRS"]) == 0:
        raise RuntimeError("No QRS triplets found in the selected annotation file.")

    qrs0 = waves_250["QRS"][0][1]
    center_t = qrs0 / fs_in
    t0 = max(0.0, center_t - 1.0)
    t1 = center_t + 1.5

    plot_with_waves(
        ecg=ecg0,
        fs=fs_in,
        waves=waves_250,
        title="Figure 1: QT record at original sampling rate with manual annotations",
        t0_s=t0,
        t1_s=t1,
        out_file=OUT_FILE_250,
    )

    fs_out = TARGET_FS
    gcd = math.gcd(fs_out, fs_in)
    up = fs_out // gcd
    down = fs_in // gcd
    ecg0_500 = resample_poly(ecg0, up=up, down=down).astype(np.float32)

    waves_500 = map_triplet_waves(waves_250, fs_in=fs_in, fs_out=fs_out)

    plot_with_waves(
        ecg=ecg0_500,
        fs=fs_out,
        waves=waves_500,
        title="Figure 2: QT record resampled to 500 Hz with mapped annotations",
        t0_s=t0,
        t1_s=t1,
        out_file=OUT_FILE_500,
    )

    print("Done.")
    print("fs_in:", fs_in, "fs_out:", fs_out)
    print("Length in:", len(ecg0), "Length out:", len(ecg0_500))
    print("Saved:")
    print(" ", OUT_FILE_250)
    print(" ", OUT_FILE_500)


if __name__ == "__main__":
    main()