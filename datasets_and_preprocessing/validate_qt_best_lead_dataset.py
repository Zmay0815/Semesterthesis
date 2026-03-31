"""Validate the QT best-lead P/QRS/T window dataset.

This script performs two types of checks on the saved QT best-lead dataset:

1. Statistical validation:
   - P-wave duration distribution
   - QRS duration distribution
   - T-wave duration distribution

2. Visual validation:
   - random example windows
   - region centers marked for P, QRS, and T

The validation figures are saved to the thesis output folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

N_SHOW = 6
SEED = 0

CLS_BG = 0
CLS_P = 1
CLS_QRS = 2
CLS_T = 3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures" / "qt_dataset_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH = NPZ_DIR / "qt_bestlead_pqt_windows_500hz.npz"


def region_centers(mask: np.ndarray, cls_id: int) -> np.ndarray:
    """Compute the center index of each contiguous class region.

    Args:
        mask: Dense class mask for one window.
        cls_id: Target class identifier.

    Returns:
        Array of center indices for contiguous regions of the selected class.
    """
    idx = np.where(mask == cls_id)[0]
    if idx.size == 0:
        return np.array([], dtype=int)

    centers = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            centers.append((start + prev) // 2)
            start = i
            prev = i

    centers.append((start + prev) // 2)
    return np.array(centers, dtype=int)


def collect_region_durations(mask: np.ndarray, cls_id: int, fs: int) -> List[float]:
    """Collect all contiguous region durations for one class.

    Args:
        mask: Dense class mask for one window.
        cls_id: Target class identifier.
        fs: Sampling rate in Hz.

    Returns:
        List of region durations in milliseconds.
    """
    durations_ms: List[float] = []

    idx = np.where(mask == cls_id)[0]
    if idx.size == 0:
        return durations_ms

    start = idx[0]
    prev = idx[0]

    for j in idx[1:]:
        if j == prev + 1:
            prev = j
        else:
            duration = (prev - start) / fs * 1000.0
            durations_ms.append(duration)
            start = j
            prev = j

    duration = (prev - start) / fs * 1000.0
    durations_ms.append(duration)

    return durations_ms


def print_stats(name: str, values: List[float]) -> None:
    """Print summary statistics for one duration distribution.

    Args:
        name: Distribution name.
        values: Duration list in milliseconds.
    """
    arr = np.array(values, dtype=float)

    print(f"\n{name}")
    print("Count:", len(arr))

    if len(arr) == 0:
        print("No values found.")
        return

    print("Mean:", np.mean(arr))
    print("Std:", np.std(arr))
    print("5th percentile:", np.percentile(arr, 5))
    print("95th percentile:", np.percentile(arr, 95))


def plot_random_examples(X: np.ndarray, Y: np.ndarray, fs: int) -> None:
    """Plot and save random example windows from the dataset.

    Args:
        X: Window tensor of shape (N, 1, L).
        Y: Dense mask tensor of shape (N, L).
        fs: Sampling rate in Hz.
    """
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(X.shape[0], size=min(N_SHOW, X.shape[0]), replace=False)

    for idx in idxs:
        x = X[idx, 0]
        mask = Y[idx]
        t = np.arange(len(x)) / fs

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, x, color="black")

        for cls, color, label in [
            (CLS_P, "blue", "P"),
            (CLS_QRS, "red", "QRS"),
            (CLS_T, "green", "T"),
        ]:
            centers = region_centers(mask, cls)
            if centers.size > 0:
                ax.scatter(centers / fs, x[centers], color=color, label=label)

        ax.set_title(f"Window {idx}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"qt_dataset_window_{idx}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    """Run the statistical and visual validation of the QT dataset."""
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"]
    Y = data["Y"]
    fs = int(data["fs"][0])

    print("Dataset shape:", X.shape)

    p_durations: List[float] = []
    qrs_durations: List[float] = []
    t_durations: List[float] = []

    for i in range(Y.shape[0]):
        mask = Y[i]
        p_durations.extend(collect_region_durations(mask, CLS_P, fs))
        qrs_durations.extend(collect_region_durations(mask, CLS_QRS, fs))
        t_durations.extend(collect_region_durations(mask, CLS_T, fs))

    print_stats("P-wave duration (ms)", p_durations)
    print_stats("QRS duration (ms)", qrs_durations)
    print_stats("T-wave duration (ms)", t_durations)

    plot_random_examples(X, Y, fs)

    print("\nSaved validation figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()