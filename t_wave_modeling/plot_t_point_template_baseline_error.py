"""Plot the timing-error distribution of the point-template T baseline.

This script builds the point-template T model from a small QT training subset,
evaluates it on a held-out QT subset, and plots the timing-error histogram for
the detected T peaks relative to the explicit T ground truth peaks.

A compact statistics box is added to the figure with:
- number of evaluated beats
- bias
- standard deviation
- mean absolute error
- root mean square error
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from t_wave_modeling_utils import (
    FS,
    build_point_t_template,
    compute_zs,
    detect_point_t_from_cached,
    load_qt_record_500hz,
    pair_r_with_explicit_t,
    parse_explicit_r_t,
    pick_training_and_eval_records,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "t_point_template_baseline_error.png"

# Optional restricted evaluation subset for faster figure generation.
FAST_EVAL_RECORDS = [
    "sel114",
    "sel116",
    "sel117",
    "sel123",
    "sel221",
    "sel231",
    "sel232",
    "sel233",
]


def choose_eval_records(eval_records_all: List[str]) -> List[str]:
    """Choose the QT evaluation records used for the histogram figure.

    Args:
        eval_records_all: Full held-out QT evaluation set.

    Returns:
        Selected evaluation records.
    """
    eval_records = [record for record in FAST_EVAL_RECORDS if record in eval_records_all]
    if len(eval_records) == 0:
        eval_records = eval_records_all[:8]
    return eval_records


def collect_timing_errors(point_template: np.ndarray, eval_records: List[str]) -> np.ndarray:
    """Collect point-template T-wave timing errors in milliseconds.

    Args:
        point_template: Point-template T model.
        eval_records: QT evaluation records.

    Returns:
        Timing error array in milliseconds.

    Raises:
        RuntimeError: If no usable timing errors were found.
    """
    errors_ms: List[float] = []

    for record_name in eval_records:
        print(f"Evaluating point-template baseline on {record_name} ...")
        try:
            signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
            _ = used_lead, ann_ext

            r_peaks, t_peaks = parse_explicit_r_t(samples, symbols)
            beats = pair_r_with_explicit_t(r_peaks, t_peaks)

            if len(beats) == 0:
                continue

            _, whitened_states = compute_zs(signal)

            for beat in beats:
                det_t, _, _ = detect_point_t_from_cached(
                    whitened_states,
                    beat["r"],
                    point_template,
                )
                if det_t is None:
                    continue

                err_ms = 1000.0 * (det_t - beat["t_peak"]) / FS
                errors_ms.append(err_ms)

        except Exception as exc:
            print(f"  skipped {record_name}: {exc}")

    error_array = np.array(errors_ms, dtype=float)

    if len(error_array) == 0:
        raise RuntimeError("No baseline timing errors found.")

    return error_array


def plot_histogram(errors_ms: np.ndarray) -> None:
    """Plot and save the timing-error histogram.

    Args:
        errors_ms: Timing error array in milliseconds.
    """
    bias = float(np.mean(errors_ms))
    std = float(np.std(errors_ms))
    mae = float(np.mean(np.abs(errors_ms)))
    rmse = float(np.sqrt(np.mean(errors_ms ** 2)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors_ms, bins=40)

    ax.set_xlabel("Timing error (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Point-template baseline T-wave timing error")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    text = (
        f"N = {len(errors_ms)}\n"
        f"Bias = {bias:.2f} ms\n"
        f"STD = {std:.2f} ms\n"
        f"MAE = {mae:.2f} ms\n"
        f"RMSE = {rmse:.2f} ms"
    )

    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUT_FILE}")


def main() -> None:
    """Build the point-template baseline and plot the error histogram."""
    train_records, eval_records_all = pick_training_and_eval_records()
    print("Training records:", train_records)

    eval_records = choose_eval_records(eval_records_all)
    print("Evaluation records:", eval_records)

    point_template = build_point_t_template(train_records)
    errors_ms = collect_timing_errors(point_template, eval_records)

    plot_histogram(errors_ms)


if __name__ == "__main__":
    main()