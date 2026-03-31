"""Plot representative point-template T-wave baseline failures.

This script builds the point-template T model from a small QT training subset,
evaluates it on held-out QT records, and plots four representative examples
with larger timing errors.

Each panel shows:
- the ECG signal segment
- the broad post-R T search window
- the R anchor
- the explicit T ground truth peak
- the detected T peak from the point-template baseline
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from t_wave_modeling_utils import (
    FS,
    T_WINDOW_SEC,
    build_point_t_template,
    evaluate_point_baseline,
    pick_training_and_eval_records,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "t_point_template_baseline_examples.png"


def select_examples(examples: List[Dict[str, object]], n_examples: int = 4) -> List[Dict[str, object]]:
    """Select representative larger-error examples.

    The selection prefers larger absolute timing errors and tries to avoid
    repeating the same record too often.

    Args:
        examples: List of baseline example dictionaries.
        n_examples: Number of examples to keep.

    Returns:
        Selected example list.
    """
    sorted_examples = sorted(
        examples,
        key=lambda x: abs(float(x["err_ms"])),
        reverse=True,
    )

    selected: List[Dict[str, object]] = []
    used_records = set()

    for example in sorted_examples:
        if abs(float(example["err_ms"])) < 40:
            continue
        if example["record"] in used_records and len(selected) < 2:
            continue

        selected.append(example)
        used_records.add(example["record"])

        if len(selected) == n_examples:
            break

    if len(selected) < n_examples:
        selected = sorted_examples[:n_examples]

    return selected


def plot_example_panel(ax: plt.Axes, example: Dict[str, object]) -> None:
    """Plot one representative baseline example.

    Args:
        ax: Matplotlib axis.
        example: Example dictionary returned by the evaluation helper.
    """
    signal = np.asarray(example["signal"])
    r_idx = int(example["r"])
    t_gt = int(example["t_gt"])
    t_det = int(example["t_det"])
    err_ms = float(example["err_ms"])

    left = max(0, r_idx - int(0.20 * FS))
    right = min(len(signal), r_idx + int(0.60 * FS))
    t_axis = np.arange(left, right) / FS

    ax.plot(t_axis, signal[left:right], color="black", linewidth=1.2, label="ECG")

    window_left = r_idx + int(T_WINDOW_SEC[0] * FS)
    window_right = r_idx + int(T_WINDOW_SEC[1] * FS)
    ax.axvspan(
        window_left / FS,
        window_right / FS,
        color="gray",
        alpha=0.12,
        label="Broad T window",
    )

    ax.axvline(
        r_idx / FS,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="R anchor",
    )
    ax.scatter(
        t_gt / FS,
        signal[t_gt],
        color="green",
        s=40,
        zorder=5,
        label="T ground truth",
    )
    ax.scatter(
        t_det / FS,
        signal[t_det],
        color="magenta",
        s=40,
        zorder=5,
        label="T detected",
    )

    ax.set_title(f"{example['record']}, error = {err_ms:.1f} ms", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def main() -> None:
    """Build the point-template baseline and plot representative failures."""
    train_records, eval_records = pick_training_and_eval_records()
    print("Training records:", train_records)

    point_template = build_point_t_template(train_records)
    errors_ms, examples = evaluate_point_baseline(point_template, eval_records)
    _ = errors_ms

    if len(examples) == 0:
        raise RuntimeError("No baseline examples found.")

    selected = select_examples(examples, n_examples=4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, example in zip(axes, selected):
        plot_example_panel(ax, example)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUT_FILE}")


if __name__ == "__main__":
    main()