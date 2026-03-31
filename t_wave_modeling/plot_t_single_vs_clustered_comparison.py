"""Plot the comparison between single and clustered T-wave trajectory templates.

This script builds:
1. A single trajectory-template T model
2. A clustered trajectory-template T model with K = 3

It then evaluates both methods on held-out QT records and creates a summary
figure with:
- timing-error histograms
- absolute-error boxplots

A small text summary with standard deviation and mean absolute error is added
to the boxplot panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from t_wave_modeling_utils import (
    build_clustered_traj_templates,
    build_single_traj_template,
    evaluate_single_vs_clustered,
    pick_training_and_eval_records,
    summarize_errors,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "t_single_vs_clustered_templates.png"


def plot_comparison(err_single: np.ndarray, err_clustered: np.ndarray) -> None:
    """Plot the comparison figure for single and clustered templates.

    Args:
        err_single: Timing errors of the single trajectory-template method.
        err_clustered: Timing errors of the clustered trajectory-template method.
    """
    stats_single = summarize_errors(err_single)
    stats_clustered = summarize_errors(err_clustered)

    if stats_single is None or stats_clustered is None:
        raise RuntimeError("Could not compute single versus clustered comparison.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bins = np.linspace(
        min(np.min(err_single), np.min(err_clustered)),
        max(np.max(err_single), np.max(err_clustered)),
        40,
    )

    axes[0].hist(err_single, bins=bins, alpha=0.65, label="single trajectory")
    axes[0].hist(err_clustered, bins=bins, alpha=0.65, label="clustered trajectory")
    axes[0].set_xlabel("Timing error (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Timing error distributions")
    axes[0].legend(fontsize=8, frameon=True)

    abs_single = np.abs(err_single)
    abs_clustered = np.abs(err_clustered)

    axes[1].boxplot(
        [abs_single, abs_clustered],
        labels=["single", "clustered"],
        showfliers=False,
    )
    axes[1].set_ylabel("Absolute timing error (ms)")
    axes[1].set_title("Absolute error comparison")

    text = (
        f"Single:    STD = {stats_single['std']:.2f} ms, "
        f"MAE = {stats_single['mae']:.2f} ms\n"
        f"Clustered: STD = {stats_clustered['std']:.2f} ms, "
        f"MAE = {stats_clustered['mae']:.2f} ms"
    )
    axes[1].text(
        0.5,
        0.02,
        text,
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUT_FILE}")


def main() -> None:
    """Build both trajectory-template variants and plot the comparison."""
    train_records, eval_records = pick_training_and_eval_records()
    print("Training records:", train_records)

    single_template, _ = build_single_traj_template(train_records)
    clustered_templates, _ = build_clustered_traj_templates(train_records, k=3)

    err_single, err_clustered = evaluate_single_vs_clustered(
        single_template,
        clustered_templates,
        eval_records,
    )

    plot_comparison(err_single, err_clustered)


if __name__ == "__main__":
    main()