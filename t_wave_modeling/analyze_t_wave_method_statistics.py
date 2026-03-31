"""Analyze QT T-wave method statistics and create a comparison figure.

This script compares three QT-based T-wave detection methods:

1. Point-template baseline
2. Single trajectory-template model
3. Clustered trajectory-template model

It computes:
- global timing-error summary statistics
- paired Wilcoxon tests on absolute timing errors
- per-record mean absolute errors

It also saves:
- a JSON report with all statistics
- a figure with absolute-error boxplots and per-record MAE curves
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from t_wave_modeling_utils import (
    build_clustered_traj_templates,
    build_point_t_template,
    build_single_traj_template,
    compute_zs,
    detect_clustered_traj_from_cached,
    detect_point_t_from_cached,
    detect_single_traj_from_cached,
    load_qt_record_500hz,
    pair_r_with_explicit_t,
    parse_explicit_r_t,
    pick_training_and_eval_records,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSON_DIR = PROJECT_ROOT / "json_reports"
FIGURE_DIR = PROJECT_ROOT / "generated_outputs" / "figures"

JSON_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = JSON_DIR / "t_method_statistics_report.json"
OUT_FIG = FIGURE_DIR / "t_method_statistics_summary.png"

# Optional restricted evaluation subset for faster runs.
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


def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    """Summarize timing errors in milliseconds.

    Args:
        errors: Timing error array in milliseconds.

    Returns:
        Dictionary with summary statistics.
    """
    errors = np.asarray(errors, dtype=float)

    return {
        "n": int(len(errors)),
        "bias_ms": float(np.mean(errors)),
        "std_ms": float(np.std(errors)),
        "mae_ms": float(np.mean(np.abs(errors))),
        "rmse_ms": float(np.sqrt(np.mean(errors ** 2))),
        "median_abs_ms": float(np.median(np.abs(errors))),
    }


def paired_wilcoxon_abs(err_a: np.ndarray, err_b: np.ndarray) -> Dict[str, float]:
    """Run a paired Wilcoxon test on absolute timing errors.

    Args:
        err_a: First paired error array in milliseconds.
        err_b: Second paired error array in milliseconds.

    Returns:
        Dictionary with test summary values.

    Raises:
        ValueError: If the paired arrays do not have the same length.
    """
    a = np.abs(np.asarray(err_a, dtype=float))
    b = np.abs(np.asarray(err_b, dtype=float))

    if len(a) != len(b):
        raise ValueError("Paired arrays must have the same length.")

    diff = a - b
    if np.allclose(diff, 0):
        return {
            "n": int(len(a)),
            "p_value": 1.0,
            "statistic": 0.0,
            "median_abs_improvement_ms": 0.0,
        }

    statistic, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "n": int(len(a)),
        "p_value": float(p_value),
        "statistic": float(statistic),
        "median_abs_improvement_ms": float(np.median(a - b)),
    }


def choose_eval_records(eval_records_all: List[str]) -> List[str]:
    """Choose the QT evaluation records used for the comparison.

    Args:
        eval_records_all: Full held-out QT evaluation set.

    Returns:
        Selected evaluation record list.
    """
    eval_records = [record for record in FAST_EVAL_RECORDS if record in eval_records_all]
    if len(eval_records) == 0:
        eval_records = eval_records_all[:8]
    return eval_records


def collect_paired_rows(
    point_template: np.ndarray,
    single_template: np.ndarray,
    clustered_templates: np.ndarray,
    eval_records: List[str],
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, float | int]]]:
    """Collect paired timing-error rows across methods.

    Args:
        point_template: Point-template T model.
        single_template: Single trajectory-template model.
        clustered_templates: Clustered trajectory-template matrix.
        eval_records: QT evaluation record names.

    Returns:
        Tuple containing:
            - list of paired beat rows
            - per-record summary dictionary
    """
    paired_rows: List[Dict[str, float | int | str]] = []
    per_record: Dict[str, Dict[str, float | int]] = {}

    for record_name in eval_records:
        print(f"Evaluating {record_name} ...")
        try:
            signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
            _ = used_lead, ann_ext

            r_peaks, t_peaks = parse_explicit_r_t(samples, symbols)
            beats = pair_r_with_explicit_t(r_peaks, t_peaks)

            if len(beats) == 0:
                continue

            _, whitened_states = compute_zs(signal)
            record_rows: List[Dict[str, float | int | str]] = []

            for beat in beats:
                t_gt = int(beat["t_peak"])

                det_point, _, _ = detect_point_t_from_cached(
                    whitened_states,
                    beat["r"],
                    point_template,
                )
                det_single, _, _ = detect_single_traj_from_cached(
                    whitened_states,
                    beat["r"],
                    single_template,
                )
                det_cluster, _, _ = detect_clustered_traj_from_cached(
                    whitened_states,
                    beat["r"],
                    clustered_templates,
                )

                if det_point is None or det_single is None or det_cluster is None:
                    continue

                err_point = 1000.0 * (det_point - t_gt) / 500.0
                err_single = 1000.0 * (det_single - t_gt) / 500.0
                err_cluster = 1000.0 * (det_cluster - t_gt) / 500.0

                row = {
                    "record": record_name,
                    "r": int(beat["r"]),
                    "t_gt": t_gt,
                    "err_point": float(err_point),
                    "err_single": float(err_single),
                    "err_cluster": float(err_cluster),
                }
                paired_rows.append(row)
                record_rows.append(row)

            if len(record_rows) > 0:
                per_record[record_name] = {
                    "point_mae_ms": float(np.mean(np.abs([x["err_point"] for x in record_rows]))),
                    "single_mae_ms": float(np.mean(np.abs([x["err_single"] for x in record_rows]))),
                    "clustered_mae_ms": float(np.mean(np.abs([x["err_cluster"] for x in record_rows]))),
                    "n_beats": int(len(record_rows)),
                }

        except Exception as exc:
            print(f"  skipped {record_name}: {exc}")

    return paired_rows, per_record


def plot_summary_figure(
    err_point: np.ndarray,
    err_single: np.ndarray,
    err_cluster: np.ndarray,
    per_record: Dict[str, Dict[str, float | int]],
) -> None:
    """Plot the summary comparison figure.

    Args:
        err_point: Point-template timing errors in milliseconds.
        err_single: Single-trajectory timing errors in milliseconds.
        err_cluster: Clustered-trajectory timing errors in milliseconds.
        per_record: Per-record MAE summary dictionary.
    """
    records_sorted = sorted(per_record.keys())
    point_mae = [float(per_record[r]["point_mae_ms"]) for r in records_sorted]
    single_mae = [float(per_record[r]["single_mae_ms"]) for r in records_sorted]
    cluster_mae = [float(per_record[r]["clustered_mae_ms"]) for r in records_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].boxplot(
        [np.abs(err_point), np.abs(err_single), np.abs(err_cluster)],
        labels=["point", "single", "clustered"],
        showfliers=False,
    )
    axes[0].set_ylabel("Absolute timing error (ms)")
    axes[0].set_title("Absolute timing error by method")

    x_axis = np.arange(len(records_sorted))
    axes[1].plot(x_axis, point_mae, marker="o", linewidth=1.1, label="point")
    axes[1].plot(x_axis, single_mae, marker="o", linewidth=1.1, label="single")
    axes[1].plot(x_axis, cluster_mae, marker="o", linewidth=1.1, label="clustered")
    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels(records_sorted, rotation=90, fontsize=7)
    axes[1].set_ylabel("Per-record MAE (ms)")
    axes[1].set_title("Per-record mean absolute error")
    axes[1].legend(fontsize=8, frameon=True)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    """Run the T-wave method comparison and save the outputs."""
    train_records, eval_records_all = pick_training_and_eval_records()
    eval_records = choose_eval_records(eval_records_all)

    print("Training records:", train_records)
    print("Evaluation records:", eval_records)

    point_template = build_point_t_template(train_records)
    single_template, _ = build_single_traj_template(train_records)
    clustered_templates, _ = build_clustered_traj_templates(train_records, k=3)

    paired_rows, per_record = collect_paired_rows(
        point_template=point_template,
        single_template=single_template,
        clustered_templates=clustered_templates,
        eval_records=eval_records,
    )

    if len(paired_rows) == 0:
        raise RuntimeError("No paired comparisons could be computed.")

    err_point = np.array([row["err_point"] for row in paired_rows], dtype=float)
    err_single = np.array([row["err_single"] for row in paired_rows], dtype=float)
    err_cluster = np.array([row["err_cluster"] for row in paired_rows], dtype=float)

    stats = {
        "point_template": summarize_errors(err_point),
        "single_trajectory": summarize_errors(err_single),
        "clustered_trajectory": summarize_errors(err_cluster),
        "paired_tests": {
            "point_vs_single": paired_wilcoxon_abs(err_point, err_single),
            "point_vs_clustered": paired_wilcoxon_abs(err_point, err_cluster),
            "single_vs_clustered": paired_wilcoxon_abs(err_single, err_cluster),
        },
        "per_record": per_record,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    plot_summary_figure(err_point, err_single, err_cluster, per_record)

    print("\nSaved:")
    print(" ", OUT_JSON)
    print(" ", OUT_FIG)

    print("\nSummary:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()