"""Evaluate MIT R-template generalization across multiple MIT-BIH records.

This script evaluates the main MIT-derived R template on several MIT-BIH test
records. The workflow follows the original thesis logic:

1. Load one ECG record
2. Resample it to 500 Hz if needed
3. Estimate ALSSM states
4. Use the saved non-whitened MIT R template
5. Compute the pointwise distance and LCR score
6. Detect candidate R peaks by percentile thresholding
7. Compare predicted peaks with annotated beats using a fixed tolerance

The output is a compact per-record summary and an overall summary across all
selected records.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lmlib as lm
import numpy as np
import wfdb
from scipy.signal import find_peaks, resample

FS_TARGET = 500

R_PARAMS = {
    "polydegree": 2,
    "l_side": 15,
    "g": 40,
}

RECORDS = ["100", "101", "106", "103", "104"]

VALID_BEAT_SYMBOLS = {"N", "L", "R", "A", "V", "F", "e", "j"}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"

TEMPLATE_FILE = NPZ_DIR / "r_template_mitbih_500hz.npy"


def resolve_mit_database_dir() -> Path:
    """Find the MIT-BIH database folder.

    Returns:
        Path to the MIT-BIH database directory.

    Raises:
        FileNotFoundError: If no valid MIT-BIH directory was found.
    """
    env_path = os.environ.get("MIT_BIH_DIR")

    candidates = [
        Path(env_path) if env_path else None,
        SCRIPT_DIR / "mit-bih-arrhythmia-database-1.0.0",
        PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0",
        PROJECT_ROOT / "datasets" / "mit-bih-arrhythmia-database-1.0.0",
    ]

    valid_candidates = [c for c in candidates if c is not None]

    for candidate in valid_candidates:
        if (candidate / "100.hea").exists():
            return candidate

    searched = "\n".join(str(c) for c in valid_candidates)
    raise FileNotFoundError(
        "Could not find the MIT-BIH database folder.\n"
        "Set MIT_BIH_DIR or place the database in one of these locations:\n"
        f"{searched}"
    )


MIT_PATH = resolve_mit_database_dir()


def load_r_template() -> np.ndarray:
    """Load the saved MIT-derived R template.

    Returns:
        Saved R template array.

    Raises:
        FileNotFoundError: If the template file is missing.
    """
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find R template file: {TEMPLATE_FILE}"
        )
    return np.load(TEMPLATE_FILE)


def load_mit_record_500hz(record_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one MIT-BIH record and resample it to 500 Hz if needed.

    Args:
        record_name: MIT-BIH record name.

    Returns:
        Tuple containing:
            - signal at 500 Hz
            - valid beat annotations at 500 Hz
    """
    record_path = MIT_PATH / record_name

    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, 0].astype(np.float64)
    fs_orig = int(record.fs)

    ann = wfdb.rdann(str(record_path), "atr")

    valid_beats: List[int] = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in VALID_BEAT_SYMBOLS:
            valid_beats.append(int(sample))

    beat_indices = np.array(valid_beats, dtype=int)

    if fs_orig != FS_TARGET:
        new_len = int(len(signal) * FS_TARGET / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)
        beat_indices = (beat_indices * FS_TARGET / fs_orig).astype(int)

    return signal, beat_indices


def run_alssm(signal: np.ndarray) -> np.ndarray:
    """Estimate ALSSM states for one ECG signal.

    Args:
        signal: ECG signal.

    Returns:
        State sequence.
    """
    model = lm.AlssmPoly(poly_degree=R_PARAMS["polydegree"])
    segment = lm.Segment(
        a=-R_PARAMS["l_side"],
        b=R_PARAMS["l_side"],
        direction=lm.FORWARD,
        g=R_PARAMS["g"],
    )
    cost = lm.CompositeCost((model,), (segment,), [[1]])

    rls = lm.create_rls(cost, multi_channel_set=False, steady_state=True)
    rls.filter(signal)
    xhat = rls.minimize_x()

    return xhat


def detect_r_peaks(signal: np.ndarray, r_template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect R peaks using the saved non-whitened MIT R template.

    Args:
        signal: ECG signal.
        r_template: Saved MIT R template.

    Returns:
        Tuple containing:
            - detected peak indices
            - LCR curve
    """
    states = run_alssm(signal)

    # Intentionally no whitening here, matching the original thesis workflow.
    distances = np.linalg.norm(states - r_template, axis=1) ** 2
    reference_energy = np.linalg.norm(r_template) ** 2

    lcr = -0.5 * np.log(np.maximum(distances / reference_energy, 1e-12))
    threshold = np.percentile(lcr, 97.5)

    peaks, _ = find_peaks(
        lcr,
        height=threshold,
        distance=int(0.35 * FS_TARGET),
    )

    return peaks, lcr


def evaluate_record(
    record_name: str,
    r_template: np.ndarray,
    tolerance_sec: float = 0.05,
) -> Dict[str, float]:
    """Evaluate the R detector on one MIT-BIH record.

    Args:
        record_name: MIT-BIH record name.
        r_template: Saved MIT R template.
        tolerance_sec: Matching tolerance in seconds.

    Returns:
        Dictionary with the per-record metrics.
    """
    signal, valid_beats = load_mit_record_500hz(record_name)
    pred_peaks, lcr = detect_r_peaks(signal, r_template)
    _ = lcr

    tolerance = int(tolerance_sec * FS_TARGET)

    true_positives = 0
    timing_errors = []

    for gt_peak in valid_beats:
        diffs = np.abs(pred_peaks - gt_peak)
        if len(diffs) > 0 and np.min(diffs) <= tolerance:
            true_positives += 1
            best_idx = int(np.argmin(diffs))
            timing_errors.append(pred_peaks[best_idx] - gt_peak)

    recall = true_positives / len(valid_beats) if len(valid_beats) > 0 else 0.0
    precision = true_positives / len(pred_peaks) if len(pred_peaks) > 0 else 0.0

    timing_ms = (
        np.array(timing_errors, dtype=np.float64) / FS_TARGET * 1000.0
        if len(timing_errors) > 0
        else np.array([], dtype=np.float64)
    )

    mean_error_ms = float(np.mean(timing_ms)) if len(timing_ms) > 0 else 0.0
    std_error_ms = float(np.std(timing_ms)) if len(timing_ms) > 0 else 0.0

    return {
        "gt_count": float(len(valid_beats)),
        "pred_count": float(len(pred_peaks)),
        "recall": float(recall),
        "precision": float(precision),
        "mean_error_ms": mean_error_ms,
        "std_error_ms": std_error_ms,
    }


def print_record_result(record_name: str, result: Dict[str, float]) -> None:
    """Print the result for one record.

    Args:
        record_name: MIT-BIH record name.
        result: Per-record metric dictionary.
    """
    print("\n==============================")
    print("Record:", record_name)
    print("GT:", int(result["gt_count"]))
    print("Pred:", int(result["pred_count"]))
    print("Recall:", result["recall"])
    print("Precision:", result["precision"])
    print("Mean error (ms):", result["mean_error_ms"])
    print("Std error (ms):", result["std_error_ms"])


def print_overall_results(results: Sequence[Dict[str, float]]) -> None:
    """Print the overall mean and standard deviation across records.

    Args:
        results: Per-record result dictionaries.
    """
    recalls = np.array([r["recall"] for r in results], dtype=np.float64)
    precisions = np.array([r["precision"] for r in results], dtype=np.float64)
    mean_errors = np.array([r["mean_error_ms"] for r in results], dtype=np.float64)
    std_errors = np.array([r["std_error_ms"] for r in results], dtype=np.float64)

    print("\n==============================")
    print("OVERALL RESULTS")
    print("Mean Recall:", np.mean(recalls))
    print("Std Recall:", np.std(recalls))
    print("Mean Precision:", np.mean(precisions))
    print("Std Precision:", np.std(precisions))
    print("Mean Timing Error (ms):", np.mean(mean_errors))
    print("Std Timing Error (ms):", np.mean(std_errors))


def main() -> None:
    """Run the MIT R-template generalization evaluation."""
    r_template = load_r_template()

    results = []
    for record_name in RECORDS:
        result = evaluate_record(record_name, r_template)
        print_record_result(record_name, result)
        results.append(result)

    print_overall_results(results)


if __name__ == "__main__":
    main()