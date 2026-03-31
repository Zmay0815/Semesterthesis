"""Plot one MIT example with detected P, R, and T waves and their scores.

This script loads one MIT-BIH record, runs three detector branches, and plots
a qualitative example window:

1. R detection using the non-whitened MIT-derived R template and true LCR
2. P detection using a whitened point-template score
3. T detection using the dynamic trajectory score from the thesis workflow

The final figure shows:
- ECG with detected P, R, and T waves
- normalized R score
- normalized P score
- normalized T dynamic score

The normalization is only used for visualization.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import lmlib as lm
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import find_peaks, resample

FS = 500
EPS = 1e-6

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECORD = "102"
START_SECOND = 8.0
WINDOW_SECONDS = 6.0

OUT_FILE = OUTPUT_DIR / f"mit_example_lcr_trajectory_{RECORD}.png"

R_POLYDEGREE = 2
R_L_SIDE = 15
R_G = 40
R_THRESH_PERCENTILE = 97.5
R_REFRACTORY_SEC = 0.35
R_TEMPLATE_FILE = NPZ_DIR / "r_template_mitbih_500hz.npy"

P_POLYDEGREE = 2
P_L_SIDE = 20
P_G = 120
P_WINDOW_SEC = (-0.25, -0.08)
P_TEMPLATE_FILE = NPZ_DIR / "p_template_mitbih_500hz.npy"

T_POLYDEGREE = 2
T_L_SIDE = 40
T_G = 10
T_WINDOW_SEC = (0.10, 0.50)
T_HALF_WIDTH = 20
T_TEMPLATE_FILE = NPZ_DIR / "t_template_dynamic_iterative.npy"

T_BIAS_CORRECTION_MS = 66
T_BIAS_CORRECTION_SAMPLES = int(T_BIAS_CORRECTION_MS / 1000 * FS)


def resolve_mit_database_dir() -> Path:
    """Find the MIT-BIH database folder.

    Returns:
        Path to the MIT-BIH database directory.

    Raises:
        FileNotFoundError: If no valid MIT-BIH database directory was found.
    """
    env_path = os.environ.get("MIT_BIH_DIR")

    candidates = [
        Path(env_path) if env_path else None,
        SCRIPT_DIR / "mit-bih-arrhythmia-database-1.0.0",
        PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0",
        PROJECT_ROOT / "datasets" / "mit-bih-arrhythmia-database-1.0.0",
        PROJECT_ROOT / "datasets_and_preprocessing" / "mit-bih-arrhythmia-database-1.0.0",
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


def load_existing_file(path: Path, label: str) -> Path:
    """Validate that a required file exists.

    Args:
        path: File path.
        label: Human-readable label.

    Returns:
        Existing file path.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_mit_record(record_name: str, fs_target: int) -> Tuple[np.ndarray, int]:
    """Load one MIT-BIH record and resample it to the target rate.

    Args:
        record_name: MIT-BIH record name.
        fs_target: Desired sampling rate.

    Returns:
        Tuple containing:
            - ECG signal at the target rate
            - original sampling rate
    """
    record = wfdb.rdrecord(str(MIT_PATH / record_name))
    signal = record.p_signal[:, 0].astype(np.float64)
    fs_orig = int(record.fs)

    if fs_orig != fs_target:
        new_len = int(len(signal) * fs_target / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)

    return signal, fs_orig


def run_alssm(signal: np.ndarray, polydegree: int, l_side: int, g: float) -> np.ndarray:
    """Estimate ALSSM states for one ECG signal.

    Args:
        signal: ECG signal.
        polydegree: Polynomial degree.
        l_side: Half-width of the ALSSM support window.
        g: Segment weighting parameter.

    Returns:
        State sequence.
    """
    model = lm.AlssmPoly(poly_degree=polydegree)
    segment = lm.Segment(a=-l_side, b=l_side, direction=lm.FORWARD, g=g)
    cost = lm.CompositeCost((model,), (segment,), [[1]])
    rls = lm.create_rls(cost, multi_channel_set=False, steady_state=True)
    rls.filter(signal)
    return rls.minimize_x()


def whiten_states(states: np.ndarray) -> np.ndarray:
    """Whiten an ALSSM state sequence.

    Args:
        states: Raw ALSSM states.

    Returns:
        Whitened state sequence.
    """
    covariance = np.cov(states.T)
    covariance_reg = covariance + EPS * np.eye(covariance.shape[0])
    chol = np.linalg.cholesky(covariance_reg)
    whitening_matrix = np.linalg.solve(chol, np.eye(chol.shape[0]))
    return (whitening_matrix @ states.T).T


def build_trajectory(
    whitened_states: np.ndarray,
    center: int,
    half_width: int,
) -> Optional[np.ndarray]:
    """Build one stacked trajectory vector around a center sample.

    Args:
        whitened_states: Whitened ALSSM states.
        center: Candidate center index.
        half_width: Half-width of the trajectory window.

    Returns:
        Flattened trajectory vector, or None if the window is invalid.
    """
    if center - half_width < 0 or center + half_width >= len(whitened_states):
        return None
    return whitened_states[center - half_width : center + half_width + 1].reshape(-1)


def normalize_curve(values: np.ndarray) -> np.ndarray:
    """Normalize a curve to [0, 1] for plotting only.

    Args:
        values: Input curve.

    Returns:
        Normalized curve.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values)

    return (values - vmin) / (vmax - vmin)


def safe_normalize_with_nans(values: np.ndarray) -> np.ndarray:
    """Normalize a curve while preserving NaN values outside valid windows.

    Args:
        values: Input curve with NaNs.

    Returns:
        Normalized curve with NaNs preserved.
    """
    values = np.asarray(values, dtype=float)

    if not np.any(np.isfinite(values)):
        return np.full_like(values, np.nan, dtype=float)

    fill_value = np.nanmin(values[np.isfinite(values)])
    normalized = normalize_curve(np.nan_to_num(values, nan=fill_value))
    return np.where(np.isfinite(values), normalized, np.nan)


def detect_r(
    signal: np.ndarray,
    r_template: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect R peaks with the MIT-derived non-whitened R detector.

    Args:
        signal: ECG signal.
        r_template: R template.

    Returns:
        Tuple containing:
            - detected R indices
            - R LCR curve
    """
    xhat_r = run_alssm(signal, R_POLYDEGREE, R_L_SIDE, R_G)

    # Intentionally not whitened, matching the thesis R-template workflow.
    states_r = xhat_r

    distance = np.linalg.norm(states_r - r_template, axis=1) ** 2
    reference_energy = np.linalg.norm(r_template) ** 2
    lcr = -0.5 * np.log(np.maximum(distance / reference_energy, 1e-12))

    threshold = np.percentile(lcr, R_THRESH_PERCENTILE)
    r_detected, _ = find_peaks(
        lcr,
        height=threshold,
        distance=int(R_REFRACTORY_SEC * FS),
    )

    return r_detected, lcr


def detect_p(
    signal: np.ndarray,
    r_detected: np.ndarray,
    p_template: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect P peaks using the pointwise whitened-state score.

    Args:
        signal: ECG signal.
        r_detected: Detected R indices.
        p_template: P template.

    Returns:
        Tuple containing:
            - detected P indices
            - full P score curve with NaNs outside search windows
    """
    xhat_p = run_alssm(signal, P_POLYDEGREE, P_L_SIDE, P_G)
    states_p = whiten_states(xhat_p)

    p_score_full = np.full(len(signal), np.nan, dtype=float)
    p_detected = []

    for r_idx in r_detected:
        start = r_idx + int(P_WINDOW_SEC[0] * FS)
        end = r_idx + int(P_WINDOW_SEC[1] * FS)

        if start < 0 or end >= len(signal) or end <= start:
            continue

        state_segment = states_p[start:end]
        distances = np.linalg.norm(state_segment - p_template, axis=1) ** 2
        lcr_like = -0.5 * distances

        p_score_full[start:end] = lcr_like

        best_local = int(np.argmax(lcr_like))
        p_idx = int(start + best_local)
        p_detected.append(p_idx)

    return np.array(p_detected, dtype=int), p_score_full


def detect_t_dynamic(
    signal: np.ndarray,
    r_detected: np.ndarray,
    t_template: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect T peaks with the dynamic trajectory score.

    This follows the thesis workflow, including the practical +66 ms bias
    correction after the minimum-cost center was found.

    Args:
        signal: ECG signal.
        r_detected: Detected R indices.
        t_template: Dynamic T trajectory template.

    Returns:
        Tuple containing:
            - detected T indices
            - full T score curve with NaNs outside search windows
    """
    xhat_t = run_alssm(signal, T_POLYDEGREE, T_L_SIDE, T_G)
    states_t = whiten_states(xhat_t)

    t_score_full = np.full(len(signal), np.nan, dtype=float)
    t_detected = []

    for r_idx in r_detected:
        t_start = r_idx + int(T_WINDOW_SEC[0] * FS)
        t_end = r_idx + int(T_WINDOW_SEC[1] * FS)

        if t_end >= len(signal) or t_end <= t_start:
            continue

        best_cost = np.inf
        best_t = None

        for t_idx in range(t_start, t_end):
            traj = build_trajectory(states_t, t_idx, T_HALF_WIDTH)
            if traj is None:
                continue

            distance = np.linalg.norm(traj - t_template) ** 2

            # This is used only as a display score, not as a formal LCR.
            t_score_full[t_idx] = -0.5 * distance

            if distance < best_cost:
                best_cost = distance
                best_t = t_idx

        if best_t is not None:
            corrected_t = best_t + T_BIAS_CORRECTION_SAMPLES
            if 0 <= corrected_t < len(signal):
                t_detected.append(corrected_t)

    return np.array(t_detected, dtype=int), t_score_full


def build_plot_window(
    signal: np.ndarray,
    r_detected: np.ndarray,
    p_detected: np.ndarray,
    t_detected: np.ndarray,
    r_lcr: np.ndarray,
    p_score_full: np.ndarray,
    t_score_full: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build the plotting window and normalize curves for display.

    Args:
        signal: ECG signal.
        r_detected: Detected R indices.
        p_detected: Detected P indices.
        t_detected: Detected T indices.
        r_lcr: R LCR curve.
        p_score_full: P score curve.
        t_score_full: T score curve.

    Returns:
        Dictionary with plot-ready arrays.
    """
    start = int(START_SECOND * FS)
    end = int((START_SECOND + WINDOW_SECONDS) * FS)
    end = min(end, len(signal))

    time_axis = np.arange(start, end) / FS

    r_in = r_detected[(r_detected >= start) & (r_detected < end)]
    p_in = p_detected[(p_detected >= start) & (p_detected < end)]
    t_in = t_detected[(t_detected >= start) & (t_detected < end)]

    r_lcr_win = r_lcr[start:end]
    p_score_win = p_score_full[start:end]
    t_score_win = t_score_full[start:end]

    r_lcr_plot = normalize_curve(r_lcr_win)
    p_score_plot = safe_normalize_with_nans(p_score_win)
    t_score_plot = safe_normalize_with_nans(t_score_win)

    return {
        "start": np.array([start]),
        "end": np.array([end]),
        "time_axis": time_axis,
        "r_in": r_in,
        "p_in": p_in,
        "t_in": t_in,
        "r_lcr_plot": r_lcr_plot,
        "p_score_plot": p_score_plot,
        "t_score_plot": t_score_plot,
    }


def plot_result(
    signal: np.ndarray,
    plot_data: Dict[str, np.ndarray],
) -> None:
    """Plot and save the qualitative MIT example figure.

    Args:
        signal: ECG signal.
        plot_data: Plot-ready arrays.
    """
    start = int(plot_data["start"][0])
    end = int(plot_data["end"][0])
    time_axis = plot_data["time_axis"]

    r_in = plot_data["r_in"]
    p_in = plot_data["p_in"]
    t_in = plot_data["t_in"]

    r_lcr_plot = plot_data["r_lcr_plot"]
    p_score_plot = plot_data["p_score_plot"]
    t_score_plot = plot_data["t_score_plot"]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3, 1.3, 1.3]},
    )

    axes[0].plot(time_axis, signal[start:end], color="black", linewidth=1.2, label="ECG")

    if len(r_in) > 0:
        axes[0].scatter(r_in / FS, signal[r_in], color="red", s=50, label="R detected")
    if len(p_in) > 0:
        axes[0].scatter(p_in / FS, signal[p_in], color="blue", s=45, label="P detected")
    if len(t_in) > 0:
        axes[0].scatter(t_in / FS, signal[t_in], color="green", s=45, label="T detected")

    axes[0].set_title(f"MIT-BIH record {RECORD}, detected waves only")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)

    axes[1].plot(time_axis, r_lcr_plot, color="red", linewidth=1.4)
    axes[1].set_ylabel("R LCR\n(norm)")
    axes[1].set_title("R score")

    axes[2].plot(time_axis, p_score_plot, color="blue", linewidth=1.4)
    axes[2].set_ylabel("P LCR\n(norm)")
    axes[2].set_title("P score")

    axes[3].plot(time_axis, t_score_plot, color="green", linewidth=1.4)
    axes[3].set_ylabel("T score\n(norm)")
    axes[3].set_title("T dynamic score")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUT_FILE}")


def main() -> None:
    """Run the MIT example figure pipeline."""
    r_template = np.load(load_existing_file(R_TEMPLATE_FILE, "R template"))
    p_template = np.load(load_existing_file(P_TEMPLATE_FILE, "P template"))
    t_template = np.load(load_existing_file(T_TEMPLATE_FILE, "dynamic T template"))

    signal, fs_orig = load_mit_record(RECORD, FS)
    print(f"Original sampling rate: {fs_orig} Hz")

    r_detected, r_lcr = detect_r(signal, r_template)
    p_detected, p_score_full = detect_p(signal, r_detected, p_template)
    t_detected, t_score_full = detect_t_dynamic(signal, r_detected, t_template)

    plot_data = build_plot_window(
        signal=signal,
        r_detected=r_detected,
        p_detected=p_detected,
        t_detected=t_detected,
        r_lcr=r_lcr,
        p_score_full=p_score_full,
        t_score_full=t_score_full,
    )
    plot_result(signal, plot_data)


if __name__ == "__main__":
    main()