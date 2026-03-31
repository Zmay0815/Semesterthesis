"""Plot QRS detection examples with raw and whitened ALSSM states.

This script loads one MIT-BIH record, applies the non-whitened MIT R-template
detector, ranks detected QRS candidates, and saves paired figures for the
selected examples.

For each selected QRS candidate, two figures are created:
1. One with the raw ALSSM state trajectory
2. One with the whitened ALSSM state trajectory

Both figures show:
- ECG signal
- state trajectory
- pointwise cost
- log-cost-ratio (LCR) score
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import lmlib as lm
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import find_peaks, resample

FS = 500
RECORD = "100"
EPS = 1e-6

POLYDEGREE = 2
L_SIDE = 15
G = 40

DISPLAY_MARGIN_SEC = 0.7
N_EXAMPLES = 1

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures" / "figure11_qrs_raw_and_whitened"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

R_TEMPLATE_FILE = NPZ_DIR / "r_template_mitbih_500hz.npy"


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


def load_signal_500hz(record_name: str) -> np.ndarray:
    """Load one MIT-BIH signal and resample it to 500 Hz if needed.

    Args:
        record_name: MIT-BIH record name.

    Returns:
        ECG signal at 500 Hz.
    """
    record_path = MIT_PATH / record_name
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, 0].astype(np.float64)
    fs_orig = int(record.fs)

    if fs_orig != FS:
        new_len = int(len(signal) * FS / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)

    return signal


def load_r_template() -> np.ndarray:
    """Load the saved MIT-derived R template.

    Returns:
        R template array.

    Raises:
        FileNotFoundError: If the template file is missing.
    """
    if not R_TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {R_TEMPLATE_FILE}")
    return np.load(R_TEMPLATE_FILE)


def run_alssm(signal: np.ndarray) -> np.ndarray:
    """Estimate ALSSM states for one ECG signal.

    Args:
        signal: ECG signal.

    Returns:
        State sequence.
    """
    model = lm.AlssmPoly(poly_degree=POLYDEGREE)
    segment = lm.Segment(
        a=-L_SIDE,
        b=L_SIDE,
        direction=lm.FORWARD,
        g=G,
    )

    cost_model = lm.CompositeCost((model,), (segment,), [[1]])
    rls = lm.create_rls(cost_model, multi_channel_set=False, steady_state=True)
    rls.filter(signal)

    return rls.minimize_x()


def whiten_states(states: np.ndarray) -> np.ndarray:
    """Whiten the ALSSM state sequence.

    Args:
        states: Raw state sequence.

    Returns:
        Whitened state sequence.
    """
    covariance = np.cov(states.T)
    covariance_reg = covariance + EPS * np.eye(covariance.shape[0])
    chol = np.linalg.cholesky(covariance_reg)
    whitening_matrix = np.linalg.solve(chol, np.eye(chol.shape[0]))
    return (whitening_matrix @ states.T).T


def detect_r_candidates(
    raw_states: np.ndarray,
    r_template: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect candidate R peaks with the real non-whitened R detector.

    Args:
        raw_states: Raw ALSSM state sequence.
        r_template: Saved MIT R template.

    Returns:
        Tuple containing:
            - detected R peak indices
            - pointwise cost curve
            - LCR curve
    """
    distance = np.linalg.norm(raw_states - r_template, axis=1) ** 2
    reference_energy = np.linalg.norm(r_template) ** 2
    lcr = -0.5 * np.log(np.maximum(distance / reference_energy, 1e-12))

    threshold = np.percentile(lcr, 97.5)
    peaks, _ = find_peaks(
        lcr,
        height=threshold,
        distance=int(0.35 * FS),
    )

    return peaks, distance, lcr


def rank_candidates(
    signal: np.ndarray,
    lcr: np.ndarray,
    r_detected: np.ndarray,
    display_margin_sec: float,
) -> List[Dict[str, float]]:
    """Rank detected QRS candidates for figure selection.

    Args:
        signal: ECG signal.
        lcr: LCR curve.
        r_detected: Detected R peak indices.
        display_margin_sec: Display margin around each candidate in seconds.

    Returns:
        Ranked list of candidate dictionaries.
    """
    candidates: List[Dict[str, float]] = []
    margin = int(display_margin_sec * FS)

    for idx, r_idx in enumerate(r_detected):
        left = r_idx - margin
        right = r_idx + margin

        if left < 0 or right >= len(signal):
            continue

        local_lcr = lcr[left:right]
        local_signal = signal[left:right]

        score = float(
            np.max(local_lcr) + 0.8 * (signal[r_idx] - np.median(local_signal))
        )

        candidates.append(
            {
                "det_idx": float(idx),
                "r": float(r_idx),
                "score": score,
                "max_lcr": float(np.max(local_lcr)),
            }
        )

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def save_figure(
    t_axis: np.ndarray,
    signal_segment: np.ndarray,
    state1: np.ndarray,
    state2: np.ndarray,
    state3: np.ndarray,
    cost_segment: np.ndarray,
    lcr_segment: np.ndarray,
    full_signal: np.ndarray,
    r_idx: int,
    state_mode: str,
    out_file: Path,
) -> None:
    """Save one QRS diagnostic figure.

    Args:
        t_axis: Time axis for the segment.
        signal_segment: ECG segment.
        state1: First state trajectory.
        state2: Second state trajectory.
        state3: Third state trajectory.
        cost_segment: Pointwise cost segment.
        lcr_segment: LCR segment.
        full_signal: Full ECG signal.
        r_idx: Detected R peak index.
        state_mode: Either ``"raw"`` or ``"whitened"``.
        out_file: Output image path.
    """
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.25, 1.0, 1.0]},
    )

    axes[0].plot(t_axis, signal_segment, color="black", linewidth=1.2, label="ECG")
    axes[0].scatter(r_idx / FS, full_signal[r_idx], color="red", s=45, label="R peak", zorder=5)
    axes[0].axvline(r_idx / FS, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_title("ECG signal", fontsize=10)
    axes[0].set_ylabel("Amplitude (mV)", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)

    if state_mode == "raw":
        axes[1].plot(t_axis, state1, color="#1f77b4", linewidth=1.2, label=r"State 1 ($x_1$)")
        axes[1].plot(t_axis, state2, color="#ff7f0e", linewidth=1.2, label=r"State 2 ($x_2$)")
        axes[1].plot(t_axis, state3, color="#2ca02c", linewidth=1.2, label=r"State 3 ($x_3$)")
        axes[1].set_title("ALSSM state trajectory", fontsize=10)
    else:
        axes[1].plot(t_axis, state1, color="#1f77b4", linewidth=1.2, label=r"State 1 ($z_1$)")
        axes[1].plot(t_axis, state2, color="#ff7f0e", linewidth=1.2, label=r"State 2 ($z_2$)")
        axes[1].plot(t_axis, state3, color="#2ca02c", linewidth=1.2, label=r"State 3 ($z_3$)")
        axes[1].set_title("Whitened ALSSM state trajectory", fontsize=10)

    axes[1].axvline(r_idx / FS, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("State value", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=8, frameon=True)

    axes[2].plot(t_axis, cost_segment, color="darkgreen", linewidth=1.4, label=r"Cost $d(k)$")
    axes[2].axvline(r_idx / FS, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[2].set_title("Pointwise cost", fontsize=10)
    axes[2].set_ylabel(r"$d(k)$", fontsize=9)
    axes[2].legend(loc="upper right", fontsize=8, frameon=True)

    axes[3].plot(t_axis, lcr_segment, color="purple", linewidth=1.4, label="LCR score")
    axes[3].axvline(r_idx / FS, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[3].set_title("Log-cost-ratio detection score", fontsize=10)
    axes[3].set_ylabel("LCR score", fontsize=9)
    axes[3].set_xlabel("Time (s)", fontsize=9)
    axes[3].legend(loc="upper right", fontsize=8, frameon=True)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create raw and whitened QRS detection figures."""
    signal = load_signal_500hz(RECORD)
    r_template = load_r_template()

    raw_states = run_alssm(signal)
    whitened_states = whiten_states(raw_states)

    r_detected, distance, lcr = detect_r_candidates(raw_states, r_template)

    candidates = rank_candidates(
        signal=signal,
        lcr=lcr,
        r_detected=r_detected,
        display_margin_sec=DISPLAY_MARGIN_SEC,
    )

    if len(candidates) == 0:
        raise RuntimeError("No valid QRS candidates found.")

    selected = candidates[:N_EXAMPLES]

    print("\nSelected QRS candidates:")
    for i, candidate in enumerate(selected):
        print(
            f"{i:02d}: detected_index={int(candidate['det_idx'])}, "
            f"R={int(candidate['r'])}, score={candidate['score']:.4f}, "
            f"maxLCR={candidate['max_lcr']:.4f}"
        )

    margin = int(DISPLAY_MARGIN_SEC * FS)

    for rank, candidate in enumerate(selected):
        r_idx = int(candidate["r"])

        left = r_idx - margin
        right = r_idx + margin

        t_axis = np.arange(left, right) / FS
        signal_segment = signal[left:right]

        x1 = raw_states[left:right, 0]
        x2 = raw_states[left:right, 1]
        x3 = raw_states[left:right, 2]

        z1 = whitened_states[left:right, 0]
        z2 = whitened_states[left:right, 1]
        z3 = whitened_states[left:right, 2]

        cost_segment = distance[left:right]
        lcr_segment = lcr[left:right]

        raw_file = OUTPUT_DIR / f"figure11_qrs_raw_{rank:02d}_R{r_idx}.png"
        whitened_file = OUTPUT_DIR / f"figure11_qrs_whitened_{rank:02d}_R{r_idx}.png"

        save_figure(
            t_axis=t_axis,
            signal_segment=signal_segment,
            state1=x1,
            state2=x2,
            state3=x3,
            cost_segment=cost_segment,
            lcr_segment=lcr_segment,
            full_signal=signal,
            r_idx=r_idx,
            state_mode="raw",
            out_file=raw_file,
        )

        save_figure(
            t_axis=t_axis,
            signal_segment=signal_segment,
            state1=z1,
            state2=z2,
            state3=z3,
            cost_segment=cost_segment,
            lcr_segment=lcr_segment,
            full_signal=signal,
            r_idx=r_idx,
            state_mode="whitened",
            out_file=whitened_file,
        )

    print(f"\nSaved {len(selected)} raw and whitened figure pairs to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()