"""Plot narrow and broad multiscale QRS examples.

This script builds two real QRS templates from MIT-BIH training records:
one narrow-scale template and one broad-scale template. It then scans a set of
held-out candidate records, finds suitable QRS examples, and selects:

1. One narrow-dominant example
2. One broad-dominant example

For each example, the script plots:
- the ECG segment
- the narrow-scale LCR
- the broad-scale LCR
- the final multi-scale LCR

The final figure is saved to the generated outputs folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lmlib as lm
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import resample

FS = 500
EPS = 1e-6

TRAIN_RECORDS = ["100", "103", "105", "119"]

CANDIDATE_EXAMPLE_RECORDS = [
    "106",
    "108",
    "114",
    "123",
    "200",
    "201",
    "202",
    "203",
    "208",
    "210",
    "214",
    "219",
    "221",
    "223",
]

POLYDEGREE = 2

# Narrow model, about 100 ms total support.
NARROW_LSIDE = 25
NARROW_G = 40

# Broad model, about 180 ms total support.
BROAD_LSIDE = 45
BROAD_G = 40

DISPLAY_MARGIN_SEC = 0.7
MATCH_TOL_SEC = 0.05

BEAT_SYMBOLS = {
    "N",
    "L",
    "R",
    "A",
    "a",
    "J",
    "S",
    "V",
    "E",
    "F",
    "e",
    "j",
    "/",
    "f",
    "Q",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "alssm_multiscale_qrs_two_examples.png"


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


def load_record_500hz(record_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one MIT-BIH record and resample it to 500 Hz if needed.

    Args:
        record_name: MIT-BIH record name.

    Returns:
        Tuple containing:
            - ECG signal at 500 Hz
            - annotated beat indices at 500 Hz
    """
    record_path = MIT_PATH / record_name
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, 0].astype(np.float64)
    fs_orig = int(record.fs)

    ann = wfdb.rdann(str(record_path), "atr")
    samples = ann.sample.copy()
    symbols = np.array(ann.symbol)

    if fs_orig != FS:
        new_len = int(len(signal) * FS / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)
        samples = (samples * FS / fs_orig).astype(int)

    r_annotations = np.array(
        [sample for sample, symbol in zip(samples, symbols) if symbol in BEAT_SYMBOLS],
        dtype=int,
    )

    return signal, r_annotations


def run_alssm(
    signal: np.ndarray,
    l_side: int,
    g: float,
    polydegree: int = POLYDEGREE,
) -> np.ndarray:
    """Estimate ALSSM states for one ECG signal.

    Args:
        signal: ECG signal.
        l_side: Half-width of the ALSSM support window.
        g: Segment weighting parameter.
        polydegree: Polynomial degree.

    Returns:
        State sequence.
    """
    model = lm.AlssmPoly(poly_degree=polydegree)
    segment = lm.Segment(
        a=-l_side,
        b=l_side,
        direction=lm.FORWARD,
        g=g,
    )
    cost_model = lm.CompositeCost((model,), (segment,), [[1]])
    rls = lm.create_rls(cost_model, multi_channel_set=False, steady_state=True)
    rls.filter(signal)
    return rls.minimize_x()


def build_template_from_records(
    records: Sequence[str],
    l_side: int,
    g: float,
    polydegree: int = POLYDEGREE,
) -> np.ndarray:
    """Build one robust QRS template from several MIT-BIH records.

    Args:
        records: Training record names.
        l_side: Half-width of the ALSSM support window.
        g: Segment weighting parameter.
        polydegree: Polynomial degree.

    Returns:
        Median QRS template in raw ALSSM state space.

    Raises:
        RuntimeError: If no valid states were collected.
    """
    all_states: List[np.ndarray] = []

    for record_name in records:
        print(f"Building template from record {record_name} ...")
        signal, r_annotations = load_record_500hz(record_name)
        states = run_alssm(signal, l_side, g, polydegree=polydegree)

        valid_r = [r_idx for r_idx in r_annotations if 0 <= r_idx < len(states)]
        if len(valid_r) == 0:
            continue

        all_states.append(states[valid_r])

    if len(all_states) == 0:
        raise RuntimeError("No states collected for template building.")

    state_matrix = np.vstack(all_states)
    return np.median(state_matrix, axis=0)


def compute_lcr(
    states: np.ndarray,
    template: np.ndarray,
    eps: float = EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pointwise cost and LCR for one template.

    Args:
        states: ALSSM state sequence.
        template: Template vector.
        eps: Small numerical stabilizer.

    Returns:
        Tuple containing:
            - squared Euclidean distance
            - LCR score
    """
    distance = np.linalg.norm(states - template, axis=1) ** 2
    reference_energy = np.linalg.norm(template) ** 2
    lcr = -0.5 * np.log((distance + eps) / (reference_energy + eps))
    return distance, lcr


def collect_candidates(
    example_records: Sequence[str],
    template_narrow: np.ndarray,
    template_broad: np.ndarray,
) -> List[Dict[str, object]]:
    """Collect candidate QRS examples from held-out records.

    Args:
        example_records: Held-out candidate records.
        template_narrow: Narrow QRS template.
        template_broad: Broad QRS template.

    Returns:
        Ranked candidate dictionaries.

    Raises:
        RuntimeError: If no suitable candidates were found.
    """
    margin = int(DISPLAY_MARGIN_SEC * FS)
    tolerance = int(MATCH_TOL_SEC * FS)
    candidates: List[Dict[str, object]] = []

    for record_name in example_records:
        print(f"Scanning record {record_name} for candidate beats ...")
        signal, r_annotations = load_record_500hz(record_name)

        states_narrow = run_alssm(
            signal,
            NARROW_LSIDE,
            NARROW_G,
            polydegree=POLYDEGREE,
        )
        states_broad = run_alssm(
            signal,
            BROAD_LSIDE,
            BROAD_G,
            polydegree=POLYDEGREE,
        )

        _, lcr_narrow = compute_lcr(states_narrow, template_narrow, eps=EPS)
        _, lcr_broad = compute_lcr(states_broad, template_broad, eps=EPS)
        lcr_final = np.maximum(lcr_narrow, lcr_broad)

        for r_ref in r_annotations:
            left = r_ref - margin
            right = r_ref + margin
            if left < 0 or right >= len(signal):
                continue

            local_left = max(0, r_ref - tolerance)
            local_right = min(len(signal), r_ref + tolerance + 1)

            local_final = lcr_final[local_left:local_right]
            peak_idx = local_left + int(np.argmax(local_final))

            if abs(peak_idx - r_ref) > tolerance:
                continue

            narrow_peak = float(lcr_narrow[peak_idx])
            broad_peak = float(lcr_broad[peak_idx])
            final_peak = float(lcr_final[peak_idx])

            local_signal = signal[left:right]
            ecg_prominence = float(signal[peak_idx] - np.median(local_signal))

            if final_peak < 2.0:
                continue
            if ecg_prominence < 0.25:
                continue

            candidates.append(
                {
                    "record": record_name,
                    "signal": signal,
                    "r_ref": int(r_ref),
                    "r_det": int(peak_idx),
                    "lcr_narrow": lcr_narrow,
                    "lcr_broad": lcr_broad,
                    "lcr_final": lcr_final,
                    "narrow_peak": narrow_peak,
                    "broad_peak": broad_peak,
                    "final_peak": final_peak,
                    "delta": narrow_peak - broad_peak,
                    "ecg_prom": ecg_prominence,
                }
            )

    if len(candidates) == 0:
        raise RuntimeError("No suitable candidates found.")

    return candidates


def choose_examples(
    candidates: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Choose one narrow-dominant and one broad-dominant example.

    Args:
        candidates: Candidate dictionaries.

    Returns:
        Tuple containing:
            - narrow-dominant example
            - broad-dominant example
    """
    narrow_sorted = sorted(
        candidates,
        key=lambda c: float(c["delta"]) + 0.15 * float(c["final_peak"]),
        reverse=True,
    )
    narrow_example = narrow_sorted[0]

    broad_sorted = sorted(
        candidates,
        key=lambda c: (-float(c["delta"])) + 0.15 * float(c["final_peak"]),
        reverse=True,
    )

    broad_example = None
    for candidate in broad_sorted:
        if not (
            candidate["record"] == narrow_example["record"]
            and candidate["r_ref"] == narrow_example["r_ref"]
        ):
            broad_example = candidate
            break

    if broad_example is None:
        broad_example = broad_sorted[0]

    return narrow_example, broad_example


def plot_examples(
    narrow_example: Dict[str, object],
    broad_example: Dict[str, object],
) -> None:
    """Plot the final narrow- and broad-dominant QRS examples.

    Args:
        narrow_example: Selected narrow-dominant example.
        broad_example: Selected broad-dominant example.
    """
    margin = int(DISPLAY_MARGIN_SEC * FS)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13, 9),
        sharex="col",
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 1.0]},
    )

    examples = [narrow_example, broad_example]
    titles = ["Narrow-dominant example", "Broad-dominant example"]

    for col, (example, title) in enumerate(zip(examples, titles)):
        signal = np.asarray(example["signal"])
        r_ref = int(example["r_ref"])
        r_det = int(example["r_det"])

        left = r_det - margin
        right = r_det + margin

        t_axis = np.arange(left, right) / FS
        signal_segment = signal[left:right]
        narrow_segment = np.asarray(example["lcr_narrow"])[left:right]
        broad_segment = np.asarray(example["lcr_broad"])[left:right]
        final_segment = np.asarray(example["lcr_final"])[left:right]

        local_det = r_det - left

        axes[0, col].plot(t_axis, signal_segment, color="black", linewidth=1.2, label="ECG")
        axes[0, col].scatter(
            r_det / FS,
            signal[r_det],
            color="red",
            s=40,
            label="Detected R",
            zorder=5,
        )
        axes[0, col].axvline(
            r_det / FS,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        axes[0, col].axvline(
            r_ref / FS,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            label="Reference R",
        )
        axes[0, col].legend(loc="upper left", fontsize=8, frameon=True)

        axes[1, col].plot(
            t_axis,
            narrow_segment,
            color="#1f77b4",
            linewidth=1.4,
            label="Narrow scale LCR",
        )
        axes[1, col].scatter(
            t_axis[local_det],
            narrow_segment[local_det],
            color="#1f77b4",
            s=30,
            zorder=5,
        )
        axes[1, col].axvline(
            r_det / FS,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        axes[1, col].legend(loc="upper right", fontsize=8, frameon=True)

        axes[2, col].plot(
            t_axis,
            broad_segment,
            color="#ff7f0e",
            linewidth=1.4,
            label="Broad scale LCR",
        )
        axes[2, col].scatter(
            t_axis[local_det],
            broad_segment[local_det],
            color="#ff7f0e",
            s=30,
            zorder=5,
        )
        axes[2, col].axvline(
            r_det / FS,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        axes[2, col].legend(loc="upper right", fontsize=8, frameon=True)

        axes[3, col].plot(
            t_axis,
            final_segment,
            color="purple",
            linewidth=1.5,
            label="Final multi-scale LCR",
        )
        axes[3, col].scatter(
            t_axis[local_det],
            final_segment[local_det],
            color="purple",
            s=30,
            zorder=5,
        )
        axes[3, col].axvline(
            r_det / FS,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        axes[3, col].legend(loc="upper right", fontsize=8, frameon=True)

        axes[0, col].set_title(f"{title}\nRecord {example['record']}", fontsize=11)

    axes[0, 0].set_ylabel("Amplitude (mV)", fontsize=9)
    axes[1, 0].set_ylabel("LCR", fontsize=9)
    axes[2, 0].set_ylabel("LCR", fontsize=9)
    axes[3, 0].set_ylabel("LCR", fontsize=9)

    axes[1, 0].text(
        0.5,
        1.05,
        r"Narrow QRS model, $\approx 100$ ms",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    axes[1, 1].text(
        0.5,
        1.05,
        r"Narrow QRS model, $\approx 100$ ms",
        transform=axes[1, 1].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    axes[2, 0].text(
        0.5,
        1.05,
        r"Broad QRS model, $\approx 180$ ms",
        transform=axes[2, 0].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    axes[2, 1].text(
        0.5,
        1.05,
        r"Broad QRS model, $\approx 180$ ms",
        transform=axes[2, 1].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    final_formula = (
        r"Final score "
        r"$\mathrm{LCR}_{\mathrm{final}}(k)=\max\{\mathrm{LCR}_{\mathrm{narrow}}(k),"
        r"\mathrm{LCR}_{\mathrm{broad}}(k)\}$"
    )
    axes[3, 0].text(
        0.5,
        1.05,
        final_formula,
        transform=axes[3, 0].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    axes[3, 1].text(
        0.5,
        1.05,
        final_formula,
        transform=axes[3, 1].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    axes[3, 0].set_xlabel("Time (s)", fontsize=9)
    axes[3, 1].set_xlabel("Time (s)", fontsize=9)

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSaved thesis figure to:\n{OUT_FILE}")


def main() -> None:
    """Build templates, select examples, and plot the multiscale QRS figure."""
    print("Building real narrow and broad QRS templates ...")
    template_narrow = build_template_from_records(
        TRAIN_RECORDS,
        NARROW_LSIDE,
        NARROW_G,
        polydegree=POLYDEGREE,
    )
    template_broad = build_template_from_records(
        TRAIN_RECORDS,
        BROAD_LSIDE,
        BROAD_G,
        polydegree=POLYDEGREE,
    )

    candidates = collect_candidates(
        CANDIDATE_EXAMPLE_RECORDS,
        template_narrow,
        template_broad,
    )

    narrow_example, broad_example = choose_examples(candidates)

    print("\nChosen examples:")
    print(
        f"NARROW winner: record={narrow_example['record']}, "
        f"Rref={narrow_example['r_ref']}, Rdet={narrow_example['r_det']}, "
        f"narrow={float(narrow_example['narrow_peak']):.3f}, "
        f"broad={float(narrow_example['broad_peak']):.3f}, "
        f"delta={float(narrow_example['delta']):.3f}"
    )
    print(
        f"BROAD winner:  record={broad_example['record']}, "
        f"Rref={broad_example['r_ref']}, Rdet={broad_example['r_det']}, "
        f"narrow={float(broad_example['narrow_peak']):.3f}, "
        f"broad={float(broad_example['broad_peak']):.3f}, "
        f"delta={float(broad_example['delta']):.3f}"
    )

    plot_examples(narrow_example, broad_example)


if __name__ == "__main__":
    main()