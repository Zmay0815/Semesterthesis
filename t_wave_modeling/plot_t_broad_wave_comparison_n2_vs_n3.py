"""Plot a broad T-wave comparison between n=2 and n=3 models.

This script builds two QT-based T-wave trajectory templates from a small
training subset:

1. Baseline model:
   - polynomial degree 2
   - narrower trajectory support

2. Exploratory broadened model:
   - polynomial degree 3
   - broader trajectory support

A held-out QT record with a broad annotated T wave is selected automatically.
The script then compares both detectors on the same beat and plots:

- ECG segment with annotated T span
- pointwise cost curve
- LCR curve

This figure is used as an exploratory broad T-wave example and is not the main
final detector result.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lmlib as lm
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.signal import resample

FS = 500
EPS = 1e-6

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUTPUT_DIR / "t_compare_n2_vs_n3_broad.png"

CAND_EXTS = ["pu", "q1c", "qt1", "atr"]

PREFERRED_TRAIN_RECORDS = ["sel100", "sel102", "sel103", "sel104"]
N_TRAIN_RECORDS = 4

BASE_POLYDEGREE = 2
BASE_L_SIDE = 40
BASE_G = 10
BASE_HALF_WIDTH = 20  # 41 samples

EXP_POLYDEGREE = 3
EXP_L_SIDE = 60
EXP_G = 10
EXP_HALF_WIDTH = 30  # 61 samples

T_WINDOW_SEC = (0.10, 0.50)

DISPLAY_LEFT_SEC = 0.35
DISPLAY_RIGHT_SEC = 0.55

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


def list_qt_records(base_path: Path) -> List[str]:
    """List QT record names from a QT Database directory.

    Args:
        base_path: QT Database folder.

    Returns:
        Sorted list of record names without extension.
    """
    records = [
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(str(base_path / "*.hea"))
    ]
    return sorted(records)


def try_rdann(record_path: Path) -> Tuple[Optional[str], Optional[wfdb.Annotation]]:
    """Try multiple annotation extensions for a QT record.

    Args:
        record_path: Path without extension to the QT record.

    Returns:
        Tuple containing:
            - annotation extension if found
            - loaded annotation object if found
    """
    for ext in CAND_EXTS:
        try:
            ann = wfdb.rdann(str(record_path), ext)
            return ext, ann
        except Exception:
            continue
    return None, None


def choose_best_lead(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    """Choose the best ECG lead using a robust derivative-based score.

    The score is:
        Q99(|dx|) / MAD(dx)

    Args:
        signal: ECG signal array.

    Returns:
        Tuple containing:
            - selected 1D signal
            - selected lead index
    """
    if signal.ndim == 1:
        return signal.astype(np.float64), 0

    best_score = -np.inf
    best_lead = 0

    for lead_idx in range(signal.shape[1]):
        x = signal[:, lead_idx]
        dx = np.diff(x)
        mad = np.median(np.abs(dx - np.median(dx))) + 1e-12
        score = np.quantile(np.abs(dx), 0.99) / mad

        if score > best_score:
            best_score = score
            best_lead = lead_idx

    return signal[:, best_lead].astype(np.float64), best_lead


def load_qt_record_500hz(
    record_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Load one QT record, choose the best lead, and resample to 500 Hz.

    Args:
        record_name: QT record name.

    Returns:
        Tuple containing:
            - signal at 500 Hz
            - annotation sample indices at 500 Hz
            - annotation symbols
            - annotation aux notes
            - used lead index
            - annotation extension
    """
    record_path = QT_PATH / record_name
    record = wfdb.rdrecord(str(record_path))
    signal, used_lead = choose_best_lead(record.p_signal)
    fs_orig = int(record.fs)

    ann_ext, ann = try_rdann(record_path)
    if ann is None or ann_ext is None:
        raise RuntimeError(f"No usable annotation found for {record_name}")

    samples = ann.sample.copy()
    symbols = np.array(ann.symbol)
    aux = np.array(ann.aux_note) if ann.aux_note is not None else np.array([""] * len(samples))

    if fs_orig != FS:
        new_len = int(len(signal) * FS / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)
        samples = (samples * FS / fs_orig).astype(int)

    return signal, samples, symbols, aux, used_lead, ann_ext


def parse_qt_annotations(
    samples: np.ndarray,
    symbols: np.ndarray,
    aux: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse R peaks, explicit T peaks, and T onset/offset annotations.

    T onset and T offset are parsed from the auxiliary notes if present.
    If no explicit T peak exists later, a midpoint fallback can be used.

    Args:
        samples: Annotation sample indices.
        symbols: Annotation symbols.
        aux: Annotation auxiliary notes.

    Returns:
        Tuple containing:
            - R peak indices
            - T peak indices
            - T onset indices
            - T offset indices
    """
    r_peaks: List[int] = []
    t_peaks: List[int] = []
    t_on: List[int] = []
    t_off: List[int] = []

    for sample, symbol, aux_note in zip(samples, symbols, aux):
        aux_low = (aux_note or "").strip().lower()

        if symbol in BEAT_SYMBOLS:
            r_peaks.append(int(sample))

        if symbol.lower() == "t":
            t_peaks.append(int(sample))

        if "t on" in aux_low:
            t_on.append(int(sample))
        if "t off" in aux_low:
            t_off.append(int(sample))

    return (
        np.array(sorted(r_peaks), dtype=int),
        np.array(sorted(t_peaks), dtype=int),
        np.array(sorted(t_on), dtype=int),
        np.array(sorted(t_off), dtype=int),
    )


def pair_t_beats(
    r_peaks: np.ndarray,
    t_peaks: np.ndarray,
    t_on: np.ndarray,
    t_off: np.ndarray,
) -> List[Dict[str, Optional[float]]]:
    """Build beat-wise T annotations for broad-T analysis.

    The function first tries to use an explicit T peak. If none exists in the
    physiological post-R range, it falls back to the midpoint between T onset
    and T offset if both are available.

    Args:
        r_peaks: R peak indices.
        t_peaks: Explicit T peak indices.
        t_on: T onset indices.
        t_off: T offset indices.

    Returns:
        List of beat dictionaries containing R, T peak, onset, offset, and
        estimated T width in milliseconds.
    """
    beats: List[Dict[str, Optional[float]]] = []

    for r_peak in r_peaks:
        t_peak = None

        valid_t = t_peaks[
            (t_peaks > r_peak + int(0.06 * FS))
            & (t_peaks < r_peak + int(0.60 * FS))
        ]
        if len(valid_t) > 0:
            t_peak = int(valid_t[0])
        else:
            valid_on = t_on[
                (t_on > r_peak + int(0.04 * FS))
                & (t_on < r_peak + int(0.50 * FS))
            ]
            valid_off = t_off[
                (t_off > r_peak + int(0.08 * FS))
                & (t_off < r_peak + int(0.80 * FS))
            ]

            if len(valid_on) > 0 and len(valid_off) > 0:
                onset = int(valid_on[0])
                offsets_after_onset = valid_off[valid_off > onset]
                if len(offsets_after_onset) > 0:
                    offset = int(offsets_after_onset[0])
                    t_peak = int((onset + offset) // 2)

        if t_peak is None:
            continue

        on_candidates = t_on[(t_on < t_peak) & (t_on > t_peak - int(0.40 * FS))]
        off_candidates = t_off[(t_off > t_peak) & (t_off < t_peak + int(0.40 * FS))]

        t_on_i = int(on_candidates[-1]) if len(on_candidates) > 0 else None
        t_off_i = int(off_candidates[0]) if len(off_candidates) > 0 else None

        width_ms = None
        if t_on_i is not None and t_off_i is not None and t_off_i > t_on_i:
            width_ms = 1000.0 * (t_off_i - t_on_i) / FS

        beats.append(
            {
                "r": int(r_peak),
                "t_peak": int(t_peak),
                "t_on": t_on_i,
                "t_off": t_off_i,
                "t_width_ms": width_ms,
            }
        )

    return beats


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
        states: Raw state sequence.

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
    """Build one stacked trajectory vector around a candidate center.

    Args:
        whitened_states: Whitened ALSSM state sequence.
        center: Candidate center index.
        half_width: Half-width of the trajectory window.

    Returns:
        Flattened trajectory vector, or ``None`` if the window is invalid.
    """
    left = center - half_width
    right = center + half_width + 1

    if left < 0 or right > len(whitened_states):
        return None

    return whitened_states[left:right].reshape(-1)


def build_t_template(
    train_records: Sequence[str],
    polydegree: int,
    l_side: int,
    g: float,
    half_width: int,
) -> np.ndarray:
    """Build one T trajectory template from QT training records.

    Args:
        train_records: QT training records.
        polydegree: Polynomial degree.
        l_side: Half-width of the ALSSM support window.
        g: Segment weighting parameter.
        half_width: Half-width of the trajectory window.

    Returns:
        Median T trajectory template.

    Raises:
        RuntimeError: If no trajectories were found.
    """
    trajectories: List[np.ndarray] = []

    for record_name in train_records:
        print(f"Building T template from {record_name} ...")
        signal, samples, symbols, aux, used_lead, ann_ext = load_qt_record_500hz(record_name)
        _ = used_lead, ann_ext

        r_peaks, t_peaks, t_on, t_off = parse_qt_annotations(samples, symbols, aux)
        beats = pair_t_beats(r_peaks, t_peaks, t_on, t_off)

        if len(beats) == 0:
            continue

        states = run_alssm(signal, polydegree, l_side, g)
        whitened_states = whiten_states(states)

        for beat in beats:
            traj = build_trajectory(whitened_states, int(beat["t_peak"]), half_width)
            if traj is not None:
                trajectories.append(traj)

    if len(trajectories) == 0:
        raise RuntimeError("No T trajectories found for template building.")

    trajectory_matrix = np.array(trajectories, dtype=np.float64)
    return np.median(trajectory_matrix, axis=0)


def detect_t_for_beat(
    signal: np.ndarray,
    r_index: int,
    template: np.ndarray,
    polydegree: int,
    l_side: int,
    g: float,
    half_width: int,
) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
    """Detect a T peak for one beat with one trajectory template.

    Args:
        signal: ECG signal.
        r_index: R anchor index.
        template: Trajectory template.
        polydegree: Polynomial degree.
        l_side: Half-width of the ALSSM support window.
        g: Segment weighting parameter.
        half_width: Half-width of the trajectory window.

    Returns:
        Tuple containing:
            - detected T index or ``None``
            - pointwise cost curve
            - pointwise LCR curve
    """
    states = run_alssm(signal, polydegree, l_side, g)
    whitened_states = whiten_states(states)

    start = r_index + int(T_WINDOW_SEC[0] * FS)
    end = r_index + int(T_WINDOW_SEC[1] * FS)

    cost = np.full(len(signal), np.nan, dtype=float)
    lcr = np.full(len(signal), np.nan, dtype=float)

    reference_energy = np.linalg.norm(template) ** 2
    best_t = None
    best_cost = np.inf

    for t_idx in range(start, end):
        traj = build_trajectory(whitened_states, t_idx, half_width)
        if traj is None:
            continue

        distance = np.linalg.norm(traj - template) ** 2
        cost[t_idx] = distance
        lcr[t_idx] = -0.5 * np.log((distance + EPS) / (reference_energy + EPS))

        if distance < best_cost:
            best_cost = distance
            best_t = t_idx

    return best_t, cost, lcr


def pick_training_and_example_records() -> Tuple[List[str], List[str]]:
    """Split QT records into training and held-out example records.

    Returns:
        Tuple containing:
            - training record list
            - held-out example candidate list
    """
    all_records = list_qt_records(QT_PATH)
    if len(all_records) < 6:
        raise RuntimeError("Too few QT records found.")

    train_records = [r for r in PREFERRED_TRAIN_RECORDS if r in all_records]

    if len(train_records) < N_TRAIN_RECORDS:
        for record_name in all_records:
            if record_name not in train_records:
                train_records.append(record_name)
            if len(train_records) == N_TRAIN_RECORDS:
                break

    example_candidates = [r for r in all_records if r not in train_records]

    if len(example_candidates) == 0:
        raise RuntimeError("No held-out QT records available.")

    return train_records, example_candidates


def choose_broad_example_record(
    example_candidates: Sequence[str],
) -> Dict[str, object]:
    """Choose the held-out QT record containing the broadest annotated T wave.

    Args:
        example_candidates: Held-out QT record names.

    Returns:
        Dictionary containing:
            - record name
            - ECG signal
            - selected broad T beat

    Raises:
        RuntimeError: If no suitable broad example was found.
    """
    best = None

    for record_name in example_candidates:
        try:
            signal, samples, symbols, aux, used_lead, ann_ext = load_qt_record_500hz(record_name)
            _ = used_lead, ann_ext

            r_peaks, t_peaks, t_on, t_off = parse_qt_annotations(samples, symbols, aux)
            beats = pair_t_beats(r_peaks, t_peaks, t_on, t_off)
        except Exception:
            continue

        valid_beats = [
            beat
            for beat in beats
            if beat["t_width_ms"] is not None and 120 <= beat["t_peak"] - beat["r"] <= 300
        ]
        if len(valid_beats) == 0:
            continue

        broadest = max(valid_beats, key=lambda beat: float(beat["t_width_ms"]))

        if best is None or float(broadest["t_width_ms"]) > float(best["beat"]["t_width_ms"]):
            best = {
                "record": record_name,
                "signal": signal,
                "beat": broadest,
            }

    if best is None:
        raise RuntimeError("No suitable broad T-wave example found in held-out QT records.")

    return best


def plot_comparison(
    record_name: str,
    signal: np.ndarray,
    beat: Dict[str, Optional[float]],
    t_base: Optional[int],
    cost_base: np.ndarray,
    lcr_base: np.ndarray,
    t_exp: Optional[int],
    cost_exp: np.ndarray,
    lcr_exp: np.ndarray,
) -> None:
    """Plot the broad-T comparison figure.

    Args:
        record_name: Selected QT record name.
        signal: ECG signal.
        beat: Selected broad T beat dictionary.
        t_base: Baseline detected T index.
        cost_base: Baseline cost curve.
        lcr_base: Baseline LCR curve.
        t_exp: Exploratory detected T index.
        cost_exp: Exploratory cost curve.
        lcr_exp: Exploratory LCR curve.
    """
    r_idx = int(beat["r"])
    t_gt = int(beat["t_peak"])
    t_on = beat["t_on"]
    t_off = beat["t_off"]
    t_width_ms = float(beat["t_width_ms"]) if beat["t_width_ms"] is not None else np.nan

    left = max(0, r_idx - int(DISPLAY_LEFT_SEC * FS))
    right = min(len(signal), r_idx + int(DISPLAY_RIGHT_SEC * FS))
    t_axis = np.arange(left, right) / FS

    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")

    titles = [
        "Baseline: polydegree = 2",
        "Experimental: broadened T + polydegree = 3",
    ]
    plot_sets = [
        (t_base, cost_base, lcr_base),
        (t_exp, cost_exp, lcr_exp),
    ]

    for col, (det_t, cost_curve, lcr_curve) in enumerate(plot_sets):
        axes[0, col].plot(t_axis, signal[left:right], color="black", linewidth=1.2, label="ECG")
        axes[0, col].axvline(
            r_idx / FS,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="R",
        )
        axes[0, col].axvline(
            t_gt / FS,
            color="green",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="T ground truth",
        )

        if t_on is not None and t_off is not None:
            axes[0, col].axvspan(
                int(t_on) / FS,
                int(t_off) / FS,
                color="green",
                alpha=0.12,
                label="Annotated T span",
            )

        if det_t is not None:
            axes[0, col].scatter(
                det_t / FS,
                signal[det_t],
                color="magenta",
                s=40,
                zorder=5,
                label="Detected T",
            )

        axes[0, col].set_title(titles[col], fontsize=11)
        axes[0, col].set_ylabel("Amplitude (mV)", fontsize=9)
        axes[0, col].legend(loc="upper right", fontsize=8, frameon=True)

        axes[1, col].plot(
            t_axis,
            cost_curve[left:right],
            color="darkgreen",
            linewidth=1.4,
            label=r"Cost $d(k)$",
        )
        axes[1, col].axvline(
            t_gt / FS,
            color="green",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        if det_t is not None:
            axes[1, col].axvline(
                det_t / FS,
                color="magenta",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
            )
        axes[1, col].set_ylabel(r"$d(k)$", fontsize=9)
        axes[1, col].legend(loc="upper right", fontsize=8, frameon=True)

        axes[2, col].plot(
            t_axis,
            lcr_curve[left:right],
            color="purple",
            linewidth=1.4,
            label="LCR",
        )
        axes[2, col].axvline(
            t_gt / FS,
            color="green",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        if det_t is not None:
            axes[2, col].axvline(
                det_t / FS,
                color="magenta",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
            )
        axes[2, col].set_ylabel("LCR", fontsize=9)
        axes[2, col].set_xlabel("Time (s)", fontsize=9)
        axes[2, col].legend(loc="upper right", fontsize=8, frameon=True)

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

    fig.suptitle(
        f"Held-out QT example: broad T-wave comparison on {record_name} "
        f"(annotated width: {t_width_ms:.1f} ms)",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {OUT_FILE}")


def main() -> None:
    """Build the baseline and exploratory templates and plot the broad-T example."""
    train_records, example_candidates = pick_training_and_example_records()
    print("Training records:", train_records)

    template_base = build_t_template(
        train_records,
        polydegree=BASE_POLYDEGREE,
        l_side=BASE_L_SIDE,
        g=BASE_G,
        half_width=BASE_HALF_WIDTH,
    )

    template_exp = build_t_template(
        train_records,
        polydegree=EXP_POLYDEGREE,
        l_side=EXP_L_SIDE,
        g=EXP_G,
        half_width=EXP_HALF_WIDTH,
    )

    example = choose_broad_example_record(example_candidates)

    record_name = str(example["record"])
    signal = np.asarray(example["signal"])
    beat = example["beat"]

    r_idx = int(beat["r"])
    t_width_ms = float(beat["t_width_ms"]) if beat["t_width_ms"] is not None else np.nan

    print(f"Chosen example record: {record_name}")
    print(f"Broad T width: {t_width_ms:.1f} ms")

    t_base, cost_base, lcr_base = detect_t_for_beat(
        signal,
        r_idx,
        template_base,
        polydegree=BASE_POLYDEGREE,
        l_side=BASE_L_SIDE,
        g=BASE_G,
        half_width=BASE_HALF_WIDTH,
    )

    t_exp, cost_exp, lcr_exp = detect_t_for_beat(
        signal,
        r_idx,
        template_exp,
        polydegree=EXP_POLYDEGREE,
        l_side=EXP_L_SIDE,
        g=EXP_G,
        half_width=EXP_HALF_WIDTH,
    )

    plot_comparison(
        record_name=record_name,
        signal=signal,
        beat=beat,
        t_base=t_base,
        cost_base=cost_base,
        lcr_base=lcr_base,
        t_exp=t_exp,
        cost_exp=cost_exp,
        lcr_exp=lcr_exp,
    )


if __name__ == "__main__":
    main()