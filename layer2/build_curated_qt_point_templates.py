"""Build curated QT-derived point templates for Layer 2.

This script builds wave-specific R, P, and T point templates from a curated
subset of QT Database records. For each wave type, it:

1. Loads the curated QT records
2. Selects the best lead automatically
3. Resamples each record to 500 Hz
4. Extracts explicit P, R, and T peak annotations
5. Estimates wave-specific ALSSM states
6. Computes a whitening matrix from a large state pool
7. Applies an outlier filter in whitened state space
8. Saves the final mean, median, and main template
9. Saves the whitening matrices and a JSON summary

Optional aligned ECG snippet preview plots are also generated and saved to the
generated outputs folder.
"""

from __future__ import annotations

import json
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
STATE_SAMPLE_STRIDE = 10

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

NPZ_DIR = PROJECT_ROOT / "npz_templates"
JSON_DIR = PROJECT_ROOT / "json_reports"
FIGURE_DIR = PROJECT_ROOT / "generated_outputs" / "figures" / "qt_curated_templates"

NPZ_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CAND_EXTS = ["pu", "q1c", "qt1", "atr"]

# Curated record lists after visual inspection.
CURATED_R_RECORDS = ["sel117", "sel123", "sel116", "sel114", "sel100", "sel104"]
CURATED_P_RECORDS = ["sel117", "sel123", "sel116", "sel114", "sel100", "sel104"]
CURATED_T_RECORDS = ["sel117", "sel123", "sel116", "sel114", "sel100", "sel104"]

# Keep only the closest fraction to the provisional mean in whitened state space.
KEEP_FRACTION = 0.80

# Main output template statistic.
MAIN_TEMPLATE_STAT = "mean"

WAVE_CONFIGS = {
    "R": {
        "polydegree": 2,
        "l_side": 15,
        "g": 40,
        "preview_left_sec": 0.10,
        "preview_right_sec": 0.14,
    },
    "P": {
        "polydegree": 2,
        "l_side": 20,
        "g": 120,
        "preview_left_sec": 0.12,
        "preview_right_sec": 0.12,
    },
    "T": {
        "polydegree": 2,
        "l_side": 40,
        "g": 10,
        "preview_left_sec": 0.18,
        "preview_right_sec": 0.22,
    },
}

BEAT_SYMBOLS = {
    "N", "L", "R", "A", "a", "J", "S", "V", "E", "F", "e", "j", "/", "f", "Q"
}


def resolve_qt_database_dir() -> Path:
    """Find the QT Database folder.

    Returns:
        Path to the QT Database directory.

    Raises:
        FileNotFoundError: If the database folder could not be found.
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
        signal: ECG signal array of shape (n_samples, n_channels) or (n_samples,).

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Load one QT record, choose the best lead, and resample to 500 Hz.

    Args:
        record_name: QT record name, for example ``"sel100"``.

    Returns:
        Tuple containing:
            - signal at 500 Hz
            - annotation sample indices at 500 Hz
            - annotation symbols
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

    if fs_orig != FS:
        new_len = int(len(signal) * FS / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)
        samples = (samples * FS / fs_orig).astype(int)

    return signal, samples, symbols, used_lead, ann_ext


def parse_explicit_prt(
    samples: np.ndarray,
    symbols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract explicit P, R, and T peak annotations.

    Args:
        samples: Annotation sample indices.
        symbols: Annotation symbols.

    Returns:
        Tuple containing arrays of R, P, and T peak sample indices.
    """
    r_peaks: List[int] = []
    p_peaks: List[int] = []
    t_peaks: List[int] = []

    for sample, symbol in zip(samples, symbols):
        if symbol in BEAT_SYMBOLS:
            r_peaks.append(int(sample))
        if symbol.lower() == "p":
            p_peaks.append(int(sample))
        if symbol.lower() == "t":
            t_peaks.append(int(sample))

    return (
        np.array(sorted(r_peaks), dtype=int),
        np.array(sorted(p_peaks), dtype=int),
        np.array(sorted(t_peaks), dtype=int),
    )


def pair_gt_beats(
    r_peaks: np.ndarray,
    p_peaks: np.ndarray,
    t_peaks: np.ndarray,
) -> List[Dict[str, int]]:
    """Pair explicit P, R, and T peaks into complete beats.

    Args:
        r_peaks: Explicit R peaks.
        p_peaks: Explicit P peaks.
        t_peaks: Explicit T peaks.

    Returns:
        List of beat dictionaries with explicit P, R, and T ground-truth peaks.
    """
    beats: List[Dict[str, int]] = []

    for r_peak in r_peaks:
        p_before = p_peaks[
            (p_peaks > r_peak - int(0.30 * FS)) &
            (p_peaks < r_peak - int(0.05 * FS))
        ]
        t_after = t_peaks[
            (t_peaks > r_peak + int(0.08 * FS)) &
            (t_peaks < r_peak + int(0.55 * FS))
        ]

        if len(p_before) == 0 or len(t_after) == 0:
            continue

        beats.append(
            {
                "r_gt": int(r_peak),
                "p_gt": int(p_before[-1]),
                "t_gt": int(t_after[0]),
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
    xhat = rls.minimize_x()

    return xhat


def compute_whitening_matrix(states: np.ndarray) -> np.ndarray:
    """Compute the whitening matrix for a state pool.

    Args:
        states: State matrix.

    Returns:
        Whitening matrix ``V``.
    """
    covariance = np.cov(states.T)
    covariance_reg = covariance + EPS * np.eye(covariance.shape[0])
    chol = np.linalg.cholesky(covariance_reg)
    whitening_matrix = np.linalg.solve(chol, np.eye(chol.shape[0]))
    return whitening_matrix


def aligned_signal_snippets(
    signal: np.ndarray,
    indices: Sequence[int],
    left_samples: int,
    right_samples: int,
) -> Optional[np.ndarray]:
    """Extract aligned ECG snippets around peak indices.

    Args:
        signal: ECG signal.
        indices: Peak indices.
        left_samples: Number of samples to include left of the center.
        right_samples: Number of samples to include right of the center.

    Returns:
        Array of aligned signal snippets or ``None`` if none were valid.
    """
    snippets = []

    for idx in indices:
        start = idx - left_samples
        end = idx + right_samples + 1

        if start < 0 or end > len(signal):
            continue

        snippets.append(signal[start:end])

    if len(snippets) == 0:
        return None

    return np.array(snippets)


def save_overlay_plot(snippets: np.ndarray, wave_name: str, out_file: Path) -> None:
    """Save an overlay plot of aligned ECG snippets.

    Args:
        snippets: Aligned ECG snippets.
        wave_name: Wave name, one of ``"R"``, ``"P"``, or ``"T"``.
        out_file: Output image path.
    """
    if snippets is None or len(snippets) == 0:
        return

    center = snippets.shape[1] // 2
    x_axis = np.arange(snippets.shape[1]) - center

    plt.figure(figsize=(8.0, 4.5))

    for i in range(min(len(snippets), 120)):
        plt.plot(x_axis, snippets[i], linewidth=0.5, alpha=0.15, color="gray")

    plt.plot(x_axis, np.mean(snippets, axis=0), linewidth=2.0, label="Mean")
    plt.plot(x_axis, np.median(snippets, axis=0), linewidth=2.0, label="Median")
    plt.axvline(0, linestyle="--", linewidth=1.0, color="black")

    plt.title(f"Curated QT aligned ECG snippets for {wave_name}")
    plt.xlabel("Samples relative to annotated peak")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_wave_outputs(
    wave_name: str,
    final_mean: np.ndarray,
    final_median: np.ndarray,
    final_main: np.ndarray,
    whitening_matrix: np.ndarray,
) -> None:
    """Save wave-specific templates and whitening matrix.

    Args:
        wave_name: Wave name.
        final_mean: Final mean template.
        final_median: Final median template.
        final_main: Main selected template.
        whitening_matrix: Whitening matrix for the wave.
    """
    wave_lower = wave_name.lower()

    np.save(NPZ_DIR / f"{wave_lower}_template_qt_curated_mean_500hz.npy", final_mean)
    np.save(
        NPZ_DIR / f"{wave_lower}_template_qt_curated_median_500hz.npy",
        final_median,
    )
    np.save(
        NPZ_DIR / f"{wave_lower}_template_qt_curated_main_500hz.npy",
        final_main,
    )
    np.save(NPZ_DIR / f"V_{wave_lower}_qt_curated_500hz.npy", whitening_matrix)


def main() -> None:
    """Build the curated QT point templates and save all outputs."""
    all_records_needed = sorted(
        set(CURATED_R_RECORDS + CURATED_P_RECORDS + CURATED_T_RECORDS)
    )

    raw_event_states = {"R": [], "P": [], "T": []}
    whitening_pool = {"R": [], "P": [], "T": []}
    aligned_signals = {"R": [], "P": [], "T": []}
    per_record_counts = {"R": {}, "P": {}, "T": {}}
    record_leads: Dict[str, Dict[str, object]] = {}

    for record_name in all_records_needed:
        print(f"Processing {record_name} ...")

        signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
        record_leads[record_name] = {
            "lead": int(used_lead),
            "ann_ext": ann_ext,
        }

        r_peaks, p_peaks, t_peaks = parse_explicit_prt(samples, symbols)
        beats = pair_gt_beats(r_peaks, p_peaks, t_peaks)

        if len(beats) == 0:
            print(f"  No usable explicit P/R/T beats found in {record_name}")
            continue

        for wave_name, cfg in WAVE_CONFIGS.items():
            use_this_record = (
                (wave_name == "R" and record_name in CURATED_R_RECORDS)
                or (wave_name == "P" and record_name in CURATED_P_RECORDS)
                or (wave_name == "T" and record_name in CURATED_T_RECORDS)
            )

            if not use_this_record:
                continue

            xhat = run_alssm(signal, cfg["polydegree"], cfg["l_side"], cfg["g"])
            whitening_pool[wave_name].append(xhat[::STATE_SAMPLE_STRIDE])

            count = 0
            wave_indices: List[int] = []

            for beat in beats:
                idx = beat[f"{wave_name.lower()}_gt"]

                if idx < cfg["l_side"] + 2 or idx >= len(signal) - cfg["l_side"] - 2:
                    continue

                raw_event_states[wave_name].append(xhat[idx].copy())
                wave_indices.append(idx)
                count += 1

            per_record_counts[wave_name][record_name] = count

            left_samples = int(cfg["preview_left_sec"] * FS)
            right_samples = int(cfg["preview_right_sec"] * FS)
            snippets = aligned_signal_snippets(
                signal,
                wave_indices,
                left_samples,
                right_samples,
            )
            if snippets is not None:
                aligned_signals[wave_name].append(snippets)

    summary: Dict[str, object] = {
        "fs": FS,
        "main_template_stat": MAIN_TEMPLATE_STAT,
        "keep_fraction": KEEP_FRACTION,
        "state_sample_stride": STATE_SAMPLE_STRIDE,
        "records": {
            "R": CURATED_R_RECORDS,
            "P": CURATED_P_RECORDS,
            "T": CURATED_T_RECORDS,
        },
        "record_leads": record_leads,
        "waves": {},
    }

    for wave_name, cfg in WAVE_CONFIGS.items():
        if len(raw_event_states[wave_name]) == 0:
            print(f"\nNo states collected for {wave_name}, skipping.")
            continue

        whitening_states = np.vstack(whitening_pool[wave_name])
        whitening_matrix = compute_whitening_matrix(whitening_states)

        raw_states = np.vstack(raw_event_states[wave_name])
        whitened_event_states = (whitening_matrix @ raw_states.T).T

        provisional_mean = np.mean(whitened_event_states, axis=0)

        distances = np.linalg.norm(whitened_event_states - provisional_mean, axis=1)
        threshold = np.quantile(distances, KEEP_FRACTION)
        keep_mask = distances <= threshold
        kept_event_states = whitened_event_states[keep_mask]

        final_mean = np.mean(kept_event_states, axis=0)
        final_median = np.median(kept_event_states, axis=0)

        if MAIN_TEMPLATE_STAT == "mean":
            final_main = final_mean
        elif MAIN_TEMPLATE_STAT == "median":
            final_main = final_median
        else:
            raise ValueError("MAIN_TEMPLATE_STAT must be 'mean' or 'median'")

        save_wave_outputs(
            wave_name=wave_name,
            final_mean=final_mean,
            final_median=final_median,
            final_main=final_main,
            whitening_matrix=whitening_matrix,
        )

        if len(aligned_signals[wave_name]) > 0:
            snippets = np.vstack(aligned_signals[wave_name])
            save_overlay_plot(
                snippets=snippets,
                wave_name=wave_name,
                out_file=FIGURE_DIR / f"{wave_name.lower()}_curated_aligned_snippets.png",
            )

        summary["waves"][wave_name] = {
            "polydegree": cfg["polydegree"],
            "l_side": cfg["l_side"],
            "g": cfg["g"],
            "n_whitening_samples": int(whitening_states.shape[0]),
            "n_event_states_before_filter": int(whitened_event_states.shape[0]),
            "n_event_states_after_filter": int(kept_event_states.shape[0]),
            "per_record_counts": per_record_counts[wave_name],
            "template_dimension": int(final_main.shape[0]),
            "main_template_stat": MAIN_TEMPLATE_STAT,
        }

    summary_file = JSON_DIR / "qt_curated_templates_summary.json"
    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nDone.")
    print(f"Saved curated templates to: {NPZ_DIR}")
    print(f"Saved summary to: {summary_file}")
    print(f"Saved preview plots to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()