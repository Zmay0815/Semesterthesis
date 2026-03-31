"""Plot a qualitative end-to-end QT delineation example.

This script searches for a clean QT record with explicit P, R, and T ground
truth peaks and then runs the full qualitative delineation pipeline:

1. R detection with the non-whitened MIT-derived R template
2. P detection with a QT/MIT point template in whitened state space
3. T detection with a trajectory template in whitened state space

The selected example is plotted as a four-panel figure showing:
- ECG with ground truth and detections
- whitened ALSSM state trajectory
- normalized cost curves
- log-cost-ratio (LCR) curves

This figure is used as the qualitative end-to-end QT example in the thesis.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

OUT_FILE = OUTPUT_DIR / "qt_full_prt_alssm_figure.png"

R_TEMPLATE_FILE = NPZ_DIR / "r_template_mitbih_500hz.npy"
P_TEMPLATE_FILE = NPZ_DIR / "p_template_mitbih_500hz.npy"

# The thesis workflow sometimes used one of two T-template filenames.
T_TEMPLATE_CANDIDATES = [
    NPZ_DIR / "t_template_dynamic_iterative.npy",
    NPZ_DIR / "t_template_mitbih_trajectory.npy",
]

R_POLYDEGREE = 2
R_L_SIDE = 15
R_G = 40
R_THRESHOLD_PERCENTILE = 97.5
R_REFRACTORY_SEC = 0.35

P_POLYDEGREE = 2
P_L_SIDE = 20
P_G = 120
P_WINDOW_SEC = (-0.25, -0.08)

T_POLYDEGREE = 2
T_L_SIDE = 40
T_G = 10
T_WINDOW_SEC = (0.10, 0.50)

PREFERRED_RECORDS = [
    "sel100",
    "sel102",
    "sel103",
    "sel104",
    "sel114",
    "sel116",
    "sel117",
    "sel123",
]

CAND_EXTS = ["pu", "q1c", "qt1", "atr"]

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


def resolve_existing_file(candidates: List[Path], label: str) -> Path:
    """Resolve the first existing file from a candidate list.

    Args:
        candidates: Candidate file paths.
        label: Human-readable file label for error messages.

    Returns:
        Existing file path.

    Raises:
        FileNotFoundError: If none of the candidates exist.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find required {label} file.\n"
        f"Searched in:\n{searched}"
    )


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Load one QT record, choose the best lead, and resample to 500 Hz.

    Args:
        record_name: QT record name.

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
        Tuple containing arrays of explicit R, P, and T peak indices.
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
        p_valid = p_peaks[
            (p_peaks > r_peak + int(P_WINDOW_SEC[0] * FS))
            & (p_peaks < r_peak + int(P_WINDOW_SEC[1] * FS))
        ]
        t_valid = t_peaks[
            (t_peaks > r_peak + int(T_WINDOW_SEC[0] * FS))
            & (t_peaks < r_peak + int(T_WINDOW_SEC[1] * FS))
        ]

        if len(p_valid) == 0 or len(t_valid) == 0:
            continue

        beats.append(
            {
                "r_gt": int(r_peak),
                "p_gt": int(p_valid[-1]),
                "t_gt": int(t_valid[0]),
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
        Flattened trajectory vector, or ``None`` if invalid.
    """
    left = center - half_width
    right = center + half_width + 1

    if left < 0 or right > len(whitened_states):
        return None

    return whitened_states[left:right].reshape(-1)


def normalize_curve(values: np.ndarray) -> np.ndarray:
    """Normalize a curve to the range [0, 1] for plotting.

    Args:
        values: Input curve with possible NaN values.

    Returns:
        Normalized curve with NaN values preserved.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    output = np.full_like(values, np.nan, dtype=float)

    if valid.sum() == 0:
        return output

    valid_values = values[valid]
    vmin = np.min(valid_values)
    vmax = np.max(valid_values)

    if vmax - vmin < 1e-12:
        output[valid] = 0.0
    else:
        output[valid] = (valid_values - vmin) / (vmax - vmin)

    return output


def load_templates() -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load the R, P, and T templates used in the figure.

    Returns:
        Tuple containing:
            - R template
            - P template
            - T trajectory template
            - inferred T trajectory half-width

    Raises:
        ValueError: If the T template has the wrong shape.
    """
    r_template_file = resolve_existing_file([R_TEMPLATE_FILE], "R template")
    p_template_file = resolve_existing_file([P_TEMPLATE_FILE], "P template")
    t_template_file = resolve_existing_file(T_TEMPLATE_CANDIDATES, "T template")

    r_template = np.load(r_template_file)
    p_template = np.load(p_template_file)
    t_template = np.load(t_template_file)

    if t_template.ndim != 1:
        raise ValueError("T template must be a 1D trajectory template.")
    if len(t_template) % 3 != 0:
        raise ValueError("T trajectory template length is not divisible by 3.")

    t_len = len(t_template) // 3
    t_half_width = (t_len - 1) // 2

    return r_template, p_template, t_template, t_half_width


def detect_record(
    record_name: str,
    r_template: np.ndarray,
    p_template: np.ndarray,
    t_template: np.ndarray,
    t_half_width: int,
) -> Optional[Dict[str, object]]:
    """Run the qualitative delineation pipeline on one QT record.

    Args:
        record_name: QT record name.
        r_template: MIT-derived R template.
        p_template: Point-template P model.
        t_template: Trajectory-template T model.
        t_half_width: Half-width inferred from the T trajectory template.

    Returns:
        Result dictionary for one clean record, or ``None`` if unsuitable.
    """
    signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
    _ = used_lead, ann_ext

    r_gt_all, p_gt_all, t_gt_all = parse_explicit_prt(samples, symbols)
    gt_beats = pair_gt_beats(r_gt_all, p_gt_all, t_gt_all)

    if len(gt_beats) < 5:
        return None

    xhat_r = run_alssm(signal, R_POLYDEGREE, R_L_SIDE, R_G)
    d_r = np.linalg.norm(xhat_r - r_template, axis=1) ** 2
    e_r = np.linalg.norm(r_template) ** 2
    lcr_r = -0.5 * np.log((d_r + EPS) / (e_r + EPS))

    r_detected, _ = find_peaks(
        lcr_r,
        height=np.percentile(lcr_r, R_THRESHOLD_PERCENTILE),
        distance=int(R_REFRACTORY_SEC * FS),
    )

    xhat_p = run_alssm(signal, P_POLYDEGREE, P_L_SIDE, P_G)
    zs_p = whiten_states(xhat_p)
    d_p = np.linalg.norm(zs_p - p_template, axis=1) ** 2
    e_p = np.linalg.norm(p_template) ** 2
    lcr_p = -0.5 * np.log((d_p + EPS) / (e_p + EPS))

    xhat_t = run_alssm(signal, T_POLYDEGREE, T_L_SIDE, T_G)
    zs_t = whiten_states(xhat_t)

    d_t = np.full(len(signal), np.nan, dtype=float)
    lcr_t = np.full(len(signal), np.nan, dtype=float)
    e_t = np.linalg.norm(t_template) ** 2

    for idx in range(t_half_width, len(signal) - t_half_width):
        traj = build_trajectory(zs_t, idx, t_half_width)
        if traj is None:
            continue
        value = np.linalg.norm(traj - t_template) ** 2
        d_t[idx] = value
        lcr_t[idx] = -0.5 * np.log((value + EPS) / (e_t + EPS))

    matched_beats = []

    for beat in gt_beats:
        r_gt = beat["r_gt"]

        if len(r_detected) == 0:
            continue

        r_idx = int(np.argmin(np.abs(r_detected - r_gt)))
        r_det = int(r_detected[r_idx])

        if abs(r_det - r_gt) > int(0.05 * FS):
            continue

        p_start = r_det + int(P_WINDOW_SEC[0] * FS)
        p_end = r_det + int(P_WINDOW_SEC[1] * FS)
        if p_start < 0 or p_end >= len(signal) or p_end <= p_start:
            continue

        p_local = int(np.argmax(lcr_p[p_start:p_end]))
        p_det = int(p_start + p_local)

        t_start = r_det + int(T_WINDOW_SEC[0] * FS)
        t_end = r_det + int(T_WINDOW_SEC[1] * FS)
        if t_start < 0 or t_end >= len(signal) or t_end <= t_start:
            continue

        t_window = lcr_t[t_start:t_end]
        if np.all(~np.isfinite(t_window)):
            continue

        t_local = int(np.nanargmax(t_window))
        t_det = int(t_start + t_local)

        p_err = abs(p_det - beat["p_gt"])
        r_err = abs(r_det - beat["r_gt"])
        t_err = abs(t_det - beat["t_gt"])

        matched_beats.append(
            {
                "p_gt": beat["p_gt"],
                "r_gt": beat["r_gt"],
                "t_gt": beat["t_gt"],
                "p_det": p_det,
                "r_det": r_det,
                "t_det": t_det,
                "score": p_err + r_err + t_err,
            }
        )

    if len(matched_beats) < 5:
        return None

    matched_beats = sorted(matched_beats, key=lambda x: int(x["score"]))
    best_beat = matched_beats[0]

    return {
        "record": record_name,
        "signal": signal,
        "gt_beats": matched_beats,
        "best_beat": best_beat,
        "zs_p": zs_p,
        "d_r": d_r,
        "lcr_r": lcr_r,
        "d_p": d_p,
        "lcr_p": lcr_p,
        "d_t": d_t,
        "lcr_t": lcr_t,
    }


def choose_best_result(
    r_template: np.ndarray,
    p_template: np.ndarray,
    t_template: np.ndarray,
    t_half_width: int,
) -> Dict[str, object]:
    """Choose the cleanest available QT record for the figure.

    Args:
        r_template: MIT-derived R template.
        p_template: P point template.
        t_template: T trajectory template.
        t_half_width: Half-width inferred from the T trajectory template.

    Returns:
        Best result dictionary.

    Raises:
        RuntimeError: If no suitable record was found.
    """
    available = list_qt_records(QT_PATH)
    candidate_records = [r for r in PREFERRED_RECORDS if r in available]
    if len(candidate_records) == 0:
        candidate_records = available[:8]

    best_result = None

    for record_name in candidate_records:
        print(f"Testing record {record_name} ...")
        result = detect_record(
            record_name=record_name,
            r_template=r_template,
            p_template=p_template,
            t_template=t_template,
            t_half_width=t_half_width,
        )
        if result is None:
            continue

        if best_result is None or int(result["best_beat"]["score"]) < int(best_result["best_beat"]["score"]):
            best_result = result

    if best_result is None:
        raise RuntimeError("No good QT record found for the figure.")

    return best_result


def build_plot_segment(best_result: Dict[str, object]) -> Dict[str, object]:
    """Build a clean multi-beat segment around the best matched beat.

    Args:
        best_result: Selected record result dictionary.

    Returns:
        Plot-ready segment dictionary.
    """
    signal = np.asarray(best_result["signal"])
    best_beat = best_result["best_beat"]
    gt_beats = sorted(best_result["gt_beats"], key=lambda x: int(x["r_gt"]))

    best_r = int(best_beat["r_gt"])
    r_list = np.array([int(beat["r_gt"]) for beat in gt_beats], dtype=int)
    best_idx = int(np.argmin(np.abs(r_list - best_r)))

    left_idx = max(0, best_idx - 2)
    right_idx = min(len(gt_beats) - 1, best_idx + 2)

    seg_left = max(0, int(gt_beats[left_idx]["r_gt"]) - int(0.40 * FS))
    seg_right = min(len(signal), int(gt_beats[right_idx]["r_gt"]) + int(0.60 * FS))

    t_axis = np.arange(seg_left, seg_right) / FS
    sig_seg = signal[seg_left:seg_right]

    p_gt_seg, r_gt_seg, t_gt_seg = [], [], []
    p_det_seg, r_det_seg, t_det_seg = [], [], []

    for beat in gt_beats:
        if seg_left <= int(beat["r_gt"]) < seg_right:
            p_gt_seg.append(int(beat["p_gt"]))
            r_gt_seg.append(int(beat["r_gt"]))
            t_gt_seg.append(int(beat["t_gt"]))
            p_det_seg.append(int(beat["p_det"]))
            r_det_seg.append(int(beat["r_det"]))
            t_det_seg.append(int(beat["t_det"]))

    cost_r_norm = normalize_curve(np.asarray(best_result["d_r"])[seg_left:seg_right])
    cost_p_norm = normalize_curve(np.asarray(best_result["d_p"])[seg_left:seg_right])
    cost_t_norm = normalize_curve(np.asarray(best_result["d_t"])[seg_left:seg_right])

    return {
        "t_axis": t_axis,
        "signal_segment": sig_seg,
        "signal_full": signal,
        "p_gt_seg": np.array(p_gt_seg, dtype=int),
        "r_gt_seg": np.array(r_gt_seg, dtype=int),
        "t_gt_seg": np.array(t_gt_seg, dtype=int),
        "p_det_seg": np.array(p_det_seg, dtype=int),
        "r_det_seg": np.array(r_det_seg, dtype=int),
        "t_det_seg": np.array(t_det_seg, dtype=int),
        "z1": np.asarray(best_result["zs_p"])[seg_left:seg_right, 0],
        "z2": np.asarray(best_result["zs_p"])[seg_left:seg_right, 1],
        "z3": np.asarray(best_result["zs_p"])[seg_left:seg_right, 2],
        "cost_r_norm": cost_r_norm,
        "cost_p_norm": cost_p_norm,
        "cost_t_norm": cost_t_norm,
        "lcr_r_seg": np.asarray(best_result["lcr_r"])[seg_left:seg_right],
        "lcr_p_seg": np.asarray(best_result["lcr_p"])[seg_left:seg_right],
        "lcr_t_seg": np.asarray(best_result["lcr_t"])[seg_left:seg_right],
    }


def plot_result(best_result: Dict[str, object], plot_data: Dict[str, object]) -> None:
    """Plot and save the qualitative QT delineation figure.

    Args:
        best_result: Selected record result dictionary.
        plot_data: Plot-ready segment dictionary.
    """
    t_axis = np.asarray(plot_data["t_axis"])
    signal_segment = np.asarray(plot_data["signal_segment"])
    signal_full = np.asarray(plot_data["signal_full"])

    p_gt_seg = np.asarray(plot_data["p_gt_seg"])
    r_gt_seg = np.asarray(plot_data["r_gt_seg"])
    t_gt_seg = np.asarray(plot_data["t_gt_seg"])

    p_det_seg = np.asarray(plot_data["p_det_seg"])
    r_det_seg = np.asarray(plot_data["r_det_seg"])
    t_det_seg = np.asarray(plot_data["t_det_seg"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t_axis, signal_segment, color="black", linewidth=1.1, label="ECG")
    axes[0].scatter(p_gt_seg / FS, signal_full[p_gt_seg], marker="x", s=50, label="P ground truth")
    axes[0].scatter(r_gt_seg / FS, signal_full[r_gt_seg], marker="x", s=50, label="R ground truth")
    axes[0].scatter(t_gt_seg / FS, signal_full[t_gt_seg], marker="x", s=50, label="T ground truth")
    axes[0].scatter(p_det_seg / FS, signal_full[p_det_seg], marker="o", s=30, label="P detected")
    axes[0].scatter(r_det_seg / FS, signal_full[r_det_seg], marker="o", s=30, label="R detected")
    axes[0].scatter(t_det_seg / FS, signal_full[t_det_seg], marker="o", s=30, label="T detected")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title(
        f"QT record {best_result['record']}: ECG with ground truth and ALSSM detections"
    )
    axes[0].legend(loc="upper right", ncol=3, fontsize=8, frameon=True)

    axes[1].plot(t_axis, np.asarray(plot_data["z1"]), linewidth=1.1, label=r"$z_1$")
    axes[1].plot(t_axis, np.asarray(plot_data["z2"]), linewidth=1.1, label=r"$z_2$")
    axes[1].plot(t_axis, np.asarray(plot_data["z3"]), linewidth=1.1, label=r"$z_3$")
    axes[1].set_ylabel("State value")
    axes[1].set_title("Whitened ALSSM state trajectory")
    axes[1].legend(loc="upper right", fontsize=8, frameon=True)

    axes[2].plot(t_axis, np.asarray(plot_data["cost_r_norm"]), linewidth=1.2, label="R cost")
    axes[2].plot(t_axis, np.asarray(plot_data["cost_p_norm"]), linewidth=1.2, label="P cost")
    axes[2].plot(t_axis, np.asarray(plot_data["cost_t_norm"]), linewidth=1.2, label="T cost")
    axes[2].set_ylabel("Normalized cost")
    axes[2].set_title("Normalized cost curves")
    axes[2].legend(loc="upper right", fontsize=8, frameon=True)

    axes[3].plot(t_axis, np.asarray(plot_data["lcr_r_seg"]), linewidth=1.2, label="R LCR")
    axes[3].plot(t_axis, np.asarray(plot_data["lcr_p_seg"]), linewidth=1.2, label="P LCR")
    axes[3].plot(t_axis, np.asarray(plot_data["lcr_t_seg"]), linewidth=1.2, label="T LCR")
    axes[3].set_ylabel("LCR")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Log-cost-ratio curves")
    axes[3].legend(loc="upper right", fontsize=8, frameon=True)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSaved figure to: {OUT_FILE}")
    print(f"Chosen record: {best_result['record']}")


def main() -> None:
    """Run the full qualitative QT delineation figure pipeline."""
    r_template, p_template, t_template, t_half_width = load_templates()

    best_result = choose_best_result(
        r_template=r_template,
        p_template=p_template,
        t_template=t_template,
        t_half_width=t_half_width,
    )

    plot_data = build_plot_segment(best_result)
    plot_result(best_result, plot_data)


if __name__ == "__main__":
    main()