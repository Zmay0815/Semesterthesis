"""Plot single-beat template diagnostics for R, P, and T models.

This script visualizes one QT beat using one of three template sources:

1. MIT-derived pointwise templates
2. QT-derived pointwise templates
3. Curated QT-derived pointwise templates

For the selected beat, the figure shows:
- ECG with annotated and detected P, R, and T points
- unwhitened and whitened ALSSM state trajectories for R, P, and T
- raw pointwise cost curves
- log-cost-ratio (LCR) curves

The script is designed to work from the unzipped thesis folder without
hard-coded user-specific paths.
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
from scipy.signal import find_peaks, resample

FS = 500
EPS = 1e-6

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
OUTPUT_DIR = PROJECT_ROOT / "generated_outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_SOURCE = "qt_curated"
RECORD = "sel114"

OUT_FILE = OUTPUT_DIR / f"fig16_clean_{TEMPLATE_SOURCE}_{RECORD}.png"

CAND_EXTS = ["pu", "q1c", "qt1", "atr"]

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
T_WINDOW_SEC = (0.10, 0.45)

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
        List of beat dictionaries with explicit P, R, and T annotated peaks.
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


def whiten_local(states: np.ndarray) -> np.ndarray:
    """Whiten a state sequence using its local covariance.

    Args:
        states: Raw state sequence.

    Returns:
        Locally whitened state sequence.
    """
    covariance = np.cov(states.T)
    covariance_reg = covariance + EPS * np.eye(covariance.shape[0])
    chol = np.linalg.cholesky(covariance_reg)
    whitening_matrix = np.linalg.solve(chol, np.eye(chol.shape[0]))
    return (whitening_matrix @ states.T).T


def apply_fixed_whitening(states: np.ndarray, whitening_matrix: np.ndarray) -> np.ndarray:
    """Apply a fixed whitening matrix to a state sequence.

    Args:
        states: Raw state sequence.
        whitening_matrix: Fixed whitening matrix.

    Returns:
        Whitened state sequence.
    """
    return (whitening_matrix @ states.T).T


def resolve_existing_file(candidates: Sequence[Path], label: str) -> Path:
    """Resolve the first existing file from a candidate list.

    Args:
        candidates: Candidate file paths.
        label: Human-readable label for error messages.

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


def resolve_existing_file_or_none(candidates: Sequence[Path]) -> Optional[Path]:
    """Resolve the first existing file from a candidate list, or return None.

    Args:
        candidates: Candidate file paths.

    Returns:
        Existing file path, or None if none exist.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_qt_spec() -> Dict[str, object]:
    """Build the specification for QT-derived pointwise templates."""
    return {
        "title_name": "QT-derived pointwise templates",
        "R_template_file": resolve_existing_file(
            [
                NPZ_DIR / "r_template_qt_layer2_500hz.npy",
            ],
            "QT R template",
        ),
        "P_template_file": resolve_existing_file(
            [
                NPZ_DIR / "p_template_qt_layer2_500hz.npy",
            ],
            "QT P template",
        ),
        "T_template_file": resolve_existing_file(
            [
                NPZ_DIR / "t_template_qt_layer2_500hz.npy",
            ],
            "QT T template",
        ),
        "V_r_file": resolve_existing_file(
            [
                NPZ_DIR / "V_r_qt_layer2_500hz.npy",
            ],
            "QT R whitening matrix",
        ),
        "V_p_file": resolve_existing_file(
            [
                NPZ_DIR / "V_p_qt_layer2_500hz.npy",
            ],
            "QT P whitening matrix",
        ),
        "V_t_file": resolve_existing_file(
            [
                NPZ_DIR / "V_t_qt_layer2_500hz.npy",
            ],
            "QT T whitening matrix",
        ),
        "R_space": "fixed_white",
        "P_space": "fixed_white",
        "T_space": "fixed_white",
    }


def build_qt_curated_spec() -> Dict[str, object]:
    """Build the specification for curated QT-derived pointwise templates."""
    return {
        "title_name": "Curated QT-derived pointwise templates",
        "R_template_file": resolve_existing_file(
            [NPZ_DIR / "r_template_qt_curated_main_500hz.npy"],
            "curated QT R template",
        ),
        "P_template_file": resolve_existing_file(
            [NPZ_DIR / "p_template_qt_curated_main_500hz.npy"],
            "curated QT P template",
        ),
        "T_template_file": resolve_existing_file(
            [NPZ_DIR / "t_template_qt_curated_main_500hz.npy"],
            "curated QT T template",
        ),
        "V_r_file": resolve_existing_file(
            [NPZ_DIR / "V_r_qt_curated_500hz.npy"],
            "curated QT R whitening matrix",
        ),
        "V_p_file": resolve_existing_file(
            [NPZ_DIR / "V_p_qt_curated_500hz.npy"],
            "curated QT P whitening matrix",
        ),
        "V_t_file": resolve_existing_file(
            [NPZ_DIR / "V_t_qt_curated_500hz.npy"],
            "curated QT T whitening matrix",
        ),
        "R_space": "fixed_white",
        "P_space": "fixed_white",
        "T_space": "fixed_white",
    }


def build_mit_spec() -> Dict[str, object]:
    """Build the specification for MIT-derived pointwise templates."""
    return {
        "title_name": "MIT-derived pointwise templates",
        "R_template_file": resolve_existing_file(
            [NPZ_DIR / "r_template_mitbih_500hz.npy"],
            "MIT R template",
        ),
        "P_template_file": resolve_existing_file(
            [NPZ_DIR / "p_template_mitbih_500hz.npy"],
            "MIT P template",
        ),
        "T_template_file": resolve_existing_file(
            [
                NPZ_DIR / "t_template_mitbih_500hz.npy",
                NPZ_DIR / "t_template_pointwise_mitbih_500hz.npy",
            ],
            "MIT T template",
        ),
        "V_r_file": resolve_existing_file_or_none(
            [NPZ_DIR / "V_r_mitbih_500hz.npy"]
        ),
        "V_p_file": resolve_existing_file_or_none(
            [NPZ_DIR / "V_p_mitbih_500hz.npy"]
        ),
        "V_t_file": resolve_existing_file_or_none(
            [NPZ_DIR / "V_t_mitbih_500hz.npy"]
        ),
        "R_space": "raw",
        "P_space": "fixed_white_or_local",
        "T_space": "fixed_white_or_local",
    }


def get_states_for_template(
    raw_states: np.ndarray,
    space_mode: str,
    whitening_file: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare states in the template space and local whitened visualization space.

    Args:
        raw_states: Raw ALSSM state sequence.
        space_mode: Template-space mode.
        whitening_file: Optional fixed whitening matrix file.

    Returns:
        Tuple containing:
            - states in the template space
            - locally whitened states for visualization
    """
    local_whitened = whiten_local(raw_states)

    if space_mode == "raw":
        return raw_states, local_whitened

    if space_mode == "fixed_white":
        whitening_matrix = np.load(whitening_file)
        fixed_whitened = apply_fixed_whitening(raw_states, whitening_matrix)
        return fixed_whitened, fixed_whitened

    if space_mode == "fixed_white_or_local":
        if whitening_file is not None and whitening_file.exists():
            whitening_matrix = np.load(whitening_file)
            fixed_whitened = apply_fixed_whitening(raw_states, whitening_matrix)
            return fixed_whitened, fixed_whitened
        return local_whitened, local_whitened

    raise ValueError(f"Unknown space mode: {space_mode}")


def detect_record(record_name: str, spec: Dict[str, object]) -> Dict[str, object]:
    """Run the single-beat diagnostic detection pipeline on one QT record.

    Args:
        record_name: QT record name.
        spec: Template specification dictionary.

    Returns:
        Result dictionary for plotting.

    Raises:
        RuntimeError: If too few usable beats were found.
    """
    signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
    _ = used_lead, ann_ext

    r_gt_all, p_gt_all, t_gt_all = parse_explicit_prt(samples, symbols)
    all_gt_beats = pair_gt_beats(r_gt_all, p_gt_all, t_gt_all)

    if len(all_gt_beats) < 3:
        raise RuntimeError("Too few explicit P/R/T beats in this record.")

    r_template = np.load(spec["R_template_file"])
    p_template = np.load(spec["P_template_file"])
    t_template = np.load(spec["T_template_file"])

    xhat_r = run_alssm(signal, R_POLYDEGREE, R_L_SIDE, R_G)
    states_r, zs_r = get_states_for_template(
        xhat_r,
        str(spec["R_space"]),
        spec["V_r_file"],
    )
    d_r = np.linalg.norm(states_r - r_template, axis=1) ** 2
    e_r = np.linalg.norm(r_template) ** 2
    lcr_r = -0.5 * np.log((d_r + EPS) / (e_r + EPS))

    r_detected, _ = find_peaks(
        lcr_r,
        height=np.percentile(lcr_r, R_THRESHOLD_PERCENTILE),
        distance=int(R_REFRACTORY_SEC * FS),
    )

    xhat_p = run_alssm(signal, P_POLYDEGREE, P_L_SIDE, P_G)
    states_p, zs_p = get_states_for_template(
        xhat_p,
        str(spec["P_space"]),
        spec["V_p_file"],
    )
    d_p = np.linalg.norm(states_p - p_template, axis=1) ** 2
    e_p = np.linalg.norm(p_template) ** 2
    lcr_p = -0.5 * np.log((d_p + EPS) / (e_p + EPS))

    xhat_t = run_alssm(signal, T_POLYDEGREE, T_L_SIDE, T_G)
    states_t, zs_t = get_states_for_template(
        xhat_t,
        str(spec["T_space"]),
        spec["V_t_file"],
    )
    d_t = np.linalg.norm(states_t - t_template, axis=1) ** 2
    e_t = np.linalg.norm(t_template) ** 2
    lcr_t = -0.5 * np.log((d_t + EPS) / (e_t + EPS))

    matched_beats = []

    for beat in all_gt_beats:
        r_gt = int(beat["r_gt"])

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
        p_det = int(p_start + np.argmax(lcr_p[p_start:p_end]))

        t_start = r_det + int(T_WINDOW_SEC[0] * FS)
        t_end = r_det + int(T_WINDOW_SEC[1] * FS)
        if t_start < 0 or t_end >= len(signal) or t_end <= t_start:
            continue
        t_det = int(t_start + np.argmax(lcr_t[t_start:t_end]))

        p_err = abs(p_det - int(beat["p_gt"]))
        r_err = abs(r_det - int(beat["r_gt"]))
        t_err = abs(t_det - int(beat["t_gt"]))

        matched_beats.append(
            {
                "p_gt": int(beat["p_gt"]),
                "r_gt": int(beat["r_gt"]),
                "t_gt": int(beat["t_gt"]),
                "p_det": p_det,
                "r_det": r_det,
                "t_det": t_det,
                "score": p_err + r_err + t_err,
            }
        )

    if len(matched_beats) == 0:
        raise RuntimeError("No matched beats found.")

    matched_beats = sorted(matched_beats, key=lambda x: int(x["score"]))
    best_beat = matched_beats[0]

    return {
        "spec_title": spec["title_name"],
        "record": record_name,
        "signal": signal,
        "all_gt_beats": all_gt_beats,
        "matched_beats": matched_beats,
        "best_beat": best_beat,
        "xhat_r": xhat_r,
        "xhat_p": xhat_p,
        "xhat_t": xhat_t,
        "zs_r": zs_r,
        "zs_p": zs_p,
        "zs_t": zs_t,
        "d_r": d_r,
        "d_p": d_p,
        "d_t": d_t,
        "lcr_r": lcr_r,
        "lcr_p": lcr_p,
        "lcr_t": lcr_t,
    }


def add_guides(
    ax: plt.Axes,
    p_win_left: int,
    p_win_right: int,
    t_win_left: int,
    t_win_right: int,
    r_win_left: int,
    r_win_right: int,
    p_gt: int,
    r_gt: int,
    t_gt: int,
    p_det: int,
    r_det: int,
    t_det: int,
) -> None:
    """Add windows and guide lines to one axis."""
    ax.axvspan(p_win_left / FS, p_win_right / FS, color="C0", alpha=0.08)
    ax.axvspan(t_win_left / FS, t_win_right / FS, color="C2", alpha=0.08)
    ax.axvspan(r_win_left / FS, r_win_right / FS, color="C1", alpha=0.06)

    ax.axvline(p_gt / FS, color="C0", linestyle=":", linewidth=0.8, alpha=0.85)
    ax.axvline(r_gt / FS, color="C1", linestyle=":", linewidth=0.8, alpha=0.85)
    ax.axvline(t_gt / FS, color="C2", linestyle=":", linewidth=0.8, alpha=0.85)

    ax.axvline(p_det / FS, color="C3", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.axvline(r_det / FS, color="C4", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.axvline(t_det / FS, color="C5", linestyle="--", linewidth=0.8, alpha=0.85)


def masked_curve(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask a curve outside the region of interest."""
    out = np.full_like(values, np.nan, dtype=float)
    out[mask] = values[mask]
    return out


def plot_clean_singlebeat(result: Dict[str, object], out_file: Path) -> None:
    """Plot and save the single-beat diagnostic figure.

    Args:
        result: Detection result dictionary.
        out_file: Output image path.
    """
    signal = np.asarray(result["signal"])
    best_beat = result["best_beat"]

    r_gt = int(best_beat["r_gt"])
    p_gt = int(best_beat["p_gt"])
    t_gt = int(best_beat["t_gt"])

    r_det = int(best_beat["r_det"])
    p_det = int(best_beat["p_det"])
    t_det = int(best_beat["t_det"])

    seg_left = max(0, r_gt - int(DISPLAY_LEFT_SEC * FS))
    seg_right = min(len(signal), r_gt + int(DISPLAY_RIGHT_SEC * FS))

    t_axis = np.arange(seg_left, seg_right) / FS
    sig_seg = signal[seg_left:seg_right]

    p_win_left = r_det + int(P_WINDOW_SEC[0] * FS)
    p_win_right = r_det + int(P_WINDOW_SEC[1] * FS)

    t_win_left = r_det + int(T_WINDOW_SEC[0] * FS)
    t_win_right = r_det + int(T_WINDOW_SEC[1] * FS)

    r_win_left = r_det - int(0.06 * FS)
    r_win_right = r_det + int(0.06 * FS)

    xr1 = np.asarray(result["xhat_r"])[seg_left:seg_right, 0]
    xr2 = np.asarray(result["xhat_r"])[seg_left:seg_right, 1]
    xr3 = np.asarray(result["xhat_r"])[seg_left:seg_right, 2]

    xp1 = np.asarray(result["xhat_p"])[seg_left:seg_right, 0]
    xp2 = np.asarray(result["xhat_p"])[seg_left:seg_right, 1]
    xp3 = np.asarray(result["xhat_p"])[seg_left:seg_right, 2]

    xt1 = np.asarray(result["xhat_t"])[seg_left:seg_right, 0]
    xt2 = np.asarray(result["xhat_t"])[seg_left:seg_right, 1]
    xt3 = np.asarray(result["xhat_t"])[seg_left:seg_right, 2]

    zr1 = np.asarray(result["zs_r"])[seg_left:seg_right, 0]
    zr2 = np.asarray(result["zs_r"])[seg_left:seg_right, 1]
    zr3 = np.asarray(result["zs_r"])[seg_left:seg_right, 2]

    zp1 = np.asarray(result["zs_p"])[seg_left:seg_right, 0]
    zp2 = np.asarray(result["zs_p"])[seg_left:seg_right, 1]
    zp3 = np.asarray(result["zs_p"])[seg_left:seg_right, 2]

    zt1 = np.asarray(result["zs_t"])[seg_left:seg_right, 0]
    zt2 = np.asarray(result["zs_t"])[seg_left:seg_right, 1]
    zt3 = np.asarray(result["zs_t"])[seg_left:seg_right, 2]

    cost_r_seg = np.asarray(result["d_r"])[seg_left:seg_right]
    cost_p_seg = np.asarray(result["d_p"])[seg_left:seg_right]
    cost_t_seg = np.asarray(result["d_t"])[seg_left:seg_right]

    lcr_r_seg = np.asarray(result["lcr_r"])[seg_left:seg_right]
    lcr_p_seg = np.asarray(result["lcr_p"])[seg_left:seg_right]
    lcr_t_seg = np.asarray(result["lcr_t"])[seg_left:seg_right]

    seg_indices = np.arange(seg_left, seg_right)

    r_mask = (seg_indices >= r_win_left) & (seg_indices <= r_win_right)
    p_mask = (seg_indices >= p_win_left) & (seg_indices <= p_win_right)
    t_mask = (seg_indices >= t_win_left) & (seg_indices <= t_win_right)

    cost_r_focus = masked_curve(cost_r_seg, r_mask)
    cost_p_focus = masked_curve(cost_p_seg, p_mask)
    cost_t_focus = masked_curve(cost_t_seg, t_mask)

    lcr_r_focus = masked_curve(lcr_r_seg, r_mask)
    lcr_p_focus = masked_curve(lcr_p_seg, p_mask)
    lcr_t_focus = masked_curve(lcr_t_seg, t_mask)

    fig = plt.figure(figsize=(15, 13))
    grid_spec = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0],
        hspace=0.42,
        wspace=0.22,
    )

    ax_ecg = fig.add_subplot(grid_spec[0, :])

    ax_r_raw = fig.add_subplot(grid_spec[1, 0], sharex=ax_ecg)
    ax_r_white = fig.add_subplot(grid_spec[1, 1], sharex=ax_ecg)

    ax_p_raw = fig.add_subplot(grid_spec[2, 0], sharex=ax_ecg)
    ax_p_white = fig.add_subplot(grid_spec[2, 1], sharex=ax_ecg)

    ax_t_raw = fig.add_subplot(grid_spec[3, 0], sharex=ax_ecg)
    ax_t_white = fig.add_subplot(grid_spec[3, 1], sharex=ax_ecg)

    ax_cost = fig.add_subplot(grid_spec[4, 0], sharex=ax_ecg)
    ax_lcr = fig.add_subplot(grid_spec[4, 1], sharex=ax_ecg)

    ax_ecg.plot(t_axis, sig_seg, color="black", linewidth=1.25, label="ECG")
    ax_ecg.axvspan(p_win_left / FS, p_win_right / FS, color="C0", alpha=0.10, label="P window")
    ax_ecg.axvspan(t_win_left / FS, t_win_right / FS, color="C2", alpha=0.10, label="T window")

    ax_ecg.scatter(p_gt / FS, signal[p_gt], marker="x", s=75, label="P annotated", zorder=5)
    ax_ecg.scatter(r_gt / FS, signal[r_gt], marker="x", s=75, label="R annotated", zorder=5)
    ax_ecg.scatter(t_gt / FS, signal[t_gt], marker="x", s=75, label="T annotated", zorder=5)

    ax_ecg.scatter(p_det / FS, signal[p_det], marker="o", s=48, label="P detected", zorder=6)
    ax_ecg.scatter(r_det / FS, signal[r_det], marker="o", s=48, label="R detected", zorder=6)
    ax_ecg.scatter(t_det / FS, signal[t_det], marker="o", s=48, label="T detected", zorder=6)

    for idx, col in [(p_gt, "C0"), (r_gt, "C1"), (t_gt, "C2")]:
        ax_ecg.axvline(idx / FS, color=col, linestyle=":", linewidth=0.9, alpha=0.8)

    for idx, col in [(p_det, "C3"), (r_det, "C4"), (t_det, "C5")]:
        ax_ecg.axvline(idx / FS, color=col, linestyle="--", linewidth=0.9, alpha=0.8)

    ax_ecg.set_ylabel("Amplitude (mV)")
    ax_ecg.set_title(f"Single-beat example using {result['spec_title']}", fontsize=12)
    ax_ecg.legend(loc="upper right", ncol=4, fontsize=8, frameon=True)

    ax_r_raw.plot(t_axis, xr1, linewidth=1.0, label=r"$x_1$")
    ax_r_raw.plot(t_axis, xr2, linewidth=1.0, label=r"$x_2$")
    ax_r_raw.plot(t_axis, xr3, linewidth=1.0, label=r"$x_3$")
    add_guides(
        ax_r_raw,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_r_raw.set_title("R model: unwhitened ALSSM state trajectory", fontsize=10)
    ax_r_raw.set_ylabel("State value")
    ax_r_raw.legend(loc="upper right", fontsize=8, frameon=True)

    ax_r_white.plot(t_axis, zr1, linewidth=1.0, label=r"$z_1$")
    ax_r_white.plot(t_axis, zr2, linewidth=1.0, label=r"$z_2$")
    ax_r_white.plot(t_axis, zr3, linewidth=1.0, label=r"$z_3$")
    add_guides(
        ax_r_white,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_r_white.set_title("R model: whitened ALSSM state trajectory", fontsize=10)
    ax_r_white.legend(loc="upper right", fontsize=8, frameon=True)

    ax_p_raw.plot(t_axis, xp1, linewidth=1.0, label=r"$x_1$")
    ax_p_raw.plot(t_axis, xp2, linewidth=1.0, label=r"$x_2$")
    ax_p_raw.plot(t_axis, xp3, linewidth=1.0, label=r"$x_3$")
    add_guides(
        ax_p_raw,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_p_raw.set_title("P model: unwhitened ALSSM state trajectory", fontsize=10)
    ax_p_raw.set_ylabel("State value")
    ax_p_raw.legend(loc="upper right", fontsize=8, frameon=True)

    ax_p_white.plot(t_axis, zp1, linewidth=1.0, label=r"$z_1$")
    ax_p_white.plot(t_axis, zp2, linewidth=1.0, label=r"$z_2$")
    ax_p_white.plot(t_axis, zp3, linewidth=1.0, label=r"$z_3$")
    add_guides(
        ax_p_white,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_p_white.set_title("P model: whitened ALSSM state trajectory", fontsize=10)
    ax_p_white.legend(loc="upper right", fontsize=8, frameon=True)

    ax_t_raw.plot(t_axis, xt1, linewidth=1.0, label=r"$x_1$")
    ax_t_raw.plot(t_axis, xt2, linewidth=1.0, label=r"$x_2$")
    ax_t_raw.plot(t_axis, xt3, linewidth=1.0, label=r"$x_3$")
    add_guides(
        ax_t_raw,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_t_raw.set_title("T model: unwhitened ALSSM state trajectory", fontsize=10)
    ax_t_raw.set_ylabel("State value")
    ax_t_raw.legend(loc="upper right", fontsize=8, frameon=True)

    ax_t_white.plot(t_axis, zt1, linewidth=1.0, label=r"$z_1$")
    ax_t_white.plot(t_axis, zt2, linewidth=1.0, label=r"$z_2$")
    ax_t_white.plot(t_axis, zt3, linewidth=1.0, label=r"$z_3$")
    add_guides(
        ax_t_white,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_t_white.set_title("T model: whitened ALSSM state trajectory", fontsize=10)
    ax_t_white.legend(loc="upper right", fontsize=8, frameon=True)

    ax_cost.plot(t_axis, cost_r_seg, color="0.80", linewidth=1.0)
    ax_cost.plot(t_axis, cost_p_seg, color="0.80", linewidth=1.0)
    ax_cost.plot(t_axis, cost_t_seg, color="0.80", linewidth=1.0)

    ax_cost.plot(t_axis, cost_r_focus, linewidth=1.8, label="R cost")
    ax_cost.plot(t_axis, cost_p_focus, linewidth=1.8, label="P cost")
    ax_cost.plot(t_axis, cost_t_focus, linewidth=1.8, label="T cost")

    add_guides(
        ax_cost,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_cost.set_title("Raw pointwise cost curves", fontsize=10)
    ax_cost.set_ylabel("Cost")
    ax_cost.set_yscale("log")
    ax_cost.legend(loc="upper right", fontsize=8, frameon=True)

    ax_lcr.plot(t_axis, lcr_r_seg, color="0.80", linewidth=1.0)
    ax_lcr.plot(t_axis, lcr_p_seg, color="0.80", linewidth=1.0)
    ax_lcr.plot(t_axis, lcr_t_seg, color="0.80", linewidth=1.0)

    ax_lcr.plot(t_axis, lcr_r_focus, linewidth=1.8, label="R LCR")
    ax_lcr.plot(t_axis, lcr_p_focus, linewidth=1.8, label="P LCR")
    ax_lcr.plot(t_axis, lcr_t_focus, linewidth=1.8, label="T LCR")

    add_guides(
        ax_lcr,
        p_win_left,
        p_win_right,
        t_win_left,
        t_win_right,
        r_win_left,
        r_win_right,
        p_gt,
        r_gt,
        t_gt,
        p_det,
        r_det,
        t_det,
    )
    ax_lcr.set_title("Log-cost-ratio curves", fontsize=10)
    ax_lcr.set_ylabel("LCR")
    ax_lcr.legend(loc="upper right", fontsize=8, frameon=True)

    for ax in [
        ax_ecg,
        ax_r_raw,
        ax_r_white,
        ax_p_raw,
        ax_p_white,
        ax_t_raw,
        ax_t_white,
        ax_cost,
        ax_lcr,
    ]:
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    ax_cost.set_xlabel("Time (s)")
    ax_lcr.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {out_file}")


def build_template_spec(template_source: str) -> Dict[str, object]:
    """Build the template specification for one source name.

    Args:
        template_source: One of ``'mit'``, ``'qt'``, or ``'qt_curated'``.

    Returns:
        Template specification dictionary.

    Raises:
        ValueError: If the source name is invalid.
    """
    source = template_source.lower()

    if source == "mit":
        return build_mit_spec()
    if source == "qt":
        return build_qt_spec()
    if source == "qt_curated":
        return build_qt_curated_spec()

    raise ValueError("TEMPLATE_SOURCE must be 'mit', 'qt', or 'qt_curated'.")


def main() -> None:
    """Run the single-beat template diagnostics figure pipeline."""
    available = list_qt_records(QT_PATH)
    if RECORD not in available:
        raise RuntimeError(f"Record {RECORD} was not found in the QT database.")

    spec = build_template_spec(TEMPLATE_SOURCE)
    result = detect_record(RECORD, spec)
    plot_clean_singlebeat(result, OUT_FILE)


if __name__ == "__main__":
    main()