"""Utilities for Chapter 11 T-wave modelling experiments.

This module contains the shared helper functions used by the QT-based T-wave
scripts. It supports three related approaches:

1. Point-template T-wave detection
2. Single trajectory-template T-wave detection
3. Clustered trajectory-template T-wave detection

The helpers are used for:
- loading QT records with best-lead selection
- resampling to 500 Hz
- extracting explicit R and T peaks
- computing ALSSM states and whitened states
- building point and trajectory templates
- evaluating timing errors on held-out QT records
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lmlib as lm
import numpy as np
import wfdb
from scipy.signal import resample

FS = 500
EPS = 1e-6

# Small clean training subset used in the thesis workflow.
PREFERRED_TRAIN_RECORDS = ["sel100", "sel102", "sel103", "sel104"]
N_TRAIN_RECORDS = 4

# Broad post-R T search window.
T_WINDOW_SEC = (0.10, 0.45)

# Common T-model settings for baseline and trajectory experiments.
T_POLYDEGREE = 2
T_L_SIDE = 40
T_G = 10
T_HALF_WIDTH = 20  # 41 samples total

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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


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


def list_qt_records(base_path: Path = QT_PATH) -> List[str]:
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


def parse_explicit_r_t(
    samples: np.ndarray,
    symbols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract explicit R and T peak annotations only.

    No midpoint fallback is used.

    Args:
        samples: Annotation sample indices.
        symbols: Annotation symbols.

    Returns:
        Tuple containing:
            - explicit R peak indices
            - explicit T peak indices
    """
    r_peaks: List[int] = []
    t_peaks: List[int] = []

    for sample, symbol in zip(samples, symbols):
        if symbol in BEAT_SYMBOLS:
            r_peaks.append(int(sample))
        if symbol.lower() == "t":
            t_peaks.append(int(sample))

    return (
        np.array(sorted(r_peaks), dtype=int),
        np.array(sorted(t_peaks), dtype=int),
    )


def pair_r_with_explicit_t(
    r_peaks: np.ndarray,
    t_peaks: np.ndarray,
    window_sec: Tuple[float, float] = T_WINDOW_SEC,
) -> List[Dict[str, int]]:
    """Pair each R peak with an explicit T peak in the post-R search window.

    Args:
        r_peaks: Explicit R peak indices.
        t_peaks: Explicit T peak indices.
        window_sec: T search window relative to R in seconds.

    Returns:
        List of beat dictionaries with R anchor and explicit T peak.
    """
    beats: List[Dict[str, int]] = []

    for r_peak in r_peaks:
        valid_t = t_peaks[
            (t_peaks > r_peak + int(window_sec[0] * FS))
            & (t_peaks < r_peak + int(window_sec[1] * FS))
        ]
        if len(valid_t) == 0:
            continue

        beats.append(
            {
                "r": int(r_peak),
                "t_peak": int(valid_t[0]),
            }
        )

    return beats


def run_alssm(
    signal: np.ndarray,
    polydegree: int = T_POLYDEGREE,
    l_side: int = T_L_SIDE,
    g: float = T_G,
) -> np.ndarray:
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
    segment = lm.Segment(
        a=-l_side,
        b=l_side,
        direction=lm.FORWARD,
        g=g,
    )
    cost = lm.CompositeCost((model,), (segment,), [[1]])
    rls = lm.create_rls(cost, multi_channel_set=False, steady_state=True)
    rls.filter(signal)
    return rls.minimize_x()


def whiten_states(states: np.ndarray) -> np.ndarray:
    """Whiten an ALSSM state sequence.

    Args:
        states: Raw ALSSM state sequence.

    Returns:
        Whitened state sequence.
    """
    covariance = np.cov(states.T)
    covariance_reg = covariance + EPS * np.eye(covariance.shape[0])
    chol = np.linalg.cholesky(covariance_reg)
    whitening_matrix = np.linalg.solve(chol, np.eye(chol.shape[0]))
    return (whitening_matrix @ states.T).T


def compute_zs(
    signal: np.ndarray,
    polydegree: int = T_POLYDEGREE,
    l_side: int = T_L_SIDE,
    g: float = T_G,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw and whitened ALSSM states for one signal.

    Args:
        signal: ECG signal.
        polydegree: Polynomial degree.
        l_side: Half-width of the ALSSM support window.
        g: Segment weighting parameter.

    Returns:
        Tuple containing:
            - raw state sequence
            - whitened state sequence
    """
    raw_states = run_alssm(signal, polydegree, l_side, g)
    whitened_states = whiten_states(raw_states)
    return raw_states, whitened_states


def build_trajectory(
    whitened_states: np.ndarray,
    center: int,
    half_width: int = T_HALF_WIDTH,
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


def pick_training_and_eval_records() -> Tuple[List[str], List[str]]:
    """Split QT records into a small training set and a held-out evaluation set.

    Returns:
        Tuple containing:
            - training record list
            - evaluation record list
    """
    all_records = list_qt_records()
    if len(all_records) < 6:
        raise RuntimeError("Too few QT records found.")

    train_records = [r for r in PREFERRED_TRAIN_RECORDS if r in all_records]

    if len(train_records) < N_TRAIN_RECORDS:
        for record_name in all_records:
            if record_name not in train_records:
                train_records.append(record_name)
            if len(train_records) == N_TRAIN_RECORDS:
                break

    eval_records = [r for r in all_records if r not in train_records]
    return train_records, eval_records


def build_point_t_template(train_records: Sequence[str]) -> np.ndarray:
    """Build the point-template T model from training records.

    Args:
        train_records: QT training record names.

    Returns:
        Median point-template T vector in whitened state space.

    Raises:
        RuntimeError: If no explicit-T states were found.
    """
    states: List[np.ndarray] = []

    for record_name in train_records:
        print(f"Building point T template from {record_name} ...")
        signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
        _ = used_lead, ann_ext

        r_peaks, t_peaks = parse_explicit_r_t(samples, symbols)
        beats = pair_r_with_explicit_t(r_peaks, t_peaks)

        if len(beats) == 0:
            continue

        _, whitened_states = compute_zs(signal)

        for beat in beats:
            t_peak = beat["t_peak"]
            if 0 <= t_peak < len(whitened_states):
                states.append(whitened_states[t_peak])

    if len(states) == 0:
        raise RuntimeError("No explicit-T states found for point template.")

    state_matrix = np.array(states, dtype=np.float64)
    return np.median(state_matrix, axis=0)


def detect_point_t_from_cached(
    whitened_states: np.ndarray,
    r_index: int,
    point_template: np.ndarray,
    window_sec: Tuple[float, float] = T_WINDOW_SEC,
) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
    """Detect a T peak with the point-template model.

    Args:
        whitened_states: Whitened ALSSM state sequence.
        r_index: R anchor index.
        point_template: Point-template T model.
        window_sec: T search window relative to R in seconds.

    Returns:
        Tuple containing:
            - detected T index or ``None``
            - pointwise cost curve
            - pointwise LCR curve
    """
    start = r_index + int(window_sec[0] * FS)
    end = r_index + int(window_sec[1] * FS)

    n_samples = len(whitened_states)
    distance = np.full(n_samples, np.nan, dtype=float)
    lcr = np.full(n_samples, np.nan, dtype=float)

    reference_energy = np.linalg.norm(point_template) ** 2
    best_t = None
    best_cost = np.inf

    for t_idx in range(start, end):
        if t_idx < 0 or t_idx >= n_samples:
            continue

        value = np.linalg.norm(whitened_states[t_idx] - point_template) ** 2
        distance[t_idx] = value
        lcr[t_idx] = -0.5 * np.log((value + EPS) / (reference_energy + EPS))

        if value < best_cost:
            best_cost = value
            best_t = t_idx

    return best_t, distance, lcr


def build_single_traj_template(
    train_records: Sequence[str],
    half_width: int = T_HALF_WIDTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the single trajectory-template T model.

    Args:
        train_records: QT training record names.
        half_width: Half-width of the trajectory window.

    Returns:
        Tuple containing:
            - median single trajectory template
            - full trajectory matrix

    Raises:
        RuntimeError: If no explicit-T trajectories were found.
    """
    trajectories: List[np.ndarray] = []

    for record_name in train_records:
        print(f"Building single trajectory template from {record_name} ...")
        signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
        _ = used_lead, ann_ext

        r_peaks, t_peaks = parse_explicit_r_t(samples, symbols)
        beats = pair_r_with_explicit_t(r_peaks, t_peaks)

        if len(beats) == 0:
            continue

        _, whitened_states = compute_zs(signal)

        for beat in beats:
            traj = build_trajectory(whitened_states, beat["t_peak"], half_width)
            if traj is not None:
                trajectories.append(traj)

    if len(trajectories) == 0:
        raise RuntimeError(
            "No explicit-T trajectories found for single trajectory template."
        )

    trajectory_matrix = np.array(trajectories, dtype=np.float64)
    template = np.median(trajectory_matrix, axis=0)

    return template, trajectory_matrix


def simple_kmeans(
    X: np.ndarray,
    k: int = 3,
    n_iter: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Run a simple K-means clustering on trajectory vectors.

    Args:
        X: Trajectory matrix.
        k: Number of clusters.
        n_iter: Maximum number of iterations.
        seed: Random seed.

    Returns:
        Cluster center matrix.
    """
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(X), size=k, replace=False)
    centers = X[init_idx].copy()

    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)

        new_centers = []
        for j in range(k):
            members = X[labels == j]
            if len(members) == 0:
                new_centers.append(centers[j])
            else:
                new_centers.append(np.mean(members, axis=0))

        new_centers = np.array(new_centers, dtype=np.float64)

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return centers


def build_clustered_traj_templates(
    train_records: Sequence[str],
    half_width: int = T_HALF_WIDTH,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build clustered trajectory-template T models.

    Args:
        train_records: QT training record names.
        half_width: Half-width of the trajectory window.
        k: Number of clusters.

    Returns:
        Tuple containing:
            - cluster-center matrix
            - full trajectory matrix
    """
    _, trajectories = build_single_traj_template(train_records, half_width=half_width)
    centers = simple_kmeans(trajectories, k=k, n_iter=40, seed=0)
    return centers, trajectories


def detect_single_traj_from_cached(
    whitened_states: np.ndarray,
    r_index: int,
    template: np.ndarray,
    half_width: int = T_HALF_WIDTH,
    window_sec: Tuple[float, float] = T_WINDOW_SEC,
) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
    """Detect a T peak with the single trajectory-template model.

    Args:
        whitened_states: Whitened ALSSM state sequence.
        r_index: R anchor index.
        template: Single trajectory template.
        half_width: Half-width of the trajectory window.
        window_sec: T search window relative to R in seconds.

    Returns:
        Tuple containing:
            - detected T index or ``None``
            - pointwise cost curve
            - pointwise LCR curve
    """
    start = r_index + int(window_sec[0] * FS)
    end = r_index + int(window_sec[1] * FS)

    n_samples = len(whitened_states)
    distance = np.full(n_samples, np.nan, dtype=float)
    lcr = np.full(n_samples, np.nan, dtype=float)

    reference_energy = np.linalg.norm(template) ** 2
    best_t = None
    best_cost = np.inf

    for t_idx in range(start, end):
        traj = build_trajectory(whitened_states, t_idx, half_width)
        if traj is None:
            continue

        value = np.linalg.norm(traj - template) ** 2
        distance[t_idx] = value
        lcr[t_idx] = -0.5 * np.log((value + EPS) / (reference_energy + EPS))

        if value < best_cost:
            best_cost = value
            best_t = t_idx

    return best_t, distance, lcr


def detect_clustered_traj_from_cached(
    whitened_states: np.ndarray,
    r_index: int,
    templates: np.ndarray,
    half_width: int = T_HALF_WIDTH,
    window_sec: Tuple[float, float] = T_WINDOW_SEC,
) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
    """Detect a T peak with clustered trajectory templates.

    Args:
        whitened_states: Whitened ALSSM state sequence.
        r_index: R anchor index.
        templates: Clustered trajectory template matrix.
        half_width: Half-width of the trajectory window.
        window_sec: T search window relative to R in seconds.

    Returns:
        Tuple containing:
            - detected T index or ``None``
            - pointwise cost curve
            - pointwise LCR curve
    """
    start = r_index + int(window_sec[0] * FS)
    end = r_index + int(window_sec[1] * FS)

    n_samples = len(whitened_states)
    distance = np.full(n_samples, np.nan, dtype=float)
    lcr = np.full(n_samples, np.nan, dtype=float)

    best_t = None
    best_cost = np.inf

    for t_idx in range(start, end):
        traj = build_trajectory(whitened_states, t_idx, half_width)
        if traj is None:
            continue

        values = np.array([np.linalg.norm(traj - mu) ** 2 for mu in templates])
        value = float(np.min(values))
        best_cluster = int(np.argmin(values))
        reference_energy = np.linalg.norm(templates[best_cluster]) ** 2

        distance[t_idx] = value
        lcr[t_idx] = -0.5 * np.log((value + EPS) / (reference_energy + EPS))

        if value < best_cost:
            best_cost = value
            best_t = t_idx

    return best_t, distance, lcr


def evaluate_point_baseline(
    point_template: np.ndarray,
    eval_records: Sequence[str],
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Evaluate the point-template baseline on held-out QT records.

    Args:
        point_template: Point-template T model.
        eval_records: Held-out evaluation record names.

    Returns:
        Tuple containing:
            - timing error array in milliseconds
            - example dictionaries with signals, detections, and LCR curves
    """
    errors_ms: List[float] = []
    examples: List[Dict[str, object]] = []

    for record_name in eval_records:
        print(f"Evaluating point-template baseline on {record_name} ...")
        try:
            signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
            _ = used_lead, ann_ext

            r_peaks, t_peaks = parse_explicit_r_t(samples, symbols)
            beats = pair_r_with_explicit_t(r_peaks, t_peaks)
            if len(beats) == 0:
                continue

            _, whitened_states = compute_zs(signal)

            for beat in beats:
                det_t, distance, lcr = detect_point_t_from_cached(
                    whitened_states,
                    beat["r"],
                    point_template,
                )
                _ = distance

                if det_t is None:
                    continue

                err_ms = 1000.0 * (det_t - beat["t_peak"]) / FS
                errors_ms.append(err_ms)

                examples.append(
                    {
                        "record": record_name,
                        "signal": signal,
                        "r": beat["r"],
                        "t_gt": beat["t_peak"],
                        "t_det": det_t,
                        "err_ms": err_ms,
                        "lcr": lcr,
                    }
                )

        except Exception as exc:
            print(f"  skipped {record_name}: {exc}")

    return np.array(errors_ms, dtype=float), examples


def evaluate_single_vs_clustered(
    single_template: np.ndarray,
    clustered_templates: np.ndarray,
    eval_records: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate single versus clustered trajectory templates.

    Args:
        single_template: Single trajectory template.
        clustered_templates: Clustered trajectory template matrix.
        eval_records: Held-out evaluation record names.

    Returns:
        Tuple containing:
            - single-template timing errors in milliseconds
            - clustered-template timing errors in milliseconds
    """
    err_single: List[float] = []
    err_clustered: List[float] = []

    for record_name in eval_records:
        print(f"Evaluating single vs clustered on {record_name} ...")
        try:
            signal, samples, symbols, used_lead, ann_ext = load_qt_record_500hz(record_name)
            _ = used_lead, ann_ext

            r_peaks, t_peaks = parse_explicit_r_t(samples, symbols)
            beats = pair_r_with_explicit_t(r_peaks, t_peaks)
            if len(beats) == 0:
                continue

            _, whitened_states = compute_zs(signal)

            for beat in beats:
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

                if det_single is not None:
                    err_single.append(1000.0 * (det_single - beat["t_peak"]) / FS)
                if det_cluster is not None:
                    err_clustered.append(1000.0 * (det_cluster - beat["t_peak"]) / FS)

        except Exception as exc:
            print(f"  skipped {record_name}: {exc}")

    return (
        np.array(err_single, dtype=float),
        np.array(err_clustered, dtype=float),
    )


def summarize_errors(errors_ms: np.ndarray) -> Optional[Dict[str, float]]:
    """Summarize timing errors in milliseconds.

    Args:
        errors_ms: Timing error array in milliseconds.

    Returns:
        Dictionary with summary statistics, or ``None`` if empty.
    """
    if len(errors_ms) == 0:
        return None

    bias = float(np.mean(errors_ms))
    std = float(np.std(errors_ms))
    mae = float(np.mean(np.abs(errors_ms)))
    rmse = float(np.sqrt(np.mean(errors_ms ** 2)))

    return {
        "n": int(len(errors_ms)),
        "bias": bias,
        "std": std,
        "mae": mae,
        "rmse": rmse,
    }