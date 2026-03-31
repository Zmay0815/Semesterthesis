"""ALSSM pipeline utilities for P, QRS, T, and combined PQRST processing.

This module provides an interactive ALSSM-based processing pipeline for ECG
feature channels. It supports:

1. Classical ALSSM channels for P, Q, R, S, T, and W
2. Pseudo-2D combination nodes such as QRS and PQRST
3. Optional neural-network input channels for P, QRS, and T
4. Interactive visualization of the processing graph and extracted features

The module is designed to work from the unzipped thesis folder without
hard-coded user-specific paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lmlib as lm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from adaptive_peak_thresholding import adaptive_peak_detection_med_average

DEFAULT_FS = 2400
CSV_FILENAME = "EECG_BASELINE_1CH_10S_FS2400HZ.csv"

USE_WIDE_ROUTE = False
USE_PQT_NN = True
PQTNET_FILENAME = "pqt_unet.pt"

plt.close("all")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

try:
    from nn_integration_pqt import nn_pqt_prob, probs_to_binary

    PQT_NN_AVAILABLE = True
except Exception:
    PQT_NN_AVAILABLE = False

all_models: Dict[str, Dict[str, Any]] = {
    "P": {"ref_index": 1310, "polydegree": 2, "l_side": 60, "g": 40},
    "Q": {"ref_index": 1370, "polydegree": 2, "l_side": 18, "g": 40},
    "R": {"ref_index": 1435, "polydegree": 2, "l_side": 25, "g": 40},
    "S": {"ref_index": 1482, "polydegree": 2, "l_side": 10, "g": 40},
    "T": {"ref_index": 1850, "polydegree": 2, "l_side": 60, "g": 40},
    "W": {"ref_index": 1620, "polydegree": 4, "l_side": 380, "g": 100},
}

combo_params_2d: Dict[str, Dict[str, Any]] = {
    "QRS": {"ref_index": 1450, "polydegree": 3, "l_side": 120, "g": 300, "weight": 4},
    "PQRST": {"ref_index": 1600, "polydegree": 3, "l_side": 320, "g": 250, "weight": 5},
}

if USE_WIDE_ROUTE:
    combo_params_2d.update(
        {
            "QRS_wide": {
                "ref_index": 1600,
                "polydegree": 3,
                "l_side": 220,
                "g": 400,
                "weight": 3,
            },
            "PQRST_wide": {
                "ref_index": 1600,
                "polydegree": 3,
                "l_side": 300,
                "g": 500,
                "weight": 5,
            },
        }
    )

combo_params_2d.update(
    {
        "P_NN": {"ref_index": 1310, "polydegree": 3, "l_side": 60, "g": 200, "weight": 4},
        "QRS_NN": {"ref_index": 1450, "polydegree": 3, "l_side": 80, "g": 300, "weight": 6},
        "T_NN": {"ref_index": 1850, "polydegree": 3, "l_side": 80, "g": 200, "weight": 4},
        "P": {"ref_index": 1310, "polydegree": 3, "l_side": 70, "g": 200, "weight": 4},
        "T": {"ref_index": 1850, "polydegree": 3, "l_side": 90, "g": 200, "weight": 4},
    }
)


def resolve_existing_file(candidates: List[Path], label: str) -> Path:
    """Resolve the first existing file from a list of candidates.

    Args:
        candidates: Candidate file paths.
        label: Human-readable label for error messages.

    Returns:
        Existing file path.

    Raises:
        FileNotFoundError: If none of the candidates exists.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find {label}.\nSearched in:\n{searched}")


def robust_csv_path(filename: str) -> Path:
    """Resolve the ECG CSV file path.

    Args:
        filename: CSV filename.

    Returns:
        Existing CSV file path.
    """
    candidates = [
        SCRIPT_DIR / filename,
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "datasets_and_preprocessing" / filename,
        PROJECT_ROOT / "generated_outputs" / filename,
    ]
    return resolve_existing_file(candidates, f"CSV file '{filename}'")


def resolve_pqtnet_path(filename: str) -> Path:
    """Resolve the trained PQT U-Net model path.

    Args:
        filename: Model filename.

    Returns:
        Existing model file path.
    """
    candidates = [
        SCRIPT_DIR / filename,
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "generated_outputs" / "models" / filename,
    ]
    return resolve_existing_file(candidates, f"PQT model '{filename}'")


def filter_baseline(
    y: np.ndarray,
    g_bl: float,
    poly_degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate and remove the ECG baseline using an ALSSM model.

    Args:
        y: Input ECG signal.
        g_bl: Segment weighting parameter.
        poly_degree: Polynomial degree of the baseline model.

    Returns:
        Tuple containing:
            - estimated baseline
            - baseline-corrected signal
    """
    alssm_bl = lm.AlssmPoly(poly_degree=poly_degree, label="baseline")
    seg_bl = lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=g_bl)
    cost = lm.CompositeCost((alssm_bl,), (seg_bl,), F=[[1]])
    rls = lm.RLSAlssm(cost)
    xs = rls.filter_minimize_x(y)
    baseline = cost.eval_alssm_output(xs, alssm_weights=[1])
    y_flat = y - baseline
    return baseline, y_flat


def merge_cr_with_noise(
    results: List[Dict[str, Any]],
    noise_level: float = 1e-6,
) -> np.ndarray:
    """Merge binary feature channels into a pseudo-2D matrix.

    Args:
        results: List of result dictionaries with ``binary_lcr`` and optional
            ``weight`` entries.
        noise_level: Small Gaussian noise level added for tie-breaking.

    Returns:
        Pseudo-2D channel matrix.
    """
    cols = []
    for res in results:
        binary = np.asarray(res["binary_lcr"])
        vec = binary.max(axis=1) if binary.ndim == 2 else binary
        weight = res.get("weight", 1.0)
        cols.append((vec * weight)[:, None])

    cr_matrix = np.concatenate(cols, axis=1)

    if noise_level > 0:
        cr_matrix += np.random.normal(scale=noise_level, size=cr_matrix.shape)

    return cr_matrix


def extract_alssm_features(signal: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ALSSM features, distances, LCR, and adaptive peaks.

    Pipeline:
        1. Build ALSSM polynomial model and segment from ``params``.
        2. Run RLS filtering and estimate the state sequence.
        3. Apply robust whitening channel-wise using Cholesky factors.
        4. Build a reference vector at ``ref_index``.
        5. Compute distance and LCR relative to the reference.
        6. Apply adaptive peak detection to the LCR.
        7. Convert peaks into a binary detection channel.

    Args:
        signal: Input signal, either one-dimensional or pseudo-2D.
        params: Parameter dictionary with ``polydegree``, ``l_side``, ``g``,
            and ``ref_index``.

    Returns:
        Dictionary containing the extracted features and intermediate results.
    """
    if signal.ndim == 1:
        signal = np.atleast_2d(signal).T

    polydegree = params["polydegree"]
    g = params["g"]
    l_side = params["l_side"]
    ref_index = params["ref_index"]

    model = lm.AlssmPoly(poly_degree=polydegree, label="alssm")
    segment = lm.Segment(a=-l_side, b=l_side, direction=lm.FORWARD, g=g)
    costs1d = lm.CompositeCost((model,), (segment,), [[1]])

    is_pseudo2d = signal.ndim == 2 and signal.shape[1] > 1
    steady = not is_pseudo2d
    rls = lm.create_rls(costs1d, multi_channel_set=True, steady_state=steady)

    rls.filter(signal)
    xhat = rls.minimize_x()

    if xhat.ndim == 2:
        xhat = xhat[:, :, None]

    n_samples, state_dim, n_channels = xhat.shape
    W = rls.W

    def apply_whitening_per_channel(xhat_arr: np.ndarray, W_mat: np.ndarray) -> np.ndarray:
        """Apply channel-wise whitening to the state sequence."""
        if W_mat.ndim == 2 and W_mat.shape == (state_dim, state_dim):
            V = np.linalg.cholesky(W_mat).T
            zs = np.empty_like(xhat_arr)
            for channel in range(xhat_arr.shape[2]):
                zs[:, :, channel] = (V @ xhat_arr[:, :, channel].T).T
            return zs

        if (
            W_mat.ndim == 3
            and W_mat.shape[:2] == (state_dim, state_dim)
            and W_mat.shape[2] == n_channels
        ):
            zs = np.empty_like(xhat_arr)
            for channel in range(n_channels):
                Vc = np.linalg.cholesky(W_mat[:, :, channel]).T
                zs[:, :, channel] = (Vc @ xhat_arr[:, :, channel].T).T
            return zs

        if (
            W_mat.ndim == 3
            and W_mat.shape[0] == n_channels
            and W_mat.shape[1:] == (state_dim, state_dim)
        ):
            zs = np.empty_like(xhat_arr)
            for channel in range(n_channels):
                Vc = np.linalg.cholesky(W_mat[channel, :, :]).T
                zs[:, :, channel] = (Vc @ xhat_arr[:, :, channel].T).T
            return zs

        return xhat_arr

    zs = apply_whitening_per_channel(xhat, W)
    zs_flat = zs.reshape(n_samples, state_dim * n_channels)

    z_ref = zs_flat[ref_index]
    distance = np.linalg.norm(zs_flat - z_ref, axis=1) ** 2
    energy = np.linalg.norm(zs_flat, axis=1) ** 2
    E_ref = float(energy[ref_index])

    eps = 1e-12
    cr = np.maximum(distance / max(E_ref, eps), eps)
    lcr = -0.5 * np.log(cr)

    peaks, threshold_lcr = adaptive_peak_detection_med_average(
        lcr,
        t=None,
        window_global=200,
        window_local=50,
        threshold_factor=2,
        threshold_offset=1.1,
    )

    binary_lcr = np.zeros_like(lcr, dtype=float)
    if len(peaks):
        binary_lcr[peaks] = 1.0

    fs_local = globals().get("DEFAULT_FS", 2400)
    max_duration_s = 1.2
    max_duration_samples = int(max_duration_s * fs_local)

    if len(peaks) > 1:
        valid = [peaks[0]]
        for i in range(1, len(peaks)):
            if (peaks[i] - peaks[i - 1]) <= max_duration_samples:
                valid.append(peaks[i])

        peaks = np.array(valid, dtype=int)
        binary_lcr[:] = 0.0
        binary_lcr[peaks] = 1.0

    return {
        "costs1d": costs1d,
        "lcr": lcr,
        "binary_lcr": binary_lcr,
        "threshold_lcr": threshold_lcr,
        "xhat": xhat,
        "distance": distance,
        "energy": energy,
        "peaks": peaks,
        "E_ref": E_ref,
    }


def plot_features(
    node_signal: np.ndarray,
    features: Dict[str, Any],
    params: Dict[str, Any],
    context_signal_1d: np.ndarray,
    fs: float = DEFAULT_FS,
    centers: List[Any] | None = None,
    window_title: str | None = None,
) -> None:
    """Plot diagnostic feature panels for one pipeline node.

    Args:
        node_signal: Node signal, either one-dimensional or pseudo-2D.
        features: Feature dictionary returned by ``extract_alssm_features``.
        params: Parameter dictionary of the current node.
        context_signal_1d: Baseline-corrected ECG signal for context plotting.
        fs: Sampling rate in Hz.
        centers: Optional label positions.
        window_title: Optional window title.
    """
    if node_signal.ndim == 1:
        node_signal = np.atleast_2d(node_signal).T

    n_samples = node_signal.shape[0]
    t = np.arange(n_samples) / fs

    l_side = params.get("l_side", 50)
    ref_index = params.get("ref_index", 0)
    lcr = features.get("lcr", np.zeros(n_samples))
    binary_lcr = features.get("binary_lcr", np.zeros(n_samples))
    costs1d = features.get("costs1d", None)
    xhat = features.get("xhat", None)
    distance = features.get("distance", np.zeros(n_samples))
    threshold_lcr = features.get("threshold_lcr", 0.5)
    peaks = features.get("peaks", np.array([], dtype=int))
    E_ref = features.get("E_ref", 0.0)

    if costs1d is not None and xhat is not None:
        mappedtraj = lm.map_trajectories(
            costs1d.trajectories(xhat[[ref_index]]),
            [ref_index],
            n_samples,
            merge_ks=True,
            merge_seg=True,
        )
    else:
        mappedtraj = np.zeros_like(node_signal)

    n_channels = node_signal.shape[1]
    offsets = np.arange(n_channels) * (n_channels - 1)

    fig = plt.figure(constrained_layout=True, figsize=(11, 6), dpi=120)
    if window_title:
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

    fig.suptitle(
        "ALSSM pseudo-2D feature extraction",
        fontsize=12,
        fontweight="bold",
    )
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[1, 3, 2, 2, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, context_signal_1d, color="k", lw=1.0)
    ax0.grid(True, which="major", linestyle=":", linewidth=0.6)
    ax0.minorticks_on()

    if centers:
        y_max = np.max(context_signal_1d)
        for center in sorted(centers, key=lambda x: x["center"]):
            ax0.text(
                center["center"] / fs,
                y_max * 1.2,
                center["letter"],
                ha="center",
                va="bottom",
            )

    for peak in peaks:
        left_idx = max(0, peak - l_side)
        right_idx = min(n_samples - 1, peak + l_side)
        ax0.axvspan(left_idx / fs, right_idx / fs, color="orange", alpha=0.25)

    ax0.set_ylabel("Amplitude [a.u.]")

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    for i in range(n_channels):
        ax1.plot(t, node_signal[:, i] + offsets[i], c="k", lw=0.9)
        ax1.plot(t, mappedtraj[:, i] + offsets[i], c="r", lw=1.2, alpha=0.9)

    for peak in peaks:
        ax1.axvline(peak / fs, ls="--", lw=0.6, c="k")

    proxy_sig = Line2D([0], [0], color="k", lw=1.0)
    proxy_traj = Line2D([0], [0], color="r", lw=1.2)
    channel_labels = [f"CH{j}" for j in range(n_channels - 1, -1, -1)]

    ax1.legend(
        [proxy_sig, proxy_traj],
        ["\n".join(channel_labels), "traj"],
        loc="upper right",
        fontsize=8,
    )
    ax1.set_ylabel("Channels (offset)")

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.plot(t, distance, label="distance", lw=1.0)
    if E_ref != 0.0:
        ax2.axhline(E_ref, ls="-.", lw=1.0, color="C2", label=r"$E_{ref}$")
        ax2.scatter(ref_index / fs, E_ref, marker="o", s=25, color="C2")
    for peak in peaks:
        ax2.axvline(peak / fs, ls="--", lw=0.6, c="k")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Distance [a.u.]")

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax3.plot(t, lcr, c="k", lw=1.0, label="LCR/Score")
    if np.isscalar(threshold_lcr):
        ax3.axhline(threshold_lcr, ls="--", lw=0.6, c="k", label="threshold")
    else:
        ax3.plot(t, threshold_lcr, ls="--", lw=0.6, c="k", label="adaptive threshold")
        if len(peaks):
            ax3.scatter([peak / fs for peak in peaks], lcr[peaks], marker="v", s=12, c="r", label="peaks")

    for peak in peaks:
        ax3.axvline(peak / fs, ls="--", lw=0.6, c="k")

    ax3.legend(loc="upper right")
    ax3.set_ylabel("LCR [a.u.]")

    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
    ax4.step(t, binary_lcr, where="mid", color="k", lw=1.0, label="Binary")
    for peak in peaks:
        ax4.axvline(peak / fs, ls="--", lw=0.6, c="r")

    ax4.set_ylim(-0.1, 1.1)
    ax4.set_yticks([0, 1])
    ax4.grid(True, which="major", linestyle=":", linewidth=0.6)
    ax4.minorticks_on()
    ax4.legend(loc="upper right")
    ax4.set_xlabel("Time [s]")

    plt.show()


def process_combo_plan(
    signal_1d: np.ndarray,
    combo_plan: List[Tuple[str, List[str]]],
    combo_params: Dict[str, Any],
    collect_only: bool = False,
    precomputed: Dict[str, Dict[str, Any]] | None = None,
) -> Any:
    """Run the processing graph defined by ``combo_plan``.

    Args:
        signal_1d: One-dimensional baseline-corrected ECG signal.
        combo_plan: List of target-parent relations.
        combo_params: Parameter dictionary for combination nodes.
        collect_only: If True, return full intermediate results.
        precomputed: Optional externally precomputed parent channels, such as
            NN-based P/QRS/T channels.

    Returns:
        Either:
            - full intermediate results and edges if ``collect_only=True``
            - dictionary of binary outputs otherwise
    """
    if precomputed is None:
        precomputed = {}

    results: Dict[str, Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]] = {}
    edges: List[Tuple[str, str]] = []

    for name, item in precomputed.items():
        feats = {
            "binary_lcr": np.asarray(item["binary_lcr"]).astype(float),
            "lcr": np.asarray(item.get("score", item["binary_lcr"])).astype(float),
            "distance": np.asarray(item.get("score", item["binary_lcr"])).astype(float),
            "threshold_lcr": item.get("threshold", 0.5),
            "peaks": np.flatnonzero(item["binary_lcr"]),
            "E_ref": 0.0,
            "costs1d": None,
            "xhat": None,
        }
        params_stub = combo_params.get(name, {"l_side": 50, "ref_index": 0})
        sig_stub = item.get("signal", signal_1d.reshape(-1, 1))
        results[name] = (sig_stub, feats, params_stub)

    for key, parts in combo_plan:
        for part in parts:
            if part in results:
                continue

            if len(part) == 1:
                raw = signal_1d.reshape(-1, 1)
                params_part = combo_params.get(part, all_models.get(part))
                if params_part is None:
                    raise KeyError(
                        f"Parameters for '{part}' not found in combo_params or all_models."
                    )

                feats_part = extract_alssm_features(raw, params_part)
                results[part] = (raw, feats_part, params_part)
            elif part in precomputed:
                continue
            else:
                raise KeyError(f"Unknown parent node '{part}'.")

        merge_inputs = []
        for part in parts:
            if part in results:
                _, feats_part, _ = results[part]
                weight = combo_params.get(part, {}).get(
                    "weight",
                    precomputed.get(part, {}).get("weight", 1.0),
                )
                merge_inputs.append(
                    {
                        "binary_lcr": feats_part["binary_lcr"],
                        "weight": weight,
                    }
                )
                edges.append((part, key))

        cr_matrix = merge_cr_with_noise(merge_inputs)

        params_key = combo_params.get(key)
        if params_key is None:
            raise KeyError(f"Parameters for combination node '{key}' not found.")

        feats_key = extract_alssm_features(cr_matrix, params_key)
        results[key] = (cr_matrix, feats_key, params_key)

    if collect_only:
        return results, edges

    return {k: feats["binary_lcr"] for k, (_, feats, _) in results.items()}


def run_interactive_pipeline(
    signal_raw: np.ndarray,
    signal_1d: np.ndarray,
    combo_plan: List[Tuple[str, List[str]]],
    combo_params: Dict[str, Any],
    fs: float = DEFAULT_FS,
    centers: List[Any] | None = None,
    precomputed: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    """Run the full pipeline and open interactive diagnostic figures.

    Args:
        signal_raw: Original ECG signal.
        signal_1d: Baseline-corrected ECG signal.
        combo_plan: List of target-parent relations.
        combo_params: Parameter dictionary for combination nodes.
        fs: Sampling rate.
        centers: Optional annotation centers.
        precomputed: Optional externally precomputed channels.
    """
    results, edges = process_combo_plan(
        signal_1d=signal_1d,
        combo_plan=combo_plan,
        combo_params=combo_params,
        collect_only=True,
        precomputed=precomputed,
    )

    last_key = list(results.keys())[-1]
    sig, feats, params = results[last_key]
    plot_features(
        node_signal=sig,
        features=feats,
        params=params,
        context_signal_1d=signal_raw,
        fs=fs,
        centers=centers,
        window_title=f"Final: {last_key}",
    )
    plt.show(block=False)

    G = nx.DiGraph()
    for node in results:
        G.add_node(node)
    for parent, child in edges:
        G.add_edge(parent, child)

    layer = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        layer[node] = 0 if not preds else max(layer[p] for p in preds) + 1

    nx.set_node_attributes(G, layer, "layer")
    pos = nx.multipartite_layout(G, subset_key="layer")

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, ax=ax)

    import textwrap

    labels = {n: textwrap.fill(n, width=15) for n in G.nodes()}
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="square,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
            ),
        )

    _, y_max = ax.get_ylim()
    levels = sorted(set(layer.values()))
    for lvl in levels[:-1]:
        xs = [pos[n][0] for n, value in layer.items() if value == lvl]
        xs_next = [pos[n][0] for n, value in layer.items() if value == lvl + 1]
        if xs and xs_next:
            boundary = (max(xs) + min(xs_next)) / 2
            ax.axvline(boundary, linestyle="--", color="gray", linewidth=1)
            ax.text(
                boundary,
                y_max,
                f"Layer {lvl}",
                fontsize=12,
                color="black",
                ha="center",
                va="bottom",
            )

    ax.axis("off")

    def on_click(event: Any) -> None:
        """Handle clicks on the DAG and open node diagnostics."""
        if event.inaxes != ax:
            return

        x0, y0 = event.xdata, event.ydata
        closest = None
        dist = float("inf")

        for node, (x, y) in pos.items():
            d = np.hypot(x0 - x, y0 - y)
            if d < dist:
                closest = node
                dist = d

        if closest is not None and dist < 0.1:
            sig_k, feats_k, params_k = results[closest]
            plot_features(
                node_signal=sig_k,
                features=feats_k,
                params=params_k,
                context_signal_1d=signal_raw,
                fs=fs,
                centers=centers,
                window_title=f"Step: {closest}",
            )

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def main() -> None:
    """Run the interactive ALSSM pipeline on the configured ECG CSV."""
    file_path = robust_csv_path(CSV_FILENAME)
    df = pd.read_csv(file_path, header=None)
    signal_raw = df.values.flatten().astype(np.float32)

    _, y_flat = filter_baseline(signal_raw, g_bl=50, poly_degree=2)
    signal_1d = y_flat.astype(np.float32)

    precomputed_inputs: Dict[str, Dict[str, Any]] = {}

    if USE_PQT_NN and PQT_NN_AVAILABLE:
        try:
            pqtnet_path = resolve_pqtnet_path(PQTNET_FILENAME)
            probs = nn_pqt_prob(signal_1d, str(pqtnet_path))
            P_bin, QRS_bin, T_bin = probs_to_binary(probs, 0.5, 0.5, 0.5)

            precomputed_inputs["P_NN"] = {
                "signal": signal_1d.reshape(-1, 1),
                "binary_lcr": P_bin.astype(np.float32),
                "score": probs[1].astype(np.float32),
                "weight": combo_params_2d["P_NN"]["weight"],
                "threshold": 0.5,
            }
            precomputed_inputs["QRS_NN"] = {
                "signal": signal_1d.reshape(-1, 1),
                "binary_lcr": QRS_bin.astype(np.float32),
                "score": probs[2].astype(np.float32),
                "weight": combo_params_2d["QRS_NN"]["weight"],
                "threshold": 0.5,
            }
            precomputed_inputs["T_NN"] = {
                "signal": signal_1d.reshape(-1, 1),
                "binary_lcr": T_bin.astype(np.float32),
                "score": probs[3].astype(np.float32),
                "weight": combo_params_2d["T_NN"]["weight"],
                "threshold": 0.5,
            }

            print("P/QRS/T neural-network channels are active.")
        except Exception as exc:
            print(f"Neural-network channels disabled: {exc}")

    combo_plan: List[Tuple[str, List[str]]] = []

    if "QRS_NN" in precomputed_inputs:
        combo_plan.append(("QRS", ["Q", "R", "S", "QRS_NN"]))
    else:
        combo_plan.append(("QRS", ["Q", "R", "S"]))

    # Only add a remapping node if an NN channel exists.
    # Otherwise P and T will be created automatically as leaf nodes when needed.
    if "P_NN" in precomputed_inputs:
        combo_plan.append(("P", ["P_NN"]))

    if "T_NN" in precomputed_inputs:
        combo_plan.append(("T", ["T_NN"]))

    combo_plan.append(("PQRST", ["P", "QRS", "T"]))

    if USE_WIDE_ROUTE:
        combo_plan.extend(
            [
                ("QRS_wide", ["W"]),
                ("PQRST_wide", ["P", "QRS_wide", "T"]),
            ]
        )

    run_interactive_pipeline(
        signal_raw=signal_raw,
        signal_1d=signal_1d,
        combo_plan=combo_plan,
        combo_params=combo_params_2d,
        fs=DEFAULT_FS,
        precomputed=precomputed_inputs,
    )


if __name__ == "__main__":
    main()