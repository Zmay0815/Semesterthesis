"""Adaptive peak-thresholding utilities for LCR-based ECG detection.

This module collects several adaptive thresholding strategies that were used
during development and comparison of ECG peak detectors. The functions operate
on an LCR-like detection signal and optionally use the original ECG signal to
estimate a local noise level.

Implemented methods:
    1. adaptive_threshold_peaks_SNR:
       ALSSM long-threshold baseline with SNR-based additive alpha term.

    2. adaptive_threshold_peaks_lmlib:
       ALSSM short and long smoothing with multiplicative noise adaptation.

    3. adaptive_threshold_peaks_lmlib2:
       Mean and median smoothing with multiplicative noise adaptation.

    4. adaptive_threshold_percentile:
       Local rolling quantile threshold on the LCR signal.

    5. adaptive_threshold_peaks:
       ALSSM long smoothing plus rolling quantile on normalized LCR.

    6. adaptive_peak_detection_med_average:
       Median and mean based adaptive thresholding directly on a signal.

The module is filesystem-independent and can be reused on any machine.
"""

from __future__ import annotations

from typing import Tuple

import lmlib as lm
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks


def rolling_std_cumsum(x: np.ndarray, window: int) -> np.ndarray:
    """Compute a rolling standard deviation in O(N) using cumulative sums.

    Edge values are filled by repeating the first and last valid result.

    Args:
        x: Input signal as a one-dimensional NumPy array.
        window: Rolling window length. Must be at least 1.

    Returns:
        Rolling standard deviation with the same length as ``x``.

    Raises:
        ValueError: If ``window < 1``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    x = np.asarray(x, dtype=float)
    n_samples = x.size

    c1 = np.empty(n_samples + 1, dtype=float)
    c1[0] = 0.0
    np.cumsum(x, out=c1[1:])

    c2 = np.empty(n_samples + 1, dtype=float)
    c2[0] = 0.0
    np.cumsum(x * x, out=c2[1:])

    sum1 = c1[window:] - c1[:-window]
    sum2 = c2[window:] - c2[:-window]

    var = sum2 / window - (sum1 / window) ** 2
    std = np.sqrt(np.clip(var, 0.0, None))

    half = window // 2
    left = std[0].repeat(half)
    right = std[-1].repeat(half + (window % 2 == 0))

    return np.concatenate([left, std, right])[:n_samples]


def adaptive_threshold_peaks_SNR(
    lcr_signal: np.ndarray,
    original_signal: np.ndarray,
    short_window: int = 10,
    long_window: int = 100,
    alpha_min: float = 0.5,
    alpha_max: float = 2.0,
    noise_estimation_window: int = 200,
    min_distance: int = 20,
    short_g: float = 20,
    long_g: float = 60,
    poly_degree: int = 0,
) -> Tuple[list[int], np.ndarray, np.ndarray]:
    """Detect peaks using an ALSSM baseline plus SNR-based additive alpha.

    The threshold is computed as:

        threshold(t) = y_hat_long(t) + alpha(t)

    where ``alpha(t)`` is derived from a local SNR estimate based on the
    original signal.

    Args:
        lcr_signal: Detection signal, typically an LCR-like score.
        original_signal: Original signal used for local noise estimation.
        short_window: Short ALSSM support width.
        long_window: Long ALSSM support width.
        alpha_min: Minimum alpha value.
        alpha_max: Maximum alpha value.
        noise_estimation_window: Window used for local noise estimation.
        min_distance: Minimum distance between detected peaks.
        short_g: Segment weighting for the short ALSSM model.
        long_g: Segment weighting for the long ALSSM model.
        poly_degree: Polynomial degree of the ALSSM model.

    Returns:
        Tuple containing:
            - list of adaptive peak indices
            - adaptive threshold series
            - alpha series

    Raises:
        ValueError: If the signal is too short for the chosen windows.
    """
    n_samples = len(lcr_signal)
    if n_samples < max(short_window, long_window):
        raise ValueError("Signal too short for the chosen window sizes.")

    start0 = 100
    w0 = min(noise_estimation_window, len(original_signal) - start0)
    if w0 <= 0:
        raise ValueError(
            "Original signal is too short for noise_estimation_window starting at index 100."
        )

    sigma_clean = np.std(original_signal[start0 : start0 + w0])

    sigma_local = (
        pd.Series(original_signal)
        .rolling(
            noise_estimation_window,
            center=True,
            min_periods=noise_estimation_window,
        )
        .std()
        .bfill()
        .ffill()
        .to_numpy()
    )

    eps = 1e-12
    snr_series = sigma_clean / (sigma_local + eps)
    snr_min = snr_series.min()
    snr_range = np.ptp(snr_series) + eps
    snr_norm = (snr_series - snr_min) / snr_range

    alpha = alpha_min + (alpha_max - alpha_min) * snr_norm
    alpha = np.clip(alpha, alpha_min, alpha_max)

    # Keep the initial region conservative.
    alpha[:start0] = alpha_max

    alssm_poly = lm.AlssmPoly(poly_degree=poly_degree)

    seg_short = (
        lm.Segment(a=-(short_window - 1), b=-1, direction=lm.FORWARD, g=short_g),
        lm.Segment(a=0, b=short_window - 1, direction=lm.BACKWARD, g=short_g),
    )
    seg_long = (
        lm.Segment(a=-(long_window - 1), b=-1, direction=lm.FORWARD, g=long_g),
        lm.Segment(a=0, b=long_window - 1, direction=lm.BACKWARD, g=long_g),
    )

    costs_short = lm.CompositeCost((alssm_poly,), seg_short, F=[[1, 1]])
    costs_long = lm.CompositeCost((alssm_poly,), seg_long, F=[[1, 1]])

    rls_short = lm.create_rls(costs_short, steady_state=True)
    _ = rls_short.filter_minimize_x(lcr_signal)

    rls_long = lm.create_rls(costs_long, steady_state=True)
    xs_long = rls_long.filter_minimize_x(lcr_signal)
    y_hat_long = costs_long.eval_alssm_output(xs_long, alssm_weights=[1])

    threshold_series = y_hat_long + alpha

    peak_indices, _ = find_peaks(lcr_signal, distance=min_distance)
    adaptive_peaks = [idx for idx in peak_indices if lcr_signal[idx] > threshold_series[idx]]

    return adaptive_peaks, threshold_series, alpha


def adaptive_threshold_peaks_lmlib(
    lcr_signal: np.ndarray,
    original_signal: np.ndarray,
    short_window: int = 10,
    long_window: int = 100,
    short_g: float = 20,
    long_g: float = 60,
    noise_lp_window: int = 200,
    noise_roll_window: int = 200,
    alpha_min: float = 0.5,
    alpha_max: float = 2.0,
    min_distance: int = 20,
    poly_degree: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect peaks using ALSSM smoothing and multiplicative noise adaptation.

    The threshold is computed as:

        threshold(t) = clip(y_hat_long(t) * alpha(t), alpha_min, alpha_max)

    Args:
        lcr_signal: Detection signal, typically an LCR-like score.
        original_signal: Original signal used for local noise estimation.
        short_window: Short ALSSM support width.
        long_window: Long ALSSM support width.
        short_g: Segment weighting for the short ALSSM model.
        long_g: Segment weighting for the long ALSSM model.
        noise_lp_window: Window length for the low-pass baseline estimate.
        noise_roll_window: Window length for the rolling noise estimate.
        alpha_min: Minimum alpha value.
        alpha_max: Maximum alpha value.
        min_distance: Minimum distance between detected peaks.
        poly_degree: Polynomial degree of the ALSSM model.

    Returns:
        Tuple containing:
            - detected peak indices
            - adaptive threshold series
            - alpha series

    Raises:
        ValueError: If the signal is too short for the chosen windows.
    """
    n_samples = lcr_signal.size
    need = max(short_window, long_window, noise_lp_window, noise_roll_window)
    if n_samples < need:
        raise ValueError(f"Signal too short (<{need}) for the chosen window sizes.")

    alssm = lm.AlssmPoly(poly_degree=poly_degree)

    seg_short = (
        lm.Segment(a=-(short_window - 1), b=-1, direction=lm.FORWARD, g=short_g),
        lm.Segment(a=0, b=short_window - 1, direction=lm.BACKWARD, g=short_g),
    )
    costs_short = lm.CompositeCost((alssm,), seg_short, F=[[1, 1]])
    rls_short = lm.create_rls(costs_short, steady_state=True)
    xs_short = rls_short.filter_minimize_x(lcr_signal)
    _ = costs_short.eval_alssm_output(xs_short, alssm_weights=[1])

    seg_long = (
        lm.Segment(a=-(long_window - 1), b=-1, direction=lm.FORWARD, g=long_g),
        lm.Segment(a=0, b=long_window - 1, direction=lm.BACKWARD, g=long_g),
    )
    costs_long = lm.CompositeCost((alssm,), seg_long, F=[[1, 1]])
    rls_long = lm.create_rls(costs_long, steady_state=True)
    xs_long = rls_long.filter_minimize_x(lcr_signal)
    y_hat_long = costs_long.eval_alssm_output(xs_long, alssm_weights=[1])

    baseline = uniform_filter1d(original_signal, noise_lp_window, mode="reflect")
    std_roll = rolling_std_cumsum(baseline, noise_roll_window)

    eps = 1e-12
    nmin = std_roll.min()
    nmax = std_roll.max()
    noise_norm = (std_roll - nmin) / (nmax - nmin + eps)

    alpha = alpha_max - (alpha_max - alpha_min) * noise_norm
    alpha = np.clip(alpha, alpha_min, alpha_max)

    threshold = np.clip(y_hat_long * alpha, alpha_min, alpha_max)

    peaks, _ = find_peaks(lcr_signal, distance=min_distance, height=threshold)
    return peaks, threshold, alpha


def adaptive_threshold_peaks_lmlib2(
    lcr_signal: np.ndarray,
    original_signal: np.ndarray,
    short_window: int = 10,
    long_window: int = 100,
    noise_lp_window: int = 200,
    noise_roll_window: int = 200,
    alpha_min: float = 0.5,
    alpha_max: float = 2.0,
    min_distance: int = 20,
    poly_degree: int = 0,
    mix_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect peaks using mean/median smoothing and adaptive scaling.

    Args:
        lcr_signal: Detection signal, typically an LCR-like score.
        original_signal: Original signal used for local noise estimation.
        short_window: Window for short-term mean smoothing.
        long_window: Window for long-term median smoothing.
        noise_lp_window: Window length for the low-pass baseline estimate.
        noise_roll_window: Window length for the rolling noise estimate.
        alpha_min: Minimum alpha value.
        alpha_max: Maximum alpha value.
        min_distance: Minimum distance between detected peaks.
        poly_degree: Unused. Kept for interface compatibility.
        mix_ratio: Mixing factor between short and long smoothing.

    Returns:
        Tuple containing:
            - detected peak indices
            - adaptive threshold series
            - alpha series

    Raises:
        ValueError: If the signal is too short.
    """
    _ = poly_degree

    n_samples = lcr_signal.size
    need = max(short_window, long_window, noise_lp_window, noise_roll_window)
    if n_samples < need:
        raise ValueError(f"Signal too short (<{need}).")

    y_hat_short = uniform_filter1d(lcr_signal, size=short_window, mode="reflect")
    y_hat_long = median_filter(lcr_signal, size=long_window, mode="reflect")
    mix = mix_ratio * y_hat_short + (1.0 - mix_ratio) * y_hat_long

    baseline = uniform_filter1d(original_signal, noise_lp_window, mode="reflect")
    std_roll = rolling_std_cumsum(baseline, noise_roll_window)

    eps = 1e-12
    nmin = std_roll.min()
    nmax = std_roll.max()
    noise_norm = (std_roll - nmin) / (nmax - nmin + eps)

    alpha = alpha_max - (alpha_max - alpha_min) * noise_norm
    alpha = np.clip(alpha, alpha_min, alpha_max)

    threshold = np.clip(mix * alpha, alpha_min, alpha_max)

    peaks, _ = find_peaks(lcr_signal, distance=min_distance, height=threshold)
    return peaks, threshold, alpha


def adaptive_threshold_percentile(
    lcr: np.ndarray,
    window: int = 200,
    quantile: float = 0.995,
    min_distance: int = 20,
    alpha_min: float = 0.5,
    alpha_max: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks using a local rolling quantile threshold.

    Args:
        lcr: Detection signal.
        window: Rolling quantile window.
        quantile: Quantile level.
        min_distance: Minimum distance between detected peaks.
        alpha_min: Lower clip bound for the threshold.
        alpha_max: Upper clip bound for the threshold.

    Returns:
        Tuple containing:
            - detected peak indices
            - threshold series
    """
    threshold = (
        pd.Series(lcr)
        .rolling(window, center=True, min_periods=1)
        .quantile(quantile)
        .bfill()
        .ffill()
        .to_numpy()
    )

    threshold = np.clip(threshold, alpha_min, alpha_max)
    peaks, _ = find_peaks(lcr, height=threshold, distance=min_distance)
    return peaks, threshold


def adaptive_threshold_peaks(
    lcr_signal: np.ndarray,
    original_signal: np.ndarray,
    short_window: int = 10,
    long_window: int = 100,
    short_g: float = 20,
    long_g: float = 60,
    noise_lp_window: int = 200,
    quantile_window: int = 200,
    quantile_level: float = 0.995,
    alpha_min: float = 0.5,
    alpha_max: float = 2.0,
    min_distance: int = 20,
    poly_degree: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks using ALSSM long smoothing plus local quantile thresholding.

    The threshold is computed on a normalized detection signal:

        lcr_norm(t) = lcr_signal(t) / (y_hat_long(t) + eps)

    A rolling quantile is applied to ``lcr_norm`` and then mapped back to the
    original amplitude scale.

    Args:
        lcr_signal: Detection signal, typically an LCR-like score.
        original_signal: Unused. Kept for interface compatibility.
        short_window: Unused. Kept for interface compatibility.
        long_window: Long ALSSM support width.
        short_g: Unused. Kept for interface compatibility.
        long_g: Segment weighting for the long ALSSM model.
        noise_lp_window: Unused. Kept for interface compatibility.
        quantile_window: Rolling quantile window length.
        quantile_level: Rolling quantile level.
        alpha_min: Lower clip bound for the threshold.
        alpha_max: Upper clip bound for the threshold.
        min_distance: Minimum distance between detected peaks.
        poly_degree: Polynomial degree of the ALSSM model.

    Returns:
        Tuple containing:
            - detected peak indices
            - threshold series
    """
    _ = original_signal, short_window, short_g, noise_lp_window

    eps = 1e-12
    alssm = lm.AlssmPoly(poly_degree=poly_degree)

    seg_long = (
        lm.Segment(a=-(long_window - 1), b=-1, direction=lm.FORWARD, g=long_g),
        lm.Segment(a=0, b=long_window - 1, direction=lm.BACKWARD, g=long_g),
    )
    costs_long = lm.CompositeCost((alssm,), seg_long, F=[[1, 1]])
    rls_long = lm.create_rls(costs_long, steady_state=True)
    xs_long = rls_long.filter_minimize_x(lcr_signal)
    y_hat_long = costs_long.eval_alssm_output(xs_long, alssm_weights=[1])

    lcr_norm = lcr_signal / (y_hat_long + eps)

    thr_norm = (
        pd.Series(lcr_norm)
        .rolling(window=quantile_window, center=True, min_periods=1)
        .quantile(quantile_level)
        .bfill()
        .ffill()
        .to_numpy()
    )

    threshold = thr_norm * y_hat_long
    threshold = np.clip(threshold, alpha_min, alpha_max)

    peaks, _ = find_peaks(lcr_signal, height=threshold, distance=min_distance)
    return peaks, threshold


def adaptive_peak_detection_med_average(
    signal: np.ndarray,
    t: np.ndarray | None = None,
    window_global: int = 101,
    window_local: int = 21,
    threshold_factor: float = 3.0,
    threshold_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks using median detrending and local mean absolute deviation.

    Args:
        signal: Input signal.
        t: Optional time axis. If omitted, sample indices are used.
        window_global: Window for the local mean absolute deviation.
        window_local: Window for the rolling median trend.
        threshold_factor: Multiplier for the local noise estimate.
        threshold_offset: Additive offset applied to the threshold.

    Returns:
        Tuple containing:
            - detected peak indices
            - threshold series
    """
    if t is None:
        t = np.arange(len(signal))

    df = pd.DataFrame({"signal": signal}, index=t)

    trend = df["signal"].rolling(
        window=window_local,
        center=True,
        min_periods=1,
    ).median()

    detrended = df["signal"] - trend
    local_noise = detrended.abs().rolling(
        window=window_global,
        center=True,
        min_periods=1,
    ).mean()

    thresh_global = trend
    thresh_local = threshold_factor * local_noise
    threshold = np.maximum(thresh_global, thresh_local) + threshold_offset

    all_peaks, _ = find_peaks(df["signal"].values)
    mask = df["signal"].values[all_peaks] > threshold.values[all_peaks]
    peaks = all_peaks[mask]

    return peaks, threshold.values