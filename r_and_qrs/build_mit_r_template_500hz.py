"""Build the MIT-derived R point template at 500 Hz.

This script builds the main MIT-derived R template from one selected MIT-BIH
record. The record is loaded from a local MIT-BIH database folder, resampled to
500 Hz if needed, converted into ALSSM states, and summarized into one robust
R template.

In this version, the R template is intentionally built without whitening,
matching the original thesis workflow for the MIT-based R template.

The final template is saved to the thesis template folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import lmlib as lm
import numpy as np
import wfdb
from scipy.signal import resample

FS_TARGET = 500
RECORD = "100"

R_PARAMS = {
    "polydegree": 2,
    "l_side": 15,
    "g": 40,
}

VALID_BEAT_SYMBOLS = {"N", "L", "R", "A", "V", "F", "e", "j"}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
NPZ_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_FILE = NPZ_DIR / "r_template_mitbih_500hz.npy"


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


def load_mit_record_500hz(record_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one MIT-BIH record and resample it to 500 Hz if needed.

    Args:
        record_name: MIT-BIH record name.

    Returns:
        Tuple containing:
            - signal at 500 Hz
            - valid beat indices at 500 Hz
    """
    record_path = MIT_PATH / record_name

    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, 0].astype(np.float64)
    fs_orig = int(record.fs)

    ann = wfdb.rdann(str(record_path), "atr")

    valid_beats: List[int] = []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in VALID_BEAT_SYMBOLS:
            valid_beats.append(int(sample))

    beat_indices = np.array(valid_beats, dtype=int)

    print("Original sampling rate:", fs_orig)
    print("Beat count:", len(beat_indices))

    if fs_orig != FS_TARGET:
        new_len = int(len(signal) * FS_TARGET / fs_orig)
        signal = resample(signal, new_len).astype(np.float64)
        beat_indices = (beat_indices * FS_TARGET / fs_orig).astype(int)

    print("Signal length (s):", len(signal) / FS_TARGET)

    return signal, beat_indices


def run_alssm(signal: np.ndarray) -> np.ndarray:
    """Estimate ALSSM states for the ECG signal.

    Args:
        signal: ECG signal.

    Returns:
        State sequence.
    """
    model = lm.AlssmPoly(poly_degree=R_PARAMS["polydegree"])
    segment = lm.Segment(
        a=-R_PARAMS["l_side"],
        b=R_PARAMS["l_side"],
        direction=lm.FORWARD,
        g=R_PARAMS["g"],
    )
    cost = lm.CompositeCost((model,), (segment,), [[1]])

    rls = lm.create_rls(cost, multi_channel_set=False, steady_state=True)
    rls.filter(signal)
    xhat = rls.minimize_x()

    return xhat


def build_r_template(states: np.ndarray, beat_indices: np.ndarray, window: int = 2) -> np.ndarray:
    """Build the R template from local state averages around valid beats.

    The original workflow does not apply whitening here. Instead, a small local
    mean of the raw states around each beat is collected, and the final
    template is taken as the componentwise median.

    Args:
        states: Raw ALSSM state sequence.
        beat_indices: Valid beat indices.
        window: Half-width of the local averaging window in samples.

    Returns:
        Final R template vector.

    Raises:
        RuntimeError: If no valid states were collected.
    """
    valid_states = []

    for beat_idx in beat_indices:
        if beat_idx >= len(states):
            continue

        start = max(0, beat_idx - window)
        end = min(len(states), beat_idx + window + 1)
        local_state = np.mean(states[start:end], axis=0)
        valid_states.append(local_state)

    if len(valid_states) == 0:
        raise RuntimeError("No valid R states were collected for template building.")

    state_matrix = np.array(valid_states, dtype=np.float64)
    template = np.median(state_matrix, axis=0)

    return template


def main() -> None:
    """Build and save the MIT-derived R template."""
    signal, beat_indices = load_mit_record_500hz(RECORD)
    states = run_alssm(signal)

    # Intentionally no whitening here, matching the original thesis workflow.
    r_template = build_r_template(states, beat_indices, window=2)

    print("Template dimension:", r_template.shape)
    print("Template norm:", np.linalg.norm(r_template))

    np.save(TEMPLATE_FILE, r_template)

    print("Template saved as:", TEMPLATE_FILE)


if __name__ == "__main__":
    main()