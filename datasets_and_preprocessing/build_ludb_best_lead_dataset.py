"""Build a LUDB best-lead P/QRS/T window dataset.

This script scans the Lobachevsky University Electrocardiography Database
(LUDB), chooses the best annotated lead for each record, and builds fixed-size
windows with dense sample-wise labels for:

    0. background
    1. P wave
    2. QRS complex
    3. T wave

Only windows containing at least one QRS segment are kept. The final dataset is
saved as one compressed NPZ file.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wfdb

WIN_SEC = 1.2
HOP_SEC = 0.2

LEAD_SUFFIXES = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]

CLS_BG = 0
CLS_P = 1
CLS_QRS = 2
CLS_T = 3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_DIR = PROJECT_ROOT / "npz_templates"
NPZ_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = NPZ_DIR / "ludb_bestlead_pqt_windows.npz"


def resolve_ludb_dir() -> Path:
    """Find the LUDB data folder.

    Returns:
        Path to the LUDB data directory.

    Raises:
        FileNotFoundError: If no valid LUDB folder was found.
    """
    env_path = os.environ.get("LUDB_DIR")

    candidates = [
        Path(env_path) if env_path else None,
        SCRIPT_DIR / "lobachevsky-university-electrocardiography-database-1.0.1" / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
        SCRIPT_DIR / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
        PROJECT_ROOT / "lobachevsky-university-electrocardiography-database-1.0.1" / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
        PROJECT_ROOT / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
        PROJECT_ROOT / "datasets" / "lobachevsky-university-electrocardiography-database-1.0.1" / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
        PROJECT_ROOT / "datasets" / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
        PROJECT_ROOT / "datasets_and_preprocessing" / "lobachevsky-university-electrocardiography-database-1.0.1" / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
        PROJECT_ROOT / "datasets_and_preprocessing" / "lobachevsky-university-electrocardiography-database-1.0.1" / "data",
    ]

    valid_candidates = [c for c in candidates if c is not None]

    for candidate in valid_candidates:
        if candidate.exists() and any(candidate.glob("*.hea")):
            return candidate

    searched = "\n".join(str(c) for c in valid_candidates)
    raise FileNotFoundError(
        "Could not find the LUDB data folder.\n"
        "Set LUDB_DIR or place the data in one of these locations:\n"
        f"{searched}"
    )


BASE = resolve_ludb_dir()


def parse_ludb_ann(path: Path) -> Optional[Dict[str, np.ndarray | int]]:
    """Parse one LUDB lead annotation file.

    Expected formats:
        - 9 columns: Pon, Pk, Poff, Qon, Rk, Qoff, Ton, Tk, Toff
        - 6 columns: Pon, Poff, Qon, Qoff, Ton, Toff

    In the 6-column case, missing peaks are filled with -1.

    Args:
        path: Annotation file path.

    Returns:
        Parsed annotation dictionary, or None if no usable rows were found.
    """
    rows: List[List[int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            s = line.strip().lower()
            if not s or s.startswith("#"):
                continue

            nums = re.findall(r"-?\d+", s)
            if not nums:
                continue

            vals = [int(x) for x in nums]

            if len(vals) >= 9:
                vals = vals[:9]
            elif len(vals) == 6:
                pon, poff, qon, qoff, ton, toff = vals
                vals = [pon, -1, poff, qon, -1, qoff, ton, -1, toff]
            else:
                continue

            rows.append(vals)

    if not rows:
        return None

    arr = np.vstack(rows).astype(int)
    pon, pk, poff, qon, rk, qoff, ton, tk, toff = [arr[:, i] for i in range(9)]

    p_cnt = int(np.sum(poff > pon))
    qrs_cnt = int(np.sum(qoff > qon))
    t_cnt = int(np.sum(toff > ton))

    return {
        "Pon": pon,
        "Poff": poff,
        "Qon": qon,
        "Qoff": qoff,
        "Ton": ton,
        "Toff": toff,
        "Pk": pk,
        "Rk": rk,
        "Tk": tk,
        "p_cnt": p_cnt,
        "qrs_cnt": qrs_cnt,
        "t_cnt": t_cnt,
    }


def choose_best_lead(base: Path, record_id: str) -> Optional[Tuple[Tuple[int, int, int], str, Dict[str, np.ndarray | int]]]:
    """Choose the best LUDB lead based on available annotation counts.

    The score is ordered as:
        (number of QRS segments, number of P segments, number of T segments)

    Args:
        base: LUDB data directory.
        record_id: Record identifier.

    Returns:
        Tuple containing:
            - score tuple
            - chosen lead suffix
            - parsed annotation dictionary

        Returns None if no usable lead was found.
    """
    best = None

    for suffix in LEAD_SUFFIXES:
        ann_path = base / f"{record_id}.{suffix}"
        if not ann_path.exists():
            continue

        info = parse_ludb_ann(ann_path)
        if info is None:
            continue

        score = (int(info["qrs_cnt"]), int(info["p_cnt"]), int(info["t_cnt"]))

        if best is None or score > best[0]:
            best = (score, suffix, info)

    return best


def put_span(y: np.ndarray, a: int, b: int, cls_id: int) -> None:
    """Write one class label into a sample range.

    Args:
        y: Label array for one window.
        a: Start sample index.
        b: End sample index.
        cls_id: Class identifier.
    """
    a = int(max(0, a))
    b = int(min(len(y), b))

    if b > a:
        y[a:b] = cls_id


def build_windows_for_record(
    sig_1d: np.ndarray,
    ann: Dict[str, np.ndarray | int],
    fs: int,
    win_sec: float,
    hop_sec: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build labeled windows for one LUDB record.

    Args:
        sig_1d: One-dimensional ECG signal.
        ann: Parsed annotation dictionary.
        fs: Sampling rate.
        win_sec: Window length in seconds.
        hop_sec: Hop size in seconds.

    Returns:
        Tuple containing:
            - X: window tensor of shape (n_windows, 1, window_length)
            - Y: dense class labels of shape (n_windows, window_length)
            - centers: window-center sample indices

        Returns None if no usable windows were found.
    """
    n_samples = len(sig_1d)
    window_len = int(round(win_sec * fs))
    hop_len = int(round(hop_sec * fs))

    Xs = []
    Ys = []
    centers = []

    pon = np.asarray(ann["Pon"], dtype=int)
    poff = np.asarray(ann["Poff"], dtype=int)
    qon = np.asarray(ann["Qon"], dtype=int)
    qoff = np.asarray(ann["Qoff"], dtype=int)
    ton = np.asarray(ann["Ton"], dtype=int)
    toff = np.asarray(ann["Toff"], dtype=int)

    for start in range(0, n_samples - window_len + 1, hop_len):
        end = start + window_len
        x = sig_1d[start:end].astype(np.float32).copy()

        y = np.zeros(window_len, dtype=np.int64)

        for a, b in zip(pon, poff):
            put_span(y, a - start, b - start, CLS_P)

        for a, b in zip(qon, qoff):
            put_span(y, a - start, b - start, CLS_QRS)

        for a, b in zip(ton, toff):
            put_span(y, a - start, b - start, CLS_T)

        if not (y == CLS_QRS).any():
            continue

        mu = float(np.median(x))
        sd = float(np.std(x)) + 1e-6
        x = (x - mu) / sd

        Xs.append(x[None, :])
        Ys.append(y)
        centers.append(start + window_len // 2)

    if not Xs:
        return None

    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.int64)
    centers_arr = np.asarray(centers, dtype=np.int64)

    return X, Y, centers_arr


def main() -> None:
    """Build the full LUDB best-lead dataset and save it as NPZ."""
    record_ids = sorted(
        set(
            os.path.splitext(os.path.basename(path))[0]
            for path in glob.glob(str(BASE / "*.hea"))
        )
    )

    if not record_ids:
        raise RuntimeError("No .hea files found. Check LUDB_DIR.")

    X_all = []
    Y_all = []
    meta = []

    used = 0
    last_fs = None

    for record_id in record_ids:
        best = choose_best_lead(BASE, record_id)
        if best is None:
            continue

        _score, suffix, ann = best

        rec = wfdb.rdrecord(str(BASE / record_id))
        fs = int(rec.fs)
        last_fs = fs

        lead_idx = LEAD_SUFFIXES.index(suffix)
        sig_1d = rec.p_signal[:, lead_idx].astype(np.float32)

        out = build_windows_for_record(sig_1d, ann, fs, WIN_SEC, HOP_SEC)
        if out is None:
            continue

        X, Y, centers = out

        for i in range(X.shape[0]):
            meta.append((record_id, suffix, int(centers[i])))

        X_all.append(X)
        Y_all.append(Y)
        used += 1

    if not X_all or last_fs is None:
        raise RuntimeError("No windows were built. Check LUDB annotations and paths.")

    X_all_arr = np.concatenate(X_all, axis=0)
    Y_all_arr = np.concatenate(Y_all, axis=0)
    meta_arr = np.asarray(meta, dtype=object)

    np.savez_compressed(
        OUT_NPZ,
        X=X_all_arr,
        Y=Y_all_arr,
        fs=np.array([last_fs], dtype=np.int64),
        win_sec=np.array([WIN_SEC], dtype=np.float32),
        hop_sec=np.array([HOP_SEC], dtype=np.float32),
        meta=meta_arr,
    )

    print(f"[OK] saved: {OUT_NPZ}")
    print(f"Records used: {used} / {len(record_ids)}")
    print("X shape:", X_all_arr.shape, "Y shape:", Y_all_arr.shape)
    print("meta example:", meta_arr[0])


if __name__ == "__main__":
    main()