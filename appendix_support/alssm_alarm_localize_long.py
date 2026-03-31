# -*- coding: utf-8 -*-
"""
ALSSM Alarm + Lokalisierung auf langem EKG (Sliding Window)

Voraussetzungen:
- synth_leadII_pqt_pathology_windows.npz
- ekg_u.py (umbenannt von EKG-U.py)
  mit: extract_alssm_features, merge_cr_with_noise
  optional: filter_baseline

Ablauf:
1) Kalibriert Alarm-Schwelle aus "normal" Windows (Quantil)
2) Baut ein langes Signal aus vielen Windows (normal + eingemischte Pathologien)
3) Sliding-Window Scan: ALSSM Score pro Window
4) Lokalisierung: Zeitbereiche + Hauptursache (P/QRS/T)
5) Plots: langes Signal mit Alarm-Spans + Score-Plot
"""

import numpy as np
import matplotlib.pyplot as plt

from ekg_u import extract_alssm_features, merge_cr_with_noise
# from ekg_u import filter_baseline  # optional

# -----------------------------
# Konfiguration
# -----------------------------
NPZ_PATH = "synth_leadII_pqt_pathology_windows.npz"

# Sliding-Window Scan auf langem Signal
WIN_S = 1.2
HOP_S = 0.2

# Alarm-Kalibrierung: Schwelle = Quantil der NORMAL-score_min Verteilung
# Beispiel 0.01: nur 1% der normalen Windows dürfen Alarm auslösen
NORMAL_FALSE_ALARM_RATE = 0.01

# Langes Signal: viele Windows aneinanderhängen
LONG_N_WINDOWS = 60
PATHOLOGY_INSERT_PROB = 0.10  # 10% der Windows sind Pathologie statt normal
PATHOLOGY_CHOICES = ["af_like", "long_pr", "wide_qrs", "t_inversion", "peaked_t", "st_elev", "pvc"]

# Alarm-Events zusammenfassen
MERGE_GAP_S = 0.15  # wenn zwischen Alarm-Windows weniger als das liegt, wird ein Event daraus


# -----------------------------
# ALSSM Score P/QRS/T
# -----------------------------
def _scale_params_for_fs(p, fs):
    scale = fs / 2400.0
    p = dict(p)
    p["l_side"] = max(3, int(round(p["l_side"] * scale)))
    p["ref_index"] = int(round(p["ref_index"] * scale))
    return p


def alssm_scores_pqt(x_win, fs):
    """
    Liefert (score_P, score_QRS, score_T, score_min, reason)
    score = 5%-Quantil(LCR). kleiner => untypischer => mehr Alarm.
    reason = P oder QRS oder T, je nachdem welcher score minimal ist.
    """
    x_use = x_win.astype(np.float32)

    # Parameter (von deinem Setup, auf fs skaliert)
    P_params = _scale_params_for_fs({'ref_index': 1310, 'polydegree': 2, 'l_side': 60,  'g': 40}, fs)
    Q_params = _scale_params_for_fs({'ref_index': 1370, 'polydegree': 2, 'l_side': 18,  'g': 40}, fs)
    R_params = _scale_params_for_fs({'ref_index': 1435, 'polydegree': 2, 'l_side': 25,  'g': 40}, fs)
    S_params = _scale_params_for_fs({'ref_index': 1482, 'polydegree': 2, 'l_side': 10,  'g': 40}, fs)
    T_params = _scale_params_for_fs({'ref_index': 1850, 'polydegree': 2, 'l_side': 60,  'g': 40}, fs)
    QRS_params = _scale_params_for_fs({'ref_index': 1450, 'polydegree': 3, 'l_side': 120, 'g': 300}, fs)

    raw = x_use.reshape(-1, 1)

    feats_P = extract_alssm_features(raw, P_params)
    feats_Q = extract_alssm_features(raw, Q_params)
    feats_R = extract_alssm_features(raw, R_params)
    feats_S = extract_alssm_features(raw, S_params)
    feats_T = extract_alssm_features(raw, T_params)

    cr = merge_cr_with_noise([
        {"binary_lcr": feats_Q["binary_lcr"], "weight": 1.0},
        {"binary_lcr": feats_R["binary_lcr"], "weight": 1.0},
        {"binary_lcr": feats_S["binary_lcr"], "weight": 1.0},
    ], noise_level=0.0)

    feats_QRS = extract_alssm_features(cr, QRS_params)

    def score_from_feats(feats):
        lcr = np.asarray(feats.get("lcr", []), dtype=float)
        if lcr.size == 0:
            return np.nan
        return float(np.percentile(lcr, 5))

    sP = score_from_feats(feats_P)
    sQ = score_from_feats(feats_QRS)
    sT = score_from_feats(feats_T)

    scores = {"P": sP, "QRS": sQ, "T": sT}
    reason = min(scores, key=lambda k: np.inf if np.isnan(scores[k]) else scores[k])
    sMin = float(np.nanmin([sP, sQ, sT]))

    return sP, sQ, sT, sMin, reason


# -----------------------------
# Schwelle aus normalen Windows kalibrieren
# -----------------------------
def calibrate_alarm_threshold_from_normal(X, Y_cls, pathology_names, fs, false_alarm_rate=0.01, max_samples=8000):
    if "normal" not in pathology_names.tolist():
        raise RuntimeError("Keine Klasse 'normal' in pathology_names gefunden.")

    normal_id = int(np.where(pathology_names == "normal")[0][0])
    idx = np.where(Y_cls.astype(int) == normal_id)[0]
    if idx.size == 0:
        raise RuntimeError("Keine normal Samples gefunden.")

    rng = np.random.default_rng(0)
    if idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)

    scores = []
    for i in idx:
        x = X[i, 0, :]
        _, _, _, sMin, _ = alssm_scores_pqt(x, fs)
        if np.isfinite(sMin):
            scores.append(sMin)

    scores = np.asarray(scores, dtype=float)
    if scores.size < 200:
        raise RuntimeError("Zu wenige normale Scores für stabile Kalibrierung.")

    thr = float(np.quantile(scores, false_alarm_rate))
    return thr, scores


# -----------------------------
# Langes Signal aus Windows bauen
# -----------------------------
def build_long_stream(X, Y_cls, pathology_names, n_windows=60, insert_prob=0.1, pathology_choices=None, seed=1):
    rng = np.random.default_rng(seed)
    if pathology_choices is None:
        pathology_choices = []

    by_name = {}
    for name in pathology_names.tolist():
        cid = int(np.where(pathology_names == name)[0][0])
        by_name[name] = np.where(Y_cls.astype(int) == cid)[0]

    if by_name.get("normal", np.array([])).size == 0:
        raise RuntimeError("Keine normal Samples vorhanden.")

    L = X.shape[2]
    xs = []
    meta = []
    cur = 0

    for k in range(n_windows):
        use_path = (rng.random() < insert_prob) and (len(pathology_choices) > 0)
        if use_path:
            name = str(rng.choice(pathology_choices))
            pool = by_name.get(name, np.array([]))
            if pool.size == 0:
                name = "normal"
                pool = by_name["normal"]
        else:
            name = "normal"
            pool = by_name["normal"]

        pick = int(rng.choice(pool))
        x = X[pick, 0, :].astype(np.float32)

        xs.append(x)
        meta.append({"k": k, "idx": pick, "label": name, "start": cur, "end": cur + L})
        cur += L

    x_long = np.concatenate(xs, axis=0).astype(np.float32)
    return x_long, meta


# -----------------------------
# Sliding Window Scan
# -----------------------------
def scan_long_ecg_for_alarm(x_long, fs, win_s, hop_s, alarm_thresh):
    L = int(round(win_s * fs))
    H = int(round(hop_s * fs))

    times = []
    sMin = []
    sP = []
    sQ = []
    sT = []
    alarms = []

    for start in range(0, len(x_long) - L + 1, H):
        xw = x_long[start:start + L]
        sp, sq, st, smin, reason = alssm_scores_pqt(xw, fs)

        t0 = start / fs
        t1 = (start + L) / fs
        times.append((t0, t1))
        sMin.append(smin)
        sP.append(sp)
        sQ.append(sq)
        sT.append(st)

        if np.isfinite(smin) and (smin < alarm_thresh):
            alarms.append((t0, t1, float(smin), reason, float(sp), float(sq), float(st)))

    return (np.asarray(sMin), np.asarray(sP), np.asarray(sQ), np.asarray(sT), times, alarms)


def merge_overlapping_alarms(alarms, gap_s=0.1):
    if not alarms:
        return []

    alarms = sorted(alarms, key=lambda a: a[0])
    events = []

    cur = list(alarms[0])  # [t0,t1,smin,reason,sP,sQ,sT]
    for a in alarms[1:]:
        if a[0] <= cur[1] + gap_s:
            cur[1] = max(cur[1], a[1])
            if a[2] < cur[2]:
                cur[2:] = list(a[2:])
        else:
            events.append(tuple(cur))
            cur = list(a)

    events.append(tuple(cur))
    return events


# -----------------------------
# Plots
# -----------------------------
def plot_long_signal_with_events(x_long, fs, events, title):
    t = np.arange(len(x_long)) / fs
    plt.figure(figsize=(14, 5))
    plt.plot(t, x_long, lw=1.0)
    plt.title(title)
    plt.xlabel("t [s]")

    y_top = float(np.max(x_long))
    for (t0, t1, smin, reason, *_rest) in events:
        plt.axvspan(t0, t1, alpha=0.25)
        plt.text((t0 + t1) / 2, y_top * 0.9, reason, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_scores(times, sMin, sP, sQ, sT, alarm_thresh):
    mid = np.array([(a + b) / 2 for (a, b) in times], dtype=float)

    plt.figure(figsize=(14, 4))
    plt.plot(mid, sMin, lw=1.0, label="score_min")
    plt.plot(mid, sP, lw=1.0, label="score_P")
    plt.plot(mid, sQ, lw=1.0, label="score_QRS")
    plt.plot(mid, sT, lw=1.0, label="score_T")
    plt.axhline(alarm_thresh, ls="--", lw=1.0, label="alarm_thresh")
    plt.title("ALSSM Scores pro Sliding Window")
    plt.xlabel("t [s]")
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    print("[STEP] loaded npz")
    d = np.load(NPZ_PATH, allow_pickle=True)
    X = d["X"]              # (N,1,L)
    Y_cls = d["Y_cls"]      # (N,)
    fs = int(d["fs"][0])
    pathology_names = np.array(d["pathology_names"].tolist(), dtype=object)

    print("pathology_names:", pathology_names.tolist())
    print("fs:", fs, "X:", X.shape)

    # 1) Schwelle kalibrieren aus normal
    print("[STEP] start calibrate")
    thr, normal_scores = calibrate_alarm_threshold_from_normal(
        X, Y_cls, pathology_names, fs,
        false_alarm_rate=NORMAL_FALSE_ALARM_RATE,
        max_samples=8000
    )

    print(f"[CALIB] false_alarm_rate={NORMAL_FALSE_ALARM_RATE} -> alarm_thresh={thr:.3f}")
    print(f"[CALIB] normal score_min: mean={float(np.mean(normal_scores)):.3f} std={float(np.std(normal_scores)):.3f}")

    # 2) langes Signal bauen
    print("[STEP] build long stream")
    x_long, meta = build_long_stream(
        X, Y_cls, pathology_names,
        n_windows=LONG_N_WINDOWS,
        insert_prob=PATHOLOGY_INSERT_PROB,
        pathology_choices=PATHOLOGY_CHOICES,
        seed=3
    )
    print("[STEP] long stream done")

    print("[LONG] windows:", len(meta), "signal_len:", len(x_long), "seconds:", len(x_long) / fs)
    print("[LONG] inserted pathologies (truth per window):")
    for m in meta[:10]:
        print(" ", m["k"], m["label"])

    # 3) scan
    print("[STEP] start scan")
    sMin, sP, sQ, sT, times, alarms = scan_long_ecg_for_alarm(
        x_long, fs, WIN_S, HOP_S, thr
    )
    print("[STEP] scan done")
    events = merge_overlapping_alarms(alarms, gap_s=MERGE_GAP_S)

    print(f"[SCAN] alarms windows: {len(alarms)}  merged events: {len(events)}")
    for e in events[:10]:
        t0, t1, smin, reason, sp, sq, st = e
        print(f"  EVENT {t0:.2f}s-{t1:.2f}s  reason={reason}  sMin={smin:.3f} (P={sp:.3f} QRS={sq:.3f} T={st:.3f})")

    # 4) plots
    plot_long_signal_with_events(
        x_long, fs, events,
        title=f"ALSSM Alarm Lokalisation | thr={thr:.3f} | win={WIN_S}s hop={HOP_S}s"
    )
    plot_scores(times, sMin, sP, sQ, sT, thr)

    print("Fertig.")


if __name__ == "__main__":
    main()
