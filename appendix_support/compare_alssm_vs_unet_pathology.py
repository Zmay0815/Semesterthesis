# -*- coding: utf-8 -*-
"""
Vergleich: ALSSM vs U-Net (Segmentierung + Pathologie Alarm) auf synthetischen Pathologien.

Voraussetzungen:
- unet_pqt_pathology.pt existiert (aus deinem Training)
- synth_leadII_pqt_pathology_windows.npz existiert (oder du generierst direkt im Code)
- Du ergänzt run_alssm_detector(x, fs) passend zu deinem ALSSM Code.

Outputs:
- Plot mit Signal + GT + U-Net Peaks + ALSSM Peaks
- Metriken: F1 für P/QRS/T (Peak matching), Timing error, Pathologie-Accuracy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ekg_u import extract_alssm_features, merge_cr_with_noise
from ekg_u import filter_baseline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segment Klassen
CLS_BG  = 0
CLS_P   = 1
CLS_QRS = 2
CLS_T   = 3


def conv_block(cin, cout, k=7):
    pad = k // 2
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=pad),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
        nn.Conv1d(cout, cout, k, padding=pad),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
    )


class UNet1D_MultiHead(nn.Module):
    def __init__(self, in_ch=1, base=32, n_seg=4, n_cls=8):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = conv_block(base*2, base*4)
        self.pool3 = nn.MaxPool1d(2)

        self.bottleneck = conv_block(base*4, base*8)

        self.up3 = nn.ConvTranspose1d(base*8, base*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base*8, base*4)

        self.up2 = nn.ConvTranspose1d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base*4, base*2)

        self.up1 = nn.ConvTranspose1d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = conv_block(base*2, base)

        self.head_seg = nn.Conv1d(base, n_seg, kernel_size=1)

        self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.head_cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*8, base*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base*4, n_cls)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        seg_logits = self.head_seg(d1)
        cls_feat = self.cls_pool(b)
        cls_logits = self.head_cls(cls_feat)
        return seg_logits, cls_logits


def mask_to_peaks(y_mask, cls_id):
    """
    Aus Segment-Maske (L,) Peaks als Zentrum jeder zusammenhängenden Region holen.
    """
    idx = np.where(y_mask == cls_id)[0]
    if idx.size == 0:
        return np.array([], dtype=int)

    peaks = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            mid = (start + prev) // 2
            peaks.append(mid)
            start = i
            prev = i
    mid = (start + prev) // 2
    peaks.append(mid)
    return np.array(peaks, dtype=int)


def match_peaks(gt, pred, tol):
    """
    Greedy Matching von Peak-Positionen.
    tol in samples.
    Returns: tp, fp, fn, mean_abs_error_samples
    """
    gt = np.asarray(gt, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if gt.size == 0 and pred.size == 0:
        return 0, 0, 0, np.nan
    if gt.size == 0:
        return 0, pred.size, 0, np.nan
    if pred.size == 0:
        return 0, 0, gt.size, np.nan

    gt_used = np.zeros(gt.size, dtype=bool)
    pred_used = np.zeros(pred.size, dtype=bool)

    errors = []
    tp = 0

    for i, p in enumerate(pred):
        best_j = -1
        best_d = 10**9
        for j, g in enumerate(gt):
            if gt_used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            tp += 1
            pred_used[i] = True
            gt_used[best_j] = True
            errors.append(best_d)

    fp = int((~pred_used).sum())
    fn = int((~gt_used).sum())
    mae = float(np.mean(errors)) if errors else np.nan
    return tp, fp, fn, mae


def f1_from_counts(tp, fp, fn):
    denom = (2*tp + fp + fn)
    return (2*tp / denom) if denom > 0 else 0.0


def load_model(pt_path):
    ckpt = torch.load(pt_path, map_location=DEVICE)
    pathology_names = ckpt["pathology_names"]
    print(pathology_names)
    model = UNet1D_MultiHead(in_ch=1, base=32, n_seg=4, n_cls=len(pathology_names)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, pathology_names


# ----------------------------------------------------------------------
# HIER musst du deinen ALSSM Detector einhängen
# ----------------------------------------------------------------------
def run_alssm_detector(x, fs):
    """
    ALSSM Peaks für P, QRS, T + Alarm-Score (Anomalie).
    Returns:
      peaks_p, peaks_qrs, peaks_t, alarm_dict
    alarm_dict enthält:
      - score_global: percentil-basierter LCR Score
      - is_alarm: bool
      - details: optionale Scores pro Welle
    """

    x_use = x.astype(np.float32)

    # Parameter-Skalierung von fs=2400 auf aktuelles fs
    scale = fs / 2400.0

    def scale_params(p):
        p = dict(p)
        p["l_side"] = max(3, int(round(p["l_side"] * scale)))
        p["ref_index"] = int(round(p["ref_index"] * scale))
        return p

    P_params = scale_params({'ref_index': 1310, 'polydegree': 2, 'l_side': 60,  'g': 40})
    Q_params = scale_params({'ref_index': 1370, 'polydegree': 2, 'l_side': 18,  'g': 40})
    R_params = scale_params({'ref_index': 1435, 'polydegree': 2, 'l_side': 25,  'g': 40})
    S_params = scale_params({'ref_index': 1482, 'polydegree': 2, 'l_side': 10,  'g': 40})
    T_params = scale_params({'ref_index': 1850, 'polydegree': 2, 'l_side': 60,  'g': 40})
    QRS_params = scale_params({'ref_index': 1450, 'polydegree': 3, 'l_side': 120, 'g': 300})

    raw = x_use.reshape(-1, 1)

    feats_P = extract_alssm_features(raw, P_params)
    feats_Q = extract_alssm_features(raw, Q_params)
    feats_R = extract_alssm_features(raw, R_params)
    feats_S = extract_alssm_features(raw, S_params)
    feats_T = extract_alssm_features(raw, T_params)

    peaks_p = np.asarray(feats_P["peaks"], dtype=int)
    peaks_t = np.asarray(feats_T["peaks"], dtype=int)

    merge_inputs = [
        {"binary_lcr": feats_Q["binary_lcr"], "weight": 1.0},
        {"binary_lcr": feats_R["binary_lcr"], "weight": 1.0},
        {"binary_lcr": feats_S["binary_lcr"], "weight": 1.0},
    ]
    cr = merge_cr_with_noise(merge_inputs, noise_level=0.0)
    feats_QRS = extract_alssm_features(cr, QRS_params)
    peaks_qrs = np.asarray(feats_QRS["peaks"], dtype=int)

    # ------------------------
    # ALSSM Alarm-Score
    # Idee: bei Pathologie passt die Morphologie schlechter zum Normal-Referenzzustand
    # -> LCR wird "schlechter" (tiefer). Wir nehmen ein robustes Percentil.
    # ------------------------

    def robust_alarm_score(feats):
        lcr = np.asarray(feats["lcr"], dtype=float)
        if lcr.size == 0:
            return np.nan
        return float(np.percentile(lcr, 5))  # 5%-Quantil = "schlechteste" Stellen

    sP = robust_alarm_score(feats_P)
    sQRS = robust_alarm_score(feats_QRS)
    sT = robust_alarm_score(feats_T)

    # globaler Score: min von (P, QRS, T)
    score_global = np.nanmin([sP, sQRS, sT])

    # Alarm-Schwelle: musst du einmal auf Normaldaten kalibrieren!
    # Für den Start: -1.5 ist oft brauchbar, kann aber je nach Simulation abweichen.
    ALARM_THRESH = -1.5
    is_alarm = bool(score_global < ALARM_THRESH)

    alarm = {
        "score_global": score_global,
        "is_alarm": is_alarm,
        "threshold": ALARM_THRESH,
        "score_P": sP,
        "score_QRS": sQRS,
        "score_T": sT,
    }

    return peaks_p, peaks_qrs, peaks_t, alarm





def main():
    npz_path = "synth_leadII_pqt_pathology_windows.npz"
    pt_path = "unet_pqt_pathology.pt"

    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]          # (N,1,L)
    Y_seg = d["Y"]      # (N,L)
    Y_cls = d["Y_cls"]  # (N,)
    fs = int(d["fs"][0])
    pathology_names = d["pathology_names"].tolist()

    model, pathology_names_model = load_model(pt_path)

    # zufällige Beispiele anschauen
    rng = np.random.default_rng(0)
    idxs = rng.choice(X.shape[0], size=5, replace=False)

    for k, idx in enumerate(idxs, start=1):
        x = X[idx, 0, :]
        y_gt = Y_seg[idx, :]
        pid_gt = int(Y_cls[idx])

        # U-Net Inferenz
        xt = torch.from_numpy(x[None, None, :]).to(DEVICE)
        with torch.no_grad():
            seg_logits, cls_logits = model(xt)
            seg_pred = torch.argmax(seg_logits, dim=1).cpu().numpy()[0]  # (L,)
            pid_pred = int(torch.argmax(cls_logits, dim=1).cpu().item())

        # Peaks aus GT und U-Net Masken
        gt_p = mask_to_peaks(y_gt, CLS_P)
        gt_qrs = mask_to_peaks(y_gt, CLS_QRS)
        gt_t = mask_to_peaks(y_gt, CLS_T)

        un_p = mask_to_peaks(seg_pred, CLS_P)
        un_qrs = mask_to_peaks(seg_pred, CLS_QRS)
        un_t = mask_to_peaks(seg_pred, CLS_T)

        # ALSSM Peaks
        al_p, al_qrs, al_t, al_alarm = run_alssm_detector(x, fs)


        # Metriken (Timing tolerances typisch: QRS enger, P/T breiter)
        tol_qrs = int(0.05 * fs)  # 50 ms
        tol_pt  = int(0.08 * fs)  # 80 ms

        tp, fp, fn, mae = match_peaks(gt_p, un_p, tol_pt)
        f1_un_p = f1_from_counts(tp, fp, fn)
        tp, fp, fn, mae2 = match_peaks(gt_qrs, un_qrs, tol_qrs)
        f1_un_qrs = f1_from_counts(tp, fp, fn)
        tp, fp, fn, mae3 = match_peaks(gt_t, un_t, tol_pt)
        f1_un_t = f1_from_counts(tp, fp, fn)

        tp, fp, fn, mae = match_peaks(gt_p, al_p, tol_pt)
        f1_al_p = f1_from_counts(tp, fp, fn)
        tp, fp, fn, mae2 = match_peaks(gt_qrs, al_qrs, tol_qrs)
        f1_al_qrs = f1_from_counts(tp, fp, fn)
        tp, fp, fn, mae3 = match_peaks(gt_t, al_t, tol_pt)
        f1_al_t = f1_from_counts(tp, fp, fn)

        # Plot
        t = np.arange(x.size) / fs
        plt.figure(figsize=(12, 4))
        plt.plot(t, x, lw=1.0)
        plt.title(
            f"Sample {idx} | GT={pathology_names[pid_gt]} | U-Net={pathology_names_model[pid_pred]}\n"
            f"U-Net F1 P={f1_un_p:.3f} QRS={f1_un_qrs:.3f} T={f1_un_t:.3f} | "
            f"ALSSM F1 P={f1_al_p:.3f} QRS={f1_al_qrs:.3f} T={f1_al_t:.3f}"
        )
        plt.xlabel("t [s]")
        if al_alarm["is_alarm"]:
            plt.text(
                0.02, 0.92,
                f"ALSSM ALARM  score={al_alarm['score_global']:.2f}  thr={al_alarm['threshold']:.2f}",
                transform=plt.gca().transAxes,
                color="red",
                fontsize=11,
                fontweight="bold"
            )
        else:
            plt.text(
                0.02, 0.92,
                f"ALSSM OK  score={al_alarm['score_global']:.2f}  thr={al_alarm['threshold']:.2f}",
                transform=plt.gca().transAxes,
                color="green",
                fontsize=11,
                fontweight="bold"
            )


        # GT markers
        plt.scatter(gt_p/fs, x[gt_p], s=25, label="GT P")
        plt.scatter(gt_qrs/fs, x[gt_qrs], s=25, label="GT QRS")
        plt.scatter(gt_t/fs, x[gt_t], s=25, label="GT T")

        # U-Net markers
        plt.scatter(un_p/fs, x[un_p], s=25, marker="x", label="U-Net P")
        plt.scatter(un_qrs/fs, x[un_qrs], s=25, marker="x", label="U-Net QRS")
        plt.scatter(un_t/fs, x[un_t], s=25, marker="x", label="U-Net T")

        # ALSSM markers
        if al_p.size:
            plt.scatter(al_p/fs, x[al_p], s=40, marker="^", label="ALSSM P")
        if al_qrs.size:
            plt.scatter(al_qrs/fs, x[al_qrs], s=40, marker="^", label="ALSSM QRS")
        if al_t.size:
            plt.scatter(al_t/fs, x[al_t], s=40, marker="^", label="ALSSM T")

        plt.legend(loc="upper right", ncol=2)
        plt.tight_layout()
        plt.show()

        print(
            f"[{idx}] GT pathology={pathology_names[pid_gt]} | U-Net pathology={pathology_names_model[pid_pred]} | "
            f"ALSSM alarm={al_alarm['is_alarm']} score={al_alarm['score_global']:.3f} "
            f"(P={al_alarm['score_P']:.3f} QRS={al_alarm['score_QRS']:.3f} T={al_alarm['score_T']:.3f})"
        )


    print("Fertig.")


if __name__ == "__main__":
    main()
