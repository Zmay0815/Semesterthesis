import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# DATASET PATH
# --------------------------------------------------

MIT_PATH = r"C:\Users\giann\OneDrive\Desktop\exploration_alssm_mg-main\1D_ALSSM_NN\mit-bih-arrhythmia-database-1.0.0"

RECORD = "100"

# search windows (seconds)
P_WINDOW = (-0.25, -0.08)
T_WINDOW = (0.10, 0.45)

# --------------------------------------------------
# load signal
# --------------------------------------------------

record = wfdb.rdrecord(os.path.join(MIT_PATH, RECORD))
signal = record.p_signal[:,0]
fs = record.fs

annotation = wfdb.rdann(os.path.join(MIT_PATH, RECORD), "atr")

r_peaks = np.array([
    annotation.sample[i]
    for i,s in enumerate(annotation.symbol)
    if s == "N"
])

# --------------------------------------------------
# choose central beat
# --------------------------------------------------

r = r_peaks[len(r_peaks)//2]

p_start = int(r + P_WINDOW[0]*fs)
p_end   = int(r + P_WINDOW[1]*fs)

t_start = int(r + T_WINDOW[0]*fs)
t_end   = int(r + T_WINDOW[1]*fs)

# plotting window
window = int(0.6*fs)

start = r - window
end   = r + window

time = np.arange(start,end)/fs

segment = signal[start:end]

# --------------------------------------------------
# plotting
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR,"figures")
os.makedirs(FIG_DIR,exist_ok=True)

plt.figure(figsize=(10,3))

plt.plot(time,segment,color="black",linewidth=1.2)

# R peak
plt.scatter(r/fs,signal[r],color="red",s=80,label="R peak")

# P window
plt.axvspan(
    p_start/fs,
    p_end/fs,
    color="blue",
    alpha=0.2,
    label="P search window"
)

# T window
plt.axvspan(
    t_start/fs,
    t_end/fs,
    color="green",
    alpha=0.2,
    label="T search window"
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")

plt.title("Anchor-based search windows for P and T detection")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR,"pt_search_windows.png"),
    dpi=300
)

plt.close()

print("Figure saved.")