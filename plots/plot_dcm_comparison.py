"""Plot all 9 DCM elements for EqF and FC side by side."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

FC_CSV   = "data/20241011_NIMBUS24_Flight_FC_Data.csv"
EQF_CSV  = "outputs/tg_eqf_output_full.csv"

_FC = {"t": 0, "roll": 29, "pitch": 30, "yaw": 31}

# --- Load EqF ---
eqf = np.genfromtxt(EQF_CSV, delimiter=",", skip_header=1)
eqf_t   = eqf[:, 0]
eqf_dcm = eqf[:, 7:16].reshape(-1, 3, 3)

# --- Load FC (Euler ZYX radians → DCM) ---
raw = np.genfromtxt(FC_CSV, delimiter=",", skip_header=1)
fc_mask = np.isfinite(raw[:, _FC["roll"]]) & np.isfinite(raw[:, _FC["pitch"]]) & np.isfinite(raw[:, _FC["yaw"]])
raw = raw[fc_mask]
fc_t = raw[:, _FC["t"]]
fc_dcm = Rotation.from_euler(
    "ZYX",
    np.column_stack([raw[:, _FC["yaw"]], raw[:, _FC["pitch"]], raw[:, _FC["roll"]]]),
    degrees=False,
).as_matrix()

# --- Plot ---
labels = [
    ("R₀₀", 0, 0), ("R₀₁", 0, 1), ("R₀₂", 0, 2),
    ("R₁₀", 1, 0), ("R₁₁", 1, 1), ("R₁₂", 1, 2),
    ("R₂₀", 2, 0), ("R₂₁", 2, 1), ("R₂₂", 2, 2),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
fig.suptitle("Rotation Matrix Elements — EqF vs FC", fontsize=13, fontweight="bold")

for ax, (name, i, j) in zip(axes.flat, labels):
    ax.plot(fc_t,  fc_dcm[:, i, j],  color="gray",   lw=1.0, alpha=0.7, label="FC")
    ax.plot(eqf_t, eqf_dcm[:, i, j], color="tomato", lw=1.0, alpha=0.9, label="EqF")
    ax.set_title(name, fontsize=10)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)

for ax in axes[2]:
    ax.set_xlabel("Time [s]")

handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.04, 1, 1])

out = "outputs/dcm_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
