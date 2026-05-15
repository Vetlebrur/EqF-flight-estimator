"""Compare gyro and accel biases from EqF and EKF filter outputs."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EQF_CSV = Path("outputs/tg_eqf_output_full.csv")
EKF_CSV = Path("outputs/ekf_output_full.csv")


def load(path: Path):
    with open(path) as f:
        cols = [c.strip() for c in f.readline().split(",")]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return {name: data[:, i] for i, name in enumerate(cols)}


eqf = load(EQF_CSV)
ekf = load(EKF_CSV)

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
fig.suptitle("Gyro & Accel Bias — EqF vs EKF", fontsize=13, fontweight="bold")

gyro_labels = ["X", "Y", "Z"]
gyro_keys   = ["bgx", "bgy", "bgz"]
accel_keys  = ["bax", "bay", "baz"]

for i, (key, label) in enumerate(zip(gyro_keys, gyro_labels)):
    ax = axes[0, i]
    ax.plot(eqf["t"], eqf[key], lw=1.0, label="EqF")
    ax.plot(ekf["t"], ekf[key], lw=1.0, ls="--", label="EKF")
    ax.set_title(f"Gyro Bias {label} [rad/s]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

for i, (key, label) in enumerate(zip(accel_keys, gyro_labels)):
    ax = axes[1, i]
    ax.plot(eqf["t"], eqf[key], lw=1.0, label="EqF")
    ax.plot(ekf["t"], ekf[key], lw=1.0, ls="--", label="EKF")
    ax.set_title(f"Accel Bias {label} [m/s²]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
out = Path("outputs/bias_comparison.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
