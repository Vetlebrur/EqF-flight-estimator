"""Plot raw gyro and accelerometer from the NIMBUS24 flight."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FC_CSV = Path("data/20241011_NIMBUS24_Flight_FC_Data.csv")

_C = {"t": 0, "ax": 9, "ay": 10, "az": 11, "gx": 15, "gy": 16, "gz": 17}
g = 9.81

raw = np.genfromtxt(FC_CSV, delimiter=",", skip_header=1)
raw = raw[np.isfinite(raw[:, _C["t"]])]
t = raw[:, _C["t"]]

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("NIMBUS24 Raw IMU", fontsize=13, fontweight="bold")

axes[0].plot(t, raw[:, _C["gx"]], lw=0.8, label="gx")
axes[0].plot(t, raw[:, _C["gy"]], lw=0.8, label="gy")
axes[0].plot(t, raw[:, _C["gz"]], lw=0.8, label="gz")
axes[0].set_ylabel("Gyro [rad/s]")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, raw[:, _C["ax"]] * g, lw=0.8, label="ax")
axes[1].plot(t, raw[:, _C["ay"]] * g, lw=0.8, label="ay")
axes[1].plot(t, raw[:, _C["az"]] * g, lw=0.8, label="az")
axes[1].set_ylabel("Accel [m/s²]")
axes[1].set_xlabel("Time [s]")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = Path("outputs/raw_imu.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
