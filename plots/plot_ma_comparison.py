"""Compare TG-EqF output with and without moving-average pre-filter."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

DATASET = "full"

_suffix = {
    "full":    "full",
    "30s":     "30s",
    "1s_loop": "1s_loop",
}[DATASET]

RAW_CSV = Path(f"outputs/tg_eqf_output_{_suffix}.csv")
MA_CSV  = Path(f"outputs/tg_eqf_output_{_suffix}_ma.csv")

for p in (RAW_CSV, MA_CSV):
    if not p.exists():
        print(f"Missing: {p}  — run eqf_filter.py with USE_MA_FILTER toggled.")
        raise SystemExit(1)


def load(path: Path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    valid = np.isfinite(data[:, :25]).all(axis=1)
    data = data[valid]
    t  = data[:, 0]
    px, py, pz = data[:, 1], data[:, 2], data[:, 3]
    vx, vy, vz = data[:, 4], data[:, 5], data[:, 6]
    dcm = data[:, 7:16].reshape(-1, 3, 3)
    euler = Rotation.from_matrix(dcm).as_euler('ZYX', degrees=True)
    yaw, pitch, roll = euler[:, 0], euler[:, 1], euler[:, 2]
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    return dict(t=t, px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz,
                roll=roll, pitch=pitch, yaw=yaw, speed=speed)


raw = load(RAW_CSV)
ma  = load(MA_CSV)

fig, axes = plt.subplots(3, 2, figsize=(15, 11))
fig.suptitle(f"TG-EqF: Raw IMU vs Moving-Average pre-filter  [{DATASET}]",
             fontsize=13, fontweight="bold")

ALPHA = 0.8
LW = 1.0

# ── Position ──────────────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(raw["t"], raw["px"], lw=LW, alpha=ALPHA, label="Raw North")
ax.plot(raw["t"], raw["py"], lw=LW, alpha=ALPHA, label="Raw East")
ax.plot(raw["t"], -raw["pz"], lw=LW, alpha=ALPHA, label="Raw Alt")
ax.plot(ma["t"],  ma["px"],  lw=LW, alpha=ALPHA, ls="--", label="MA North")
ax.plot(ma["t"],  ma["py"],  lw=LW, alpha=ALPHA, ls="--", label="MA East")
ax.plot(ma["t"],  -ma["pz"], lw=LW, alpha=ALPHA, ls="--", label="MA Alt")
ax.set_ylabel("Position [m]")
ax.set_title("Position")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Altitude only (easier to see vertical) ────────────────────────────────────
ax = axes[0, 1]
ax.plot(raw["t"], -raw["pz"], lw=LW, color="steelblue", label="Raw")
ax.plot(ma["t"],  -ma["pz"],  lw=LW, color="darkorange", ls="--", label="MA")
ax.set_ylabel("Altitude [m]")
ax.set_title("Altitude")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Velocity ──────────────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(raw["t"], raw["vx"], lw=LW, alpha=ALPHA, label="Raw N")
ax.plot(raw["t"], raw["vy"], lw=LW, alpha=ALPHA, label="Raw E")
ax.plot(raw["t"], -raw["vz"], lw=LW, alpha=ALPHA, label="Raw Up")
ax.plot(ma["t"],  ma["vx"],  lw=LW, alpha=ALPHA, ls="--", label="MA N")
ax.plot(ma["t"],  ma["vy"],  lw=LW, alpha=ALPHA, ls="--", label="MA E")
ax.plot(ma["t"],  -ma["vz"], lw=LW, alpha=ALPHA, ls="--", label="MA Up")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Velocity")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Speed ─────────────────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(raw["t"], raw["speed"], lw=LW, color="steelblue", label="Raw")
ax.plot(ma["t"],  ma["speed"],  lw=LW, color="darkorange", ls="--", label="MA")
ax.set_ylabel("Speed [m/s]")
ax.set_title("Total Speed")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Roll & Yaw ────────────────────────────────────────────────────────────────
ax = axes[2, 0]
ax.plot(raw["t"], np.unwrap(raw["roll"],  period=360), lw=LW, color="steelblue",  alpha=ALPHA, label="Raw Roll")
ax.plot(raw["t"], np.unwrap(raw["yaw"],   period=360), lw=LW, color="steelblue",  alpha=0.5,   label="Raw Yaw", ls=":")
ax.plot(ma["t"],  np.unwrap(ma["roll"],   period=360), lw=LW, color="darkorange", alpha=ALPHA, label="MA Roll",  ls="--")
ax.plot(ma["t"],  np.unwrap(ma["yaw"],    period=360), lw=LW, color="darkorange", alpha=0.5,   label="MA Yaw",   ls="-.")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Angle [deg]")
ax.set_title("Roll & Yaw (unwrapped)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Pitch ─────────────────────────────────────────────────────────────────────
ax = axes[2, 1]
ax.plot(raw["t"], raw["pitch"], lw=LW, color="steelblue",  label="Raw")
ax.plot(ma["t"],  ma["pitch"],  lw=LW, color="darkorange", ls="--", label="MA")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Angle [deg]")
ax.set_title("Pitch")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path(f"outputs/ma_comparison_{_suffix}.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
