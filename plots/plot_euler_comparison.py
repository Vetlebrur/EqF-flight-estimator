"""Compare TG-EqF with expm vs Euler discretization of A."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

DATASET     = "full"
USE_MA_FILTER = True   # must match which outputs exist on disk

_suffix = {"full": "full", "30s": "30s", "1s_loop": "1s_loop"}[DATASET]
_ma = "_ma" if USE_MA_FILTER else ""

EXPM_CSV  = Path(f"outputs/tg_eqf_output_{_suffix}{_ma}.csv")
EULER_CSV = Path(f"outputs/tg_eqf_output_{_suffix}{_ma}_euler.csv")

for p in (EXPM_CSV, EULER_CSV):
    if not p.exists():
        print(f"Missing: {p}  — run eqf_filter.py with USE_EULER_DISCR toggled.")
        raise SystemExit(1)


def load(path: Path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    valid = np.isfinite(data[:, :25]).all(axis=1)
    data  = data[valid]
    t  = data[:, 0]
    px, py, pz = data[:, 1], data[:, 2], data[:, 3]
    vx, vy, vz = data[:, 4], data[:, 5], data[:, 6]
    dcm   = data[:, 7:16].reshape(-1, 3, 3)
    euler = Rotation.from_matrix(dcm).as_euler("ZYX", degrees=True)
    yaw, pitch, roll = euler[:, 0], euler[:, 1], euler[:, 2]
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    return dict(t=t, px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz,
                roll=roll, pitch=pitch, yaw=yaw, speed=speed)


e  = load(EXPM_CSV)
eu = load(EULER_CSV)

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
fig.suptitle(
    f"TG-EqF: expm vs Euler discretization  [{DATASET}{_ma}]",
    fontsize=13, fontweight="bold"
)

LW    = 0.9
ALPHA = 0.85

# ── Position: North, East, Altitude ──────────────────────────────────────────
ax = axes[0]
ax.plot(e["t"],  e["px"],   lw=LW, alpha=ALPHA, color="steelblue",  label="expm N")
ax.plot(e["t"],  e["py"],   lw=LW, alpha=0.55,  color="steelblue",  ls=":",  label="expm E")
ax.plot(e["t"],  -e["pz"],  lw=LW, alpha=0.75,  color="steelblue",  ls="-.", label="expm Alt")
ax.plot(eu["t"], eu["px"],  lw=LW, alpha=ALPHA, color="darkorange", ls="--", label="Euler N")
ax.plot(eu["t"], eu["py"],  lw=LW, alpha=0.55,  color="darkorange", ls=(0, (3,1,1,1)), label="Euler E")
ax.plot(eu["t"], -eu["pz"], lw=LW, alpha=0.75,  color="darkorange", ls=(0, (5,2,1,2)), label="Euler Alt")
ax.set_ylabel("Position [m]", fontsize=9)
ax.set_title("Position — North, East & Altitude")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

# ── Speed ─────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(e["t"],  e["speed"],  lw=LW, alpha=ALPHA, color="steelblue",  label="expm")
ax.plot(eu["t"], eu["speed"], lw=LW, alpha=ALPHA, color="darkorange", ls="--", label="Euler")
ax.set_ylabel("Speed [m/s]", fontsize=9)
ax.set_title("Total Speed")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Roll & Yaw (unwrapped) ────────────────────────────────────────────────────
ax = axes[2]
e_roll  = np.unwrap(e["roll"],   period=360)
e_yaw   = np.unwrap(e["yaw"],    period=360)
eu_roll = np.unwrap(eu["roll"],  period=360)
eu_yaw  = np.unwrap(eu["yaw"],   period=360)
ax.plot(e["t"],  e_roll,  lw=LW, alpha=ALPHA, color="steelblue",  label="expm Roll")
ax.plot(e["t"],  e_yaw,   lw=LW, alpha=0.6,   color="steelblue",  ls=":",  label="expm Yaw")
ax.plot(eu["t"], eu_roll, lw=LW, alpha=ALPHA, color="darkorange", ls="--", label="Euler Roll")
ax.plot(eu["t"], eu_yaw,  lw=LW, alpha=0.6,   color="darkorange", ls="-.", label="Euler Yaw")
ax.set_ylabel("Angle [deg]", fontsize=9)
ax.set_title("Roll & Yaw (unwrapped)")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

# ── Pitch ─────────────────────────────────────────────────────────────────────
ax = axes[3]
ax.plot(e["t"],  e["pitch"],  lw=LW, alpha=ALPHA, color="steelblue",  label="expm")
ax.plot(eu["t"], eu["pitch"], lw=LW, alpha=ALPHA, color="darkorange", ls="--", label="Euler")
ax.set_ylabel("Angle [deg]", fontsize=9)
ax.set_title("Pitch")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Time [s]", fontsize=9)

plt.tight_layout()
out = Path(f"outputs/euler_comparison_{_suffix}{_ma}.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
