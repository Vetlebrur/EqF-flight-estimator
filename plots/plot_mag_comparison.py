"""Compare TG-EqF with and without magnetometer updates."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))
import eqf_filter

DATASET = "full"
_DATA_FILES = {
    "full":    "data/20241011_NIMBUS24_Flight_FC_Data.csv",
    "30s":     "data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",
    "1s_loop": "data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",
}
CSV_IN = _DATA_FILES[DATASET]

NO_MAG_CSV = Path(f"outputs/tg_eqf_output_{DATASET}_nomag.csv")
MAG_CSV    = Path(f"outputs/tg_eqf_output_{DATASET}_mag.csv")

print("Running filter WITHOUT magnetometer...")
eqf_filter.run(csv_in=CSV_IN, csv_out=str(NO_MAG_CSV), use_mag_update=False, silent=False)

print("\nRunning filter WITH magnetometer...")
eqf_filter.run(csv_in=CSV_IN, csv_out=str(MAG_CSV), use_mag_update=True, silent=False)


def load(path: Path) -> dict:
    data  = np.genfromtxt(path, delimiter=",", skip_header=1)
    valid = np.isfinite(data[:, :25]).all(axis=1)
    data  = data[valid]
    t     = data[:, 0]
    px, py, pz = data[:, 1], data[:, 2], data[:, 3]
    vx, vy, vz = data[:, 4], data[:, 5], data[:, 6]
    dcm   = data[:, 7:16].reshape(-1, 3, 3)
    euler = Rotation.from_matrix(dcm).as_euler("ZYX", degrees=True)
    yaw, pitch, roll = euler[:, 0], euler[:, 1], euler[:, 2]
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    return dict(t=t, px=px, py=py, pz=pz,
                roll=roll, pitch=pitch, yaw=yaw, speed=speed)


nm = load(NO_MAG_CSV)
mg = load(MAG_CSV)

LW    = 0.9
ALPHA = 0.85
C_NM  = "steelblue"
C_MAG = "darkorange"

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
fig.suptitle(f"TG-EqF: GNSS-only vs GNSS + Magnetometer  [{DATASET}]",
             fontsize=13, fontweight="bold")

# ── Roll & Yaw ────────────────────────────────────────────────────────────────
ax = axes[0]
nm_roll = np.unwrap(nm["roll"], period=360)
mg_roll = np.unwrap(mg["roll"], period=360)
nm_yaw  = np.unwrap(nm["yaw"],  period=360)
mg_yaw  = np.unwrap(mg["yaw"],  period=360)
ax.plot(nm["t"], nm_roll, lw=LW, alpha=ALPHA, color=C_NM,  label="No Mag — Roll")
ax.plot(nm["t"], nm_yaw,  lw=LW, alpha=0.6,   color=C_NM,  ls=":",  label="No Mag — Yaw")
ax.plot(mg["t"], mg_roll, lw=LW, alpha=ALPHA, color=C_MAG, ls="--", label="Mag — Roll")
ax.plot(mg["t"], mg_yaw,  lw=LW, alpha=0.6,   color=C_MAG, ls="-.", label="Mag — Yaw")
ax.set_ylabel("Angle [deg]", fontsize=9)
ax.set_title("Roll & Yaw (unwrapped)")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

# ── Pitch ─────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(nm["t"], nm["pitch"], lw=LW, alpha=ALPHA, color=C_NM,  label="No Mag")
ax.plot(mg["t"], mg["pitch"], lw=LW, alpha=ALPHA, color=C_MAG, ls="--", label="Mag")
ax.set_ylabel("Angle [deg]", fontsize=9)
ax.set_title("Pitch")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Position ──────────────────────────────────────────────────────────────────
ax = axes[2]
ax.plot(nm["t"], nm["px"],   lw=LW, alpha=ALPHA, color=C_NM,  label="No Mag N")
ax.plot(nm["t"], nm["py"],   lw=LW, alpha=0.55,  color=C_NM,  ls=":",  label="No Mag E")
ax.plot(nm["t"], -nm["pz"],  lw=LW, alpha=0.75,  color=C_NM,  ls="-.", label="No Mag Alt")
ax.plot(mg["t"], mg["px"],   lw=LW, alpha=ALPHA, color=C_MAG, ls="--", label="Mag N")
ax.plot(mg["t"], mg["py"],   lw=LW, alpha=0.55,  color=C_MAG, ls=(0, (3,1,1,1)), label="Mag E")
ax.plot(mg["t"], -mg["pz"],  lw=LW, alpha=0.75,  color=C_MAG, ls=(0, (5,2,1,2)), label="Mag Alt")
ax.set_ylabel("Position [m]", fontsize=9)
ax.set_title("Position — North, East & Altitude")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

# ── Speed ─────────────────────────────────────────────────────────────────────
ax = axes[3]
ax.plot(nm["t"], nm["speed"], lw=LW, alpha=ALPHA, color=C_NM,  label="No Mag")
ax.plot(mg["t"], mg["speed"], lw=LW, alpha=ALPHA, color=C_MAG, ls="--", label="Mag")
ax.set_ylabel("Speed [m/s]", fontsize=9)
ax.set_title("Total Speed")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Time [s]", fontsize=9)

plt.tight_layout()
out = Path(f"outputs/mag_comparison_{DATASET}.png")
plt.savefig(out, dpi=150)
print(f"\nSaved to {out}")
plt.show()
