"""Compare TG-EqF vs EKF — attitude and trajectory, stacked layout."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

DATASET       = "full"
USE_MA_FILTER = False

_suffix = {"full": "full", "30s": "30s", "1s_loop": "1s_loop"}[DATASET]
_ma = "_ma" if USE_MA_FILTER else ""

EQF_CSV = Path(f"outputs/tg_eqf_output_{_suffix}{_ma}.csv")
EKF_CSV = Path(f"outputs/ekf_output_{_suffix}.csv")

for p in (EQF_CSV, EKF_CSV):
    if not p.exists():
        print(f"Missing: {p}")
        raise SystemExit(1)


def load(path: Path):
    with open(path) as f:
        header = [c.strip() for c in f.readline().split(",")]
    has_quat = "qw" in header
    bg_start = 20 if has_quat else 16

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    valid = np.isfinite(data[:, :16]).all(axis=1)
    data  = data[valid]

    t   = data[:, 0]
    pos = data[:, 1:4]
    vel = data[:, 4:7]
    dcm = data[:, 7:16].reshape(-1, 3, 3)
    euler = Rotation.from_matrix(dcm).as_euler("ZYX", degrees=True)
    yaw, pitch, roll = euler[:, 0], euler[:, 1], euler[:, 2]
    speed = np.linalg.norm(vel, axis=1)
    return dict(t=t, pos=pos, vel=vel, roll=roll, pitch=pitch, yaw=yaw, speed=speed)


eqf = load(EQF_CSV)
ekf = load(EKF_CSV)

# ── FC reference ──────────────────────────────────────────────────────────────
_FC_CSV = Path(f"data/20241011_NIMBUS24_Flight_FC_Data.csv")
_Cc = {"t":0,"lat":2,"lon":1,"alt":3,"pn":36,"pe":37,"pd":38,
       "vn":39,"ve":40,"vd":41,"roll":29,"pitch":30,"yaw":31}
R_EARTH = 6_378_137.0

_raw = np.genfromtxt(_FC_CSV, delimiter=",", skip_header=1)
fc_t, fc_pos, fc_vel, fc_roll, fc_pitch, fc_yaw = [], [], [], [], [], []
for _row in _raw:
    _t = _row[_Cc["t"]]
    if not np.isfinite(_t): continue
    _pn, _pe, _pd = _row[_Cc["pn"]], _row[_Cc["pe"]], _row[_Cc["pd"]]
    if not (np.isfinite(_pn) and np.isfinite(_pe) and np.isfinite(_pd)): continue
    _vn, _ve, _vd = _row[_Cc["vn"]], _row[_Cc["ve"]], _row[_Cc["vd"]]
    _r, _p, _y = _row[_Cc["roll"]], _row[_Cc["pitch"]], _row[_Cc["yaw"]]
    if not (np.isfinite(_r) and np.isfinite(_p) and np.isfinite(_y)): continue
    fc_t.append(_t)
    fc_pos.append([_pn, _pe, _pd])
    fc_vel.append([_vn, _ve, _vd])
    fc_roll.append(np.degrees(_r)); fc_pitch.append(np.degrees(_p)); fc_yaw.append(np.degrees(_y))

fc_t     = np.array(fc_t)
fc_pos   = np.array(fc_pos)
fc_vel   = np.array(fc_vel)
fc_roll  = np.array(fc_roll)
fc_pitch = np.array(fc_pitch)
fc_yaw   = np.array(fc_yaw)
fc_speed = np.linalg.norm(fc_vel, axis=1)

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
fig.suptitle(
    f"TG-EqF vs EKF vs FC  [{DATASET}]",
    fontsize=13, fontweight="bold"
)

LW    = 0.9
ALPHA = 0.85
C_EQF = "steelblue"
C_EKF = "darkorange"
C_FC  = "seagreen"

# ── Position: North, East, Altitude ──────────────────────────────────────────
ax = axes[0]
ax.plot(eqf["t"], eqf["pos"][:, 0],  lw=LW, alpha=ALPHA, color=C_EQF, label="EqF N")
ax.plot(eqf["t"], eqf["pos"][:, 1],  lw=LW, alpha=0.55,  color=C_EQF, ls=":",  label="EqF E")
ax.plot(eqf["t"], -eqf["pos"][:, 2], lw=LW, alpha=0.75,  color=C_EQF, ls="-.", label="EqF Alt")
ax.plot(ekf["t"], ekf["pos"][:, 0],  lw=LW, alpha=ALPHA, color=C_EKF, ls="--",          label="EKF N")
ax.plot(ekf["t"], ekf["pos"][:, 1],  lw=LW, alpha=0.55,  color=C_EKF, ls=(0,(3,1,1,1)), label="EKF E")
ax.plot(ekf["t"], -ekf["pos"][:, 2], lw=LW, alpha=0.75,  color=C_EKF, ls=(0,(5,2,1,2)), label="EKF Alt")
ax.plot(fc_t, fc_pos[:, 0],  lw=LW, alpha=ALPHA, color=C_FC, ls="--",          label="FC N")
ax.plot(fc_t, fc_pos[:, 1],  lw=LW, alpha=0.55,  color=C_FC, ls=(0,(3,1,1,1)), label="FC E")
ax.plot(fc_t, -fc_pos[:, 2], lw=LW, alpha=0.75,  color=C_FC, ls=(0,(5,2,1,2)), label="FC Alt")
ax.set_ylabel("Position [m]", fontsize=9)
ax.set_title("Position — North, East & Altitude")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

# ── Speed ─────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(eqf["t"], eqf["speed"], lw=LW, alpha=ALPHA, color=C_EQF, label="TG-EqF")
ax.plot(ekf["t"], ekf["speed"], lw=LW, alpha=ALPHA, color=C_EKF, ls="--", label="EKF")
ax.plot(fc_t, fc_speed, lw=LW, alpha=ALPHA, color=C_FC, ls=":", label="FC")
ax.set_ylabel("Speed [m/s]", fontsize=9)
ax.set_title("Total Speed")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Roll & Yaw (unwrapped) ────────────────────────────────────────────────────
ax = axes[2]
eqf_roll = np.unwrap(eqf["roll"], period=360)
eqf_yaw  = np.unwrap(eqf["yaw"],  period=360)
ekf_roll = np.unwrap(ekf["roll"], period=360)
ekf_yaw  = np.unwrap(ekf["yaw"],  period=360)
ax.plot(eqf["t"], eqf_roll, lw=LW, alpha=ALPHA, color=C_EQF, label="EqF Roll")
ax.plot(eqf["t"], eqf_yaw,  lw=LW, alpha=0.6,   color=C_EQF, ls=":",  label="EqF Yaw")
ax.plot(ekf["t"], ekf_roll, lw=LW, alpha=ALPHA, color=C_EKF, ls="--", label="EKF Roll")
ax.plot(ekf["t"], ekf_yaw,  lw=LW, alpha=0.6,   color=C_EKF, ls="-.", label="EKF Yaw")
ax.plot(fc_t, np.unwrap(fc_roll, period=360), lw=LW, alpha=ALPHA, color=C_FC, ls=":",         label="FC Roll")
ax.plot(fc_t, np.unwrap(fc_yaw,  period=360), lw=LW, alpha=0.6,   color=C_FC, ls=(0,(1,1,1)), label="FC Yaw")
ax.set_ylabel("Angle [deg]", fontsize=9)
ax.set_title("Roll & Yaw (unwrapped)")
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)

# ── Pitch ─────────────────────────────────────────────────────────────────────
ax = axes[3]
ax.plot(eqf["t"], eqf["pitch"], lw=LW, alpha=ALPHA, color=C_EQF, label="TG-EqF")
ax.plot(ekf["t"], ekf["pitch"], lw=LW, alpha=ALPHA, color=C_EKF, ls="--", label="EKF")
ax.plot(fc_t, fc_pitch, lw=LW, alpha=ALPHA, color=C_FC, ls=":", label="FC")
ax.set_ylabel("Angle [deg]", fontsize=9)
ax.set_title("Pitch")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Time [s]", fontsize=9)

plt.tight_layout()
out = Path(f"outputs/filter_comparison_{_suffix}.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
