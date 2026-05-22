"""Compare EqF with vs without magnetometer updates, alongside FC — in plot_trajectory style."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))
import eqf_filter

FC_CSV  = "data/20241011_NIMBUS24_Flight_FC_Data.csv"
OUT_ON  = "outputs/tg_eqf_output_mag_on.csv"
OUT_OFF = "outputs/tg_eqf_output_mag_off.csv"
R_EARTH = 6_378_137.0

_C = {
    "t": 0, "lat": 2, "lon": 1, "alt": 3,
    "gps_vn": 4, "gps_ve": 5, "gps_vd": 6,
    "pn": 36, "pe": 37, "pd": 38,
    "vn": 39, "ve": 40, "vd": 41,
    "roll": 29, "pitch": 30, "yaw": 31,
}

# =============================================================================
# Run filter
# =============================================================================

_CSV_IN = {
    "full":    "data/20241011_NIMBUS24_Flight_FC_Data.csv",
    "30s":     "data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",
    "1s_loop": "data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",
}[eqf_filter.DATASET]

print("Running EqF WITH magnetometer...")
eqf_filter.run(csv_in=_CSV_IN, csv_out=OUT_ON,  use_mag_update=True,  silent=True)
print("Running EqF WITHOUT magnetometer...")
eqf_filter.run(csv_in=_CSV_IN, csv_out=OUT_OFF, use_mag_update=False, silent=True)

# =============================================================================
# Load helpers
# =============================================================================

def load_eqf(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    valid = np.isfinite(d[:, :16]).all(axis=1)
    d = d[valid]
    t   = d[:, 0]
    pos = d[:, 1:4]
    vel = d[:, 4:7]
    dcm = d[:, 7:16].reshape(-1, 3, 3)
    e   = Rotation.from_matrix(dcm).as_euler("ZYX", degrees=True)
    roll  = np.unwrap(e[:, 2], discont=180)
    pitch = np.unwrap(e[:, 1], discont=180)
    yaw   = np.unwrap(e[:, 0], discont=180)
    speed = np.linalg.norm(vel, axis=1)
    return t, pos, vel, speed, roll, pitch, yaw

def gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    return np.array([dlat * R_EARTH, dlon * np.cos(np.radians(lat0)) * R_EARTH, -(alt - alt0)])

# =============================================================================
# Load FC + GNSS
# =============================================================================

raw = np.genfromtxt(FC_CSV, delimiter=",", skip_header=1)
valid_gps = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
first = int(np.argmax(valid_gps))
lat0, lon0, alt0 = raw[first, _C["lat"]], raw[first, _C["lon"]], raw[first, _C["alt"]] / 1000.0

gnss_pos, gnss_vel, gnss_t = [], [], []
fc_pos, fc_vel, fc_att, fc_att_t, fc_t = [], [], [], [], []

for row in raw:
    t_row = row[_C["t"]]
    if not np.isfinite(t_row):
        continue
    lat, lon = row[_C["lat"]], row[_C["lon"]]
    if lat != 0 and lon != 0:
        alt = row[_C["alt"]] / 1000.0
        gnss_pos.append(gps_to_ned(lat, lon, alt, lat0, lon0, alt0))
        gnss_vel.append(np.array([row[_C["gps_vn"]], row[_C["gps_ve"]], row[_C["gps_vd"]]]) / 1000.0)
        gnss_t.append(t_row)
    pn, pe, pd = row[_C["pn"]], row[_C["pe"]], row[_C["pd"]]
    if np.isfinite(pn) and np.isfinite(pe) and np.isfinite(pd):
        fc_pos.append([pn, pe, pd])
        vn, ve, vd = row[_C["vn"]], row[_C["ve"]], row[_C["vd"]]
        if np.isfinite(vn) and np.isfinite(ve) and np.isfinite(vd):
            fc_vel.append([vn, ve, vd])
        r, p, y = row[_C["roll"]], row[_C["pitch"]], row[_C["yaw"]]
        if np.isfinite(r) and np.isfinite(p) and np.isfinite(y):
            fc_att.append([r, p, y])
            fc_att_t.append(t_row)
        fc_t.append(t_row)

gnss_pos  = np.array(gnss_pos)  if gnss_pos  else np.empty((0, 3))
gnss_vel  = np.array(gnss_vel)  if gnss_vel  else np.empty((0, 3))
gnss_t    = np.array(gnss_t)
fc_pos    = np.array(fc_pos)    if fc_pos    else np.empty((0, 3))
fc_vel    = np.array(fc_vel)    if fc_vel    else np.empty((0, 3))
fc_att    = np.array(fc_att)    if fc_att    else np.empty((0, 3))
fc_att_t  = np.array(fc_att_t)
fc_t      = np.array(fc_t)

fc_roll  = np.degrees(np.unwrap(fc_att[:, 0])) if len(fc_att) else np.zeros(0)
fc_pitch = np.degrees(np.unwrap(fc_att[:, 1])) if len(fc_att) else np.zeros(0)
fc_yaw   = np.degrees(np.unwrap(fc_att[:, 2])) if len(fc_att) else np.zeros(0)
fc_speed = np.linalg.norm(fc_vel, axis=1)       if len(fc_vel) else np.zeros(0)

t_on,  pos_on,  vel_on,  spd_on,  roll_on,  pitch_on,  yaw_on  = load_eqf(OUT_ON)
t_off, pos_off, vel_off, spd_off, roll_off, pitch_off, yaw_off = load_eqf(OUT_OFF)

# =============================================================================
# Plot — same 3×2 layout as plot_trajectory.py
# =============================================================================

fig = plt.figure(figsize=(16, 12))
fig.suptitle("EqF: Magnetometer ON vs OFF vs FC", fontsize=14, fontweight="bold")

kw_on  = dict(color="tomato",    lw=1.2, label="EqF + mag")
kw_off = dict(color="steelblue", lw=1.2, label="EqF no mag", ls="--")
kw_fc  = dict(color="gray",      lw=1.0, alpha=0.7)
kw_gps = dict(color="green",     lw=1.0, ls="--", alpha=0.7)

# --- 3D Trajectory ---
ax1 = fig.add_subplot(3, 2, 1, projection="3d")
ax1.plot(pos_on[:, 0],  pos_on[:, 1],  -pos_on[:, 2],  **kw_on)
ax1.plot(pos_off[:, 0], pos_off[:, 1], -pos_off[:, 2], **kw_off)
if len(gnss_pos):
    ax1.plot(gnss_pos[:, 0], gnss_pos[:, 1], -gnss_pos[:, 2], color="green", lw=1, ls="--", alpha=0.7, label="GNSS")
if len(fc_pos):
    ax1.plot(fc_pos[:, 0], fc_pos[:, 1], -fc_pos[:, 2], color="gray", lw=1, alpha=0.7, label="FC")
ax1.set_xlabel("North [m]"); ax1.set_ylabel("East [m]"); ax1.set_zlabel("Altitude [m]")
ax1.set_title("3D Trajectory"); ax1.legend(fontsize=7); ax1.grid(True)

# --- Position components ---
ax2 = fig.add_subplot(3, 2, 2)
for arr, kw in [(pos_on, kw_on), (pos_off, kw_off)]:
    t_arr = t_on if kw is kw_on else t_off
    ax2.plot(t_arr, arr[:, 0], lw=kw["lw"], color=kw["color"], ls=kw.get("ls", "-"), label=kw["label"] + " N")
    ax2.plot(t_arr, arr[:, 1], lw=kw["lw"], color=kw["color"], ls=kw.get("ls", "-"), alpha=0.7)
    ax2.plot(t_arr, -arr[:, 2], lw=kw["lw"], color=kw["color"], ls=kw.get("ls", "-"), alpha=0.5)
if len(gnss_pos):
    ax2.plot(gnss_t, gnss_pos[:, 0], color="green", lw=1, ls="--", alpha=0.7, label="GNSS N")
    ax2.plot(gnss_t, gnss_pos[:, 1], color="green", lw=1, ls="--", alpha=0.5)
    ax2.plot(gnss_t, -gnss_pos[:, 2], color="green", lw=1, ls="--", alpha=0.3)
if len(fc_pos):
    ax2.plot(fc_t, fc_pos[:, 0], color="gray", lw=1, alpha=0.7, label="FC N")
    ax2.plot(fc_t, fc_pos[:, 1], color="gray", lw=1, alpha=0.5)
    ax2.plot(fc_t, -fc_pos[:, 2], color="gray", lw=1, alpha=0.3)
ax2.set_ylabel("Position [m]"); ax2.set_title("Position vs Time"); ax2.legend(fontsize=7, ncol=2); ax2.grid(True)

# --- Velocity components ---
ax3 = fig.add_subplot(3, 2, 3)
for arr, t_arr, kw in [(vel_on, t_on, kw_on), (vel_off, t_off, kw_off)]:
    ax3.plot(t_arr, arr[:, 0], lw=kw["lw"], color=kw["color"], ls=kw.get("ls", "-"), label=kw["label"] + " N")
    ax3.plot(t_arr, arr[:, 1], lw=kw["lw"], color=kw["color"], ls=kw.get("ls", "-"), alpha=0.7)
    ax3.plot(t_arr, -arr[:, 2], lw=kw["lw"], color=kw["color"], ls=kw.get("ls", "-"), alpha=0.5)
if len(gnss_vel):
    ax3.plot(gnss_t, gnss_vel[:, 0], color="green", lw=1, ls="--", alpha=0.7, label="GNSS N")
    ax3.plot(gnss_t, gnss_vel[:, 1], color="green", lw=1, ls="--", alpha=0.5)
    ax3.plot(gnss_t, -gnss_vel[:, 2], color="green", lw=1, ls="--", alpha=0.3)
if len(fc_vel):
    ax3.plot(fc_t[:len(fc_vel)], fc_vel[:, 0], color="gray", lw=1, alpha=0.7, label="FC N")
    ax3.plot(fc_t[:len(fc_vel)], fc_vel[:, 1], color="gray", lw=1, alpha=0.5)
    ax3.plot(fc_t[:len(fc_vel)], -fc_vel[:, 2], color="gray", lw=1, alpha=0.3)
ax3.set_ylabel("Velocity [m/s]"); ax3.set_title("Velocity vs Time"); ax3.legend(fontsize=7, ncol=2); ax3.grid(True)

# --- Speed ---
ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(t_on,  spd_on,  **kw_on)
ax4.plot(t_off, spd_off, **kw_off)
if len(fc_speed):
    ax4.plot(fc_t[:len(fc_speed)], fc_speed, color="gray", lw=1, alpha=0.7, label="FC")
ax4.set_ylabel("Speed [m/s]"); ax4.set_title("Total Speed"); ax4.legend(fontsize=8); ax4.grid(True)

# --- Roll & Yaw ---
ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(t_on,  roll_on,  color="tomato",    lw=1.2, label="EqF+mag Roll")
ax5.plot(t_on,  yaw_on,   color="tomato",    lw=1.2, ls="-.", label="EqF+mag Yaw", alpha=0.7)
ax5.plot(t_off, roll_off, color="steelblue", lw=1.2, ls="--", label="no-mag Roll")
ax5.plot(t_off, yaw_off,  color="steelblue", lw=1.2, ls=":",  label="no-mag Yaw", alpha=0.7)
if len(fc_att):
    ax5.plot(fc_att_t, fc_roll, color="gray", lw=1, alpha=0.7, label="FC Roll")
    ax5.plot(fc_att_t, fc_yaw,  color="gray", lw=1, ls="--", alpha=0.7, label="FC Yaw")
ax5.set_xlabel("Time [s]"); ax5.set_ylabel("Angle [deg]")
ax5.set_title("Roll & Yaw"); ax5.legend(fontsize=7, ncol=2); ax5.grid(True)

# --- Pitch ---
ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(t_on,  pitch_on,  **kw_on)
ax6.plot(t_off, pitch_off, **kw_off)
if len(fc_att):
    ax6.plot(fc_att_t, fc_pitch, color="gray", lw=1, alpha=0.7, label="FC")
ax6.set_xlabel("Time [s]"); ax6.set_ylabel("Angle [deg]")
ax6.set_title("Pitch"); ax6.legend(fontsize=8); ax6.grid(True)

plt.tight_layout(rect=(0, 0, 1, 0.97))
out = "outputs/mag_effect.png"
plt.savefig(out, dpi=150)
print(f"\nSaved to {out}")
plt.show()
