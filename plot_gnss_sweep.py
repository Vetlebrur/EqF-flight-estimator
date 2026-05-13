"""Compare EqF outputs across GNSS frequency sweep (outputs/gnss_freq_sweep/).

Plots the same quantities as plot_trajectory.py but overlays every sweep run
on each subplot so the effect of GNSS update rate is immediately visible.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial.transform import Rotation

# =============================================================================
# Configuration
# =============================================================================

SWEEP_DIR   = Path("outputs/gnss_freq_sweep")
FC_CSV      = Path("data/20241011_NIMBUS24_Flight_FC_Data.csv")
OUTPUT_FILE = Path("outputs/gnss_sweep_comparison.png")

_C = {
    "t": 0, "lon": 1, "lat": 2, "alt": 3,
    "gps_vn": 4, "gps_ve": 5, "gps_vd": 6,
    "ax": 9, "ay": 10, "az": 11,
    "gx": 15, "gy": 16, "gz": 17,
    "pn": 36, "pe": 37, "pd": 38,
    "vn": 39, "ve": 40, "vd": 41,
    "roll": 29, "pitch": 30, "yaw": 31,
}
R_EARTH = 6_378_137.0


# =============================================================================
# Helpers
# =============================================================================

def _freq_from_path(p: Path) -> float:
    """Parse freq from filename like tg_eqf_output_0p05Hz.csv -> 0.05."""
    m = re.search(r"_([\dp]+)Hz\.csv$", p.name)
    if m:
        return float(m.group(1).replace("p", "."))
    return float("nan")


def gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    north = dlat * R_EARTH
    east  = dlon * np.cos(np.radians(lat0)) * R_EARTH
    down  = -(alt - alt0)
    return np.array([north, east, down])


def load_sweep(path: Path):
    """Load one EqF output CSV. Returns dict of named arrays."""
    out = np.genfromtxt(path, delimiter=",", skip_header=1)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    valid = np.isfinite(out[:, :25]).all(axis=1)
    out = out[valid]
    if len(out) == 0:
        return None

    t  = out[:, 0]
    px, py, pz = out[:, 1], out[:, 2], out[:, 3]
    vx, vy, vz = out[:, 4], out[:, 5], out[:, 6]

    dcm = out[:, 7:16].reshape(-1, 3, 3)
    euler = Rotation.from_matrix(dcm).as_euler("ZYX", degrees=True)
    yaw_d   = np.unwrap(euler[:, 0], period=360)
    pitch_d = np.unwrap(euler[:, 1], period=360)
    roll_d  = np.unwrap(euler[:, 2], period=360)

    return dict(t=t, px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz,
                roll=roll_d, pitch=pitch_d, yaw=yaw_d)


# =============================================================================
# Load reference data (GNSS + FC)
# =============================================================================

raw = np.genfromtxt(FC_CSV, delimiter=",", skip_header=1)
if raw.ndim == 1:
    raw = raw.reshape(1, -1)

valid_gps = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
first = int(np.argmax(valid_gps))
lat0 = raw[first, _C["lat"]]
lon0 = raw[first, _C["lon"]]
alt0 = raw[first, _C["alt"]] / 1000.0

gnss_pos, gnss_vel, gnss_t = [], [], []
fc_pos, fc_vel, fc_att, fc_att_t, fc_t = [], [], [], [], []
fc_gyro, fc_gyro_t = [], []

for row in raw:
    t = row[_C["t"]]
    if not np.isfinite(t):
        continue

    lat, lon = row[_C["lat"]], row[_C["lon"]]
    if lat != 0 and lon != 0:
        alt = row[_C["alt"]] / 1000.0
        gnss_pos.append(gps_to_ned(lat, lon, alt, lat0, lon0, alt0))
        gnss_vel.append(np.array([row[_C["gps_vn"]], row[_C["gps_ve"]], row[_C["gps_vd"]]]) / 1000.0)
        gnss_t.append(t)

    pn, pe, pd = row[_C["pn"]], row[_C["pe"]], row[_C["pd"]]
    if np.isfinite(pn) and np.isfinite(pe) and np.isfinite(pd):
        fc_pos.append([pn, pe, pd])
        vn, ve, vd = row[_C["vn"]], row[_C["ve"]], row[_C["vd"]]
        if np.isfinite(vn) and np.isfinite(ve) and np.isfinite(vd):
            fc_vel.append([vn, ve, vd])
        r, p, y = row[_C["roll"]], row[_C["pitch"]], row[_C["yaw"]]
        if np.isfinite(r) and np.isfinite(p) and np.isfinite(y):
            fc_att.append([r, p, y])
            fc_att_t.append(t)
        gx, gy, gz = row[_C["gx"]], row[_C["gy"]], row[_C["gz"]]
        if np.isfinite(gx) and np.isfinite(gy) and np.isfinite(gz):
            fc_gyro.append(np.array([gx, gy, gz]) * (np.pi / 180.0))
            fc_gyro_t.append(t)
        fc_t.append(t)

gnss_pos  = np.array(gnss_pos)  if gnss_pos  else np.empty((0, 3))
gnss_vel  = np.array(gnss_vel)  if gnss_vel  else np.empty((0, 3))
gnss_t    = np.array(gnss_t)
fc_pos    = np.array(fc_pos)    if fc_pos    else np.empty((0, 3))
fc_vel    = np.array(fc_vel)    if fc_vel    else np.empty((0, 3))
fc_att    = np.array(fc_att)    if fc_att    else np.empty((0, 3))
fc_att_t  = np.array(fc_att_t)
fc_gyro   = np.array(fc_gyro)   if fc_gyro   else np.empty((0, 3))
fc_gyro_t = np.array(fc_gyro_t)
fc_t      = np.array(fc_t)

if len(fc_att) > 0:
    for i in range(3):
        fc_att[:, i] = np.unwrap(fc_att[:, i], discont=np.pi)

# =============================================================================
# Load sweep runs
# =============================================================================

csv_files = sorted(SWEEP_DIR.glob("tg_eqf_output_*.csv"), key=_freq_from_path)
if not csv_files:
    print(f"No sweep CSVs found in {SWEEP_DIR}. Run sweep_gnss_freq.py first.")
    exit(1)

runs = []
for path in csv_files:
    freq = _freq_from_path(path)
    data = load_sweep(path)
    if data is not None:
        runs.append((freq, data))
        print(f"Loaded {path.name}  ({freq} Hz,  {len(data['t'])} rows)")

print(f"\n{len(runs)} runs loaded.")

# Select 3 representative runs (first, middle, last)
_n = len(runs)
_sel = [0, _n // 2, _n - 1] if _n >= 3 else list(range(_n))
plot_runs = [runs[i] for i in _sel]

# Colormap spread across the full range so colours contrast well
cmap        = cm.get_cmap("plasma", _n)
plot_colors = [cmap(i) for i in _sel]

# =============================================================================
# Plot
# =============================================================================

fig = plt.figure(figsize=(18, 20))
gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.3)
axes = np.empty((4, 3), dtype=object)
for r in range(3):
    for c in range(3):
        axes[r, c] = fig.add_subplot(gs[r, c])
axes[3, 0] = fig.add_subplot(gs[3, :])   # position error spans full width
fig.suptitle("EqF GNSS Frequency Sweep Comparison", fontsize=14, fontweight="bold")

def ref_style(ax):
    ax.grid(True, alpha=0.3)

def add_gnss_ref(ax, y_gnss, label="GNSS"):
    if len(gnss_t) > 0:
        ax.plot(gnss_t, y_gnss, "k--", lw=1.2, alpha=0.5, label=label, zorder=0)

def add_fc_ref(ax, y_fc, label="FC"):
    if len(fc_t) > 0:
        ax.plot(fc_t, y_fc, color="gray", lw=1.0, alpha=0.5, ls=":", label=label, zorder=0)

# --- Row 0: Position North / East / Altitude ---
ax = axes[0, 0]
add_gnss_ref(ax, gnss_pos[:, 0], "GNSS North")
add_fc_ref(ax, fc_pos[:, 0],     "FC North")
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], d["px"], color=c, lw=1, label=f"{freq} Hz")
ax.set_title("Position North [m]"); ax.set_xlabel("Time [s]"); ref_style(ax)

ax = axes[0, 1]
add_gnss_ref(ax, gnss_pos[:, 1], "GNSS East")
add_fc_ref(ax, fc_pos[:, 1],     "FC East")
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], d["py"], color=c, lw=1, label=f"{freq} Hz")
ax.set_title("Position East [m]"); ax.set_xlabel("Time [s]"); ref_style(ax)

ax = axes[0, 2]
add_gnss_ref(ax, -gnss_pos[:, 2], "GNSS Alt")
add_fc_ref(ax,   -fc_pos[:, 2],   "FC Alt")
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], -d["pz"], color=c, lw=1, label=f"{freq} Hz")
ax.set_title("Altitude [m]"); ax.set_xlabel("Time [s]"); ref_style(ax)

# --- Row 1: Velocity North / East / Up + Speed ---
ax = axes[1, 0]
add_gnss_ref(ax, gnss_vel[:, 0], "GNSS Vn")
add_fc_ref(ax, fc_vel[:, 0],     "FC Vn")
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], d["vx"], color=c, lw=1)
ax.set_title("Velocity North [m/s]"); ax.set_xlabel("Time [s]"); ref_style(ax)

ax = axes[1, 1]
add_gnss_ref(ax, gnss_vel[:, 1], "GNSS Ve")
add_fc_ref(ax, fc_vel[:, 1],     "FC Ve")
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], d["vy"], color=c, lw=1)
ax.set_title("Velocity East [m/s]"); ax.set_xlabel("Time [s]"); ref_style(ax)

ax = axes[1, 2]
add_gnss_ref(ax, -gnss_vel[:, 2], "GNSS Vu")
add_fc_ref(ax, -fc_vel[:, 2],     "FC Vu")
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], -d["vz"], color=c, lw=1)
ax.set_title("Velocity Up [m/s]"); ax.set_xlabel("Time [s]"); ref_style(ax)

# --- Row 2: Attitude Roll / Pitch / Yaw ---
ax = axes[2, 0]
if len(fc_att) > 0:
    ax.plot(fc_att_t, np.degrees(fc_att[:, 0]), "k--", lw=1, alpha=0.5, label="FC", zorder=0)
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], d["roll"], color=c, lw=1)
ax.set_title("Roll [deg]"); ax.set_xlabel("Time [s]"); ref_style(ax)

ax = axes[2, 1]
if len(fc_att) > 0:
    ax.plot(fc_att_t, np.degrees(fc_att[:, 1]), "k--", lw=1, alpha=0.5, label="FC", zorder=0)
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], d["pitch"], color=c, lw=1)
ax.set_title("Pitch [deg]"); ax.set_xlabel("Time [s]"); ref_style(ax)

ax = axes[2, 2]
if len(fc_att) > 0:
    ax.plot(fc_att_t, np.degrees(fc_att[:, 2]), "k--", lw=1, alpha=0.5, label="FC", zorder=0)
for (freq, d), c in zip(plot_runs, plot_colors):
    ax.plot(d["t"], d["yaw"], color=c, lw=1)
ax.set_title("Yaw [deg]"); ax.set_xlabel("Time [s]"); ref_style(ax)

# --- Row 3: Position error vs GNSS (full width) ---
ax = axes[3, 0]
if len(gnss_pos) > 0:
    for (freq, d), c in zip(plot_runs, plot_colors):
        p_i = np.column_stack([np.interp(gnss_t, d["t"], d[k]) for k in ("px", "py", "pz")])
        err = np.linalg.norm(p_i - gnss_pos, axis=1)
        ax.plot(gnss_t, err, color=c, lw=1, label=f"{freq} Hz")
ax.set_yscale("log"); ax.set_title("Position Error vs GNSS [m, log]"); ax.set_xlabel("Time [s]"); ref_style(ax)

# --- Shared legend ---
handles = [plt.Line2D([0], [0], color=c, lw=2, label=f"{freq} Hz")
           for (freq, _), c in zip(plot_runs, plot_colors)]
handles += [plt.Line2D([0], [0], color="k",    lw=1.2, ls="--", label="GNSS ref"),
            plt.Line2D([0], [0], color="gray",  lw=1.0, ls=":",  label="FC ref")]
fig.legend(handles=handles, loc="lower center", ncol=len(handles),
           fontsize=8, bbox_to_anchor=(0.5, 0.0))

fig.subplots_adjust(top=0.95, bottom=0.07)
plt.savefig(OUTPUT_FILE, dpi=150)
print(f"\nSaved to {OUTPUT_FILE}")
plt.show()
