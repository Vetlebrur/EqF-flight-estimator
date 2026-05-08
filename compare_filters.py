"""Compare EqF and EKF filter outputs against FC and GNSS reference."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# =============================================================================
# Configuration — match DATASET in eqf_filter.py and ekf_filter.py
# =============================================================================

DATASET = "30s"

_DATASETS = {
    "full":    ("data/20241011_NIMBUS24_Flight_FC_Data.csv",
                "outputs/tg_eqf_output_full.csv",
                "outputs/ekf_output_full.csv",
                "FULL FLIGHT"),
    "30s":     ("data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",
                "outputs/tg_eqf_output_30s.csv",
                "outputs/ekf_output_30s.csv",
                "FLIGHT (first 30s)"),
    "1s_loop": ("data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",
                "outputs/tg_eqf_output_1s_loop.csv",
                "outputs/ekf_output_1s_loop.csv",
                "FLIGHT (1s loop)"),
}

if DATASET not in _DATASETS:
    raise ValueError(f"Unknown DATASET {DATASET!r}. Choose from: {list(_DATASETS)}")

input_csv, eqf_csv, ekf_csv, data_type = _DATASETS[DATASET]

# =============================================================================
# Constants
# =============================================================================

R_EARTH = 6_378_137.0

_C = {
    "t":      0,
    "lon":    1,
    "lat":    2,
    "alt":    3,
    "gps_vn": 4,
    "gps_ve": 5,
    "gps_vd": 6,
    "roll":   29,
    "pitch":  30,
    "yaw":    31,
    "pn":     36,
    "pe":     37,
    "pd":     38,
    "vn":     39,
    "ve":     40,
    "vd":     41,
}


def gps_to_ned(lat: float, lon: float, alt: float,
               lat0: float, lon0: float, alt0: float) -> np.ndarray:
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    return np.array([
        dlat * R_EARTH,
        dlon * np.cos(np.radians(lat0)) * R_EARTH,
        -(alt - alt0),
    ])


def load_filter_output(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load a filter output CSV (EqF or EKF format).

    Returns (t, pos, vel, q_wxyz, bias_gyro) or None if file missing.
    Both filters share: t=0, pos=1:4, vel=4:7, R=7:16, q=16:20, bgxyz=20:23.
    """
    try:
        out = np.genfromtxt(path, delimiter=",", skip_header=1)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        t    = out[:, 0]
        pos  = out[:, 1:4]
        vel  = out[:, 4:7]
        q    = out[:, 16:20]   # [w,x,y,z]
        bg   = out[:, 20:23]   # gyro bias
        return t, pos, vel, q, bg
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None


def quat_to_euler_deg(q_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert [w,x,y,z] quaternion array to unwrapped ZYX Euler in degrees."""
    rot = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])  # scipy wants [x,y,z,w]
    euler = rot.as_euler('ZYX', degrees=True)           # [yaw, pitch, roll]
    yaw   = np.unwrap(euler[:, 0], discont=180)
    pitch = np.unwrap(euler[:, 1], discont=180)
    roll  = np.unwrap(euler[:, 2], discont=180)
    return roll, pitch, yaw


def angular_error_deg(q_filt: np.ndarray, q_fc: np.ndarray) -> np.ndarray:
    """Quaternion angular error in degrees between two [w,x,y,z] arrays."""
    dot = np.clip(np.abs(np.sum(q_filt * q_fc, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


# =============================================================================
# Load FC / GNSS reference data
# =============================================================================

raw = np.genfromtxt(input_csv, delimiter=",", skip_header=1)
if raw.ndim == 1:
    raw = raw.reshape(1, -1)

valid_gps = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
first = int(np.argmax(valid_gps))
lat0 = raw[first, _C["lat"]]
lon0 = raw[first, _C["lon"]]
alt0 = raw[first, _C["alt"]] / 1000.0

gnss_pos, gnss_vel, gnss_t = [], [], []
fc_pos, fc_vel, fc_att, fc_att_t, fc_t = [], [], [], [], []

for row in raw:
    t = row[_C["t"]]
    if not np.isfinite(t):
        continue

    lat, lon = row[_C["lat"]], row[_C["lon"]]
    if lat != 0 and lon != 0:
        alt = row[_C["alt"]] / 1000.0
        gnss_pos.append(gps_to_ned(lat, lon, alt, lat0, lon0, alt0))
        gnss_vel.append(row[[_C["gps_vn"], _C["gps_ve"], _C["gps_vd"]]] / 1000.0)
        gnss_t.append(t)

    pn, pe, pd = row[_C["pn"]], row[_C["pe"]], row[_C["pd"]]
    if np.isfinite(pn) and np.isfinite(pe) and np.isfinite(pd):
        fc_pos.append([pn, pe, pd])
        vn, ve, vd = row[_C["vn"]], row[_C["ve"]], row[_C["vd"]]
        if np.isfinite(vn) and np.isfinite(ve) and np.isfinite(vd):
            fc_vel.append([vn, ve, vd])
        roll_fc, pitch_fc, yaw_fc = row[_C["roll"]], row[_C["pitch"]], row[_C["yaw"]]
        if np.isfinite(roll_fc) and np.isfinite(pitch_fc) and np.isfinite(yaw_fc):
            fc_att.append([roll_fc, pitch_fc, yaw_fc])
            fc_att_t.append(t)
        fc_t.append(t)

gnss_pos  = np.array(gnss_pos)  if gnss_pos  else np.empty((0, 3))
gnss_vel  = np.array(gnss_vel)  if gnss_vel  else np.empty((0, 3))
gnss_t    = np.array(gnss_t)
fc_pos    = np.array(fc_pos)    if fc_pos    else np.empty((0, 3))
fc_vel    = np.array(fc_vel)    if fc_vel    else np.empty((0, 3))
fc_att    = np.array(fc_att)    if fc_att    else np.empty((0, 3))
fc_att_t  = np.array(fc_att_t)
fc_t      = np.array(fc_t)

# Build FC quaternion array for angular error computation
fc_q_wxyz: np.ndarray = np.empty((0, 4))
if len(fc_att) > 0:
    fc_rot   = Rotation.from_euler('ZYX', fc_att[:, [2, 1, 0]])  # [yaw, pitch, roll]
    fc_q_xyzw = fc_rot.as_quat()
    fc_q_wxyz = fc_q_xyzw[:, [3, 0, 1, 2]]

# =============================================================================
# Load filter outputs
# =============================================================================

eqf = load_filter_output(eqf_csv)
ekf = load_filter_output(ekf_csv)

if eqf is None and ekf is None:
    print("No filter outputs found. Run filters first.")
    exit(1)

have_eqf = eqf is not None
have_ekf = ekf is not None

_empty3 = np.empty((0, 3))
_empty4 = np.empty((0, 4))
_zero   = np.zeros(0)

eqf_t:     np.ndarray = _zero;   eqf_pos = _empty3; eqf_vel = _empty3
eqf_q:     np.ndarray = _empty4; eqf_bg  = _empty3
eqf_roll:  np.ndarray = _zero;   eqf_pitch = _zero; eqf_yaw = _zero

ekf_t:     np.ndarray = _zero;   ekf_pos = _empty3; ekf_vel = _empty3
ekf_q:     np.ndarray = _empty4; ekf_bg  = _empty3
ekf_roll:  np.ndarray = _zero;   ekf_pitch = _zero; ekf_yaw = _zero

if have_eqf:
    eqf_t, eqf_pos, eqf_vel, eqf_q, eqf_bg = eqf  # type: ignore[misc]
    eqf_roll, eqf_pitch, eqf_yaw = quat_to_euler_deg(eqf_q)

if have_ekf:
    ekf_t, ekf_pos, ekf_vel, ekf_q, ekf_bg = ekf  # type: ignore[misc]
    ekf_roll, ekf_pitch, ekf_yaw = quat_to_euler_deg(ekf_q)

# =============================================================================
# Plotting
# =============================================================================

fig = plt.figure(figsize=(16, 14))
fig.suptitle(f'Filter Comparison — {data_type}', fontsize=14, fontweight='bold')

# --- 3D Trajectory ---
ax = fig.add_subplot(3, 3, 1, projection='3d')
if have_eqf:
    ax.plot(eqf_pos[:, 0], eqf_pos[:, 1], eqf_pos[:, 2], 'b-', lw=1, label='EqF', alpha=0.8)
if have_ekf:
    ax.plot(ekf_pos[:, 0], ekf_pos[:, 1], ekf_pos[:, 2], 'r-', lw=1, label='EKF', alpha=0.8)
if len(gnss_pos) > 0:
    ax.plot(gnss_pos[:, 0], gnss_pos[:, 1], gnss_pos[:, 2], 'g--', lw=1, label='GNSS', alpha=0.5)
if len(fc_pos) > 0:
    ax.plot(fc_pos[:, 0], fc_pos[:, 1], fc_pos[:, 2], 'm:', lw=1.5, label='FC', alpha=0.6)
ax.set_xlabel('North [m]'); ax.set_ylabel('East [m]'); ax.set_zlabel('Down [m]')
ax.set_title('3D Trajectory'); ax.legend(fontsize=7); ax.grid(True)

# --- Position North ---
ax = fig.add_subplot(3, 3, 2)
if have_eqf:  ax.plot(eqf_t, eqf_pos[:, 0], 'b-', lw=1, label='EqF')
if have_ekf:  ax.plot(ekf_t, ekf_pos[:, 0], 'r-', lw=1, label='EKF')
if len(gnss_pos) > 0: ax.plot(gnss_t, gnss_pos[:, 0], 'g--', lw=1, label='GNSS', alpha=0.5)
if len(fc_pos) > 0:   ax.plot(fc_t,   fc_pos[:, 0],   'm:',  lw=1.5, label='FC', alpha=0.6)
ax.set_ylabel('Position North [m]'); ax.set_title('Position — North')
ax.legend(fontsize=7); ax.grid(True)

# --- Position East ---
ax = fig.add_subplot(3, 3, 3)
if have_eqf:  ax.plot(eqf_t, eqf_pos[:, 1], 'b-', lw=1, label='EqF')
if have_ekf:  ax.plot(ekf_t, ekf_pos[:, 1], 'r-', lw=1, label='EKF')
if len(gnss_pos) > 0: ax.plot(gnss_t, gnss_pos[:, 1], 'g--', lw=1, label='GNSS', alpha=0.5)
if len(fc_pos) > 0:   ax.plot(fc_t,   fc_pos[:, 1],   'm:',  lw=1.5, label='FC', alpha=0.6)
ax.set_ylabel('Position East [m]'); ax.set_title('Position — East')
ax.legend(fontsize=7); ax.grid(True)

# --- Velocity North ---
ax = fig.add_subplot(3, 3, 4)
if have_eqf:  ax.plot(eqf_t, eqf_vel[:, 0], 'b-', lw=1, label='EqF')
if have_ekf:  ax.plot(ekf_t, ekf_vel[:, 0], 'r-', lw=1, label='EKF')
if len(gnss_vel) > 0: ax.plot(gnss_t, gnss_vel[:, 0], 'g--', lw=1, label='GNSS', alpha=0.5)
if len(fc_vel) > 0:   ax.plot(fc_t[:len(fc_vel)], fc_vel[:, 0], 'm:', lw=1.5, label='FC', alpha=0.6)
ax.set_ylabel('Velocity North [m/s]'); ax.set_title('Velocity — North')
ax.legend(fontsize=7); ax.grid(True)

# --- Velocity East ---
ax = fig.add_subplot(3, 3, 5)
if have_eqf:  ax.plot(eqf_t, eqf_vel[:, 1], 'b-', lw=1, label='EqF')
if have_ekf:  ax.plot(ekf_t, ekf_vel[:, 1], 'r-', lw=1, label='EKF')
if len(gnss_vel) > 0: ax.plot(gnss_t, gnss_vel[:, 1], 'g--', lw=1, label='GNSS', alpha=0.5)
if len(fc_vel) > 0:   ax.plot(fc_t[:len(fc_vel)], fc_vel[:, 1], 'm:', lw=1.5, label='FC', alpha=0.6)
ax.set_ylabel('Velocity East [m/s]'); ax.set_title('Velocity — East')
ax.legend(fontsize=7); ax.grid(True)

# --- Velocity Down ---
ax = fig.add_subplot(3, 3, 6)
if have_eqf:  ax.plot(eqf_t, eqf_vel[:, 2], 'b-', lw=1, label='EqF')
if have_ekf:  ax.plot(ekf_t, ekf_vel[:, 2], 'r-', lw=1, label='EKF')
if len(gnss_vel) > 0: ax.plot(gnss_t, gnss_vel[:, 2], 'g--', lw=1, label='GNSS', alpha=0.5)
if len(fc_vel) > 0:   ax.plot(fc_t[:len(fc_vel)], fc_vel[:, 2], 'm:', lw=1.5, label='FC', alpha=0.6)
ax.set_ylabel('Velocity Down [m/s]'); ax.set_title('Velocity — Down')
ax.legend(fontsize=7); ax.grid(True)

# --- Attitude: Roll ---
ax = fig.add_subplot(3, 3, 7)
if have_eqf: ax.plot(eqf_t, eqf_roll, 'b-', lw=1, label='EqF')
if have_ekf: ax.plot(ekf_t, ekf_roll, 'r-', lw=1, label='EKF')
if len(fc_att) > 0:
    ax.plot(fc_att_t, np.degrees(fc_att[:, 0]), 'm--', lw=1, alpha=0.7, label='FC')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Roll [deg]'); ax.set_title('Attitude — Roll')
ax.legend(fontsize=7); ax.grid(True)

# --- Attitude: Pitch ---
ax = fig.add_subplot(3, 3, 8)
if have_eqf: ax.plot(eqf_t, eqf_pitch, 'b-', lw=1, label='EqF')
if have_ekf: ax.plot(ekf_t, ekf_pitch, 'r-', lw=1, label='EKF')
if len(fc_att) > 0:
    ax.plot(fc_att_t, np.degrees(fc_att[:, 1]), 'm--', lw=1, alpha=0.7, label='FC')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Pitch [deg]'); ax.set_title('Attitude — Pitch')
ax.legend(fontsize=7); ax.grid(True)

# --- Angular Error vs FC ---
ax = fig.add_subplot(3, 3, 9)
if len(fc_att) > 0 and len(fc_q_wxyz) > 0:
    def interp_q(t_filt: np.ndarray, q_filt: np.ndarray) -> np.ndarray:
        """Interpolate filter quaternion to FC attitude timestamps."""
        return np.column_stack([
            np.interp(fc_att_t, t_filt, q_filt[:, i]) for i in range(4)
        ])

    if have_eqf:
        eq = interp_q(eqf_t, eqf_q)
        norm = np.linalg.norm(eq, axis=1, keepdims=True)
        eq /= np.where(norm > 0, norm, 1.0)
        ax.plot(fc_att_t, angular_error_deg(eq, fc_q_wxyz), 'b-', lw=1, label='EqF error')
    if have_ekf:
        ek = interp_q(ekf_t, ekf_q)
        norm = np.linalg.norm(ek, axis=1, keepdims=True)
        ek /= np.where(norm > 0, norm, 1.0)
        ax.plot(fc_att_t, angular_error_deg(ek, fc_q_wxyz), 'r-', lw=1, label='EKF error')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Angular error [deg]')
ax.set_title('Attitude Angular Error vs FC')
ax.legend(fontsize=7); ax.grid(True)

plt.tight_layout(rect=(0, 0, 1, 0.97))
out_file = f'outputs/filter_comparison_{DATASET}.png'
plt.savefig(out_file, dpi=150)
print(f"Saved comparison plot to {out_file}")

# =============================================================================
# Statistics
# =============================================================================

print("\n" + "="*70)
print(f"FILTER COMPARISON — {data_type}")
print("="*70)

if have_eqf:
    speed = np.sqrt(np.sum(eqf_vel**2, axis=1))
    print(f"\nEQF:")
    print(f"  Final pos: [{eqf_pos[-1,0]:.1f}, {eqf_pos[-1,1]:.1f}, {eqf_pos[-1,2]:.1f}] m")
    print(f"  Final vel: [{eqf_vel[-1,0]:.1f}, {eqf_vel[-1,1]:.1f}, {eqf_vel[-1,2]:.1f}] m/s")
    print(f"  Final att (ZYX): roll={eqf_roll[-1]:.1f}°  pitch={eqf_pitch[-1]:.1f}°  yaw={eqf_yaw[-1]:.1f}°")
    print(f"  Max speed: {np.max(speed):.1f} m/s")
    if len(gnss_pos) > 0:
        p_i = np.column_stack([np.interp(gnss_t, eqf_t, eqf_pos[:, i]) for i in range(3)])
        err = np.linalg.norm(p_i - gnss_pos, axis=1)
        print(f"  Pos error vs GNSS — mean: {np.mean(err):.1f} m  max: {np.max(err):.1f} m")

if have_ekf:
    speed = np.sqrt(np.sum(ekf_vel**2, axis=1))
    print(f"\nEKF:")
    print(f"  Final pos: [{ekf_pos[-1,0]:.1f}, {ekf_pos[-1,1]:.1f}, {ekf_pos[-1,2]:.1f}] m")
    print(f"  Final vel: [{ekf_vel[-1,0]:.1f}, {ekf_vel[-1,1]:.1f}, {ekf_vel[-1,2]:.1f}] m/s")
    print(f"  Final att (ZYX): roll={ekf_roll[-1]:.1f}°  pitch={ekf_pitch[-1]:.1f}°  yaw={ekf_yaw[-1]:.1f}°")
    print(f"  Max speed: {np.max(speed):.1f} m/s")
    if len(gnss_pos) > 0:
        p_i = np.column_stack([np.interp(gnss_t, ekf_t, ekf_pos[:, i]) for i in range(3)])
        err = np.linalg.norm(p_i - gnss_pos, axis=1)
        print(f"  Pos error vs GNSS — mean: {np.mean(err):.1f} m  max: {np.max(err):.1f} m")

if len(gnss_pos) > 0:
    print(f"\nGNSS ref:")
    print(f"  Final pos: [{gnss_pos[-1,0]:.1f}, {gnss_pos[-1,1]:.1f}, {gnss_pos[-1,2]:.1f}] m")

plt.show()
