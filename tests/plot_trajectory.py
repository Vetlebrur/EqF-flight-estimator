"""Plot the trajectory from EqF filter output."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def safe_normalize(vec: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize vector with guard against zero-norm."""
    norm = np.linalg.norm(vec)
    if norm < epsilon:
        return vec / epsilon  # Return small vector instead of dividing by zero
    return vec / norm


# =============================================================================
# Configuration
# =============================================================================
# "full"    -> data/20241011_NIMBUS24_Flight_FC_Data.csv          (complete flight)
# "30s"     -> data/20241011_NIMBUS24_Flight_FC_Data_30s.csv      (first 30 s)
# "1s_loop" -> data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv  (first 1 s looped for 30 s)
DATASET = "30s"

# =============================================================================
# Constants
# =============================================================================
R_EARTH = 6_378_137.0
_C = {
    "t": 0,
    "lon": 1,
    "lat": 2,
    "alt": 3,
    "gps_vn": 4,
    "gps_ve": 5,
    "gps_vd": 6,
    "ax": 9,       # FC accel X (g)
    "ay": 10,      # FC accel Y (g)
    "az": 11,      # FC accel Z (g)
    "pn": 36,      # FC position North (m)
    "pe": 37,      # FC position East (m)
    "pd": 38,      # FC position Down (m)
    "vn": 40,      # FC velocity North (m/s)
    "ve": 41,      # FC velocity East (m/s)
    "vd": 42,      # FC velocity Down (m/s)
    "roll": 29,    # FC roll (rad)
    "pitch": 30,   # FC pitch (rad)
    "yaw": 31,     # FC yaw (rad)
}

def gps_to_ned(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)  # Positive down (negative altitude difference)
    return np.array([north, east, down])

# Load CSV data based on configuration
_datasets = {
    "full":    ("data/20241011_NIMBUS24_Flight_FC_Data.csv",        "outputs/tg_eqf_output_full.csv",     "FULL FLIGHT"),
    "30s":     ("data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",     "outputs/tg_eqf_output_30s.csv",      "FLIGHT (first 30s)"),
    "1s_loop": ("data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv", "outputs/tg_eqf_output_1s_loop.csv",  "FLIGHT (1s loop)"),
}
if DATASET not in _datasets:
    raise ValueError(f"Unknown DATASET {DATASET!r}. Choose from: {list(_datasets)}")
input_csv, output_csv, data_type = _datasets[DATASET]
print(f"Loading {data_type} data for plotting")

# Verify output file exists
if not Path(output_csv).exists():
    print(f"Error: Output file not found: {output_csv}")
    print(f"Make sure to run eqf_filter.py with DATASET={DATASET!r}")
    exit(1)

raw = np.genfromtxt(input_csv, delimiter=",", skip_header=1)
if raw.ndim == 1:
    raw = raw.reshape(1, -1)

# Find first valid GPS fix for reference
valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)

first = np.argmax(valid)
lat0 = raw[first, _C["lat"]]
lon0 = raw[first, _C["lon"]]
alt0 = raw[first, _C["alt"]] / 1000.0
print(f"Using first valid GPS fix as reference: lat={lat0:.6f}, lon={lon0:.6f}, alt={alt0:.1f}m")

# Extract GNSS and FC data
gnss_pos = []
gnss_vel = []
fc_pos = []
fc_vel = []
fc_att = []  # FC attitude (roll, pitch, yaw)
fc_accel = []  # FC acceleration
gnss_t = []
fc_t = []
fc_att_t = []  # Time for attitude data
fc_accel_t = []  # Time for acceleration data

for i, row in enumerate(raw):
    t = row[_C["t"]]
    if not np.isfinite(t):
        continue

    # GNSS data
    lat = row[_C["lat"]]
    lon = row[_C["lon"]]

    if lat != 0 and lon != 0:
        alt = row[_C["alt"]] / 1000.0
        pos_NED = gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
        vel_NED = np.array([row[_C["gps_vn"]], row[_C["gps_ve"]], row[_C["gps_vd"]]]) / 1000.0

        gnss_pos.append(pos_NED)
        gnss_vel.append(vel_NED)
        gnss_t.append(t)

    # FC data (pn, pe, pd in meters)
    pn = row[_C["pn"]]
    pe = row[_C["pe"]]
    pd = row[_C["pd"]]

    if np.isfinite(pn) and np.isfinite(pe) and np.isfinite(pd):
        fc_pos.append(np.array([pn, pe, pd]))
        vn = row[_C["vn"]]
        ve = row[_C["ve"]]
        vd = row[_C["vd"]]
        if np.isfinite(vn) and np.isfinite(ve) and np.isfinite(vd):
            fc_vel.append(np.array([vn, ve, vd]))

        # FC attitude (roll, pitch, yaw in radians)
        roll_fc = row[_C["roll"]]
        pitch_fc = row[_C["pitch"]]
        yaw_fc = row[_C["yaw"]]
        if np.isfinite(roll_fc) and np.isfinite(pitch_fc) and np.isfinite(yaw_fc):
            fc_att.append(np.array([roll_fc, pitch_fc, yaw_fc]))
            fc_att_t.append(t)

        # FC acceleration (in g)
        ax = row[_C["ax"]]
        ay = row[_C["ay"]]
        az = row[_C["az"]]
        if np.isfinite(ax) and np.isfinite(ay) and np.isfinite(az):
            fc_accel.append(np.array([ax, ay, az]) * 9.81)  # Convert g to m/s²
            fc_accel_t.append(t)

        fc_t.append(t)

gnss_pos = np.array(gnss_pos) if gnss_pos else np.empty((0, 3))
gnss_vel = np.array(gnss_vel) if gnss_vel else np.empty((0, 3))
gnss_t = np.array(gnss_t)
fc_pos = np.array(fc_pos) if fc_pos else np.empty((0, 3))
fc_vel = np.array(fc_vel) if fc_vel else np.empty((0, 3))
fc_att = np.array(fc_att) if fc_att else np.empty((0, 3))
fc_att_t = np.array(fc_att_t)

# Unwrap FC attitude to match filter representation
if len(fc_att) > 0:
    fc_att[:, 0] = np.unwrap(fc_att[:, 0], discont=np.pi)
    fc_att[:, 1] = np.unwrap(fc_att[:, 1], discont=np.pi)
    fc_att[:, 2] = np.unwrap(fc_att[:, 2], discont=np.pi)

fc_accel = np.array(fc_accel) if fc_accel else np.empty((0, 3))
fc_accel_t = np.array(fc_accel_t)
fc_t = np.array(fc_t)

# Load output data
out = np.genfromtxt(output_csv, delimiter=",", skip_header=1)

# Apply NaN filtering early
valid_rows = np.isfinite(out).all(axis=1)
out = out[valid_rows]

if len(out) == 0:
    print("Error: No valid data rows in output CSV")
    exit(1)

# Extract columns
t = out[:, 0]
px = out[:, 1]
py = out[:, 2]
pz = out[:, 3]  # NED (positive down)
vx = out[:, 4]
vy = out[:, 5]
vz = out[:, 6]  # NED (positive down)

# Extract rotation matrix for validation
R_list = []
for i in range(len(out)):
    R = np.array([
        [out[i, 7], out[i, 8], out[i, 9]],
        [out[i, 10], out[i, 11], out[i, 12]],
        [out[i, 13], out[i, 14], out[i, 15]]
    ])
    R_list.append(R)

# Validate rotation matrices (det=1, R^T R = I)
invalid_count = 0
for i, rot in enumerate(R_list):
    det = np.linalg.det(rot)
    ortho_error = np.linalg.norm(rot.T @ rot - np.eye(3))
    if abs(det - 1.0) > 0.01 or ortho_error > 0.01:
        invalid_count += 1

if invalid_count > len(R_list) * 0.1:  # More than 10% invalid
    print(f"Warning: {invalid_count}/{len(R_list)} rotation matrices are invalid (det≠1 or not orthogonal)")
    print("This suggests incorrect orthonormalization in the filter upstream")

# Extract Euler angles from output (columns 16-18)
roll_arr = out[:, 16].copy()
pitch_arr = out[:, 17].copy()
yaw_arr = out[:, 18].copy()

# Unwrap to track continuous rotation across multiple 2π revolutions
roll_arr = np.unwrap(roll_arr, discont=np.pi)
pitch_arr = np.unwrap(pitch_arr, discont=np.pi)
yaw_arr = np.unwrap(yaw_arr, discont=np.pi)

# Load diagnostic data if available
diag_csv = output_csv.replace("tg_eqf_output", "tg_eqf_diagnostics")
diag_data = None
try:
    diag_data = np.genfromtxt(diag_csv, delimiter=',', skip_header=1,
                             dtype=[('time', 'f8'), ('type', 'U10'), ('anis', 'f8'), ('anees', 'f8')],
                             filling_values=np.nan)
    print(f"Loaded diagnostic data from {diag_csv}")
except Exception as e:
    print(f"Could not load diagnostic data: {e}")

# Create main trajectory figure (5 rows for trajectory, biases, etc.)
fig = plt.figure(figsize=(16, 20))

# 3D trajectory
ax1 = fig.add_subplot(5, 2, 1, projection='3d')
ax1.plot(px, py, pz, 'b-', linewidth=1, label='Filter Estimate')
if len(gnss_pos) > 0:
    ax1.plot(gnss_pos[:, 0], gnss_pos[:, 1], gnss_pos[:, 2], 'r--', linewidth=1, label='GNSS')
if len(fc_pos) > 0:
    ax1.plot(fc_pos[:, 0], fc_pos[:, 1], fc_pos[:, 2], 'g:', linewidth=1.5, label='FC Estimate')
ax1.scatter(px[0], py[0], pz[0], c='blue', s=100, marker='o')
ax1.scatter(px[-1], py[-1], pz[-1], c='blue', s=100, marker='s')
if len(gnss_pos) > 0:
    ax1.scatter(gnss_pos[0, 0], gnss_pos[0, 1], gnss_pos[0, 2], c='red', s=100, marker='o')
    ax1.scatter(gnss_pos[-1, 0], gnss_pos[-1, 1], gnss_pos[-1, 2], c='red', s=100, marker='s')
if len(fc_pos) > 0:
    ax1.scatter(fc_pos[0, 0], fc_pos[0, 1], fc_pos[0, 2], c='green', s=100, marker='o')
    ax1.scatter(fc_pos[-1, 0], fc_pos[-1, 1], fc_pos[-1, 2], c='green', s=100, marker='s')
ax1.set_xlabel('North [m]')
ax1.set_ylabel('East [m]')
ax1.set_zlabel('Down [m]')
ax1.set_title('3D Trajectory')
ax1.legend(fontsize=8)
ax1.grid(True)

# Position components vs time
ax2 = fig.add_subplot(5, 2, 2)
ax2.plot(t, px, 'b-', label='Filter North', linewidth=1)
ax2.plot(t, py, 'b-', label='Filter East', linewidth=1, alpha=0.7)
ax2.plot(t, pz, 'b-', label='Filter Down', linewidth=1, alpha=0.5)
if len(gnss_pos) > 0:
    ax2.plot(gnss_t, gnss_pos[:, 0], 'r--', label='GNSS North', linewidth=1)
    ax2.plot(gnss_t, gnss_pos[:, 1], 'r--', label='GNSS East', linewidth=1, alpha=0.7)
    ax2.plot(gnss_t, gnss_pos[:, 2], 'r--', label='GNSS Down', linewidth=1, alpha=0.5)
if len(fc_pos) > 0:
    ax2.plot(fc_t, fc_pos[:, 0], 'g:', label='FC North', linewidth=1.5)
    ax2.plot(fc_t, fc_pos[:, 1], 'g:', label='FC East', linewidth=1.5, alpha=0.7)
    ax2.plot(fc_t, fc_pos[:, 2], 'g:', label='FC Down', linewidth=1.5, alpha=0.5)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Position [m]')
ax2.set_title('Position Components vs Time')
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True)

# Velocity components vs time
ax3 = fig.add_subplot(5, 2, 3)
ax3.plot(t, vx, 'b-', label='Filter North', linewidth=1)
ax3.plot(t, vy, 'b-', label='Filter East', linewidth=1, alpha=0.7)
ax3.plot(t, vz, 'b-', label='Filter Down', linewidth=1, alpha=0.5)
if len(gnss_vel) > 0:
    ax3.plot(gnss_t, gnss_vel[:, 0], 'r--', label='GNSS North', linewidth=1)
    ax3.plot(gnss_t, gnss_vel[:, 1], 'r--', label='GNSS East', linewidth=1, alpha=0.7)
    ax3.plot(gnss_t, gnss_vel[:, 2], 'r--', label='GNSS Down', linewidth=1, alpha=0.5)
if len(fc_vel) > 0:
    ax3.plot(fc_t, fc_vel[:, 0], 'g:', label='FC North', linewidth=1.5)
    ax3.plot(fc_t, fc_vel[:, 1], 'g:', label='FC East', linewidth=1.5, alpha=0.7)
    ax3.plot(fc_t, fc_vel[:, 2], 'g:', label='FC Down', linewidth=1.5, alpha=0.5)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Velocity [m/s]')
ax3.set_title('Velocity Components vs Time')
ax3.legend(fontsize=7, ncol=2)
ax3.grid(True)

# Speed vs time
speed = np.sqrt(vx**2 + vy**2 + vz**2)
ax4 = fig.add_subplot(5, 2, 4)
ax4.plot(t, speed, 'r-', linewidth=1, label='Filter Speed')
if len(fc_vel) > 0:
	fc_speed = np.sqrt(fc_vel[:, 0]**2 + fc_vel[:, 1]**2 + fc_vel[:, 2]**2)
	ax4.plot(fc_t, fc_speed, 'g--', linewidth=1, alpha=0.7, label='FC Speed')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Speed [m/s]')
ax4.set_title('Total Speed vs Time')
ax4.legend(fontsize=8)
ax4.grid(True)

# Attitude (Euler angles) - Filter vs FC
ax5 = fig.add_subplot(5, 2, 6)
ax5.plot(t, np.degrees(roll_arr), 'r-', linewidth=1.5, label='Filter Roll')
ax5.plot(t, np.degrees(pitch_arr), 'g-', linewidth=1.5, label='Filter Pitch')
ax5.plot(t, np.degrees(yaw_arr), 'b-', linewidth=1.5, label='Filter Yaw')
if len(fc_att) > 0:
	ax5.plot(fc_att_t, np.degrees(fc_att[:, 0]), 'r--', linewidth=1, alpha=0.7, label='FC Roll')
	ax5.plot(fc_att_t, np.degrees(fc_att[:, 1]), 'g--', linewidth=1, alpha=0.7, label='FC Pitch')
	ax5.plot(fc_att_t, np.degrees(fc_att[:, 2]), 'b--', linewidth=1, alpha=0.7, label='FC Yaw')
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Angle [deg]')
ax5.set_title('Attitude (Euler Angles) - Filter vs FC')
ax5.legend(fontsize=7, ncol=2)
ax5.grid(True)

# Acceleration components vs time
ax6_accel = fig.add_subplot(5, 2, 5)
# Note: Filter doesn't output acceleration directly, so we can't compare
# But we can show FC acceleration for reference
if len(fc_accel) > 0:
	ax6_accel.plot(fc_accel_t, fc_accel[:, 0], 'r-', label='FC Accel X', linewidth=1)
	ax6_accel.plot(fc_accel_t, fc_accel[:, 1], 'g-', label='FC Accel Y', linewidth=1, alpha=0.7)
	ax6_accel.plot(fc_accel_t, fc_accel[:, 2], 'b-', label='FC Accel Z', linewidth=1, alpha=0.5)
ax6_accel.set_xlabel('Time [s]')
ax6_accel.set_ylabel('Acceleration [m/s²]')
ax6_accel.set_title('FC Acceleration Components')
ax6_accel.legend(fontsize=7, ncol=2)
ax6_accel.grid(True)

# Bias estimates - Gyroscope (columns 19-21)
ax6 = fig.add_subplot(5, 2, 7)
bgx = out[:, 19]
bgy = out[:, 20]
bgz = out[:, 21]
ax6.plot(t, bgx, linewidth=1, label='Gyro X bias', alpha=0.7, color='red')
ax6.plot(t, bgy, linewidth=1, label='Gyro Y bias', alpha=0.7, color='green')
ax6.plot(t, bgz, linewidth=1, label='Gyro Z bias', alpha=0.7, color='blue')
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('Bias [rad/s]')
ax6.set_title('Gyroscope Bias Estimates')
ax6.legend(fontsize=8)
ax6.grid(True)

# Bias estimates - Accelerometer (columns 22-24)
ax7 = fig.add_subplot(5, 2, 8)
bax = out[:, 22]
bay = out[:, 23]
baz = out[:, 24]
ax7.plot(t, bax, linewidth=1, label='Accel X bias', alpha=0.7, color='red')
ax7.plot(t, bay, linewidth=1, label='Accel Y bias', alpha=0.7, color='green')
ax7.plot(t, baz, linewidth=1, label='Accel Z bias', alpha=0.7, color='blue')
ax7.set_xlabel('Time [s]')
ax7.set_ylabel('Bias [m/s²]')
ax7.set_title('Accelerometer Bias Estimates')
ax7.legend(fontsize=8)
ax7.grid(True)

# Bias estimates - Virtual bias (b_mu, columns 25-27)
ax8 = fig.add_subplot(5, 2, 10)
bmux = out[:, 25]
bmuy = out[:, 26]
bmuz = out[:, 27]
ax8.plot(t, bmux, linewidth=1, label='b_mu X', alpha=0.7, color='red')
ax8.plot(t, bmuy, linewidth=1, label='b_mu Y', alpha=0.7, color='green')
ax8.plot(t, bmuz, linewidth=1, label='b_mu Z', alpha=0.7, color='blue')
ax8.set_xlabel('Time [s]')
ax8.set_ylabel('Virtual Bias [m/s-like]')
ax8.set_title('Virtual Position Bias (b_mu) Estimates')
ax8.legend(fontsize=8)
ax8.grid(True)

# Add title indicating data source
# (data_type already defined in file loading section)
fig.suptitle(f'EqF Trajectory Estimation - {data_type}', fontsize=14, fontweight='bold')

plt.tight_layout(rect=(0, 0, 1, 0.99))  # Adjust for suptitle

# Save with data type in filename
output_suffix = f"_{DATASET}"
output_filename = f'outputs/trajectory_plot{output_suffix}.png'
plt.savefig(output_filename, dpi=150)
print(f"Saved {data_type.lower()} trajectory plot to {output_filename}")

# Print statistics
print(f"\n=== Trajectory Statistics ({data_type}) ===")
print(f"Total time: {t[-1] - t[0]:.2f} s")
print(f"\nFilter final position: [{px[-1]:.1f}, {py[-1]:.1f}, {pz[-1]:.1f}] m")
print(f"Filter start position: [{px[0]:.1f}, {py[0]:.1f}, {pz[0]:.1f}] m")
if len(gnss_pos) > 0:
    print(f"GNSS final position: [{gnss_pos[-1, 0]:.1f}, {gnss_pos[-1, 1]:.1f}, {gnss_pos[-1, 2]:.1f}] m")
    print(f"GNSS start position: [{gnss_pos[0, 0]:.1f}, {gnss_pos[0, 1]:.1f}, {gnss_pos[0, 2]:.1f}] m")
if len(fc_pos) > 0:
    print(f"FC final position: [{fc_pos[-1, 0]:.1f}, {fc_pos[-1, 1]:.1f}, {fc_pos[-1, 2]:.1f}] m")
    print(f"FC start position: [{fc_pos[0, 0]:.1f}, {fc_pos[0, 1]:.1f}, {fc_pos[0, 2]:.1f}] m")
print(f"\nMax North: {np.max(px):.1f} m, Min: {np.min(px):.1f} m")
print(f"Max East: {np.max(py):.1f} m, Min: {np.min(py):.1f} m")
print(f"Max Down: {np.max(pz):.1f} m, Min: {np.min(pz):.1f} m")
print(f"Max speed: {np.max(speed):.1f} m/s")
print(f"Final velocity: [{vx[-1]:.1f}, {vy[-1]:.1f}, {vz[-1]:.1f}] m/s")
print(f"\nAttitude (Euler angles):")
print(f"Final Roll:  {np.degrees(roll_arr[-1]):.1f} deg")
print(f"Final Pitch: {np.degrees(pitch_arr[-1]):.1f} deg")
print(f"Final Yaw:   {np.degrees(yaw_arr[-1]):.1f} deg")
print(f"Max Roll:  {np.degrees(np.max(np.abs(roll_arr))):.1f} deg")
print(f"Max Pitch: {np.degrees(np.max(np.abs(pitch_arr))):.1f} deg")
print(f"Max Yaw:   {np.degrees(np.max(np.abs(yaw_arr))):.1f} deg")

# Create separate diagnostic figure if data available
if diag_data is not None:
    fig_diag = plt.figure(figsize=(14, 10))

    # Combined ANIS and ANEES (log scale)
    ax_combined = fig_diag.add_subplot(2, 2, 1)
    anis_times = []
    anis_vals = []
    anees_times = []
    anees_vals = []
    for row in diag_data:
        if not np.isnan(row['anis']):
            anis_times.append(row['time'])
            anis_vals.append(row['anis'])
        if not np.isnan(row['anees']):
            anees_times.append(row['time'])
            anees_vals.append(row['anees'])

    if anis_times:
        ax_combined.scatter(anis_times, anis_vals, s=8, alpha=0.6, label='ANIS', color='blue')
    if anees_times:
        ax_combined.scatter(anees_times, anees_vals, s=8, alpha=0.6, label='ANEES', color='red')
    ax_combined.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Target (1.0)')
    ax_combined.set_xlabel('Time [s]', fontsize=11)
    ax_combined.set_ylabel('Value', fontsize=11)
    ax_combined.set_title('ANIS & ANEES (log scale)', fontsize=12, fontweight='bold')
    ax_combined.set_yscale('log')
    ax_combined.legend(fontsize=10)
    ax_combined.grid(True, alpha=0.3)

    # ANIS over time (linear)
    ax_anis = fig_diag.add_subplot(2, 2, 2)
    if anis_times:
        ax_anis.plot(anis_times, anis_vals, 'b.-', linewidth=1, markersize=4, label='ANIS')
        anis_mean = np.mean(anis_vals)
        anis_std = np.std(anis_vals)
        anis_p95 = np.percentile(anis_vals, 95)
        ax_anis.axhline(y=float(anis_mean), color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Mean: {anis_mean:.3f}')
        ax_anis.axhline(y=float(anis_p95), color='darkblue', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'95th %ile: {anis_p95:.3f}')
        ax_anis.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Target (1.0)')
        ax_anis.fill_between(anis_times, anis_mean - anis_std, anis_mean + anis_std,
                            color='blue', alpha=0.1, label=f'±1σ: {anis_std:.3f}')
    ax_anis.set_xlabel('Time [s]', fontsize=11)
    ax_anis.set_ylabel('ANIS Value', fontsize=11)
    ax_anis.set_title('Average Normalized Innovation Squared', fontsize=12, fontweight='bold')
    ax_anis.legend(fontsize=9, loc='best')
    ax_anis.grid(True, alpha=0.3)

    # ANEES over time (linear)
    ax_anees = fig_diag.add_subplot(2, 2, 3)
    if anees_times:
        ax_anees.plot(anees_times, anees_vals, 'r.-', linewidth=1, markersize=4, label='ANEES')
        anees_mean = np.mean(anees_vals)
        anees_std = np.std(anees_vals)
        anees_p95 = np.percentile(anees_vals, 95)
        ax_anees.axhline(y=float(anees_mean), color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                        label=f'Mean: {anees_mean:.1f}')
        ax_anees.axhline(y=float(anees_p95), color='darkred', linestyle=':', linewidth=1.5, alpha=0.7,
                        label=f'95th %ile: {anees_p95:.1f}')
        ax_anees.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Target (1.0)')
        ax_anees.fill_between(anees_times, np.maximum(0, anees_mean - anees_std), anees_mean + anees_std,
                             color='red', alpha=0.1, label=f'±1σ: {anees_std:.1f}')
    ax_anees.set_xlabel('Time [s]', fontsize=11)
    ax_anees.set_ylabel('ANEES Value', fontsize=11)
    ax_anees.set_title('Avg Normalized Estimation Error Squared', fontsize=12, fontweight='bold')
    ax_anees.legend(fontsize=9, loc='best')
    ax_anees.grid(True, alpha=0.3)

    # Statistics summary
    ax_stats = fig_diag.add_subplot(2, 2, 4)
    ax_stats.axis('off')
    stats_text = "Filter Diagnostic Summary\n" + "="*40 + "\n\n"
    if anis_times:
        stats_text += f"ANIS Statistics:\n"
        stats_text += f"  Mean:      {np.mean(anis_vals):.4f}\n"
        stats_text += f"  Std Dev:   {np.std(anis_vals):.4f}\n"
        stats_text += f"  Min:       {np.min(anis_vals):.4f}\n"
        stats_text += f"  25th %ile: {np.percentile(anis_vals, 25):.4f}\n"
        stats_text += f"  Median:    {np.percentile(anis_vals, 50):.4f}\n"
        stats_text += f"  75th %ile: {np.percentile(anis_vals, 75):.4f}\n"
        stats_text += f"  95th %ile: {np.percentile(anis_vals, 95):.4f}\n"
        stats_text += f"  Max:       {np.max(anis_vals):.4f}\n"
    stats_text += "\n"
    if anees_times:
        stats_text += f"ANEES Statistics:\n"
        stats_text += f"  Mean:      {np.mean(anees_vals):.2f}\n"
        stats_text += f"  Std Dev:   {np.std(anees_vals):.2f}\n"
        stats_text += f"  Min:       {np.min(anees_vals):.2f}\n"
        stats_text += f"  25th %ile: {np.percentile(anees_vals, 25):.2f}\n"
        stats_text += f"  Median:    {np.percentile(anees_vals, 50):.2f}\n"
        stats_text += f"  75th %ile: {np.percentile(anees_vals, 75):.2f}\n"
        stats_text += f"  95th %ile: {np.percentile(anees_vals, 95):.2f}\n"
        stats_text += f"  Max:       {np.max(anees_vals):.2f}\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig_diag.suptitle(f'Filter Diagnostics - {data_type}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    # Save diagnostic figure
    diag_filename = f'outputs/diagnostics_plot{output_suffix}.png'
    plt.savefig(diag_filename, dpi=150)
    print(f"\nSaved diagnostic plot to {diag_filename}")

plt.show()
