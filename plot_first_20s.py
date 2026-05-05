"""Plot the first 20 seconds of trajectory from EqF filter output."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
R_EARTH = 6_378_137.0
_C = {
    "t": 0,
    "lon": 1,
    "lat": 2,
    "alt": 3,
    "gps_vn": 4,
    "gps_ve": 5,
    "gps_vd": 6,
    "pn": 36,
    "pe": 37,
    "pd": 38,
    "vn": 40,
    "ve": 41,
    "vd": 42,
}

def gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)
    return np.array([north, east, down])

# Load CSV data
raw = np.genfromtxt("data/20241011_NIMBUS24_Flight_FC_Data.csv", delimiter=",", skip_header=1)
if raw.ndim == 1:
    raw = raw.reshape(1, -1)

# Find first valid GPS fix for reference
valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
first = np.argmax(valid)
lat0 = raw[first, _C["lat"]]
lon0 = raw[first, _C["lon"]]
alt0 = raw[first, _C["alt"]] / 1000.0

# Extract GNSS and FC data for first 20 seconds
gnss_pos = []
gnss_vel = []
fc_pos = []
fc_vel = []
gnss_t = []
fc_t = []

for i, row in enumerate(raw):
    t = row[_C["t"]]
    if not np.isfinite(t) or t > 20:
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

    # FC data
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
        fc_t.append(t)

gnss_pos = np.array(gnss_pos) if gnss_pos else np.empty((0, 3))
gnss_vel = np.array(gnss_vel) if gnss_vel else np.empty((0, 3))
gnss_t = np.array(gnss_t)
fc_pos = np.array(fc_pos) if fc_pos else np.empty((0, 3))
fc_vel = np.array(fc_vel) if fc_vel else np.empty((0, 3))
fc_t = np.array(fc_t)

# Load output data
out = np.genfromtxt("outputs/tg_eqf_output.csv", delimiter=",", skip_header=1)

# Filter to first 20 seconds
mask = out[:, 0] <= 20
out = out[mask]

# Extract columns
t = out[:, 0]
px = out[:, 1]
py = out[:, 2]
pz = out[:, 3]
vx = out[:, 4]
vy = out[:, 5]
vz = out[:, 6]

# Extract rotation matrix and compute Euler angles
R_list = []
for i in range(len(out)):
    R = np.array([
        [out[i, 7], out[i, 8], out[i, 9]],
        [out[i, 10], out[i, 11], out[i, 12]],
        [out[i, 13], out[i, 14], out[i, 15]]
    ])
    R_list.append(R)

roll_arr = np.array([np.arctan2(R[1, 0], R[0, 0]) for R in R_list])
pitch_arr = np.array([np.arcsin(np.clip(-R[2, 0], -1, 1)) for R in R_list])
yaw_arr = np.array([np.arctan2(R[2, 1], R[2, 2]) for R in R_list])

# Create figure with subplots
fig = plt.figure(figsize=(16, 14))

# 3D trajectory
ax1 = fig.add_subplot(3, 2, 1, projection='3d')
ax1.plot(px, py, pz, 'b-', linewidth=1, label='Filter Estimate')
if len(gnss_pos) > 0:
    ax1.plot(gnss_pos[:, 0], gnss_pos[:, 1], gnss_pos[:, 2], 'r--', linewidth=1, label='GNSS')
if len(fc_pos) > 0:
    ax1.plot(fc_pos[:, 0], fc_pos[:, 1], fc_pos[:, 2], 'g:', linewidth=1.5, label='FC Estimate')
ax1.scatter(px[0], py[0], pz[0], c='blue', s=100, marker='o')
if len(px) > 1:
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
ax1.set_title('3D Trajectory (First 20s)')
ax1.legend(fontsize=8)
ax1.grid(True)

# Position components vs time
ax2 = fig.add_subplot(3, 2, 2)
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
ax3 = fig.add_subplot(3, 2, 3)
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
ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(t, speed, 'r-', linewidth=1)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Speed [m/s]')
ax4.set_title('Total Speed vs Time')
ax4.grid(True)

# Attitude (Euler angles)
ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(t, np.degrees(roll_arr), 'r-', linewidth=1, label='Roll')
ax5.plot(t, np.degrees(pitch_arr), 'g-', linewidth=1, label='Pitch')
ax5.plot(t, np.degrees(yaw_arr), 'b-', linewidth=1, label='Yaw')
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Angle [deg]')
ax5.set_title('Attitude (Euler Angles)')
ax5.legend(fontsize=8)
ax5.grid(True)

# Bias estimates
ax6 = fig.add_subplot(3, 2, 6)
bgx = out[:, 16]
bgy = out[:, 17]
bgz = out[:, 18]
bax = out[:, 19]
bay = out[:, 20]
baz = out[:, 21]
ax6.plot(t, bgx, linewidth=1, label='Gyro X bias', alpha=0.7)
ax6.plot(t, bgy, linewidth=1, label='Gyro Y bias', alpha=0.7)
ax6.plot(t, bgz, linewidth=1, label='Gyro Z bias', alpha=0.7)
ax6.plot(t, bax, linewidth=1, label='Accel X bias', alpha=0.7)
ax6.plot(t, bay, linewidth=1, label='Accel Y bias', alpha=0.7)
ax6.plot(t, baz, linewidth=1, label='Accel Z bias', alpha=0.7)
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('Bias [m/s^2 or rad/s]')
ax6.set_title('Bias Estimates')
ax6.legend(fontsize=7, ncol=2)
ax6.grid(True)

plt.tight_layout()
plt.savefig('outputs/trajectory_plot_20s.png', dpi=150)
print(f"Saved trajectory plot to outputs/trajectory_plot_20s.png")

# Print statistics
print(f"\n=== First 20 Seconds Statistics ===")
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

plt.show()
