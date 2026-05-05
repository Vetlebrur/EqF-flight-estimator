"""Compare EqF and EKF filter outputs."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R_EARTH = 6_378_137.0

_C_gnss = {
    "t": 0,
    "lon": 1,
    "lat": 2,
    "alt": 3,
    "gps_vn": 4,
    "gps_ve": 5,
    "gps_vd": 6,
}


def gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """Convert GPS to NED."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)
    return np.array([north, east, down])


# Load GNSS reference
raw = np.genfromtxt("data/20241011_NIMBUS24_Flight_FC_Data.csv", delimiter=",", skip_header=1)
if raw.ndim == 1:
    raw = raw.reshape(1, -1)

valid = (raw[:, _C_gnss["lat"]] != 0) & (raw[:, _C_gnss["lon"]] != 0)
first = np.argmax(valid)
lat0 = raw[first, _C_gnss["lat"]]
lon0 = raw[first, _C_gnss["lon"]]
alt0 = raw[first, _C_gnss["alt"]] / 1000.0

gnss_pos = []
gnss_vel = []
gnss_t = []

for row in raw:
    t = row[_C_gnss["t"]]
    if not np.isfinite(t):
        continue

    lat = row[_C_gnss["lat"]]
    lon = row[_C_gnss["lon"]]

    if lat != 0 and lon != 0:
        alt = row[_C_gnss["alt"]] / 1000.0
        pos_NED = gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
        vel_NED = np.array([row[_C_gnss["gps_vn"]], row[_C_gnss["gps_ve"]], row[_C_gnss["gps_vd"]]]) / 1000.0
        gnss_pos.append(pos_NED)
        gnss_vel.append(vel_NED)
        gnss_t.append(t)

gnss_pos = np.array(gnss_pos) if gnss_pos else np.empty((0, 3))
gnss_vel = np.array(gnss_vel) if gnss_vel else np.empty((0, 3))
gnss_t = np.array(gnss_t)

# Load EqF output
try:
    eqf_out = np.genfromtxt("outputs/tg_eqf_output.csv", delimiter=",", skip_header=1)
    if eqf_out.ndim == 1:
        eqf_out = eqf_out.reshape(1, -1)
    eqf_t = eqf_out[:, 0]
    eqf_pos = eqf_out[:, 1:4]
    eqf_vel = eqf_out[:, 4:7]
    eqf_roll = np.degrees(eqf_out[:, 16] if eqf_out.shape[1] > 16 else np.zeros(len(eqf_out)))
    eqf_pitch = np.degrees(eqf_out[:, 17] if eqf_out.shape[1] > 17 else np.zeros(len(eqf_out)))
    eqf_yaw = np.degrees(eqf_out[:, 18] if eqf_out.shape[1] > 18 else np.zeros(len(eqf_out)))
    have_eqf = True
except:
    have_eqf = False
    print("EqF output not found")

# Load EKF output
try:
    ekf_out = np.genfromtxt("outputs/ekf_output.csv", delimiter=",", skip_header=1)
    if ekf_out.ndim == 1:
        ekf_out = ekf_out.reshape(1, -1)
    ekf_t = ekf_out[:, 0]
    ekf_pos = ekf_out[:, 1:4]
    ekf_vel = ekf_out[:, 4:7]
    ekf_roll = ekf_out[:, 16]
    ekf_pitch = ekf_out[:, 17]
    ekf_yaw = ekf_out[:, 18]
    have_ekf = True
except:
    have_ekf = False
    print("EKF output not found")

if not (have_eqf or have_ekf):
    print("No filter outputs found. Run filters first.")
    exit(1)

# Create comparison figure
fig = plt.figure(figsize=(16, 12))

# 3D Trajectory
ax = fig.add_subplot(3, 3, 1, projection='3d')
if have_eqf:
    ax.plot(eqf_pos[:, 0], eqf_pos[:, 1], eqf_pos[:, 2], 'b-', linewidth=1, label='EqF', alpha=0.7)
if have_ekf:
    ax.plot(ekf_pos[:, 0], ekf_pos[:, 1], ekf_pos[:, 2], 'r-', linewidth=1, label='EKF', alpha=0.7)
if len(gnss_pos) > 0:
    ax.plot(gnss_pos[:, 0], gnss_pos[:, 1], gnss_pos[:, 2], 'g--', linewidth=1, label='GNSS', alpha=0.5)
ax.set_xlabel('North [m]')
ax.set_ylabel('East [m]')
ax.set_zlabel('Down [m]')
ax.set_title('3D Trajectory Comparison')
ax.legend()
ax.grid(True)

# Position North
ax = fig.add_subplot(3, 3, 2)
if have_eqf:
    ax.plot(eqf_t, eqf_pos[:, 0], 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_pos[:, 0], 'r-', label='EKF', linewidth=1)
if len(gnss_pos) > 0:
    ax.plot(gnss_t, gnss_pos[:, 0], 'g--', label='GNSS', linewidth=1, alpha=0.5)
ax.set_ylabel('Position North [m]')
ax.set_title('Position - North Component')
ax.legend()
ax.grid(True)

# Position East
ax = fig.add_subplot(3, 3, 3)
if have_eqf:
    ax.plot(eqf_t, eqf_pos[:, 1], 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_pos[:, 1], 'r-', label='EKF', linewidth=1)
if len(gnss_pos) > 0:
    ax.plot(gnss_t, gnss_pos[:, 1], 'g--', label='GNSS', linewidth=1, alpha=0.5)
ax.set_ylabel('Position East [m]')
ax.set_title('Position - East Component')
ax.legend()
ax.grid(True)

# Velocity North
ax = fig.add_subplot(3, 3, 4)
if have_eqf:
    ax.plot(eqf_t, eqf_vel[:, 0], 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_vel[:, 0], 'r-', label='EKF', linewidth=1)
if len(gnss_vel) > 0:
    ax.plot(gnss_t, gnss_vel[:, 0], 'g--', label='GNSS', linewidth=1, alpha=0.5)
ax.set_ylabel('Velocity North [m/s]')
ax.set_title('Velocity - North Component')
ax.legend()
ax.grid(True)

# Velocity East
ax = fig.add_subplot(3, 3, 5)
if have_eqf:
    ax.plot(eqf_t, eqf_vel[:, 1], 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_vel[:, 1], 'r-', label='EKF', linewidth=1)
if len(gnss_vel) > 0:
    ax.plot(gnss_t, gnss_vel[:, 1], 'g--', label='GNSS', linewidth=1, alpha=0.5)
ax.set_ylabel('Velocity East [m/s]')
ax.set_title('Velocity - East Component')
ax.legend()
ax.grid(True)

# Velocity Down
ax = fig.add_subplot(3, 3, 6)
if have_eqf:
    ax.plot(eqf_t, eqf_vel[:, 2], 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_vel[:, 2], 'r-', label='EKF', linewidth=1)
if len(gnss_vel) > 0:
    ax.plot(gnss_t, gnss_vel[:, 2], 'g--', label='GNSS', linewidth=1, alpha=0.5)
ax.set_ylabel('Velocity Down [m/s]')
ax.set_title('Velocity - Down Component')
ax.legend()
ax.grid(True)

# Attitude - Roll
ax = fig.add_subplot(3, 3, 7)
if have_eqf:
    ax.plot(eqf_t, eqf_roll, 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_roll, 'r-', label='EKF', linewidth=1)
ax.set_ylabel('Roll [deg]')
ax.set_title('Attitude - Roll')
ax.legend()
ax.grid(True)

# Attitude - Pitch
ax = fig.add_subplot(3, 3, 8)
if have_eqf:
    ax.plot(eqf_t, eqf_pitch, 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_pitch, 'r-', label='EKF', linewidth=1)
ax.set_ylabel('Pitch [deg]')
ax.set_title('Attitude - Pitch')
ax.legend()
ax.grid(True)

# Attitude - Yaw
ax = fig.add_subplot(3, 3, 9)
if have_eqf:
    ax.plot(eqf_t, eqf_yaw, 'b-', label='EqF', linewidth=1)
if have_ekf:
    ax.plot(ekf_t, ekf_yaw, 'r-', label='EKF', linewidth=1)
ax.set_ylabel('Yaw [deg]')
ax.set_title('Attitude - Yaw')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('outputs/filter_comparison.png', dpi=150)
print(f"Saved comparison plot to outputs/filter_comparison.png")

# Print statistics
print("\n" + "="*80)
print("FILTER COMPARISON STATISTICS")
print("="*80)

if have_eqf:
    print("\nEQF FILTER:")
    print(f"  Final position: [{eqf_pos[-1, 0]:.1f}, {eqf_pos[-1, 1]:.1f}, {eqf_pos[-1, 2]:.1f}] m")
    print(f"  Final velocity: [{eqf_vel[-1, 0]:.1f}, {eqf_vel[-1, 1]:.1f}, {eqf_vel[-1, 2]:.1f}] m/s")
    speed_eqf = np.sqrt(np.sum(eqf_vel**2, axis=1))
    print(f"  Final attitude: roll={eqf_roll[-1]:.1f}°, pitch={eqf_pitch[-1]:.1f}°, yaw={eqf_yaw[-1]:.1f}°")
    print(f"  Max speed: {np.max(speed_eqf):.1f} m/s")

if have_ekf:
    print("\nEKF FILTER:")
    print(f"  Final position: [{ekf_pos[-1, 0]:.1f}, {ekf_pos[-1, 1]:.1f}, {ekf_pos[-1, 2]:.1f}] m")
    print(f"  Final velocity: [{ekf_vel[-1, 0]:.1f}, {ekf_vel[-1, 1]:.1f}, {ekf_vel[-1, 2]:.1f}] m/s")
    speed_ekf = np.sqrt(np.sum(ekf_vel**2, axis=1))
    print(f"  Final attitude: roll={ekf_roll[-1]:.1f}°, pitch={ekf_pitch[-1]:.1f}°, yaw={ekf_yaw[-1]:.1f}°")
    print(f"  Max speed: {np.max(speed_ekf):.1f} m/s")

if len(gnss_pos) > 0:
    print("\nGNSS REFERENCE:")
    print(f"  Final position: [{gnss_pos[-1, 0]:.1f}, {gnss_pos[-1, 1]:.1f}, {gnss_pos[-1, 2]:.1f}] m")
    print(f"  Final velocity: [{gnss_vel[-1, 0]:.1f}, {gnss_vel[-1, 1]:.1f}, {gnss_vel[-1, 2]:.1f}] m/s")

# Position error vs GNSS
if have_eqf and len(gnss_pos) > 0:
    # Interpolate filter output to GNSS times
    eqf_pos_interp = np.interp(gnss_t, eqf_t, eqf_pos[:, 0]), \
                     np.interp(gnss_t, eqf_t, eqf_pos[:, 1]), \
                     np.interp(gnss_t, eqf_t, eqf_pos[:, 2])
    eqf_pos_interp = np.array(eqf_pos_interp).T
    eqf_pos_error = np.linalg.norm(eqf_pos_interp - gnss_pos, axis=1)
    print(f"\nEQF Position Error vs GNSS:")
    print(f"  Mean: {np.mean(eqf_pos_error):.1f} m")
    print(f"  Max: {np.max(eqf_pos_error):.1f} m")

if have_ekf and len(gnss_pos) > 0:
    ekf_pos_interp = np.interp(gnss_t, ekf_t, ekf_pos[:, 0]), \
                     np.interp(gnss_t, ekf_t, ekf_pos[:, 1]), \
                     np.interp(gnss_t, ekf_t, ekf_pos[:, 2])
    ekf_pos_interp = np.array(ekf_pos_interp).T
    ekf_pos_error = np.linalg.norm(ekf_pos_interp - gnss_pos, axis=1)
    print(f"\nEKF Position Error vs GNSS:")
    print(f"  Mean: {np.mean(ekf_pos_error):.1f} m")
    print(f"  Max: {np.max(ekf_pos_error):.1f} m")

plt.show()
