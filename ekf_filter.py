"""Extended Kalman Filter for rocket trajectory and attitude estimation using IMU and GNSS."""

import os
from typing import Any
import numpy as np
from scipy.linalg import expm

# =============================================================================
# Constants
# =============================================================================

g = 9.81
R_EARTH = 6_378_137.0

# Process noise (tuning parameters)
Q_pos = np.eye(3) * (0.01**2)      # Position process noise
Q_vel = np.eye(3) * (0.1**2)       # Velocity process noise
Q_att = np.eye(3) * (0.001**2)     # Attitude process noise
Q_gyro_bias = np.eye(3) * (1e-6**2)  # Gyro bias process noise (very slow drift)
Q_accel_bias = np.eye(3) * (1e-6**2) # Accel bias process noise (very slow drift)

# Measurement noise (from sensor specs)
R_gnss_pos = np.eye(3) * (5.0**2)    # GNSS position error ~5m
R_gnss_vel = np.eye(3) * (0.1**2)    # GNSS velocity error ~0.1 m/s


# =============================================================================
# Euler Angle and Rotation Utilities
# =============================================================================

def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (rad) to Direction Cosine Matrix (DCM).

    Uses ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    return R


def dcm_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert DCM to Euler angles (roll, pitch, yaw) in radians."""
    pitch = np.arcsin(-R[2, 0])

    # Avoid singularity when pitch = ±π/2
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return roll, pitch, yaw


def skew(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def normalize_dcm(R: np.ndarray) -> np.ndarray:
    """Orthonormalize DCM using SVD."""
    U, _, VT = np.linalg.svd(R)
    return U @ VT


# =============================================================================
# EKF Filter
# =============================================================================

class EKF:
    """Extended Kalman Filter for attitude and trajectory estimation."""

    # State indices
    PX, PY, PZ = 0, 1, 2           # Position (North, East, Down)
    VX, VY, VZ = 3, 4, 5           # Velocity
    ROLL, PITCH, YAW = 6, 7, 8     # Attitude (Euler angles)
    BGX, BGY, BGZ = 9, 10, 11      # Gyro bias
    BAX, BAY, BAZ = 12, 13, 14     # Accel bias

    STATE_SIZE = 15

    def __init__(self):
        """Initialize filter state and covariance."""
        self.x = np.zeros(self.STATE_SIZE)

        # Initial covariance (large uncertainty)
        self.cov: np.ndarray = np.diag([
            100.0**2,  # pos
            100.0**2,
            100.0**2,
            10.0**2,   # vel
            10.0**2,
            10.0**2,
            (10*np.pi/180)**2,  # att (10 deg)
            (10*np.pi/180)**2,
            (10*np.pi/180)**2,
            (0.01)**2,   # gyro bias
            (0.01)**2,
            (0.01)**2,
            (0.5)**2,    # accel bias
            (0.5)**2,
            (0.5)**2
        ])

        self.t_prev = None
        self.last_gnss_update = -999.0

    def predict(self, t: float, accel_meas: np.ndarray, gyro_meas: np.ndarray) -> None:
        """Predict step using IMU measurements.

        Args:
            t: Current time (seconds)
            accel_meas: Measured acceleration in body frame [m/s^2]
            gyro_meas: Measured angular velocity in body frame [rad/s]
        """
        if self.t_prev is None:
            self.t_prev = t
            return

        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return

        # Extract current state
        pos = self.x[self.PX:self.PZ+1]
        vel = self.x[self.VX:self.VZ+1]
        roll, pitch, yaw = self.x[self.ROLL:self.YAW+1]
        b_gyro = self.x[self.BGX:self.BGZ+1]
        b_accel = self.x[self.BAX:self.BAZ+1]

        # Get rotation matrix
        R = euler_to_dcm(roll, pitch, yaw)

        # Remove bias from measurements
        accel_unbiased = accel_meas - b_accel
        gyro_unbiased = gyro_meas - b_gyro

        # Integrate kinematics
        # Position: p_dot = v
        pos_new = pos + vel * dt

        # Velocity: v_dot = R @ a - g_vec
        g_vec = np.array([0, 0, g])  # Gravity in NED (positive down)
        accel_world = R @ accel_unbiased
        vel_new = vel + (accel_world - g_vec) * dt

        # Attitude: integrate gyroscope with exponential map on SO(3)
        # w_norm = ||w||, R_new = R * exp(skew(w) * dt)
        w_mag = np.linalg.norm(gyro_unbiased)
        if w_mag > 1e-8:
            dR = expm(skew(gyro_unbiased) * dt)
            R_new = R @ dR
            R_new = normalize_dcm(R_new)
        else:
            R_new = R

        roll_new, pitch_new, yaw_new = dcm_to_euler(R_new)

        # Biases: constant (random walk with very small noise)
        b_gyro_new = b_gyro
        b_accel_new = b_accel

        # Update state
        x_new = self.x.copy()
        x_new[self.PX:self.PZ+1] = pos_new
        x_new[self.VX:self.VZ+1] = vel_new
        x_new[self.ROLL:self.YAW+1] = [roll_new, pitch_new, yaw_new]
        x_new[self.BGX:self.BGZ+1] = b_gyro_new
        x_new[self.BAX:self.BAZ+1] = b_accel_new
        self.x = x_new

        # Compute Jacobian F (for covariance prediction)
        F = self._jacobian_F(vel, accel_unbiased, gyro_unbiased, R, dt)

        # Process noise
        Q = np.diag(np.concatenate([
            np.diag(Q_pos),
            np.diag(Q_vel),
            np.diag(Q_att),
            np.diag(Q_gyro_bias),
            np.diag(Q_accel_bias)
        ]))

        # Covariance prediction: P = F @ P @ F.T + Q
        self.cov = F @ self.cov @ F.T + Q
        self.cov = 0.5 * (self.cov + self.cov.T)  # Ensure symmetry

        self.t_prev = t

    def _jacobian_F(self, vel: np.ndarray, accel: np.ndarray, gyro: np.ndarray, R: np.ndarray, dt: float) -> np.ndarray:
        """Compute Jacobian of state transition."""
        F = np.eye(self.STATE_SIZE)

        # dp/dv = I * dt
        F[self.PX:self.PZ+1, self.VX:self.VZ+1] = np.eye(3) * dt

        # dv/datt and dv/d(b_accel)
        accel_world = R @ accel
        # dv/droll, dv/dpitch, dv/dyaw using finite differences for simplicity
        eps = 1e-7
        for i in range(3):
            x_plus = self.x.copy()
            x_plus[self.ROLL + i] += eps
            roll_p, pitch_p, yaw_p = x_plus[self.ROLL:self.YAW+1]
            R_plus = euler_to_dcm(roll_p, pitch_p, yaw_p)
            accel_plus = R_plus @ accel
            dv = (accel_plus - accel_world) / eps
            F[self.VX:self.VZ+1, self.ROLL + i] = dv * dt

        # dv/d(b_accel) = -R
        F[self.VX:self.VZ+1, self.BAX:self.BAZ+1] = -R * dt

        # droll/d(b_gyro), dpitch/d(b_gyro), dyaw/d(b_gyro)
        # Simplified: attitude changes proportional to gyro * dt
        for i in range(3):
            x_plus = self.x.copy()
            x_plus[self.BGX + i] += eps
            gyro_plus = gyro - (x_plus[self.BGX:self.BGZ+1] - self.x[self.BGX:self.BGZ+1])

            w_mag = np.linalg.norm(gyro)
            if w_mag > 1e-8:
                dR = expm(skew(gyro) * dt)
                R_new = R @ dR
            else:
                R_new = R

            w_mag_plus = np.linalg.norm(gyro_plus)
            if w_mag_plus > 1e-8:
                dR_plus = expm(skew(gyro_plus) * dt)
                R_plus = R @ dR_plus
            else:
                R_plus = R

            _, pitch_new, yaw_new = dcm_to_euler(R_new)
            _, pitch_plus, yaw_plus = dcm_to_euler(R_plus)

            datt = np.array([0, (pitch_plus - pitch_new) / eps, (yaw_plus - yaw_new) / eps])
            F[self.ROLL:self.YAW+1, self.BGX + i] = datt

        return F

    def update_gnss(self, pos_meas: np.ndarray, vel_meas: np.ndarray) -> None:
        """GNSS measurement update.

        Args:
            pos_meas: Measured position in NED [m]
            vel_meas: Measured velocity in NED [m/s]
        """
        # Measurement vector (6D: 3 pos + 3 vel)
        z = np.concatenate([pos_meas, vel_meas])

        # Measurement matrix H (identity for position and velocity)
        H = np.zeros((6, self.STATE_SIZE))
        H[0:3, self.PX:self.PZ+1] = np.eye(3)  # Position measurement
        H[3:6, self.VX:self.VZ+1] = np.eye(3)  # Velocity measurement

        # Innovation
        z_pred = np.concatenate([
            self.x[self.PX:self.PZ+1],
            self.x[self.VX:self.VZ+1]
        ])
        y = z - z_pred

        # Measurement covariance
        R_meas = np.diag(np.concatenate([np.diag(R_gnss_pos), np.diag(R_gnss_vel)]))

        # Kalman gain
        S = H @ self.cov @ H.T + R_meas
        K = self.cov @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        self.cov = (np.eye(self.STATE_SIZE) - K @ H) @ self.cov
        self.cov = 0.5 * (self.cov + self.cov.T)  # Ensure symmetry

    def output_row(self, t: float) -> list[Any]:
        """Extract state as output row."""
        roll, pitch, yaw = self.x[self.ROLL:self.YAW+1]
        rot = euler_to_dcm(roll, pitch, yaw)

        return [
            t,
            self.x[self.PX],
            self.x[self.PY],
            self.x[self.PZ],
            self.x[self.VX],
            self.x[self.VY],
            self.x[self.VZ],
            rot[0, 0], rot[0, 1], rot[0, 2],
            rot[1, 0], rot[1, 1], rot[1, 2],
            rot[2, 0], rot[2, 1], rot[2, 2],
            np.degrees(roll),
            np.degrees(pitch),
            np.degrees(yaw),
            self.x[self.BGX],
            self.x[self.BGY],
            self.x[self.BGZ],
            self.x[self.BAX],
            self.x[self.BAY],
            self.x[self.BAZ],
        ]


# =============================================================================
# GPS Coordinate Conversion
# =============================================================================

def gps_to_ned(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)
    return np.array([north, east, down])


# =============================================================================
# CSV I/O
# =============================================================================

_C = {
    "t": 0,
    "lon": 1,
    "lat": 2,
    "alt": 3,
    "gps_vn": 4,
    "gps_ve": 5,
    "gps_vd": 6,
    "ax": 9,
    "ay": 10,
    "az": 11,
    "gx": 15,
    "gy": 16,
    "gz": 17,
}


def run(csv_in: str = "data/20241011_NIMBUS24_Flight_FC_Data.csv",
        csv_out: str = "outputs/ekf_output.csv") -> None:
    """Run EKF on NIMBUS24 FC CSV data."""
    raw = np.genfromtxt(csv_in, delimiter=",", skip_header=1)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    print(f"Loaded {len(raw)} rows")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # ICM_20608 gyro scale factor: raw ADC counts to degrees/sec, then to rad/s
    gyro_scale_factor = (2000.0 / 32768.0) * (np.pi / 180.0)  # raw ADC -> rad/s

    # Find first valid GPS fix for reference
    valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
    first = np.argmax(valid)
    lat0 = raw[first, _C["lat"]]
    lon0 = raw[first, _C["lon"]]
    alt0 = raw[first, _C["alt"]] / 1000.0

    # Initialize filter
    filt = EKF()
    out = []
    gnss_count = 0
    last_progress_t = 0
    last_gnss_update_t = -999.0

    for i, row in enumerate(raw):
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        # Read body-frame measurements
        gyro_body = np.array([row[_C["gx"]], row[_C["gy"]], row[_C["gz"]]]) * gyro_scale_factor  # Convert from raw ADC to rad/s
        accel_body = np.array([row[_C["ax"]], row[_C["ay"]], row[_C["az"]]]) * g

        if not np.all(np.isfinite(np.concatenate([gyro_body, accel_body]))):
            continue

        # Propagate filter
        filt.predict(t, accel_body, gyro_body)

        # Output filter state
        out.append(filt.output_row(t))

        # Check for GNSS update
        lat = row[_C["lat"]]
        lon = row[_C["lon"]]
        if lat != 0 and lon != 0 and (t - last_gnss_update_t) >= 5.0:
            alt = row[_C["alt"]] / 1000.0
            pos_NED = gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
            vel_NED = np.array([row[_C["gps_vn"]], row[_C["gps_ve"]], row[_C["gps_vd"]]]) / 1000.0
            filt.update_gnss(pos_NED, vel_NED)
            last_gnss_update_t = t
            gnss_count += 1

        # Progress reporting
        if t - last_progress_t >= 30:
            progress = 100.0 * i / len(raw)
            print(f"[PROGRESS] {progress:.1f}% | t={t:.2f}s | rows={i}/{len(raw)} | GNSS_updates={gnss_count}")
            last_progress_t = t

    # Save output
    out = np.asarray(out)
    header = (
        "t,px,py,pz,vx,vy,vz,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
        "roll,pitch,yaw,"
        "bgx,bgy,bgz,bax,bay,baz"
    )

    np.savetxt(csv_out, out, delimiter=",", header=header, comments="")
    print(f"Wrote {len(out)} rows to {csv_out}")


if __name__ == "__main__":
    cin = "data/20241011_NIMBUS24_Flight_FC_Data.csv"
    cout = "outputs/ekf_output.csv"
    run(cin, cout)
