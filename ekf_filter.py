"""Extended Kalman Filter for rocket trajectory and attitude estimation using IMU and GNSS."""

import os
from typing import Any
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as ScipyRot

# =============================================================================
# Configuration
# =============================================================================

# "full"    -> data/20241011_NIMBUS24_Flight_FC_Data.csv
# "30s"     -> data/20241011_NIMBUS24_Flight_FC_Data_30s.csv
# "1s_loop" -> data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv
DATASET = "30s"

GNSS_UPDATE_FREQ_HZ = 1.0

# =============================================================================
# Constants
# =============================================================================

g = 9.81
R_EARTH = 6_378_137.0

# Process noise
Q_pos       = np.eye(3) * (0.01**2)
Q_vel       = np.eye(3) * (0.1**2)
Q_att       = np.eye(3) * (0.001**2)
Q_gyro_bias = np.eye(3) * (1e-6**2)
Q_accel_bias = np.eye(3) * (1e-6**2)

# Measurement noise
R_gnss_pos = np.eye(3) * (5.0**2)
R_gnss_vel = np.eye(3) * (0.1**2)


# =============================================================================
# Rotation Utilities
# =============================================================================

def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])


def dcm_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert DCM to (roll, pitch, yaw) in radians."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw  = float(np.arctan2(-R[0, 1], R[1, 1]))
    return roll, pitch, yaw


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ])


def normalize_dcm(R: np.ndarray) -> np.ndarray:
    U, _, VT = np.linalg.svd(R)
    return U @ VT


# =============================================================================
# EKF Filter
# =============================================================================

class EKF:
    """Extended Kalman Filter for attitude and trajectory estimation."""

    PX, PY, PZ       = 0, 1, 2
    VX, VY, VZ       = 3, 4, 5
    ROLL, PITCH, YAW = 6, 7, 8
    BGX, BGY, BGZ    = 9, 10, 11
    BAX, BAY, BAZ    = 12, 13, 14
    STATE_SIZE = 15

    def __init__(self) -> None:
        self.x = np.zeros(self.STATE_SIZE)
        self.cov: np.ndarray = np.diag([
            100.0**2, 100.0**2, 100.0**2,
            10.0**2,  10.0**2,  10.0**2,
            (10*np.pi/180)**2, (10*np.pi/180)**2, (10*np.pi/180)**2,
            0.01**2, 0.01**2, 0.01**2,
            0.5**2,  0.5**2,  0.5**2,
        ])
        self.t_prev: float | None = None
        self.t_last_gnss: float | None = None

    def predict(self, t: float, accel_meas: np.ndarray, gyro_meas: np.ndarray) -> None:
        if self.t_prev is None:
            self.t_prev = t
            return
        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return

        roll, pitch, yaw = self.x[self.ROLL:self.YAW+1]
        b_gyro  = self.x[self.BGX:self.BGZ+1]
        b_accel = self.x[self.BAX:self.BAZ+1]
        R = euler_to_dcm(roll, pitch, yaw)

        accel_unbiased = accel_meas - b_accel
        gyro_unbiased  = gyro_meas  - b_gyro

        pos_new = self.x[self.PX:self.PZ+1] + self.x[self.VX:self.VZ+1] * dt
        vel_new = self.x[self.VX:self.VZ+1] + (R @ accel_unbiased - np.array([0, 0, g])) * dt

        w_mag = np.linalg.norm(gyro_unbiased)
        R_new = R @ expm(skew(gyro_unbiased) * dt) if w_mag > 1e-8 else R
        R_new = normalize_dcm(R_new)
        roll_new, pitch_new, yaw_new = dcm_to_euler(R_new)

        x_new = self.x.copy()
        x_new[self.PX:self.PZ+1] = pos_new
        x_new[self.VX:self.VZ+1] = vel_new
        x_new[self.ROLL:self.YAW+1] = [roll_new, pitch_new, yaw_new]
        self.x = x_new

        F = self._jacobian_F(accel_unbiased, gyro_unbiased, R, dt)
        Q = np.diag(np.concatenate([
            np.diag(Q_pos), np.diag(Q_vel), np.diag(Q_att),
            np.diag(Q_gyro_bias), np.diag(Q_accel_bias)
        ]))
        self.cov = F @ self.cov @ F.T + Q
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.t_prev = t

    def _jacobian_F(self, accel: np.ndarray, gyro: np.ndarray,
                    R: np.ndarray, dt: float) -> np.ndarray:
        F = np.eye(self.STATE_SIZE)
        F[self.PX:self.PZ+1, self.VX:self.VZ+1] = np.eye(3) * dt

        accel_world = R @ accel
        eps = 1e-7
        for i in range(3):
            x_p = self.x.copy()
            x_p[self.ROLL + i] += eps
            R_p = euler_to_dcm(*x_p[self.ROLL:self.YAW+1])
            F[self.VX:self.VZ+1, self.ROLL + i] = (R_p @ accel - accel_world) / eps * dt

        F[self.VX:self.VZ+1, self.BAX:self.BAZ+1] = -R * dt

        w_mag = np.linalg.norm(gyro)
        R_new = R @ expm(skew(gyro) * dt) if w_mag > 1e-8 else R
        _, pitch_new, yaw_new = dcm_to_euler(R_new)
        for i in range(3):
            gyro_p = gyro.copy()
            gyro_p[i] += eps
            w_p = np.linalg.norm(gyro_p)
            R_p = R @ expm(skew(gyro_p) * dt) if w_p > 1e-8 else R
            _, pitch_p, yaw_p = dcm_to_euler(R_p)
            F[self.ROLL:self.YAW+1, self.BGX + i] = [
                0,
                (pitch_p - pitch_new) / eps,
                (yaw_p   - yaw_new)   / eps,
            ]

        return F

    def update_gnss(self, pos_meas: np.ndarray, vel_meas: np.ndarray) -> None:
        z = np.concatenate([pos_meas, vel_meas])
        H = np.zeros((6, self.STATE_SIZE))
        H[0:3, self.PX:self.PZ+1] = np.eye(3)
        H[3:6, self.VX:self.VZ+1] = np.eye(3)

        z_pred = np.concatenate([self.x[self.PX:self.PZ+1], self.x[self.VX:self.VZ+1]])
        y = z - z_pred
        R_meas = np.diag(np.concatenate([np.diag(R_gnss_pos), np.diag(R_gnss_vel)]))
        S = H @ self.cov @ H.T + R_meas
        K = self.cov @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(self.STATE_SIZE) - K @ H
        self.cov = I_KH @ self.cov @ I_KH.T + K @ R_meas @ K.T
        self.cov = 0.5 * (self.cov + self.cov.T)

    def output_row(self, t: float) -> list[Any]:
        """Extract state as output row with quaternion attitude (same layout as EqF cols 0-25)."""
        roll, pitch, yaw = self.x[self.ROLL:self.YAW+1]
        R = euler_to_dcm(roll, pitch, yaw)
        q = ScipyRot.from_matrix(R).as_quat()  # [x,y,z,w]
        qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])

        return [
            t,
            self.x[self.PX], self.x[self.PY], self.x[self.PZ],
            self.x[self.VX], self.x[self.VY], self.x[self.VZ],
            R[0,0], R[0,1], R[0,2],
            R[1,0], R[1,1], R[1,2],
            R[2,0], R[2,1], R[2,2],
            qw, qx, qy, qz,          # cols 16-19
            self.x[self.BGX], self.x[self.BGY], self.x[self.BGZ],  # cols 20-22
            self.x[self.BAX], self.x[self.BAY], self.x[self.BAZ],  # cols 23-25
        ]


# =============================================================================
# GPS Coordinate Conversion
# =============================================================================

def gps_to_ned(lat: float, lon: float, alt: float,
               lat0: float, lon0: float, alt0: float) -> np.ndarray:
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    return np.array([
        dlat * R_EARTH,
        dlon * np.cos(lat0_rad) * R_EARTH,
        -(alt - alt0),
    ])


# =============================================================================
# CSV I/O
# =============================================================================

_C = {
    "t":      0,
    "lon":    1,
    "lat":    2,
    "alt":    3,
    "gps_vn": 4,
    "gps_ve": 5,
    "gps_vd": 6,
    "ax":     9,
    "ay":     10,
    "az":     11,
    "gx":     15,
    "gy":     16,
    "gz":     17,
}

_DATASETS = {
    "full":    ("data/20241011_NIMBUS24_Flight_FC_Data.csv",        "outputs/ekf_output_full.csv"),
    "30s":     ("data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",    "outputs/ekf_output_30s.csv"),
    "1s_loop": ("data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv","outputs/ekf_output_1s_loop.csv"),
}


def run(csv_in: str | None = None, csv_out: str | None = None) -> None:
    """Run EKF on NIMBUS24 FC CSV data."""
    if csv_in is None:
        if DATASET not in _DATASETS:
            raise ValueError(f"Unknown DATASET {DATASET!r}. Choose from: {list(_DATASETS)}")
        csv_in, csv_out_default = _DATASETS[DATASET]
        if csv_out is None:
            csv_out = csv_out_default
    if csv_out is None:
        csv_out = "outputs/ekf_output.csv"

    raw = np.genfromtxt(csv_in, delimiter=",", skip_header=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    print(f"Loaded {len(raw)} rows from {csv_in}")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    gyro_scale = (2000.0 / 32768.0) * (np.pi / 180.0)  # raw ADC → rad/s

    valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
    first = int(np.argmax(valid))
    lat0 = raw[first, _C["lat"]]
    lon0 = raw[first, _C["lon"]]
    alt0 = raw[first, _C["alt"]] / 1000.0

    filt = EKF()
    out = []
    gnss_count = 0
    last_progress_t = 0.0
    gnss_period = 1.0 / GNSS_UPDATE_FREQ_HZ

    for i, row in enumerate(raw):
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        # Sensor→body frame: az→X(N), ay→Y(E), -ax→Z(D)  (confirmed from accel integration vs GNSS)
        gyro  = row[[_C["gz"], _C["gy"], _C["gx"]]] * gyro_scale * np.array([1.0, 1.0, -1.0])
        accel = row[[_C["az"], _C["ay"], _C["ax"]]] * g           * np.array([1.0, 1.0, -1.0])

        if not np.all(np.isfinite(np.concatenate([gyro, accel]))):
            continue

        filt.predict(t, accel, gyro)
        out.append(filt.output_row(t))

        lat = row[_C["lat"]]
        lon = row[_C["lon"]]
        if lat != 0 and lon != 0:
            time_since = gnss_period if filt.t_last_gnss is None else (t - filt.t_last_gnss)
            if filt.t_last_gnss is None or time_since >= gnss_period:
                alt = row[_C["alt"]] / 1000.0
                pos_NED = gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
                vel_NED = row[[_C["gps_vn"], _C["gps_ve"], _C["gps_vd"]]] / 1000.0
                filt.update_gnss(pos_NED, vel_NED)
                filt.t_last_gnss = t
                gnss_count += 1

        if t - last_progress_t >= 30:
            print(f"[PROGRESS] {100*i/len(raw):.1f}% | t={t:.2f}s | GNSS_updates={gnss_count}")
            last_progress_t = t

    out_arr = np.asarray(out)
    header = (
        "t,px,py,pz,vx,vy,vz,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
        "qw,qx,qy,qz,"
        "bgx,bgy,bgz,bax,bay,baz"
    )
    np.savetxt(csv_out, out_arr, delimiter=",", header=header, comments="")
    print(f"Wrote {len(out_arr)} rows to {csv_out}")


if __name__ == "__main__":
    run()
