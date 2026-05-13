"""Extended Kalman Filter for rocket trajectory and attitude estimation using IMU and GNSS."""

import os
import time
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

# =============================================================================
# Configuration
# =============================================================================

# "full"    -> data/20241011_NIMBUS24_Flight_FC_Data.csv
# "30s"     -> data/20241011_NIMBUS24_Flight_FC_Data_30s.csv
# "1s_loop" -> data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv
DATASET = "full"

GNSS_UPDATE_FREQ_HZ = 0.1

# =============================================================================
# Constants
# =============================================================================

g = 9.81
R_EARTH = 6_378_137.0

# Magnetometer reference (WMM2025, Azores)
WMM_DECLINATION = -13.4
WMM_INCLINATION =  56.8
_dec_r = np.radians(WMM_DECLINATION)
_inc_r = np.radians(WMM_INCLINATION)
_wmm_h = np.cos(_inc_r)
_wmm_raw = np.array([_wmm_h * np.cos(_dec_r), _wmm_h * np.sin(_dec_r), np.sin(_inc_r)])
MAG_FIELD_NED = _wmm_raw / np.linalg.norm(_wmm_raw)

# Mag sensor→body axis config
MAG_AXIS_ORDER = np.array([0, 2, 1])
MAG_AXIS_SIGNS = np.array([1.0, -1.0, -1.0])

# Process noise
Q_pos        = np.eye(3) * 1e-2
Q_vel        = np.eye(3) * 1e-1
Q_att        = np.eye(3) * 1e-2
Q_gyro_bias  = np.eye(3) * (1e-2)**2
Q_accel_bias = np.eye(3) * (1e-2)**2

# Measurement noise
R_gnss_pos = np.eye(3) * 1.0
R_gnss_vel = np.eye(3) * 5.0
R_mag_var  = 100.0   # scalar yaw noise (rad²)

# =============================================================================
# Offset frame — avoids ZYX Euler singularity at pitch≈90°
#
# R_OFFSET = Ry(-90°): maps NED frame so "rocket pointing up" → pitch_off ≈ 0°
# Proof: a rocket with R_body_to_ned = Ry(90°) gives R_body_to_off = I (pitch=0°).
# =============================================================================

R_OFFSET     = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])   # Ry(-90°)
G_OFF        = R_OFFSET @ np.array([0., 0., g])   # gravity in offset frame = [-g, 0, 0]
MAG_FIELD_OFF = R_OFFSET @ MAG_FIELD_NED           # mag reference in offset frame


# =============================================================================
# Rotation Utilities
# =============================================================================

def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles → body-to-world DCM."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])


def euler_kin(roll: float, pitch: float) -> np.ndarray:
    """E matrix: euler_dot = E @ omega_body (ZYX kinematics, yaw-independent)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    tp = sp / (cp + 1e-15)
    return np.array([
        [1.,  sr * tp,  cr * tp],
        [0.,  cr,      -sr     ],
        [0.,  sr / (cp + 1e-15), cr / (cp + 1e-15)],
    ])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])


def quat_to_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion [w,x,y,z] → (roll, pitch, yaw) in degrees, ZYX convention."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw  = float(np.arctan2(-R[0, 1], R[1, 1]))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def dcm_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """DCM → (roll, pitch, yaw) radians, ZYX convention."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = 0.0
        yaw  = float(np.arctan2(-R[0, 1], R[1, 1]))
    return roll, pitch, yaw


def triad_attitude(accel_body: np.ndarray, mag_body: np.ndarray) -> "np.ndarray | None":
    """Compute body→NED rotation matrix via TRIAD from accelerometer + magnetometer."""
    g_body = -accel_body
    g_n = np.linalg.norm(g_body)
    m_n = np.linalg.norm(mag_body)
    if g_n < 1.0 or m_n < 0.01:
        return None
    g_b = g_body / g_n
    m_b = mag_body / m_n

    g_ned = np.array([0.0, 0.0, 1.0])
    m_ned = MAG_FIELD_NED.flatten()

    cross_b = np.cross(g_b, m_b)
    cross_n = np.cross(g_ned, m_ned)
    if np.linalg.norm(cross_b) < 1e-6 or np.linalg.norm(cross_n) < 1e-6:
        return None
    t2_b = cross_b / np.linalg.norm(cross_b)
    t2_n = cross_n / np.linalg.norm(cross_n)

    T_body = np.column_stack([g_b,   t2_b,   np.cross(g_b,   t2_b)])
    T_ned  = np.column_stack([g_ned, t2_n,   np.cross(g_ned, t2_n)])
    return T_ned @ T_body.T


# =============================================================================
# EKF Filter — normal Euler-state EKF in offset frame
# =============================================================================

class EKF:
    """Normal EKF with Euler angle state parameterized in the offset frame.

    The offset frame is NED rotated by Ry(-90°), which maps the rocket's typical
    ~84.5° pitch in NED to ~-5.5° pitch in the offset frame, safely away from
    the ZYX singularity at ±90°.

    State: x = [pos_off(3), vel_off(3), roll_off, pitch_off, yaw_off, bg(3), ba(3)]
    """

    P  = slice(0, 3)    # position in offset frame
    V  = slice(3, 6)    # velocity in offset frame
    AT = slice(6, 9)    # [roll_off, pitch_off, yaw_off]
    BG = slice(9, 12)   # gyro bias (body frame)
    BA = slice(12, 15)  # accel bias (body frame)
    STATE_SIZE = 15

    ROLL  = 6
    PITCH = 7
    YAW   = 8

    def __init__(self) -> None:
        self.x: np.ndarray = np.zeros(15)
        self.cov: np.ndarray = np.diag([
            1.0**2,  1.0**2,  1.0**2,
            10.0**2, 10.0**2, 10.0**2,
            1.0**2,  1.0**2,  1.0**2,
            0.1**2,  0.1**2,  0.1**2,
            0.1**2,  0.1**2,  0.1**2,
        ])
        self.t_prev:      float | None = None
        self.t_last_gnss: float | None = None

    def _dcm(self) -> np.ndarray:
        """Body-to-offset DCM from current state."""
        return euler_to_dcm(self.x[self.ROLL], self.x[self.PITCH], self.x[self.YAW])

    def predict(self, t: float, accel_meas: np.ndarray, gyro_meas: np.ndarray) -> None:
        if self.t_prev is None:
            self.t_prev = t
            return
        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return

        e = self.x[self.AT].copy()
        R = euler_to_dcm(*e)
        a = accel_meas - self.x[self.BA]
        w = gyro_meas  - self.x[self.BG]
        E = euler_kin(e[0], e[1])

        # Nominal state propagation
        self.x[self.P]  += self.x[self.V] * dt
        self.x[self.V]  += (R @ a + G_OFF) * dt
        self.x[self.AT] += E @ w * dt

        # F matrix
        eps = 1e-6
        F = np.eye(15)
        F[0:3, 3:6]   = np.eye(3) * dt   # dp/dv
        F[3:6, 12:15] = -R * dt           # dv/dba
        F[6:9,  9:12] = -E * dt           # deuler/dbg

        # Finite-diff partials for attitude-coupled rows
        for i in range(3):
            de = np.zeros(3); de[i] = eps
            Rp = euler_to_dcm(*(e + de)); Rm = euler_to_dcm(*(e - de))
            F[3:6, 6 + i] += (Rp @ a - Rm @ a) / (2 * eps) * dt      # dv/deuler
            ep = e + de; em = e - de
            Ep = euler_kin(ep[0], ep[1]); Em = euler_kin(em[0], em[1])
            F[6:9, 6 + i] += (Ep @ w - Em @ w) / (2 * eps) * dt      # deuler/deuler

        Q = np.diag([
            *([Q_pos[0, 0]]        * 3),
            *([Q_vel[0, 0]]        * 3),
            *([Q_att[0, 0]]        * 3),
            *([Q_gyro_bias[0, 0]]  * 3),
            *([Q_accel_bias[0, 0]] * 3),
        ])
        self.cov = F @ self.cov @ F.T + Q * dt
        self.cov = (self.cov + self.cov.T) * 0.5
        self.t_prev = t

    def update_gnss(self, pos_ned: np.ndarray, vel_ned: np.ndarray) -> None:
        pos_meas = R_OFFSET @ pos_ned
        vel_meas = R_OFFSET @ vel_ned

        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        y = np.concatenate([pos_meas - self.x[self.P], vel_meas - self.x[self.V]])
        R_meas = np.diag([*np.diag(R_gnss_pos), *np.diag(R_gnss_vel)])
        S = H @ self.cov @ H.T + R_meas
        K = self.cov @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        IKH = np.eye(15) - K @ H
        self.cov = IKH @ self.cov @ IKH.T + K @ R_meas @ K.T
        self.cov = (self.cov + self.cov.T) * 0.5

    def update_mag(self, mag_body: np.ndarray) -> None:
        """Yaw-only update — SO3.log innovation projected onto body-Z axis."""
        mag_n = np.linalg.norm(mag_body)
        if mag_n < 1e-6:
            return
        mag_body = mag_body / mag_n

        e = self.x[self.AT].copy()
        R = euler_to_dcm(*e)
        y_hat = R.T @ MAG_FIELD_OFF
        y_hat /= np.linalg.norm(y_hat) + 1e-8

        def delta_yaw(yh: np.ndarray) -> float:
            cross = np.cross(yh, mag_body)
            dot   = float(np.clip(np.dot(yh, mag_body), -1.0, 1.0))
            angle = np.arccos(dot)
            cn    = np.linalg.norm(cross)
            d3    = (cross / cn) * angle if cn > 1e-10 else np.zeros(3)
            return float(d3[2])  # body-Z component

        dyw = delta_yaw(y_hat)
        if abs(dyw) < 0.01:
            return

        eps = 1e-6
        H = np.zeros((1, 15))
        for i in range(3):
            de = np.zeros(3); de[i] = eps
            Rp = euler_to_dcm(*(e + de))
            yp = Rp.T @ MAG_FIELD_OFF
            yp /= np.linalg.norm(yp) + 1e-8
            H[0, 6 + i] = (delta_yaw(yp) - dyw) / eps

        R_m = np.array([[R_mag_var]])
        S = H @ self.cov @ H.T + R_m
        K = self.cov @ H.T @ np.linalg.inv(S)

        self.x += (K * dyw).flatten()
        IKH = np.eye(15) - K @ H
        self.cov = IKH @ self.cov @ IKH.T + K @ R_m @ K.T
        self.cov = (self.cov + self.cov.T) * 0.5

    def output_row(self, t: float) -> list[Any]:
        R_off = self._dcm()
        R_ned = R_OFFSET.T @ R_off          # body-to-NED
        p_ned = R_OFFSET.T @ self.x[self.P]
        v_ned = R_OFFSET.T @ self.x[self.V]
        sq = ScipyRot.from_matrix(R_ned).as_quat()  # [x,y,z,w]
        qw, qx, qy, qz = float(sq[3]), float(sq[0]), float(sq[1]), float(sq[2])
        return [
            t,
            p_ned[0], p_ned[1], p_ned[2],
            v_ned[0], v_ned[1], v_ned[2],
            R_ned[0,0], R_ned[0,1], R_ned[0,2],
            R_ned[1,0], R_ned[1,1], R_ned[1,2],
            R_ned[2,0], R_ned[2,1], R_ned[2,2],
            qw, qx, qy, qz,
            self.x[self.BG][0], self.x[self.BG][1], self.x[self.BG][2],
            self.x[self.BA][0], self.x[self.BA][1], self.x[self.BA][2],
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
    "mx":     18,
    "my":     19,
    "mz":     20,
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

    # TRIAD initialization
    MIN_PRE_INIT = 5
    accel_accum = np.zeros(3)
    accel_n = 0
    attitude_initialized = False
    for _row in raw:
        if not np.isfinite(_row[_C["t"]]):
            continue
        _accel = _row[[_C["ax"], _C["ay"], _C["az"]]] * g
        _mag_raw = _row[[_C["mx"], _C["my"], _C["mz"]]]
        if not np.all(np.isfinite(_accel)) or not np.all(np.isfinite(_mag_raw)):
            continue
        accel_accum += _accel
        accel_n += 1
        _mag = _mag_raw[MAG_AXIS_ORDER] * MAG_AXIS_SIGNS
        _mag_n = np.linalg.norm(_mag)
        if _mag_n > 1e-6 and accel_n >= MIN_PRE_INIT:
            R_init = triad_attitude(accel_accum / accel_n, _mag / _mag_n)
            if R_init is not None:
                # R_init is body-to-NED; convert to offset frame for EKF state
                R_off_init = R_OFFSET @ R_init
                roll0, pitch0, yaw0 = dcm_to_euler(R_off_init)
                filt.x[filt.ROLL]  = roll0
                filt.x[filt.PITCH] = pitch0
                filt.x[filt.YAW]   = yaw0
                attitude_initialized = True
                # Print in NED frame for readability
                rn, pn, yn = dcm_to_euler(R_init)
                print(f"TRIAD init ({accel_n} accel samples): "
                      f"roll={np.degrees(rn):.1f}°  pitch={np.degrees(pn):.1f}°  yaw={np.degrees(yn):.1f}°")
            break

    prev_mag = None

    _prop_total = 0.0
    _prop_count = 0

    for i, row in enumerate(raw):
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        gyro  = row[[_C["gx"], _C["gy"], _C["gz"]]] * (np.pi / 180.0)
        accel = row[[_C["ax"], _C["ay"], _C["az"]]] * g

        if not np.all(np.isfinite(np.concatenate([gyro, accel]))):
            continue

        _t0 = time.perf_counter()
        filt.predict(t, accel, gyro)
        _prop_total += time.perf_counter() - _t0
        _prop_count += 1

        # Magnetometer update
        mag_raw = row[[_C["mx"], _C["my"], _C["mz"]]]
        if np.all(np.isfinite(mag_raw)) and attitude_initialized:
            mag = mag_raw[MAG_AXIS_ORDER] * MAG_AXIS_SIGNS
            mag_norm = np.linalg.norm(mag)
            if mag_norm > 1e-6:
                mag_unit = mag / mag_norm
                if prev_mag is None or not np.allclose(prev_mag, mag_unit):
                    filt.update_mag(mag_unit)
                    prev_mag = mag_unit

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
    if _prop_count > 0:
        avg_us = _prop_total / _prop_count * 1e6
        print(f"Propagate: {_prop_count} steps, avg {avg_us:.1f} µs/step  (total {_prop_total*1e3:.1f} ms)")


if __name__ == "__main__":
    run()
