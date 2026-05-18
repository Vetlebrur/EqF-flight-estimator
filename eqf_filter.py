"""TG-EqF inertial/GNSS filter on SE₂(3) using pylie."""

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as ScipyRot
from pylie import SO3, SE23

ref_path = Path(__file__).parent / "eqf-reference"
sys.path.insert(0, str(ref_path))
utils_path = ref_path / "Utils"
sys.path.insert(0, str(utils_path))
from matrix_math import *
from Symmetries.Calibrated.SE23_se23.Symmetry import SymGroup, State, InputSpace, stateAction, velocityAction

# =============================================================================
# Configuration
# =============================================================================

# "full"    -> data/20241011_NIMBUS24_Flight_FC_Data.csv          (complete flight)
# "30s"     -> data/20241011_NIMBUS24_Flight_FC_Data_30s.csv      (first 30 s)
# "1s_loop" -> data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv  (first 1 s looped for 30 s)
DATASET = "full"

GNSS_UPDATE_FREQ_HZ = 0.1   # GNSS update frequency (Hz) — update every 1/f seconds
MAG_UPDATE_FREQ_HZ  = 1.0   # Magnetometer update frequency (Hz)

# --- Update toggles ---
USE_GNSS_UPDATE       = True
USE_MAG_UPDATE        = True
USE_MA_FILTER         = False   # Moving average pre-filter on gyro/accel (window=5)
USE_EULER_DISCR       = False  # Euler discretization of A (Φ=I+A·dt) instead of expm(A·dt)
USE_RESET             = True   # post-update covariance reset on Lie group manifold (arXiv:2309.03765)
# =============================================================================
# Constants and Physical Parameters
# =============================================================================

g = 9.81  # Gravitational acceleration (m/s²)
G = np.zeros((5, 5))
G[2, 3] = g
N = np.zeros((5, 5))
N[3, 4] = 1.0

# =============================================================================
# Noise Parameters
# =============================================================================
# --- State Covariance (P) ---
P_0_blocks = [
    (1)**2 * np.eye(3),       # [0:3] attitude error (rad²) 
    (10.0)**2 * np.eye(3),       # [3:6] velocity error (m/s)²
    (1.0)**2 * np.eye(3),      # [6:9] position error (m²)
    (0.01)**2 * np.eye(3),     # [9:12] gyro bias error (rad/s)²
    (0.01)**2 * np.eye(3),      # [12:15] accel bias error (m/s²)² - tight initial
    1e-9 * np.eye(3)           # [15:18] virtual accel bias (frozen at zero)
]

# --- Process Noise (Q) ---
Q_gyro_var = 1e-3               # [0:3] rotation process noise
Q_accel_var = 1e-1                # [3:6] velocity process noise
Q_virt_var = 1e-2                # [6:9] position process noise
Q_gyro_bias_var = (1e-6)**2     # [9:12] gyro bias random walk (very tight)
Q_accel_bias_var = (1e-5)**2    # [12:15] accel bias random walk (very tight)
Q_virtual_bias_var = 1e-7      # [15:18] virtual accel bias (frozen)

# --- Measurement Noise (R) ---

# GNSS measurement noise
R_gnss_pos_var = 1            # GNSS position measurement variance (m²)
R_gnss_vel_var = 5.0           # GNSS velocity measurement variance (m/s)²

# Magnetometer noise: innovation is SO3 log of R_triad @ R_hat^T (radians).
# VN200 calibrated AHRS uses 0.04 rad² (~11.5° std). Raw TRIAD from accel+mag is noisier.
R_mag_var = 10  # rad²  (~57° std per axis); ANIS tuned: raw TRIAD is noisy

# =============================================================================
# Magnetometer configuration
# =============================================================================
# MAG_AXIS_ORDER: which raw sensor axis goes to body [x, y, z]
#   default [0,1,2] = identity (mx→x, my→y, mz→z)
# MAG_AXIS_SIGNS: sign applied to each body axis after permutation
#   default [1, 1, 1]; negate if sensor axis is mounted in opposite direction
MAG_AXIS_ORDER = np.array([0, 2, 1])   # sensor: mx→body X, mz→body Y, my→body Z
MAG_AXIS_SIGNS = np.array([1.0, -1.0, -1.0])   # tuned: confirmed by FC/GNSS yaw alignment

# WMM2025 for NIMBUS24 launch site (Ribeira Grande, Azores, 39.39°N, 8.29°W)
# Declination: -13.4° (West), Inclination: 56.8°, Magnitude: ~47000 nT
WMM_DECLINATION = -13.4  # degrees (West = negative)
WMM_INCLINATION = 56.8   # degrees (Down from horizontal)
WMM_MAGNITUDE = 47000.0  # nanoTesla

# Compute WMM field vector in NED from declination and inclination
dec_rad = np.radians(WMM_DECLINATION)
inc_rad = np.radians(WMM_INCLINATION)
wmm_h = WMM_MAGNITUDE * np.cos(inc_rad)  # Horizontal component
wmm_north = wmm_h * np.cos(dec_rad)
wmm_east = wmm_h * np.sin(dec_rad)
wmm_down = WMM_MAGNITUDE * np.sin(inc_rad)
_mag_raw = np.array([wmm_north, wmm_east, wmm_down])
MAG_FIELD_NED = _mag_raw / np.linalg.norm(_mag_raw)

# =============================================================================
# Helpers
# =============================================================================

def col(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)

def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def blockdiag(*arrs: np.ndarray) -> np.ndarray:
    """Create block diagonal matrix from array list."""
    n = sum(a.shape[0] for a in arrs)
    m = sum(a.shape[1] for a in arrs)
    out = np.zeros((n, m))
    r = 0
    c = 0
    for a in arrs:
        rr, cc = a.shape
        out[r : r + rr, c : c + cc] = a
        r += rr
        c += cc
    return out

# =============================================================================
# Symmetry Lift
# =============================================================================

def calculate_lift(xi: State, U: InputSpace) -> np.ndarray:
    W     = U.as_W_mat()
    B     = SE23.wedge(xi.b)
    T     = xi.T.as_matrix()
    T_inv = xi.T.inv().as_matrix()

    Lambda_1 = SE23.vee(W - B + N + T_inv @ (G - N) @ T)

    Lambda = np.zeros((18, 1))
    Lambda[0:9]  = Lambda_1
    Lambda[9:18] = SE23.adjoint(xi.b) @ Lambda_1 - U.tau
    return Lambda

# =============================================================================
# Filter
# =============================================================================


class TGEqF:
    """TG-EqF Inertial/GNSS Filter."""

    def initialize_attitude_triad(self, mag_body: np.ndarray) -> None:
        MIN_INIT_SAMPLES = 5
        if self.attitude_initialized or self._init_accel_n < MIN_INIT_SAMPLES:
            return

        accel_avg = self._init_accel_accum / self._init_accel_n
        g_body = -accel_avg
        g_norm = np.linalg.norm(g_body)
        if g_norm < 1.0:  # sanity check: at least 1 m/s² average
            return
        g_body = g_body / g_norm

        m_body = mag_body / (np.linalg.norm(mag_body) + 1e-10)

        g_ned = np.array([0.0, 0.0, 1.0])
        m_ned = MAG_FIELD_NED.flatten()

        cross_b = np.cross(g_body, m_body)
        cross_n = np.cross(g_ned, m_ned)
        cn = np.linalg.norm(cross_n)
        cb = np.linalg.norm(cross_b)
        if cn < 1e-6 or cb < 1e-6:
            return  # degenerate: gravity and mag nearly collinear

        t2_b = cross_b / cb
        t2_n = cross_n / cn

        T_body = np.column_stack([g_body,            t2_b,            np.cross(g_body, t2_b)])
        T_ned  = np.column_stack([g_ned,             t2_n,            np.cross(g_ned,  t2_n)])

        R = T_ned @ T_body.T  # maps body → NED

        se23_init = SE23(R)  # type: ignore[arg-type]
        self.X_hat = SymGroup(se23_init, np.zeros((5, 5)))
        self.attitude_initialized = True

        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        print(f"TRIAD init ({self._init_accel_n} accel samples): "
              f"roll={roll:.1f}°  pitch={pitch:.1f}°  yaw={yaw:.1f}°")

    def __init__(self):
        self.X_hat = SymGroup.identity()
        self.xi_0 = State()
        self.t_prev = None
        self.t_last_gnss = None
        self.t_last_imu = None

        self.attitude_initialized = False
        self._init_accel_accum = np.zeros(3)
        self._init_accel_n = 0
        self._last_accel: np.ndarray | None = None

        self.ma_window = 10
        self.gyro_buffer = []
        self.accel_buffer = []

        self.mag_meas_0 = MAG_FIELD_NED.reshape(-1, 1)
        self.mag_update_count = 0
        self.mag_euler: np.ndarray = np.full(3, float('nan'))

        self.anis_values: list[float] = []
        self.update_times: list[tuple[str, float, float]] = []

        self.Sigma = blockdiag(*P_0_blocks)
        self.R_gnss = blockdiag(np.eye(3) * R_gnss_pos_var, np.eye(3) * R_gnss_vel_var)
        self.R_mag  = np.eye(3) * R_mag_var
        self.Q = blockdiag(
            np.eye(3) * Q_gyro_var,
            np.eye(3) * Q_accel_var,
            np.eye(3) * Q_virt_var,
            np.eye(3) * Q_gyro_bias_var,
            np.eye(3) * Q_accel_bias_var,
            np.eye(3) * Q_virtual_bias_var,
        )

    def xi_hat(self) -> State:
        xi = stateAction(self.X_hat, self.xi_0)
        # Freeze b_mu at zero (unobservable, prevents affecting dynamics)
        xi.b[6:9] = 0
        return xi

    def _apply_moving_average(self, measurement: np.ndarray, buffer: list[np.ndarray]) -> np.ndarray:
        """Apply moving average filter with window size of self.ma_window."""
        buffer.append(measurement.flatten())
        if len(buffer) > self.ma_window:
            buffer.pop(0)
        return np.mean(buffer, axis=0).reshape(-1, 1)

    # =========================================================================
    # Propagation
    # =========================================================================

    def propagate(self, t: float, gyro: np.ndarray, accel: np.ndarray) -> None:
        gyro = col(gyro)
        accel = col(accel)

        if self.t_prev is None:
            self.t_prev = t
            return

        if not self.attitude_initialized:
            self._init_accel_accum += accel.flatten()
            self._init_accel_n += 1
            self.t_prev = t
            return

        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return

        if USE_MA_FILTER:
            gyro = self._apply_moving_average(gyro, self.gyro_buffer)
            accel = self._apply_moving_average(accel, self.accel_buffer)

        self._last_accel = accel.flatten()

        U = InputSpace()
        U.w = SO3.wedge(gyro)
        U.a = accel

        xi_hat = self.xi_hat()
        U.mu = xi_hat.b[6:9, 0:1]
        U.tau = np.zeros((9, 1))

        lift = calculate_lift(xi_hat, U)

        A = self.calculate_A(U)
        Phi = np.eye(18) + A * dt if USE_EULER_DISCR else expm(A * dt)
        Bt_block = self.X_hat.B.Adjoint()
        Bt = np.zeros((18, 18))
        Bt[0:9,  0:9]  = Bt_block
        Bt[9:18, 9:18] = Bt_block

        self.X_hat = self.X_hat * SymGroup.exp(lift * dt)  # type: ignore[attr-defined]
        self.Sigma = Phi @ self.Sigma @ Phi.T + Bt @ self.Q @ Bt.T * dt
        self.Sigma = sym(self.Sigma)

        # Only decompose when diagonal hints at overflow (avoids per-step eigh)
        max_eig_threshold = 1e8
        if np.max(np.diag(self.Sigma)) > max_eig_threshold:
            eigs, V = np.linalg.eigh(self.Sigma)
            eigs_clipped = np.clip(eigs, 0, max_eig_threshold)
            self.Sigma = V @ np.diag(eigs_clipped) @ V.T
            self.Sigma = sym(self.Sigma)


        self.t_prev = t
        self.t_last_imu = t

    # =========================================================================
    # Update functions
    # =========================================================================

    def magnetometer_update(self, mag: np.ndarray, t: float | None = None) -> None:
        mag = np.asarray(mag, dtype=float).reshape(3)
        mag_n = np.linalg.norm(mag)
        if mag_n < 1e-6 or self._last_accel is None:
            return

        # Quasi-static gate: skip during powered flight where accel ≠ gravity
        accel_norm = np.linalg.norm(self._last_accel)
        if abs(accel_norm - g) > 3.0:
            return

        # Build R_triad (body→NED) via TRIAD from accel + mag
        g_body = -self._last_accel / accel_norm  # gravity direction in body frame
        m_body = mag / mag_n

        g_ned = np.array([0.0, 0.0, 1.0])
        m_ned = MAG_FIELD_NED.flatten()

        cross_b = np.cross(g_body, m_body)
        cross_n = np.cross(g_ned, m_ned)
        cb, cn = np.linalg.norm(cross_b), np.linalg.norm(cross_n)
        if cb < 1e-6 or cn < 1e-6:
            return  # degenerate: gravity and mag nearly collinear

        t2_b = cross_b / cb
        t2_n = cross_n / cn
        T_body = np.column_stack([g_body, t2_b, np.cross(g_body, t2_b)])
        T_ned  = np.column_stack([g_ned,  t2_n, np.cross(g_ned,  t2_n)])
        R_triad = T_ned @ T_body.T

        # Project onto SO(3)
        U, _, Vt = np.linalg.svd(R_triad)
        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1
        R_triad = U @ Vt

        # SO3 log innovation: delta = log(R_triad @ R_hat^T)  (same form as VN200 MagUpdate)
        xi_hat = self.xi_hat()
        R_hat = xi_hat.T.R().as_matrix()
        R_err = R_triad @ R_hat.T
        U2, _, Vt2 = np.linalg.svd(R_err)
        if np.linalg.det(U2 @ Vt2) < 0:
            U2[:, -1] *= -1
        R_err = U2 @ Vt2
        delta = ScipyRot.from_matrix(R_err).as_rotvec().reshape(3, 1)

        # C = [I_3 | 0_{3×15}]  (same as VN200 MagUpdate)
        C = np.zeros((3, 18))
        C[0:3, 0:3] = np.eye(3)

        S = C @ self.Sigma @ C.T + self.R_mag
        K = self.Sigma @ C.T @ np.linalg.inv(S)
        K_delta = K @ delta

        if t is not None:
            anis_raw = float(np.squeeze(delta.T @ np.linalg.inv(S) @ delta))
            self.update_times.append(("mag", t, anis_raw / 3.0))

        self.X_hat = SymGroup.exp(K_delta) * self.X_hat
        IKC = np.eye(18) - K @ C
        self.Sigma = sym(IKC @ self.Sigma @ IKC.T + K @ self.R_mag @ K.T)
        if USE_RESET:
            self._reset(K_delta)

        euler_zyx = ScipyRot.from_matrix(self.xi_hat().T.R().as_matrix()).as_euler('ZYX')
        self.mag_euler = np.array([euler_zyx[2], euler_zyx[1], euler_zyx[0]])
        self.mag_update_count += 1

    def GNSS_update(self, pos_NED: np.ndarray, vel_NED: np.ndarray, t: float | None = None) -> None:
        xi_hat = self.xi_hat()
        pos_est = xi_hat.T.w().as_vector().flatten()
        vel_est = xi_hat.T.x().as_vector().flatten()

        delta_u = np.hstack([pos_NED - pos_est, vel_NED - vel_est]).reshape(-1, 1)

        C = self.calculate_C_gnss()
        S = C @ self.Sigma @ C.T + self.R_gnss
        K = self.Sigma @ C.T @ np.linalg.inv(S)
        K_delta = K @ delta_u
        Delta = K_delta

        anis_raw = self.compute_anis(delta_u, S)
        if t is not None and anis_raw is not None:
            self.update_times.append(('gnss', t, anis_raw / delta_u.size))

        self.X_hat = SymGroup.exp(Delta) * self.X_hat  # type: ignore[attr-defined]
        IKC = np.eye(18) - K @ C
        self.Sigma = sym(IKC @ self.Sigma @ IKC.T + K @ self.R_gnss @ K.T)

        if USE_RESET:
            self._reset(K_delta)
        if t is not None:
            self.t_last_gnss = t

    # =========================================================================
    # Filter Diagnostics
    # =========================================================================

    def compute_anis(self, innovation: Any, S: Any) -> float | None:
        try:
            inn: np.ndarray = np.asarray(innovation, dtype=float)
            s: np.ndarray = np.asarray(S, dtype=float)
            S_inv: np.ndarray = np.linalg.inv(s)
            anis_mat: np.ndarray = inn.T @ S_inv @ inn
            anis = float(np.squeeze(anis_mat))
            self.anis_values.append(anis)
            return anis
        except (np.linalg.LinAlgError, ValueError):
            return None

    # =========================================================================

    def _grp_adj(self, l: np.ndarray) -> np.ndarray:
        ad = np.zeros((18, 18))
        ad[0:9,  0:9]  = SE23.adjoint(l[0:9])
        ad[9:18, 0:9]  = SE23.adjoint(l[9:18])
        ad[9:18, 9:18] = SE23.adjoint(l[0:9])
        return ad

    def _reset(self, K_delta: np.ndarray) -> None:
        """Post-update covariance reset. Left Jacobian via expm: J_l = expm(½ grp_adj(K·δ))."""
        J = expm(0.5 * self._grp_adj(K_delta))
        self.Sigma = sym(J @ self.Sigma @ J.T)

    # =========================================================================

    def output_row(self, t: float) -> list[float]:
        """Extract state as output row [t, p, v, R, euler, b_gyro, b_accel]."""
        xi_hat = self.xi_hat()
        R = xi_hat.T.R().as_matrix()
        v = col(xi_hat.T.x().as_vector())
        p = col(xi_hat.T.w().as_vector())
        b = xi_hat.b

        return [
            t,
            p[0, 0], p[1, 0], p[2, 0],
            v[0, 0], v[1, 0], v[2, 0],
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2],
            *(b[i, 0] if b.shape[0] > i else 0.0 for i in range(9)),
            *self.mag_euler.tolist(),
        ]
    
    # =============================================================================
    # Propagation Jacobians
    # =============================================================================

    def calculate_A(self, u: InputSpace) -> np.ndarray:
        u_0 = velocityAction(self.X_hat.inv(), u)
        _2A =  np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [SO3.wedge(col((0,0,g))), np.zeros((3, 3)), np.zeros((3,3))],
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))]
        ])
        w_vec = u_0.as_W_vec()
        g_vec = SE23.vee(G)
        At = np.zeros((18, 18))
        At[0:9, 0:9] = _2A
        At[9:18, 9:18] = SE23.adjoint(w_vec + g_vec)
        At[0:9, 9:18] = np.eye(9)

        return At

    def calculate_C_mag(self, y_hat: np.ndarray) -> np.ndarray:
        Ct = np.zeros((3, 18))
        y_wedge = SO3.wedge(y_hat.reshape(3, 1))
        Ct[0:3, 0:3] = -(y_wedge @ y_wedge)   # = I - y_hat @ y_hat.T
        return Ct

    def calculate_C_gnss(self) -> np.ndarray:
        Ct = np.zeros((6, 18))
        xi_hat = self.xi_hat()
        Ct[0:3, 0:3] = -SO3.wedge(xi_hat.T.w().as_vector())
        Ct[0:3, 6:9] = np.eye(3)
        Ct[3:6, 0:3] = -SO3.wedge(xi_hat.T.x().as_vector())
        Ct[3:6, 3:6] = np.eye(3)
        return Ct
    
# =============================================================================
# CSV I/O
# =============================================================================

# NIMBUS24 FC CSV column indices
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
    "mx": 18,
    "my": 19,
    "mz": 20,
    "roll_fc": 29,
    "pitch_fc": 30,
    "yaw_fc": 31,
}
R_EARTH = 6_378_137.0


def _gps_to_ned(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)
    return np.array([north, east, down])

def run(csv_in: str | None = None, csv_out: str = "outputs/tg_eqf_output.csv",
        mag_axis_order: np.ndarray | None = None,
        mag_axis_signs: np.ndarray | None = None,
        r_mag_var: float | None = None,
        gnss_freq_hz: float | None = None,
        mag_freq_hz: float | None = None,
        use_mag_update: bool | None = None,
        silent: bool = False) -> dict[str, float]:
    """Run filter on NIMBUS24 FC CSV data. Returns RMSE dict (keys: roll, pitch, yaw, deg)."""
    axis_order = MAG_AXIS_ORDER if mag_axis_order is None else mag_axis_order
    axis_signs = MAG_AXIS_SIGNS if mag_axis_signs is None else mag_axis_signs
    _gnss_freq = GNSS_UPDATE_FREQ_HZ if gnss_freq_hz is None else gnss_freq_hz
    _mag_freq  = MAG_UPDATE_FREQ_HZ  if mag_freq_hz  is None else mag_freq_hz
    _use_mag   = USE_MAG_UPDATE if use_mag_update is None else use_mag_update

    # Select data source based on configuration
    if csv_in is None:
        _tag = ("_ma" if USE_MA_FILTER else "") + ("_euler" if USE_EULER_DISCR else "")
        _datasets = {
            "full":    ("data/20241011_NIMBUS24_Flight_FC_Data.csv",         f"outputs/tg_eqf_output_full{_tag}.csv"),
            "30s":     ("data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",      f"outputs/tg_eqf_output_30s{_tag}.csv"),
            "1s_loop": ("data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",  f"outputs/tg_eqf_output_1s_loop{_tag}.csv"),
        }
        if DATASET not in _datasets:
            raise ValueError(f"Unknown DATASET {DATASET!r}. Choose from: {list(_datasets)}")
        csv_in, csv_out = _datasets[DATASET]

    raw = np.genfromtxt(csv_in, delimiter=",", skip_header=1)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if not silent:
        print(f"Loaded {len(raw)} rows")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
    first = np.argmax(valid)
    lat0 = raw[first, _C["lat"]]
    lon0 = raw[first, _C["lon"]]
    alt0 = raw[first, _C["alt"]] / 1000.0

    filt = TGEqF()
    if r_mag_var is not None:
        filt.R_mag = np.eye(3) * r_mag_var
    out = []
    fc_att_list: list[list[float]] = []
    prev_mag = None
    t_last_mag = None
    gnss_count = 0
    last_progress_t = 0

    gyro_scale_factor = np.pi / 180.0

    MIN_PRE_INIT = 5
    for _row in raw:
        if not np.isfinite(_row[_C["t"]]):
            continue
        _accel = _row[[_C["ax"], _C["ay"], _C["az"]]] * g
        _mag_raw = _row[[_C["mx"], _C["my"], _C["mz"]]]
        if not np.all(np.isfinite(_accel)) or not np.all(np.isfinite(_mag_raw)):
            continue
        filt._init_accel_accum += _accel
        filt._init_accel_n += 1
        _mag = _mag_raw[axis_order] * axis_signs
        _mag_n = np.linalg.norm(_mag)
        if _mag_n > 1e-6 and filt._init_accel_n >= MIN_PRE_INIT:
            filt.initialize_attitude_triad(_mag / _mag_n)
            break

    _prop_total = 0.0
    _prop_count = 0

    for i, row in enumerate(raw):
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        gyro  = row[[_C["gx"], _C["gy"], _C["gz"]]] * gyro_scale_factor
        accel = row[[_C["ax"], _C["ay"], _C["az"]]] * g

        if not np.all(np.isfinite(np.concatenate([gyro, accel]))):
            continue

        _t0 = time.perf_counter()
        filt.propagate(t, gyro, accel)
        _prop_total += time.perf_counter() - _t0
        _prop_count += 1

        mag_raw = row[[_C["mx"], _C["my"], _C["mz"]]]
        mag = mag_raw[axis_order] * axis_signs
        mag_norm = np.linalg.norm(mag)
        if mag_norm > 1e-6:
            mag = mag / mag_norm
        if not filt.attitude_initialized:
            filt.initialize_attitude_triad(mag)

        _mag_rate_ok = t_last_mag is None or (t - t_last_mag) >= 1.0 / _mag_freq
        if _use_mag and filt.attitude_initialized and _mag_rate_ok and (prev_mag is None or not np.allclose(prev_mag, mag)):
            filt.magnetometer_update(mag, t=t)
            prev_mag = mag
            t_last_mag = t

        lat = row[_C["lat"]]
        lon = row[_C["lon"]]
        if USE_GNSS_UPDATE and lat != 0 and lon != 0:
            gnss_update_period = 1.0 / _gnss_freq
            rate_ok = filt.t_last_gnss is None or (t - filt.t_last_gnss) >= gnss_update_period
            if rate_ok:
                alt = row[_C["alt"]] / 1000.0
                pos_NED = _gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
                vel_NED = np.array([row[_C["gps_vn"]], row[_C["gps_ve"]], row[_C["gps_vd"]]]) / 1000.0
                filt.GNSS_update(pos_NED, vel_NED, t)
                gnss_count += 1

        out.append(filt.output_row(t))
        fc_att_list.append([float(row[_C["roll_fc"]]), float(row[_C["pitch_fc"]]), float(row[_C["yaw_fc"]])])

        if not silent and t - last_progress_t >= 30:
            progress = 100.0 * i / len(raw)
            print(f"[PROGRESS] {progress:.1f}% | t={t:.2f}s | rows={i}/{len(raw)} | GNSS_updates={gnss_count}")
            sigmas = np.sqrt(np.diag(filt.Sigma))
            print(f"  Covariance (std dev):  Att:{sigmas[0:3].mean():.2f}rad  Vel:{sigmas[3:6].mean():.2f}m/s  "
                  f"Pos:{sigmas[6:9].mean():.1f}m  GyBias:{sigmas[9:12].mean():.4f}rad/s  "
                  f"AcBias:{sigmas[12:15].mean():.3f}m/s²  VBias:{sigmas[15:18].mean():.2e}")
            last_progress_t = t

    out = np.asarray(out)
    header = (
        "t,px,py,pz,vx,vy,vz,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
        "bgx,bgy,bgz,bax,bay,baz,bmux,bmuy,bmuz,"
        "mag_roll,mag_pitch,mag_yaw"
    )

    np.savetxt(csv_out, out, delimiter=",", header=header, comments="")
    if not silent:
        print(f"Wrote {len(out)} rows to {csv_out}")
        print(f"Magnetometer updates applied: {filt.mag_update_count}")

    # Save diagnostic data (normalised ANIS per update, target ~1.0)
    if filt.update_times:
        diag_out = csv_out.replace("tg_eqf_output", "tg_eqf_diagnostics")
        with open(diag_out, 'w') as f:
            f.write("time,update_type,anis,anees\n")
            for update_type, ts, anis_val in sorted(filt.update_times, key=lambda x: x[1]):
                f.write(f"{ts:.4f},{update_type},{anis_val:.4f},\n")
        if not silent:
            print(f"Wrote diagnostic data to {diag_out}")

    # Compute angular RMSE between filter and FC attitude (DCM→quaternion, no gimbal lock)
    fc_att_arr = np.array(fc_att_list)
    valid_att = (np.all(np.isfinite(fc_att_arr), axis=1) &
                 np.all(np.isfinite(out[:, 7:16]), axis=1))
    rmse: dict[str, float] = {"angular": float('nan')}
    if valid_att.any():
        dcm_rows = out[valid_att, 7:16].reshape(-1, 3, 3)
        q_filt_xyzw = ScipyRot.from_matrix(dcm_rows).as_quat()
        q_filt = q_filt_xyzw[:, [3, 0, 1, 2]]  # → [w,x,y,z]
        roll_fc  = fc_att_arr[valid_att, 0]
        pitch_fc = fc_att_arr[valid_att, 1]
        yaw_fc   = fc_att_arr[valid_att, 2]
        q_fc_xyzw = ScipyRot.from_euler('ZYX', np.column_stack([yaw_fc, pitch_fc, roll_fc])).as_quat()
        q_fc = q_fc_xyzw[:, [3, 0, 1, 2]]  # → [w,x,y,z]
        dot = np.clip(np.abs(np.sum(q_filt * q_fc, axis=1)), 0.0, 1.0)
        ang_err_deg = np.degrees(2.0 * np.arccos(dot))
        rmse = {"angular": float(np.sqrt(np.mean(ang_err_deg**2)))}

    if _prop_count > 0:
        avg_us = _prop_total / _prop_count * 1e6
        print(f"Propagate: {_prop_count} steps, avg {avg_us:.1f} µs/step  (total {_prop_total*1e3:.1f} ms)")

    if not silent:
        print(f"\n=== Attitude RMSE vs FC ===")
        print(f"  Angular RMSE: {rmse['angular']:.2f} deg")

        print(f"\n=== Filter Diagnostics ===")
        if filt.update_times:
            from collections import defaultdict
            type_anis: dict[str, list[float]] = defaultdict(list)
            for mtype, _t, anis_val in filt.update_times:
                type_anis[mtype].append(anis_val)
            for mtype, vals in sorted(type_anis.items()):
                arr = np.array(vals)
                print(f"ANIS [{mtype}]: mean={np.mean(arr):.2f}  max={np.max(arr):.2f}  n={len(arr)}  (target ~1.0)")

    return rmse

if __name__ == "__main__":
    import sys
    run()