"""TG-EqF inertial/GNSS filter on SE₂(3) ⋉ se₂(3)."""

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
sys.path.insert(0, str(ref_path / "Utils"))
from matrix_math import *
from Symmetries.Calibrated.SE23_se23.Symmetry import SymGroup, State, InputSpace, stateAction

# =============================================================================
# Configuration
# =============================================================================

# "full"    -> data/20241011_NIMBUS24_Flight_FC_Data.csv
# "30s"     -> data/20241011_NIMBUS24_Flight_FC_Data_30s.csv
# "1s_loop" -> data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv
DATASET = "full"

GNSS_UPDATE_FREQ_HZ = 1     # Hz — how often to apply GNSS corrections
MAG_UPDATE_FREQ_HZ  = 0.5   # Hz — how often to apply magnetometer corrections

USE_GNSS_UPDATE = True
USE_MAG_UPDATE  = True
USE_MA_FILTER   = False  # moving-average pre-filter on IMU (window=5)
USE_EULER_DISCR = False  # Φ = I + A·dt instead of expm(A·dt)
USE_RESET       = True   # post-update covariance reset (arXiv:2309.03765)

# =============================================================================
# Physical constants
# =============================================================================

g = 9.81

# SE23 matrices used in lift and A computations
G = np.zeros((5, 5)); G[2, 3] = g   # gravity term in velocity slot
N = np.zeros((5, 5)); N[3, 4] = 1.0  # velocity→position transport

# Precomputed constants to avoid repeated construction inside the propagation loop
_SKEW_G  = SO3.wedge(np.array([[0.], [0.], [g]]))  # skew([0,0,g])
_G_VEC   = SE23.vee(G)                              # 9×1 vee of G
_A_UPPER = np.block([                               # upper-left 9×9 block of A (constant)
    [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
    [_SKEW_G,          np.zeros((3, 3)), np.zeros((3, 3))],
    [np.zeros((3, 3)), np.eye(3),        np.zeros((3, 3))],
])

# =============================================================================
# Noise parameters
# =============================================================================

P_0_blocks = [
    (2.0)**2  * np.eye(3),   # [0:3]   attitude (rad²)
    (0.5)**2  * np.eye(3),   # [3:6]   velocity (m/s)²
    (5.0)**2  * np.eye(3),   # [6:9]   position (m²)
    (0.02)**2 * np.eye(3),   # [9:12]  gyro bias (rad/s)²
    (0.02)**2 * np.eye(3),   # [12:15] accel bias (m/s²)²
    (0.5)**2  * np.eye(3),   # [15:18] virtual bias
]

Q_gyro_var         = 1e-1
Q_accel_var        = 1e-0
Q_virt_var         = 1.0
Q_gyro_bias_var    = (1e-3)**2
Q_accel_bias_var   = (1e-2)**2
Q_virtual_bias_var = (1e-7)**2

R_gnss_pos_var = 1.0   # m²
R_gnss_vel_var = 10.0  # (m/s)²
R_mag_var      = 50.0  # rad²

# =============================================================================
# Magnetometer configuration
# =============================================================================

MAG_AXIS_ORDER = np.array([0, 2, 1])
MAG_AXIS_SIGNS = np.array([-1.0, 1.0, 1.0])

WMM_DECLINATION = -13.4  # degrees West
WMM_INCLINATION =  56.8  # degrees Down
WMM_MAGNITUDE   = 47000.0  # nT

_dec = np.radians(WMM_DECLINATION)
_inc = np.radians(WMM_INCLINATION)
_wmm_h = WMM_MAGNITUDE * np.cos(_inc)
MAG_FIELD_NED = np.array([_wmm_h * np.cos(_dec), _wmm_h * np.sin(_dec),
                           WMM_MAGNITUDE * np.sin(_inc)])
MAG_FIELD_NED /= np.linalg.norm(MAG_FIELD_NED)

# =============================================================================
# Helpers
# =============================================================================

def col(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1, 1)

def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def blockdiag(*arrs: np.ndarray) -> np.ndarray:
    n = sum(a.shape[0] for a in arrs)
    m = sum(a.shape[1] for a in arrs)
    out = np.zeros((n, m))
    r = c = 0
    for a in arrs:
        rr, cc = a.shape
        out[r:r+rr, c:c+cc] = a
        r += rr; c += cc
    return out

def from_two_vectors_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray | None:
    """Minimum-angle rotation R such that R @ v_to ≈ v_from."""
    f = v_from / (np.linalg.norm(v_from) + 1e-12)
    t = v_to   / (np.linalg.norm(v_to)   + 1e-12)
    cosine = float(np.clip(f @ t, -1.0, 1.0))
    if cosine >  1.0 - 1e-12: return np.eye(3)
    if cosine < -1.0 + 1e-12:
        perp = np.array([1., 0., 0.]) if abs(f[0]) < 0.9 else np.array([0., 1., 0.])
        axis = np.cross(f, perp); axis /= np.linalg.norm(axis)
        return ScipyRot.from_rotvec(np.pi * axis).as_matrix()
    cross = np.cross(f, t)
    K = SO3.wedge(cross.reshape(3, 1))
    return (np.eye(3) + K + K @ K / (1.0 + cosine)).T

# =============================================================================
# Filter class
# =============================================================================

class TGEqF:
    """TG-EqF: Trajectory-Generating Equivariant Filter on SE₂(3) ⋉ se₂(3)."""

    def __init__(self):
        self.X_hat = SymGroup.identity()
        self.xi_0  = State()

        self.t_prev     = None
        self.t_last_gnss = None
        self.t_last_imu  = None

        self.attitude_initialized = False
        self._init_accel_accum = np.zeros(3)
        self._init_accel_n     = 0
        self._init_mag_accum   = np.zeros(3)
        self._init_mag_n       = 0
        self._last_accel: np.ndarray | None = None
        self.hard_iron_bias = np.zeros(3)
        self._heading_corrected = False

        self.ma_window   = 10
        self.gyro_buffer: list = []
        self.accel_buffer: list = []

        self.mag_update_count = 0
        self.mag_euler: np.ndarray = np.full(3, float('nan'))
        self.anis_values: list[float] = []
        self.update_times: list[tuple[str, float, float]] = []

        self.Sigma  = blockdiag(*P_0_blocks)
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

        # Cached numpy arrays extracted from X_hat — updated once per propagation step.
        # Using these avoids calling stateAction() (which constructs many pylie objects)
        # in the hot path.
        self._R        = np.eye(3)        # body→NED rotation  (3×3)
        self._v        = np.zeros((3, 1)) # velocity NED        (3×1)
        self._p        = np.zeros((3, 1)) # position NED        (3×1)
        self._b        = np.zeros((9, 1)) # bias vector         (9×1), virtual=0
        self._B_mat    = np.eye(5)        # SE23 matrix of B    (5×5)
        self._B_mat_inv = np.eye(5)       # analytical inverse  (5×5)

    # -------------------------------------------------------------------------
    # State cache
    # -------------------------------------------------------------------------

    def _cache_state(self) -> None:
        """Extract key arrays from X_hat.B analytically — avoids stateAction overhead."""
        B = self.X_hat.B.as_matrix()   # 5×5 SE23 element
        R = B[:3, :3]
        v = B[:3, 3:4]
        p = B[:3, 4:5]

        # SE23 inverse: B^{-1} = | R^T  -R^T v  -R^T p |
        RT = R.T
        Bi = np.eye(5)
        Bi[:3, :3] = RT
        Bi[:3, 3:4] = -RT @ v
        Bi[:3, 4:5] = -RT @ p

        # Bias: b = -vee(B^{-1} · beta · B), virtual slot frozen to 0
        b = -SE23.vee(Bi @ self.X_hat.beta @ B)
        b[6:9] = 0.0

        self._R = R; self._v = v; self._p = p; self._b = b
        self._B_mat = B; self._B_mat_inv = Bi

    def xi_hat(self) -> State:
        """Full State object from stateAction — kept for debugging/compatibility."""
        xi = stateAction(self.X_hat, self.xi_0)
        xi.b[6:9] = 0.0
        return xi

    # -------------------------------------------------------------------------
    # Attitude initialisation (TRIAD)
    # -------------------------------------------------------------------------

    def initialize_attitude_triad(self, mag_body: np.ndarray) -> None:
        MIN_SAMPLES = 5
        if self.attitude_initialized or self._init_accel_n < MIN_SAMPLES:
            return

        accel_avg = self._init_accel_accum / self._init_accel_n
        g_body = -accel_avg
        g_norm = np.linalg.norm(g_body)
        if g_norm < 1.0:
            return
        g_body /= g_norm

        m_body = mag_body / (np.linalg.norm(mag_body) + 1e-10)
        g_ned  = np.array([0., 0., 1.])
        m_ned  = MAG_FIELD_NED.flatten()

        cross_b = np.cross(g_body, m_body); cb = np.linalg.norm(cross_b)
        cross_n = np.cross(g_ned,  m_ned);  cn = np.linalg.norm(cross_n)
        if cb < 1e-6 or cn < 1e-6:
            return

        t2_b = cross_b / cb; t2_n = cross_n / cn
        T_body = np.column_stack([g_body, t2_b, np.cross(g_body, t2_b)])
        T_ned  = np.column_stack([g_ned,  t2_n, np.cross(g_ned,  t2_n)])
        R = T_ned @ T_body.T  # body→NED

        if self._init_mag_n > 0:
            m_avg = self._init_mag_accum / self._init_mag_n
            m_avg_n = np.linalg.norm(m_avg)
            if m_avg_n > 1e-6:
                self.hard_iron_bias = m_avg - R.T @ MAG_FIELD_NED * m_avg_n

        self.X_hat = SymGroup(SE23(R), np.zeros((5, 5)))  # type: ignore[arg-type]
        self.attitude_initialized = True
        self._cache_state()

        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        b = self.hard_iron_bias
        print(f"TRIAD init ({self._init_accel_n} accel samples): "
              f"roll={roll:.1f}°  pitch={pitch:.1f}°  yaw={yaw:.1f}°")
        print(f"Hard-iron bias: [{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}]")

    # -------------------------------------------------------------------------
    # Propagation
    # -------------------------------------------------------------------------

    def propagate(self, t: float, gyro: np.ndarray, accel: np.ndarray) -> None:
        gyro  = col(gyro)
        accel = col(accel)

        if self.t_prev is None:
            self.t_prev = t; return

        if not self.attitude_initialized:
            self._init_accel_accum += accel.flatten()
            self._init_accel_n += 1
            self.t_prev = t; return

        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t; return

        if USE_MA_FILTER:
            gyro  = self._moving_average(gyro,  self.gyro_buffer)
            accel = self._moving_average(accel, self.accel_buffer)

        self._last_accel = accel.flatten()

        U = InputSpace()
        U.w   = SO3.wedge(gyro)
        U.a   = accel
        U.mu  = np.zeros((3, 1))  # virtual bias is frozen at 0
        U.tau = np.zeros((9, 1))

        lift = self._lift(U)
        A    = self._A(U)
        Phi  = np.eye(18) + A * dt if USE_EULER_DISCR else expm(A * dt)

        self.X_hat = self.X_hat * SymGroup.exp(lift * dt)  # type: ignore[attr-defined]
        self.Sigma = sym(Phi @ self.Sigma @ Phi.T + self.Q * dt)

        if np.max(np.diag(self.Sigma)) > 1e8:
            eigs, V = np.linalg.eigh(self.Sigma)
            self.Sigma = sym(V @ np.diag(np.clip(eigs, 0, 1e8)) @ V.T)

        self._cache_state()
        self.t_prev    = t
        self.t_last_imu = t

    # -------------------------------------------------------------------------
    # Update steps
    # -------------------------------------------------------------------------

    def GNSS_update(self, pos_NED: np.ndarray, vel_NED: np.ndarray,
                    t: float | None = None) -> None:
        # Innovation: difference between GNSS measurement and filter prediction
        pos_err = (pos_NED - self._p.flatten()).reshape(-1, 1)
        vel_err = (vel_NED - self._v.flatten()).reshape(-1, 1)
        delta   = np.vstack([pos_err, vel_err])

        C = self._C_gnss()
        S = C @ self.Sigma @ C.T + self.R_gnss
        K = self.Sigma @ C.T @ np.linalg.inv(S)
        K_delta = K @ delta

        anis = self._anis(delta, S)
        if t is not None and anis is not None:
            self.update_times.append(('gnss', t, anis / delta.size))

        self.X_hat = SymGroup.exp(K_delta) * self.X_hat  # type: ignore[attr-defined]
        IKC = np.eye(18) - K @ C
        self.Sigma = sym(IKC @ self.Sigma @ IKC.T + K @ self.R_gnss @ K.T)

        if USE_RESET:
            self._reset(K_delta)

        self._cache_state()

        if t is not None:
            self.t_last_gnss = t

    def magnetometer_update(self, mag: np.ndarray, t: float | None = None) -> None:
        mag = np.asarray(mag, dtype=float).reshape(3) - self.hard_iron_bias
        mag_n = np.linalg.norm(mag)
        if mag_n < 1e-6 or self._last_accel is None:
            return

        # Quasi-static gate: skip during powered flight / free-fall
        if abs(np.linalg.norm(self._last_accel) - g) > 3.0:
            return

        # TRIAD: build body→NED rotation from accel + mag
        accel_n = np.linalg.norm(self._last_accel)
        g_body  = -self._last_accel / accel_n
        m_body  =  mag / mag_n
        g_ned   = np.array([0., 0., 1.])
        m_ned   = MAG_FIELD_NED.flatten()

        cb = np.linalg.norm(np.cross(g_body, m_body))
        cn = np.linalg.norm(np.cross(g_ned,  m_ned))
        if cb < 1e-6 or cn < 1e-6:
            return

        t2_b = np.cross(g_body, m_body) / cb
        t2_n = np.cross(g_ned,  m_ned)  / cn
        T_body = np.column_stack([g_body, t2_b, np.cross(g_body, t2_b)])
        T_ned  = np.column_stack([g_ned,  t2_n, np.cross(g_ned,  t2_n)])
        R_triad = T_ned @ T_body.T

        # Project onto SO(3)
        U_svd, _, Vt = np.linalg.svd(R_triad)
        if np.linalg.det(U_svd @ Vt) < 0:
            U_svd[:, -1] *= -1
        R_triad = U_svd @ Vt

        # SO3 log innovation
        R_err = R_triad @ self._R.T
        U_svd, _, Vt = np.linalg.svd(R_err)
        if np.linalg.det(U_svd @ Vt) < 0:
            U_svd[:, -1] *= -1
        delta = ScipyRot.from_matrix(U_svd @ Vt).as_rotvec().reshape(3, 1)

        C = np.zeros((3, 18))
        C[0:3, 0:3] = -np.eye(3)

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

        self._cache_state()

        euler = ScipyRot.from_matrix(self._R).as_euler('ZYX')
        self.mag_euler = np.array([euler[2], euler[1], euler[0]])
        self.mag_update_count += 1

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def output_row(self, t: float) -> list[float]:
        R, v, p, b = self._R, self._v, self._p, self._b
        return [
            t,
            p[0, 0], p[1, 0], p[2, 0],
            v[0, 0], v[1, 0], v[2, 0],
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2],
            *(float(b[i, 0]) for i in range(9)),
            *self.mag_euler.tolist(),
        ]

    # -------------------------------------------------------------------------
    # Internal: lift, A, C matrices
    # -------------------------------------------------------------------------

    def _lift(self, U: InputSpace) -> np.ndarray:
        """Compute the symmetry lift Λ from the cached state and IMU input U."""
        W      = U.as_W_mat()
        B_bias = SE23.wedge(self._b)
        lam1   = SE23.vee(W - B_bias + N + self._B_mat_inv @ (G - N) @ self._B_mat)
        lam    = np.zeros((18, 1))
        lam[0:9]  = lam1
        lam[9:18] = SE23.adjoint(self._b) @ lam1
        return lam

    def _A(self, U: InputSpace) -> np.ndarray:
        """Linearised dynamics matrix A using cached B_mat / B_mat_inv."""
        # Velocity action: transform U to the origin without creating X_hat.inv()
        X_inv_beta = -self._B_mat_inv @ self.X_hat.beta @ self._B_mat
        f10 = np.zeros((5, 5))
        f10[0:3, 4:5] = self._B_mat_inv[0:3, 3:4]
        u0_vec = SE23.vee(
            self._B_mat_inv @ (U.as_W_mat() - X_inv_beta) @ self._B_mat + f10
        )
        At = np.zeros((18, 18))
        At[0:9,  0:9]  = _A_UPPER
        At[9:18, 9:18] = SE23.adjoint(u0_vec + _G_VEC)
        At[0:9,  9:18] = np.eye(9)
        return At

    def _C_gnss(self) -> np.ndarray:
        """GNSS output Jacobian using cached position and velocity."""
        C = np.zeros((6, 18))
        C[0:3, 0:3] = -SO3.wedge(self._p)
        C[0:3, 6:9] = np.eye(3)
        C[3:6, 0:3] = -SO3.wedge(self._v)
        C[3:6, 3:6] = np.eye(3)
        return C

    # -------------------------------------------------------------------------
    # Internal: covariance reset, ANIS, moving average
    # -------------------------------------------------------------------------

    def _reset(self, K_delta: np.ndarray) -> None:
        """Left-Jacobian covariance reset: J_l = expm(½ · grp_adj(K·δ))."""
        ad = np.zeros((18, 18))
        d0 = K_delta[0:9]; d1 = K_delta[9:18]
        ad[0:9,  0:9]  = SE23.adjoint(d0)
        ad[9:18, 0:9]  = SE23.adjoint(d1)
        ad[9:18, 9:18] = SE23.adjoint(d0)
        J = expm(0.5 * ad)
        self.Sigma = sym(J @ self.Sigma @ J.T)

    def _anis(self, innovation: np.ndarray, S: np.ndarray) -> float | None:
        try:
            val = float(np.squeeze(innovation.T @ np.linalg.inv(S) @ innovation))
            self.anis_values.append(val)
            return val
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _moving_average(self, x: np.ndarray, buf: list) -> np.ndarray:
        buf.append(x.flatten())
        if len(buf) > self.ma_window:
            buf.pop(0)
        return np.mean(buf, axis=0).reshape(-1, 1)

# =============================================================================
# Data I/O
# =============================================================================

_C = {
    "t": 0, "lon": 1, "lat": 2, "alt": 3,
    "gps_vn": 4, "gps_ve": 5, "gps_vd": 6,
    "ax": 9, "ay": 10, "az": 11,
    "gx": 15, "gy": 16, "gz": 17,
    "mx": 18, "my": 19, "mz": 20,
    "roll_fc": 29, "pitch_fc": 30, "yaw_fc": 31,
}
R_EARTH = 6_378_137.0


def _gps_to_ned(lat, lon, alt, lat0, lon0, alt0) -> np.ndarray:
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    return np.array([
        dlat * R_EARTH,
        dlon * np.cos(np.radians(lat0)) * R_EARTH,
        -(alt - alt0),
    ])


def run(csv_in: str | None = None,
        csv_out: str = "outputs/tg_eqf_output.csv",
        mag_axis_order: np.ndarray | None = None,
        mag_axis_signs: np.ndarray | None = None,
        r_mag_var: float | None = None,
        gnss_freq_hz: float | None = None,
        mag_freq_hz: float | None = None,
        use_mag_update: bool | None = None,
        silent: bool = False) -> dict[str, float]:
    """Run the TG-EqF on NIMBUS24 FC data. Returns {'angular': RMSE_deg}."""

    axis_order = MAG_AXIS_ORDER if mag_axis_order is None else mag_axis_order
    axis_signs = MAG_AXIS_SIGNS if mag_axis_signs is None else mag_axis_signs
    _gnss_freq = GNSS_UPDATE_FREQ_HZ if gnss_freq_hz   is None else gnss_freq_hz
    _mag_freq  = MAG_UPDATE_FREQ_HZ  if mag_freq_hz    is None else mag_freq_hz
    _use_mag   = USE_MAG_UPDATE      if use_mag_update is None else use_mag_update

    if csv_in is None:
        _tag = ("_ma" if USE_MA_FILTER else "") + ("_euler" if USE_EULER_DISCR else "")
        _datasets = {
            "full":    ("data/20241011_NIMBUS24_Flight_FC_Data.csv",
                        f"outputs/tg_eqf_output_full{_tag}.csv"),
            "30s":     ("data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",
                        f"outputs/tg_eqf_output_30s{_tag}.csv"),
            "1s_loop": ("data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",
                        f"outputs/tg_eqf_output_1s_loop{_tag}.csv"),
        }
        if DATASET not in _datasets:
            raise ValueError(f"Unknown DATASET {DATASET!r}")
        csv_in, csv_out = _datasets[DATASET]

    raw = np.genfromtxt(csv_in, delimiter=",", skip_header=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if not silent:
        print(f"Loaded {len(raw)} rows")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    # Reference GPS origin
    valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
    first = np.argmax(valid)
    lat0 = raw[first, _C["lat"]]
    lon0 = raw[first, _C["lon"]]
    alt0 = raw[first, _C["alt"]] / 1000.0

    filt = TGEqF()
    if r_mag_var is not None:
        filt.R_mag = np.eye(3) * r_mag_var

    # Pre-initialisation: accumulate accel + mag for TRIAD
    for row in raw:
        if not np.isfinite(row[_C["t"]]):
            continue
        accel = row[[_C["ax"], _C["ay"], _C["az"]]] * g
        mag_raw = row[[_C["mx"], _C["my"], _C["mz"]]]
        if not (np.all(np.isfinite(accel)) and np.all(np.isfinite(mag_raw))):
            continue
        filt._init_accel_accum += accel
        filt._init_accel_n += 1
        mag = mag_raw[axis_order] * axis_signs
        mag_n = np.linalg.norm(mag)
        if mag_n > 1e-6:
            filt._init_mag_accum += mag
            filt._init_mag_n += 1
            if filt._init_accel_n >= 5:
                filt.initialize_attitude_triad(mag / mag_n)
                break

    out: list = []
    fc_att_list: list = []
    prev_mag  = None
    t_last_mag = None
    gnss_count = 0
    last_progress_t = 0.0
    gyro_scale = np.pi / 180.0

    prop_t0 = time.perf_counter()

    for i, row in enumerate(raw):
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        gyro  = row[[_C["gx"], _C["gy"], _C["gz"]]] * gyro_scale
        accel = row[[_C["ax"], _C["ay"], _C["az"]]] * g
        if not np.all(np.isfinite(np.concatenate([gyro, accel]))):
            continue

        filt.propagate(t, gyro, accel)

        # Magnetometer update
        mag_raw  = row[[_C["mx"], _C["my"], _C["mz"]]]
        mag      = mag_raw[axis_order] * axis_signs
        mag_norm = np.linalg.norm(mag)

        if not filt.attitude_initialized and mag_norm > 1e-6:
            filt._init_mag_accum += mag
            filt._init_mag_n += 1
            filt.initialize_attitude_triad(mag / mag_norm)

        mag_rate_ok = t_last_mag is None or (t - t_last_mag) >= 1.0 / _mag_freq
        if (_use_mag and filt.attitude_initialized and mag_rate_ok
                and (prev_mag is None or not np.allclose(prev_mag, mag))):
            filt.magnetometer_update(mag, t=t)
            prev_mag   = mag
            t_last_mag = t

        # GNSS update
        lat, lon = row[_C["lat"]], row[_C["lon"]]
        if USE_GNSS_UPDATE and lat != 0 and lon != 0:
            gnss_period = 1.0 / _gnss_freq
            if filt.t_last_gnss is None or (t - filt.t_last_gnss) >= gnss_period:
                alt    = row[_C["alt"]] / 1000.0
                pos_NED = _gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
                vel_NED = np.array([row[_C["gps_vn"]], row[_C["gps_ve"]],
                                    row[_C["gps_vd"]]]) / 1000.0
                filt.GNSS_update(pos_NED, vel_NED, t)
                gnss_count += 1

        out.append(filt.output_row(t))
        fc_att_list.append([float(row[_C["roll_fc"]]),
                             float(row[_C["pitch_fc"]]),
                             float(row[_C["yaw_fc"]])])

        if not silent and t - last_progress_t >= 30:
            sig = np.sqrt(np.diag(filt.Sigma))
            print(f"[PROGRESS] {100*i/len(raw):.1f}% | t={t:.2f}s | GNSS={gnss_count}")
            print(f"  std: Att={sig[0:3].mean():.2f}rad  Vel={sig[3:6].mean():.2f}m/s  "
                  f"Pos={sig[6:9].mean():.1f}m  Gbias={sig[9:12].mean():.4f}rad/s  "
                  f"Abias={sig[12:15].mean():.3f}m/s2")
            last_progress_t = t

    prop_total = time.perf_counter() - prop_t0

    out_arr = np.asarray(out)
    header = ("t,px,py,pz,vx,vy,vz,"
              "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
              "bgx,bgy,bgz,bax,bay,baz,bmux,bmuy,bmuz,"
              "mag_roll,mag_pitch,mag_yaw")
    np.savetxt(csv_out, out_arr, delimiter=",", header=header, comments="")
    if not silent:
        print(f"Wrote {len(out_arr)} rows to {csv_out}")
        print(f"Magnetometer updates: {filt.mag_update_count}")

    if filt.update_times:
        diag_out = csv_out.replace("tg_eqf_output", "tg_eqf_diagnostics")
        with open(diag_out, 'w') as f:
            f.write("time,update_type,anis,anees\n")
            for utype, ts, aval in sorted(filt.update_times, key=lambda x: x[1]):
                f.write(f"{ts:.4f},{utype},{aval:.4f},\n")
        if not silent:
            print(f"Wrote diagnostics to {diag_out}")

    # Angular RMSE vs FC
    fc_att = np.array(fc_att_list)
    valid_att = np.all(np.isfinite(fc_att), axis=1) & np.all(np.isfinite(out_arr[:, 7:16]), axis=1)
    rmse: dict[str, float] = {"angular": float('nan')}
    if valid_att.any():
        dcm   = out_arr[valid_att, 7:16].reshape(-1, 3, 3)
        q_f   = ScipyRot.from_matrix(dcm).as_quat()[:, [3, 0, 1, 2]]
        rpy   = fc_att[valid_att]
        q_fc  = ScipyRot.from_euler('ZYX', rpy[:, [2, 1, 0]]).as_quat()[:, [3, 0, 1, 2]]
        dot   = np.clip(np.abs((q_f * q_fc).sum(axis=1)), 0.0, 1.0)
        rmse  = {"angular": float(np.sqrt(np.mean(np.degrees(2 * np.arccos(dot))**2)))}

    n_steps = len(out)
    avg_us = prop_total / n_steps * 1e6 if n_steps else 0
    print(f"Propagate: {n_steps} steps, avg {avg_us:.1f} µs/step  "
          f"(total {prop_total*1e3:.1f} ms)")

    if not silent:
        print(f"\n=== Attitude RMSE vs FC ===")
        print(f"  Angular RMSE: {rmse['angular']:.2f} deg")
        if filt.update_times:
            from collections import defaultdict
            by_type: dict[str, list[float]] = defaultdict(list)
            for mtype, _, v in filt.update_times:
                by_type[mtype].append(v)
            print(f"\n=== Filter Diagnostics ===")
            for mtype, vals in sorted(by_type.items()):
                arr = np.array(vals)
                print(f"ANIS [{mtype}]: mean={arr.mean():.2f}  "
                      f"max={arr.max():.2f}  n={len(arr)}  (target ~1.0)")

    return rmse


if __name__ == "__main__":
    run()
