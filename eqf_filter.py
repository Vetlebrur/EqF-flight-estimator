"""Standalone CNS TG-EqF inspired inertial/GNSS filter using pylie."""

import os
import sys
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
from Symmetries.Calibrated.SE23_se23.Symmetry import SymGroup, State, InputSpace, stateAction, velocityAction, f_10, stateActionDiff, grp_adj

# =============================================================================
# Configuration
# =============================================================================

# "full"    -> data/20241011_NIMBUS24_Flight_FC_Data.csv          (complete flight)
# "30s"     -> data/20241011_NIMBUS24_Flight_FC_Data_30s.csv      (first 30 s)
# "1s_loop" -> data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv  (first 1 s looped for 30 s)
DATASET = "full"

GNSS_UPDATE_FREQ_HZ = 0.3   # GNSS update frequency (Hz) — update every 1/f seconds
# =============================================================================
# Constants and Physical Parameters
# =============================================================================

g = 9.81  # Gravitational acceleration (m/s²)

# =============================================================================
# Noise Parameters
# =============================================================================
# --- State Covariance (P) ---
# Realistic initial uncertainties for rocket inertial/GNSS fusion
P_0_blocks = [
    (1)**2 * np.eye(3),       # [0:3] attitude error (rad²) 
    (10.0)**2 * np.eye(3),       # [3:6] velocity error (m/s)²
    (1.0)**2 * np.eye(3),      # [6:9] position error (m²)
    (0.1)**2 * np.eye(3),     # [9:12] gyro bias error (rad/s)²
    (0.1)**2 * np.eye(3),      # [12:15] accel bias error (m/s²)² - tight initial
    1e-9 * np.eye(3)           # [15:18] virtual accel bias (frozen at zero)
]

# --- Process Noise (Q) ---
# Conservative random walk for biases (prevent explosion)
Q_rot_var = 1e-2               # [0:3] rotation process noise
Q_vel_var = 1e-1                # [3:6] velocity process noise
Q_pos_var = 1e-2                # [6:9] position process noise
Q_gyro_bias_var = (1e-2)**2     # [9:12] gyro bias random walk (very tight)
Q_accel_bias_var = (1e-2)**2    # [12:15] accel bias random walk (very tight)
Q_virtual_bias_var = 1e-7      # [15:18] virtual accel bias (frozen)

# --- Measurement Noise (R) ---
# GNSS measurement noise
R_gnss_pos_var = 1            # GNSS position measurement variance (m²)
R_gnss_vel_var = 5.0           # GNSS velocity measurement variance (m/s)²

# Magnetometer noise: innovation is SO3.log(R_err) in radians.
# R_mag=400.0 rad² tuned via 48-config sweep with correct body frame (accel=[ax,ay,az]).
R_mag_var = 100.0  # rad²

# TRIAD-derived full attitude update noise (rad²)
# Keep large — TRIAD from a vibrating rocket pad is noisy; let gyro-integration dominate
R_triad_var = (0.3)**2  # ~17° per axis

# --- Magnetometer Configuration ---
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


# Magnetometer gating thresholds
MAG_FIELD_MAGNITUDE_NOM = 47000.0  # nanoTesla
MAG_FIELD_DEVIATION_THRESHOLD = 0.15  # Reject if >15% deviation
MAG_ACCEL_GATE_THRESHOLD = 15.0  # Disable during boost (accel > 15 m/s²)


# =============================================================================
# Helpers
# =============================================================================


def from_two_vectors_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix mapping unit vector a to unit vector b.

    Mirrors C++ EqFparser.cpp fromTwoVectorsRotation used to build magData for MagUpdate.
    """
    a = a.flatten() / (np.linalg.norm(a) + 1e-12)
    b = b.flatten() / (np.linalg.norm(b) + 1e-12)
    cross = np.cross(a, b)
    cross_n = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cross_n < 1e-8:
        if dot > 0:
            return np.eye(3)
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        ax = np.cross(a, perp)
        ax /= np.linalg.norm(ax)
        return -np.eye(3) + 2 * np.outer(ax, ax)
    axis = cross / cross_n
    angle = np.arctan2(cross_n, dot)
    K = SO3.wedge(axis.reshape(-1, 1))  # (3,3) skew-symmetric
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def triad_attitude(accel_body: np.ndarray, mag_body: np.ndarray) -> "np.ndarray | None":
    """Compute body→NED rotation matrix via TRIAD from accelerometer + magnetometer.

    Only valid when accel ≈ gravity (quasi-static). Returns None for degenerate geometry.
    """
    g_body = -accel_body
    g_body_n = np.linalg.norm(g_body)
    m_body_n = np.linalg.norm(mag_body)
    if g_body_n < 1.0 or m_body_n < 0.01:
        return None
    g_b = g_body / g_body_n
    m_b = mag_body / m_body_n

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
# Gravity Matrix
# =============================================================================

G = np.zeros((5, 5))
G[2, 3] = g


# =============================================================================
# Symmetry Lift
# =============================================================================

def calculate_lift(xi: State, U: InputSpace) -> np.ndarray:
    """Compute continuous lift (tangent-space dynamics)."""
    L = np.zeros((18, 1))

    # Decompose the lift computation
    U_minus_b = U.as_W_vec() - xi.b
    T_inv_mat = xi.T.inv().as_matrix()
    f10_val = f_10(xi.T.as_matrix())

    # Create gravity term with attitude-dependent compensation
    gravity_term = G + f10_val
    # Modify gravity to account for the rotation-dependent mu feedback
    # by reducing its effect when mu is being fed back
    T_inv_grav = T_inv_mat @ gravity_term
    grav_vee = SE23.vee(T_inv_grav)

    L[0:9, 0:1] = U_minus_b + grav_vee
    L[9:18, 0:1] = SE23.adjoint(xi.b) @ L[0:9, 0:1] - U.tau

    return L

# =============================================================================
# Filter
# =============================================================================


class TGEqF:
    """TG-EqF Inertial/GNSS Filter."""

    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix using intrinsic Z-Y-X rotations.

        This matches the extraction formula used in output_row():
          roll = arctan2(R[2,1], R[2,2])
          pitch = arcsin(-R[2,0])
          yaw = arctan2(R[1,0], R[0,0])

        Args:
            roll, pitch, yaw: Euler angles in radians (intrinsic body-frame rotations)

        Returns:
            3x3 rotation matrix (body to world)
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Intrinsic Z-Y-X rotations: R = Rx(roll) * Ry(pitch) * Rz(yaw)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        return R

    def initialize_attitude_triad(self, mag_body: np.ndarray) -> None:
        """Initialize attitude via TRIAD using averaged accel (gravity) + magnetometer.

        TRIAD constructs an orthonormal rotation matrix from two vector observations
        (gravity and magnetic field) in both the body frame and the NED reference frame.
        No Euler angles need to be specified — works for any initial orientation.
        """
        MIN_INIT_SAMPLES = 5
        if self.attitude_initialized or self._init_accel_n < MIN_INIT_SAMPLES:
            return

        accel_avg = self._init_accel_accum / self._init_accel_n

        # Gravity direction in body = opposite of specific force (accel reads -gravity)
        g_body = -accel_avg
        g_norm = np.linalg.norm(g_body)
        if g_norm < 1.0:  # sanity check: at least 1 m/s² average
            return
        g_body = g_body / g_norm

        m_body = mag_body / (np.linalg.norm(mag_body) + 1e-10)

        # Reference vectors in NED
        g_ned = np.array([0.0, 0.0, 1.0])   # gravity points down (+NED-Z)
        m_ned = MAG_FIELD_NED.flatten()

        # TRIAD: build orthonormal triad in body frame and NED frame
        # First basis vector: gravity
        # Second: perpendicular to both gravity and mag (cross product)
        # Third: completes the right-handed frame
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
        """Initialize filter state and covariance matrices."""
        self.X_hat = SymGroup.identity()
        self.xi_0 = State()
        self.t_prev = None
        self.t_last_gnss = None
        self.t_last_imu = None
        self.innovationLift = np.linalg.pinv(stateActionDiff(self.xi_0))
        self.attitude_initialized = False

        # Accel accumulator for TRIAD initialization (averaged over static samples)
        self._init_accel_accum = np.zeros(3)
        self._init_accel_n = 0

        # Moving average filter for smoothing noisy sensor measurements
        self.ma_window = 5  # Reduced to minimize lag (5-10 samples)
        self.gyro_buffer = []
        self.accel_buffer = []
        self.mag_buffer = []

        # Magnetometer values
        self.mag_meas_0 = MAG_FIELD_NED.reshape(-1, 1)  # WMM field in NED
        self.magnetometer_initialized = True  # Pre-initialized with WMM
        self.accel_norm_prev = 0.0
        self.mag_ref_initialized: bool = False

        self.prev_altitude = 0.0
        self.mag_update_count = 0  # Track magnetometer update calls
        self.update_count = 0  # Counter for periodic re-orthonormalization

        # Magnetometer attitude snapshot (quaternion [w,x,y,z] at last mag update, nan until first)
        self.mag_q: np.ndarray = np.full(4, float('nan'))

        # Filter diagnostics: ANIS and ANEES
        self.anis_values: list[float] = []
        self.anees_values: list[float] = []
        self.update_times: list[tuple[str, float, float]] = []

        # =====================================================================
        # Assemble noise covariance matrices from parameters
        # =====================================================================
        # State covariance (P) 
        self.Sigma = blockdiag(*P_0_blocks)

        # Measurement noise covariance (R)
        self.R_gnss = blockdiag(
            np.eye(3) * R_gnss_pos_var,    # Position variance [0:3, 0:3]
            np.eye(3) * R_gnss_vel_var     # Velocity variance [3:6, 3:6]
        )
        self.R_gnss_pos_only = np.eye(3) * R_gnss_pos_var
        self.R_mag = blockdiag(
            np.eye(3) * R_mag_var
        )
        self.R_triad = np.eye(3) * R_triad_var
        self.triad_update_count = 0
        self.t_last_triad: float | None = None

        # Process noise covariance (Q)
        # 18x18 block diagonal with different noise levels for each state component
        self.Q = blockdiag(
            np.eye(3) * Q_rot_var,           # [0:3, 0:3] rotation process noise
            np.eye(3) * Q_vel_var,           # [3:6, 3:6] velocity process noise
            np.eye(3) * Q_pos_var,           # [6:9, 6:9] position process noise
            np.eye(3) * Q_gyro_bias_var,     # [9:12, 9:12] gyro bias random walk
            np.eye(3) * Q_accel_bias_var,    # [12:15, 12:15] accel bias random walk
            np.eye(3) * Q_virtual_bias_var   # [15:18, 15:18] virtual accel bias
        )

    def xi_hat(self) -> State:
        xi = stateAction(self.X_hat, self.xi_0)
        # Freeze b_mu at zero (unobservable, prevents affecting dynamics)
        xi.b[6:9] = 0
        return xi

    def _apply_moving_average(self, measurement: np.ndarray, buffer: list[np.ndarray]) -> np.ndarray:
        """Apply moving average filter with window size of 10."""
        buffer.append(measurement.flatten())
        if len(buffer) > self.ma_window:
            buffer.pop(0)
        return np.mean(buffer, axis=0).reshape(-1, 1)

    @staticmethod
    def _unwrap_angle(angle_raw: float, angle_prev: float) -> float:
        """Unwrap angle to avoid discontinuous jumps at ±π.

        Maps angle_raw to be within π of angle_prev, creating continuous output.
        """
        delta = angle_raw - angle_prev
        if delta > np.pi:
            return angle_raw - 2 * np.pi
        elif delta < -np.pi:
            return angle_raw + 2 * np.pi
        else:
            return angle_raw

    # =========================================================================
    # Propagation
    # =========================================================================

    def propagate(self, t: float, gyro: np.ndarray, accel: np.ndarray) -> None:
        """Propagate state using gyro and accel measurements."""
        gyro = col(gyro)
        accel = col(accel)

        if self.t_prev is None:
            self.t_prev = t
            return

        # Accumulate accel samples for TRIAD initialization; skip propagation until ready
        if not self.attitude_initialized:
            self._init_accel_accum += accel.flatten()
            self._init_accel_n += 1
            self.t_prev = t
            return

        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return

        # Skip moving average smoothing to test if it causes divergence
        #gyro = self._apply_moving_average(gyro, self.gyro_buffer)
        #accel = self._apply_moving_average(accel, self.accel_buffer)

        U = InputSpace()
        U.w = SO3.wedge(gyro)
        U.a = accel

        xi_hat = self.xi_hat()
        U.mu = xi_hat.b[6:9, 0:1]
        U.tau = np.zeros((9, 1))

        lift = calculate_lift(self.xi_hat(), U)

        self.X_hat = self.X_hat * SymGroup.exp(lift * dt)  # type: ignore[attr-defined]

        A = self.calculate_A(U)
        Phi = expm(A * dt)
        self.Sigma = Phi @ self.Sigma @ Phi.T + self.Q * dt
        self.Sigma = sym(self.Sigma)
        

        # Prevent covariance explosion by bounding maximum eigenvalue
        # If any eigenvalue exceeds threshold, clip it
        eigs, V = np.linalg.eigh(self.Sigma)
        max_eig_threshold = 1e8  # Cap maximum uncertainty
        if np.max(eigs) > max_eig_threshold:
            eigs_clipped = np.clip(eigs, 0, max_eig_threshold)
            self.Sigma = V @ np.diag(eigs_clipped) @ V.T
            self.Sigma = sym(self.Sigma)


        self.t_prev = t
        self.t_last_imu = t

    # =========================================================================
    # Update functions
    # =========================================================================

    def magnetometer_update(self, mag: np.ndarray, t: float | None = None) -> None:
        """Yaw-only update from body-frame magnetometer — projects innovation onto NED-Z axis."""
        # Gate on measurement norm
        MAG_NORM_THRESHOLD = 0.15  # Loosened threshold to accept more measurements
        mag_norm = np.linalg.norm(mag)

        if mag_norm < MAG_NORM_THRESHOLD:
            return  # Measurement is noise-dominated, skip update

        # Normalize to unit length
        mag = mag / mag_norm

        # Predicted body-frame magnetic field direction
        R_hat = self.xi_hat().T.R().as_matrix()
        y_hat = R_hat.T @ MAG_FIELD_NED.flatten()
        y_hat /= np.linalg.norm(y_hat) + 1e-8

        # Full 3D rotation innovation (shortest-arc rotation from predicted to measured)
        R_err = from_two_vectors_rotation(y_hat, mag)
        delta_3d = SO3.log(SO3(R_err))  # (3,1), radians

        # Project onto body-Z (body yaw axis) → scalar yaw innovation
        body_z = np.array([[0.0], [0.0], [1.0]])
        delta_yaw = (body_z.T @ delta_3d).item()

        MAG_YAW_GATE = 0.01  # rad — skip when heading innovation is negligible
        if abs(delta_yaw) < MAG_YAW_GATE:
            return

        # 1×18 C matrix: full 3×3 attitude block projected onto yaw axis
        C_att = -SO3.wedge(y_hat.reshape(-1, 1))  # (3,3)
        C_yaw = np.zeros((1, 18))
        C_yaw[0, 0:3] = (body_z.T @ C_att).flatten()

        R_yaw = np.array([[self.R_mag[0, 0]]])  # scalar noise variance (1,1)

        # Kalman update
        S = C_yaw @ self.Sigma @ C_yaw.T + R_yaw  # (1,1)
        K = self.Sigma @ C_yaw.T @ np.linalg.inv(S)  # (18,1)
        Delta = self.innovationLift @ K * delta_yaw  # (18,1)

        # Compute ANIS for filter diagnostics
        anis = self.compute_anis(np.array([[delta_yaw]]), S)
        if t is not None and anis is not None:
            self.update_times.append(('mag', t, anis))

        self.X_hat = SymGroup.exp(Delta) * self.X_hat  # type: ignore[attr-defined]
        I = np.eye(18)
        IKC = I - K @ C_yaw
        self.Sigma = IKC @ self.Sigma @ IKC.T + K @ R_yaw @ K.T  # Joseph form
        self.Sigma = sym(self.Sigma)

        # Snapshot attitude after mag update; swap ZYX roll↔yaw slots to match FC convention
        R_post = self.xi_hat().T.R().as_matrix()
        _rot = ScipyRot.from_matrix(R_post)
        _e = _rot.as_euler('ZYX')
        _rot_mapped = ScipyRot.from_euler('ZYX', [_e[2], _e[1], _e[0]])  # swap roll↔yaw
        q = _rot_mapped.as_quat()  # [x,y,z,w]
        self.mag_q = np.array([q[3], q[0], q[1], q[2]])  # → [w,x,y,z]

        # Track successful magnetometer updates
        self.mag_update_count += 1


    def triad_attitude_update(self, R_meas: np.ndarray, t: float | None = None) -> None:
        """Full 3-DOF attitude update from a TRIAD-derived body→NED rotation matrix.

        Mirrors the C++ EqFalgo.cpp MagUpdate(Mat3) approach:
          innovation = SO3.log(R_meas @ R_hat.T)  [3-vector in so(3)]
          C = [I₃ | 0_{3×15}]                     rotation block only
        """
        R_hat = self.xi_hat().T.R().as_matrix()

        R_err = R_meas @ R_hat.T
        delta = SO3.log(SO3(R_err))  # (3,1)

        C = np.zeros((3, 18))
        C[0:3, 0:3] = np.eye(3)

        S = C @ self.Sigma @ C.T + self.R_triad
        K = self.Sigma @ C.T @ np.linalg.inv(S)
        Delta = self.innovationLift @ K @ delta

        anis = self.compute_anis(delta, S)
        if t is not None and anis is not None:
            self.update_times.append(('triad', t, anis))

        self.X_hat = SymGroup.exp(Delta) * self.X_hat  # type: ignore[attr-defined]
        I = np.eye(18)
        self.Sigma = (I - K @ C) @ self.Sigma
        Gamma = 0.5 * grp_adj(K @ delta)
        GammaExp = expm(Gamma)
        self.Sigma = GammaExp @ self.Sigma @ GammaExp.T
        self.Sigma = sym(self.Sigma)

        self.triad_update_count += 1


    def GNSS_update(self, pos_NED: np.ndarray, vel_NED: np.ndarray, t: float | None = None) -> None:
        """Correct position and velocity estimates using GNSS measurements."""
        # Get current estimates from composed state
        xi_hat = self.xi_hat()
        pos_est = xi_hat.T.w().as_vector().flatten()
        vel_est = xi_hat.T.x().as_vector().flatten()

        # Combine position and velocity innovations (6 measurements)
        delta = np.hstack([pos_NED - pos_est, vel_NED - vel_est])
        delta_u = delta.reshape(-1, 1)

        C = self.calculate_C_gnss()
        S = C @ self.Sigma @ C.T + self.R_gnss
        Sinv = np.linalg.inv(S)
        K = self.Sigma @ C.T @ Sinv
        Delta = K @ delta_u

        # Compute ANIS for filter diagnostics
        anis = self.compute_anis(delta_u, S)
        if t is not None and anis is not None:
            self.update_times.append(('gnss', t, anis))

        self.X_hat = SymGroup.exp(Delta) * self.X_hat  # type: ignore[attr-defined]
        I = np.eye(18)
        IKC = I - K @ C
        self.Sigma = IKC @ self.Sigma @ IKC.T + K @ self.R_gnss @ K.T  # Joseph form
        self.Sigma = sym(self.Sigma)

        # Compute ANEES using GNSS as ground truth (position and velocity only)
        # State error = [pos_est - pos_meas, vel_est - vel_meas]
        state_error = delta.reshape(-1, 1)  # Use innovation as state error estimate
        # Extract 6x6 covariance block for position (6:9) and velocity (3:6)
        P_pos_vel = np.block([
            [self.Sigma[6:9, 6:9], self.Sigma[6:9, 3:6]],
            [self.Sigma[3:6, 6:9], self.Sigma[3:6, 3:6]]
        ])
        anees = self.compute_anees(state_error, P_pos_vel)
        if t is not None and anees is not None:
            self.update_times.append(('gnss_anees', t, anees))

        # Track when last GNSS update occurred
        if t is not None:
            self.t_last_gnss = t

    # =========================================================================
    # Filter Diagnostics
    # =========================================================================

    def compute_anis(self, innovation: Any, S: Any) -> float | None:
        """Compute Average Normalized Innovation Squared.

        ANIS = innovation^T * S^{-1} * innovation
        Should be close to measurement dimension (~3 for 3D measurements)
        """
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

    def compute_anees(self, state_error: Any, P: Any) -> float | None:
        """Compute Average Normalized Estimation Error Squared.

        ANEES = error^T * P^{-1} * error
        Should be close to state dimension (~18 for full state)
        """
        try:
            err: np.ndarray = np.asarray(state_error, dtype=float)
            p: np.ndarray = np.asarray(P, dtype=float)
            P_inv: np.ndarray = np.linalg.inv(p)
            anees_mat: np.ndarray = err.T @ P_inv @ err
            anees = float(np.squeeze(anees_mat))
            self.anees_values.append(anees)
            return anees
        except (np.linalg.LinAlgError, ValueError):
            return None

    # =========================================================================

    def output_row(self, t: float) -> list[float]:
        """Extract state as output row [t, p, v, R, euler, b_gyro, b_accel]."""
        xi_hat = self.xi_hat()
        R = xi_hat.T.R().as_matrix()
        v = col(xi_hat.T.x().as_vector())
        p = col(xi_hat.T.w().as_vector())
        b = xi_hat.b  # 9-vector: [b_gyro(3); b_accel(3); b_mu(3)]

        # Convert R to unit quaternion — no Euler singularity
        q = ScipyRot.from_matrix(R).as_quat()  # [x,y,z,w]
        qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])

        return [
            t,
            p[0, 0],
            p[1, 0],
            p[2, 0],
            v[0, 0],
            v[1, 0],
            v[2, 0],
            R[0, 0],
            R[0, 1],
            R[0, 2],
            R[1, 0],
            R[1, 1],
            R[1, 2],
            R[2, 0],
            R[2, 1],
            R[2, 2],
            qw,                  # quaternion w
            qx,                  # quaternion x
            qy,                  # quaternion y
            qz,                  # quaternion z
            b[0, 0] if b.shape[0] > 0 else 0.0,  # Gyro X bias
            b[1, 0] if b.shape[0] > 1 else 0.0,  # Gyro Y bias
            b[2, 0] if b.shape[0] > 2 else 0.0,  # Gyro Z bias
            b[3, 0] if b.shape[0] > 3 else 0.0,  # Accel X bias
            b[4, 0] if b.shape[0] > 4 else 0.0,  # Accel Y bias
            b[5, 0] if b.shape[0] > 5 else 0.0,  # Accel Z bias
            b[6, 0] if b.shape[0] > 6 else 0.0,  # Virtual bias X (b_mu)
            b[7, 0] if b.shape[0] > 7 else 0.0,  # Virtual bias Y
            b[8, 0] if b.shape[0] > 8 else 0.0,  # Virtual bias Z
            *self.mag_q.tolist(),                  # quaternion [w,x,y,z] at last mag update
        ]
    
    # =============================================================================
    # Propagation Jacobians
    # =============================================================================

    def calculate_A(self, u: InputSpace) -> np.ndarray:
        u_0 = velocityAction(self.X_hat.inv(), u)
        At = np.zeros((18, 18))
        At[0:9, 0:9] = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [SO3.wedge(col((0,0,g))), np.zeros((3, 3)), np.zeros((3,3))],
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))]
        ])
        w_vec = u_0.as_W_vec()
        g_vec = SE23.vee(G)
        At[9:18, 9:18] = SE23.adjoint(w_vec + g_vec)
        At[0:9, 9:18] = np.eye(9)


        return At

    def calculate_C_mag(self) -> np.ndarray:
        Ct = np.zeros((3, 18))

        # Project reference magnetometer to body frame using composed state
        xi_hat = self.xi_hat()
        y_hat = xi_hat.T.R().as_matrix().T @ self.mag_meas_0
        y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-8)  # Normalize (keep as column vector for SO3.wedge)

        Ct[0:3, 0:3] = -SO3.wedge(y_hat)

        return Ct



    def calculate_C_gnss(self) -> np.ndarray:
        # 6x18 C matrix for position and velocity measurements
        Ct = np.zeros((6, 18))

        # Get composed state for measurement Jacobian
        xi_hat = self.xi_hat()

        # Position measurement: C_p = [-wedge(p), 0, I3, 0, 0, 0]
        Ct[0:3, 0:3] = -SO3.wedge(xi_hat.T.w().as_vector())
        Ct[0:3, 6:9] = np.eye(3)

        # Velocity measurement: C_v = [-wedge(v), I3, 0, 0, 0, 0]
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


# =============================================================================
# Magnetometer axis configuration
# =============================================================================
# MAG_AXIS_ORDER: which raw sensor axis goes to body [x, y, z]
#   default [0,1,2] = identity (mx→x, my→y, mz→z)
# MAG_AXIS_SIGNS: sign applied to each body axis after permutation
#   default [1, 1, 1]; negate if sensor axis is mounted in opposite direction
MAG_AXIS_ORDER = np.array([0, 2, 1])   # sensor: mx→body X, mz→body Y, my→body Z
MAG_AXIS_SIGNS = np.array([1.0, -1.0, -1.0])   # tuned: confirmed by FC/GNSS yaw alignment

def run(csv_in: str | None = None, csv_out: str = "outputs/tg_eqf_output.csv",
        mag_axis_order: np.ndarray | None = None,
        mag_axis_signs: np.ndarray | None = None,
        r_mag_var: float | None = None,
        silent: bool = False) -> dict[str, float]:
    """Run filter on NIMBUS24 FC CSV data. Returns RMSE dict (keys: roll, pitch, yaw, deg)."""
    axis_order = MAG_AXIS_ORDER if mag_axis_order is None else mag_axis_order
    axis_signs = MAG_AXIS_SIGNS if mag_axis_signs is None else mag_axis_signs

    # Select data source based on configuration
    if csv_in is None:
        _datasets = {
            "full":    ("data/20241011_NIMBUS24_Flight_FC_Data.csv",         "outputs/tg_eqf_output_full.csv"),
            "30s":     ("data/20241011_NIMBUS24_Flight_FC_Data_30s.csv",      "outputs/tg_eqf_output_30s.csv"),
            "1s_loop": ("data/20241011_NIMBUS24_Flight_FC_Data_1s_loop.csv",  "outputs/tg_eqf_output_1s_loop.csv"),
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
    gnss_count = 0
    last_progress_t = 0

    # ICM_20608 gyro scale factor: raw ADC counts to rad/s
    gyro_scale_factor =  (np.pi / 180.0) #* (2000.0 / 32768.0)

    # Pre-initialize attitude via TRIAD before the main loop, so the filter
    # propagates from t=0 with the correct initial orientation.
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

    for i, row in enumerate(raw):
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        # Sensor→body frame: ax→X, ay→Y, az→Z (matches FC body frame, TRIAD gives pitch≈84.5°)
        gyro  = row[[_C["gx"], _C["gy"], _C["gz"]]] * gyro_scale_factor
        accel = row[[_C["ax"], _C["ay"], _C["az"]]] * g


        if not np.all(np.isfinite(np.concatenate([gyro, accel]))):
            continue

        filt.propagate(t, gyro, accel)

        # Extract magnetometer data
        mag_raw = row[[_C["mx"], _C["my"], _C["mz"]]]

        # Remap sensor axes to body frame
        mag = mag_raw[axis_order] * axis_signs

        # Normalize to unit length
        mag_norm = np.linalg.norm(mag)
        if mag_norm > 1e-6:  # Avoid division by zero
            mag = mag / mag_norm
        # TRIAD initialization: trigger on first valid mag reading before filter starts
        if not filt.attitude_initialized:
            filt.initialize_attitude_triad(mag)

        # Magnetometer update
        _DEBUG_WINDOW = 75.0 <= t <= 85.0
        if filt.attitude_initialized and (prev_mag is None or not np.allclose(prev_mag, mag)):
            _b_before = filt.xi_hat().b[3:6].flatten().copy() if _DEBUG_WINDOW else None
            filt.magnetometer_update(mag, t=t)
            prev_mag = mag
            if _DEBUG_WINDOW:
                _b_after = filt.xi_hat().b[3:6].flatten()
                print(f"[DBG t={t:.2f}] MAG update  | ba_before={_b_before}  ba_after={_b_after}  delta={_b_after - _b_before}")

        lat = row[_C["lat"]]
        lon = row[_C["lon"]]
        if lat != 0 and lon != 0:
            # Apply GNSS update at configured frequency, regardless of position change
            gnss_update_period = 1.0 / GNSS_UPDATE_FREQ_HZ
            time_since_last_gnss = gnss_update_period if filt.t_last_gnss is None else (t - filt.t_last_gnss)
            rate_ok = filt.t_last_gnss is None or time_since_last_gnss >= gnss_update_period

            if rate_ok:
                # Apply GNSS update
                alt = row[_C["alt"]] / 1000.0
                pos_NED = _gps_to_ned(lat, lon, alt, lat0, lon0, alt0)
                vel_NED = np.array([row[_C["gps_vn"]], row[_C["gps_ve"]], row[_C["gps_vd"]]]) / 1000.0
                if _DEBUG_WINDOW:
                    _b_before = filt.xi_hat().b[3:6].flatten().copy()
                    _pos_est = filt.xi_hat().T.w().as_vector().flatten()
                    _vel_est = filt.xi_hat().T.x().as_vector().flatten()
                    print(f"[DBG t={t:.2f}] GNSS update | pos_inn={pos_NED - _pos_est}  vel_inn={vel_NED - _vel_est}")
                filt.GNSS_update(pos_NED, vel_NED, t)
                gnss_count += 1
                if _DEBUG_WINDOW:
                    _b_after = filt.xi_hat().b[3:6].flatten()
                    print(f"[DBG t={t:.2f}]              | ba_before={_b_before}  ba_after={_b_after}  delta={_b_after - _b_before}")


        out.append(filt.output_row(t))
        fc_att_list.append([
            float(row[_C["roll_fc"]]),
            float(row[_C["pitch_fc"]]),
            float(row[_C["yaw_fc"]]),
        ])

        # Progress reporting with covariance diagnostics
        if not silent and t - last_progress_t >= 30:  # Every 30 seconds
            progress = 100.0 * i / len(raw)
            print(f"[PROGRESS] {progress:.1f}% | t={t:.2f}s | rows={i}/{len(raw)} | GNSS_updates={gnss_count}")

            # Pretty-print covariance (diagonal standard deviations)
            sigmas = np.sqrt(np.diag(filt.Sigma))
            print(f"  Covariance (std dev):  Att:{sigmas[0:3].mean():.2f}rad  Vel:{sigmas[3:6].mean():.2f}m/s  "
                  f"Pos:{sigmas[6:9].mean():.1f}m  GyBias:{sigmas[9:12].mean():.4f}rad/s  "
                  f"AcBias:{sigmas[12:15].mean():.3f}m/s²  VBias:{sigmas[15:18].mean():.2e}")
            last_progress_t = t

    out = np.asarray(out)
    header = (
        "t,px,py,pz,vx,vy,vz,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
        "qw,qx,qy,qz,"
        "bgx,bgy,bgz,bax,bay,baz,bmux,bmuy,bmuz,"
        "mag_qw,mag_qx,mag_qy,mag_qz"
    )

    if not silent:
        np.savetxt(csv_out, out, delimiter=",", header=header, comments="")
        print(f"Wrote {len(out)} rows to {csv_out}")
        print(f"Magnetometer updates applied: {filt.mag_update_count}")
        print(f"TRIAD attitude updates applied: {filt.triad_update_count}")

    # Save diagnostic data (ANIS/ANEES values with timestamps)
    if filt.update_times:
        diag_out = csv_out.replace("tg_eqf_output", "tg_eqf_diagnostics")
        with open(diag_out, 'w') as f:
            f.write("time,update_type,anis,anees\n")
            anis_dict: dict[tuple[str, float], float] = {}
            anees_dict: dict[tuple[str, float], float] = {}
            # Organize by (update_type, time) to combine anis and anees
            for update_type, ts, value in filt.update_times:
                key = (update_type.split('_')[0], ts)
                if 'anees' in update_type:
                    anees_dict[key] = value
                else:
                    anis_dict[key] = value

            # Merge and write
            all_times: dict[tuple[str, float], dict[str, float | None]] = {}
            for key, val in anis_dict.items():
                if key not in all_times:
                    all_times[key] = {'anis': None, 'anees': None}
                all_times[key]['anis'] = val
            for key, val in anees_dict.items():
                if key not in all_times:
                    all_times[key] = {'anis': None, 'anees': None}
                all_times[key]['anees'] = val

            for (update_type, ts), values in sorted(all_times.items()):
                anis_str = f"{values['anis']:.4f}" if values['anis'] is not None else ""
                anees_str = f"{values['anees']:.4f}" if values['anees'] is not None else ""
                f.write(f"{ts:.4f},{update_type},{anis_str},{anees_str}\n")
        if not silent:
            print(f"Wrote diagnostic data to {diag_out}")

    # Compute angular RMSE between filter and FC attitude (quaternion-based, no gimbal lock)
    fc_att_arr = np.array(fc_att_list)
    valid_att = (np.all(np.isfinite(fc_att_arr), axis=1) &
                 np.all(np.isfinite(out[:, 16:20]), axis=1))
    rmse: dict[str, float] = {"angular": float('nan')}
    if valid_att.any():
        q_filt = out[valid_att, 16:20]  # [w,x,y,z]
        roll_fc  = fc_att_arr[valid_att, 0]
        pitch_fc = fc_att_arr[valid_att, 1]
        yaw_fc   = fc_att_arr[valid_att, 2]
        q_fc_xyzw = ScipyRot.from_euler('ZYX', np.column_stack([yaw_fc, pitch_fc, roll_fc])).as_quat()
        q_fc = q_fc_xyzw[:, [3, 0, 1, 2]]  # → [w,x,y,z]
        dot = np.clip(np.abs(np.sum(q_filt * q_fc, axis=1)), 0.0, 1.0)
        ang_err_deg = np.degrees(2.0 * np.arccos(dot))
        rmse = {"angular": float(np.sqrt(np.mean(ang_err_deg**2)))}

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
                print(f"ANIS [{mtype}]: mean={np.mean(arr):.2f}  max={np.max(arr):.2f}  n={len(arr)}")

        if filt.anees_values:
            anees_array = np.array(filt.anees_values)
            print(f"ANEES (avg normalized estimation error squared):")
            print(f"  Mean: {np.mean(anees_array):.2f} (should be ~1 for measurement dimension)")
            print(f"  Std:  {np.std(anees_array):.2f}")
            print(f"  Min:  {np.min(anees_array):.2f}, Max: {np.max(anees_array):.2f}")

    return rmse


def tune_magnetometer() -> None:
    """Try all 8 sign combinations × 6 axis permutations, then sweep R_mag_var on the best."""
    import itertools

    sign_combos = list(itertools.product([-1.0, 1.0], repeat=3))
    axis_orders = [
        [0, 1, 2],  # identity
        [0, 2, 1],  # swap y/z
        [1, 0, 2],  # swap x/y
        [1, 2, 0],  # x→y, y→z, z→x
        [2, 0, 1],  # x→z, y→x, z→y
        [2, 1, 0],  # swap x/z
    ]

    # --- Phase 1: axis sweep ---
    axis_results: list[tuple[list[int], tuple[float, float, float], dict[str, float]]] = []
    n_total = len(axis_orders) * len(sign_combos)
    print(f"Phase 1: Testing {n_total} axis configurations ...")

    for order in axis_orders:
        for signs in sign_combos:
            rmse = run(
                mag_axis_order=np.array(order),
                mag_axis_signs=np.array(signs),
                silent=True,
            )
            axis_results.append((order, signs, rmse))

    axis_results.sort(key=lambda r: r[2]["angular"])

    print(f"\n{'Order':<12} {'Signs':<18} {'Angular RMSE':>14}")
    print("-" * 48)
    for order, signs, rmse in axis_results:
        sign_str = f"[{signs[0]:+.0f},{signs[1]:+.0f},{signs[2]:+.0f}]"
        order_str = f"[{order[0]},{order[1]},{order[2]}]"
        print(f"{order_str:<12} {sign_str:<18} {rmse['angular']:>12.2f}°")

    best_order, best_signs, _ = axis_results[0]
    print(f"\nBest axis config: order={best_order}  signs={list(best_signs)}")

    # --- Phase 2: R_mag_var sweep using best axis ---
    r_vars = [0.01, 0.1, 0.5, 1.0, 4.0, 9.0, 16.0, 36.0, 100.0, 400.0, 1600.0]
    print(f"\nPhase 2: Sweeping R_mag_var over {r_vars} ...")
    print(f"\n{'R_mag_var':>12} {'Angular RMSE':>14}")
    print("-" * 28)
    noise_results: list[tuple[float, dict[str, float]]] = []
    for rv in r_vars:
        rmse = run(
            mag_axis_order=np.array(best_order),
            mag_axis_signs=np.array(best_signs),
            r_mag_var=rv,
            silent=True,
        )
        noise_results.append((rv, rmse))
        print(f"{rv:>12.3f} {rmse['angular']:>12.2f}°")

    noise_results.sort(key=lambda r: r[1]["angular"])
    best_rv, best_rmse = noise_results[0]
    print(f"\nBest R_mag_var: {best_rv}")
    print(f"  Angular RMSE: {best_rmse['angular']:.2f}°")
    print(f"\nFinal recommended config:")
    print(f"  MAG_AXIS_ORDER = {best_order}")
    print(f"  MAG_AXIS_SIGNS = {list(best_signs)}")
    print(f"  R_mag_var      = {best_rv}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        tune_magnetometer()
    else:
        run()