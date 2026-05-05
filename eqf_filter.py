"""Standalone CNS TG-EqF inspired inertial/GNSS filter using pylie."""

import os
import sys
from pathlib import Path
from enum import Enum

import numpy as np
from scipy.linalg import expm
from pylie import SO3, SE23, R3

# =============================================================================
# Todo:
# =============================================================================
"""
Make it actually work.
Make sure the math is proper.
Make sure that we are not doing things not described in the paper

"""


ref_path = Path(__file__).parent / "eqf-reference"
sys.path.insert(0, str(ref_path))
utils_path = ref_path / "Utils"
sys.path.insert(0, str(utils_path))
from matrix_math import *
from Symmetries.Calibrated.SE23_se23.Symmetry import SymGroup, State, InputSpace, stateAction, velocityAction, f_10, grp_adj, local_coords, local_coords_inv, stateActionDiff
# =============================================================================
# Configuration
# =============================================================================

USE_STATIC_DATA = False  # "combined" = static + flight, False = flight only
                         # - False: runs on real NIMBUS24 flight data
                         # - True:  runs on truly_static_30s.csv (zero gyro/accel)
                         # - "gravity": runs on gravity_only_30s.csv (ax=1g, rest=0)
                         # - "combined": runs on static (30s) + full flight (~150s)

GNSS_UPDATE_FREQ_HZ = 1.0   # GNSS update frequency (Hz) — update every 1/f seconds
GNSS_POS_ONLY = False
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
Q_rot_var = 1e-1               # [0:3] rotation process noise
Q_vel_var = 1e+1                # [3:6] velocity process noise
Q_pos_var = 1e+1                # [6:9] position process noise
Q_gyro_bias_var = (1e-3)**2     # [9:12] gyro bias random walk (very tight)
Q_accel_bias_var = (1e-2)**2    # [12:15] accel bias random walk (very tight)
Q_virtual_bias_var = 1e-7      # [15:18] virtual accel bias (frozen)

# --- Measurement Noise (R) ---
# GNSS measurement noise
R_gnss_pos_var = 1            # GNSS position measurement variance (m²)
R_gnss_vel_var = 5.0           # GNSS velocity measurement variance (m/s)²

# Magnetometer measurement noise (realistic 1-2 degree heading sensor)
R_mag_var = (6.0)**2         # Magnetometer variance (nanoTesla²)

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
MAG_FIELD_NED = np.array([wmm_north, wmm_east, wmm_down])
MAG_FIELD_NED = MAG_FIELD_NED / np.linalg.norm(MAG_FIELD_NED)  # Normalize

# Hard-iron bias correction (body frame offset)
# Computed from static phase analysis: mag_body_mean from diagnose_magnetometer.py
MAG_HARD_IRON_BIAS = np.array([-0.92683517, 0.11646937, -0.20916137])

# Magnetometer gating thresholds
MAG_FIELD_MAGNITUDE_NOM = 47000.0  # nanoTesla
MAG_FIELD_DEVIATION_THRESHOLD = 0.15  # Reject if >15% deviation
MAG_ACCEL_GATE_THRESHOLD = 15.0  # Disable during boost (accel > 15 m/s²)


# =============================================================================
# Helpers
# =============================================================================


def col(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)


def sym(A):
    return 0.5 * (A + A.T)

def blockdiag(*arrs):
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
G[2, 3] = -g


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

    def initialize_attitude_from_gravity(self, accel: np.ndarray, gyro: np.ndarray):
        """Initialize attitude from FC's initial estimate (roll, pitch, yaw from flight computer).

        Uses the flight computer's initial Euler angles to set up the filter's rotation matrix.
        """
        if self.attitude_initialized or self.t_prev is None:
            return

        if True:  # Accept any valid data
            # FC initial attitude (from 20241011_NIMBUS24_Flight_FC_Data.csv first row)
            roll_fc = -1.043745  # radians
            pitch_fc = 1.478157  # radians
            yaw_fc = 2.772124    # radians

            # Convert to rotation matrix using intrinsic Z-Y-X rotations
            R_init = self.euler_to_rotation_matrix(roll_fc, pitch_fc, yaw_fc)

            # Verify orthogonality
            R_check = R_init.T @ R_init
            is_orthogonal = np.allclose(R_check, np.eye(3), atol=1e-6)
            det_R = np.linalg.det(R_init)

            # Create SE(2,3) with FC's initial rotation
            se23_init = SE23(R_init)
            self.X_hat = SymGroup(se23_init, np.zeros((5, 5)))

            self.attitude_initialized = True

    def __init__(self):
        """Initialize filter state and covariance matrices."""
        self.X_hat = SymGroup.identity()
        self.xi_0 = State()
        self.t_prev = None
        self.t_last_gnss = None
        self.t_last_imu = None
        self.innovationLift = np.linalg.pinv(stateActionDiff(self.xi_0))
        self.attitude_initialized = False  # Smart initialization from gravity

        # Moving average filter for smoothing noisy sensor measurements
        self.ma_window = 5  # Reduced to minimize lag (5-10 samples)
        self.gyro_buffer = []
        self.accel_buffer = []
        self.mag_buffer = []

        # Magnetometer values
        self.mag_meas_0 = MAG_FIELD_NED.reshape(-1, 1)  # WMM field in NED
        self.magnetometer_initialized = True  # Pre-initialized with WMM
        self.accel_norm_prev = 0.0  # For magnetometer gating

        self.prev_altitude = 0.0
        self.mag_update_count = 0  # Track magnetometer update calls
        self.update_count = 0  # Counter for periodic re-orthonormalization

        # Euler angle unwrapping for continuous output
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0

        # Filter diagnostics: ANIS and ANEES
        self.anis_values = []  # Average Normalized Innovation Squared
        self.anees_values = []  # Average Normalized Estimation Error Squared
        self.update_times = []  # Times of updates for diagnostics

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

    def _apply_moving_average(self, measurement, buffer):
        """Apply moving average filter with window size of 10."""
        buffer.append(measurement.flatten())
        if len(buffer) > self.ma_window:
            buffer.pop(0)
        return np.mean(buffer, axis=0).reshape(-1, 1)

    @staticmethod
    def _unwrap_angle(angle_raw, angle_prev):
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

    def propagate(self, t, gyro, accel):
        """Propagate state using gyro and accel measurements."""
        gyro = col(gyro)
        accel = col(accel)

        if self.t_prev is None:
            self.t_prev = t
            return

        # Smart initialization: estimate attitude from gravity during static phase
        if not self.attitude_initialized:
            self.initialize_attitude_from_gravity(accel, gyro)

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


        # Debug: Print input and state before propagation
        #if int(t * 10) % 10 == 0:  # Every 1 second
        #    p_before = self.X_hat.B.w().as_vector().flatten()
        #    v_before = self.X_hat.B.x().as_vector().flatten()
        #    b_before = xi_hat.b.flatten()
        #    print(f"[BEFORE] t={t:.2f}s | gyro={gyro.flatten()} | accel={accel.flatten()}")
        #    print(f"         p=[{p_before[0]:7.1f}, {p_before[1]:7.1f}, {p_before[2]:7.1f}]m | v=[{v_before[0]:6.2f}, {v_before[1]:6.2f}, {v_before[2]:6.2f}]m/s")
        #    print(f"         b_mu={b_before[6:9]}")

        lift = calculate_lift(self.xi_hat(), U)

        self.X_hat = self.X_hat * SymGroup.exp(lift * dt)

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

        # Debug: Print filter state periodically
        #if int(t * 10) % 10 == 0:  # Every 1 second
        #    p = self.X_hat.B.w().as_vector().flatten()
        #    v = self.X_hat.B.x().as_vector().flatten()
        #    b = self.xi_hat().b.flatten()
        #    trace_P = np.trace(self.Sigma)
        #    print(f"[AFTER]  t={t:.2f}s | p=[{p[0]:7.1f}, {p[1]:7.1f}, {p[2]:7.1f}]m | v=[{v[0]:6.2f}, {v[1]:6.2f}, {v[2]:6.2f}]m/s")
        #    print(f"         b_mu={b[6:9]} | P_trace={trace_P:.2e}\n")

    # =========================================================================
    # Update functions
    # =========================================================================

    def magnetometer_update(self, mag: np.ndarray, t=None):
        """Update state using magnetometer measurement (hard-iron corrected).

        Args:
            mag: hard-iron corrected magnetometer measurement (3,)
            t: timestamp (optional)
        """
        # Gate on corrected measurement norm
        # After hard-iron bias subtraction, small norms indicate noise-dominated signal
        MAG_NORM_THRESHOLD = 0.15  # Loosened threshold to accept more measurements
        mag_norm = np.linalg.norm(mag)

        if mag_norm < MAG_NORM_THRESHOLD:
            return  # Measurement is noise-dominated, skip update

        # Normalize to unit length
        mag = mag / mag_norm

        # Get C matrix for magnetometer measurement
        C = self.calculate_C_mag()

        # Compute innovation using composed state with WMM reference field
        xi_hat = self.xi_hat()
        y_hat = xi_hat.T.R().as_matrix().T @ self.mag_meas_0
        y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-8)  # Normalize prediction (keep as column vector)

        # Innovation: project measurement residual into SO(3) tangent space
        # C matrix is -SO3.wedge(y_hat), so delta should be SO3.wedge(y_hat) @ mag
        mag_col = mag.reshape(-1, 1)
        delta = SO3.wedge(y_hat) @ mag_col
        delta_u = delta

        # Kalman update
        S = C @ self.Sigma @ C.T + self.R_mag
        Sinv = np.linalg.inv(S)
        K = self.Sigma @ C.T @ Sinv
        Delta = self.innovationLift @ K @ delta_u

        # Compute ANIS for filter diagnostics
        anis = self.compute_anis(delta_u, S)
        if t is not None and anis is not None:
            self.update_times.append(('mag', t, anis))

        self.X_hat = SymGroup.exp(Delta) * self.X_hat
        # Joseph form
        I = np.eye(18)
        self.Sigma = (I - K @ C) @ self.Sigma @ (I - K @ C).T + K @ self.R_mag @ K.T
        self.Sigma = sym(self.Sigma)

        # Track successful magnetometer updates
        self.mag_update_count += 1


    def GNSS_update(self, pos_NED:np.ndarray ,vel_NED:np.ndarray, t=None):
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

        self.X_hat = SymGroup.exp(Delta) * self.X_hat
        #Joseph form
        I = np.eye(18)
        self.Sigma = (I - K @ C) @ self.Sigma @ (I - K @ C).T + K @ self.R_gnss @ K.T
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

    def GNSS_update_pos_only(self, pos_NED:np.ndarray, t=None):
        """Correct position and velocity estimates using GNSS measurements."""
        # Get current estimates from composed state
        xi_hat = self.xi_hat()
        pos_est = xi_hat.T.w().as_vector().flatten()

        #position innovation
        delta = np.hstack([pos_NED - pos_est])
        delta_u = delta.reshape(-1, 1)

        C = self.calculate_C_gnss_pos_only(pos_NED)
        S = C @ self.Sigma @ C.T + self.R_gnss_pos_only
        Sinv = np.linalg.inv(S)
        K = self.Sigma @ C.T @ Sinv
        Delta = self.innovationLift @ K @ delta_u

        self.X_hat = SymGroup.exp(Delta) * self.X_hat
        #Joseph form
        I = np.eye(18)
        self.Sigma = (I - K @ C) @ self.Sigma @ (I - K @ C).T + K @ self.R_gnss_pos_only @ K.T
        self.Sigma = sym(self.Sigma)

        # Track when last GNSS update occurred
        if t is not None:
            self.t_last_gnss = t

    # =========================================================================
    # Filter Diagnostics
    # =========================================================================

    def compute_anis(self, innovation, S):
        """Compute Average Normalized Innovation Squared.

        ANIS = innovation^T * S^{-1} * innovation
        Should be close to measurement dimension (~3 for 3D measurements)
        """
        try:
            S_inv = np.linalg.inv(S)
            anis_mat = innovation.T @ S_inv @ innovation
            anis = float(np.squeeze(anis_mat))  # Extract scalar from matrix
            self.anis_values.append(anis)
            return anis
        except (np.linalg.LinAlgError, ValueError):
            return None

    def compute_anees(self, state_error, P):
        """Compute Average Normalized Estimation Error Squared.

        ANEES = error^T * P^{-1} * error
        Should be close to state dimension (~18 for full state)
        """
        try:
            P_inv = np.linalg.inv(P)
            anees_mat = state_error.T @ P_inv @ state_error
            anees = float(np.squeeze(anees_mat))  # Extract scalar from matrix
            self.anees_values.append(anees)
            return anees
        except (np.linalg.LinAlgError, ValueError):
            return None

    # =========================================================================

    def output_row(self, t):
        """Extract state as output row [t, p, v, R, euler, b_gyro, b_accel]."""
        # Get full composed state (group action applied to reference state)
        xi_hat = self.xi_hat()
        R = xi_hat.T.R().as_matrix()
        v = col(xi_hat.T.x().as_vector())
        p = col(xi_hat.T.w().as_vector())
        b = xi_hat.b  # 9-vector: [b_gyro(3); b_accel(3); b_mu(3)]

        # Compute Euler angles (roll, pitch, yaw) from rotation matrix
        # FIXED: SE23.exp() produces intrinsic (body-frame) rotations, not extrinsic
        # For body-frame Z-Y-X rotations: R = Rx(roll) * Ry(pitch) * Rz(yaw)
        # Correct extraction:
        roll_raw = np.arctan2(R[2, 1], R[2, 2])
        pitch_raw = np.arcsin(np.clip(-R[2, 0], -1, 1))
        yaw_raw = np.arctan2(R[1, 0], R[0, 0])

        # Unwrap Euler angles to avoid discontinuous jumps at ±π
        roll = self._unwrap_angle(roll_raw, self.prev_roll)
        pitch = self._unwrap_angle(pitch_raw, self.prev_pitch)
        yaw = self._unwrap_angle(yaw_raw, self.prev_yaw)

        # Store for next unwrapping
        self.prev_roll = roll
        self.prev_pitch = pitch
        self.prev_yaw = yaw

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
            roll,                # Euler angle (rad)
            pitch,               # Euler angle (rad)
            yaw,                 # Euler angle (rad)
            b[0, 0] if b.shape[0] > 0 else 0.0,  # Gyro X bias
            b[1, 0] if b.shape[0] > 1 else 0.0,  # Gyro Y bias
            b[2, 0] if b.shape[0] > 2 else 0.0,  # Gyro Z bias
            b[3, 0] if b.shape[0] > 3 else 0.0,  # Accel X bias
            b[4, 0] if b.shape[0] > 4 else 0.0,  # Accel Y bias
            b[5, 0] if b.shape[0] > 5 else 0.0,  # Accel Z bias
            b[6, 0] if b.shape[0] > 6 else 0.0,  # Virtual bias X (b_mu)
            b[7, 0] if b.shape[0] > 7 else 0.0,  # Virtual bias Y
            b[8, 0] if b.shape[0] > 8 else 0.0,  # Virtual bias Z
        ]
    
    # =============================================================================
    # Propagation Jacobians
    # =============================================================================

    def calculate_A(self, u : InputSpace) -> np.ndarray:
        u_0 = velocityAction(self.X_hat.inv(), u)
        At = np.zeros((18, 18))
        At[0:9, 0:9] = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [SO3.wedge(col((0,0,-9.81))), np.zeros((3, 3)), np.zeros((3,3))],
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))]
        ])
        w_vec = u_0.as_W_vec()
        g_vec = SE23.vee(G)
        At[9:18, 9:18] = SE23.adjoint(w_vec + g_vec)
        At[0:9, 9:18] = np.eye(9)


        return At

    def calculate_B(self, u : InputSpace) -> np.ndarray:
        Bt = np.block([self.X_hat.B.Adjoint(),np.zeros((9,9))],
                      [self.X_hat.B.Adjoint(),np.zeros((9,9))])
        return Bt

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

        # Compute numerical Jacobian for bias columns (9:18)
        # This captures how position and velocity depend on gyro/accel/virtual biases
        epsilon = 1e-6

        def measure_state(state_delta):
            """Measure position and velocity given state perturbation."""
            # Create perturbed state by group exponential
            X_perturbed = SymGroup.exp(state_delta) * self.X_hat
            # Get composed state
            xi_perturbed = stateAction(X_perturbed, self.xi_0)
            # Extract position and velocity
            pos = xi_perturbed.T.w().as_vector().flatten()
            vel = xi_perturbed.T.x().as_vector().flatten()
            return np.hstack([pos, vel])

        # Reference measurement
        z_ref = np.hstack([xi_hat.T.w().as_vector().flatten(),
                          xi_hat.T.x().as_vector().flatten()])

        # Numerical differentiation for bias columns
        for j in range(9, 18):  # Columns 9:18 correspond to biases
            delta = np.zeros((18, 1))
            delta[j, 0] = epsilon
            z_perturbed = measure_state(delta)
            Ct[:, j] = (z_perturbed - z_ref) / epsilon

        return Ct
    
    def calculate_C_gnss_pos_only(self, meas) -> np.ndarray:
        # 3x18 C matrix for position-only measurements
        Ct = np.zeros((3, 18))

        # Get composed state for measurement Jacobian
        xi_hat = self.xi_hat()

        # Position measurement: C_p = [-wedge(p), 0, I3, 0, 0, 0]
        Ct[0:3, 0:3] = -0.5 * (SO3.wedge(xi_hat.T.w().as_vector()) + meas)
        Ct[0:3, 6:9] = np.eye(3)

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
}
R_EARTH = 6_378_137.0


def _gps_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """Convert GPS (lat/lon/alt) to NED relative to reference."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    north = dlat * R_EARTH
    east = dlon * np.cos(lat0_rad) * R_EARTH
    down = -(alt - alt0)
    return np.array([north, east, down])


def run(csv_in=None, csv_out="outputs/tg_eqf_output.csv"):
    """Run filter on NIMBUS24 FC CSV data."""
    # Select data source based on configuration
    if csv_in is None:
        if USE_STATIC_DATA == "combined":
            csv_in = "data/20241011_NIMBUS24_combined_static_flight.csv"
            csv_out = "outputs/tg_eqf_output_combined.csv"
        elif USE_STATIC_DATA == "gravity":
            csv_in = "data/20241011_NIMBUS24_gravity_only_30s.csv"
            csv_out = "outputs/tg_eqf_output_gravity.csv"
        elif USE_STATIC_DATA:
            csv_in = "data/20241011_NIMBUS24_truly_static_30s.csv"
            csv_out = "outputs/tg_eqf_output_static.csv"
        else:
            csv_in = "data/20241011_NIMBUS24_Flight_FC_Data.csv"

    raw = np.genfromtxt(csv_in, delimiter=",", skip_header=1)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    print(f"Loaded {len(raw)} rows")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    valid = (raw[:, _C["lat"]] != 0) & (raw[:, _C["lon"]] != 0)
    first = np.argmax(valid)
    lat0 = raw[first, _C["lat"]]
    lon0 = raw[first, _C["lon"]]
    alt0 = raw[first, _C["alt"]] / 1000.0

    filt = TGEqF()
    out = []
    prev_mag = None
    prev_lat, prev_lon = None, None
    R_pos = np.eye(3) * 5.0**2
    gnss_count = 0
    last_progress_t = 0

    # ICM_20608 gyro scale factor: raw ADC counts to degrees/sec
    # Then convert to rad/s: deg/s * pi/180. This took way too long to find out
    gyro_scale_factor =  (np.pi / 180.0) * (2000.0 / 32768.0) # raw ADC -> rad/s

    for i, row in enumerate(raw):
        t = row[_C["t"]]
        if not np.isfinite(t):
            continue

        gyro = row[[_C["gx"], _C["gy"], _C["gz"]]] * gyro_scale_factor  # rad/s
        accel = row[[_C["ax"], _C["ay"], _C["az"]]] * g  # m/s²


        if not np.all(np.isfinite(np.concatenate([gyro, accel]))):
            continue

        filt.propagate(t, gyro, accel)

        # Extract magnetometer data
        mag_raw = row[[_C["mx"], _C["my"], _C["mz"]]]

        # Apply hard-iron bias correction (remove uncalibrated offset)
        mag = mag_raw - MAG_HARD_IRON_BIAS

        # Negate Y axis (empirically improves heading alignment)
        #mag[1] = -mag[1]

        # Normalize to unit length
        mag_norm = np.linalg.norm(mag)
        if mag_norm > 1e-6:  # Avoid division by zero
            mag = mag / mag_norm
        # Initialize magnetometer reference with first measurement after attitude initialization
        # This captures the actual sensor field (body frame) at the initial attitude
        if filt.attitude_initialized and not hasattr(filt, '_mag_ref_initialized'):
            # Project first measurement into NED frame using the initialized attitude
            xi_hat = filt.xi_hat()
            R_body_to_ned = xi_hat.T.R().as_matrix()  # Body frame to NED
            filt.mag_meas_0 = (R_body_to_ned @ 
            mag).reshape(-1, 1)
            filt._mag_ref_initialized = True

        # Magnetometer update (with hard-iron bias correction applied above)
        if prev_mag is None or not np.allclose(prev_mag, mag):
            filt.magnetometer_update(mag, t=t)
            prev_mag = mag

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
                if GNSS_POS_ONLY:
                    filt.GNSS_update_pos_only(pos_NED,t)
                else:
                    filt.GNSS_update(pos_NED, vel_NED, t)
                gnss_count += 1

            prev_lat, prev_lon = lat, lon

        out.append(filt.output_row(t))

        # Progress reporting with covariance diagnostics
        if t - last_progress_t >= 30:  # Every 30 seconds
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
        "roll,pitch,yaw,"
        "bgx,bgy,bgz,bax,bay,baz,bmux,bmuy,bmuz"
    )

    np.savetxt(csv_out, out, delimiter=",", header=header, comments="")
    print(f"Wrote {len(out)} rows to {csv_out}")
    print(f"Magnetometer updates applied: {filt.mag_update_count}")

    # Save diagnostic data (ANIS/ANEES values with timestamps)
    if filt.update_times:
        diag_out = csv_out.replace("tg_eqf_output", "tg_eqf_diagnostics")
        with open(diag_out, 'w') as f:
            f.write("time,update_type,anis,anees\n")
            anis_dict = {}
            anees_dict = {}
            # Organize by (update_type, time) to combine anis and anees
            for update_type, t, value in filt.update_times:
                key = (update_type.split('_')[0], t)  # Use 'mag' or 'gnss' as key
                if 'anees' in update_type:
                    anees_dict[key] = value
                else:
                    anis_dict[key] = value

            # Merge and write
            all_times = {}
            for key, val in anis_dict.items():
                if key not in all_times:
                    all_times[key] = {'anis': None, 'anees': None}
                all_times[key]['anis'] = val
            for key, val in anees_dict.items():
                if key not in all_times:
                    all_times[key] = {'anis': None, 'anees': None}
                all_times[key]['anees'] = val

            for (update_type, t), values in sorted(all_times.items()):
                anis_str = f"{values['anis']:.4f}" if values['anis'] is not None else ""
                anees_str = f"{values['anees']:.4f}" if values['anees'] is not None else ""
                f.write(f"{t:.4f},{update_type},{anis_str},{anees_str}\n")
        print(f"Wrote diagnostic data to {diag_out}")

    # Print filter diagnostics
    print(f"\n=== Filter Diagnostics ===")
    if filt.anis_values:
        anis_array = np.array(filt.anis_values)
        print(f"ANIS (avg normalized innovation squared):")
        print(f"  Mean: {np.mean(anis_array):.2f} (should be ~1)")
        print(f"  Std:  {np.std(anis_array):.2f}")
        print(f"  Min:  {np.min(anis_array):.2f}, Max: {np.max(anis_array):.2f}")

    if filt.anees_values:
        anees_array = np.array(filt.anees_values)
        print(f"ANEES (avg normalized estimation error squared):")
        print(f"  Mean: {np.mean(anees_array):.2f} (should be ~1 for measurement dimension)")
        print(f"  Std:  {np.std(anees_array):.2f}")
        print(f"  Min:  {np.min(anees_array):.2f}, Max: {np.max(anees_array):.2f}")


if __name__ == "__main__":
    if USE_STATIC_DATA:
        print("=" * 60)
        print("Running on STATIC DATA (30s initial conditions)")
        print("=" * 60)
        run()  # Uses static data file
    else:
        print("=" * 60)
        print("Running on REAL FLIGHT DATA")
        print("=" * 60)
        run()  # Uses real flight data