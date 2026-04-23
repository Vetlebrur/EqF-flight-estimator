# Example: Core EqF components using PyLie
# This is pseudocode/reference for the actual implementation

from pylie import SE23, se23
import numpy as np
from typing import Tuple


# =============================================================================
# Phase 1: State Representation with PyLie
# =============================================================================

class FilterStateEqFPyLie:
    """
    State on SE(2,3) ⋉ se(2,3) using pylie for SE(2,3) component.

    State: (T, b) where:
      - T ∈ SE(2,3): extended pose (from pylie)
      - b ∈ ℝ^9: bias vector [gyro_bias, velocity_bias, accel_bias]
      - Σ ∈ ℝ^{18×18}: covariance matrix
    """

    def __init__(self):
        self.T = SE23.identity()              # pylie SE(2,3)
        self.b = np.zeros(9)                   # 9-vec bias
        self.Sigma = np.eye(18) * 0.1          # Covariance

    @staticmethod
    def identity():
        """Identity element of SE(2,3) ⋉ se(2,3)"""
        state = FilterStateEqFPyLie()
        state.T = SE23.identity()
        state.b = np.zeros(9)
        return state

    def compose_semidirect(self, other: 'FilterStateEqFPyLie') -> 'FilterStateEqFPyLie':
        """
        Semi-direct product composition: (T2, b2) * (T1, b1)

        Formula:
          T_result = T2 * T1                  (group multiplication)
          b_result = b2 + Ad_{T2^{-1}}[b1]   (bias with adjoint coupling)

        Args:
            other: The state to compose (right operand)

        Returns:
            New composed state
        """
        result = FilterStateEqFPyLie()

        # Group part: T2 * T1 using pylie
        result.T = self.T * other.T

        # Bias part: b2 + Ad_{T2^{-1}}[b1]
        T2_inv = self.T.inverse()

        # Get adjoint matrix (9×9 for SE(2,3))
        Ad_T2_inv_matrix = T2_inv.Ad()  # pylie returns 9×9

        # Apply adjoint to other's bias
        b1_adjoint = Ad_T2_inv_matrix @ other.b

        # Add to self's bias
        result.b = self.b + b1_adjoint

        return result

    def inverse(self) -> 'FilterStateEqFPyLie':
        """
        Inverse: (T, b)^{-1} = (T^{-1}, -Ad_{T^{-1}}[b])
        """
        result = FilterStateEqFPyLie()

        # T^{-1} using pylie
        result.T = self.T.inverse()

        # Bias: -Ad_{T^{-1}}[b]
        Ad_T_inv = result.T.Ad()
        result.b = -Ad_T_inv @ self.b

        return result

    def adjoint_action_9d(self, v: np.ndarray) -> np.ndarray:
        """
        Apply adjoint action of (T, b) on a 9-vector.

        For semi-direct product:
          Ad_{(T,b)}[v] = Ad_T[v]

        The bias b itself affects the transformation via the Lie bracket,
        but for the linear action on vectors, we use Ad_T.
        """
        Ad_T = self.T.Ad()  # 9×9 matrix from pylie
        return Ad_T @ v

    def extract_pose_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract rotation, position, velocity from T"""
        R = self.T.R().as_matrix()        # 3×3 SO(3)
        p = np.array(self.T.p()).flatten()         # 3-vec
        v = np.array(self.T.v()).flatten()         # 3-vec
        return R, p, v

    def extract_bias_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract bias components"""
        b_gyro = self.b[0:3]
        b_velocity = self.b[3:6]
        b_accel = self.b[6:9]
        return b_gyro, b_velocity, b_accel


# =============================================================================
# Phase 2: Equivariant Lift using PyLie
# =============================================================================

def compute_lift_Lambda1(
    state: FilterStateEqFPyLie,
    gyro: np.ndarray,
    accel: np.ndarray,
    velocity_bias: np.ndarray,
    gravity_world: np.ndarray = np.array([0, 0, 9.81])
) -> np.ndarray:
    """
    Compute Λ_1 (Equation 12 from paper).

    Λ_1 = (w∧ - b∧) + Ad_{T^{-1}}[g∧] + T^{-1} f_1^0(T)

    Where:
      - w = [gyro, velocity_bias, accel]  (9-vec: combined input)
      - b = state.b                        (9-vec: bias)
      - g = gravity (3-vec)
      - f_1^0 is the right-invariant drift field

    Args:
        state: Current filter state
        gyro: IMU angular velocity (3-vec)
        accel: IMU linear acceleration (3-vec)
        velocity_bias: Virtual velocity input (3-vec)
        gravity_world: Gravity in world frame (3-vec)

    Returns:
        Lambda_1: 9-vector in se(2,3) space
    """

    # Construct input vector
    w = np.concatenate([gyro, velocity_bias, accel])  # 9-vec

    # Term 1: (w∧ - b∧)
    term1 = w - state.b

    # Term 2: Ad_{T^{-1}}[g∧]
    # We need gravity in se(2,3) form. Gravity affects velocity dynamics.
    # In the algebra, gravity is represented in the velocity component.
    # Create a pseudo-algebra element for gravity
    g_algebra_vec = np.concatenate([np.zeros(3), np.zeros(3), gravity_world])  # ⟂0,0,g⟩

    # Get inverse and its adjoint
    T_inv = state.T.inverse()
    Ad_T_inv = T_inv.Ad()

    # Ad_{T^{-1}}[g]
    term2 = Ad_T_inv @ g_algebra_vec

    # Term 3: T^{-1} f_1^0(T)
    # f_1^0(T) is the right-invariant drift field: [ 0  v  0 ]
    # This is a matrix form, we need to convert to vector
    term3 = compute_drift_term_pylie(state.T)

    # Λ_1 = term1 + term2 + term3
    Lambda_1 = term1 + term2 + term3

    return Lambda_1


def compute_drift_term_pylie(T: SE23) -> np.ndarray:
    """
    Compute T^{-1} f_1^0(T) using pylie.

    f_1^0(T) = [ 0   v   0 ]  where v is velocity from T
               [ 0   0   0 ]

    This represents the right-invariant drift field.
    """

    # Extract velocity from T
    v = np.array(T.v()).flatten()  # 3-vec

    # Create drift matrix (4×4)
    drift_matrix = np.zeros((4, 4))
    drift_matrix[0:3, 1:4] = np.hstack([v[:, np.newaxis], np.zeros((3, 3))])
    # Actually: drift_matrix[0:3, 1] is position column, drift_matrix[0:3, 2] is velocity column
    # For SE(2,3): f_1^0 = [[0, p_dot, v_dot], [0, 0, 0], [0, 0, 0]]
    # In the matrix form of se(2,3) element

    # Convert to se(2,3) algebra element and extract vector
    # This is tricky with pylie - convert to vector form

    drift_vector = np.concatenate([
        np.zeros(3),      # No rotation part
        v,                # Velocity as "acceleration" in the lift
        np.zeros(3)       # Bias component unaffected
    ])

    # Apply T^{-1}: this is right-multiplying by T^{-1} in Lie group context
    T_inv = T.inverse()
    Ad_T_inv = T_inv.Ad()
    drift_transformed = Ad_T_inv @ drift_vector

    return drift_transformed


def compute_lift_Lambda2(
    state: FilterStateEqFPyLie,
    gyro: np.ndarray,
    accel: np.ndarray,
    velocity_bias: np.ndarray,
    tau_gyro: np.ndarray = None,
    tau_velocity: np.ndarray = None,
    tau_accel: np.ndarray = None,
    gravity_world: np.ndarray = np.array([0, 0, 9.81])
) -> np.ndarray:
    """
    Compute Λ_2 (Equation 13 from paper).

    Λ_2 = ad_{b∧}[Λ_1] - τ∧

    Where:
      - ad_{b∧}[·] is the Lie bracket (commutator)
      - τ = [τ_gyro, τ_velocity, τ_accel] (virtual inputs for bias)

    Args:
        state: Current filter state
        gyro: IMU angular velocity
        accel: IMU acceleration
        velocity_bias: Virtual velocity input
        tau_*: Virtual inputs for bias dynamics
        gravity_world: Gravity vector

    Returns:
        Lambda_2: 9-vector
    """

    # Construct virtual bias inputs
    if tau_gyro is None:
        tau_gyro = np.zeros(3)
    if tau_velocity is None:
        tau_velocity = np.zeros(3)
    if tau_accel is None:
        tau_accel = np.zeros(3)

    tau = np.concatenate([tau_gyro, tau_velocity, tau_accel])  # 9-vec

    # Compute Λ_1 (needed for the bracket)
    Lambda_1 = compute_lift_Lambda1(
        state, gyro, accel, velocity_bias, gravity_world
    )

    # Term 1: ad_{b∧}[Λ_1] = [b, Λ_1] (Lie bracket)
    bracket = lie_bracket_9d(state.b, Lambda_1)

    # Term 2: τ∧
    # (τ is already in vector form)

    # Λ_2 = bracket - τ
    Lambda_2 = bracket - tau

    return Lambda_2


def lie_bracket_9d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute Lie bracket [u, v] in se(2,3) space.

    The Lie bracket is [u, v] = uv - vu where multiplication is matrix product.

    We need to convert 9-vectors to matrices, compute bracket, convert back.
    """

    # Convert vectors to matrix form (this depends on how we represent se(2,3))
    # Typical representation:
    # u = [ω_x, ω_y, ω_z, p_x, p_y, p_z, v_x, v_y, v_z]
    #
    # Matrix form (4×4):
    # [ω∧  p  v]
    # [0   0  0]  where ω∧ is 3×3 skew-symmetric
    # [0   0  0]

    # Extract components
    omega_u = u[0:3]
    p_u = u[3:6]
    v_u = u[6:9]

    omega_v = v[0:3]
    p_v = v[3:6]
    v_v = v[6:9]

    # Create matrix forms
    omega_u_skew = skew_symmetric(omega_u)
    omega_v_skew = skew_symmetric(omega_v)

    u_matrix = np.zeros((4, 4))
    u_matrix[0:3, 0:3] = omega_u_skew
    u_matrix[0:3, 3] = p_u
    # Need another column for velocity - adjust based on se(2,3) convention

    v_matrix = np.zeros((4, 4))
    v_matrix[0:3, 0:3] = omega_v_skew
    v_matrix[0:3, 3] = p_v

    # Compute bracket [u, v] = u*v - v*u
    bracket_matrix = u_matrix @ v_matrix - v_matrix @ u_matrix

    # Convert back to 9-vector
    bracket_vector = matrix_to_se23_vector(bracket_matrix)

    return bracket_vector


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Convert 3-vector to 3×3 skew-symmetric matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def matrix_to_se23_vector(M: np.ndarray) -> np.ndarray:
    """Convert 4×4 se(2,3) matrix to 9-vector"""
    # Extract components from matrix
    omega_vec = np.array([M[2, 1], M[0, 2], M[1, 0]])  # From skew-symmetric
    p_vec = M[0:3, 3]
    # For velocity, depends on se(2,3) convention
    # Placeholder:
    v_vec = np.zeros(3)

    return np.concatenate([omega_vec, p_vec, v_vec])


# =============================================================================
# Phase 3: Propagation with PyLie
# =============================================================================

def propagate_state_pylie(
    state: FilterStateEqFPyLie,
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: float,
    velocity_bias: np.ndarray = None,
    tau_gyro: np.ndarray = None,
    tau_velocity: np.ndarray = None,
    tau_accel: np.ndarray = None,
    gravity_world: np.ndarray = np.array([0, 0, 9.81])
) -> FilterStateEqFPyLie:
    """
    Propagate state using lifted system dynamics with pylie.

    Ṫ = T(w∧ + Ad_{T^{-1}}[b̂]) + g∧·T + f_1^0(T)
    ḃ = ad_{-b̂}[(Ad_T[w∧] + b̂) + g∧ + f_1^0(T)] - Ad_T[τ∧] + ad_{∆_1}[b̂]

    For propagation without innovation (∆ = 0), the second term simplifies.
    """

    if velocity_bias is None:
        velocity_bias = np.zeros(3)

    # Compute lifts (without innovation terms ∆)
    Lambda_1 = compute_lift_Lambda1(
        state, gyro, accel, velocity_bias, gravity_world
    )
    Lambda_2 = compute_lift_Lambda2(
        state, gyro, accel, velocity_bias,
        tau_gyro, tau_velocity, tau_accel, gravity_world
    )

    # Create new state
    new_state = FilterStateEqFPyLie()

    # Integrate T using pylie exponential
    # T_new = T_old * exp(Λ_1 * dt)

    # Convert Lambda_1 to se(2,3) algebra element
    # (This requires mapping from 9-vec to 6-vec se(2,3) representation)
    Lambda_1_algebra = se23_vector_to_algebra(Lambda_1)

    # Exponential map using pylie
    exp_term = SE23.exp(Lambda_1_algebra * dt)

    # Update T using pylie group multiplication
    new_state.T = state.T * exp_term

    # Integrate b
    # Simple Euler: b_new = b_old + Λ_2 * dt
    new_state.b = state.b + Lambda_2 * dt

    # Copy covariance (updated separately)
    new_state.Sigma = state.Sigma.copy()

    return new_state


def se23_vector_to_algebra(v: np.ndarray) -> se23:
    """
    Convert 9-vector to se(2,3) algebra element.

    This is a challenge because pylie uses 6-vec for se(2,3),
    but we use 9-vec (including bias).

    We need to extract the relevant 6 components for SE(2,3) propagation.
    """
    # Take first 6 components (rotation + velocity dynamics)
    v_se23 = v[0:6]

    # Use pylie's from_vector (or from_matrix after converting)
    # This is pseudo-code; actual implementation depends on pylie's API
    algebra = se23.from_vector(v_se23)

    return algebra


# =============================================================================
# Phase 4: Innovation and Update with PyLie
# =============================================================================

def compute_innovation_pylie(
    state: FilterStateEqFPyLie,
    measurement_pose: np.ndarray,  # 4×4 measured SE(2,3) matrix
    output_noise_Q: np.ndarray,     # 9×9 or 6×6
    C0: np.ndarray                  # Output matrix (9×18 or 6×18)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute innovation using pylie.

    Output residual: δ = ρ(X̂^{-1}, y) = y * T̂^{-1}

    In Lie algebra: log(y * T̂^{-1})
    """

    # Convert measurement to pylie
    y = SE23.from_matrix(measurement_pose)

    # Get state inverse
    T_hat_inv = state.T.inverse()

    # Residual on the group: y * T̂^{-1}
    residual_group = y * T_hat_inv

    # Convert to Lie algebra
    residual_algebra = residual_group.log()
    residual_vec = residual_algebra.as_vector()  # 6-vec from pylie

    # Pad to 9-vec (add zeros for unobserved bias part)
    residual_vec_full = np.concatenate([residual_vec, np.zeros(3)])

    # Kalman gain: K = Σ * C^0T * Q^{-1}
    Q_inv = np.linalg.inv(output_noise_Q)
    K = state.Sigma @ C0.T @ Q_inv  # 18×9 or 18×6

    # Correction
    correction = K @ residual_vec_full  # 18-vec

    # Split
    Delta_1 = correction[0:9]
    Delta_2 = correction[9:18]

    return Delta_1, Delta_2


def update_state_pylie(
    state: FilterStateEqFPyLie,
    Delta_1: np.ndarray,  # 9-vec correction for T
    Delta_2: np.ndarray   # 9-vec correction for b
) -> FilterStateEqFPyLie:
    """
    Update state with equivariant correction.

    T_new = T_old * exp(∆_1)
    b_new = b_old + ∆_2 + [∆_1, b_old]  (with Lie bracket coupling)
    """

    new_state = FilterStateEqFPyLie()

    # Update T via exponential using pylie
    Delta_1_algebra = se23_vector_to_algebra(Delta_1)
    exp_correction = SE23.exp(Delta_1_algebra)
    new_state.T = state.T * exp_correction

    # Update bias with coupling term
    bracket_term = lie_bracket_9d(Delta_1, state.b)
    new_state.b = state.b + Delta_2 + bracket_term

    # Covariance updated via Riccati (separate)
    new_state.Sigma = state.Sigma.copy()

    return new_state


# =============================================================================
# Main Filter Class Using PyLie
# =============================================================================

class EquivariantFilterPyLie:
    """Complete EqF implementation using pylie"""

    def __init__(self):
        self.state = FilterStateEqFPyLie()
        self.t_prev = None

        # Tuning
        self.Q_nav = 1e-2
        self.Q_bias = 1e-5
        self.P = self._build_process_noise()
        self.Q = np.eye(9) * (0.01**2)

    def _build_process_noise(self) -> np.ndarray:
        """Build 18×18 process noise matrix"""
        P = np.zeros((18, 18))
        P[0:9, 0:9] = np.eye(9) * self.Q_nav
        P[9:18, 9:18] = np.eye(9) * self.Q_bias
        return P

    def propagate(self, t: float, gyro: np.ndarray, accel: np.ndarray):
        """Propagate without measurement"""
        if self.t_prev is None:
            self.t_prev = t
            return

        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return

        # State propagation using pylie
        self.state = propagate_state_pylie(
            self.state, gyro, accel, dt
        )

        # Covariance propagation (via Riccati)
        # ... (standard Kalman covariance update)

        self.t_prev = t

    def correct(self, position_NED: np.ndarray, R_pos: np.ndarray):
        """Correct with position measurement"""
        # Construct measurement SE(2,3) (requires extracting pose from position)
        # ... (measurement construction)

        # Innovation using pylie
        C0 = np.zeros((9, 18))
        C0[0:9, 0:9] = np.eye(9)

        Delta_1, Delta_2 = compute_innovation_pylie(
            self.state, measurement_pose, self.Q, C0
        )

        # Update state using pylie
        self.state = update_state_pylie(self.state, Delta_1, Delta_2)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Create initial state
    state = FilterStateEqFPyLie()

    # Test composition
    state2 = FilterStateEqFPyLie()
    state3 = state.compose_semidirect(state2)
    print(f"Composition works: T shape {state3.T.as_matrix().shape}, b shape {state3.b.shape}")

    # Test propagation
    gyro = np.array([0.01, 0.02, 0.03])
    accel = np.array([0, 0, 9.81])
    new_state = propagate_state_pylie(state, gyro, accel, dt=0.01)
    print(f"Propagation works: T shape {new_state.T.as_matrix().shape}")

    # Test adjoint
    v = np.random.randn(9)
    v_adj = state.adjoint_action_9d(v)
    print(f"Adjoint works: input shape {v.shape}, output shape {v_adj.shape}")
