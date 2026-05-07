"""
Prediction step for TGEqF: IMU-based state propagation.

Implements prediction methods: imu_predict(), predict_bias_zero_order(), predict_bias_random_walk(),
calculate_A(), calculate_lift()
"""

import numpy as np
from typing import Optional, Tuple
from pylie import SE23, SO3
import scipy
from old.tg_eqf import TGEqF
from old.SE23xxse23 import SE23xxse23

def calculate_A(
    self: TGEqF,
    gyro: np.ndarray,
    accel: np.ndarray,
    mu: np.ndarray = None,
) -> np.ndarray:
    """Match reference SE23_se23_EqF stateMatrixA_CT exactly"""
    if mu is None:
        mu = self.b[6:9]

    T_inv_mat = self.T.inv().as_matrix()
    T_mat = self.T.as_matrix()

    # U in matrix form: [skew(gyro), accel, mu; 0, I, 0; 0, 0, 1]
    U_mat = np.zeros((5, 5))
    U_mat[0:3, 0:3] = SO3.wedge(gyro.reshape(3, 1))
    U_mat[0:3, 3:4] = accel.reshape(3, 1)
    U_mat[0:3, 4:5] = mu.reshape(3, 1)

    # Beta in matrix form (bias)
    beta_mat = SE23.wedge(self.b)

    # f_10: extract velocity from T_inv and put in position slot
    vel_from_inv = T_inv_mat[0:3, 3:4]
    f_10_mat = np.zeros((5, 5))
    f_10_mat[0:3, 4:5] = vel_from_inv

    # velocityAction result: T_inv @ (U - beta) @ T + f_10(T_inv)
    transformed_measurement_mat = T_inv_mat @ (U_mat - beta_mat) @ T_mat + T_inv_mat @ f_10_mat
    u_0_vec = SE23.vee(transformed_measurement_mat)

    # Build A matrix following reference exactly
    A = np.zeros((18, 18))

    # Upper-left block: reference uses specific structure
    # A0t[0:9, 0:9] = np.hstack((blockDiag(np.vstack((np.zeros((3, 3)), SO3.skew(np.vstack((0.0, 0.0, -9.81))))), np.eye(3)), np.zeros((9, 3))))
    # Decode: blockDiag([zeros(3,3); skew(-g)], I3) then hstack with zeros(9,3)

    # Build the block diagonal part
    gravity_skew = SO3.skew(np.array([0.0, 0.0, -9.81]))  # skew of -g
    top_part = np.vstack((np.zeros((3, 3)), gravity_skew))  # (6, 3)
    # blockDiag creates [[top_part, 0], [0, I3]] = (9, 6)
    block1 = np.vstack((np.hstack((top_part, np.zeros((6, 3)))),
                        np.hstack((np.zeros((3, 3)), np.eye(3)))))
    # hstack with zeros(9, 3) to make (9, 9)
    A_upper_left = np.hstack((block1, np.zeros((9, 3))))

    A[0:9, 0:9] = A_upper_left

    # Upper-right block: I9
    A[0:9, 9:18] = np.eye(9)

    # Lower-right block: adjoint(u_0 + G)
    G_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.81, 0.0, 0.0, 0.0])
    u_0_plus_G = u_0_vec + G_vec
    A[9:18, 9:18] = SE23.adjoint(u_0_plus_G)

    return A


def calculate_lift(
    self: TGEqF,
    gyro: np.ndarray,
    accel: np.ndarray,
    mu: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Exactly match reference implementation: continuous_lift from SE23_se23_EqF.py
    # L[0:9] = (U.as_W_vec() - xi.b) + SE23.vee(xi.T.inv() @ (G + f_10(xi.T)))
    # L[9:18] = SE23.adjoint(xi.b) @ L[0:9] - tau

    if mu is None:
        mu = self.b[6:9]

    # Measurement vector: U.as_W_vec() = [vee(SO3.wedge(gyro)), accel, mu]
    # Since gyro is already a vector, vee(wedge(gyro)) = gyro
    U_vec = np.concatenate([gyro, accel, mu])

    # First part: U - b
    measurement_minus_bias = U_vec - self.b

    # Second part: vee(T_inv @ (G + f_10(T)))
    # NOTE: continuous_lift formula uses f_10(T), not f_10(T_inv)!
    T_mat = self.T.as_matrix()
    T_inv_mat = self.T.inv().as_matrix()

    # Construct G matrix: gravity in z-velocity component
    G_mat = np.zeros((5, 5))
    G_mat[2, 3:4] = -9.81

    # Construct f_10 matrix: extract velocity from T (current pose) and put in position slot
    vel_extracted = T_mat[0:3, 3:4]
    f_10_mat = np.zeros((5, 5))
    f_10_mat[0:3, 4:5] = vel_extracted

    # Combine: G + f_10 (exactly matching reference formula)
    G_plus_f10 = G_mat + f_10_mat

    # Conjugate by T_inv: T_inv @ (G + f_10)
    conjugated = T_inv_mat @ G_plus_f10

    # Convert to vector form
    coupling_term = SE23.vee(conjugated)

    # Combine: (U - b) + coupling_term
    lift_upper = measurement_minus_bias + coupling_term

    # Lower part: SE23.adjoint(b) @ L[0:9] - tau (tau=0 for IMU-only)
    lift_lower = SE23.adjoint(self.b) @ lift_upper

    return lift_upper, lift_lower

def imu_predict(
    self: TGEqF,
    dt: float,
    accel: np.ndarray,
    gyro: np.ndarray,
    accel_cov: Optional[np.ndarray] = None,
    gyro_cov: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None
) -> None:
    """
    IMU prediction step: propagate state using accelerometer and gyroscope.

    Implements: dT/dt = T * (...), db/dt = ...

    Args:
        dt: Time step (seconds).
        accel: Accelerometer measurement (3,).
        gyro: Gyroscope measurement (3,).
        accel_cov: Accelerometer covariance (3×3), optional.
        gyro_cov: Gyroscope covariance (3×3), optional.
        mu: Structural input (3,). Defaults to b_mu.
    """
    # Compute A matrix for covariance propagation
    A = self.calculate_A(gyro, accel)
    A_norm = np.linalg.norm(A)
    F = scipy.linalg.expm(A*dt)
    F_norm = np.linalg.norm(F)

    # Compute lift with mu term
    lift_upper, lift_lower = self.calculate_lift(gyro, accel, mu)
    # Add bias input (tau) - typically zero for IMU-only prediction
    Lift = np.concatenate([lift_upper, lift_lower]) * dt

    # Propagate state via group exponential
    X_hat = SE23xxse23(self.T, self.b)
    X_hat = X_hat * SE23xxse23.exp(Lift)

    self.T = X_hat.T
    self.b = X_hat.b

    # Update time
    self.time += dt

    # Propagate covariance (no process noise - IMU model assumed perfect)
    sigma_before = np.linalg.norm(self.Sigma)
    self.Sigma = F @ self.Sigma @ F.T
    self.P = self.Sigma.copy()  # Keep P in sync
    sigma_after = np.linalg.norm(self.Sigma)

    if sigma_after > 1e6 or np.isnan(sigma_after):
        print(f"  [t={self.time:.4f}] COVARIANCE SPIKE in imu_predict!")
        print(f"    A norm: {A_norm:.4e}, F norm: {F_norm:.4e}")
        print(f"    Sigma: {sigma_before:.4e} -> {sigma_after:.4e}")
        print(f"    dt: {dt:.4e}, Lift norm: {np.linalg.norm(Lift):.4e}")


def predict_bias_zero_order(self: TGEqF) -> None:
    """Assume bias is constant (zero-order hold)."""
    # Bias stays the same: db/dt = 0
    self.P[9:18, 9:18] = np.diag([self.Z9,self.Z9])


def predict_bias_random_walk(
    self: TGEqF,
    dt: float,
    Q_bias: Optional[np.ndarray] = None
) -> None:
    """
    Predict bias with random walk model: db/dt = w, E[ww^T] = Q_bias.

    Args:
        dt: Time step.
        Q_bias: Process noise covariance for bias (9×9).
    """
    # For now, assume constant bias
    if Q_bias is not None:
        # Add process noise to covariance
        self.P[9:18, 9:18] += Q_bias * dt


# Attach methods to TGEqF
TGEqF.calculate_A = calculate_A
TGEqF.calculate_lift = calculate_lift
TGEqF.imu_predict = imu_predict
TGEqF.predict_bias_zero_order = predict_bias_zero_order
TGEqF.predict_bias_random_walk = predict_bias_random_walk

__all__ = ['TGEqF']


