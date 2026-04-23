"""
Prediction step for TGEqF: IMU-based state propagation.

Implements prediction methods: imu_predict(), predict_bias_zero_order(), predict_bias_random_walk(),
calculate_A(), calculate_lift()
"""

import numpy as np
from typing import Optional, Tuple
from pylie import SE23, SO3
from tg_eqf import TGEqF
from SE23xxse23 import SE23xxse23

def calculate_A(
    self: TGEqF,
    gyro: np.ndarray,
    accel: np.ndarray,
    g: np.ndarray = np.array([0, 0, 9.81]),
) -> np.ndarray:
    """
    Calculate the A matrix (Jacobian of dynamics) for linearized EqF prediction.

    Computes the 18×18 state transition matrix for the tangent-space dynamics.

    Args:
        gyro: Gyroscope measurement (3,).
        accel: Accelerometer measurement (3,).
        g: Gravity vector in world frame (3,), default [0, 0, 9.81].

    Returns:
        A: 18×18 Jacobian matrix.
    """
    # Extract bias in se(2,3) algebra
    beta_hat = SE23.vee(self.T.inv().Adjoint() @ self.b)

    #2A matrix from the paper
    A_2 = np.block([self.Z3,self.Z3,self.Z3],
                   [SO3.wedge(g),self.Z3,self.Z3],
                   [self.Z3,self.I3,self.Z3])

    # Construct measurement vector
    w = np.zeros(9)
    w[0:3] = gyro
    w[3:6] = accel
    w[6:9] = beta_hat[6:9]

    # Velocity component in se(2,3)
    velterm = np.vstack([np.zeros(6),self.b[3:6]])
    #f10[3:6] = self.get_velocity()  # Velocity in position-rate slot
    
    # w0 = phi(X_hat^{-1}, xi)
    w0 = self.T.Adjoint() @ w + SE23.vee(self.b) + SE23.vee(velterm)
    g_9x9 = np.vstack([np.zeros(3),g,np.zeros(3)])
    A = np.block([A_2,self.I9],
                 [self.Z9,SE23.adjoint(w0+g_9x9)]
                 )
    return A


def calculate_lift(
    self: TGEqF,
    gyro: np.ndarray, #3x1
    accel: np.ndarray, #3x1
    g: np.ndarray = np.array([0, 0, 9.81]), #3x1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the lift (equivariant frame dynamics) for EqF update laws.

    Returns the lifting matrices that project measurements into the equivariant frame.

    Args:
        gyro: Gyroscope measurement (3,).
        accel: Accelerometer measurement (3,).
        g: Gravity vector in world frame (3,), default [0, 0, 9.81].

    Returns:
        Tuple of (lift_pose, lift_bias) as 9-vectors in se(2,3).
    """
    N = np.zeros([5,5])
    N[3,3] = 1
    B = self.wedge(self.b)
    G = self.wedge(np.vstack([np.zeros(3),g,np.zeros(3)])) 
    W = self.wedge(np.vstack([gyro,accel,np.zeros(3)]))
    lift_pose_vec = (W-B+N)+self.T.inv() @ (G-N) @ self.T

    lift_bias_vec = self.b.adjoint(lift_pose_vec) #Tau = 0

    return lift_pose_vec, lift_bias_vec

def imu_predict(
    self: TGEqF,
    dt: float,
    accel: np.ndarray,
    gyro: np.ndarray,
    accel_cov: Optional[np.ndarray] = None,
    gyro_cov: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = [0,0,9.81]
) -> None:
    """
    IMU prediction step: propagate state using accelerometer and gyroscope.

    Implements: dT/dt = T * (ω∧ - b_gyro∧), db/dt = ...

    Args:
        dt: Time step (seconds).
        accel: Accelerometer measurement (3,).
        gyro: Gyroscope measurement (3,).
        accel_cov: Accelerometer covariance (3×3), optional.
        gyro_cov: Gyroscope covariance (3×3), optional.
    """
    # Compute A matrix for covariance propagation, Discretize system
    A = self.calculate_A(gyro, accel)*dt
    Lift = self.calculate_lift(gyro,accel)*dt

    phi = A.exp()
    b_gyro = self.get_bias_gyro()
    b_accel = self.get_bias_accel()

    X_hat = SE23xxse23(self.T,self.b)

    #TODO: change this to correct version with not this as it is not giving
    X_hat = X_hat @ SE23xxse23.exp(Lift)

    self.T = X_hat.T
    self.b = X_hat.T

    # Update time
    self.time += dt

    self.Sigma = A @self.Sigma@ A.T + self.P


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
 

