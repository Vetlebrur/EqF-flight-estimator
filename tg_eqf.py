"""
Tangent-Group Equivariant Filter (TGEqF)

Base class for the EqF implementation. Method implementations are in separate files:
  - eqf_predict.py: imu_predict(), predict_bias_zero_order(), predict_bias_random_walk()
  - eqf_update.py: magnetometer_update(), gnss_update(), barometer_update()
"""

from pylie import SE23, SO3
import numpy as np
from typing import Tuple, Optional
import eqf_config as config


class TGEqF:
    """
    Tangent-Group Equivariant Filter on SE(2,3) ⋉ se(2,3).

    State: (T, b) where T ∈ SE(2,3) and b ∈ ℝ^9 is bias in se(2,3) algebra.

    The algebra se(2,3) is represented as 9-dimensional vectors using pylie's
    wedge/vee operators.
    """

    def __init__(self):
        """Initialize filter state."""
        self.T = SE23.identity()           # SE(2,3) pose
        self.b = np.zeros(9)               # se(2,3) bias vector

        # Initialize covariance matching reference implementation
        # sigma_vec = [att_var (3), vel_var (3), pos_var (3), bias_var (9)]
        initial_att_noise = 1.0
        initial_vel_noise = 1.0
        initial_pos_noise = 1.0
        initial_bias_noise = 0.01

        sigma_vec = np.concatenate([
            np.ones(3) * initial_att_noise**2,      # Attitude: 1.0
            np.ones(3) * initial_vel_noise**2,      # Velocity: 1.0
            np.ones(3) * initial_pos_noise**2,      # Position: 1.0
            np.ones(9) * initial_bias_noise**2      # Bias: 0.0001
        ])

        self.P = np.diag(sigma_vec)        # Covariance (18×18)
        self.Sigma = self.P.copy()         # Covariance matrix (same as P)
        self.mu = np.zeros(18)             # Information vector
        self.time = 0.0                    # Current time
        self.Delta = np.zeros((18,))       # 18x1
        self.Z3 = np.zeros((3,3))          # 3x3 matrix of zeroes
        self.Z9 = np.zeros((9,9))          # 9x9 matrix of zeroes
        self.I3 = np.eye(3)                # 3x3 Identity
        self.I9 = np.eye(9)                # 9x9 Identity
        self.Q_mag = config.Q_0_magnetometer
        self.Q_baro = config.Q_0_barometer
        self.Q_gnss = config.Q_0_gnss

    # =========================================================================
    # Algebra Operations (se(2,3))
    # =========================================================================


    # =========================================================================
    # Prediction: IMU-based state propagation
    # Implemented in eqf_predict.py
    # =========================================================================

    def imu_predict(
        self,
        dt: float,
        accel: np.ndarray,
        gyro: np.ndarray,
        accel_cov: Optional[np.ndarray] = None,
        gyro_cov: Optional[np.ndarray] = None,
    ) -> None:
        """
        IMU prediction step: propagate state using accelerometer and gyroscope.

        Implemented in eqf_predict.py
        """
        raise NotImplementedError("See eqf_predict.py")

    def predict_bias_zero_order(self) -> None:
        """
        Assume bias is constant (zero-order hold).

        Implemented in eqf_predict.py
        """
        raise NotImplementedError("See eqf_predict.py")

    def predict_bias_random_walk(self, dt: float, Q_bias: Optional[np.ndarray] = None) -> None:
        """
        Predict bias with random walk model.

        Implemented in eqf_predict.py
        """
        raise NotImplementedError("See eqf_predict.py")

    # =========================================================================
    # Update: Measurement-based state correction
    # Implemented in eqf_update.py
    # =========================================================================

    def magnetometer_update(
        self,
        mag: np.ndarray,
        mag_reference: np.ndarray = np.array([1, 0, 0]),
        R_mag: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update state using magnetometer measurement.

        Implemented in eqf_update.py
        """
        raise NotImplementedError("See eqf_update.py")

    def gnss_update(
        self,
        lat: float,
        lon: float,
        alt: Optional[float] = None,
        R_gnss: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update state using GNSS (GPS) measurement.

        Implemented in eqf_update.py
        """
        raise NotImplementedError("See eqf_update.py")

    def barometer_update(
        self,
        pressure: float,
        R_baro: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update state using barometer (altimeter) measurement.

        Implemented in eqf_update.py
        """
        raise NotImplementedError("See eqf_update.py")
