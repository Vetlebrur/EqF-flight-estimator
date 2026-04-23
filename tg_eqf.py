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
        self.P = np.eye(18) * 0.1          # Covariance (18×18)
        self.time = 0.0                    # Current time
        self.Delta = np.zeros((18,))       # 18x1
        Z3 = np.zeros((3,3))               # 3x3 matrix of zeroes
        Z9 = np.zeros((9,9))               # 9x9 matrix of zeroes
        I3 = np.eye(3)                     # 3x3 Identity
        I9 = np.eye(9)                     # 9x9 Identity
        Q_mag = config.Q_0_magnetometer
        Q_baro = config.Q_0_barometer
        Q_gnss = config.Q_0_gnss

    # =========================================================================
    # Algebra Operations (se(2,3))
    # =========================================================================

    @staticmethod
    def wedge(v: np.ndarray) -> np.ndarray:
        """Convert 9-vector to se(2,3) matrix form (5×5)."""
        return SE23.wedge(v)

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """Convert se(2,3) matrix form (5×5) to 9-vector."""
        return SE23.vee(mat)

    @staticmethod
    def exp_map(v: np.ndarray) -> SE23:
        """Exponential map: se(2,3) → SE(2,3)."""
        return SE23.exp(v)

    def log_map(self, T: Optional[SE23] = None) -> np.ndarray:
        """Logarithmic map: SE(2,3) → se(2,3)."""
        if T is None:
            T = self.T
        return T.log()

    # =========================================================================
    # Group Operations
    # =========================================================================

    def compose(self, other: 'TGEqF') -> 'TGEqF':
        """
        Semi-direct product: (T2, b2) * (T1, b1) = (T2*T1, b2 + Ad_{T2^-1}[b1]).

        Args:
            other: The state to compose with (right operand).

        Returns:
            New composed state.
        """
        result = TGEqF()

        # SE(2,3) multiplication
        result.T = self.T * other.T

        # Bias transformation: b2 + Ad_{T2^-1}[b1]
        T2_inv = self.T.inv()
        Ad_T2_inv = T2_inv.Adjoint()
        result.b = self.b + Ad_T2_inv @ other.b

        return result

    def inverse(self) -> 'TGEqF':
        """
        Inverse: (T, b)^-1 = (T^-1, -Ad_{T^-1}[b]).

        Returns:
            Inverse state.
        """
        result = TGEqF()
        result.T = self.T.inv()
        Ad_T_inv = result.T.Adjoint()
        result.b = -Ad_T_inv @ self.b
        return result

    # =========================================================================
    # Pose Access
    # =========================================================================

    def get_rotation(self) -> np.ndarray:
        """Get rotation matrix (3×3) from pose."""
        return self.T.R().as_matrix()

    def get_position(self) -> np.ndarray:
        """Get position vector (3,) from pose."""
        return self.T.x().as_vector()

    def get_velocity(self) -> np.ndarray:
        """Get velocity vector (3,) from pose."""
        return self.T.w().as_vector()

    def get_bias_gyro(self) -> np.ndarray:
        """Get gyroscope bias (3,)."""
        return self.b[0:3]

    def get_bias_velocity(self) -> np.ndarray:
        """Get velocity bias (3,)."""
        return self.b[3:6]

    def get_bias_accel(self) -> np.ndarray:
        """Get accelerometer bias (3,)."""
        return self.b[6:9]

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
