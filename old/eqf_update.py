"""
Update step for TGEqF: Measurement-based state correction.

Implements update methods: magnetometer_update(), gnss_update(), barometer_update()
"""

import numpy as np
from typing import Optional
from scipy.linalg import expm
from old.tg_eqf import TGEqF
from pylie import SO3, SE23
from old.SE23xxse23 import SE23xxse23
import old.eqf_config as config

# def calculate_gnss_C_old(
#     self: TGEqF,
#     R_hat: np.ndarray,
#     p_hat: np.ndarray,
#     ) -> np.ndarray:
#     return np.block([self.Z3,self.Z3,self.Z3,-R_hat.T,self.Z3,R_hat.T @ SO3.wedge(p_hat)])

def calculate_gnss_C(
    self: TGEqF,
    p_gnss: np.ndarray,
    v_gnss: np.ndarray,
) -> np.ndarray:
    """
    Calculate GNSS measurement Jacobian C for position and velocity.

    Measurement model: z = [p_n, p_e, p_d, v_n, v_e, v_d]^T
    GNSS directly measures position and velocity in NED frame.
    No bias terms: GNSS is absolute measurement, independent of IMU biases.

    Args:
        p_gnss: GNSS position (3,) in NED [pn, pe, pd] (for consistency, not used)
        v_gnss: GNSS velocity (3,) in NED [vn, ve, vd] (for consistency, not used)

    Returns:
        C matrix (6×18) relating GNSS measurement to SE23xxse23 state
        State format: [R(3), p(3), v(3), b_gyro(3), b_vel(3), b_accel(3)]
    """
    C = np.zeros((6, 18))

    # Position measurement: extracts p from state (columns 3-5 of pose part)
    C[0:3, 3:6] = np.eye(3)

    # Velocity measurement: extracts v from state (columns 6-8 of pose part)
    C[3:6, 6:9] = np.eye(3)

    return C

def calculate_magnetometer_C(
    self: TGEqF
    ) -> np.ndarray:
    return np.block([-SO3.wedge(self.T.R()@config.m_0),self.Z3,self.Z3,self.Z3,self.Z3,self.Z3])

def calculate_barometer_C ( #TODO: ignore
    self: TGEqF
    ) -> np.ndarray:
    return  np.block([self.Z3,self.Z3,self.Z3+np.diag([0, 0, -1]),self.Z3,self.Z3,self.Z3])

def magnetometer_update(
    self: TGEqF,
    mag: np.ndarray,
    mag_reference: np.ndarray = np.array([1, 0, 0]),
    R_mag: Optional[np.ndarray] = None,
) -> None:
    C_mag = calculate_magnetometer_C(self)
    delta = mag_reference - self.T.R() @ mag

    Q_mag = R_mag if R_mag is not None else config.Q_0_magnetometer
    S = C_mag @ self.Sigma @ C_mag.T + Q_mag
    K = self.Sigma @ C_mag.T @ np.linalg.inv(S)
    Delta = K @ delta

    X_hat = SE23xxse23(self.T, self.b)
    X_hat = SE23xxse23.exp(Delta) * X_hat
    self.T = X_hat.T
    self.b = X_hat.b

    self.Sigma = (np.eye(18) - K @ C_mag) @ self.Sigma


def gnss_update(
    self: TGEqF,
    pos: np.ndarray,
    vel: np.ndarray,
    R_gnss: Optional[np.ndarray] = None,
) -> None:
    C_gnss = calculate_gnss_C(self, pos, vel)
    y_meas = np.concatenate([pos, vel])
    y_est = np.concatenate([self.T.x().as_vector(), self.T.w().as_vector()])
    delta = y_meas - y_est

    Q_gnss = R_gnss if R_gnss is not None else config.Q_0_gnss
    S = C_gnss @ self.Sigma @ C_gnss.T + Q_gnss
    K = self.Sigma @ C_gnss.T @ np.linalg.inv(S)
    Delta = K @ delta

    X_hat = SE23xxse23(self.T, self.b)
    X_hat = SE23xxse23.exp(Delta) * X_hat
    self.T = X_hat.T
    self.b = X_hat.b

    self.Sigma = (np.eye(18) - K @ C_gnss) @ self.Sigma


def barometer_update(
    self: TGEqF,
    pressure: float,
    R_baro: Optional[np.ndarray] = None,
) -> None:
    """
    Update state using barometer (altimeter) measurement.

    Measurement model: y_baro = altitude (vertical position)

    Args:
        pressure: Barometric pressure (Pa).
        R_baro: Measurement noise variance (scalar), optional.
    """
    # TODO: Implement barometer to altitude conversion and update
    pass


# Attach methods to TGEqF
TGEqF.magnetometer_update = magnetometer_update
TGEqF.gnss_update = gnss_update
TGEqF.barometer_update = barometer_update

__all__ = ['TGEqF']
