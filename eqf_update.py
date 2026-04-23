"""
Update step for TGEqF: Measurement-based state correction.

Implements update methods: magnetometer_update(), gnss_update(), barometer_update()
"""

import numpy as np
from typing import Optional
from tg_eqf import TGEqF
from pylie import SO3, SE23

def calculate_gnss_C(
    self: TGEqF,
    R_hat: np.ndarray,
    p_hat: np.ndarray,
    ) -> np.ndarray:
    return np.block([self.Z3,self.Z3,self.Z3,-R_hat.T,self.Z3,R_hat.T @ SO3.wedge(p_hat)])

def calculate_magnetometer_C(
    self: TGEqF
    ) -> np.ndarray:
    return np.block([self.I3,self.Z3,self.Z3,self.Z3,self.Z3,self.Z3])

def calculate_barometer_C ( #TODO: ignore
    self: TGEqF
    ) -> np.ndarray:
    return  np.block([self.Z3,self.Z3,self.Z3+np.diag([0, 0, -1]),self.Z3,self.Z3,self.Z3])
    pass 

def magnetometer_update(
    self: TGEqF,
    mag: np.ndarray,
    mag_reference: np.ndarray = np.array([1, 0, 0]),
    R_mag: Optional[np.ndarray] = None,
) -> None:
    """
    Update state using magnetometer measurement.

    Measurement model: y_mag = R * mag_reference (heading)

    Args:
        mag: Magnetometer measurement (3,).
        mag_reference: Expected magnetic field direction in world frame (3,).
            Default: North pointing (1, 0, 0).
        R_mag: Measurement noise covariance (3×3), optional.
    """
    # TODO: Implement equivariant magnetometer update

    C_mag = calculate_magnetometer_C()

    self.Sigma = (self.Sigma.inv()+ C_mag.T @ self.Q_mag.inv() @ C_mag).inv()
    self.mu = self.Sigma @ C_mag.T @ self.Q_mag.inv()
    pass


def gnss_update(
    self: TGEqF,
    lat: float,
    lon: float,
    alt: Optional[float] = None,
    R_gnss: Optional[np.ndarray] = None,
) -> None:
    """
    Update state using GNSS (GPS) measurement.

    Measurement model: y_gnss = position in NED frame

    Args:
        lat: Latitude (degrees).
        lon: Longitude (degrees).
        alt: Altitude (meters), optional.
        R_gnss: Measurement noise covariance (3×3), optional.
    """
    # TODO: Implement GNSS to NED conversion and update
    pass


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
