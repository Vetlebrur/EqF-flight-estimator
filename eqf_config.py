import numpy as np
from scipy.linalg import block_diag
#TODO: read up on uncertainty in sensors and correct inital values
Q_0_magnetometer = np.diag([2,2,2])
m_0 = np.array([1,0,0])
Q_0_barometer = 2

Q_0_gnss = np.diag([16,16,16,25,25,25])

Sigma_0 = block_diag(
    np.eye(3)*2,
    np.eye(3)*4,
    np.eye(3)*3,
    np.eye(3)*2,
    np.eye(3)*2,
    np.eye(3)*2
)

P_0 = block_diag(
    np.eye(3)*2,
    np.eye(3)*4,
    np.eye(3)*3,
    np.eye(3)*2,
    np.eye(3)*2,
    np.eye(3)*2
)

