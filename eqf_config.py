import numpy as np
#TODO: read up on uncertainty in sensors and correct inital values
Q_0_magnetometer = np.diag([2,2,2])

Q_0_barometer = 2

Q_0_gnss = np.diag([4,4,4,6,6,6])

Sigma_0 = np.diag(
    np.eye(3)*2,
    np.eye(3)*4,
    np.eye(3)*3,
    np.eye(3)*2,
    np.eye(3)*2,
    np.eye(3)*2
    )

P_0 = np.diag(
    np.eye(3)*2,
    np.eye(3)*4,
    np.eye(3)*3,
    np.eye(3)*2,
    np.eye(3)*2,
    np.eye(3)*2
    )

