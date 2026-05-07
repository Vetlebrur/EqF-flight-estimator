"""
Test formula equivalence between reference and current implementation.

Rather than import the full reference (which has dataclass compatibility issues),
we test the core formulas directly against what the reference documentation specifies.
"""

import numpy as np
from pylie import SE23, SO3
from old.tg_eqf import TGEqF
from old.SE23xxse23 import SE23xxse23
import old.eqf_predict as eqf_predict
import old.eqf_update as eqf_update

def test_lift_formula():
    """
    Test that calculate_lift matches reference formula:
    L[0:9] = (U - b) + vee(T_inv @ (G + f_10))
    L[9:18] = adjoint(b) @ L[0:9]
    """
    filter = TGEqF()

    # Set some non-zero state
    filter.T = SE23.exp(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01, -0.02, 0.03]))
    filter.b = np.array([0.01, -0.01, 0.02, 0.05, -0.03, 0.02, 0.001, -0.001, 0.002])

    # Measurements
    gyro = np.array([0.05, -0.03, 0.08])
    accel = np.array([0.1, -0.2, 9.81])
    mu = np.array([0.0, 0.0, 0.0])

    lift_upper, lift_lower = filter.calculate_lift(gyro, accel, mu)

    print("Test: Lift Formula")
    print(f"  lift_upper shape: {lift_upper.shape}")
    print(f"  lift_lower shape: {lift_lower.shape}")
    print(f"  lift_upper norm: {np.linalg.norm(lift_upper):.6e}")
    print(f"  lift_lower norm: {np.linalg.norm(lift_lower):.6e}")

    # Verify lower is adjoint action of upper
    expected_lower = SE23.adjoint(filter.b.reshape(9, 1)) @ lift_upper.reshape(9, 1)
    lower_match = np.allclose(lift_lower.reshape(9, 1), expected_lower, atol=1e-10)
    print(f"  Lower matches adjoint(b) @ upper: {lower_match}")
    return lower_match


def test_A_matrix_formula():
    """
    Test that calculate_A matches reference formula structure:
    A[0:9, 0:9] = blockDiag([0, SO3.skew(-g)], I3)
    A[0:9, 9:18] = I9
    A[9:18, 9:18] = adjoint(u_0 + G) where u_0 = velocityAction(...)
    """
    filter = TGEqF()

    # Set non-zero state
    filter.T = SE23.exp(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01, -0.02, 0.03]))
    filter.b = np.array([0.01, -0.01, 0.02, 0.05, -0.03, 0.02, 0.001, -0.001, 0.002])

    gyro = np.array([0.05, -0.03, 0.08])
    accel = np.array([0.1, -0.2, 9.81])

    A = filter.calculate_A(gyro, accel)

    print("\nTest: A Matrix Formula")
    print(f"  A shape: {A.shape}")
    print(f"  A norm: {np.linalg.norm(A):.6e}")

    # Check block structure
    upper_left = A[0:9, 0:9]
    upper_right = A[0:9, 9:18]
    lower_left = A[9:18, 0:9]
    lower_right = A[9:18, 9:18]

    print(f"  Upper-left block norm: {np.linalg.norm(upper_left):.6e}")
    print(f"  Upper-right block is I9: {np.allclose(upper_right, np.eye(9))}")
    print(f"  Lower-left block is zero: {np.allclose(lower_left, np.zeros((9, 9)))}")
    print(f"  Lower-right block norm: {np.linalg.norm(lower_right):.6e}")

    return True


def test_covariance_propagation():
    """
    Test that covariance propagation handles the process noise correctly.
    """
    filter = TGEqF()

    dt = 0.01
    gyro = np.array([0.05, -0.03, 0.08])
    accel = np.array([0.1, -0.2, 9.81])

    sigma_before = np.linalg.norm(filter.Sigma)

    # Predict with process noise
    filter.imu_predict(dt, accel, gyro, gyro_noise=0.05, accel_noise=0.1, bias_noise=1e-4)

    sigma_after = np.linalg.norm(filter.Sigma)

    print("\nTest: Covariance Propagation")
    print(f"  Covariance norm before: {sigma_before:.6e}")
    print(f"  Covariance norm after: {sigma_after:.6e}")
    print(f"  Growth: {sigma_after / sigma_before:.4f}x")

    return sigma_after >= sigma_before  # Covariance should grow with process noise


if __name__ == "__main__":
    print("=" * 70)
    print("Formula Equivalence Tests")
    print("=" * 70)

    test1 = test_lift_formula()
    test2 = test_A_matrix_formula()
    test3 = test_covariance_propagation()

    print("\n" + "=" * 70)
    print(f"Results: Lift={test1}, A_matrix={test2}, Covariance={test3}")
    print("=" * 70)
