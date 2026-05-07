"""
Example: Using the TGEqF (Tangent-Group Equivariant Filter) class.

This example demonstrates the basic usage of TGEqF with predict and update steps.
"""

import numpy as np
from old.tg_eqf import TGEqF
import old.eqf_predict as eqf_predict  # Attaches predict methods to TGEqF
import old.eqf_update as eqf_update   # Attaches update methods to TGEqF


def main():
    """Run example EqF filter."""

    # Initialize filter
    print("Initializing TGEqF...")
    f = TGEqF()
    print(f"  Pose T: {type(f.T).__name__}")
    print(f"  Bias b: {f.b.shape}")
    print(f"  Covariance P: {f.P.shape}")

    # Simulate IMU measurements
    dt = 0.01  # 10 ms
    gyro = np.array([0.01, 0.01, 0.02])  # rad/s
    accel = np.array([0.0, 0.0, 9.81])   # m/s^2

    # Prediction step
    print("\n--- Prediction Step ---")
    print(f"IMU: gyro={gyro}, accel={accel}")
    try:
        f.imu_predict(dt=dt, accel=accel, gyro=gyro)
        print("imu_predict() called")
        print(f"  Position: {f.get_position()}")
        print(f"  Velocity: {f.get_velocity()}")
    except NotImplementedError as e:
        print(f"imu_predict() not yet implemented: {e}")

    # Update steps (placeholders for now)
    print("\n--- Update Steps ---")

    # Magnetometer
    mag = np.array([1.0, 0.0, 0.0])
    try:
        f.magnetometer_update(mag=mag)
        print("magnetometer_update() called")
    except NotImplementedError as e:
        print(f"magnetometer_update() not yet implemented: {e}")

    # GNSS
    try:
        f.gnss_update(lat=37.0, lon=122.0, alt=100.0)
        print("gnss_update() called")
    except NotImplementedError as e:
        print(f"gnss_update() not yet implemented: {e}")

    # Barometer
    try:
        f.barometer_update(pressure=101325.0)
        print("barometer_update() called")
    except NotImplementedError as e:
        print(f"barometer_update() not yet implemented: {e}")

    # Algebra operations
    print("\n--- Lie Algebra Operations ---")
    v = np.random.randn(9)
    print(f"Random se(2,3) vector: shape {v.shape}")

    # Wedge/vee
    mat = TGEqF.wedge(v)
    v_recovered = TGEqF.vee(mat)
    print(f"Wedge/vee roundtrip error: {np.linalg.norm(v - v_recovered):.2e}")

    # Exponential map
    T = TGEqF.exp_map(v)
    v_log = T.log()
    print(f"exp/log roundtrip error: {np.linalg.norm(v - v_log):.2e}")

    # Group operations
    print("\n--- Semi-Direct Product ---")
    f1 = TGEqF()
    f2 = TGEqF()

    f1_composed = f1.compose(f2)
    print(f"Composition works: T_result = T1 * T2")

    f1_inv = f1.inverse()
    print(f"Inverse works: (T, b)^-1 computed")

    # Adjoint action
    Ad = f1.T.Adjoint()
    print(f"Adjoint matrix: {Ad.shape}")

    print("\nExample complete!")


if __name__ == "__main__":
    main()
