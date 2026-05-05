"""Test filter propagation with analytical ground truth."""

import numpy as np
import sys
from pathlib import Path

ref_path = Path(__file__).parent / "eqf-reference"
sys.path.insert(0, str(ref_path))
utils_path = ref_path / "Utils"
sys.path.insert(0, str(utils_path))
from matrix_math import *
from Symmetries.Calibrated.SE23_se23.Symmetry import SymGroup, State, InputSpace, stateAction, SE23
from pylie import SO3, SE23 as SE23_pylie

# Import filter
from eqf_filter import TGEqF, g

def test_pure_x_rotation():
    """Test: constant rotation around x-axis only.

    Input: gyro = [0.1, 0, 0] rad/s (roll rate)
    Expected: after time t, roll angle should be 0.1*t

    This is the simplest analytical test.
    """
    print("=" * 70)
    print("TEST 1: Pure X-axis rotation (constant roll rate)")
    print("=" * 70)

    filt = TGEqF()
    dt = 0.01  # 10ms timestep
    t_end = 5.0  # 5 seconds

    # Constant gyro input: 0.1 rad/s around x-axis only
    gyro_const = np.array([0.1, 0.0, 0.0])
    accel_const = np.array([0.0, 0.0, g])  # 1g downward (no acceleration)

    results = []
    t = 0

    while t < t_end:
        filt.propagate(t, gyro_const, accel_const)

        # Extract state
        xi_hat = filt.xi_hat()
        R = xi_hat.T.R().as_matrix()

        # Compute Euler angles - FIXED for intrinsic/body-frame rotations
        # SE23.exp produces intrinsic rotations: R = Rx(roll) * Ry(pitch) * Rz(yaw)
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # Analytical expectation: pure rotation around x-axis
        roll_expected = 0.1 * t
        pitch_expected = 0.0
        yaw_expected = 0.0

        # Analytical rotation matrix for x-rotation
        c = np.cos(roll_expected)
        s = np.sin(roll_expected)
        R_expected = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

        # Error
        det_R = np.linalg.det(R)
        ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
        R_err = np.linalg.norm(R - R_expected)

        results.append({
            't': t,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'roll_exp': roll_expected,
            'pitch_exp': pitch_expected,
            'yaw_exp': yaw_expected,
            'det': det_R,
            'ortho_err': ortho_err,
            'R_err': R_err,
        })

        t += dt

    # Print results
    print(f"\n{'Time (s)':<10} {'Roll (rad)':<15} {'Expected':<15} {'Error':<15} {'Det(R)':<10} {'Ortho Err':<12}")
    print("-" * 77)
    for i in range(0, len(results), int(0.5 / dt)):  # Print every 0.5 sec
        r = results[i]
        print(f"{r['t']:<10.2f} {r['roll']:<15.6f} {r['roll_exp']:<15.6f} "
              f"{abs(r['roll'] - r['roll_exp']):<15.6f} {r['det']:<10.6f} {r['ortho_err']:<12.6e}")

    # Final state
    r_final = results[-1]
    print(f"\nFinal state (t={r_final['t']:.2f}s):")
    print(f"  Roll:       {np.degrees(r_final['roll']):.2f}° (expected {np.degrees(r_final['roll_exp']):.2f}°)")
    print(f"  Pitch:      {np.degrees(r_final['pitch']):.2f}° (expected {np.degrees(r_final['pitch_exp']):.2f}°)")
    print(f"  Yaw:        {np.degrees(r_final['yaw']):.2f}° (expected {np.degrees(r_final['yaw_exp']):.2f}°)")
    print(f"  Roll error: {np.degrees(abs(r_final['roll'] - r_final['roll_exp'])):.2f}°")

    # Check if test passes
    roll_error_deg = np.degrees(abs(r_final['roll'] - r_final['roll_exp']))
    if roll_error_deg < 0.1:
        print("PASS: Roll angle matches analytical expectation")
        return True
    else:
        print("FAIL: Roll angle diverges from analytical expectation")
        return False


def test_static_zero_input():
    """Test: zero input should maintain identity rotation.

    Input: gyro = [0, 0, 0], accel = [0, 0, g]
    Expected: rotation stays identity (roll=pitch=yaw=0)
    """
    print("\n" + "=" * 70)
    print("TEST 2: Static (zero gyro) with gravity compensation")
    print("=" * 70)

    filt = TGEqF()
    dt = 0.01
    t_end = 2.0

    # Zero gyro, just gravity
    gyro_const = np.array([0.0, 0.0, 0.0])
    accel_const = np.array([0.0, 0.0, g])

    results = []
    t = 0

    while t < t_end:
        filt.propagate(t, gyro_const, accel_const)

        xi_hat = filt.xi_hat()
        R = xi_hat.T.R().as_matrix()

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        results.append({
            't': t,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
        })

        t += dt

    print(f"\n{'Time (s)':<10} {'Roll (°)':<12} {'Pitch (°)':<12} {'Yaw (°)':<12}")
    print("-" * 46)
    for i in range(0, len(results), int(0.5 / dt)):
        r = results[i]
        print(f"{r['t']:<10.2f} {np.degrees(r['roll']):<12.4f} "
              f"{np.degrees(r['pitch']):<12.4f} {np.degrees(r['yaw']):<12.4f}")

    r_final = results[-1]
    max_angle = max(abs(np.degrees(r_final['roll'])), abs(np.degrees(r_final['pitch'])),
                    abs(np.degrees(r_final['yaw'])))

    print(f"\nFinal angles: Roll={np.degrees(r_final['roll']):.4f}°, "
          f"Pitch={np.degrees(r_final['pitch']):.4f}°, Yaw={np.degrees(r_final['yaw']):.4f}°")

    if max_angle < 0.01:
        print("PASS: Rotation stays near identity")
        return True
    else:
        print(f"FAIL: Rotation drifted by {max_angle:.4f} degrees")
        return False


if __name__ == "__main__":
    test1 = test_pure_x_rotation()
    test2 = test_static_zero_input()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Pure X-rotation):    {'PASS' if test1 else 'FAIL'}")
    print(f"Test 2 (Static zero input):  {'PASS' if test2 else 'FAIL'}")

    if test1 and test2:
        print("\nAll tests passed - propagation is correct")
    else:
        print("\nSome tests failed - PROPAGATION HAS CRITICAL ISSUES")
