"""Debug SE23 exponential map to understand axis orientation issue."""

import numpy as np
import sys
from pathlib import Path

ref_path = Path(__file__).parent / "eqf-reference"
sys.path.insert(0, str(ref_path))
utils_path = ref_path / "Utils"
sys.path.insert(0, str(utils_path))

from pylie import SO3, SE23
from Symmetries.Calibrated.SE23_se23.Symmetry import SymGroup, State, InputSpace

print("=" * 70)
print("DEBUG: SE23 Exponential Map")
print("=" * 70)

# Test 1: Pure rotation around X-axis
print("\nTEST 1: SE23 with pure X-axis rotation")
print("-" * 70)

# Create an infinitesimal rotation around X-axis: [phi, rho, psi] where phi is SO3.vee
# For X-rotation, we want [0.1, 0, 0] as the SO3 part
se23_vec = np.zeros((9, 1))
se23_vec[0:3, 0:1] = np.array([[0.1], [0.0], [0.0]])  # Rotation part (X-axis, 0.1 rad)
se23_vec[3:6, 0:1] = np.array([[0.0], [0.0], [0.0]])  # Velocity part (zero)
se23_vec[6:9, 0:1] = np.array([[0.0], [0.0], [0.0]])  # Position part (zero)

print(f"Input SE23 vector (small rotation around X):")
print(f"  phi (rotation): {se23_vec[0:3, 0:1].flatten()}")
print(f"  rho (velocity): {se23_vec[3:6, 0:1].flatten()}")
print(f"  psi (position): {se23_vec[6:9, 0:1].flatten()}")

# Compute exponential
exp_result = SE23.exp(se23_vec)
print(f"\nSE23.exp result:")
print(f"  Type: {type(exp_result)}")
print(f"  R (rotation matrix):")
print(f"    {exp_result.R().as_matrix()}")
print(f"  x (velocity vector): {exp_result.x().as_vector()}")
print(f"  w (position vector): {exp_result.w().as_vector()}")

# Extract Euler angles
R = exp_result.R().as_matrix()
roll = np.arctan2(R[1, 0], R[0, 0])
pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
yaw = np.arctan2(R[2, 1], R[2, 2])

print(f"\nEuler angles extracted from rotation matrix:")
print(f"  Roll (X-rotation): {np.degrees(roll):.2f}° (expected ~5.73°)")
print(f"  Pitch (Y-rotation): {np.degrees(pitch):.2f}° (expected 0.00°)")
print(f"  Yaw (Z-rotation): {np.degrees(yaw):.2f}° (expected 0.00°)")

# Test 2: Check if rotation matrix matches SO3.exp of just the rotation part
print("\n" + "=" * 70)
print("TEST 2: Compare SE23.exp with SO3.exp for rotation only")
print("-" * 70)

phi_only = np.array([[0.1], [0.0], [0.0]])
so3_result = SO3.exp(phi_only)
se23_result_R = SE23.exp(se23_vec).R()

print(f"SO3.exp([0.1, 0, 0]) rotation matrix:")
print(so3_result.as_matrix())
print(f"\nSE23.exp([0.1, 0, 0, ...]) rotation matrix:")
print(se23_result_R.as_matrix())
print(f"\nAre they equal? {np.allclose(so3_result.as_matrix(), se23_result_R.as_matrix())}")

# Test 3: Incrementally apply rotations
print("\n" + "=" * 70)
print("TEST 3: Accumulate rotations step by step")
print("-" * 70)

X_hat = SymGroup.identity()
dt = 0.01
steps = 10

print(f"\nStarting from identity, applying {steps} steps of X-rotation with dt={dt}")
for step in range(1, steps + 1):
    # Create small rotation
    se23_small = np.zeros((9, 1))
    se23_small[0:3, 0:1] = np.array([[0.1 * dt], [0.0], [0.0]])

    # Propagate
    exp_small = SE23.exp(se23_small)
    X_new = SymGroup(exp_small, X_hat.beta)  # Keep same beta
    X_hat = X_new * X_hat  # Group multiplication

    # Extract Euler angles
    R = X_hat.B.R().as_matrix()
    roll = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    yaw = np.arctan2(R[2, 1], R[2, 2])

    t_acc = step * dt
    roll_expected = 0.1 * t_acc

    if step % 2 == 0:  # Print every 2 steps
        print(f"  t={t_acc:.2f}s: roll={np.degrees(roll):.2f}° "
              f"(expected {np.degrees(roll_expected):.2f}°), "
              f"pitch={np.degrees(pitch):.2f}°, yaw={np.degrees(yaw):.2f}°")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("If TEST 1 shows rotation around Z instead of X -> axis mismatch in SO3/SE23")
print("If TEST 2 shows SO3 and SE23 differ -> SE23 has frame issue")
print("If TEST 3 shows yaw instead of roll -> cumulative group mult issue")
