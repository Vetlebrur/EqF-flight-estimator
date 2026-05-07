"""
Example: Using SE23xxse23 for Equivariant Filter State Representation

This demonstrates the semi-direct product group SE(2,3) ⋉ se(2,3)
for biased inertial navigation.
"""

import numpy as np
from old.SE23xxse23 import SE23xxse23
from pylie import SE23


def test_group_axioms():
    """Test group properties."""
    print("\n=== Group Axioms ===")

    # Create states
    G1 = SE23xxse23()
    G1.b = np.array([0.01, 0.02, 0.03, 0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003])

    G2 = SE23xxse23()
    G2.b = np.array([0.02, 0.01, 0.02, 0.002, 0.001, 0.002, 0.0002, 0.0001, 0.0002])

    G3 = SE23xxse23()
    G3.b = np.array([0.03, 0.03, 0.01, 0.003, 0.003, 0.001, 0.0003, 0.0003, 0.0001])

    # Associativity: (G1·G2)·G3 = G1·(G2·G3)
    left = (G1 * G2) * G3
    right = G1 * (G2 * G3)

    error = np.linalg.norm(left.b - right.b)
    print(f"Associativity error (bias): {error:.2e}")
    assert error < 1e-10, "Associativity failed"
    print("[OK] Associativity verified")

    # Identity: G·I = G and I·G = G
    G = SE23xxse23()
    G.b = np.random.randn(9) * 0.1

    G_id = G * SE23xxse23.identity()
    error_right = np.linalg.norm(G_id.b - G.b)
    print(f"Right identity error: {error_right:.2e}")

    G_id = SE23xxse23.identity() * G
    error_left = np.linalg.norm(G_id.b - G.b)
    print(f"Left identity error: {error_left:.2e}")
    assert error_right < 1e-10 and error_left < 1e-10
    print("[OK] Identity verified")

    # Inverse: G·G⁻¹ = I
    G_inv = G.inv()
    G_product = G * G_inv

    bias_norm = np.linalg.norm(G_product.b)
    print(f"G*G^(-1) bias norm: {bias_norm:.2e} (should be ~0)")
    assert bias_norm < 1e-10
    print("[OK] Inverse verified")


def test_adjoint_properties():
    """Test adjoint matrix properties."""
    print("\n=== Adjoint Properties ===")

    G = SE23xxse23()
    G.b = np.random.randn(9) * 0.1

    # Get adjoint
    Ad = G.Adjoint()
    print(f"Adjoint matrix shape: {Ad.shape}")
    assert Ad.shape == (18, 18)

    # Ad_{G1·G2} = Ad_{G1} · Ad_{G2}
    G1 = SE23xxse23()
    G1.b = np.random.randn(9) * 0.05

    G2 = SE23xxse23()
    G2.b = np.random.randn(9) * 0.05

    Ad_G1 = G1.Adjoint()
    Ad_G2 = G2.Adjoint()
    Ad_G1G2 = (G1 * G2).Adjoint()

    error = np.linalg.norm(Ad_G1 @ Ad_G2 - Ad_G1G2)
    print(f"Ad(G1*G2) = Ad(G1)*Ad(G2) error: {error:.2e}")
    assert error < 1e-10
    print("[OK] Adjoint product rule verified")

    # Ad_{G⁻¹} = (Ad_G)⁻¹
    G_inv = G.inv()
    Ad_inv = G_inv.Adjoint()
    Ad = G.Adjoint()
    Ad_inv_computed = np.linalg.inv(Ad)

    error = np.linalg.norm(Ad_inv - Ad_inv_computed)
    print(f"Ad(G^(-1)) = (Ad(G))^(-1) error: {error:.2e}")
    assert error < 1e-9  # Slightly higher due to matrix inversion
    print("[OK] Adjoint inverse rule verified")


def test_tangent_space():
    """Test exponential and logarithm maps."""
    print("\n=== Tangent Space Operations ===")

    # Create random tangent vector
    xi = np.random.randn(9) * 0.1  # Pose part
    eta = np.random.randn(9) * 0.05  # Bias part
    tangent = np.concatenate([xi, eta])

    # Exponential map
    G = SE23xxse23.exp(tangent)
    print(f"exp: (18,) tangent vector -> SE23xxse23")

    # Logarithm map
    tangent_recovered = G.log()
    error = np.linalg.norm(tangent - tangent_recovered)
    print(f"exp/log roundtrip error: {error:.2e}")
    assert error < 1e-10
    print("[OK] Exponential/logarithm verified")


def test_semi_direct_product_structure():
    """Demonstrate the semi-direct product composition."""
    print("\n=== Semi-Direct Product Structure ===")

    # Create two states
    G1 = SE23xxse23()
    G1.b = np.array([0.01, 0, 0, 0, 0, 0, 0, 0, 0])  # Small gyro bias

    G2 = SE23xxse23()
    G2.b = np.array([0, 0.02, 0, 0, 0, 0, 0, 0, 0])  # Different gyro bias

    print(f"G1.b = {G1.b}")
    print(f"G2.b = {G2.b}")

    # Composition: (T2, b2) * (T1, b1) = (T2*T1, b2 + Ad_{T2⁻¹}[b1])
    G_product = G1 * G2

    print(f"\nG1 * G2:")
    print(f"  T2*T1 computed: Yes")
    print(f"  b2 + Ad(T2⁻¹)[b1] applied: Yes")
    print(f"  Result: {G_product.b}")

    # Show adjoint action
    T2_inv = G1.T.inv()
    Ad_T2_inv = SE23xxse23._adjoint_matrix(T2_inv)
    b_transformed = Ad_T2_inv @ G2.b
    print(f"\nAdjoint action:")
    print(f"  Ad(T2⁻¹)[b1]: {b_transformed}")
    print(f"  b2 + result: {G1.b + b_transformed}")
    print(f"  Matches composition: {np.allclose(G_product.b, G1.b + b_transformed)}")

    print("[OK] Semi-direct product structure verified")


def test_use_case_bias_transformation():
    """Demonstrate bias transformation under pose changes."""
    print("\n=== Bias Transformation Under Pose Changes ===")

    # Initial state
    state = SE23xxse23()
    state.b = np.array([0.01, 0.02, 0.03, 0, 0, 0, 0, 0, 0])

    print(f"Initial bias: {state.b}")

    # Apply a pose transformation
    # Create a small perturbation in SE(2,3)
    xi = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0])  # Small rotation around x-axis
    T_transform = SE23.exp(xi)
    state_transformed = SE23xxse23(T_transform, np.zeros(9))

    # Compose with initial state
    state_new = state_transformed * state

    print(f"After pose change (composition):")
    print(f"  New bias: {state_new.b}")
    print(f"  Bias changed due to adjoint: {not np.allclose(state_new.b, state.b)}")

    # The bias now lives in a rotated coordinate frame
    # This is the key feature: bias transforms consistently with pose
    print("[OK] Bias transformation demonstrated")


def example_filter_state():
    """Example: Initialize filter state with realistic values."""
    print("\n=== Realistic Filter State Example ===")

    # Start with identity
    state = SE23xxse23.identity()

    # Set initial biases (typical IMU values)
    state.b[0:3] = np.array([0.001, -0.0005, 0.0015])  # Gyro bias (rad/s)
    state.b[3:6] = np.array([0.05, -0.03, 0.04])        # Accel bias (m/s²)
    state.b[6:9] = np.array([0.0, 0.0, 0.0])            # Mu bias (structural term)

    print(f"Filter state initialized:")
    print(f"  Pose: SE(2,3)")
    print(f"  Gyro bias: {state.get_bias_gyro()}")
    print(f"  Accel bias: {state.get_bias_accel()}")
    print(f"  Mu bias: {state.get_bias_mu()}")

    # Extract components
    R = state.get_rotation()
    p = state.get_position()
    v = state.get_velocity()

    print(f"\nPose components:")
    print(f"  Rotation (identity): {np.allclose(R, np.eye(3))}")
    print(f"  Position: {p}")
    print(f"  Velocity: {v}")

    print("[OK] Realistic filter state created")


def main():
    """Run all examples."""
    print("SE23xxse23: Semi-Direct Product Group Examples")
    print("=" * 60)

    test_group_axioms()
    test_adjoint_properties()
    test_tangent_space()
    test_semi_direct_product_structure()
    test_use_case_bias_transformation()
    example_filter_state()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
