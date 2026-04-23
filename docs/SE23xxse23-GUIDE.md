# SE23xxse23: Semi-Direct Product Group SE(2,3) ⋉ se(2,3)

## Overview

`SE23xxse23` is a Python implementation of the semi-direct product Lie group **SE(2,3) ⋉ se(2,3)** for biased inertial navigation systems.

This group is central to the Equivariant Filter (EqF) design from Fornasier et al. (2022).

## Group Structure

### Elements

Each element is a pair `(T, b)` where:
- **T ∈ SE(2,3)**: Extended pose encoding rotation R, position p, velocity v
- **b ∈ ℝ⁹**: Bias in the Lie algebra se(2,3), with components:
  - b[0:3]: Gyroscope bias
  - b[3:6]: Velocity bias
  - b[6:9]: Accelerometer bias

### Composition (Group Multiplication)

```
(T₂, b₂) ∘ (T₁, b₁) = (T₂·T₁, b₂ + Ad_{T₂⁻¹}[b₁])
```

The key feature: **bias transforms by the adjoint action** of the inverse pose. This ensures equivariance.

### Inverse

```
(T, b)⁻¹ = (T⁻¹, -Ad_{T⁻¹}[b])
```

## Usage

### Initialization

```python
from SE23xxse23 import SE23xxse23
from pylie import SE23
import numpy as np

# Create from components
T = SE23.identity()
b = np.zeros(9)
state = SE23xxse23(T, b)

# Create identity
state_id = SE23xxse23.identity()

# Create with random bias
state = SE23xxse23()
state.b = np.random.randn(9) * 0.1
```

### Group Operations

```python
# Composition
state1 = SE23xxse23()
state2 = SE23xxse23()
state_product = state1 * state2  # (T1·T2, b1 + Ad_{T1⁻¹}[b2])

# Division
state_quotient = state1 / state2  # state1 * state2.inv()

# Inverse
state_inv = state.inv()

# Verify: state * state^{-1} = identity
identity = state * state.inv()
print(f"Bias norm: {np.linalg.norm(identity.b)}")  # ~0
```

### Adjoint Action

The adjoint matrix (18×18) acts on the tangent space ℝ¹⁸:

```python
# Get adjoint matrix
Ad = state.Adjoint()  # Returns 18x18 matrix

# Apply to tangent vector [ξ; η] (9+9 components)
xi_eta = np.concatenate([xi_pose, eta_bias])
xi_eta_transformed = Ad @ xi_eta
```

Structure of adjoint for semi-direct product:
```
Ad_{(T,b)} = [ Ad_T    0   ]
             [ Bracket Ad_T ]
```

Where the lower-left "Bracket" term represents the coupling from the Lie bracket [ξ, b].

### Exponential and Logarithm Maps

```python
# Exponential map: tangent space -> group
tangent = np.random.randn(18)  # [ξ (9); η (9)]
state = SE23xxse23.exp(tangent)

# Logarithmic map: group -> tangent space
tangent_recovered = state.log()

# Verify roundtrip (at first order)
error = np.linalg.norm(tangent - tangent_recovered)
print(f"Error: {error:.2e}")
```

### Component Access

```python
# Extract pose components
R = state.get_rotation()        # 3×3 rotation matrix
p = state.get_position()        # (3,) position vector
v = state.get_velocity()        # (3,) velocity vector

# Extract bias components
b_gyro = state.get_bias_gyro()      # (3,) gyroscope bias
b_vel = state.get_bias_velocity()   # (3,) velocity bias
b_accel = state.get_bias_accel()    # (3,) accelerometer bias

# Full state
T = state.get_pose()     # SE23 element
b = state.get_bias()     # Full 9-vector
```

### Matrix Representation

```python
# Convert to 18×18 matrix (for visualization/storage)
mat = state.as_matrix()

# Convert from matrix
state_from_mat = SE23xxse23.from_matrix(mat)
```

### Wedge/Vee Operations

```python
# Wedge: tangent vector (18,) -> matrix (18×18)
tangent = np.random.randn(18)
mat = SE23xxse23.wedge(tangent)

# Vee: matrix (18×18) -> tangent vector (18,)
tangent_recovered = SE23xxse23.vee(mat)
```

## Mathematical Details

### Semi-Direct Product Law

The group law comes from the semi-direct product structure:

```
(T₂, b₂) ∘ (T₁, b₁) = (T₂·T₁, b₂ + Ad_{T₂⁻¹}[b₁])
```

**Why the adjoint?** The bias lives in the Lie algebra, which transforms under group actions by adjoint. Without the adjoint transformation, the group law wouldn't be associative.

### Adjoint Action

For an element `(T, b)`, the adjoint on tangent vectors `[ξ; η]`:

```
Ad_{(T,b)}[ξ; η] = [ Ad_T(ξ)              ]
                   [ ad_ξ(b) + Ad_T(η)  ]
```

Where:
- `Ad_T`: Standard SE(2,3) adjoint (9×9)
- `ad_ξ`: Lie bracket action
- The coupling term `ad_ξ(b)` ensures equivariance

### Lie Brackets

The Lie bracket in se(2,3) is computed via matrix commutator:

```python
# Using SE23 algebra
u_mat = SE23.wedge(u)
v_mat = SE23.wedge(v)
bracket_mat = u_mat @ v_mat - v_mat @ u_mat
bracket_vec = SE23.vee(bracket_mat)  # = [u, v]
```

## Integration with TGEqF

Use `SE23xxse23` as the underlying state representation for TGEqF:

```python
from SE23xxse23 import SE23xxse23
from tg_eqf import TGEqF
import eqf_predict
import eqf_update

# Create EqF state
state = SE23xxse23.identity()
state.b = np.random.randn(9) * 0.01

# Use with TGEqF (if modified to accept SE23xxse23)
# Or keep TGEqF as is and use SE23xxse23 separately
```

## Properties and Invariants

### Group Axioms

```python
# Associativity
G1, G2, G3 = SE23xxse23(), SE23xxse23(), SE23xxse23()
assert np.allclose((G1 * G2) * G3, G1 * (G2 * G3))  # Always true

# Identity
G = SE23xxse23()
assert np.allclose((G * SE23xxse23.identity()).b, G.b)

# Inverse
G_inv = G.inv()
G_prod = G * G_inv
assert np.allclose(G_prod.b, 0)  # Bias of identity is zero
```

### Adjoint Properties

```python
# Ad_{G1·G2} = Ad_{G1} · Ad_{G2}
Ad_G1 = G1.Adjoint()
Ad_G2 = G2.Adjoint()
Ad_G1G2 = (G1 * G2).Adjoint()
assert np.allclose(Ad_G1 @ Ad_G2, Ad_G1G2)

# Ad_{G⁻¹} = (Ad_G)⁻¹
Ad_inv = G_inv.Adjoint()
Ad = G.Adjoint()
assert np.allclose(Ad_inv, np.linalg.inv(Ad))
```

## Comparison with PyLie SE23

| Feature | SE23 | SE23xxse23 |
|---------|------|-----------|
| **Dimension** | 9 (pose only) | 18 (pose + bias) |
| **Elements** | (R, p, v) | ((R, p, v), b) |
| **Composition** | Standard SE(2,3) | Semi-direct product with adjoint |
| **Adjoint** | 9×9 matrix | 18×18 matrix |
| **Bias handling** | Not supported | Explicit + coupled to pose |

## Key Differences from SE23

1. **Bias is explicit**: SE23xxse23 stores bias as a separate 9-vector
2. **Adjoint coupling**: Semi-direct product introduces coupling in adjoint
3. **Larger state space**: 18 DOF instead of 9
4. **Equivariance structure**: Designed for EqF theory

## Testing

Run the built-in tests:

```bash
python3 SE23xxse23.py
```

Expected output:
- Identity element created
- Composition computed
- Inverse verified
- Adjoint matrix 18×18
- Exp/log roundtrip error < 1e-10

## Files

- `SE23xxse23.py` - Main implementation
- `docs/SE23xxse23-GUIDE.md` - This guide
- `example_tg_eqf.py` - Example usage

## References

- **Fornasier et al. (2022)**: "Equivariant Filter Design for Inertial Navigation Systems with Input Measurement Biases"
- **PyLie**: Python Lie groups library - SE(2,3) implementation
- **Lie Group Theory**: Hall, "Lie Groups, Lie Algebras, and Representations"

