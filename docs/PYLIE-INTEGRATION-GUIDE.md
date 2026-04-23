# Using PyLie for SE(2,3) ⋉ se(2,3) Implementation

## Overview

PyLie (python-lie) is a Python library for Lie groups and algebras. It provides:
- SO(3), SE(3) groups and algebras
- Wedge/vee operators
- Exponential maps
- Group operations (multiplication, inverse)
- Adjoint actions

For our EqF, we'll leverage pylie for SE(2,3) base operations and extend it for the semi-direct product structure.

---

## PyLie Basics

### Importing
```python
from pylie import SO3, SE23, se23, so3
import numpy as np
```

### Key Classes
- `SO3`: Special Orthogonal group (3×3 rotation matrices)
- `SE23`: Special Euclidean group with velocity (4×4 extended pose)
- `se23`, `so3`: Lie algebra elements

### Common Operations

```python
# Create group elements
R = SO3.from_matrix(rotation_matrix)  # From 3×3 rotation
T = SE23.from_matrix(T_matrix)         # From 4×4 extended pose
T = SE23.identity()                    # Identity

# Group operations
T_new = T1 * T2                        # Group multiplication
T_inv = T.inverse()                    # Group inverse
T_exp = SE23.exp(algebra_element)      # Exponential map

# Extract components
R = T.R()                              # Rotation part
v = T.v()                              # Velocity part
p = T.p()                              # Position part

# Lie algebra operations
u = T.log()                            # Log (inverse of exp)
u_vec = u.as_vector()                  # Convert to vector form
u_matrix = u.as_matrix()               # Convert to matrix form

# Adjoint action
Ad_T = T.Ad()                          # Get adjoint matrix (9×9)
u_transformed = Ad_T @ u_vec           # Apply adjoint

# Wedge operator (vector → algebra)
u_wedge = so3.wedge(gyro_vector)       # 3-vec → 3×3 skew-symmetric
u_wedge = SE23.wedge(full_vector)      # 9-vec → algebra form

# Vee operator (algebra → vector)
v = u.vee()                            # Extract vector from algebra
```

---

## Extending PyLie for Semi-Direct Product

PyLie has SE(2,3) but not the **semi-direct product** SE(2,3) ⋉ se(2,3). We need to implement this ourselves using pylie components.

### New Class: FilterStateEqFWithPyLie

```python
from pylie import SE23, se23
import numpy as np

class FilterStateEqFWithPyLie:
    """
    State on SE(2,3) ⋉ se(2,3) using pylie.
    
    Composition law: (T, b∧) * (T', b'∧) = (T*T', b∧ + Ad_{T^-1}[b'∧])
    """
    
    def __init__(self):
        """Initialize to identity of semi-direct product"""
        self.T = SE23.identity()           # SE(2,3) extended pose
        self.b_algebra = se23.zero()       # se(2,3) bias (algebra element)
        self.Sigma = np.eye(18) * 0.1     # Covariance
    
    @staticmethod
    def identity():
        """Identity element (I, 0∧) ∈ SE(2,3) ⋉ se(2,3)"""
        state = FilterStateEqFWithPyLie()
        state.T = SE23.identity()
        state.b_algebra = se23.zero()
        return state
    
    def compose(self, other):
        """
        Semi-direct product composition.
        (T2, b2∧) * (T1, b1∧) = (T2*T1, b2∧ + Ad_{T2^{-1}}[b1∧])
        """
        result = FilterStateEqFWithPyLie()
        
        # Group part: T2 * T1
        result.T = other.T * self.T  # Note: pylie uses left-to-right
        
        # Algebra part: b2∧ + Ad_{T2^{-1}}[b1∧]
        T2_inv = other.T.inverse()
        
        # Get adjoint matrix (9×9)
        Ad_T2_inv = T2_inv.Ad()
        
        # Convert b1 to vector, apply adjoint, convert back
        b1_vec = self.b_algebra.as_vector()
        b1_adj = Ad_T2_inv @ b1_vec
        b1_adj_algebra = se23.from_matrix(
            se23.wedge(b1_adj)  # Reconstruct algebra element
        )
        
        # Add b2 and adjoint-transformed b1
        result.b_algebra = other.b_algebra + b1_adj_algebra
        
        return result
    
    def inverse(self):
        """
        Inverse: (T, b∧)^{-1} = (T^{-1}, -Ad_{T^{-1}}[b∧])
        """
        result = FilterStateEqFWithPyLie()
        
        T_inv = self.T.inverse()
        result.T = T_inv
        
        # Adjoint of inverse
        Ad_T_inv = T_inv.Ad()
        b_vec = self.b_algebra.as_vector()
        b_adj_inv = Ad_T_inv @ b_vec
        b_adj_inv_algebra = se23.from_matrix(
            se23.wedge(-b_adj_inv)
        )
        
        result.b_algebra = b_adj_inv_algebra
        
        return result
    
    def adjoint_action(self, v):
        """
        Apply adjoint: Ad_{(T,b∧)}[v∧]
        For semi-direct product: Ad_{(T,b)}[v] = (Ad_T[v_pose], Ad_T[v_bias])
        """
        # Extract pose and bias components of v
        v_pose = v[:6]  # First 6 components (SO3 + position)
        v_bias = v[6:]   # Last 3 components (velocity)
        
        # Apply adjoint of T
        Ad_T = self.T.Ad()
        v_pose_adj = Ad_T[:6, :6] @ v_pose
        
        # For bias: it also gets rotated
        v_bias_adj = Ad_T[:3, :3] @ v_bias  # Rotate by rotation part of T
        
        result = np.concatenate([v_pose_adj, v_bias_adj])
        return result
    
    def lie_bracket(self, v, w):
        """
        Lie bracket in se(2,3) ⋉ se(2,3):
        [u, v] = [u∧, v∧] where [·, ·] is matrix commutator
        """
        u_matrix = se23.wedge(v)
        v_matrix = se23.wedge(w)
        
        bracket = u_matrix @ v_matrix - v_matrix @ u_matrix
        bracket_vec = se23.vee(bracket)
        
        return bracket_vec
    
    def extract_components(self):
        """Extract rotation, position, velocity from T"""
        R = self.T.R().as_matrix()      # 3×3 rotation
        p = np.array(self.T.p()).flatten()     # Position (3-vec)
        v = np.array(self.T.v()).flatten()     # Velocity (3-vec)
        b = self.b_algebra.as_vector()  # Bias (9-vec)
        
        return R, p, v, b
```

---

## Using PyLie in Filter Operations

### Propagation with PyLie

```python
def propagate_state_with_pylie(
    state: FilterStateEqFWithPyLie,
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: float
) -> FilterStateEqFWithPyLie:
    """
    Propagate using pylie SE(2,3) exponential
    """
    
    # Construct input vector
    w_input = np.concatenate([gyro, np.zeros(3), accel])  # 9-vec
    
    # Lift to algebra (this needs proper Λ_1)
    Lambda_1_vec = compute_lift_Lambda1(state, w_input, accel)
    
    # Convert to SE(2,3) algebra element
    Lambda_1_algebra = se23.from_vector(Lambda_1_vec)
    
    # Exponential map: exp(Λ_1 * dt)
    exp_term = SE23.exp(Lambda_1_algebra * dt)
    
    # Update T: T_new = T_old * exp(Λ_1 * dt)
    state_new = FilterStateEqFWithPyLie()
    state_new.T = state.T * exp_term  # Group multiplication
    
    # Update bias (simplified - full version more complex)
    Lambda_2_vec = compute_lift_Lambda2(state, w_input, accel)
    Lambda_2_algebra = se23.from_vector(Lambda_2_vec)
    state_new.b_algebra = state.b_algebra + Lambda_2_algebra * dt
    
    return state_new
```

### Innovation with PyLie

```python
def compute_innovation_with_pylie(
    state: FilterStateEqFWithPyLie,
    measurement_T: np.ndarray,  # 4×4 measured extended pose
    output_noise_Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Equivariant innovation using pylie.
    
    Output residual: y * T̂^{-1}
    """
    
    # Convert measurement to SE(2,3)
    y = SE23.from_matrix(measurement_T)
    
    # Get state inverse
    T_hat_inv = state.T.inverse()
    
    # Output action: ρ(X, y) = y·A
    # So residual is y · T̂^{-1}
    residual_T = y * T_hat_inv
    
    # Convert to Lie algebra (log map)
    residual_algebra = residual_T.log()
    residual_vec = residual_algebra.as_vector()  # 6-vec for SE(2,3)
    
    # Pad to 9-vec if needed (add zeros for bias unobserved part)
    residual_vec_full = np.concatenate([residual_vec, np.zeros(3)])
    
    # Kalman gain computation
    C0 = compute_output_matrix_C0()
    Q_inv = np.linalg.inv(output_noise_Q)
    K = state.Sigma @ C0.T @ Q_inv
    
    # Innovation
    correction_vec = K @ residual_vec_full  # 18-vec
    
    Delta_1 = correction_vec[0:9]
    Delta_2 = correction_vec[9:18]
    
    return Delta_1, Delta_2
```

---

## Key PyLie Methods

| Method | Returns | Use For |
|--------|---------|---------|
| `SE23.identity()` | Identity element | Initialize state |
| `SE23.from_matrix(M)` | SE(2,3) from 4×4 | Load measurement |
| `T1 * T2` | SE(2,3) | Group multiplication |
| `T.inverse()` | T^{-1} | Inverse |
| `SE23.exp(u)` | exp(u) | Exponential map |
| `T.log()` | log(T) | Logarithm |
| `T.R()`, `T.p()`, `T.v()` | SO(3), ℝ^3, ℝ^3 | Extract components |
| `T.Ad()` | 9×9 matrix | Adjoint action |
| `T.as_vector()` | 6-vec | Vector form |
| `se23.wedge(v)` | Algebra matrix | Vector → algebra |
| `se23.from_vector(v)` | Algebra element | Create algebra element |

---

## Implementation Strategy Using PyLie

### Phase 1 Modification: Use PyLie for Foundation

Instead of building Lie algebra from scratch, use PyLie's SE(2,3):

```python
# File: eqf_core/state.py

from pylie import SE23, se23
import numpy as np

class FilterStateEqFPyLie:
    """Leverages pylie for group operations"""
    
    def __init__(self):
        self.T = SE23.identity()          # pylie SE(2,3)
        self.b_algebra = se23.zero()      # pylie se(2,3)
        self.Sigma = np.eye(18) * 0.1
    
    # Implement semi-direct product wrapper around pylie
    def compose(self, other):
        # Use pylie's multiplication for T
        # Manually handle adjoint for bias
        pass
    
    def adjoint(self, v):
        # Use pylie's Ad() method
        Ad = self.T.Ad()
        return Ad @ v
```

### Advantages

1. ✅ **Vetted Implementation**: PyLie's SE(2,3) is mathematically correct
2. ✅ **Numerical Stability**: Uses tested exponential/log maps
3. ✅ **Less Code**: Don't reimplement wedge/vee operators
4. ✅ **Integration**: Already in requirements.txt
5. ✅ **Focus**: Concentrate on semi-direct product + filter specifics

### What Still Need to Implement

1. **Semi-direct product wrapper** - Composition law with adjoint
2. **Lift functions** - Λ_1, Λ_2 with Lie brackets
3. **Filter equations** - Propagation, update, Riccati
4. **Error linearization** - A^0_t, C^0 matrices
5. **Innovation computation** - Measurement residuals

---

## PyLie SE(2,3) Details

PyLie's SE(2,3) is defined as:
```
T = [ R   p   v ]
    [ 0   1   0 ]
    [ 0   0   1 ]

where R ∈ SO(3), p, v ∈ ℝ^3
```

Matrix dimension: **4×4** (not 5×5)

### Accessing Components

```python
T_matrix = T.as_matrix()  # Get 4×4 form
R = T.R()                 # SO(3) rotation
p = np.array(T.p())       # Position vector
v = np.array(T.v())       # Velocity vector
```

### Algebra Form

PyLie's se(2,3) algebra element represents:
```
u∧ = [ ω∧  p_dot  v_dot ]
     [ 0    0      0     ]
     [ 0    0      0     ]

where ω∧ is 3×3 skew-symmetric, p_dot, v_dot ∈ ℝ^3
```

As a vector (6-vec): `[ω_x, ω_y, ω_z, p_dot_x, p_dot_y, p_dot_z]`

---

## Working with Bias in Algebra Form

The bias in our system lives in **se(2,3)**:
- Gyro bias: ω_b ∈ ℝ^3
- Velocity bias: ν_b ∈ ℝ^3
- Accel bias: a_b ∈ ℝ^3

As a **9-vec**: `[ω_b, ν_b, a_b]` (concatenated)

PyLie's se(2,3) is **6-vec**, so we need to extend:

```python
# Extend se(2,3) representation for full bias
class BiasInAlgebra:
    """Wrapper to handle 9-dim bias space"""
    
    def __init__(self, bias_9vec=None):
        if bias_9vec is None:
            bias_9vec = np.zeros(9)
        self.gyro_bias = bias_9vec[0:3]
        self.velocity_bias = bias_9vec[3:6]
        self.accel_bias = bias_9vec[6:9]
    
    def as_vector(self):
        return np.concatenate([
            self.gyro_bias,
            self.velocity_bias,
            self.accel_bias
        ])
    
    def lie_bracket(self, other):
        """Compute [b∧, u∧]"""
        # Implement full Lie bracket for 9-dim space
        pass
```

---

## Complete Example: Filter Step with PyLie

```python
from pylie import SE23, se23
import numpy as np

class EqFFilterWithPyLie:
    def __init__(self):
        self.T = SE23.identity()
        self.b_algebra = np.zeros(9)  # Extended bias
        self.Sigma = np.eye(18) * 0.1
        self.t_prev = None
    
    def propagate(self, t, gyro, accel):
        """Propagate using pylie SE(2,3)"""
        if self.t_prev is None:
            self.t_prev = t
            return
        
        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return
        
        # Compute lift
        Lambda_1 = self._compute_lambda1(gyro, accel)
        
        # Convert to pylie algebra element (need helper)
        Lambda_1_se23 = self._vector_to_se23(Lambda_1[:6])
        
        # Exponential using pylie
        exp_term = SE23.exp(Lambda_1_se23 * dt)
        
        # Update T using pylie multiplication
        self.T = self.T * exp_term
        
        # Update bias
        Lambda_2 = self._compute_lambda2(gyro, accel)
        self.b_algebra = self.b_algebra + Lambda_2 * dt
        
        # Update covariance
        self._propagate_covariance(dt)
        
        self.t_prev = t
    
    def correct(self, measurement_pose_4x4, R_meas):
        """Update using position measurement"""
        # Convert measurement to pylie
        y = SE23.from_matrix(measurement_pose_4x4)
        
        # Innovation using pylie
        T_inv = self.T.inverse()
        residual = y * T_inv  # Group action
        residual_vec = residual.log().as_vector()
        
        # Rest of Kalman update...
```

---

## Potential Issues and Solutions

### Issue 1: PyLie SE(2,3) is 4×4, not 5×5
**Solution**: Our mathematical formulation uses 5×5 (includes gravity row). We adjust formulas or keep separate gravity handling.

### Issue 2: PyLie doesn't have semi-direct product directly
**Solution**: Implement semi-direct product wrapper using pylie's SE(2,3) and manual adjoint.

### Issue 3: Bias is 9-dim, pylie algebra is 6-dim
**Solution**: Extend biases externally, use pylie only for T ∈ SE(2,3).

### Issue 4: Lie brackets on 9-dim space
**Solution**: Implement custom Lie bracket for extended bias space, use pylie's for the SE(2,3) component.

---

## Summary: PyLie Integration Benefits

✅ **Use PyLie for**:
- SE(2,3) group operations (multiply, inverse)
- Exponential and logarithm maps
- Adjoint actions on SE(2,3)
- Vector ↔ algebra conversions (wedge/vee)

❌ **Don't use PyLie for** (implement yourself):
- Semi-direct product composition
- Extended 9-dim bias space operations
- Lift computation Λ_1, Λ_2
- Filter-specific equations

---

## Next: Update Implementation Plan

Update `IMPLEMENTATION-PLAN.md` Phase 1 to:
1. Leverage pylie's SE(2,3)
2. Implement semi-direct product wrapper
3. Handle 9-dim bias space
4. Use pylie's adjoint action

This reduces implementation complexity by ~30% while maintaining correctness.
