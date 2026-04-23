# PyLie SE(2,3) Algebra Reference

## Finding se(2,3) in PyLie

The Lie algebra **se(2,3)** is represented as **9-dimensional vectors** in PyLie. There is no explicit `se23` class - instead, use numpy arrays and PyLie's conversion functions.

### Basic Usage

```python
from pylie import SE23
import numpy as np

# se(2,3) is represented as ℝ⁹
xi = np.array([
    0.01, 0.02, 0.03,      # Rotational part [ω_x, ω_y, ω_z]
    1.0, 2.0, 3.0,          # Position part [p_x, p_y, p_z]
    0.0, 0.1, 0.2           # Velocity part [v_x, v_y, v_z]
])  # shape (9,)

# Exponential map: se(2,3) → SE(2,3)
T = SE23.exp(xi)           # Returns SE23 group element

# Logarithmic map: SE(2,3) → se(2,3)
xi_recovered = T.log()      # Returns ℝ⁹ vector
```

## Conversion Functions

### Wedge: Vector to Matrix Form

Converts a 9-vector to its matrix representation (5×5):

```python
xi = np.random.randn(9)
mat = SE23.wedge(xi)        # Returns 5×5 matrix

# Structure of wedge(xi):
# ┌                    ┐
# │  ω∧    p    v    │
# │  0    0    0    │  where ω∧ is 3×3 skew-symmetric
# │  0    0    0    │
# └                    ┘
```

### Vee: Matrix to Vector Form

Converts 5×5 matrix back to 9-vector:

```python
mat = SE23.wedge(xi)
xi_recovered = SE23.vee(mat)    # Returns ℝ⁹ vector

# Roundtrip error
error = np.linalg.norm(xi - xi_recovered)  # ~0
```

## Group Element Operations

```python
# Create SE(2,3) element
T = SE23.identity()
T = SE23.exp(xi)

# Extract components
R = T.R()               # Rotation SO(3)
p = T.x()               # Position ∈ ℝ³ (returns R3 object)
v = T.w()               # Velocity ∈ ℝ³ (returns R3 object)

# Convert R3 to numpy array
p_vec = p.as_vector()   # Returns (3,) array
v_vec = v.as_vector()   # Returns (3,) array

# Group operations
T1 = SE23.exp(xi1)
T2 = SE23.exp(xi2)
T_product = T1 * T2     # Group multiplication
T_inv = T.inv()         # Inverse

# Adjoint matrix (9×9)
Ad = T.Adjoint()        # Returns (9, 9) array
xi_transformed = Ad @ xi
```

## Practical Example: SE(2,3) ⋉ se(2,3)

```python
from pylie import SE23
import numpy as np

class State:
    """State on SE(2,3) ⋉ se(2,3)"""
    
    def __init__(self):
        self.T = SE23.identity()        # SE(2,3) element
        self.b = np.zeros(9)            # se(2,3) bias (9-vector)
    
    def compose(self, other):
        """Semi-direct product: (T2, b2) ∘ (T1, b1)"""
        result = State()
        
        # Group part
        result.T = self.T * other.T
        
        # Bias part: b2 + Ad_{T2^{-1}}[b1]
        T_inv = self.T.inv()
        Ad_inv = T_inv.Adjoint()
        result.b = self.b + Ad_inv @ other.b
        
        return result
    
    def inverse(self):
        """(T, b)^{-1} = (T^{-1}, -Ad_{T^{-1}}[b])"""
        result = State()
        result.T = self.T.inv()
        Ad_inv = result.T.Adjoint()
        result.b = -Ad_inv @ self.b
        return result

# Usage
s1 = State()
s2 = State()
s1.b = np.random.randn(9) * 0.1

# Composition
s_composed = s1.compose(s2)

# Inverse
s_inv = s1.inverse()
```

## Lie Bracket in se(2,3)

The Lie bracket [·,·] in se(2,3) can be computed via matrix commutator:

```python
def lie_bracket(u, v):
    """Compute [u, v] in se(2,3)"""
    # Convert to matrix form (5×5)
    u_mat = SE23.wedge(u)
    v_mat = SE23.wedge(v)
    
    # Commutator: [u,v] = uv - vu
    bracket_mat = u_mat @ v_mat - v_mat @ u_mat
    
    # Convert back to vector
    return SE23.vee(bracket_mat)

# Usage
xi1 = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0])
xi2 = np.array([0, 0.1, 0, 0, 0, 0, 0, 0, 0])
bracket = lie_bracket(xi1, xi2)  # Returns (9,)
```

## Adjoint Action

The adjoint action Ad_T of a group element T on an algebra element v:

```python
T = SE23.exp(np.random.randn(9))
v = np.random.randn(9)

# Ad_T[v] transforms v by the group element
Ad_matrix = T.Adjoint()             # 9×9 matrix
v_transformed = Ad_matrix @ v       # Results in (9,)

# Properties:
# - Ad_{T1·T2} = Ad_{T1} · Ad_{T2}
# - Ad_{T^{-1}} = (Ad_T)^{-1}
# - For T = identity: Ad_T = I₉
```

## Computing with se(2,3) Vectors

```python
# Component indexing
xi = np.random.randn(9)
omega = xi[0:3]         # Rotational: [ω_x, ω_y, ω_z]
p_dot = xi[3:6]         # Position derivative: [p_x, p_y, p_z]
accel = xi[6:9]         # Acceleration: [a_x, a_y, a_z]

# Reconstruct
xi_new = np.concatenate([omega, p_dot, accel])

# Matrix form for visualization
mat = SE23.wedge(xi)    # 5×5 matrix
# Display: shows how algebra elements are represented
```

## Key Properties

| Property | Expression | Code |
|----------|------------|------|
| **Roundtrip** | wedge/vee | `SE23.vee(SE23.wedge(xi)) == xi` |
| **Exp/Log** | inverse maps | `SE23.exp(T.log()) == T` |
| **Adjoint** | group action | `Ad_T @ v` gives 9-vector |
| **Composition** | group mult | `T1 * T2` gives SE23 element |
| **Inverse** | group inverse | `T.inv()` returns SE23 element |

## Common Patterns

### Initialize from Measurements
```python
gyro = np.array([0.01, 0.02, 0.03])      # 3-vec
velocity = np.array([1.0, 0.0, 0.0])     # 3-vec  
accel = np.array([0.0, 0.0, 9.81])       # 3-vec

# Construct se(2,3) element
xi = np.concatenate([gyro, velocity, accel])  # 9-vec

# Propagate: T_{k+1} = T_k * exp(ξ·dt)
T_increment = SE23.exp(xi * dt)
T_new = T_old * T_increment
```

### Extract Pose Components
```python
T = SE23.exp(xi)

R_matrix = T.R().as_matrix()         # 3×3 rotation
p_vector = T.x().as_vector()         # (3,) position
v_vector = T.w().as_vector()         # (3,) velocity
```

### Bias Transformation
```python
# Bias lives in se(2,3): b ∈ ℝ⁹
b_old = np.random.randn(9)

# Transform by adjoint: Ad_T[b]
Ad_T = T.Adjoint()
b_transformed = Ad_T @ b_old

# Inverse adjoint for semi-direct product
Ad_T_inv = T.inv().Adjoint()
b_adj_inv = Ad_T_inv @ b_old
```

## Debugging Tips

1. **Check dimensions**:
   ```python
   assert xi.shape == (9,), f"Expected (9,), got {xi.shape}"
   assert mat.shape == (5, 5), f"Expected (5, 5), got {mat.shape}"
   ```

2. **Verify roundtrips**:
   ```python
   error_wedge_vee = np.linalg.norm(SE23.vee(SE23.wedge(xi)) - xi)
   error_exp_log = np.linalg.norm(SE23.exp(T.log()).log() - T.log())
   assert error_wedge_vee < 1e-10
   assert error_exp_log < 1e-10
   ```

3. **Check adjoint properties**:
   ```python
   T_inv = T.inv()
   Ad_T = T.Adjoint()
   Ad_T_inv = T_inv.Adjoint()
   I9 = Ad_T @ Ad_T_inv  # Should be identity
   assert np.allclose(I9, np.eye(9))
   ```

## References

- **PyLie Documentation**: https://github.com/pylie/pylie
- **SE(2,3) Theory**: Fornasier et al., ICRA 2022
- **Lie Group Basics**: Hall, "Lie Groups, Lie Algebras, and Representations"

