# TGEqF (Tangent-Group Equivariant Filter) Structure

## Overview

The TGEqF class implements the Equivariant Filter on SE(2,3) ⋉ se(2,3), with methods organized across separate modules (like C/C++ headers and source files).

## Architecture

```
tg_eqf.py
├── TGEqF (base class)
│   ├── State representation (T, b, P, time)
│   ├── Algebra operations (wedge, vee, exp, log)
│   ├── Group operations (compose, inverse)
│   └── Pose/bias accessors

eqf_predict.py
├── Implements prediction methods on TGEqF:
│   ├── imu_predict()
│   ├── predict_bias_zero_order()
│   └── predict_bias_random_walk()

eqf_update.py
├── Implements update methods on TGEqF:
│   ├── magnetometer_update()
│   ├── gnss_update()
│   └── barometer_update()
```

## Why This Structure?

Like C/C++:
- **tg_eqf.py** = Header file (interface + basic operations)
- **eqf_predict.py** = Source file (predict implementations)
- **eqf_update.py** = Source file (update implementations)

Benefits:
- Separation of concerns: each file has a specific purpose
- Clear module organization: navigation, measurement updates are separate
- Easy to extend: add new sensors without modifying TGEqF
- Methods still belong to single TGEqF class (via monkey-patching)

## Usage

```python
from tg_eqf import TGEqF
import eqf_predict  # Attaches predict methods
import eqf_update   # Attaches update methods
import numpy as np

# Initialize
f = TGEqF()

# Predict (IMU)
gyro = np.array([0.01, 0.01, 0.02])
accel = np.array([0.0, 0.0, 9.81])
f.imu_predict(dt=0.01, accel=accel, gyro=gyro)

# Update (Sensors)
f.magnetometer_update(mag=np.array([1, 0, 0]))
f.gnss_update(lat=37.0, lon=122.0, alt=100.0)
f.barometer_update(pressure=101325.0)

# Algebra operations
xi = np.random.randn(9)
T = TGEqF.exp_map(xi)      # exp: se(2,3) -> SE(2,3)
xi_recovered = T.log()      # log: SE(2,3) -> se(2,3)

# Access state
R = f.get_rotation()        # Attitude (3x3)
p = f.get_position()        # Position (3,)
v = f.get_velocity()        # Velocity (3,)
b_gyro = f.get_bias_gyro()  # Gyro bias (3,)
```

## State Representation

**SE(2,3) Pose**: Represents extended pose with position, velocity, and attitude
```
T ∈ SE(2,3) encodes:
  - Rotation: R ∈ SO(3)
  - Position: p ∈ ℝ³
  - Velocity: v ∈ ℝ³
```

**Bias Vector**: 9-dimensional element of se(2,3) algebra
```
b ∈ ℝ⁹ contains:
  - Gyroscope bias: b_gyro (indices 0-2)
  - Velocity bias: b_velocity (indices 3-5)
  - Accelerometer bias: b_accel (indices 6-8)
```

**Covariance**: 18×18 matrix (9 DOF for pose tangent space + 9 for bias)
```
P ∈ ℝ^{18×18} where:
  - Upper-left 9×9: pose covariance
  - Lower-right 9×9: bias covariance
  - Off-diagonals: cross-correlations
```

## Lie Algebra Operations

The algebra se(2,3) is represented as 9-dimensional vectors using PyLie's conversions:

| Operation | Function | Maps |
|-----------|----------|------|
| Wedge | `TGEqF.wedge(v)` | ℝ⁹ → ℝ^{5×5} (matrix form) |
| Vee | `TGEqF.vee(mat)` | ℝ^{5×5} → ℝ⁹ (vector form) |
| Exponential | `TGEqF.exp_map(v)` | ℝ⁹ → SE(2,3) |
| Logarithmic | `T.log()` | SE(2,3) → ℝ⁹ |
| Adjoint | `T.Adjoint()` | SE(2,3) → ℝ^{9×9} (linear action matrix) |

## Group Operations

### Semi-Direct Product Composition
```
(T₂, b₂) ∘ (T₁, b₁) = (T₂·T₁, b₂ + Ad_{T₂⁻¹}[b₁])

Implementation:
  result = state1.compose(state2)
  # Composes two states on SE(2,3) ⋉ se(2,3)
```

### Inverse
```
(T, b)⁻¹ = (T⁻¹, -Ad_{T⁻¹}[b])

Implementation:
  result = state.inverse()
  # Returns inverse state
```

## Prediction Interface

Implements IMU-based state propagation.

```python
def imu_predict(dt, accel, gyro, accel_cov=None, gyro_cov=None):
    """Propagate state using accelerometer and gyroscope."""
    # Updates self.T and self.b
```

## Update Interface

Implements measurement-based state correction.

```python
def magnetometer_update(mag, mag_reference=np.array([1,0,0]), R_mag=None):
    """Update using heading (magnetometer)."""

def gnss_update(lat, lon, alt=None, R_gnss=None):
    """Update using position (GPS/GNSS)."""

def barometer_update(pressure, R_baro=None):
    """Update using altitude (barometer)."""
```

## PyLie Integration

Uses PyLie for vetted Lie group operations:

```python
from pylie import SE23, SO3

# Group elements
T = SE23.identity()
T.R()           # Rotation (SO3)
T.x()           # Position (R3)
T.w()           # Velocity (R3)
T.inv()         # Inverse
T.Adjoint()     # 9×9 adjoint matrix

# Algebra operations
SE23.wedge(v)   # Vector to matrix (5×5)
SE23.vee(mat)   # Matrix to vector
SE23.exp(xi)    # Exponential map
T.log()         # Logarithmic map
```

## Example Usage

See [example_tg_eqf.py](../example_tg_eqf.py) for complete example showing:
- Initialization
- IMU prediction
- Measurement updates
- Algebra operations
- Group composition

## Next Steps

1. **Implement EqF-specific methods**:
   - Replace `raise NotImplementedError` in predict/update methods
   - Add proper Kalman gain computation
   - Implement Riccati equation integration

2. **Add support methods**:
   - Error state computation (current vs. estimated)
   - Covariance propagation
   - Innovation computation

3. **Testing**:
   - Unit tests for group operations
   - Monte Carlo validation
   - Comparison with paper results

## References

- **Paper**: Fornasier et al., IEEE ICRA 2022
- **PyLie**: Python Lie groups library
- **Lie Theory**: Semi-direct products, exponential maps, adjoint actions

