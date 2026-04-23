# Comprehensive Implementation Plan: Correct Equivariant Filter

## Overview

The current EqF implementation (eqf_filter.py) deviates significantly from the paper's formulation. This plan outlines the systematic rewrite required to properly implement the equivariant filter as described in Fornasier et al. (2022).

**Scope**: Full filter rewrite + validation against paper experiments

---

## Phase 1: Foundation - Data Structures and Lie Algebra Operations

### 1.1 Create Proper State Representation

**File**: `eqf_core/state.py`

Current issue: State is SE(2,3) + additive bias vector
Required: SE(2,3) ⋉ se(2,3) state with proper group operations

```python
class FilterStateEqF:
    """Proper state on SE(2,3) ⋉ se(2,3)"""
    
    # Main state
    A: np.ndarray          # SE(2,3) extended pose (4×4 matrix form)
    a: np.ndarray          # se(2,3) bias state (3×3 block form or 6-vec)
    
    # Covariance
    Sigma: np.ndarray      # 18×18 Riccati matrix
    
    # Dimensions
    dim_pose = 9           # 3 for SO(3), 3 for position, 3 for velocity
    dim_bias = 9           # 3 gyro, 3 velocity, 3 accel
    dim_total = 18
```

**Key operations to implement**:
- `.to_matrix()` / `.from_matrix()` - Convert A to/from 4×4 form
- `.compose(other)` - Semi-direct product multiplication: `(B, b) * (A, a)`
- `.inverse()` - Semi-direct product inverse
- `.adjoint(v)` - `Ad_A[v]` for se(2,3) elements
- `.adjoint_matrix()` - Matrix form of `Ad_A` (9×9)

### 1.2 Lie Algebra Helpers

**File**: `eqf_core/lie_algebra.py`

```python
def wedge_se23(v: np.ndarray) -> np.ndarray:
    """ℝ^6 → se(2,3): Map 6-vector to Lie algebra"""
    # v = [ω, v_dot] where ω ∈ ℝ^3, v_dot ∈ ℝ^3
    # Returns 3×3 block form used in paper
    
def vee_se23(X: np.ndarray) -> np.ndarray:
    """se(2,3) → ℝ^6: Inverse map"""
    
def lie_bracket(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """[X, Y] = XY - YX in se(2,3)"""
    
def adjoint_vec(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute Ad_A[v] for v ∈ se(2,3) (9-vector form)"""
    # Returns 9-vector result
    # Uses 9×9 matrix form of adjoint
    
def adjoint_bracket(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute ad_v[w] = [v, w] in Lie algebra form"""
    # Returns ad^∨_v (9×9 matrix) or direct computation
```

**Tests needed**:
- Verify group axioms (associativity, inverses)
- Verify adjoint properties
- Numerical stability checks

---

## Phase 2: System Dynamics - Correct Lift and Propagation

### 2.1 Implement Equivariant Lift (Theorem 5.1)

**File**: `eqf_core/lift.py`

```python
def compute_lift(
    state: FilterStateEqF,
    gyro: np.ndarray,       # Iω (3-vec)
    accel: np.ndarray,      # Ia (3-vec)
    virtual_inputs: dict    # ν, τω, τν, τa
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Λ_1 and Λ_2 (Equations 12-13)
    
    Returns:
        Λ_1: se(2,3) element for pose dynamics
        Λ_2: se(2,3) element for bias dynamics
    """
    
    # Extract state
    A = state.A              # SE(2,3)
    a = state.a              # se(2,3) bias
    
    # Input components
    w = np.concatenate([gyro, virtual_inputs['nu'], accel])  # 9-vec
    w_wedge = wedge_full_9d(w)  # 9×9 or structured form
    tau = np.concatenate([virtual_inputs['tau_w'], 
                         virtual_inputs['tau_nu'], 
                         virtual_inputs['tau_a']])  # 9-vec
    tau_wedge = wedge_full_9d(tau)
    
    g = np.array([0, 0, 9.81])  # gravity in world frame
    g_wedge = wedge_se23(g)     # se(2,3) form
    
    # Λ_1: (w∧ - b∧) + Ad_{T^(-1)}[g∧] + T^(-1) f_1^0(T)
    Lambda_1 = (w_wedge - a)
    
    # Ad_{A^(-1)}[g∧]
    A_inv = state.inverse()
    g_transformed = adjoint_vec(A_inv.A, g_wedge)
    Lambda_1 = Lambda_1 + g_transformed
    
    # T^(-1) f_1^0(T) - right-invariant drift
    f1_term = compute_drift_term(A.A)  # Eq. 5
    Lambda_1 = Lambda_1 + f1_term
    
    # Λ_2: ad_{b∧}[Λ_1] - τ∧
    Lambda_2 = lie_bracket(a, Lambda_1) - tau_wedge
    
    return Lambda_1, Lambda_2
```

**Critical formula** (Eq. 5 - right-invariant drift):
```python
def compute_drift_term(T: np.ndarray) -> np.ndarray:
    """
    f_1^0(T) = [ 0   v   0 ]
              [ 0   0   0 ]
    where v = velocity from T
    
    T is SE(2,3) in matrix form, extract v and create drift
    """
```

### 2.2 Implement Lifted System Propagation

**File**: `eqf_core/propagation.py`

```python
def propagate_lifted_state(
    state: FilterStateEqF,
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: float,
    virtual_inputs: dict = None  # Default to zero
) -> FilterStateEqF:
    """
    Propagate state using lifted system (Eqs. 14-15)
    
    Ȧ = A(w∧ + Ad_{A^(-1)}[â]) + g∧·A + f_1^0(A) + ∆_1·A
    â̇ = ad_{-â}[(Ad_A[w∧] + â) + g∧ + f_1^0(A)] - Ad_A[τ∧] + ∆_2 + ad_{∆_1}[â]
    """
    
    if virtual_inputs is None:
        virtual_inputs = {
            'nu': np.zeros(3),
            'tau_w': np.zeros(3),
            'tau_nu': np.zeros(3),
            'tau_a': np.zeros(3)
        }
    
    Lambda_1, Lambda_2 = compute_lift(state, gyro, accel, virtual_inputs)
    
    # For propagation without innovation (∆ = 0):
    A_old = state.A
    a_old = state.a
    
    # A_dot = A(w∧ + Ad_{A^(-1)}[â]) + g∧·A + f_1^0(A)
    w_vec = np.concatenate([gyro, virtual_inputs['nu'], accel])
    w_wedge = wedge_full_9d(w_vec)
    
    bias_term = adjoint_vec(A_old.inverse().A, a_old)  # Ad_{A^(-1)}[â]
    A_dot = A_old.A @ (w_wedge + bias_term) + ... # Complete formula
    
    # Integrate using matrix exponential or RK45
    # A_new = A_old * exp(Lambda_1 * dt)
    A_new = A_old.compose(SE23.exp(Lambda_1 * dt))
    
    # â_dot = ad_{-â}[(...)] - ...
    a_dot = ...  # Complex - requires Lie bracket manipulations
    a_new = a_old + a_dot * dt  # First-order approximation
    
    # Copy covariance (no update in propagation phase)
    new_state = FilterStateEqF()
    new_state.A = A_new
    new_state.a = a_new
    new_state.Sigma = state.Sigma.copy()
    
    return new_state
```

---

## Phase 3: Error Dynamics and Jacobians

### 3.1 Implement Error Linearization

**File**: `eqf_core/error_dynamics.py`

```python
def compute_error_jacobian_A0t() -> np.ndarray:
    """
    Compute linearized error dynamics matrix A^0_t (Eq. 24)
    
    A^0_t = [ Υ              -I    ]
            [ 0    ad^∨_{w∧_0+Gg∧} ]
    
    where Υ = [ 0    0    0 ]
              [ 0    0    I ]
              [Gg∧  0    0 ]
    """
    g = np.array([0, 0, 9.81])
    Gg_wedge = skew(g)  # 3×3
    
    Upsilon = np.zeros((9, 9))
    Upsilon[0:3, 0:3] = 0           # attitude error - no direct feedback
    Upsilon[3:6, 3:6] = 0           # position error
    Upsilon[6:9, 6:9] = 0           # velocity error
    Upsilon[3:6, 6:9] = np.eye(3)   # velocity couples to position
    Upsilon[6:9, 0:3] = Gg_wedge    # gravity couples gravity to velocity error
    
    A0t_top_left = Upsilon
    A0t_top_right = -np.eye(9)
    
    # Bottom right: ad^∨_{w∧_0 + Gg∧}
    # This is time-varying depending on inputs/state
    # For filter design, use nominal values
    w0_nominal = np.zeros(6)  # Or use actual gyro/accel
    ad_bottom_right = compute_ad_matrix(w0_nominal, g)
    
    A0t = np.zeros((18, 18))
    A0t[0:9, 0:9] = A0t_top_left
    A0t[0:9, 9:18] = A0t_top_right
    A0t[9:18, 9:18] = ad_bottom_right
    
    return A0t

def compute_output_matrix_C0() -> np.ndarray:
    """
    Output matrix C^0 (Eq. 26) - constant!
    
    C^0 = [I  0]  ∈ ℝ^{9×18}
    
    Observes 9 pose states, not bias
    """
    C0 = np.zeros((9, 18))
    C0[0:9, 0:9] = np.eye(9)
    return C0
```

---

## Phase 4: Filter Update - Innovation and Measurement

### 4.1 Implement Innovation Term

**File**: `eqf_core/innovation.py`

```python
def compute_innovation(
    state_estimate: FilterStateEqF,
    measurement: np.ndarray,  # Measured extended pose
    output_noise_Q: np.ndarray  # 9×9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute innovation term ∆ = (∆_1, ∆_2) (Eq. 27c)
    
    ∆ = DE|_I φ(E)† · dε^(-1) · Σ · C^0T · Q^(-1) · δ(ρ_{X̂^(-1)}(y))
    
    Returns:
        Delta_1: correction for A dynamics
        Delta_2: correction for a dynamics
    """
    
    # Output residual: δ(ρ_{X̂^(-1)}(y))
    # ρ(X, y) = y·A (right action on measurement)
    # ρ_{X̂^(-1)}(y) = y·Â^(-1)
    
    X_hat_inv = state_estimate.inverse()
    measurement_residual_T = measurement @ X_hat_inv.A  # On SE(2,3)
    
    # Convert to Lie algebra
    delta_residual = log_se23(measurement_residual_T)  # ℝ^6 for first-order
    # or pad to ℝ^9 for full representation
    
    # Q^(-1) · δ(...)
    Q_inv = np.linalg.inv(output_noise_Q)
    residual_weighted = Q_inv @ delta_residual  # 9-vec
    
    # Kalman gain matrix: Σ · C^0T · Q^(-1) · C^0 · Σ + Q
    C0 = compute_output_matrix_C0()
    K = state_estimate.Sigma @ C0.T @ Q_inv  # Kalman gain (18×9)
    
    # Correction
    correction = K @ residual_weighted  # 18-vec
    
    # Split into ∆_1 and ∆_2
    Delta_1 = correction[0:9]   # correction for A
    Delta_2 = correction[9:18]  # correction for a
    
    return Delta_1, Delta_2
```

### 4.2 Implement State Update (Measurement Correction)

**File**: `eqf_core/update.py`

```python
def update_state(
    state: FilterStateEqF,
    Delta_1: np.ndarray,  # se(2,3) correction for pose
    Delta_2: np.ndarray   # se(2,3) correction for bias
) -> FilterStateEqF:
    """
    Update state using equivariant correction (Eqs. 27a-27b update terms)
    
    The innovation enters as:
      Â ← Â * exp(∆_1)    (on the group)
      â ← â + ∆_2         (in the algebra, then via Lie bracket coupling)
    """
    
    new_state = FilterStateEqF()
    
    # Pose correction via group exponential
    # Note: ∆_1 is in se(2,3), must be exponentiated on SE(2,3)
    Delta_1_wedge = wedge_se23_to_matrix(Delta_1)
    correction_group = SE23.exp(Delta_1_wedge)
    new_state.A = state.A.compose(correction_group)
    
    # Bias correction
    # The full equation includes coupling term: ad_{∆_1}[â]
    Delta_2_wedge = wedge_full_9d(Delta_2)
    coupling = lie_bracket(Delta_1_wedge, state.a)
    new_state.a = state.a + Delta_2_wedge + coupling
    
    # Covariance update via Riccati
    return new_state
```

---

## Phase 5: Covariance Propagation and Riccati Equation

### 5.1 Implement Riccati Dynamics

**File**: `eqf_core/covariance.py`

```python
def propagate_covariance(
    Sigma: np.ndarray,     # 18×18 Riccati matrix
    A0t: np.ndarray,       # 18×18 error dynamics (time-varying)
    P: np.ndarray,         # 18×18 process noise gain
    C0: np.ndarray,        # 9×18 output matrix
    Q_inv: np.ndarray,     # 9×9 inverse output noise
    dt: float
) -> np.ndarray:
    """
    Propagate covariance via Riccati equation (Eq. 27d)
    
    Σ̇ = A^0_t Σ + Σ (A^0_t)T + P - Σ C^0T Q^(-1) C^0 Σ
    """
    
    # Continuous-time Riccati
    term1 = A0t @ Sigma
    term2 = Sigma @ A0t.T
    term3 = P
    term4 = Sigma @ C0.T @ Q_inv @ C0 @ Sigma
    
    Sigma_dot = term1 + term2 + term3 - term4
    
    # Integrate (simple Euler for now, RK45 better)
    Sigma_new = Sigma + Sigma_dot * dt
    
    # Enforce symmetry (numerical stability)
    Sigma_new = (Sigma_new + Sigma_new.T) / 2
    
    return Sigma_new

def tune_process_noise(
    Q_nav: float = 1e-2,      # Process noise on navigation
    Q_bias: float = 1e-5       # Process noise on bias
) -> np.ndarray:
    """
    Build process noise matrix P (paper uses constant tuning)
    """
    P = np.zeros((18, 18))
    P[0:9, 0:9] = np.eye(9) * Q_nav
    P[9:18, 9:18] = np.eye(9) * Q_bias
    return P
```

---

## Phase 6: Integration and Filter Class

### 6.1 Main Filter Class

**File**: `eqf_core/filter.py`

```python
class EquivariantFilterEqF:
    """
    Proper Equivariant Filter implementation (Fornasier et al. 2022)
    """
    
    def __init__(self):
        self.state = FilterStateEqF()
        self.t_prev = None
        
        # Tuning (from paper or cross-validated)
        self.Q_nav = 1e-2
        self.Q_bias = 1e-5
        self.P = tune_process_noise(self.Q_nav, self.Q_bias)
        
        # Output noise (tune based on sensor specs)
        self.Q = np.eye(9) * (0.01**2)  # Measurement noise covariance
        
    def propagate(self, t: float, gyro: np.ndarray, accel: np.ndarray):
        """Propagate state (no measurement)"""
        if self.t_prev is None:
            self.t_prev = t
            return
        
        dt = t - self.t_prev
        if dt <= 0 or dt > 1.0:
            self.t_prev = t
            return
        
        # Virtual inputs (zero for propagation)
        virtual_inputs = {
            'nu': np.zeros(3),
            'tau_w': np.zeros(3),
            'tau_nu': np.zeros(3),
            'tau_a': np.zeros(3)
        }
        
        # State propagation
        self.state = propagate_lifted_state(
            self.state, gyro, accel, dt, virtual_inputs
        )
        
        # Covariance propagation
        A0t = compute_error_jacobian_A0t()  # Time-varying, ideally
        C0 = compute_output_matrix_C0()
        Q_inv = np.linalg.inv(self.Q)
        
        self.state.Sigma = propagate_covariance(
            self.state.Sigma, A0t, self.P, C0, Q_inv, dt
        )
        
        self.t_prev = t
    
    def correct_position(self, pos_NED: np.ndarray, R_pos: np.ndarray):
        """Correct using position measurement"""
        # Construct measurement (extended pose from position)
        y = construct_measurement_from_position(pos_NED, self.state)
        
        # Innovation
        Delta_1, Delta_2 = compute_innovation(self.state, y, self.Q)
        
        # Update state
        self.state = update_state(self.state, Delta_1, Delta_2)
```

---

## Phase 7: Validation and Testing

### 7.1 Unit Tests

**File**: `tests/test_eqf_implementation.py`

```python
def test_group_operations():
    """Verify semi-direct product group axioms"""
    
def test_equivariance_property():
    """Verify Theorem 4.3: equivariance under symmetry group"""
    
def test_lie_algebra_bracket():
    """Verify Lie bracket properties"""
    
def test_lift_properties():
    """Verify lift satisfies Theorem 5.1"""
    
def test_error_dynamics_linearization():
    """Verify error dynamics match paper Eqs. 24-26"""
    
def test_stability_analysis():
    """Verify Theorem 7.1 conditions (observability)"""
```

### 7.2 Integration Tests

```python
def test_monte_carlo_15_runs():
    """Replicate paper's first simulation (Table I)"""
    # Compare EqF vs MEKF on 15 Monte-Carlo runs
    # Check: transient and asymptotic RMSE
    # Target: EqF significantly better on bias estimation
    
def test_initialization_robustness():
    """Replicate paper's Figure 2: robustness to initialization"""
    # Vary initialization error, check convergence
    # EqF should converge; MEKF should diverge
    
def test_real_flight_data():
    """Run on NIMBUS24 UAV data (Table III)"""
    # Compare attitude, position, velocity RMSE
    # Target: ~10% improvement over MEKF
```

---

## Implementation Order (Critical Path)

1. **Phase 1**: Lie algebra infrastructure (2-3 days)
   - Block pylie dependency issues early
   - Test group operations thoroughly
   
2. **Phase 2.1**: Implement lift (2 days)
   - Verify Λ_1, Λ_2 formulas
   - Test with known trajectories
   
3. **Phase 3.1**: Error linearization (1 day)
   - Verify A^0_t structure
   - Confirm observable pair (A^0_t, C^0)
   
4. **Phase 4**: Innovation and update (2 days)
   - Implement measurement model
   - Test correction application
   
5. **Phase 5**: Covariance (1 day)
   - Riccati integration
   - Numerical stability
   
6. **Phase 6**: Filter integration (1 day)
   - Main filter class
   - CSV I/O (keep from existing code)
   
7. **Phase 7**: Testing (3-5 days)
   - Unit tests throughout
   - Integration with real data
   - Convergence validation

**Total estimated time**: 1.5-2 weeks for correct implementation

---

## Known Pitfalls to Avoid

1. **Lie algebra vs. matrix form**: Maintain clear distinction between se(2,3) elements and their 3×3 or 6×6 matrix representations
2. **Left vs. right actions**: Paper uses right group actions consistently
3. **Adjoint direction**: Ad_A[·] (forward) vs Ad_{A^(-1)}[·] (inverse)
4. **Lie bracket properties**: Not commutative, requires careful formula transcription
5. **Covariance symmetry**: Must be enforced after numeric integration
6. **Gravity frame**: Must be in world frame (G frame), not body frame

---

## Success Criteria

✓ Unit tests pass (group axioms, equivariance)
✓ Monte Carlo 15-run comparison: EqF >> MEKF on bias
✓ Initialization robustness: EqF converges on full error range
✓ Real flight data: 10%+ improvement on attitude, position, velocity RMSE
✓ Stability: No divergence under moderate initialization error
✓ Performance: Matches or exceeds paper's Table I, II, III results
