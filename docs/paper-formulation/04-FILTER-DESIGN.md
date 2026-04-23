# Equivariant Filter Design and Implementation

## Equivariant Filter ODE (Equations 27a-27d)

The full EqF evolution is:

```
Â̇ = Â(Iw∧ + Ad_{Â^(-1)}[â]) + Gg∧·Â + f_1^0(Â) + ∆_1·Â        (27a)

â̇ = ad_{-â}[(Ad_Â[Iw∧] + â) + Gg∧ + f_1^0(Â)] 
     - Ad_Â[τ∧] + ∆_2 + ad_{∆_1}[â]                            (27b)

∆ = [∆_1, ∆_2] = DE|_I φ_{ξ_0}(E)† dε^(-1) Σ C^0T Q^(-1) δ(ρ_{X̂^(-1)}(y))  (27c)

Σ̇ = A^0_t Σ + Σ(A^0_t)T + P - Σ C^0T Q^(-1) C^0 Σ                (27d)
```

Where:
- `X̂ = (Â, â)` is the filter state estimate
- `∆ = (∆_1, ∆_2)` is the innovation term (measurement-driven correction)
- `Σ ∈ ℝ^{18×18}` is the Riccati matrix (covariance)
- `P ∈ ℝ^{18×18}` is the process noise gain matrix
- `Q ∈ ℝ^{9×9}` is the output noise gain matrix

## Error Dynamics (Equations 24-26)

The **linearized error dynamics** are critical for understanding filter behavior:

```
A^0_t = [ Υ        -I   ]  ∈ ℝ^{18×18}
        [ 0    ad∨_{w∧_0+Gg∧}]

where:
       
Υ = [ 0   0   0  ]
    [ 0   0   I  ]
    [Gg∧ 0   0  ]

C^0 = [I  0] ∈ ℝ^{9×18}
```

### What This Means

1. **Output matrix C^0 is constant** (9 states observed, 9 unobserved bias states)
2. **A^0_t has block structure**:
   - Top-left (3×3): Zero (no direct attitude error feedback)
   - Top-right (3×9): -I (velocity errors couple back to attitude via integration)
   - Bottom-left (9×3): Gg^∧ (gravity induces bias error changes)
   - Bottom-right (9×9): ad^∨_{w∧_0+Gg∧} (body-frame time variation)

3. **Key difference from MEKF/IEKF**: The (1,2) block is constant and geometry-derived, not time-varying. This is why bias estimation is improved.

## Innovation Term

The innovation (correction) is computed as:

```
∆ = DE|_I φ_{ξ_0}(E)† · dε^(-1) · Σ · C^0T · Q^(-1) · δ(ρ_{X̂^(-1)}(y))
```

Breaking this down:
1. `δ(ρ_{X̂^(-1)}(y))` - Output residual (measurement error in Lie algebra)
2. `Q^(-1)` - Output noise weighting
3. `Σ · C^0T` - Kalman gain numerator (covariance-weighted observation)
4. `dε^(-1)` - Inverse of chart transition map
5. `DE|_I φ` - Differential of the state action at identity

This ensures the innovation respects the group geometry.

## Riccati Equation

The covariance evolution:

```
Σ̇ = A^0_t Σ + Σ(A^0_t)T + P - Σ C^0T Q^(-1) C^0 Σ
```

**Critical property**: Σ must remain:
- Symmetric positive definite
- Representative of tangent-space error covariance

The last term `-Σ C^0T Q^(-1) C^0 Σ` is the measurement update reducing uncertainty.

## Stability Analysis (Theorem 7.1)

**Local exponential stability guaranteed if**:
1. The pair (A^0_t, C^0) is uniformly observable
2. Initial error is in the local neighborhood of identity

**Why uniform observability?**
- We observe 9 states (extended pose via yaw,pitch,roll, position, velocity)
- We have 18 error states (9 pose + 9 bias)
- Bias must be observable via coupling through A^0_t

## Current Implementation Gaps

The current code:
1. **Missing**: Proper `Λ_1` and `Λ_2` computation
2. **Wrong**: Uses simple matrix exponential on dummy A_t (Eq. 142)
3. **Wrong**: Bias update is additive, not via Adjoint (line 232)
4. **Wrong**: Innovation is Euclidean residual, not equivariant (line 228)
5. **Missing**: Proper error linearization with Lie brackets
6. **Wrong**: C^0 isn't constant in the implementation

All of these contribute to the 30-40% bias estimation error and divergence under poor initialization.
