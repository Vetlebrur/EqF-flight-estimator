# System Definition - Equivariant Filter for Biased INS

## Extended System with Virtual Inputs (Equations 2a-2f)

The original system (Eq. 1) is augmented with virtual inputs to achieve equivariance:

```
Ğ_IR = Ğ_IR (Iω - Ibω)∧          [Attitude kinematics]
Ğp_I = Ğ_IR (ν - Ibν) + Ğv_I    [Position kinematics with virtual velocity]
Ğv_I = Ğ_IR (Ia - Iba) + Ğg      [Velocity dynamics]
I_bω̇ = τω                        [Gyro bias dynamics]
I_bν̇ = τν                        [Virtual velocity bias dynamics]
I_bȧ = τa                         [Accel bias dynamics]
```

**Key Addition**: Virtual velocity input `ν` and associated virtual bias `I_bν`
- These are non-physical but enable the equivariant structure
- Setting them to zero recovers the original system

## State Manifold (Equation 3)

The biased inertial navigation system is posed on:
```
ξ = (G_IT, I_b∧) ∈ M := SE(2,3) × se(2,3)
```

Where:
- `G_IT = (G_IR, G_p_I, G_v_I) ∈ SE(2,3)` is the extended pose (rotation, position, velocity)
- `I_b∧ = (I_bω∧, I_bν∧, I_ba∧) ∈ se(2,3)` is the bias state in Lie algebra form
- SE(2,3) is the SE(2,3)-Torsor (special case of Lie group for velocity-inclusive kinematics)

## Compact Affine Form (Equation 6)

```
ξ̇ = f_0(ξ) + f_u(ξ)

where:
  f_0(ξ) = (f_1^0(G_IT) · G_IT, 0∧)              [Drift field - right invariant]
  f_u(ξ) = (G_IT(Iw∧ - I_b∧) + g∧·G_IT, τ∧)     [Control input effect]

with:
  f_1^0(G_IT) := [0      G_v_I  0]  ∈ ℝ(SE(2,3))
                  [0      0      0]
```

## Input Space

```
u = (Iw∧, g∧, τ∧) ∈ L ⊆ se(2,3) × se(2,3) × se(2,3)

where:
  Iw = (Iω, ν, Ia) ∈ ℝ^9  [virtual input vector]
```

## Measurement Model (Equation 7-8)

Configuration output (extended pose only):
```
h(ξ) = G_IT ∈ SE(2,3)
```

Real-world measurement with noise:
```
y = G_IT exp(n∧)    [Local noise]
or
y = exp(n∧) G_IT    [Global noise]
```

where `n∧` is Gaussian noise on the Lie algebra.

## Critical Distinction from Current Implementation

**Current code** treats bias as simple additive perturbation on SE(2,3).
**Paper formulation** embeds bias in the Lie algebra se(2,3) as a semi-direct product structure.

This changes:
1. How bias couples to navigation states
2. How errors are measured
3. How updates are applied
4. Covariance representation and evolution
