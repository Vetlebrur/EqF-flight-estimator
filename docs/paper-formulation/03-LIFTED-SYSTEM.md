# Lifted System and Filter Design

## Equivariant Lift (Theorem 5.1)

To design an EqF, we need to "lift" the system kinematics onto the Lie algebra of the symmetry group.

The lift Λ must be a smooth map:
```
Λ: M × L → se(2,3) ⋉ se(2,3)
```

such that:
```
dφ_ξ(I)[Λ(ξ,u)] = f_0(ξ) + f_u(ξ)
```

### Lift Components (Equations 12-13)

**First component** (translational/velocity part):
```
Λ_1(ξ,u) := (w∧ - b∧) + Ad_{T^(-1)}[g∧] + T^(-1) f_1^0(T)

where:
  (w∧ - b∧)           = "measured" input minus bias
  Ad_{T^(-1)}[g∧]     = gravity transformed to body frame via T^(-1)
  T^(-1) f_1^0(T)     = correction due to right-invariant drift
```

**Second component** (bias dynamics):
```
Λ_2(ξ,u) := ad_{b∧}[Λ_1(ξ,u)] - τ∧

where:
  ad_{b∧}[·]          = Lie bracket, encodes bias coupling to state
  τ∧                  = virtual bias input
```

**Key**: The bias `b∧` appears in the Lie bracket (Lie algebra operation), not as simple addition.

## Lifted System Dynamics (Equations 14-15)

Given the lift, the system on the symmetry group becomes:

```
Ȧ = A(w∧ + Ad_{A^(-1)}[â]) + g∧·A + f_1^0(A)

â̇ = Ad_A[ad_{(-Ad_{A^(-1)}[â])}[(w∧ + Ad_{A^(-1)}[â]) 
                                    + Ad_{A^(-1)}[g∧] 
                                    + A^(-1)f_1^0(A)] - τ∧]
```

Where:
- `X = (A, â) ∈ SE(2,3) ⋉ se(2,3)` is the lifted state
- `A ∈ SE(2,3)` is the extended pose
- `â ∈ se(2,3)` is the lifted bias

### State Origin

The **state origin** is chosen as:
```
ξ_0 = (I, 0∧)   (identity extended pose, zero bias)
```

This allows global parametrization of M via:
```
ξ = φ_{ξ_0}(X)  for X ∈ SE(2,3) ⋉ se(2,3)
```

## Lifted System Structure

**Critical observation**: The lifted system is:

1. **Linear in inputs** (`w∧`, `g∧`, `τ∧`)
2. **Bilinear in state and inputs** (Lie brackets couple state and inputs)
3. **Right-invariant vector fields** (structure respects group geometry)

This is fundamentally different from the original nonlinear system, making it amenable to equivariant filter design.

## Current Implementation Missing

The current code uses:
```python
self.state.X = self.state.X * SE23.exp(lift[0:9] * dt)
self.state.b = self.state.b + lift[9:18] * dt
```

But `lift` is computed as a simple linear approximation. The paper requires:
1. Proper `Λ_1` and `Λ_2` as defined above
2. Correct Lie bracket operations for bias coupling
3. Proper Adjoint transformations
4. Equivariant lift properties verified

This is why the current filter breaks under initialization error and poor bias estimation.
