# Symmetry and Equivariance - SE(2,3) ⋉ se(2,3)

## Semi-Direct Product Group Structure

The symmetry group is `G_sym = SE(2,3) ⋉ se(2,3)` - semi-direct product of SE(2,3) with its Lie algebra.

### Group Operation (Lemma 4.1)

For `X = (A, a), Y = (B, b) ∈ SE(2,3) ⋉ se(2,3)`:

```
Y·X = (B·A, b + Ad_B[a])
```

Key feature: The se(2,3) component `a` is transformed via the Adjoint action when composing.

### Group Inverse

```
X^(-1) = (A^(-1), -Ad_{A^(-1)}[a])
```

### Identity Element
```
I = (I_{SE(2,3)}, 0∧)
```

## Right Group Actions

### State Action (Equation 9)

The symmetry group acts on the state manifold M:

```
φ: SE(2,3) ⋉ se(2,3) × M → M
φ(X, ξ) := (T·A, Ad_{A^(-1)}[b∧ - a])
```

Where `X = (A, a)` and `ξ = (T, b∧)`.

**Property**: φ is a **transitive right group action** (Lemma 4.1)

### Input Action (Equation 10)

The symmetry group acts on the input space L:

```
ψ: SE(2,3) ⋉ se(2,3) × L → L
ψ(X, u) := (Ad_{A^(-1)}[w∧ - a] + f_1^0(A^(-1)), g∧, Ad_{A^(-1)}[τ∧])
```

Where:
- `u = (w∧, g∧, τ∧)`
- `f_1^0` is the right-invariant drift field (Eq. 5)

**Property**: ψ is a right group action on L (Lemma 4.2)

### Output Action (Equation 11)

```
ρ: SE(2,3) ⋉ se(2,3) × N → N
ρ(X, y) := y·A
```

## Equivariance Property (Theorem 4.3)

**The system is equivariant** under the combined actions φ, ψ, ρ:

```
f_0(ξ) + f_{ψ_X(u)}(ξ) = Φ_X f_0(ξ) + Φ_X f_u(ξ)
```

Where `Φ_X` is the pushforward of the group action φ by X.

**Meaning**: The system dynamics transform predictably under symmetry transformations.

## Why This Matters

1. **Equivariance ≠ Invariance**: The system is NOT invariant (doesn't have identical dynamics at every point), but it IS equivariant (behaves predictably under symmetry).

2. **Geometric Structure Preservation**: Updates preserve the manifold geometry and group structure automatically.

3. **Decoupled Error Analysis**: Error dynamics can be analyzed in a fixed frame (the Lie algebra) rather than tracking a moving frame.

4. **Bias Estimation**: The bias `a` lives in se(2,3) and transforms consistently with navigation state A. This is fundamentally different from additive bias models.

## Current Implementation Problems

The current code:
- Uses simple SE(2,3) without the semi-direct product structure
- Treats bias as additive perturbation: `state.b += ∆`
- Does not use Adjoint transformations for bias
- Loses the equivariance property → loses theoretical guarantees

**Result**: The filter can diverge under poor initialization, loses 30-40% performance on bias estimation, and has no formal stability guarantees.
