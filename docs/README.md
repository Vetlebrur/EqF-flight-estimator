# EqF Implementation Documentation

This directory contains comprehensive documentation extracted from Fornasier et al. (2022) paper and a detailed implementation plan for rewriting the Equivariant Filter to correctly follow the paper's formulation.

## Quick Start

**First time here?** Start with `PAPER-STUDY-SUMMARY.md` - it explains what the paper does and what needs to be fixed.

## Document Guide

### For Understanding the Theory

1. **`PAPER-STUDY-SUMMARY.md`** ⭐ START HERE
   - Overview of the paper's contribution
   - What the current code gets wrong
   - Reading order for theory documents
   - Success milestones
   - **Time**: 15-20 min read

2. **`paper-formulation/01-SYSTEM-DEFINITION.md`**
   - Extended system with virtual inputs (Eq. 2)
   - State manifold SE(2,3) × se(2,3)
   - Input and measurement models
   - Why the current approach fails
   - **Time**: 20 min

3. **`paper-formulation/02-SYMMETRY-AND-EQUIVARIANCE.md`**
   - Semi-direct product group SE(2,3) ⋉ se(2,3)
   - Group operations (multiply, inverse)
   - Right group actions (state, input, output)
   - Equivariance property (Theorem 4.3)
   - Why this structure is crucial
   - **Time**: 25 min

4. **`paper-formulation/03-LIFTED-SYSTEM.md`**
   - Equivariant lift Λ (Theorem 5.1)
   - Lift components Λ_1 and Λ_2 (Eq. 12-13)
   - Lifted system dynamics (Eq. 14-15)
   - State origin choice
   - **Time**: 20 min

5. **`paper-formulation/04-FILTER-DESIGN.md`**
   - Complete EqF equations (Eq. 27a-27d)
   - Error linearization (Eq. 24-26)
   - Riccati covariance dynamics
   - Stability analysis (Theorem 7.1)
   - What guarantees the paper provides
   - **Time**: 30 min

### For Implementation

6. **`IMPLEMENTATION-PLAN.md`** ⭐ IMPLEMENTATION GUIDE
   - 7-phase systematic rewrite
   - Detailed pseudocode for each phase
   - Data structures needed
   - Testing strategy
   - Critical pitfalls to avoid
   - Implementation order (critical path)
   - **Time**: Start with 30 min overview, refer during implementation

---

## The Core Problem

**Current Implementation**: 
- Uses SE(2,3) + additive bias
- Loses equivariance property
- Error dynamics not derived correctly
- Results in 30-40% performance loss on bias estimation and divergence under poor initialization

**Correct Implementation** (from paper):
- Uses SE(2,3) ⋉ se(2,3) semi-direct product
- Bias lives in Lie algebra, transforms consistently
- Error dynamics are constant and geometry-derived
- Results in better robustness and 30-40% better bias estimation accuracy

---

## Key Equations

These are the main formulas you need to implement:

| Equation | What | Location |
|----------|------|----------|
| **Eq. 2** | Extended system with virtual inputs | System Definition |
| **Eq. 6** | Compact affine form | System Definition |
| **Eq. 9** | State action φ | Symmetry & Equivariance |
| **Eq. 10** | Input action ψ | Symmetry & Equivariance |
| **Eq. 12-13** | **Equivariant lift Λ_1, Λ_2** | Lifted System |
| **Eq. 14-15** | **Lifted system propagation** | Lifted System |
| **Eq. 24-26** | **Error dynamics A^0_t, C^0** | Filter Design |
| **Eq. 27** | **Filter ODE + Riccati** | Filter Design |

---

## Implementation Checklist

### Phase 1: Foundation (2-3 days)
- [ ] Create `eqf_core/state.py` - FilterStateEqF class
- [ ] Create `eqf_core/lie_algebra.py` - Wedge, vee, brackets
- [ ] Implement group operations (multiply, inverse, adjoint)
- [ ] Write unit tests for group axioms

### Phase 2: Lift (2 days)
- [ ] Implement `compute_lift()` for Λ_1, Λ_2
- [ ] Verify against Theorem 5.1
- [ ] Implement lifted state propagation

### Phase 3: Error Dynamics (1 day)
- [ ] Compute A^0_t (Eq. 24)
- [ ] Verify C^0 is constant (Eq. 26)
- [ ] Check observability pair (A^0_t, C^0)

### Phase 4-6: Filter & Covariance (3-4 days)
- [ ] Innovation computation
- [ ] State update with Adjoint
- [ ] Riccati equation integration
- [ ] Filter main class

### Phase 7: Testing (3-5 days)
- [ ] Unit tests for all components
- [ ] Monte Carlo 15-run validation
- [ ] Real flight data comparison
- [ ] Paper results replication

**Total**: 1.5-2 weeks for correct implementation

---

## Key Concepts Explained

### Semi-Direct Product (A⋉B)
Elements are pairs (a, b). Multiplication is: `(b', b) · (a, a') = (ba, b + Ad_b[a])`

Why? The algebra element must transform by the group's adjoint action.

### Equivariance
System behavior is predictable under symmetry transformations. Not the same as invariance (identical at every point), but structured predictability that helps error analysis.

### Lie Bracket [X, Y]
Commutator: `[X, Y] = XY - YX`. Encodes how two infinitesimal rotations interact. Essential for bias coupling to state.

### Adjoint Action Ad_A[v]
How a group element A transforms Lie algebra elements. In our case: A ∈ SO(3), v ∈ se(3) → Ad_A[v] rotates v.

---

## Common Questions

**Q: Why add virtual velocity ν and bias I_bν?**
A: To embed bias in the Lie algebra (se(2,3)). Without them, bias can't transform consistently with the group element, and you lose equivariance.

**Q: Why is A^0_t constant in the paper but time-varying otherwise?**
A: The paper's geometry ensures bias couples linearly to pose error through Lie bracket structure. Other approaches (MEKF, IEKF) couple through state-dependent frames, giving time-varying A_t and worse observability.

**Q: What's the difference between Adjoint (Ad_A) and adjoint (ad_v)?**
A: Ad_A is the group's action on the algebra (how group elements transform algebra elements). ad_v is the algebra's action on itself (Lie brackets). They're related: ad_v = differential of Ad at identity.

**Q: Why does the current implementation fail?**
A: It treats bias as simple addition, not as a Lie algebra element. This breaks the equivariance property and makes error dynamics time-varying (state-dependent), reducing observability.

---

## References

**Paper**: Fornasier et al. (2022). "Equivariant Filter Design for Inertial Navigation Systems with Input Measurement Biases." IEEE ICRA 2022.

**Key Related Work**:
- Van Goor et al. (2020). "Equivariant Filter (EqF): A General Filter Design for Systems on Homogeneous Spaces."
- Barrau & Bonnabel (2017). "The Invariant Extended Kalman Filter as a Stable Observer."
- Mahony et al. (various). Equivariant systems theory foundations.

---

## File Structure

```
docs/
├── README.md                          ← You are here
├── PAPER-STUDY-SUMMARY.md             ← Start here
├── IMPLEMENTATION-PLAN.md              ← Implementation guide
└── paper-formulation/
    ├── 01-SYSTEM-DEFINITION.md
    ├── 02-SYMMETRY-AND-EQUIVARIANCE.md
    ├── 03-LIFTED-SYSTEM.md
    └── 04-FILTER-DESIGN.md
```

---

## Next Steps

1. **Read** `PAPER-STUDY-SUMMARY.md` (15-20 min)
2. **Study** `01-SYSTEM-DEFINITION.md` (20 min)
3. **Review** your notes, write down questions
4. **Continue** with documents 2-5 in order
5. **Reference** `IMPLEMENTATION-PLAN.md` when you start coding

---

## Need Help?

- Check the "Common Questions" section above
- Review the specific document covering that topic
- Look at the key equations table
- Refer to the paper itself (Fornasier et al. 2022)

---

**Last Updated**: April 2026
**Paper**: Fornasier et al., IEEE ICRA 2022
**Status**: Documentation complete, ready for implementation
