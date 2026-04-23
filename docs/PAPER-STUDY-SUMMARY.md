# Paper Study Summary and Next Steps

## Paper Reference
**Title**: Equivariant Filter Design for Inertial Navigation Systems with Input Measurement Biases
**Authors**: Alessandro Fornasier, Yonhon Ng, Robert Mahony, Stephan Weiss
**Year**: 2022, ICRA (IEEE International Conference on Robotics and Automation)
**Doi**: IEEE ICRA 2022

---

## Key Contributions of the Paper

1. **Novel Symmetry Group**: SE(2,3) ⋉ se(2,3) semi-direct product for biased INS
2. **Velocity Extension**: Virtual velocity inputs + associated velocity bias to achieve equivariance
3. **Equivariant Lift**: First equivariant lift for concurrent navigation state and bias estimation
4. **Geometric Bias Model**: Bias estimated in Lie algebra (group-theoretic), not Euclidean
5. **Stability Guarantees**: Local exponential stability with formal proof (Theorem 7.1)
6. **Empirical Results**: 
   - 30-40% improvement on bias estimation accuracy
   - Better robustness to initialization errors
   - Outperforms industry-standard MEKF

---

## Core Problem Solved

**Prior state-of-art (MEKF, IEKF)**:
- Attitude: geometric (SO(3))
- Position/velocity: Euclidean (ℝ^3 × ℝ^3)
- Bias: Euclidean additive (ℝ^9)
- Issue: **Bias breaks symmetry** → time-varying A_t matrix, poor bias estimation

**EqF approach**:
- Entire system on SE(2,3) with bias embedded in Lie algebra
- Achieved via extended system: adds virtual velocity input + virtual bias
- Result: **Constant (geometry-derived) A_t** → better bias observability

---

## Documentation Created

All files in `docs/paper-formulation/`:

### 1. System Definition (`01-SYSTEM-DEFINITION.md`)
- Original system (Eq. 1) vs. extended system (Eq. 2)
- State manifold M = SE(2,3) × se(2,3)
- Compact affine form
- Why virtual inputs are necessary

### 2. Symmetry and Equivariance (`02-SYMMETRY-AND-EQUIVARIANCE.md`)
- Semi-direct product group operations
- Right group actions (state, input, output)
- Equivariance property (Theorem 4.3)
- Why this matters for filter design

### 3. Lifted System (`03-LIFTED-SYSTEM.md`)
- Equivariant lift Λ (Theorem 5.1)
- Λ_1 and Λ_2 components (Equations 12-13)
- Lifted system dynamics on symmetry group (Equations 14-15)
- State origin choice

### 4. Filter Design (`04-FILTER-DESIGN.md`)
- Full EqF equations (Equations 27a-27d)
- Error linearization (Equations 24-26)
- Riccati covariance dynamics
- Stability analysis (Theorem 7.1)
- What the paper guarantees

### 5. Implementation Plan (`IMPLEMENTATION-PLAN.md`)
- 7-phase rewrite strategy
- Data structures needed (FilterStateEqF, group operations)
- Detailed pseudocode for each component
- Testing strategy
- Known pitfalls

---

## Current Implementation Issues

**Critical mismatches** between code and paper:

| Aspect | Paper | Current Code | Impact |
|--------|-------|--------------|--------|
| **State** | SE(2,3) ⋉ se(2,3) | SE(2,3) + additive bias | Loses equivariance |
| **A_t matrix** | Constant (Eq. 24) | Dummy placeholder (Eq. 142) | No proper error analysis |
| **B_t matrix** | Symmetry-derived | Identity (Eq. 146) | Wrong noise injection |
| **Lift** | Λ_1, Λ_2 with Lie brackets (Eq. 12-13) | Ad-hoc "continuous_lift" | Violates equivariance |
| **Innovation** | Equivariant (Eq. 27c) | Plain Euclidean residual | Breaks geometric structure |
| **Bias update** | Via Adjoint + Lie brackets | Simple addition (line 232) | Violates group geometry |
| **Covariance** | Constant C^0 (Eq. 26) | Time-varying implicitly | Loses observability structure |

**Result**: 30-40% performance loss on bias, divergence under poor init, no stability guarantees.

---

## What Needs to Happen

### Short-term (Understanding)
- [ ] Study each documentation file in order
- [ ] Work through Lie algebra operations (wedge, vee, brackets)
- [ ] Understand semi-direct product (group multiplication, inverse)
- [ ] Trace through Equations 12-13 step by step

### Medium-term (Implementation)
- [ ] Implement Phase 1: Lie algebra infrastructure + group operations
- [ ] Implement Phase 2: Equivariant lift (Λ_1, Λ_2)
- [ ] Implement Phase 3: Error dynamics (A^0_t, C^0)
- [ ] Implement Phases 4-6: Filter update + covariance
- [ ] Write unit tests for each component

### Validation
- [ ] Run against paper's experiment data (15 Monte Carlo runs)
- [ ] Compare vs. MEKF: match Figure 1 results
- [ ] Test initialization robustness: replicate Figure 2
- [ ] Real flight data: match Table III results

---

## Key Equations to Implement

**In order of implementation**:

1. **Equations 12-13**: Equivariant lift Λ_1, Λ_2
2. **Equations 14-15**: Lifted system propagation
3. **Equations 24-26**: Error dynamics A^0_t, C^0
4. **Equations 27a-27d**: Filter ODE + Riccati

---

## Math Prerequisites Needed

You'll need to be comfortable with:
- Lie groups (SO(3), SE(3), SE(2,3))
- Lie algebras (so(3), se(3), se(2,3))
- Wedge/vee operators (ℝ^n ↔ Lie algebra)
- Adjoint action (Ad_A, ad_v)
- Lie brackets [·,·]
- Matrix exponentials exp(·)
- Kalman filtering basics
- Equivariance property

The documentation explains all of these in context, but having linear algebra + group theory background helps.

---

## Recommended Reading Order

1. Start with `01-SYSTEM-DEFINITION.md` - understand the extended system
2. Read `02-SYMMETRY-AND-EQUIVARIANCE.md` - why we need semi-direct product
3. Study `03-LIFTED-SYSTEM.md` - how the lift works
4. Deep dive into `04-FILTER-DESIGN.md` - filter equations + error dynamics
5. Review `IMPLEMENTATION-PLAN.md` - implementation strategy
6. Refer back to paper sections as needed

---

## Success Milestones

### Milestone 1: Lie Algebra Infrastructure
- Group operations verified (axioms)
- Wedge/vee operators working
- Adjoint action correct

### Milestone 2: Lift Implementation
- Λ_1 computed correctly
- Λ_2 computed with proper Lie brackets
- Verify against Theorem 5.1

### Milestone 3: Error Dynamics
- A^0_t matches Equation 24
- C^0 constant as Equation 26
- Observability verified

### Milestone 4: Filter Integration
- All phases working together
- No divergence on test trajectories
- Covariance positive definite

### Milestone 5: Paper Replication
- Monte Carlo results match Table I
- Initialization robustness like Figure 2
- Real flight data like Table III

---

## Notes for Implementation

### Notation
The paper uses:
- `^∧`: vector → matrix (wedge, vee)
- `·` or juxtaposition: group multiplication
- `Ad_A[·]`: Adjoint action of group element
- `ad_v[·]`: Adjoint action of Lie algebra element
- `∨`: matrix → vector (inverse of wedge)

### Coordinate Frames
- `G`: Global (inertial/Earth) frame
- `I`: IMU (body) frame
- Superscripts denote frame: G_p = position in global frame

### State Components
- `T` or `IT`: extended pose (rotation R, position p, velocity v)
- `b` or `b∧`: bias (in Lie algebra form)
- Capital letters: group elements
- Small letters: algebra elements

---

## Files and Locations

```
EqF-flight-estimator/
├── docs/
│   ├── paper-formulation/
│   │   ├── 01-SYSTEM-DEFINITION.md
│   │   ├── 02-SYMMETRY-AND-EQUIVARIANCE.md
│   │   ├── 03-LIFTED-SYSTEM.md
│   │   └── 04-FILTER-DESIGN.md
│   └── IMPLEMENTATION-PLAN.md
├── eqf_filter.py                    [TO BE REWRITTEN]
├── requirements.txt
└── tests/
    └── test_*.py                    [NEW]
```

---

## Questions to Guide Your Work

1. **On Semi-Direct Product**: Why is `(B, b) * (A, a) = (BA, b + Ad_B[a])`? Why not just `(BA, b + a)`?
   → Answer: Because a ∈ se(2,3) must transform consistently with group element A

2. **On Virtual Inputs**: Why add virtual velocity ν and bias I_bν?
   → Answer: Makes bias live in se(2,3), achieves equivariance

3. **On Lift**: What does Λ(ξ,u) actually compute?
   → Answer: Maps system dynamics to tangent space of symmetry group

4. **On A^0_t Being Constant**: Why is this better than time-varying?
   → Answer: Geometry ensures bias couples linearly to pose error → better observability

5. **On Equivariance**: How does it help the filter?
   → Answer: Ensures error stays in fixed frame, no "frame drift" issues like MEKF

---

## Next Action

Read `docs/paper-formulation/01-SYSTEM-DEFINITION.md` first.
Then ask questions on anything that's unclear.
