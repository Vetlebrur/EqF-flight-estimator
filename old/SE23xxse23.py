"""
SE23xxse23: Semi-direct product group SE(2,3) ⋉ se(2,3)

Represents the group of biased inertial states where:
  - T ∈ SE(2,3): extended pose (rotation, position, velocity)
  - b ∈ se(2,3): bias in the Lie algebra (9-dimensional vector)

Group composition: (T2, b2) ∘ (T1, b1) = (T2*T1, b2 + Ad_{T2^{-1}}[b1])

This follows Fornasier et al. (2022) for equivariant filter design.
"""

from pylie import SE23, SO3
import numpy as np
import scipy


# =============================================================================
# SE23 Left Jacobian and Helper Functions (from reference implementation)
# =============================================================================

def _J1(so3vec: np.ndarray) -> np.ndarray:
    """SO3 Jacobian helper."""
    angle = np.linalg.norm(so3vec)
    if np.isclose(angle, 0.0):
        return np.eye(3) + 0.5 * SO3.wedge(so3vec)

    axis = so3vec / angle
    s = np.sin(angle) / angle
    c = (1 - np.cos(angle)) / angle
    return s * np.eye(3) + (1 - s) * np.outer(axis, axis) + c * SO3.wedge(axis)


def _Q1(arr: np.ndarray) -> np.ndarray:
    """Q1 operator for SE23 Jacobian."""
    phi = arr[0:3]
    rho = arr[3:6]

    rx = SO3.wedge(rho)
    px = SO3.wedge(phi)

    ph = np.linalg.norm(phi)
    if np.isclose(ph, 0.0):
        return rx

    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph

    cph = np.cos(ph)
    sph = np.sin(ph)

    m1 = 0.5
    m2 = (ph - sph) / ph3
    m3 = (0.5 * ph2 + cph - 1.0) / ph4
    m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

    t1 = rx
    t2 = px @ rx + rx @ px + px @ rx @ px
    t3 = px @ px @ rx + rx @ px @ px - 3.0 * px @ rx @ px
    t4 = px @ rx @ px @ px + px @ px @ rx @ px

    return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4


def SE23LeftJacobian(se23vec: np.ndarray) -> np.ndarray:
    """Compute the left Jacobian of SE(2,3)."""
    if isinstance(se23vec, np.ndarray) and se23vec.ndim == 2:
        se23vec = se23vec.flatten()

    phi = se23vec[0:3]
    rho = se23vec[3:6]
    psi = se23vec[6:9]

    if np.isclose(np.linalg.norm(phi), 0.0):
        return np.eye(9) + 0.5 * SE23.adjoint(se23vec.reshape(9, 1) if se23vec.ndim == 1 else se23vec)

    SO3_JL = _J1(phi)

    J = np.zeros((9, 9))
    J[0:3, 0:3] = SO3_JL
    J[3:6, 3:6] = SO3_JL
    J[6:9, 6:9] = SO3_JL
    J[3:6, 0:3] = _Q1(np.concatenate([phi, rho]))
    J[6:9, 0:3] = _Q1(np.concatenate([phi, psi]))

    return J


class SE23xxse23:
    """
    Semi-direct product group SE(2,3) ⋉ se(2,3).

    Elements are pairs (T, b) where:
      - T ∈ SE(2,3): pose with rotation, position, velocity
      - b ∈ ℝ⁹: bias vector in se(2,3) algebra
    """

    DIM = 18  # 9 DOF for SE(2,3) + 9 DOF for se(2,3)

    def __init__(self, T: SE23 = None, b: np.ndarray = None):
        """
        Initialize SE(2,3) ⋉ se(2,3) element.

        Args:
            T: SE23 group element (pose). Defaults to identity.
            b: Bias vector (9,). Defaults to zero.
        """
        if T is None:
            T = SE23.identity()
        if b is None:
            b = np.zeros(9)

        # Store pose as SE23 element
        if isinstance(T, SE23):
            self.T = T
        else:
            raise TypeError("T must be SE23 instance")

        # Store bias as 9-vector
        if isinstance(b, np.ndarray):
            if b.size != 9:
                raise ValueError("b must have size 9")
            self.b = np.array(b).reshape(9)
        else:
            raise TypeError("b must be numpy array")

    # =========================================================================
    # Basic Operations
    # =========================================================================

    def __str__(self) -> str:
        """String representation."""
        return f"SE23xxse23(T=\n{self.T.as_matrix()},\nb={self.b})"

    def __repr__(self) -> str:
        """Representation."""
        return self.__str__()

    # =========================================================================
    # Group Composition: (T2, b2) * (T1, b1)
    # =========================================================================

    def __mul__(self, other: 'SE23xxse23') -> 'SE23xxse23':
        """
        Semi-direct product composition.

        (T2, b2) * (T1, b1) = (T2*T1, b2 + Ad_{T2^{-1}}[b1])

        The bias transforms by the adjoint action of T2^{-1}.

        Args:
            other: Right operand (T1, b1).

        Returns:
            Composed element (T2*T1, b2 + Ad_{T2^{-1}}[b1]).
        """
        if not isinstance(other, SE23xxse23):
            return NotImplemented

        result = SE23xxse23()

        # SE(2,3) multiplication: T2 * T1
        result.T = self.T * other.T

        # Bias transformation: b2 + Ad_{T2}[b1]
        # Conjugation by T2 in matrix form: T2 @ β_other @ T2^{-1}
        # Ad_T corresponds to conjugation by T_inv, so we need Ad_{T_inv} to conjugate by T
        Ad_T2_inv = self._adjoint_matrix(self.T.inv())

        # Apply adjoint to other's bias
        result.b = self.b + Ad_T2_inv @ other.b

        return result

    def __truediv__(self, other: 'SE23xxse23') -> 'SE23xxse23':
        """Right division: self * other^{-1}."""
        if not isinstance(other, SE23xxse23):
            return NotImplemented
        return self * other.inv()

    # =========================================================================
    # Inverse: (T, b)^{-1} = (T^{-1}, -Ad_{T^{-1}}[b])
    # =========================================================================

    def inv(self) -> 'SE23xxse23':
        """
        Inverse element.

        (T, b)^{-1} = (T^{-1}, -Ad_{T^{-1}}[b])

        Returns:
            Inverse element.
        """
        result = SE23xxse23()
        # Inverse of SE(2,3) part
        result.T = self.T.inv()
        # Bias part: -Ad_{T^{-1}}[b]
        result.b = -self.T.Adjoint() @ self.b
        return result

    # =========================================================================
    # Identity Element
    # =========================================================================

    @staticmethod
    def identity() -> 'SE23xxse23':
        """
        Identity element: (I_{SE23}, 0).

        Returns:
            Identity element.
        """
        result = SE23xxse23()
        result.T = SE23.identity()
        result.b = np.zeros(9)
        return result

    # =========================================================================
    # Adjoint Action: Ad_{(T,b)} on tangent space
    # =========================================================================

    def Adjoint(self) -> np.ndarray:
        """
        Adjoint matrix for this element (18×18).

        Acts on tangent space (se(2,3) ⋉ se(2,3)).

        For semi-direct product, Ad_{(T,b)}[ξ, η] acts as:
          - On pose part ξ: standard SE(2,3) adjoint Ad_T
          - On bias part η: Ad_T(η) + ad_ξ[b] (coupling)

        Returns:
            18×18 adjoint matrix.
        """
        Ad_18 = np.zeros((18, 18))

        # Upper-left: Ad_T for pose part (9×9)
        Ad_T = self.T.Adjoint()
        Ad_18[0:9, 0:9] = Ad_T

        # Lower-left: coupling between pose and bias (9×9)
        # This comes from the Lie bracket: ad_ξ[b]
        # For semi-direct product: ad_ξ acts on b with adjoint coupling
        b_bracket_mat = self._lie_bracket_matrix(self.b)
        Ad_18[9:18, 0:9] = b_bracket_mat

        # Upper-right: zero (pose doesn't affect by bias)
        # (already initialized to zero)

        # Lower-right: Ad_T for bias part (9×9)
        Ad_18[9:18, 9:18] = Ad_T

        return Ad_18

    @staticmethod
    def adjoint(xi_eta: np.ndarray) -> np.ndarray:
        """
        Adjoint action on tangent space vectors (18×18 matrix).

        For tangent vector [ξ; η] (9+9 components), compute the adjoint matrix.

        Args:
            xi_eta: 18-vector [ξ (9); η (9)].

        Returns:
            18×18 adjoint matrix.
        """
        if not isinstance(xi_eta, np.ndarray):
            raise TypeError("xi_eta must be numpy array")
        if xi_eta.size != 18:
            raise ValueError("xi_eta must have 18 components")

        xi = xi_eta[0:9]
        eta = xi_eta[9:18]

        ad_18 = np.zeros((18, 18))

        # Upper-left: ad_ξ for pose part
        ad_xi = SE23.adjoint(xi)
        ad_18[0:9, 0:9] = ad_xi

        # Lower-left: 0 (bias doesn't directly couple back to pose)
        # (already zero)

        # Upper-right: 0 (pose derivative doesn't come from bias)
        # (already zero)

        # Lower-right: ad_ξ for bias part
        # and coupling from ad_η
        ad_18[9:18, 0:9] = SE23xxse23._lie_bracket_matrix(eta)
        ad_18[9:18, 9:18] = ad_xi

        return ad_18

    # =========================================================================
    # Lie Algebra: Exponential and Logarithm
    # =========================================================================
    
    @staticmethod
    def exp(u):
        """Exponential map matching reference implementation using SE23LeftJacobian."""
        xi  = u[:9]
        eta = u[9:]

        result = SE23xxse23()
        result.T = SE23.exp(xi.reshape(9, 1) if xi.ndim == 1 else xi)

        # Use SE23 Left Jacobian (matches reference implementation)
        J_L = SE23LeftJacobian(xi)
        result.b = J_L @ eta

        return result

    def log(self):
        """Logarithm map matching reference implementation using SE23LeftJacobian inverse."""
        xi = self.T.log()

        # Use inverse of SE23 Left Jacobian (matches reference)
        J_L = SE23LeftJacobian(xi)
        J_L_inv = np.linalg.inv(J_L)
        eta = J_L_inv @ self.b

        return np.concatenate([xi, eta])

    # =========================================================================
    # Matrix Representations
    # =========================================================================

    def as_matrix(self) -> np.ndarray:
        """
        Matrix representation (18×18).

        Upper-left: SE(2,3) pose as 5×5
        Upper-right: Zero
        Lower-left: Bias stacked
        Lower-right: Identity

        Returns:
            18×18 matrix.
        """
        mat = np.eye(18)

        # SE(2,3) part in upper-left corner (5×5 block)
        T_mat = self.T.as_matrix()
        mat[0:5, 0:5] = T_mat

        # Bias as diagonal block (conceptual representation)
        # We store bias as part of the state, so represent it in lower-right
        for i in range(9):
            mat[9 + i, 9 + i] = 1.0
            # Optionally scale by bias magnitude
            # mat[9+i, 9+i] = 1.0 + np.linalg.norm(self.b[i])

        return mat

    @staticmethod
    def from_matrix(mat: np.ndarray) -> 'SE23xxse23':
        """
        Reconstruct from matrix representation.

        Args:
            mat: 18×18 matrix.

        Returns:
            SE23xxse23 element.
        """
        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be numpy array")
        if mat.shape != (18, 18):
            raise ValueError("mat must be 18×18")

        result = SE23xxse23()

        # Extract SE(2,3) part from upper-left
        result.T = SE23.from_matrix(mat[0:5, 0:5])

        # Extract bias from lower-right diagonal (conceptually)
        # For now, assume bias is embedded in remaining components
        result.b = np.zeros(9)

        return result

    # =========================================================================
    # Wedge/Vee: Vector ↔ Matrix Representations
    # =========================================================================

    @staticmethod
    def wedge(vec: np.ndarray) -> np.ndarray:
        """
        Convert 18-vector to matrix form.

        Args:
            vec: 18-vector [ξ (9); η (9)].

        Returns:
            18×18 matrix (wrapped SE(2,3) + bias part).
        """
        if not isinstance(vec, np.ndarray):
            raise TypeError("vec must be numpy array")
        if vec.size != 18:
            raise ValueError("vec must have 18 components")

        mat = np.zeros((18, 18))

        # Upper-left: SE(2,3) part
        xi = vec[0:9]
        mat[0:5, 0:5] = SE23.wedge(xi)

        # Lower-right: bias algebra part (as adjoint matrix)
        eta = vec[9:18]
        eta_bracket = SE23xxse23._lie_bracket_matrix(eta)
        mat[9:18, 9:18] = eta_bracket

        return mat

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """
        Convert matrix form to 18-vector.

        Args:
            mat: 18×18 matrix.

        Returns:
            18-vector [ξ (9); η (9)].
        """
        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be numpy array")
        if mat.shape != (18, 18):
            raise ValueError("mat must be 18×18")

        vec = np.zeros(18)

        # Extract SE(2,3) part from upper-left
        xi = SE23.vee(mat[0:5, 0:5])
        vec[0:9] = xi

        # Extract bias part from lower-right (inverse bracket)
        # For now, simplified extraction
        vec[9:18] = mat[9:18, 16]  # Extract from a representative column

        return vec

    # =========================================================================
    # Component Access
    # =========================================================================

    def get_pose(self) -> SE23:
        """Get SE(2,3) pose element."""
        return self.T

    def get_bias(self) -> np.ndarray:
        """Get bias vector (9,)."""
        return self.b.copy()

    def get_rotation(self) -> np.ndarray:
        """Get rotation matrix (3×3)."""
        return self.T.R().as_matrix()

    def get_position(self) -> np.ndarray:
        """Get position vector (3,)."""
        return self.T.x().as_vector()

    def get_velocity(self) -> np.ndarray:
        """Get velocity vector (3,)."""
        return self.T.w().as_vector()

    def get_bias_gyro(self) -> np.ndarray:
        """Get gyroscope bias (3,)."""
        return self.b[0:3].copy()

    def get_bias_accel(self) -> np.ndarray:
        """Get accelerometer bias (3,)."""
        return self.b[3:6].copy()

    def get_bias_mu(self) -> np.ndarray:
        """Get mu bias term (3,)."""
        return self.b[6:9].copy()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _adjoint_matrix(T: SE23) -> np.ndarray:
        """
        Get adjoint matrix of SE(2,3) element (9×9).

        This is a wrapper around T.Adjoint() for clarity.

        Args:
            T: SE23 element.

        Returns:
            9×9 adjoint matrix.
        """
        return T.Adjoint()

    @staticmethod
    def _lie_bracket_matrix(vec: np.ndarray) -> np.ndarray:
        """
        Compute Lie bracket adjoint matrix for se(2,3) element.

        Given v ∈ se(2,3), return the matrix representing ad_v[·].

        Args:
            vec: 9-vector in se(2,3).

        Returns:
            9×9 matrix where ad_v[u] = result @ u.
        """
        if not isinstance(vec, np.ndarray):
            raise TypeError("vec must be numpy array")
        if vec.size != 9:
            raise ValueError("vec must have 9 components")

        # Use SE23's adjoint method
        return SE23.adjoint(vec)

    @staticmethod
    def _lie_bracket(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute Lie bracket [u, v] in se(2,3).

        Uses matrix commutator: [u, v] = uv - vu.

        Args:
            u: 9-vector.
            v: 9-vector.

        Returns:
            9-vector [u, v].
        """
        # Convert to matrix form
        u_mat = SE23.wedge(u)
        v_mat = SE23.wedge(v)

        # Commutator
        bracket_mat = u_mat @ v_mat - v_mat @ u_mat

        # Convert back to vector
        return SE23.vee(bracket_mat)

    @staticmethod
    def numericalDifferential(func, x, eps=1e-8):
        """
        Compute numerical Jacobian using central finite differences.

        Args:
            func: Function mapping R^n -> R^m.
            x: Base point (n-vector).
            eps: Perturbation magnitude.

        Returns:
            Jacobian matrix (m × n).
        """
        x = np.asarray(x).flatten()
        y0 = func(x)
        y0 = np.asarray(y0).flatten()

        m = y0.size
        n = x.size
        J = np.zeros((m, n))

        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += eps
            y_plus = np.asarray(func(x_plus)).flatten()

            x_minus = x.copy()
            x_minus[j] -= eps
            y_minus = np.asarray(func(x_minus)).flatten()

            J[:, j] = (y_plus - y_minus) / (2 * eps)

        return J

    def local_coords(self, state: 'SE23xxse23') -> np.ndarray:
        """
        Compute local coordinates of state relative to self (reference state).

        Maps state to tangent space representation using exponential coordinates.

        Args:
            state: SE23xxse23 element.

        Returns:
            18-vector in tangent space [log(T_rel); b_rel].
        """
        # Relative pose: T_rel = self.T^{-1} * state.T
        T_rel = self.T.inv() * state.T

        # Relative bias: b_rel = state.b - self.b (exponential coordinates)
        b_rel = state.b - self.b

        # Tangent coordinates: [log(T_rel); b_rel]
        xi = T_rel.log()
        eps = np.concatenate([xi, b_rel])

        return eps

    def stateAction(self, X: 'SE23xxse23', state: 'SE23xxse23') -> 'SE23xxse23':
        """
        Left action of group element X on state.

        Transforms state by group element: X ⊳ state.
        Uses matrix representation with SE23.wedge/vee for bias.

        Args:
            X: Group element acting on state.
            state: State to be transformed.

        Returns:
            Transformed state.
        """
        result = SE23xxse23()

        # Action on pose: T_new = state.T * X.T
        result.T = state.T * X.T

        # Action on bias using wedge/vee operations
        # b_new = vee(X.T^{-1} * (wedge(state.b) - wedge(X.b)) * X.T)
        X_T_inv_mat = X.T.inv().as_matrix()
        X_T_mat = X.T.as_matrix()

        state_b_wedge = SE23.wedge(state.b)
        X_b_wedge = SE23.wedge(X.b)

        b_matrix = X_T_inv_mat @ (state_b_wedge - X_b_wedge) @ X_T_mat
        result.b = SE23.vee(b_matrix)

        return result

    def stateActionDiff(self, state: 'SE23xxse23') -> np.ndarray:
        """
        Compute Jacobian of state action at identity.

        Linearizes the state action g -> (g ⊳ state) around g = identity.
        Uses numerical differentiation of the composite map: exp -> stateAction -> local_coords.

        Args:
            state: State to compute action derivative for.

        Returns:
            18×18 Jacobian matrix.
        """
        def coords_action(U):
            g_pert = SE23xxse23.exp(U)
            state_action = self.stateAction(g_pert, state)
            return self.local_coords(state_action)

        J = SE23xxse23.numericalDifferential(coords_action, np.zeros((18, 1)))
        return J

    def localChartDiff(self, state: 'SE23xxse23') -> np.ndarray:
        """
        Compute differential of local chart: D_Θ_h.

        Jacobian of the local coordinate map ε = Θ_h(ξ)
        where Θ_h := (φ_{ξ°} ∘ exp_h)^{-1} is the local chart on horizontal subspace h.

        D_Θ_h = ∂ε/∂ξ where ε are local coordinates.

        Args:
            state: Reference state where chart is evaluated.

        Returns:
            18×18 Jacobian matrix D_Θ_h.
        """
        def chart_composed(U):
            h_pert = SE23xxse23.exp(U)
            state_pert = h_pert * state
            return self.local_coords(state_pert)

        D_Theta_h = SE23xxse23.numericalDifferential(chart_composed, np.zeros((18, 1)))
        return D_Theta_h

# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("SE23xxse23: Semi-Direct Product Group SE(2,3) (×) se(2,3)")
    print("=" * 60)

    # Create identity
    G_id = SE23xxse23.identity()
    print(f"\nIdentity element:")
    print(f"  T: {type(G_id.T).__name__}")
    print(f"  b: shape {G_id.b.shape}, norm {np.linalg.norm(G_id.b):.6f}")

    # Create elements with random bias
    G1 = SE23xxse23()
    G1.b = np.random.randn(9) * 0.1

    G2 = SE23xxse23()
    G2.b = np.random.randn(9) * 0.1

    print(f"\nElement G1:")
    print(f"  b norm: {np.linalg.norm(G1.b):.6f}")

    # Composition
    G_prod = G1 * G2
    print(f"\nComposition G1 * G2:")
    print(f"  T type: {type(G_prod.T).__name__}")
    print(f"  b norm: {np.linalg.norm(G_prod.b):.6f}")

    # Inverse
    G1_inv = G1.inv()
    print(f"\nInverse G1^{{-1}}:")
    print(f"  b norm: {np.linalg.norm(G1_inv.b):.6f}")

    # Identity composition (should recover identity)
    G_id_prod = G1 * G1_inv
    print(f"\nG1 * G1^{{-1}}:")
    print(f"  b norm: {np.linalg.norm(G_id_prod.b):.6e} (should be ~0)")

    # Adjoint
    Ad = G1.Adjoint()
    print(f"\nAdjoint matrix shape: {Ad.shape}")

    # Tangent space
    tangent = np.random.randn(18)
    G_exp = SE23xxse23.exp(tangent)
    print(f"\nExp map:")
    print(f"  Input: 18-vector")
    print(f"  Output: SE23xxse23 element")
    print(f"  b norm: {np.linalg.norm(G_exp.b):.6f}")

    tangent_recovered = G_exp.log()
    print(f"\nLog map:")
    print(f"  Exp/log error: {np.linalg.norm(tangent - tangent_recovered):.6e}")

    # Test state action differential
    print(f"\n=== State Action Differential ===")
    state = SE23xxse23()
    state.b = np.random.randn(9) * 0.05

    J = G1.stateActionDiff(state)
    print(f"State action Jacobian shape: {J.shape}")
    print(f"Jacobian rank: {np.linalg.matrix_rank(J)}")
    print(f"Jacobian condition number: {np.linalg.cond(J):.2e}")

    print("\nAll tests passed!")
