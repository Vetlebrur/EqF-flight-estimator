"""
Reset step for TGEqF: State reset and reinitialization.

Implements reset methods for filter state management.
"""

import numpy as np
from tg_eqf import TGEqF
import SE23xxse23


def calculate_Delta(
    self: TGEqF,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Calculate Delta using local chart differential.

    Implements: Δ = D_φ_c(id)^T D_Θ_b^{-1} μ

    Where:
      - D_φ_c(id): Jacobian of state action at identity (18×18)
      - D_Θ_b: Differential of local chart, bias block (9×9)
      - μ: Innovation vector (18,)

    Args:
        mu: Innovation vector (18,).

    Returns:
        Delta vector (18,) for state update in Lie algebra.
    """
    X_hat = SE23xxse23(self.T,self.b)

    # D_φ_c(id): Differential of state action at identity
    reference_state = SE23xxse23.identity()
    D_phi_c = X_hat.stateActionDiff(reference_state)

    # D_Θ_h: Differential of local chart (horizontal subspace)
    D_Theta_h = X_hat.localChartDiff(reference_state)

    # Δ = pseudoinverse(D_φ_c) D_Θ_h^{-1} μ
    Delta = np.linalg.pinv(D_phi_c) @ np.linalg.inv(D_Theta_h) @ mu

    return Delta


def reset_sigma(
    self: TGEqF,
) -> None:
    """
    Reset covariance matrix Sigma.

    Updates the filter covariance using Joseph form stabilization.
    """
    # TODO: Implement Sigma reset
    funny_expression = SE23xxse23.vee(SE23xxse23.exp(-self.Delta/2).Adjoint())

    self.Sigma =  funny_expression @ self.Sigma @ funny_expression.T


def reset_X(
    self: TGEqF,
    Delta: np.ndarray,
) -> None:
    """
    Reset state X using exponential map.

    Updates: X_hat = exp(Delta) * X_hat

    Args:
        Delta: Update vector (18,) in Lie algebra.
    """
    delta_group = SE23xxse23.exp(Delta)
    X_hat = SE23xxse23(self.T, self.b)
    X_hat = delta_group * X_hat  # Group composition
    self.T = X_hat.T
    self.b = X_hat.b


def reset(
    self: TGEqF,
    mu: np.ndarray,
) -> None:
    """
    Perform full equivariant filter reset step.

    Implements: Δ = (D_φ_c)^T D_Θ_h^{-1} μ and updates state.

    Args:
        mu: Innovation vector (18,).
    """
    # Compute innovation lift: Δ = (D_φ_c)^T D_Θ_h^{-1} μ
    self.Delta = self.calculate_Delta(mu)

    # Update state
    self.reset_X(self.Delta)

    # Reset covariance
    self.reset_sigma()


# Attach methods to TGEqF
TGEqF.calculate_Delta = calculate_Delta
TGEqF.reset_sigma = reset_sigma
TGEqF.reset_X = reset_X
TGEqF.reset = reset

__all__ = ['TGEqF']
