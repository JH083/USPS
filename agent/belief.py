from typing import Callable, Iterable, Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray


class GaussianBelief:
    """Multivariate Gaussian posterior over parameters θ.

    The model assumes a locally linear relationship between θ and the next-state
    residuals via the user-supplied Jacobian J = ∂f/∂θ evaluated at θ̂.

    Given a batch {(s_i, a_i, s'_i)} and linearization point θ̂, define:
        y_i = s'_i - f(s_i, a_i; θ̂)           # residual (d_s,)
        J_i = jacobian(s_i, a_i; θ̂)           # (d_s x d_θ)
        \tilde{y}_i = y_i + J_i θ̂             # shifted target

    Stack Y = [\tilde{y}_1; ...; \tilde{y}_N] in ℝ^{N·d_s} and
          X = [J_1; ...; J_N] in ℝ^{N·d_s × d_θ}.

    With prior N(μ, Σ) and isotropic observation noise σ²I, the conjugate
    linear-Gaussian update is:
        Σ⁻¹_new = Σ⁻¹ + (Xᵀ X)/σ²
        μ_new   = Σ_new · (Σ⁻¹ μ + (Xᵀ Y)/σ²)

    Parameters
    ----------
    mu0 : Array
        Prior mean (d_θ,).
    Sigma0 : Array
        Prior covariance (d_θ, d_θ). Must be symmetric positive definite.
    sigma2 : float
        Observation noise variance used in the linear-Gaussian update.
    jitter : float
        Small diagonal term added to improve numerical stability in Cholesky.
    """

    def __init__(self, mu0: Array, Sigma0: Array, *, sigma2: float = 1e-3, jitter: float = 1e-6) -> None:
        mu0 = np.asarray(mu0, dtype=float).reshape(-1)
        Sigma0 = np.asarray(Sigma0, dtype=float)
        if Sigma0.shape[0] != Sigma0.shape[1]:
            raise ValueError("Sigma0 must be square (d x d)")
        if Sigma0.shape[0] != mu0.shape[0]:
            raise ValueError("mu0 and Sigma0 dimension mismatch")
        self.mu: Array = mu0.copy()
        self.Sigma: Array = 0.5 * (Sigma0 + Sigma0.T)  # symmetrize defensively
        self.sigma2: float = float(sigma2)
        self.jitter: float = float(jitter)

        # Cache precision-form parameters lazily when needed
        self._Sigma_inv: Optional[Array] = None


    def sample_theta(self, rng: Optional[np.random.Generator] = None) -> Array:
        """Draw a parameter sample θ ~ N(μ, Σ).

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random generator to use. If None, uses default Generator.
        """
        if rng is None:
            rng = np.random.default_rng()
        # Cholesky factor of Σ + εI for stability
        L = np.linalg.cholesky(self.Sigma + self.jitter * np.eye(self.Sigma.shape[0]))
        z = rng.standard_normal(self.mu.shape[0])
        return self.mu + L @ z

    def mahalanobis2(self, theta: Array) -> float:
        """Return squared Mahalanobis distance (θ-μ)^T Σ^{-1} (θ-μ).
        Useful for credible-region checks by the caller.
        """
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape[0] != self.mu.shape[0]:
            raise ValueError("theta has wrong dimension")
        Sigma_inv = self._inv_cov()
        diff = theta - self.mu
        return float(diff @ (Sigma_inv @ diff))

    def set_noise(self, sigma2: float) -> None:
        """Update observation noise variance σ² used in updates."""
        self.sigma2 = float(sigma2)

    def update(
        self,
        batch: Sequence[Tuple[Array, Array, Array]],
        *,
        f: Callable[[Array, Array, Array], Array],
        jacobian: Callable[[Array, Array, Array], Array],
        theta_lin: Optional[Array] = None,
    ) -> None:
        """Conjugate linear-Gaussian update using a mini-batch.

        Parameters
        ----------
        batch : sequence of (s, a, s_next)
            Mini-batch of transitions from the replay buffer. Each element can
            be 1-D arrays; they will be flattened. All samples must share the
            same state dimension.
        f : callable
            Forward model: f(s, a, theta) -> predicted next state (same shape as s_next).
        jacobian : callable
            Jacobian wrt θ at (s, a, theta_lin): returns array of shape (d_s, d_θ).
        theta_lin : array, optional
            Linearization point θ̂. If None, uses current posterior mean μ.
        """
        if len(batch) == 0:
            return
        if theta_lin is None:
            theta_lin = self.mu
        theta_lin = np.asarray(theta_lin, dtype=float).reshape(-1)

        X_blocks = []  # list of (d_s x d_theta)
        Y_blocks = []  # list of (d_s,)

        # Build stacked design matrix X and target Y
        for (s, a, s_next) in batch:
            s = np.asarray(s, dtype=float).reshape(-1)
            a = np.asarray(a, dtype=float).reshape(-1)
            s_next = np.asarray(s_next, dtype=float).reshape(-1)

            J = np.asarray(jacobian(s, a, theta_lin), dtype=float)
            if J.ndim != 2 or J.shape[1] != self.mu.shape[0]:
                raise ValueError("jacobian must return (d_s x d_theta)")

            pred = np.asarray(f(s, a, theta_lin), dtype=float).reshape(-1)
            if pred.shape != s_next.shape or pred.shape[0] != J.shape[0]:
                raise ValueError("forward model / jacobian output shape mismatch")

            y = s_next - pred  # (d_s,)
            # Shifted target: y + J θ̂  so that model becomes linear in θ
            y_tilde = y + J @ theta_lin  # (d_s,)

            X_blocks.append(J)
            Y_blocks.append(y_tilde)

        X = np.vstack(X_blocks)  # (N*d_s, d_theta)
        Y = np.concatenate(Y_blocks, axis=0)  # (N*d_s,)

        # Prior precision and natural parameter
        d = self.mu.shape[0]
        Sigma_inv = self._inv_cov()  # (d,d)
        A = Sigma_inv + (X.T @ X) / max(self.sigma2, 1e-12)
        b = Sigma_inv @ self.mu + (X.T @ Y) / max(self.sigma2, 1e-12)

        # Solve A μ_new = b using Cholesky (SPD guaranteed with jitter)
        A_jit = A + self.jitter * np.eye(d)
        L = np.linalg.cholesky(A_jit)
        mu_new = np.linalg.solve(L.T, np.linalg.solve(L, b))

        # Covariance is the inverse of A; for small/medium d we form it explicitly
        Sigma_new = np.linalg.inv(A_jit)

        self.mu = mu_new
        # Symmetrize for numerical hygiene
        self.Sigma = 0.5 * (Sigma_new + Sigma_new.T)
        self._Sigma_inv = None  # invalidate cache


    def _inv_cov(self) -> Array:
        """Return Σ^{-1}, computing/caching if necessary."""
        if self._Sigma_inv is None:
            d = self.Sigma.shape[0]
            S = self.Sigma + self.jitter * np.eye(d)
            # Solve for inverse via Cholesky for stability
            L = np.linalg.cholesky(S)
            I = np.eye(d)
            # Σ^{-1} = (L Lᵀ)^{-1} = L^{-T} L^{-1}
            Linv = np.linalg.solve(L, I)
            self._Sigma_inv = Linv.T @ Linv
        return self._Sigma_inv