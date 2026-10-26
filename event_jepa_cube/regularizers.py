"""JEPA regularizers and representation-health diagnostics.

Includes SIGReg, WeakSIGReg, RDMReg, NormReg, SACReg, and SubSIGReg,
grounded in LeJEPA (arXiv:2511.08544), Weak-SIGReg (arXiv:2603.05924),
Rectified LpJEPA (arXiv:2602.01456), BiJEPA (arXiv:2603.00049),
lambda-JEPA (arXiv:2609.35288), Sub-JEPA (arXiv:2605.09241),
MotionJEPA (arXiv:2609.23881), and MM-LeJEPA (arXiv:2610.24681).

Requires PyTorch. Install with: pip install event-jepa-cube[torch]
"""

from __future__ import annotations

import math

try:
    import torch
    from torch import Tensor

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Tensor = None  # type: ignore[assignment,misc]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for regularizers. Install with: pip install event-jepa-cube[torch]")


class SIGReg:
    """Sketched Isotropic Gaussian Regularization (LeJEPA).

    Enforces isotropic Gaussian distribution on embeddings via random
    projections and Epps-Pulley characteristic function matching.

    The loss projects embeddings onto M random unit-norm directions and
    compares each projected 1D distribution to N(0,1) using the Epps-Pulley
    test statistic based on characteristic functions.

    Reference: LeJEPA (arXiv:2511.08544)

    Args:
        num_directions: Number of random projection directions (M). More
            directions give better approximation. Default 64.
        sigma: Width of Gaussian weighting window for Epps-Pulley test.
            Default 1.0.
        num_quadrature_points: Number of quadrature points for integral
            approximation. Default 17.
        quadrature_range: Range [-R, R] for trapezoidal quadrature. Default 5.0.
    """

    def __init__(
        self,
        num_directions: int = 64,
        sigma: float = 1.0,
        num_quadrature_points: int = 17,
        quadrature_range: float = 5.0,
    ) -> None:
        _require_torch()
        self.num_directions = num_directions
        self.sigma = sigma
        self.num_quadrature_points = num_quadrature_points
        self.quadrature_range = quadrature_range

    def _sample_directions(self, dim: int, device: torch.device) -> Tensor:
        """Sample M random unit-norm directions from the unit hypersphere."""
        directions = torch.randn(dim, self.num_directions, device=device)
        directions = directions / directions.norm(dim=0, keepdim=True)
        return directions

    def _epps_pulley(self, samples: Tensor) -> Tensor:
        """Compute Epps-Pulley statistic comparing samples to N(0,1).

        EP = N * integral |phi_hat(t) - phi(t)|^2 * w(t) dt

        where phi_hat is empirical CF, phi(t) = exp(-t^2/2) is Gaussian CF,
        and w(t) = exp(-t^2/sigma^2) is the weighting window.
        """
        n = samples.shape[0]
        # Quadrature points
        t_points = torch.linspace(
            -self.quadrature_range,
            self.quadrature_range,
            self.num_quadrature_points,
            device=samples.device,
        )
        t_points[1] - t_points[0]

        # Empirical CF: phi_hat(t) = (1/N) sum_j exp(i*t*x_j)
        # samples: (N,), t_points: (T,) -> outer product: (N, T)
        phases = samples.unsqueeze(1) * t_points.unsqueeze(0)  # (N, T)
        ecf_real = torch.cos(phases).mean(dim=0)  # (T,)
        ecf_imag = torch.sin(phases).mean(dim=0)  # (T,)

        # Theoretical Gaussian CF: phi(t) = exp(-t^2/2)
        gcf = torch.exp(-0.5 * t_points**2)

        # |phi_hat(t) - phi(t)|^2
        diff_real = ecf_real - gcf
        diff_imag = ecf_imag  # Gaussian CF is real, so imaginary part is just ecf_imag
        diff_sq = diff_real**2 + diff_imag**2

        # Weighting: w(t) = exp(-t^2/sigma^2)
        weight = torch.exp(-(t_points**2) / (self.sigma**2))

        # Trapezoidal integration
        integrand = diff_sq * weight
        ep = float(n) * torch.trapezoid(integrand, t_points)

        return ep

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        """Compute SIGReg loss on a batch of embeddings.

        Args:
            embeddings: Tensor of shape (N, d) where N is batch size
                and d is embedding dimension.

        Returns:
            Scalar loss tensor.
        """
        _require_torch()
        n, d = embeddings.shape

        # Center and normalize embeddings to unit variance per dimension
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True).clamp(min=1e-8)
        embeddings = embeddings / std

        # Sample random directions
        directions = self._sample_directions(d, embeddings.device)

        # Project embeddings onto directions: (N, d) @ (d, M) -> (N, M)
        projections = embeddings @ directions

        # Compute Epps-Pulley statistic for each direction
        total_loss = torch.tensor(0.0, device=embeddings.device)
        for i in range(self.num_directions):
            total_loss = total_loss + self._epps_pulley(projections[:, i])

        return total_loss / self.num_directions


class WeakSIGReg:
    """Covariance-targeting variant of SIGReg for supervised settings.

    Targets the covariance matrix via random sketching rather than the full
    characteristic function. More computationally efficient, suitable as a
    general training stabilizer.

    Loss = ||sketch(Cov(Z)) - sketch(I)||_F^2

    Reference: Weak-SIGReg (arXiv:2603.05924)

    Args:
        sketch_dim: Dimension of the random sketch matrix (K). Default 64.
    """

    def __init__(self, sketch_dim: int = 64) -> None:
        _require_torch()
        self.sketch_dim = sketch_dim
        self._sketch_matrix: Tensor | None = None
        self._sketch_dim_source: int | None = None

    def _get_sketch_matrix(self, d: int, device: torch.device) -> Tensor:
        """Get or create the random sketch matrix S in R^{K x d}."""
        if self._sketch_matrix is None or self._sketch_dim_source != d:
            self._sketch_matrix = torch.randn(self.sketch_dim, d, device=device) / math.sqrt(d)
            self._sketch_dim_source = d
        return self._sketch_matrix.to(device)

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        """Compute Weak-SIGReg loss on a batch of embeddings.

        Args:
            embeddings: Tensor of shape (N, d) where N is batch size
                and d is embedding dimension.

        Returns:
            Scalar loss tensor.
        """
        _require_torch()
        n, d = embeddings.shape

        # Center embeddings
        z_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Get sketch matrix
        s = self._get_sketch_matrix(d, embeddings.device)

        # Sketch the embeddings: (N, d) @ (K, d)^T -> (N, K)
        z_sketch = z_centered @ s.t()

        # Compute sketched covariance: (K, N) @ (N, K) / N -> (K, K)
        cov_sketch = (z_sketch.t() @ z_sketch) / float(n)

        # Sketched identity target: S @ I @ S^T = S @ S^T
        target = s @ s.t()

        # Frobenius norm of difference
        loss = torch.norm(cov_sketch - target, p="fro") ** 2

        return loss


class RDMReg:
    """Rectified Distribution Matching Regularization (Rectified LpJEPA).

    Matches embeddings to a Rectified Generalized Gaussian (RGG) distribution
    for sparse, non-negative representations using sliced 2-Wasserstein distance.

    Generalizes SIGReg: when target_sparsity=0 and p=2, recovers dense
    Gaussian matching.

    Reference: Rectified LpJEPA (arXiv:2602.01456)

    Args:
        p: Shape parameter of the Generalized Gaussian (2.0 = Gaussian,
            1.0 = Laplacian). Default 2.0.
        target_sparsity: Expected fraction of zero entries in [0, 1).
            Controls the mu parameter of the RGG. Default 0.0 (dense).
        num_projections: Number of random slicing directions. Default 64.
    """

    def __init__(
        self,
        p: float = 2.0,
        target_sparsity: float = 0.0,
        num_projections: int = 64,
    ) -> None:
        _require_torch()
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        self.p = p
        self.target_sparsity = target_sparsity
        self.num_projections = num_projections

    def _sample_rgg(self, n: int, d: int, device: torch.device) -> Tensor:
        """Sample from the Rectified Generalized Gaussian distribution.

        RGG mixes a Dirac at zero with a truncated generalized Gaussian on (0, inf).
        The mu parameter controls sparsity via the CDF.
        """
        if self.p == 2.0:
            # Standard Gaussian case
            samples = torch.randn(n, d, device=device)
        elif self.p == 1.0:
            # Laplacian case
            u = torch.rand(n, d, device=device).clamp(min=1e-8)
            samples = -torch.sign(torch.rand(n, d, device=device) - 0.5) * torch.log(u)
        else:
            # General case: use inverse CDF sampling approximation
            # Sample from standard normal and transform
            samples = torch.randn(n, d, device=device)
            samples = torch.sign(samples) * torch.abs(samples).pow(2.0 / self.p)

        # Apply rectification based on target_sparsity
        if self.target_sparsity > 0:
            # Shift distribution so that target_sparsity fraction falls below zero
            # For Gaussian: mu = -Phi^{-1}(1 - sparsity) where Phi is normal CDF
            # Approximate using probit function
            mu = -math.sqrt(2) * torch.erfinv(torch.tensor(1.0 - 2.0 * self.target_sparsity)).item()
            samples = samples + mu

        # Rectify: apply ReLU (max(0, x))
        samples = torch.relu(samples)

        return samples

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        """Compute RDMReg loss using sliced 2-Wasserstein distance.

        Args:
            embeddings: Tensor of shape (N, d). Will be rectified internally.

        Returns:
            Scalar loss tensor.
        """
        _require_torch()
        n, d = embeddings.shape

        # Rectify embeddings
        z_rect = torch.relu(embeddings)

        # Sample target from RGG distribution
        y_target = self._sample_rgg(n, d, embeddings.device)

        # Sample random projection directions
        directions = torch.randn(d, self.num_projections, device=embeddings.device)
        directions = directions / directions.norm(dim=0, keepdim=True)

        # Project both embeddings and target
        z_proj = z_rect @ directions  # (N, M)
        y_proj = y_target @ directions  # (N, M)

        # Sliced Wasserstein: sort and compare
        z_sorted, _ = torch.sort(z_proj, dim=0)
        y_sorted, _ = torch.sort(y_proj, dim=0)

        # L2 distance between sorted projections
        loss = ((z_sorted - y_sorted) ** 2).mean()

        return loss


class NormReg:
    """Representation-norm regularization (BiJEPA).

    Penalizes deviation of per-sample squared L2 norms from the isotropic
    Gaussian expectation E[||z||^2] = d. Bidirectional/symmetric prediction
    (no stop-gradient, no EMA teacher) admits a trivial escape where both
    predictors inflate embedding norms instead of learning structure —
    "representation explosion". NormReg closes that escape without
    constraining the distribution's shape, so it composes with SIGReg /
    WeakSIGReg / RDMReg rather than competing with them.

    Loss = mean_n ( ||z_n||^2 / d - 1 )^2

    Reference: BiJEPA (arXiv:2603.00049)

    Args:
        target_scale: Target for the normalized squared norm ||z||^2 / d.
            Default 1.0 (matches an isotropic standard Gaussian).
    """

    def __init__(self, target_scale: float = 1.0) -> None:
        _require_torch()
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive")
        self.target_scale = target_scale

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        """Compute NormReg loss on a batch of embeddings.

        Args:
            embeddings: Tensor of shape (N, d).

        Returns:
            Scalar loss tensor.
        """
        _require_torch()
        _, d = embeddings.shape
        normalized_sq_norms = embeddings.pow(2).sum(dim=1) / float(d)
        return ((normalized_sq_norms - self.target_scale) ** 2).mean()


def effective_rank(embeddings: Tensor) -> Tensor:
    """Return the exponential entropy of the covariance spectrum.

    Effective rank measures how evenly variance is distributed across feature
    directions. Zero means no variation; higher values indicate more active
    directions. Track it on both training and validation representations.
    """
    _require_torch()
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, d)")
    n, _ = embeddings.shape
    if n == 0:
        return embeddings.sum() * 0.0

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = (centered.t() @ centered) / float(n)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0.0)
    total = eigenvalues.sum()
    if total.item() <= 0.0:
        return total
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return entropy.exp()


class SACReg:
    """Spectral anti-collapse regularization for backbone representations.

    Implements the lambda-JEPA objective from arXiv:2609.35288, Definition 3.5:
    ``0.5 * (||mean||^2 + tr(cov) - logdet(cov + eps*I) - d)``.
    Apply it to the representation retained for downstream tasks, not only to
    the projected embeddings used by the prediction objective.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        _require_torch()
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        self.eps = eps

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        """Compute the SACReg loss on an ``(N, d)`` representation batch."""
        _require_torch()
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, d)")
        n, d = embeddings.shape
        if n == 0 or d == 0:
            return embeddings.sum() * 0.0

        mean = embeddings.mean(dim=0)
        centered = embeddings - mean
        covariance = (centered.t() @ centered) / float(n)
        stabilized = covariance + self.eps * torch.eye(d, device=embeddings.device, dtype=embeddings.dtype)
        _, logabsdet = torch.linalg.slogdet(stabilized)
        return 0.5 * (mean.pow(2).sum() + torch.trace(covariance) - logabsdet - float(d))


class SubSIGReg:
    """SIGReg applied to frozen row-orthonormal latent subspaces.

    Sub-JEPA (arXiv:2605.09241) relaxes the full-space isotropic Gaussian prior
    by regularizing multiple lower-dimensional views. This preserves the
    anti-collapse signal while allowing a lower intrinsic representation rank.
    """

    def __init__(
        self,
        num_subspaces: int = 4,
        subspace_dim: int = 16,
        num_directions: int = 64,
        sigma: float = 1.0,
        num_quadrature_points: int = 17,
        quadrature_range: float = 5.0,
    ) -> None:
        _require_torch()
        if num_subspaces < 1 or subspace_dim < 1:
            raise ValueError("num_subspaces and subspace_dim must be positive")
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        self.num_directions = num_directions
        self.sigma = sigma
        self.num_quadrature_points = num_quadrature_points
        self.quadrature_range = quadrature_range
        self._projections: dict[tuple[int, torch.device], Tensor] = {}

    def _get_projection(self, dim: int, device: torch.device) -> Tensor:
        if self.num_subspaces * self.subspace_dim > dim:
            raise ValueError("num_subspaces * subspace_dim must not exceed the embedding dimension")
        key = (dim, device)
        if key not in self._projections:
            width = self.num_subspaces * self.subspace_dim
            projection = torch.linalg.qr(torch.randn(dim, width, device=device)).Q
            self._projections[key] = projection
        return self._projections[key]

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        """Compute mean SIGReg loss across the frozen latent subspaces."""
        _require_torch()
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, d)")
        dim = embeddings.shape[1]
        projection = self._get_projection(dim, embeddings.device)
        regularizer = SIGReg(
            num_directions=self.num_directions,
            sigma=self.sigma,
            num_quadrature_points=self.num_quadrature_points,
            quadrature_range=self.quadrature_range,
        )
        losses = []
        for index in range(self.num_subspaces):
            start = index * self.subspace_dim
            subspace = embeddings @ projection[:, start : start + self.subspace_dim]
            losses.append(regularizer.compute_loss(subspace))
        return torch.stack(losses).mean()


class TemporalDifferenceReg:
    """Motion-aware auxiliary loss for sequential predictions.

    Combines the ordinary next-state MSE with an MSE over embedding
    differences. The second term prevents a static representation from looking
    artificially accurate when the target sequence is changing.
    """

    def __init__(self, delta_weight: float = 1.0) -> None:
        _require_torch()
        if delta_weight < 0.0:
            raise ValueError("delta_weight must be non-negative")
        self.delta_weight = delta_weight

    def compute_loss(self, current: Tensor, predicted_next: Tensor, target_next: Tensor) -> Tensor:
        """Compute next-state and temporal-difference losses on ``(N, T, D)`` tensors."""
        _require_torch()
        if current.ndim != 3 or predicted_next.ndim != 3 or target_next.ndim != 3:
            raise ValueError("current, predicted_next, and target_next must have shape (N, T, D)")
        if predicted_next.shape != target_next.shape:
            raise ValueError("predicted_next and target_next must have identical shapes")
        if current.shape[0] != predicted_next.shape[0] or current.shape[2] != predicted_next.shape[2]:
            raise ValueError("current must match prediction batch size and embedding dimension")

        base_loss = torch.nn.functional.mse_loss(predicted_next, target_next)
        if current.shape[1] == 1:
            predicted_delta = predicted_next - current
            target_delta = target_next - current
        else:
            predicted_delta = predicted_next[:, 1:] - current[:, :-1]
            target_delta = target_next[:, 1:] - current[:, :-1]
        return base_loss + self.delta_weight * torch.nn.functional.mse_loss(predicted_delta, target_delta)


class CorrelatedSIGReg:
    """Covariance regularization with a correlated Gaussian target.

    MM-LeJEPA (arXiv:2610.24681) motivates cross-modal covariance rather than
    independent isotropic targets. ``rho`` controls the shared off-diagonal
    covariance; zero recovers an isotropic Gaussian target.
    """

    def __init__(self, rho: float = 0.5) -> None:
        _require_torch()
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        self.rho = rho

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        """Compute full-covariance squared Frobenius distance to the target."""
        _require_torch()
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, d)")
        n, d = embeddings.shape
        if n == 0 or d == 0:
            return embeddings.sum() * 0.0

        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        covariance = (centered.t() @ centered) / float(n)
        target = torch.full((d, d), self.rho, device=embeddings.device, dtype=embeddings.dtype)
        target.fill_diagonal_(1.0)
        return torch.norm(covariance - target, p="fro") ** 2
