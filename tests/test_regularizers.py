"""Tests for JEPA regularizers (SIGReg, WeakSIGReg, RDMReg).

Skipped entirely if PyTorch is not available.
"""

import pytest

torch = pytest.importorskip("torch")

from event_jepa_cube.regularizers import (  # noqa: E402
    CorrelatedSIGReg,
    NormReg,
    RDMReg,
    SACReg,
    SIGReg,
    SubSIGReg,
    TemporalDifferenceReg,
    WeakSIGReg,
    effective_rank,
)


class TestSIGReg:
    """Tests for SIGReg regularizer."""

    def test_sigreg_gaussian_low_loss(self):
        torch.manual_seed(42)
        reg = SIGReg(num_directions=32)
        gaussian_emb = torch.randn(64, 16)
        loss_gaussian = reg.compute_loss(gaussian_emb)

        collapsed_emb = torch.ones(64, 16)
        loss_collapsed = reg.compute_loss(collapsed_emb)

        # Gaussian embeddings should have lower loss than collapsed
        assert loss_gaussian.item() < loss_collapsed.item()

    def test_sigreg_collapsed_high_loss(self):
        torch.manual_seed(0)
        reg = SIGReg(num_directions=32)
        # All-same embeddings (add tiny noise to avoid zero std in centering)
        collapsed = torch.ones(64, 16) + torch.randn(64, 16) * 1e-6
        loss = reg.compute_loss(collapsed)
        # Collapsed embeddings should have non-trivially high loss
        assert loss.item() > 0.0

    def test_sigreg_output_scalar(self):
        torch.manual_seed(1)
        reg = SIGReg(num_directions=8)
        emb = torch.randn(32, 8)
        loss = reg.compute_loss(emb)
        assert loss.dim() == 0  # scalar tensor


class TestWeakSIGReg:
    """Tests for WeakSIGReg regularizer."""

    def test_weak_sigreg_identity_covariance(self):
        torch.manual_seed(42)
        reg = WeakSIGReg(sketch_dim=32)
        # Embeddings drawn from isotropic Gaussian (covariance ~ I)
        good_emb = torch.randn(256, 16)
        loss_good = reg.compute_loss(good_emb)

        # Collapsed embeddings (covariance far from I)
        collapsed = torch.ones(256, 16)
        loss_collapsed = reg.compute_loss(collapsed)

        assert loss_good.item() < loss_collapsed.item()

    def test_weak_sigreg_output_scalar(self):
        torch.manual_seed(2)
        reg = WeakSIGReg(sketch_dim=8)
        emb = torch.randn(32, 8)
        loss = reg.compute_loss(emb)
        assert loss.dim() == 0


class TestRDMReg:
    """Tests for RDMReg regularizer."""

    def test_rdmreg_output_scalar(self):
        torch.manual_seed(3)
        reg = RDMReg(num_projections=16)
        emb = torch.randn(32, 8)
        loss = reg.compute_loss(emb)
        assert loss.dim() == 0

    def test_rdmreg_sparsity_validation(self):
        with pytest.raises(ValueError, match="target_sparsity"):
            RDMReg(target_sparsity=1.0)
        with pytest.raises(ValueError, match="target_sparsity"):
            RDMReg(target_sparsity=-0.1)

    def test_rdmreg_dense_vs_sparse(self):
        torch.manual_seed(42)
        reg = RDMReg(target_sparsity=0.5, num_projections=32)
        # Gaussian embeddings passed through ReLU internally
        emb = torch.randn(64, 16)
        loss = reg.compute_loss(emb)
        # After ReLU on Gaussian, roughly half should be zero
        rectified = torch.relu(emb)
        zero_fraction = (rectified == 0).float().mean().item()
        assert zero_fraction > 0.3  # some zeros should exist
        assert loss.item() >= 0.0


class TestNormReg:
    """Tests for NormReg regularizer.

    NormReg exists to close the representation-explosion escape in symmetric
    (bidirectional) prediction: inflating norms must cost more than learning
    structure, while well-scaled Gaussian embeddings stay near-free.
    """

    def test_normreg_gaussian_low_exploded_high(self):
        torch.manual_seed(42)
        reg = NormReg()
        gaussian_emb = torch.randn(256, 32)
        exploded_emb = gaussian_emb * 10.0

        loss_gaussian = reg.compute_loss(gaussian_emb)
        loss_exploded = reg.compute_loss(exploded_emb)

        # Norm inflation must be penalized far more than unit-scale embeddings,
        # otherwise NormReg cannot prevent representation explosion.
        assert loss_exploded.item() > loss_gaussian.item() * 100

    def test_normreg_collapsed_to_zero_penalized(self):
        # Collapse-to-zero (all norms 0) is the other trivial escape; it must
        # not be free either: loss -> target_scale^2.
        reg = NormReg()
        zero_emb = torch.zeros(64, 16)
        loss = reg.compute_loss(zero_emb)
        assert abs(loss.item() - 1.0) < 1e-6

    def test_normreg_output_scalar_and_grad(self):
        torch.manual_seed(1)
        reg = NormReg()
        emb = torch.randn(32, 8, requires_grad=True)
        loss = reg.compute_loss(emb)
        assert loss.dim() == 0
        loss.backward()
        assert emb.grad is not None

    def test_normreg_target_scale_validation(self):
        with pytest.raises(ValueError, match="target_scale"):
            NormReg(target_scale=0.0)
        with pytest.raises(ValueError, match="target_scale"):
            NormReg(target_scale=-1.0)


class TestRepresentationHealth:
    def test_effective_rank_detects_collapse(self):
        torch.manual_seed(42)
        full = effective_rank(torch.randn(128, 8))
        collapsed = effective_rank(torch.ones(128, 8))

        assert full.item() > 1.0
        assert collapsed.item() == pytest.approx(0.0)

    def test_sacreg_penalizes_collapse_and_mean_shift(self):
        torch.manual_seed(42)
        reg = SACReg(eps=1e-4)
        embeddings = torch.randn(128, 8)

        centered_loss = reg.compute_loss(embeddings)
        collapsed_loss = reg.compute_loss(torch.ones(128, 8))
        shifted_loss = reg.compute_loss(embeddings + 5.0)

        assert centered_loss.item() < collapsed_loss.item()
        assert shifted_loss.item() > centered_loss.item()

    def test_sacreg_is_differentiable(self):
        reg = SACReg()
        embeddings = torch.randn(32, 4, requires_grad=True)

        loss = reg.compute_loss(embeddings)
        loss.backward()

        assert loss.dim() == 0
        assert embeddings.grad is not None

    def test_sub_sigreg_checks_shape_and_penalizes_collapse(self):
        torch.manual_seed(42)
        reg = SubSIGReg(num_subspaces=2, subspace_dim=4, num_directions=8)
        embeddings = torch.randn(64, 8)

        assert reg.compute_loss(embeddings).dim() == 0
        assert reg.compute_loss(embeddings).item() < reg.compute_loss(torch.ones(64, 8)).item()
        with pytest.raises(ValueError, match="embedding dimension"):
            reg.compute_loss(torch.randn(64, 4))


class TestTemporalAndCorrelatedObjectives:
    def test_temporal_difference_penalizes_static_predictions(self):
        reg = TemporalDifferenceReg(delta_weight=1.0)
        current = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
        target = current + 1.0

        correct = reg.compute_loss(current, target, target)
        static = reg.compute_loss(current, current, target)

        assert correct.item() == pytest.approx(0.0)
        assert static.item() > 0.0

    def test_temporal_difference_handles_single_event_history(self):
        reg = TemporalDifferenceReg(delta_weight=1.0)
        current = torch.tensor([[[0.0, 0.0]]])
        target = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])

        correct = reg.compute_loss(current, target, target)
        static = reg.compute_loss(current, current.repeat(1, 2, 1), target)

        assert correct.item() == pytest.approx(0.0)
        assert static.item() > 0.0

    def test_correlated_target_penalizes_independent_dimensions_more(self):
        independent = torch.eye(4)
        target = torch.full((4, 4), 0.5).fill_diagonal_(1.0)
        factor = torch.linalg.cholesky(target).t()
        correlated = factor - factor.mean(dim=0, keepdim=True)
        reg = CorrelatedSIGReg(rho=0.5)

        independent_loss = reg.compute_loss(independent)
        correlated_loss = reg.compute_loss(correlated)

        assert correlated_loss.item() < independent_loss.item()

    def test_correlated_reg_rejects_extreme_correlation(self):
        with pytest.raises(ValueError, match="rho"):
            CorrelatedSIGReg(rho=1.0)
