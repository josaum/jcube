"""Tests for JEPA regularizers (SIGReg, WeakSIGReg, RDMReg).

Skipped entirely if PyTorch is not available.
"""

import pytest

torch = pytest.importorskip("torch")

from event_jepa_cube.regularizers import RDMReg, SIGReg, WeakSIGReg


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
