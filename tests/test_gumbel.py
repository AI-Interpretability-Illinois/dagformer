"""Tests specifically for Gumbel-Sigmoid correctness and mathematical properties."""

import pytest
import torch

from src.model.predictor import gumbel_sigmoid
from src.model.olmo_graph import create_block_upper_triangular_mask


class TestGumbelSigmoidMath:
    """Test mathematical properties of Gumbel-Sigmoid."""

    def test_logistic_noise_distribution(self):
        """Gumbel noise G = log(U) - log(1-U) should follow Logistic(0,1)."""
        torch.manual_seed(42)
        n = 100_000
        U = torch.rand(n).clamp(1e-8, 1 - 1e-8)
        G = torch.log(U) - torch.log(1 - U)
        # Logistic(0,1) has mean=0, variance=pi^2/3
        assert abs(G.mean().item()) < 0.05, f"Logistic noise mean should be ~0, got {G.mean():.4f}"
        expected_var = torch.pi ** 2 / 3
        assert abs(G.var().item() - expected_var) < 0.1, \
            f"Logistic noise var should be ~{expected_var:.4f}, got {G.var():.4f}"

    def test_sigmoid_saturation_at_large_logits(self):
        """Large positive logits → A ≈ 1, large negative → A ≈ 0."""
        Z = torch.tensor([[[100.0, -100.0]]])
        A = gumbel_sigmoid(Z, tau=1.0, mode="eval_soft")
        assert A[0, 0, 0] > 0.999
        assert A[0, 0, 1] < 0.001

    def test_zero_logit_gives_half(self):
        """σ(0/τ) = 0.5 for any τ."""
        Z = torch.tensor([[[0.0]]])
        A = gumbel_sigmoid(Z, tau=1.0, mode="eval_soft")
        assert abs(A[0, 0, 0].item() - 0.5) < 1e-6

    def test_hard_threshold_at_zero(self):
        """Hard mode thresholds at logit=0 (prob=0.5)."""
        Z = torch.tensor([[[0.1, -0.1, 0.0]]])
        A = gumbel_sigmoid(Z, tau=1.0, mode="eval_hard")
        assert A[0, 0, 0] == 1.0  # > 0
        assert A[0, 0, 1] == 0.0  # < 0
        assert A[0, 0, 2] == 0.0  # = 0 → not > 0

    def test_train_mean_converges_to_sigmoid(self):
        """With many samples, training mode mean should converge to σ(Z/τ)."""
        torch.manual_seed(0)
        Z = torch.tensor([[[1.5]]])
        tau = 2.0
        n_samples = 10_000
        samples = torch.stack([gumbel_sigmoid(Z, tau, mode="train") for _ in range(n_samples)])
        empirical_mean = samples.mean().item()
        expected = torch.sigmoid(Z / tau).item()
        assert abs(empirical_mean - expected) < 0.05, \
            f"Empirical mean {empirical_mean:.4f} != σ(Z/τ) {expected:.4f}"

    def test_masked_positions_stay_zero(self):
        """After masking, invalid positions should be ~0 regardless of Z values."""
        mask = create_block_upper_triangular_mask()
        Z = torch.ones(1, 256, 256) * 10.0  # all high logits
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        for mode in ["train", "eval_soft", "eval_hard"]:
            A = gumbel_sigmoid(Z_masked, tau=1.0, mode=mode)
            invalid = A[0][~mask.bool()]
            assert (invalid < 1e-6).all(), \
                f"Invalid positions not zero in {mode}: max={invalid.max():.6f}"
