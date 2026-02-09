"""Tests for the structure predictor components (no GPU or model loading required)."""

import pytest
import torch
import torch.nn as nn

from src.model.predictor import (
    PredictorMLP,
    cascading_gate,
    gumbel_sigmoid,
)
from src.model.olmo_graph import create_block_upper_triangular_mask


class TestPredictorMLP:
    """Test MLP decoder shapes and gradient flow."""

    def setup_method(self):
        self.batch = 2
        self.input_dim = 1024  # Qwen embed_dim
        self.hidden_dim = 256  # small for testing
        self.rank = 8
        self.mlp = PredictorMLP(self.input_dim, self.hidden_dim, self.rank)

    def test_output_shape(self):
        e = torch.randn(self.batch, self.input_dim)
        Z = self.mlp(e)
        assert Z.shape == (self.batch, 256, 256)

    def test_low_rank_structure(self):
        """Z - logit_bias = UV^T should have rank <= r."""
        e = torch.randn(1, self.input_dim)
        Z = self.mlp(e)
        Z_2d = Z.squeeze(0)
        # Subtract the scalar logit_bias (constant across all entries)
        # so we test the rank of UV^T alone
        Z_no_bias = Z_2d - self.mlp.logit_bias.detach()
        S = torch.linalg.svdvals(Z_no_bias)
        # Values beyond rank r should be ~0 (up to numerical precision)
        assert S[self.rank:].abs().max() < 0.05, \
            f"UV^T has effective rank > {self.rank}: max singular value beyond rank = {S[self.rank:].abs().max()}"

    def test_gradient_flow(self):
        e = torch.randn(self.batch, self.input_dim)
        Z = self.mlp(e)
        loss = Z.sum()
        loss.backward()
        for name, p in self.mlp.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_batch_independence(self):
        """Different inputs should produce different outputs."""
        e1 = torch.randn(1, self.input_dim)
        e2 = torch.randn(1, self.input_dim)
        Z1 = self.mlp(e1)
        Z2 = self.mlp(e2)
        assert not torch.allclose(Z1, Z2), "Different inputs produced identical Z"


class TestGumbelSigmoid:
    """Test Gumbel-Sigmoid in all 3 modes."""

    def setup_method(self):
        self.batch = 2
        mask = create_block_upper_triangular_mask()
        # Create Z_masked with valid structure
        Z = torch.randn(self.batch, 256, 256)
        self.Z_masked = Z * mask.unsqueeze(0) + (-1e9) * (1 - mask.unsqueeze(0))
        self.tau = 2.0

    def test_train_mode_range(self):
        A = gumbel_sigmoid(self.Z_masked, self.tau, mode="train")
        assert A.shape == (self.batch, 256, 256)
        assert (A >= 0).all() and (A <= 1).all(), "Train mode values out of [0, 1]"

    def test_train_mode_stochastic(self):
        """Two calls with same input should give different results (stochastic)."""
        A1 = gumbel_sigmoid(self.Z_masked, self.tau, mode="train")
        A2 = gumbel_sigmoid(self.Z_masked, self.tau, mode="train")
        assert not torch.allclose(A1, A2), "Train mode is deterministic (should be stochastic)"

    def test_eval_soft_range(self):
        A = gumbel_sigmoid(self.Z_masked, self.tau, mode="eval_soft")
        assert (A >= 0).all() and (A <= 1).all(), "Eval soft values out of [0, 1]"

    def test_eval_soft_deterministic(self):
        A1 = gumbel_sigmoid(self.Z_masked, self.tau, mode="eval_soft")
        A2 = gumbel_sigmoid(self.Z_masked, self.tau, mode="eval_soft")
        assert torch.allclose(A1, A2), "Eval soft is not deterministic"

    def test_eval_hard_binary(self):
        A = gumbel_sigmoid(self.Z_masked, self.tau, mode="eval_hard")
        unique_values = A.unique()
        assert all(v in [0.0, 1.0] for v in unique_values), \
            f"Eval hard should produce binary 0/1, got {unique_values}"

    def test_eval_hard_deterministic(self):
        A1 = gumbel_sigmoid(self.Z_masked, self.tau, mode="eval_hard")
        A2 = gumbel_sigmoid(self.Z_masked, self.tau, mode="eval_hard")
        assert torch.allclose(A1, A2), "Eval hard is not deterministic"

    def test_invalid_positions_zero(self):
        """Invalid positions (same/backward layer) should be ~0 in all modes."""
        mask = create_block_upper_triangular_mask()
        invalid_mask = (1 - mask).bool()
        for mode in ["train", "eval_soft", "eval_hard"]:
            A = gumbel_sigmoid(self.Z_masked, self.tau, mode=mode)
            invalid_vals = A[0][invalid_mask]
            assert (invalid_vals < 1e-6).all(), \
                f"Invalid positions not zero in {mode}: max={invalid_vals.max()}"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            gumbel_sigmoid(self.Z_masked, self.tau, mode="unknown")

    def test_temperature_effect(self):
        """Lower temperature → sharper distribution (closer to binary)."""
        A_high_tau = gumbel_sigmoid(self.Z_masked, tau=10.0, mode="eval_soft")
        A_low_tau = gumbel_sigmoid(self.Z_masked, tau=0.1, mode="eval_soft")
        mask = create_block_upper_triangular_mask().bool()
        # Low tau should be more extreme (values closer to 0 or 1)
        valid_high = A_high_tau[0][mask]
        valid_low = A_low_tau[0][mask]
        # Measure "sharpness": distance from 0.5
        sharp_high = (valid_high - 0.5).abs().mean()
        sharp_low = (valid_low - 0.5).abs().mean()
        assert sharp_low > sharp_high, \
            f"Lower tau should be sharper: sharp_low={sharp_low:.4f}, sharp_high={sharp_high:.4f}"

    def test_gradient_through_train_mode(self):
        """Gradients should flow through Gumbel-Sigmoid in train mode."""
        Z = torch.randn(1, 256, 256, requires_grad=True)
        mask = create_block_upper_triangular_mask()
        Z_masked = Z * mask + (-1e9) * (1 - mask)
        A = gumbel_sigmoid(Z_masked, tau=2.0, mode="train")
        loss = A.sum()
        loss.backward()
        assert Z.grad is not None
        # Gradients should be nonzero at valid positions
        valid_grads = Z.grad[0][mask.bool()]
        assert (valid_grads != 0).any(), "No nonzero gradients at valid positions"


class TestCascadingGate:
    """Test cascading activation gate."""

    def setup_method(self):
        self.batch = 2

    def test_output_shape(self):
        A = torch.rand(self.batch, 256, 256)
        A_gated = cascading_gate(A, k=5.0, hard=False)
        assert A_gated.shape == A.shape

    def test_soft_mode_range(self):
        A = torch.rand(self.batch, 256, 256)
        A_gated = cascading_gate(A, k=5.0, hard=False)
        assert (A_gated >= 0).all() and (A_gated <= 1).all()

    def test_hard_mode_kills_disconnected(self):
        """Nodes with no incoming edges should have all outgoing edges zeroed."""
        A = torch.zeros(1, 256, 256)
        # Only set edges from node 0 to node 16 (layer 0 → layer 1)
        A[0, 0, 16] = 1.0
        A_gated = cascading_gate(A, k=5.0, hard=True)
        # Node 0 has no incoming edges → its outgoing should be zeroed
        assert A_gated[0, 0, 16] == 0.0, "Node 0 has no incoming but wasn't gated to 0"
        # Node 16 has incoming from node 0 (but node 0 was gated to 0)
        # In one-pass mode, inc uses ORIGINAL A, so node 16 has inc > 0

    def test_hard_mode_preserves_connected(self):
        """Nodes with incoming edges keep their outgoing edges."""
        A = torch.zeros(1, 256, 256)
        # Set edges: node 0→16, node 16→32
        A[0, 0, 16] = 1.0
        A[0, 16, 32] = 1.0
        A_gated = cascading_gate(A, k=5.0, hard=True)
        # Node 16 has incoming (from 0) → g_16 = 1 → outgoing preserved
        assert A_gated[0, 16, 32] == 1.0

    def test_soft_mode_differentiable(self):
        A = torch.rand(1, 256, 256, requires_grad=True)
        A_gated = cascading_gate(A, k=5.0, hard=False)
        loss = A_gated.sum()
        loss.backward()
        assert A.grad is not None
        assert A.grad.abs().sum() > 0

    def test_all_zeros_all_killed(self):
        """If A is all zeros, cascading gate should keep it all zeros."""
        A = torch.zeros(1, 256, 256)
        A_gated = cascading_gate(A, k=5.0, hard=True)
        assert (A_gated == 0).all()

    def test_one_pass_uses_original(self):
        """Verify cascading gate uses original A for incoming sums (one-pass)."""
        # If it were iterative, node 0 being gated off would affect node 16's incoming
        # But one-pass uses original A, so node 16's incoming is computed from original
        A = torch.zeros(1, 256, 256)
        A[0, 0, 16] = 1.0  # 0 → 16
        A[0, 16, 32] = 1.0  # 16 → 32

        A_gated = cascading_gate(A, k=5.0, hard=True)
        # One-pass: inc[16] = A[:,16].sum() = A[0,16] = 1.0 (from original A)
        # g[16] = (inc[16] > 0) = 1.0
        # So A_gated[16, 32] = A[16, 32] * g[16] = 1.0 * 1.0 = 1.0
        assert A_gated[0, 16, 32] == 1.0
