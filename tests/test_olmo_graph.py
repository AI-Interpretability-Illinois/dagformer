"""Unit tests for olmo_graph.py.

Tests that don't require model download run with synthetic tensors.
Integration tests (baseline reproduction) require the model and are
skipped if model is not available.
"""

import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.olmo_graph import (
    create_block_upper_triangular_mask,
    InputNormalizer,
)


class TestBlockUpperTriangularMask:
    """Test the DAG constraint mask."""

    def test_shape(self):
        mask = create_block_upper_triangular_mask(256, 16)
        assert mask.shape == (256, 256)

    def test_dtype(self):
        mask = create_block_upper_triangular_mask(256, 16)
        assert mask.dtype == torch.float32

    def test_no_self_connections(self):
        """Diagonal should be 0 — a node cannot connect to itself."""
        mask = create_block_upper_triangular_mask(256, 16)
        assert mask.diag().sum() == 0

    def test_no_same_layer_connections(self):
        """Nodes in the same layer should NOT be connected."""
        mask = create_block_upper_triangular_mask(256, 16)
        for layer in range(16):
            start = layer * 16
            end = start + 16
            block = mask[start:end, start:end]
            assert block.sum() == 0, f"Layer {layer} has same-layer connections"

    def test_no_backward_connections(self):
        """No connections from higher layer to lower layer."""
        mask = create_block_upper_triangular_mask(256, 16)
        for src_layer in range(16):
            for tgt_layer in range(src_layer):  # tgt < src = backward
                src_start = src_layer * 16
                tgt_start = tgt_layer * 16
                block = mask[src_start:src_start+16, tgt_start:tgt_start+16]
                assert block.sum() == 0, f"Backward connection from layer {src_layer} to {tgt_layer}"

    def test_forward_connections_exist(self):
        """Forward connections (higher layer targets) should be 1."""
        mask = create_block_upper_triangular_mask(256, 16)
        for src_layer in range(15):
            for tgt_layer in range(src_layer + 1, 16):
                src_start = src_layer * 16
                tgt_start = tgt_layer * 16
                block = mask[src_start:src_start+16, tgt_start:tgt_start+16]
                assert block.sum() == 16 * 16, \
                    f"Missing connections from layer {src_layer} to layer {tgt_layer}"

    def test_total_valid_entries(self):
        """Should have exactly 30,720 valid entries."""
        mask = create_block_upper_triangular_mask(256, 16)
        assert mask.sum().item() == 30720

    def test_adjacent_connections_count(self):
        """Adjacent layer connections: 15 × 16 × 16 = 3840."""
        mask = create_block_upper_triangular_mask(256, 16)
        count = 0
        for src_layer in range(15):
            tgt_layer = src_layer + 1
            src_start = src_layer * 16
            tgt_start = tgt_layer * 16
            count += mask[src_start:src_start+16, tgt_start:tgt_start+16].sum().item()
        assert count == 3840

    def test_skip_connections_count(self):
        """Skip connections: 105 × 16 × 16 = 26880."""
        mask = create_block_upper_triangular_mask(256, 16)
        count = 0
        for src_layer in range(14):
            for tgt_layer in range(src_layer + 2, 16):
                src_start = src_layer * 16
                tgt_start = tgt_layer * 16
                count += mask[src_start:src_start+16, tgt_start:tgt_start+16].sum().item()
        assert count == 26880

    def test_not_torch_triu(self):
        """Verify this is NOT element-upper-triangular.

        torch.triu would set mask[0,15]=1 (both in layer 0), which is wrong.
        """
        mask = create_block_upper_triangular_mask(256, 16)
        # Node 0 (layer 0, head 0) to node 15 (layer 0, head 15)
        assert mask[0, 15] == 0, "Same-layer connection detected — did you use torch.triu()?"
        # Node 0 (layer 0, head 0) to node 16 (layer 1, head 0)
        assert mask[0, 16] == 1, "Adjacent-layer connection should be 1"


class TestInputNormalizer:
    """Test input normalization methods."""

    def test_none(self):
        norm = InputNormalizer("none")
        x = torch.randn(2, 16, 32, 2048)
        out = norm(x)
        assert torch.allclose(out, x)

    def test_gate_mean(self):
        norm = InputNormalizer("gate_mean")
        gated_sum = torch.randn(2, 16, 32, 2048)
        A_slice = torch.rand(2, 48, 16)  # 3 prior layers
        out = norm(gated_sum, A_slice=A_slice)
        assert out.shape == gated_sum.shape
        assert torch.isfinite(out).all()

    def test_rms_post(self):
        norm = InputNormalizer("rms_post", model_dim=2048)
        x = torch.randn(2, 16, 32, 2048)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_ln_post(self):
        norm = InputNormalizer("ln_post", model_dim=2048)
        x = torch.randn(2, 16, 32, 2048)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_rms_pre(self):
        norm = InputNormalizer("rms_pre", model_dim=64, num_nodes=32)  # small for test
        prior = torch.randn(2, 32, 8, 64)
        A_slice = torch.rand(2, 32, 4)
        gated_sum = torch.einsum('bih,bisd->bhsd', A_slice, prior)
        out = norm(gated_sum, A_slice=A_slice, prior_head_outs=prior)
        assert out.shape == gated_sum.shape
        assert torch.isfinite(out).all()

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown input_norm"):
            InputNormalizer("unknown_method")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
