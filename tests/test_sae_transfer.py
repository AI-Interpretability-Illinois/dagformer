"""The SAE transfer controls and uncertainty retain their intended units."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_sae_transfer import permute_whole_heads, permute_within_streams, ratio_summary
from interp_editing import layer_chunks, stream_slices
from summarize_sae_transfer import span_summary


def test_masked_loss_is_weighted_by_tokens_not_windows():
    reference = np.full((2, 4), 3.)
    delta = np.array([[1., 0., 0., 0.], [2., 2., 2., 0.]])
    mask = np.array([[True, False, False, False], [True, True, True, False]])
    indexes = np.array([[0, 0], [0, 1], [1, 1]])
    result = ratio_summary(delta, reference, mask, indexes)
    assert result["delta"] == pytest.approx(7 / 4)
    assert result["reference_nll"] == 3.
    assert result["edited_nll"] == pytest.approx(4.75)
    assert result["window_token_count"] == [1, 3]
    assert result["n_tokens"] == 4


def test_random_direction_preserves_each_layer_and_stream_multiset():
    # Three routed layers, two heads: Q/K/V/R widths are 4/4/4/2,
    # then 6/6/6/3, then 8/8/8/4. A global shuffle would fail this check.
    direction = torch.arange(63, dtype=torch.float32) - 31
    control = permute_within_streams(direction, [(0, 14), (14, 35), (35, 63)], 2, 42)
    boundaries = [0, 4, 8, 12, 14, 20, 26, 32, 35, 43, 51, 59, 63]
    for start, stop in zip(boundaries, boundaries[1:]):
        assert torch.equal(control[start:stop].sort().values, direction[start:stop].sort().values)
    assert not torch.equal(control, direction)


def test_dose_span_keeps_pairing_and_target_token_weights():
    # Both dose effects vary across windows, but their paired difference is
    # exactly two nats per target token in each window.
    minus = {"window_token_count": [1, 3], "window_delta_sum": [-5., 12.]}
    plus = {"window_token_count": [1, 3], "window_delta_sum": [-3., 18.]}
    result = span_summary(minus, plus, draws=100, seed=42)
    assert result["delta"] == 2.
    assert result["paired_bootstrap_95ci"] == [2., 2.]


def test_whole_head_control_preserves_sources_qkv_alignment_and_residual():
    chunks = layer_chunks(4, 4)
    direction = torch.arange(chunks[-1][1], dtype=torch.float32)
    control = permute_whole_heads(direction, chunks, 4, 42)
    assert not torch.equal(control, direction)
    for layer, (start, _) in enumerate(chunks, 1):
        slices, n_sources = stream_slices(layer, 4)
        permutations = []
        for stream in ("q", "k", "v"):
            lo, hi = slices[stream]
            original = direction[start + lo:start + hi].reshape(4, n_sources)
            actual = control[start + lo:start + hi].reshape(4, n_sources)
            torch.testing.assert_close(actual.sort(dim=0).values, original.sort(dim=0).values)
            permutations.append((actual[:, 0] - original[0, 0]) / n_sources)
        assert torch.equal(permutations[0], permutations[1])
        assert torch.equal(permutations[1], permutations[2])
        lo, hi = slices["r"]
        assert torch.equal(control[start + lo:start + hi], direction[start + lo:start + hi])
