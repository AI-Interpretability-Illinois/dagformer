import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_copy_head_transfer import head_edges, matched_heads, next_token_metrics


def test_copy_metric_includes_first_repeated_target():
    rows = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    logits = torch.full((1, 8, 4), -20.)
    for position in range(7):
        logits[0, position, rows[0, position + 1]] = 20.
    assert next_token_metrics(logits, rows, 4)["accuracy"] == [1.]
    # Corrupt just the prediction of the first token of the second block.
    logits[0, 3] = torch.tensor([-20., 20., -20., -20.])
    assert next_token_metrics(logits, rows, 4)["accuracy"] == [pytest.approx(.75)]


def test_copy_controls_keep_whole_heads_and_layer_counts():
    named = [(3, 2), (3, 11), (3, 13), (4, 1), (4, 4), (6, 11)]
    controls = matched_heads(named, 16, 20260921)
    assert len(set(controls)) == len(named)
    assert not set(controls) & set(named)
    for layer in (3, 4, 6):
        assert sum(l == layer for l, _ in controls) == sum(l == layer for l, _ in named)
    edges = head_edges(controls)
    for layer, head in controls:
        assert {(s, src) for l, s, h, src in edges if l == layer and h == head} == {
            (s, src) for s in ("q", "k") for src in range(layer + 1)}
