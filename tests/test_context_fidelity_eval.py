"""Check the intervention's algebra and discovery/evaluation separation."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_context_fidelity import apply_edges, heldout_items, matched_edges
from interp_liar_cloze import build_items


def test_simultaneous_edits_use_unmodified_mean_and_commute():
    # Two edited heads share one source: sequentially recomputing the mean
    # would give [4, 5.333, 8], while the simultaneous edit must give [4, 4, 8].
    chunk = torch.zeros(1, 1, (3 * 3 + 1) * 2)
    chunk[0, 0, :6] = torch.tensor([0., 10., 4., 20., 8., 30.])
    edges = [(1, "q", 0, 0), (1, "q", 1, 0)]
    edited = apply_edges(chunk, 1, 3, edges, 0.)
    assert torch.equal(edited, apply_edges(chunk, 1, 3, edges[::-1], 0.))
    assert torch.equal(edited[0, 0, :6], torch.tensor([4., 10., 4., 20., 8., 30.]))
    assert chunk[0, 0, 0].item() == 0.
    assert torch.equal(apply_edges(chunk, 1, 3, edges, 1.), chunk)


def test_eval_items_are_disjoint_and_random_controls_preserve_coordinates():
    key = lambda item: (item["fact"], item["ask"], item["a"], item["b"])
    discovery = {key(it) for it in build_items(240, 0)}
    evaluation = heldout_items(128, 20260917)
    assert len({key(it) for it in evaluation}) == 128
    assert not discovery.intersection(key(it) for it in evaluation)
    edges = [(4, "q", 1, 0), (4, "q", 3, 0), (7, "v", 9, 4)]
    control = matched_edges(edges, 16, 1000)
    assert not set(edges).intersection(control)
    assert [(l, s, src) for l, s, h, src in edges] == [
        (l, s, src) for l, s, h, src in control]
