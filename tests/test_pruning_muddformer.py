"""The masker on MUDDFormer (layerwise dynamic-dense routers): same guarantees
as on OLMo-2, with per-layer MLP widths."""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.muddformer.wrapper import build_muddformer  # noqa: E402
from src.pruning import StructuredMasker  # noqa: E402

L, H, D, I, V = 3, 4, 32, 256, 128   # MLP width scales with depth: 128 / 256 / 384


def tiny_mudd():
    cfg = types.SimpleNamespace(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                                intermediate_size=I, vocab_size=V, max_position_embeddings=64,
                                tie_word_embeddings=True)
    torch.manual_seed(1)
    m = build_muddformer(cfg).eval()
    with torch.no_grad():                       # non-identity routing so mixing matters
        for db in m.model.dense_bs:
            db.add_(0.1 * torch.randn_like(db))
    return m


@pytest.fixture
def ids():
    torch.manual_seed(2)
    return torch.randint(0, V, (2, 16))


def logits(m, ids):
    with torch.no_grad():
        return m(input_ids=ids).logits.float()


def test_adapter_and_identity(ids):
    m = tiny_mudd()
    ref = logits(m, ids)
    masker = StructuredMasker(m, unit_types=("head", "neuron", "attn", "mlp"))
    assert masker.arch.name == "muddformer"
    assert masker.I_per_layer == [layer.feed_forward.w2.weight.shape[1] for layer in m.model.layers]
    assert len(set(masker.I_per_layer)) > 1          # width scales with depth
    assert torch.equal(logits(m, ids), ref)
    assert masker.counts()["neuron"] == (0, sum(masker.I_per_layer))


def test_head_and_neuron_masks_are_exact(ids):
    m = tiny_mudd()
    masker = StructuredMasker(m, unit_types=("head", "neuron"))
    masker.masks["head"][1, 2] = 0.0
    masker.masks["neuron"][2, :5] = 0.0
    masked = logits(m, ids)
    masker.remove_hooks()
    hd = D // H
    with torch.no_grad():
        m.model.layers[1].attention.wo.weight[:, 2 * hd:3 * hd] = 0.0
        m.model.layers[2].feed_forward.w2.weight[:, :5] = 0.0
    assert torch.allclose(logits(m, ids), masked, atol=1e-5)


def test_block_masks_and_accounting(ids):
    m = tiny_mudd()
    masker = StructuredMasker(m, unit_types=("head", "neuron", "attn", "mlp"))
    ref = logits(m, ids)
    masker.masks["attn"][0, 0] = 0.0
    masker.masks["mlp"][2, 0] = 0.0
    out = logits(m, ids)
    assert not torch.allclose(out, ref)
    c0, c2 = masker.arch.unit_params(0), masker.arch.unit_params(2)
    assert masker.pruned_params() == c0["attn"] + c2["mlp"]
    layer_params = sum(p.numel() for p in m.model.layers[2].parameters())
    assert masker.arch.unit_params(2)["attn"] + c2["mlp"] == layer_params


def test_importance_and_prune_skip_padding(ids):
    m = tiny_mudd().train()
    masker = StructuredMasker(m, unit_types=("head", "neuron"))
    loss = F.cross_entropy(m(input_ids=ids).logits.reshape(-1, V), ids.reshape(-1))
    loss.backward()
    masker.accumulate_importance("taylor")
    newly = masker.prune_to({"neuron": 0.5, "head": 0.25}, criterion="taylor")
    assert newly["neuron"] == round(0.5 * sum(masker.I_per_layer))
    # nothing pruned in padding columns
    for l, w in enumerate(masker.I_per_layer):
        assert (masker.masks["neuron"][l, w:] == 1).all()
    assert newly["head"] == 3
