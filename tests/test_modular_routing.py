"""CPU tests for src/model/modular_routing.py.

The invariant that matters most: with the predictor at its all-ones identity
init, the module-granular routed forward equals the vanilla OLMo-2 forward.
Then: a zeroed source column really deletes the module, column mass reports
it, shapes match the layout, gradients reach the predictor.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import Olmo2Config, Olmo2ForCausalLM  # noqa: E402

from src.model.modular_routing import (  # noqa: E402
    FourWayModularDAGFormer, FourWayModularPredictor, column_group_penalty,
    identity_modular_routing, modular_n_sources, modular_source_labels,
    module_importance_from_mass, source_column_mass,
)
from src.pruning import StructuredMasker  # noqa: E402

L, H, D, I, V = 3, 4, 32, 64, 128


def tiny_olmo() -> Olmo2ForCausalLM:
    cfg = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                      num_key_value_heads=H, intermediate_size=I, vocab_size=V,
                      tie_word_embeddings=True, max_position_embeddings=64)
    torch.manual_seed(1)
    return Olmo2ForCausalLM(cfg).eval()


def predictor(**kw) -> FourWayModularPredictor:
    torch.manual_seed(3)
    return FourWayModularPredictor(vocab_size=V, encoder_dim=16, encoder_layers=1,
                                   encoder_heads=2, max_seq_len=64, num_layers=L,
                                   num_heads=H, hidden_dim=16, **kw).eval()


@pytest.fixture
def ids():
    torch.manual_seed(2)
    return torch.randint(0, V, (2, 12))


def test_source_bookkeeping():
    assert [modular_n_sources(l) for l in range(4)] == [1, 3, 5, 7]
    assert modular_source_labels(2) == ["emb", "a0", "m0", "a1", "m1"]


@pytest.mark.parametrize("v_norm", [False])
def test_identity_init_matches_vanilla(ids, v_norm):
    base = tiny_olmo()
    ref = base(input_ids=ids).logits.float()
    fw = FourWayModularDAGFormer(base, num_layers=L, num_heads=H, use_v_norm=v_norm).eval()
    pred = predictor()
    with torch.no_grad():
        rw = pred(ids)
        out = fw(ids, rw).float()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()
    # explicit all-ones routing gives the same thing
    with torch.no_grad():
        out2 = fw(ids, identity_modular_routing(2, ids.shape[1], L, H, ids.device)).float()
    assert torch.allclose(out2, ref, atol=1e-4, rtol=1e-4)


def test_identity_init_matches_vanilla_with_corrections(ids):
    base = tiny_olmo()
    ref = base(input_ids=ids).logits.float()
    fw = FourWayModularDAGFormer(base, num_layers=L, num_heads=H, use_local_correction=True,
                                 correction_hidden=8).eval()
    with torch.no_grad():
        out = fw(ids, predictor()(ids)).float()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert len(fw.get_routing_parameters()) == 2 * (L - 1)


def test_predictor_layout(ids):
    pred = predictor()
    rw = pred(ids)
    B, T = ids.shape
    assert len(rw["q"]) == L - 1 and len(rw["m"]) == L
    for l in range(1, L):
        n = modular_n_sources(l)
        for s in ("q", "k", "v"):
            assert rw[s][l - 1].shape == (B, T, H, n)
        assert rw["r"][l - 1].shape == (B, T, n)
        assert rw["m"][l].shape == (B, T, n + 1)
    assert rw["m"][0].shape == (B, T, 2)
    assert rw["o"].shape == (B, T, 2 * L + 1)
    # identity init: everything is exactly 1
    for s in ("q", "k", "v", "r", "m"):
        for t in rw[s]:
            assert torch.equal(t, torch.ones_like(t))
    assert torch.equal(rw["o"], torch.ones_like(rw["o"]))


def test_zero_column_deletes_module(ids):
    """Zeroing every reader's weight on source a_1 == pruning attention block 1."""
    base = tiny_olmo()
    fw = FourWayModularDAGFormer(base, num_layers=L, num_heads=H).eval()
    B, T = ids.shape
    rw = identity_modular_routing(B, T, L, H, ids.device)
    src = 1 + 2 * 1          # a_1
    with torch.no_grad():
        for l in range(2, L):              # readers of a_1 are layers >= 2
            for s in ("q", "k", "v"):
                rw[s][l - 1][..., src] = 0.0
            rw["r"][l - 1][..., src] = 0.0
            rw["m"][l][..., src] = 0.0
        rw["m"][1][..., modular_n_sources(1)] = 0.0    # layer 1's own MLP reads a_1 as its last source
        rw["o"][..., src] = 0.0
        routed = fw(ids, rw).float()

        masker = StructuredMasker(base, unit_types=("attn",))
        masker.masks["attn"][1, 0] = 0.0
        pruned = fw(ids, identity_modular_routing(B, T, L, H, ids.device)).float()
    assert torch.allclose(routed, pruned, atol=1e-4), (routed - pruned).abs().max()
    mass = source_column_mass(rw, L)
    labels = modular_source_labels(L)
    assert mass[labels.index("a1")] == 0.0
    assert (mass[[i for i in range(len(labels)) if labels[i] != "a1"]] > 0).all()
    imp = module_importance_from_mass(mass, L)
    assert imp["attn"].shape == (L, 1) and imp["mlp"].shape == (L, 1)
    assert imp["attn"][1, 0] == 0.0 and imp["attn"][0, 0] > 0


def test_column_mass_counts_every_reader():
    B, T = 1, 1
    rw = identity_modular_routing(B, T, L, H, "cpu")
    mass = source_column_mass(rw, L, power=1)
    # readers of the embedding: q/k/v (H each) + r for l>=1, m for all l, o once
    expected_emb = sum(3 * H + 1 for _ in range(1, L)) + L + 1
    assert mass[0].item() == expected_emb
    # last MLP output m_{L-1} is read only by the read-out
    assert mass[-1].item() == 1.0
    # penalty is finite and positive at identity
    assert column_group_penalty(rw, L) > 0


def test_gradients_reach_predictor_and_backbone(ids):
    base = tiny_olmo().train()
    fw = FourWayModularDAGFormer(base, num_layers=L, num_heads=H).train()
    pred = predictor().train()
    logits = fw(ids, pred(ids))
    loss = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
    loss.backward()
    for l, head in enumerate(pred.layer_heads):
        assert head.weight.grad is not None and (head.weight.grad != 0).any(), l
        assert pred.layer_biases[l].grad is not None
    assert (pred.out_head.weight.grad != 0).any()
    assert base.model.layers[0].mlp.down_proj.weight.grad is not None


def test_masker_hooks_apply_inside_modular_forward(ids):
    base = tiny_olmo()
    fw = FourWayModularDAGFormer(base, num_layers=L, num_heads=H).eval()
    B, T = ids.shape
    rw = identity_modular_routing(B, T, L, H, ids.device)
    masker = StructuredMasker(base, unit_types=("head", "neuron", "attn", "mlp"))
    with torch.no_grad():
        ref = fw(ids, rw).float()
        masker.masks["head"][1, 0] = 0.0
        masker.masks["neuron"][2, :8] = 0.0
        masked = fw(ids, rw).float()
    assert not torch.allclose(ref, masked)
    fw.train()
    loss = F.cross_entropy(fw(ids, rw).reshape(-1, V), ids.reshape(-1))
    loss.backward()
    for t in masker.unit_types:
        assert masker.gates[t].grad is not None and torch.isfinite(masker.gates[t].grad).all()
