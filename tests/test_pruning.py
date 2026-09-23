"""CPU tests for src/pruning: masks are exact, importance flows, schedule ramps.

Uses a tiny random OLMo-2 (2 layers, 4 heads) so everything runs in seconds
without a checkpoint. The dense HF forward and the FourWay routed forward are
both exercised because the masker must behave identically on the two.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import Olmo2Config, Olmo2ForCausalLM  # noqa: E402

from src.model.olmo_graph import FourWayDAGFormer  # noqa: E402
from src.model.predictor import FourWayPredictor  # noqa: E402
from src.pruning import (  # noqa: E402
    StructuredMasker, cubic_sparsity, prune_steps, unit_param_counts,
)
from src.pruning.masks import count_unique_params  # noqa: E402

torch.manual_seed(0)

L, H, D, I, V = 3, 4, 32, 64, 128


def tiny_olmo() -> Olmo2ForCausalLM:
    cfg = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                      num_key_value_heads=H, intermediate_size=I, vocab_size=V,
                      tie_word_embeddings=True, max_position_embeddings=64)
    torch.manual_seed(1)
    return Olmo2ForCausalLM(cfg).eval()


@pytest.fixture
def ids():
    torch.manual_seed(2)
    return torch.randint(0, V, (2, 16))


def dense_logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def fourway_pair():
    base = tiny_olmo()
    fw = FourWayDAGFormer(base, num_layers=L, num_heads=H).eval()
    torch.manual_seed(3)
    pred = FourWayPredictor(vocab_size=V, encoder_dim=16, encoder_layers=1,
                            encoder_heads=2, max_seq_len=64, num_layers=L,
                            num_heads=H, hidden_dim=16).eval()
    # perturb the predictor so routing is non-identity (W=0 init otherwise)
    with torch.no_grad():
        for head in pred.layer_heads:
            head.weight.normal_(0, 0.05)
    return fw, pred


def fourway_logits(fw, pred, ids):
    with torch.no_grad():
        return fw(ids, pred(ids)).float()


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------

def test_all_ones_masks_are_identity_dense(ids):
    model = tiny_olmo()
    ref = dense_logits(model, ids)
    masker = StructuredMasker(model, unit_types=("head", "neuron", "attn", "mlp"))
    out = dense_logits(model, ids)
    assert torch.equal(out, ref)
    masker.remove_hooks()
    assert torch.equal(dense_logits(model, ids), ref)


def test_all_ones_masks_are_identity_fourway(ids):
    fw, pred = fourway_pair()
    ref = fourway_logits(fw, pred, ids)
    StructuredMasker(fw.olmo, unit_types=("head", "neuron", "attn", "mlp"))
    out = fourway_logits(fw, pred, ids)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("routed", [False, True])
def test_head_mask_equals_zeroed_o_proj_columns(ids, routed):
    if routed:
        fw, pred = fourway_pair()
        model = fw.olmo
        run = lambda: fourway_logits(fw, pred, ids)  # noqa: E731
    else:
        model = tiny_olmo()
        run = lambda: dense_logits(model, ids)  # noqa: E731
    masker = StructuredMasker(model, unit_types=("head",))
    l, h = 1, 2
    masker.masks["head"][l, h] = 0.0
    masked = run()
    masker.remove_hooks()
    hd = D // H
    with torch.no_grad():
        model.model.layers[l].self_attn.o_proj.weight[:, h * hd:(h + 1) * hd] = 0.0
    surgical = run()
    assert torch.allclose(masked, surgical, atol=1e-5), (masked - surgical).abs().max()


@pytest.mark.parametrize("routed", [False, True])
def test_neuron_mask_equals_zeroed_down_proj_column(ids, routed):
    if routed:
        fw, pred = fourway_pair()
        model = fw.olmo
        run = lambda: fourway_logits(fw, pred, ids)  # noqa: E731
    else:
        model = tiny_olmo()
        run = lambda: dense_logits(model, ids)  # noqa: E731
    masker = StructuredMasker(model, unit_types=("neuron",))
    l, cols = 2, [0, 5, 17]
    masker.masks["neuron"][l, cols] = 0.0
    masked = run()
    masker.remove_hooks()
    with torch.no_grad():
        model.model.layers[l].mlp.down_proj.weight[:, cols] = 0.0
    surgical = run()
    assert torch.allclose(masked, surgical, atol=1e-5)


def test_module_masks_match_all_units_pruned(ids):
    """Pruning every head (neuron) of a layer == pruning the attn (mlp) block."""
    model = tiny_olmo()
    masker = StructuredMasker(model, unit_types=("head", "neuron", "attn", "mlp"))
    masker.masks["attn"][0, 0] = 0.0
    masker.masks["mlp"][2, 0] = 0.0
    by_module = dense_logits(model, ids)
    masker.masks["attn"][0, 0] = 1.0
    masker.masks["mlp"][2, 0] = 1.0
    masker.masks["head"][0, :] = 0.0
    masker.masks["neuron"][2, :] = 0.0
    by_units = dense_logits(model, ids)
    assert torch.allclose(by_module, by_units, atol=1e-5)
    # and it is not a no-op
    masker.masks["head"][0, :] = 1.0
    masker.masks["neuron"][2, :] = 1.0
    assert not torch.allclose(by_module, dense_logits(model, ids))


def test_disabled_masker_is_passthrough(ids):
    model = tiny_olmo()
    ref = dense_logits(model, ids)
    masker = StructuredMasker(model, unit_types=("head", "mlp"))
    masker.masks["head"][0, 0] = 0.0
    masker.enabled = False
    assert torch.equal(dense_logits(model, ids), ref)


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

def test_unit_param_counts_tile_the_model():
    model = tiny_olmo()
    c = unit_param_counts(model.config)
    layer_params = sum(p.numel() for p in model.model.layers[0].parameters())
    assert c["attn"] + c["mlp"] == layer_params
    assert c["head"] * H == 4 * D * D + 2 * D           # heads tile the projections + qk norms
    assert c["neuron"] * I == 3 * D * I                  # neurons tile the MLP matrices
    total = count_unique_params(model)
    assert total == V * D + L * layer_params + D          # tied embedding + layers + final norm


def test_pruned_params_is_structure_aware():
    model = tiny_olmo()
    masker = StructuredMasker(model, unit_types=("head", "neuron", "attn", "mlp"))
    c = unit_param_counts(model.config)
    assert masker.pruned_params() == 0
    masker.masks["head"][0, :2] = 0.0
    assert masker.pruned_params() == 2 * c["head"]
    masker.masks["attn"][0, 0] = 0.0                     # whole block: heads not double counted
    assert masker.pruned_params() == c["attn"]
    masker.masks["head"][1, :] = 0.0                     # all heads == block
    assert masker.pruned_params() == 2 * c["attn"]
    masker.masks["neuron"][2, :5] = 0.0
    masker.masks["mlp"][1, 0] = 0.0
    assert masker.pruned_params() == 2 * c["attn"] + 5 * c["neuron"] + c["mlp"]
    rep = masker.report()
    assert rep["prune/base_params_pruned"] == masker.pruned_params()
    assert 0 < rep["prune/base_param_sparsity"] < 1


# ---------------------------------------------------------------------------
# importance + pruning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("routed", [False, True])
def test_taylor_importance_flows_to_every_gate(ids, routed):
    if routed:
        fw, pred = fourway_pair()
        model = fw.olmo
        fw.train()
        forward = lambda: fw(ids, pred(ids))  # noqa: E731
    else:
        model = tiny_olmo().train()
        forward = lambda: model(input_ids=ids).logits  # noqa: E731
    masker = StructuredMasker(model, unit_types=("head", "neuron", "attn", "mlp"))
    logits = forward()
    loss = F.cross_entropy(logits.view(-1, V), ids.view(-1))
    loss.backward()
    for t in masker.unit_types:
        g = masker.gates[t].grad
        assert g is not None, t
        assert torch.isfinite(g).all(), t
        assert (g != 0).any(), t
    masker.accumulate_importance("taylor")
    assert masker.n_accumulated == 1
    for t in masker.unit_types:
        assert masker.gates[t].grad is None
        assert (masker.importance[t] > 0).any(), t


def test_prune_to_reaches_target_and_never_revives(ids):
    model = tiny_olmo().train()
    masker = StructuredMasker(model, unit_types=("head", "neuron"))
    loss = F.cross_entropy(model(input_ids=ids).logits.view(-1, V), ids.view(-1))
    loss.backward()
    masker.accumulate_importance("taylor")

    newly = masker.prune_to({"head": 0.25, "neuron": 0.1}, criterion="taylor")
    assert newly == {"head": 3, "neuron": round(0.1 * L * I)}
    first = {t: m.clone() for t, m in masker.masks.items()}
    # the pruned heads were the least important alive ones
    imp = masker.importance["head"].flatten()
    pruned_idx = (masker.masks["head"].flatten() == 0).nonzero().flatten()
    kept_idx = (masker.masks["head"].flatten() == 1).nonzero().flatten()
    assert imp[pruned_idx].max() <= imp[kept_idx].min()

    # lower target: nothing changes; higher target: strictly more pruned, old stay pruned
    assert masker.prune_to({"head": 0.1}, criterion="taylor") == {"head": 0}
    assert torch.equal(masker.masks["head"], first["head"])
    masker.prune_to({"head": 0.5}, criterion="taylor")
    assert (masker.masks["head"] == 0).sum() == L * H // 2
    assert ((first["head"] == 0) & (masker.masks["head"] == 0)).sum() == (first["head"] == 0).sum()


def test_min_alive_per_layer_guard(ids):
    model = tiny_olmo()
    masker = StructuredMasker(model, unit_types=("head",))
    masker.prune_to({"head": 1.0}, criterion="magnitude", min_alive_per_layer={"head": 1})
    assert ((masker.masks["head"] != 0).sum(dim=1) == 1).all()


def test_random_and_magnitude_criteria_need_no_gradients():
    model = tiny_olmo()
    masker = StructuredMasker(model, unit_types=("head", "neuron", "attn", "mlp"))
    g = torch.Generator().manual_seed(0)
    masker.prune_to({"head": 0.5, "attn": 1 / L}, criterion="random", generator=g)
    assert (masker.masks["head"] == 0).sum() == L * H // 2
    assert (masker.masks["attn"] == 0).sum() == 1
    masker.prune_to({"neuron": 0.5, "mlp": 1 / L}, criterion="magnitude")
    assert (masker.masks["neuron"] == 0).sum() == L * I // 2
    with pytest.raises(AssertionError):
        masker.prune_to({"head": 0.9}, criterion="taylor")   # nothing accumulated


def test_state_dict_roundtrip():
    model = tiny_olmo()
    masker = StructuredMasker(model, unit_types=("head", "neuron"))
    masker.prune_to({"head": 0.5}, criterion="magnitude")
    state = masker.state_dict()
    other = StructuredMasker(tiny_olmo(), unit_types=("head", "neuron"))
    other.load_state_dict(state)
    for t in masker.unit_types:
        assert torch.equal(other.masks[t], masker.masks[t])
    assert other.n_prune_events == 1


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

def test_cubic_schedule_endpoints_and_monotone():
    assert cubic_sparsity(0, 0.0, 0.5, 100, 500) == 0.0
    assert cubic_sparsity(100, 0.0, 0.5, 100, 500) == 0.0
    assert cubic_sparsity(500, 0.0, 0.5, 100, 500) == 0.5
    assert cubic_sparsity(10_000, 0.0, 0.5, 100, 500) == 0.5
    vals = [cubic_sparsity(t, 0.0, 0.5, 100, 500) for t in range(100, 501)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # cubic: half of the sparsity is reached well before half of the window
    assert cubic_sparsity(300, 0.0, 0.5, 100, 500) > 0.25 + 0.15
    assert cubic_sparsity(0, 0.2, 0.2, 0, 0) == 0.2       # one-shot degenerate


def test_prune_steps_include_endpoints():
    assert prune_steps(100, 500, 100) == [100, 200, 300, 400, 500]
    assert prune_steps(100, 450, 100) == [100, 200, 300, 400, 450]
    assert prune_steps(7, 7, 3) == [7]


# ---------------------------------------------------------------------------
# baking masks into weights (standalone pruned checkpoint)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("routed", [False, True])
def test_baked_weights_reproduce_masked_model(ids, routed):
    from scripts.prune_finetune import bake_masks_into_weights
    if routed:
        fw, pred = fourway_pair()
        model = fw.olmo
        run = lambda: fourway_logits(fw, pred, ids)  # noqa: E731
    else:
        model = tiny_olmo()
        run = lambda: dense_logits(model, ids)  # noqa: E731
    masker = StructuredMasker(model, unit_types=("head", "neuron", "attn", "mlp"))
    masker.masks["head"][0, 1] = 0.0
    masker.masks["neuron"][1, :10] = 0.0
    masker.masks["attn"][2, 0] = 0.0
    masker.masks["mlp"][0, 0] = 0.0
    masked = run()
    bake_masks_into_weights(masker)
    masker.remove_hooks()
    assert torch.allclose(run(), masked, atol=1e-5)
