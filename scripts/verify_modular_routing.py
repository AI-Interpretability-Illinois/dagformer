"""Verify that fourway_modular really routes the formerly hard-wired edges.

Run on a GPU against a real-size model (fresh or a checkpoint). Checks:

  1. identity     all-ones routing == vanilla OLMo-2 forward (bf16 tolerance)
  2. attn->MLP    zeroing the MLP-input weight on a_l changes the output, and
                  equals a reference where mlp_in = R_l (edge really removable)
  3. read-out     zeroing the final read-out weight on m_{L-1} equals dropping
                  the last MLP block's contribution
  4. column       zeroing every reader's weight on a_l == masking attention
                  block l with StructuredMasker
  5. gradients    after one backward, every intra-layer edge (m stream) and
                  read-out edge receives a non-zero gradient at the predictor
  6. movement     if a checkpoint is given: the predictor's m / o biases and
                  weights have moved away from identity during training,
                  i.e. the optimiser actually touched those edges.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_lm_harness import build_base_model, load_fourway, _replace_olmo_rmsnorm  # noqa: E402
from src.model.modular_routing import (  # noqa: E402
    build_modular_pair, identity_modular_routing, modular_n_sources, source_column_mass,
)
from src.pruning import StructuredMasker  # noqa: E402


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default="", help="optional checkpoint from pretrain_dagformer.py")
    p.add_argument("--eval-cache", default="/work/hdd/bfqt/xiaocong/dagformer_pruning/data/mathinstruct/eval_cache.pt")
    p.add_argument("--seq-len", type=int, default=256)
    args = p.parse_args()
    device = torch.device("cuda")
    cfg = yaml.safe_load(open(args.config))
    assert str(cfg.get("routing_mode", "")).startswith("fourway_modular"), cfg.get("routing_mode")
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]

    if args.ckpt:
        fw, pred = load_fourway(args.ckpt, cfg, device)
        for p_ in list(fw.parameters()) + list(pred.parameters()):   # the loader freezes everything
            p_.requires_grad_(True)
    else:
        base = build_base_model(cfg, device)
        if cfg.get("replace_rmsnorm", False):
            _replace_olmo_rmsnorm(base)
            base = base.to(device=device, dtype=torch.bfloat16)
        fw, pred = build_modular_pair(cfg, base, device)
    fw.eval()
    pred.eval()
    batches = torch.load(args.eval_cache, map_location="cpu", weights_only=False)
    ids = batches[0]["olmo_ids"][:4, :args.seq_len].to(device)
    B, T = ids.shape
    ok = True

    with torch.no_grad():
        # 1. identity == vanilla
        rw = identity_modular_routing(B, T, L, H, device, dtype=torch.float32)
        vanilla = fw.olmo(input_ids=ids).logits.float()
        routed = fw(ids, rw).float()
        diff = (vanilla - routed).abs().max().item()
        ok &= report("identity routing == vanilla OLMo-2", diff < 0.05, f"max|diff|={diff:.4f} (bf16)")

        # 2. attention -> MLP edge of layer l
        l = L // 2
        rw2 = identity_modular_routing(B, T, L, H, device)
        rw2["m"][l][..., modular_n_sources(l)] = 0.0          # last m-source of layer l is a_l
        out2 = fw(ids, rw2).float()
        ok &= report(f"zeroing attn->MLP edge in layer {l} changes the output",
                     (out2 - routed).abs().max().item() > 1e-3)
        # reference: same thing done by hand through hooks -- subtract a_l from the MLP input
        captured = {}
        layer = fw.olmo.model.layers[l]
        h1 = layer.post_attention_layernorm.register_forward_hook(lambda m, i, o: captured.__setitem__("a", o))
        h2 = layer.mlp.register_forward_pre_hook(lambda m, i: (i[0] - captured["a"],))
        ref2 = fw(ids, identity_modular_routing(B, T, L, H, device)).float()
        h1.remove(); h2.remove()
        d = (out2 - ref2).abs().max().item()
        ok &= report("  ... and equals mlp_in = R_l computed by hand", d < 0.05, f"max|diff|={d:.4f}")

        # 3. read-out edge on the last MLP block
        rw3 = identity_modular_routing(B, T, L, H, device)
        rw3["o"][..., -1] = 0.0
        out3 = fw(ids, rw3).float()
        masker = StructuredMasker(fw.olmo, unit_types=("attn", "mlp"))
        masker.masks["mlp"][L - 1, 0] = 0.0
        ref3 = fw(ids, identity_modular_routing(B, T, L, H, device)).float()
        masker.masks["mlp"][L - 1, 0] = 1.0
        d = (out3 - ref3).abs().max().item()
        ok &= report("read-out column of m_{L-1} zeroed == last MLP block masked", d < 0.05, f"max|diff|={d:.4f}")

        # 4. whole column of a_l == attention block l masked
        src = 1 + 2 * l
        rw4 = identity_modular_routing(B, T, L, H, device)
        for j in range(l + 1, L):
            for s in ("q", "k", "v"):
                rw4[s][j - 1][..., src] = 0.0
            rw4["r"][j - 1][..., src] = 0.0
            rw4["m"][j][..., src] = 0.0
        rw4["m"][l][..., modular_n_sources(l)] = 0.0
        rw4["o"][..., src] = 0.0
        out4 = fw(ids, rw4).float()
        masker.masks["attn"][l, 0] = 0.0
        ref4 = fw(ids, identity_modular_routing(B, T, L, H, device)).float()
        masker.masks["attn"][l, 0] = 1.0
        masker.remove_hooks()
        d = (out4 - ref4).abs().max().item()
        mass = source_column_mass(rw4, L)
        ok &= report(f"zero column of a_{l} == attention block {l} masked", d < 0.05,
                     f"max|diff|={d:.4f}, column mass={mass[src].item():.3f}")

    # 5. gradients reach every intra-layer / read-out edge
    fw.train(); pred.train()
    for p_ in list(fw.parameters()) + list(pred.parameters()):
        p_.grad = None
    logits = fw(ids, pred(ids))
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids.reshape(-1))
    loss.backward()
    m_grads_ok, o_ok = True, (pred.out_bias.grad is not None and (pred.out_bias.grad != 0).all())
    for li in range(L):
        g = pred.layer_biases[li].grad
        n = modular_n_sources(li)
        m_slice = g[-(n + 1):] if li >= 1 else g            # m stream is the tail of each layer's head
        m_grads_ok &= g is not None and (m_slice != 0).all().item()
    ok &= report("non-zero gradient on every attn->MLP / MLP-input edge (m stream)", bool(m_grads_ok))
    ok &= report("non-zero gradient on every read-out edge (o stream)", bool(o_ok))
    ok &= report("gradient reaches the backbone", fw.olmo.model.layers[0].mlp.down_proj.weight.grad is not None)

    # 6. movement away from identity (checkpoint only)
    if args.ckpt:
        with torch.no_grad():
            m_dev = max((pred.layer_biases[li][-(modular_n_sources(li) + 1):] - 1).abs().max().item()
                        for li in range(1, L))
            m0_dev = (pred.layer_biases[0] - 1).abs().max().item()
            o_dev = (pred.out_bias - 1).abs().max().item()
            w_norm = max(h.weight.norm().item() for h in pred.layer_heads) + pred.out_head.weight.norm().item()
            rw = pred(ids)
            mass = source_column_mass(rw, L)
        ok &= report("trained biases moved off identity on attn->MLP / read-out edges", m_dev > 0 and o_dev > 0,
                     f"max|m-1|={m_dev:.4f} (layer0 {m0_dev:.4f}), max|o-1|={o_dev:.4f}, head W norm={w_norm:.4f}")
        print("  column mass per source:", [round(x, 3) for x in mass.tolist()])
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
