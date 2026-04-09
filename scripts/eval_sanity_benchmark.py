"""Eval sanity benchmark — does eval code compute the same NLL as training?

Tests:
1. FourWay checkpoint: run eval code on held-out eval cache.
   → compare to training log's reported eval NLL.
   → also compare TRAIN mode vs EVAL mode predictor on same data.

2. Dense baseline checkpoint: run on the SAME eval cache.
   → should get ~3.85. If not → eval data/pipeline is broken.

3. FourWay checkpoint: run WITHOUT routing (base model only) on eval cache.
   → should match dense_baseline result (7.08 for trained-with-routing base).
   → if much different → something wrong with base model loading.

Usage:
    python scripts/eval_sanity_benchmark.py \\
        --fourway_ckpt checkpoints/fourway_corrected_joint_causal/checkpoint_step5000.pt \\
        --fourway_config configs/fourway_corrected_joint_causal.yaml \\
        --dense_ckpt checkpoints/pretrain_300m_baseline_5k/checkpoint_step5000.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.olmo_graph import FourWayDAGFormer
from src.model.predictor import FourWayPredictor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_base(config, device, dtype):
    mc = Olmo2Config(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        tie_word_embeddings=config.get("tie_word_embeddings", True),
        max_position_embeddings=config.get("max_position_embeddings", 4096),
    )
    return Olmo2ForCausalLM(mc).to(device=device, dtype=dtype)


def load_base_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_path" in ckpt:
        model_path = ckpt["model_state_path"]
        if not Path(model_path).is_absolute():
            cand = Path(ckpt_path).parent / Path(model_path).name
            if cand.exists():
                model_path = str(cand)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "base_model_state_dict" in ckpt:
        state = ckpt["base_model_state_dict"]
    else:
        state = ckpt

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    stripped = {}
    for k, v in state.items():
        nk = k
        for p in ("module.base_model.", "module.", "base_model.olmo.", "olmo."):
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        stripped[nk] = v

    m, u = model.load_state_dict(stripped, strict=False)
    print(f"  Load: missing={len(m)}, unexpected={len(u)}")
    return ckpt


def nll(logits, labels, V):
    return F.cross_entropy(logits.float().view(-1, V), labels.view(-1)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fourway_ckpt", type=str, required=True)
    parser.add_argument("--fourway_config", type=str, required=True)
    parser.add_argument("--dense_ckpt", type=str, default=None)
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.fourway_config)
    device = torch.device("cuda")
    V = config["vocab_size"]

    # Load eval cache
    eval_cache_path = Path(config.get("save_dir", "checkpoints/")) / "eval_cache.pt"
    assert eval_cache_path.exists(), f"No eval cache at {eval_cache_path}"
    eval_cache = torch.load(str(eval_cache_path), map_location="cpu", weights_only=False)
    eval_batches = eval_cache if isinstance(eval_cache, list) else eval_cache.get("batches", list(eval_cache))
    eval_batches = eval_batches[:args.n_batches]
    print(f"Eval cache: {len(eval_batches)} batches from {eval_cache_path}")

    # ================================================================
    # TEST 1: FourWay on eval cache — train() vs eval() mode
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: FourWay checkpoint on eval cache (train vs eval mode)")
    print("=" * 60)

    base = build_base(config, device, torch.bfloat16)
    routing_mode = config.get("routing_mode", "fourway_corrected")
    use_corr = (routing_mode == "fourway_corrected")
    fw = FourWayDAGFormer(
        model=base, num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        use_local_correction=use_corr,
        correction_hidden=config.get("correction_hidden", 128),
        use_v_norm=config.get("use_v_norm", False),
    ).to(device)
    pred = FourWayPredictor(
        vocab_size=V, encoder_dim=config["predictor_encoder_dim"],
        encoder_layers=config["predictor_encoder_layers"],
        encoder_heads=config["predictor_encoder_heads"],
        max_seq_len=config["predictor_max_seq_len"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        hidden_dim=config["fourway_hidden"],
        causal=config.get("predictor_causal", True),
        dropout=config.get("predictor_dropout", 0.0),
    ).to(device)

    print(f"Loading: {args.fourway_ckpt}")
    ckpt = load_base_state(fw.olmo, args.fourway_ckpt)
    if "predictor_state_dict" in ckpt:
        pred.load_state_dict(ckpt["predictor_state_dict"], strict=False)
    if "routing_state_dict" in ckpt:
        fw.load_state_dict(ckpt["routing_state_dict"], strict=False)

    nlls_eval_mode = []
    nlls_train_mode = []
    nlls_dense = []

    with torch.no_grad():
        for i, batch in enumerate(eval_batches):
            ids = batch["olmo_ids"].to(device)
            labels = batch["olmo_labels"].to(device)

            # Eval mode (standard eval path)
            fw.eval(); pred.eval()
            rw = pred(ids)
            rw_cast = {k: [a.to(dtype=torch.bfloat16) for a in v] for k, v in rw.items()}
            logits_e = fw(ids, rw_cast)
            nlls_eval_mode.append(nll(logits_e, labels, V))

            # Train mode (same data, same forward, just mode flag)
            fw.train(); pred.train()
            rw2 = pred(ids)
            rw2_cast = {k: [a.to(dtype=torch.bfloat16) for a in v] for k, v in rw2.items()}
            logits_t = fw(ids, rw2_cast)
            nlls_train_mode.append(nll(logits_t, labels, V))

            # Dense (base model only, no routing)
            fw.olmo.eval()
            logits_d = fw.olmo(input_ids=ids, use_cache=False).logits
            nlls_dense.append(nll(logits_d, labels, V))

    mean_e = sum(nlls_eval_mode) / len(nlls_eval_mode)
    mean_t = sum(nlls_train_mode) / len(nlls_train_mode)
    mean_d = sum(nlls_dense) / len(nlls_dense)

    print(f"\n  FW eval-mode NLL:   {mean_e:.4f}  (should match training log ~5.49)")
    print(f"  FW train-mode NLL:  {mean_t:.4f}  (should ≈ eval-mode)")
    print(f"  Dense (no routing): {mean_d:.4f}  (base model as dense)")
    print(f"  train-eval diff:    {mean_t - mean_e:+.4f}")

    # ================================================================
    # TEST 2: Dense baseline checkpoint on SAME eval cache
    # ================================================================
    if args.dense_ckpt:
        print("\n" + "=" * 60)
        print("TEST 2: Dense baseline checkpoint on SAME eval cache")
        print("=" * 60)

        dense_model = build_base(config, device, torch.bfloat16)
        print(f"Loading: {args.dense_ckpt}")
        load_base_state(dense_model, args.dense_ckpt)
        dense_model.eval()

        nlls_dense_baseline = []
        with torch.no_grad():
            for batch in eval_batches:
                ids = batch["olmo_ids"].to(device)
                labels = batch["olmo_labels"].to(device)
                logits = dense_model(input_ids=ids, use_cache=False).logits
                nlls_dense_baseline.append(nll(logits, labels, V))

        mean_db = sum(nlls_dense_baseline) / len(nlls_dense_baseline)
        print(f"\n  Dense baseline eval NLL:  {mean_db:.4f}")
        print(f"  Expected:                 ~3.85")
        print(f"  Deviation:                {mean_db - 3.85:+.4f}")

        if abs(mean_db - 3.85) > 0.5:
            print("  ⚠ LARGE DEVIATION — eval cache/pipeline might be broken!")
        else:
            print("  ✓ Within tolerance — eval infrastructure is correct")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if args.dense_ckpt:
        print(f"  Dense baseline on eval cache:     {mean_db:.4f} (expect ~3.85)")
    print(f"  FW eval-mode on eval cache:       {mean_e:.4f} (expect ~5.49)")
    print(f"  FW train-mode on eval cache:      {mean_t:.4f} (expect ≈ eval-mode)")
    print(f"  FW base-only (no routing):        {mean_d:.4f} (expect ~7.08)")
    print()
    if args.dense_ckpt and abs(mean_db - 3.85) < 0.5:
        print("  ✓ Dense baseline matches → eval pipeline is correct")
    if abs(mean_t - mean_e) < 0.01:
        print("  ✓ Train/eval mode identical → no hidden mode-dependent behavior")
    gap = mean_e - (mean_db if args.dense_ckpt else 3.85)
    print(f"  Gap (FW eval - dense baseline):   {gap:+.4f}")
    if gap > 1.0:
        print("  → Genuine overfit: FW model generalizes worse than dense baseline")
    else:
        print("  → Gap is small — routing is not hurting much")


if __name__ == "__main__":
    main()
