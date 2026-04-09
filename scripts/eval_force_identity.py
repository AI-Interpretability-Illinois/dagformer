"""Load a trained FourWay checkpoint and evaluate NLL in three modes:
1. Normal: use the trained predictor's output
2. Identity: force routing weights to identity [0,...,0,1]
3. Dense (sanity): run the base OLMo directly (no FourWay wrapper)

This implements Codex's suggested Q5 experiment: disambiguate whether
the ~2.0 train/eval gap is caused by (A) predictor overfitting producing
bad eval-time routings, or (B) the base model being co-adapted/corrupted
by routed training (R-stream amplification hypothesis).

Expected results under each hypothesis:
- (A) predictor overfit: identity-eval NLL ≈ dense baseline (~3.85)
- (B) base corrupted:    identity-eval NLL >> dense baseline

Usage:
    python scripts/eval_force_identity.py \\
        --checkpoint checkpoints/fourway_corrected_joint_causal/checkpoint_step5000.pt \\
        --config configs/fourway_corrected_joint_causal.yaml
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


def build_base(config: dict, device, dtype) -> Olmo2ForCausalLM:
    model_config = Olmo2Config(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        tie_word_embeddings=config.get("tie_word_embeddings", True),
        max_position_embeddings=config.get("max_position_embeddings", 4096),
    )
    model = Olmo2ForCausalLM(model_config).to(device=device, dtype=dtype)
    return model


def build_fourway_and_predictor(config, device):
    base = build_base(config, device, torch.bfloat16)
    # Match training's construction logic (pretrain_dagformer.py:674-683):
    # use_local_correction is tied to routing_mode == "fourway_corrected"
    routing_mode = config.get("routing_mode", "fourway_corrected")
    use_correction = (routing_mode == "fourway_corrected")
    fw = FourWayDAGFormer(
        model=base,
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        use_local_correction=use_correction,
        correction_hidden=config.get("correction_hidden", 128),
        use_triton_kernel=False,
        use_v_norm=config.get("use_v_norm", False),
    ).to(device)
    pred = FourWayPredictor(
        vocab_size=config["vocab_size"],
        encoder_dim=config["predictor_encoder_dim"],
        encoder_layers=config["predictor_encoder_layers"],
        encoder_heads=config["predictor_encoder_heads"],
        max_seq_len=config["predictor_max_seq_len"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        hidden_dim=config["fourway_hidden"],
        causal=config.get("predictor_causal", True),
        dropout=config.get("predictor_dropout", 0.0),
    ).to(device)
    return base, fw, pred


def load_checkpoint_into(fw: FourWayDAGFormer, pred: FourWayPredictor, ckpt_path: str) -> dict:
    """Load trained predictor + base model weights from a checkpoint.

    Handles the key-prefix bug noted in Codex's review: the saved state dict may have
    bare `model.layers.0...` keys but FourWayDAGFormer expects `olmo.model.layers.0...`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {list(ckpt.keys())[:20]}")

    # Load predictor state
    if "predictor_state_dict" in ckpt:
        missing, unexpected = pred.load_state_dict(ckpt["predictor_state_dict"], strict=False)
        print(f"Predictor load: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("WARNING: no predictor_state_dict in checkpoint")

    # Load base model state — need to unwrap potential prefixes
    state_keys = [
        "base_model_state_dict",
        "model_state_dict",
        "state_dict",
    ]
    base_state = None
    for k in state_keys:
        if k in ckpt:
            base_state = ckpt[k]
            print(f"Using base state from key: '{k}'")
            break

    # NEW: handle separate _model.pt file referenced by 'model_state_path'
    if base_state is None and "model_state_path" in ckpt:
        model_path = ckpt["model_state_path"]
        if not Path(model_path).is_absolute():
            # Path may be relative to repo root or checkpoint dir
            candidates = [
                Path(model_path),
                Path(ckpt_path).parent / Path(model_path).name,
            ]
            for c in candidates:
                if c.exists():
                    model_path = str(c)
                    break
        print(f"Loading base model state from: {model_path}")
        base_state = torch.load(model_path, map_location="cpu", weights_only=False)
        # If nested in a dict
        if isinstance(base_state, dict) and "state_dict" in base_state:
            base_state = base_state["state_dict"]

    if base_state is None:
        print("WARNING: could not find base model state in checkpoint")
        return ckpt

    # Strip common DDP/wrapper prefixes
    stripped = {}
    for k, v in base_state.items():
        new_k = k
        for prefix in ("module.base_model.", "module.", "base_model.olmo.", "olmo."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        stripped[new_k] = v

    # Load into fw.olmo (which IS an Olmo2ForCausalLM)
    missing, unexpected = fw.olmo.load_state_dict(stripped, strict=False)
    print(f"Base OLMo load: missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"  first 5 missing: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"  first 5 unexpected: {unexpected[:5]}")

    # Also try loading routing_state_dict if it exists (correction MLPs or similar)
    if "routing_state_dict" in ckpt:
        missing, unexpected = fw.load_state_dict(ckpt["routing_state_dict"], strict=False)
        print(f"Routing state load: missing={len(missing)}, unexpected={len(unexpected)}")

    return ckpt


def build_identity_routing(
    num_layers: int, num_heads: int, batch: int, seq_len: int, device, dtype
) -> dict[str, list[torch.Tensor]]:
    """Produce routing_weights dict that is exact identity for all streams/layers."""
    rw = {"q": [], "k": [], "v": [], "r": []}
    H = num_heads
    for l in range(1, num_layers):
        n_src = l + 1
        # q/k/v: [B, T, H, n_src], identity = last source = 1
        for key in ("q", "k", "v"):
            α = torch.zeros(batch, seq_len, H, n_src, device=device, dtype=dtype)
            α[..., -1] = 1.0
            rw[key].append(α)
        # r: [B, T, n_src]
        α_r = torch.zeros(batch, seq_len, n_src, device=device, dtype=dtype)
        α_r[..., -1] = 1.0
        rw["r"].append(α_r)
    return rw


def compute_nll(logits: torch.Tensor, labels: torch.Tensor, pad_id: int = -100) -> float:
    """Standard cross-entropy NLL. Labels are already shifted by the dataloader."""
    V = logits.shape[-1]
    loss = F.cross_entropy(
        logits.float().view(-1, V), labels.view(-1), ignore_index=pad_id, reduction="mean"
    )
    return loss.item()


def load_eval_batches(eval_cache_path: str):
    """Load the saved eval cache produced by pretrain_dagformer.py."""
    cache = torch.load(eval_cache_path, map_location="cpu", weights_only=False)
    return cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--eval_cache",
        type=str,
        default=None,
        help="path to eval_cache.pt (if None, inferred from config.save_dir)",
    )
    parser.add_argument("--max_batches", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda")

    # Build models
    print("Building models...")
    base, fw, pred = build_fourway_and_predictor(config, device)
    fw.eval()
    pred.eval()

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint_into(fw, pred, args.checkpoint)
    print(f"Checkpoint step: {ckpt.get('step', '?')}")

    # Load eval batches
    if args.eval_cache is None:
        save_dir = config.get("save_dir", "checkpoints/")
        args.eval_cache = str(Path(save_dir) / "eval_cache.pt")
    print(f"\nLoading eval cache: {args.eval_cache}")
    eval_cache = load_eval_batches(args.eval_cache)
    if isinstance(eval_cache, list):
        eval_batches = eval_cache
    elif isinstance(eval_cache, dict) and "batches" in eval_cache:
        eval_batches = eval_cache["batches"]
    else:
        print(f"Eval cache type: {type(eval_cache)}")
        eval_batches = list(eval_cache)
    eval_batches = eval_batches[: args.max_batches]
    print(f"Using {len(eval_batches)} eval batches")

    # Peek at first batch to understand format
    b0 = eval_batches[0]
    print(f"First batch type: {type(b0)}, keys: {list(b0.keys()) if isinstance(b0, dict) else 'N/A'}")

    # Evaluation loop — 5 modes to disentangle predictor vs correction contributions
    #
    # When use_local_correction=True, the correction MLPs add deltas to α regardless
    # of what we pass in. So "fw_id + corrections_on" is NOT the same as pure dense.
    # We temporarily flip the flag for modes that want corrections bypassed.
    results = {
        "dense_baseline": [],                       # direct OLMo, no FourWay at all
        "fw_pred":  [],                             # FW + predictor α + corrections (true training eval)
        "fw_id":    [],                             # FW + identity α + corrections (kill predictor, keep corrections)
        "fw_pred_noc": [],                          # FW + predictor α, corrections disabled
        "fw_id_noc":   [],                          # FW + identity α, corrections disabled (= dense if math is right)
    }

    original_use_correction = fw.use_local_correction

    with torch.no_grad():
        for i, batch in enumerate(eval_batches):
            if isinstance(batch, dict):
                input_ids = batch["olmo_ids"].to(device)
                labels = batch["olmo_labels"].to(device)
            else:
                raise ValueError(f"Unknown batch format: {type(batch)}")

            batch_size, seq_len = input_ids.shape
            model_dtype = next(fw.olmo.parameters()).dtype

            # Precompute predictor output and identity routing once per batch
            rw_pred = pred(input_ids)
            rw_pred_cast = {
                k: [a.to(dtype=model_dtype) for a in v] for k, v in rw_pred.items()
            }
            rw_id = build_identity_routing(
                num_layers=config["num_hidden_layers"],
                num_heads=config["num_attention_heads"],
                batch=batch_size,
                seq_len=seq_len,
                device=device,
                dtype=model_dtype,
            )

            # (1) Dense baseline: direct forward through base OLMo (no FourWay forward)
            dense_logits = fw.olmo(input_ids=input_ids, use_cache=False).logits
            results["dense_baseline"].append(compute_nll(dense_logits, labels))

            # (2) FW + predictor + corrections (if any) — true training eval
            fw.use_local_correction = original_use_correction
            fw_pred_logits = fw(input_ids, rw_pred_cast)
            results["fw_pred"].append(compute_nll(fw_pred_logits, labels))

            # (3) FW + identity + corrections
            fw_id_logits = fw(input_ids, rw_id)
            results["fw_id"].append(compute_nll(fw_id_logits, labels))

            # (4) FW + predictor, corrections disabled
            fw.use_local_correction = False
            fw_pred_noc_logits = fw(input_ids, rw_pred_cast)
            results["fw_pred_noc"].append(compute_nll(fw_pred_noc_logits, labels))

            # (5) FW + identity, corrections disabled (should bit-exact equal dense)
            fw_id_noc_logits = fw(input_ids, rw_id)
            results["fw_id_noc"].append(compute_nll(fw_id_noc_logits, labels))

            # restore flag
            fw.use_local_correction = original_use_correction

            if (i + 1) % 10 == 0:
                print(
                    f"  batch {i+1}/{len(eval_batches)}: "
                    f"dense={results['dense_baseline'][-1]:.3f} "
                    f"fw_pred={results['fw_pred'][-1]:.3f} "
                    f"fw_id={results['fw_id'][-1]:.3f} "
                    f"fw_pred_noc={results['fw_pred_noc'][-1]:.3f} "
                    f"fw_id_noc={results['fw_id_noc'][-1]:.3f}"
                )

    # Report
    print("\n" + "=" * 60)
    print("EVAL NLL (averaged over batches)")
    print("=" * 60)
    for key, vals in results.items():
        mean = sum(vals) / len(vals)
        print(f"  {key:30s} = {mean:.4f}")

    dense_mean = sum(results["dense_baseline"]) / len(results["dense_baseline"])
    fw_pred_mean = sum(results["fw_pred"]) / len(results["fw_pred"])
    fw_id_mean = sum(results["fw_id"]) / len(results["fw_id"])
    fw_pred_noc_mean = sum(results["fw_pred_noc"]) / len(results["fw_pred_noc"])
    fw_id_noc_mean = sum(results["fw_id_noc"]) / len(results["fw_id_noc"])

    print("\n" + "=" * 60)
    print("DIAGNOSIS — 5 modes + deltas")
    print("=" * 60)
    print(f"  dense_baseline      = {dense_mean:.4f}  (direct OLMo forward, no FourWay)")
    print(f"  fw_pred (full)      = {fw_pred_mean:.4f}  (predictor α + corrections)")
    print(f"  fw_id   (kill pred) = {fw_id_mean:.4f}  (identity α + corrections)")
    print(f"  fw_pred_noc         = {fw_pred_noc_mean:.4f}  (predictor α, corrections OFF)")
    print(f"  fw_id_noc           = {fw_id_noc_mean:.4f}  (identity α, corrections OFF — should ≈ dense)")
    print()
    print(f"  fw_pred - dense     = {fw_pred_mean - dense_mean:+.4f}  (total routing effect)")
    print(f"  fw_pred - fw_id     = {fw_pred_mean - fw_id_mean:+.4f}  (predictor contribution)")
    print(f"  fw_id   - fw_id_noc = {fw_id_mean - fw_id_noc_mean:+.4f}  (correction contribution alone)")
    print(f"  fw_id_noc - dense   = {fw_id_noc_mean - dense_mean:+.4f}  (should be ≈ 0)")

    # Sanity check
    if abs(fw_id_noc_mean - dense_mean) > 0.01:
        print(f"\n  WARNING: fw_id_noc should bit-exact equal dense, got diff {fw_id_noc_mean - dense_mean:.4e}")
        print("  Indicates a bug in FourWay forward at identity init.")

    # Also report α statistics from the first batch
    print("\n" + "=" * 60)
    print("α STATISTICS (from first eval batch, predictor output)")
    print("=" * 60)
    with torch.no_grad():
        batch = eval_batches[0]
        input_ids = batch["olmo_ids"].to(device)
        rw = pred(input_ids)
        for stream in ("q", "k", "v", "r"):
            norms, maxes, mins = [], [], []
            for l_idx, α in enumerate(rw[stream]):
                α_f = α.float()
                norms.append(α_f.norm(dim=-1).mean().item())
                maxes.append(α_f.max().item())
                mins.append(α_f.min().item())
            print(f"  stream {stream}: "
                  f"mean_norm={sum(norms)/len(norms):.3f}, "
                  f"max={max(maxes):.3f}, "
                  f"min={min(mins):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
