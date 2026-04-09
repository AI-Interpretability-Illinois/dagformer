"""Diagnostic: check whether FourWayPredictor produces different output
in train() vs eval() mode on the SAME input batch.

With the current config (no predictor_dropout), the predictor has NO
stochastic layers. train() and eval() should produce bit-identical results.
If they don't, there's a hidden mode-dependent code path causing a silent
train/eval asymmetry.

Also compares the full FourWayDAGFormer forward (predictor + wrapper)
in both modes.

Usage:
    python scripts/diff_train_eval_predictor.py \\
        --checkpoint checkpoints/fourway_corrected_joint_causal/checkpoint_step5000.pt \\
        --config configs/fourway_corrected_joint_causal.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.olmo_graph import FourWayDAGFormer
from src.model.predictor import FourWayPredictor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build(config, device):
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
    base = Olmo2ForCausalLM(mc).to(device=device, dtype=torch.bfloat16)
    routing_mode = config.get("routing_mode", "fourway_corrected")
    use_corr = (routing_mode == "fourway_corrected")
    fw = FourWayDAGFormer(
        model=base,
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        use_local_correction=use_corr,
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


def load_ckpt(fw, pred, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "predictor_state_dict" in ckpt:
        pred.load_state_dict(ckpt["predictor_state_dict"], strict=False)
    # Base model state from separate file
    model_path = ckpt.get("model_state_path")
    if model_path:
        if not Path(model_path).is_absolute():
            cand = Path(path).parent / Path(model_path).name
            if cand.exists():
                model_path = str(cand)
        base_state = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(base_state, dict) and "state_dict" in base_state:
            base_state = base_state["state_dict"]
        stripped = {}
        for k, v in base_state.items():
            nk = k
            for p in ("module.base_model.", "module.", "base_model.olmo.", "olmo."):
                if nk.startswith(p):
                    nk = nk[len(p):]
                    break
            stripped[nk] = v
        fw.olmo.load_state_dict(stripped, strict=False)
    # Routing state (correction MLPs, v_norms)
    if "routing_state_dict" in ckpt:
        fw.load_state_dict(ckpt["routing_state_dict"], strict=False)


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32 = a.float()
    b32 = b.float()
    d = (a32 - b32).abs()
    return {
        "max": d.max().item(),
        "mean": d.mean().item(),
        "any_nonzero": (d > 0).any().item(),
        "num_diff": (d > 0).sum().item(),
        "total": d.numel(),
        "a_norm": a32.norm().item(),
        "b_norm": b32.norm().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda")

    print("Building models...")
    base, fw, pred = build(config, device)
    print(f"Loading checkpoint: {args.checkpoint}")
    load_ckpt(fw, pred, args.checkpoint)

    # Fixed random input
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, config["vocab_size"], (args.batch, args.seq_len), device=device
    )
    print(f"Input: {tuple(input_ids.shape)}")

    # ============================================================
    # TEST 1: Predictor alone — train() vs eval()
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: FourWayPredictor(input_ids) in train() vs eval() mode")
    print("=" * 60)

    pred.train()
    with torch.no_grad():
        rw_train = pred(input_ids)

    pred.eval()
    with torch.no_grad():
        rw_eval = pred(input_ids)

    any_bad = False
    for stream in ("q", "k", "v", "r"):
        for i, (t, e) in enumerate(zip(rw_train[stream], rw_eval[stream])):
            s = diff_stats(t, e)
            if s["any_nonzero"]:
                any_bad = True
                print(f"  DIFF stream={stream} layer_idx={i}: "
                      f"max={s['max']:.4e} mean={s['mean']:.4e} "
                      f"num_diff={s['num_diff']}/{s['total']}")

    if not any_bad:
        print("  PASS — predictor output is bit-identical in train() and eval() mode")
        print("         (expected, since no dropout/BN in current config)")
    else:
        print("  FAIL — predictor produces DIFFERENT output in train vs eval!")
        print("         This is a hidden mode-dependent code path.")

    # ============================================================
    # TEST 2: Full FourWay forward — train() vs eval() with same α
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: FourWayDAGFormer forward in train() vs eval() (same α)")
    print("=" * 60)

    pred.eval()  # use fixed predictor output
    with torch.no_grad():
        rw = pred(input_ids)
        rw_cast = {k: [a.to(dtype=torch.bfloat16) for a in v] for k, v in rw.items()}

        fw.train()
        logits_train = fw(input_ids, rw_cast)

        fw.eval()
        logits_eval = fw(input_ids, rw_cast)

    s = diff_stats(logits_train, logits_eval)
    print(f"  logits diff: max={s['max']:.4e} mean={s['mean']:.4e} "
          f"any_nonzero={s['any_nonzero']}")
    print(f"  num_diff={s['num_diff']}/{s['total']}")
    if not s["any_nonzero"]:
        print("  PASS — FourWayDAGFormer produces bit-identical output")
    else:
        print(f"  FAIL — FourWayDAGFormer diverges in train vs eval by "
              f"max {s['max']:.4e}")

    # ============================================================
    # TEST 3: Full pipeline — predictor train/eval × wrapper train/eval
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Full pipeline — 4 combinations of train/eval modes")
    print("=" * 60)

    def run_pipeline(pred_mode: str, fw_mode: str) -> torch.Tensor:
        (pred.train() if pred_mode == "train" else pred.eval())
        (fw.train() if fw_mode == "train" else fw.eval())
        with torch.no_grad():
            rw = pred(input_ids)
            rw_cast = {k: [a.to(dtype=torch.bfloat16) for a in v] for k, v in rw.items()}
            logits = fw(input_ids, rw_cast)
        return logits

    combos = [
        ("train", "train"),  # "training mode"
        ("eval",  "eval"),   # "eval mode"
        ("train", "eval"),
        ("eval",  "train"),
    ]
    results = {}
    for pm, fm in combos:
        logits = run_pipeline(pm, fm)
        results[(pm, fm)] = logits

    reference = results[("eval", "eval")]
    print("  Reference: (pred=eval, fw=eval)")
    print(f"  reference norm = {reference.float().norm().item():.4f}")
    for combo, logits in results.items():
        s = diff_stats(logits, reference)
        label = f"(pred={combo[0]}, fw={combo[1]})"
        print(f"  {label:30s} max_diff={s['max']:.4e} mean_diff={s['mean']:.4e}")

    # Compare pipeline NLL
    print("\n  NLL comparison (random labels)")
    import torch.nn.functional as F
    torch.manual_seed(args.seed + 1)
    labels = torch.randint(0, config["vocab_size"], input_ids.shape, device=device)
    for combo, logits in results.items():
        nll = F.cross_entropy(
            logits.float().view(-1, config["vocab_size"]), labels.view(-1)
        ).item()
        label = f"(pred={combo[0]}, fw={combo[1]})"
        print(f"  {label:30s} NLL = {nll:.6f}")

    # ============================================================
    # TEST 4: REAL eval data — pred.train() vs pred.eval()
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: REAL eval data — NLL with pred.train() vs pred.eval()")
    print("=" * 60)

    # Load eval cache
    save_dir = config.get("save_dir", "checkpoints/")
    eval_cache_path = Path(save_dir) / "eval_cache.pt"
    if not eval_cache_path.exists():
        print(f"  SKIP — no eval cache at {eval_cache_path}")
        return 0

    eval_cache = torch.load(str(eval_cache_path), map_location="cpu", weights_only=False)
    if isinstance(eval_cache, dict) and "batches" in eval_cache:
        eval_batches = eval_cache["batches"]
    else:
        eval_batches = list(eval_cache) if not isinstance(eval_cache, list) else eval_cache
    eval_batches = eval_batches[:20]
    print(f"  Using {len(eval_batches)} real eval batches from {eval_cache_path}")

    nll_train_list = []
    nll_eval_list = []

    fw.eval()  # fw doesn't matter (Test 2 showed identical)

    with torch.no_grad():
        for batch in eval_batches:
            eids = batch["olmo_ids"].to(device)
            elabels = batch["olmo_labels"].to(device)

            # pred.train()
            pred.train()
            rw_t = pred(eids)
            rw_t_cast = {k: [a.to(dtype=torch.bfloat16) for a in v] for k, v in rw_t.items()}
            logits_t = fw(eids, rw_t_cast)
            nll_t = F.cross_entropy(
                logits_t.float().view(-1, config["vocab_size"]), elabels.view(-1)
            ).item()
            nll_train_list.append(nll_t)

            # pred.eval()
            pred.eval()
            rw_e = pred(eids)
            rw_e_cast = {k: [a.to(dtype=torch.bfloat16) for a in v] for k, v in rw_e.items()}
            logits_e = fw(eids, rw_e_cast)
            nll_e = F.cross_entropy(
                logits_e.float().view(-1, config["vocab_size"]), elabels.view(-1)
            ).item()
            nll_eval_list.append(nll_e)

    mean_train = sum(nll_train_list) / len(nll_train_list)
    mean_eval = sum(nll_eval_list) / len(nll_eval_list)
    print(f"\n  pred.train() mean NLL = {mean_train:.4f}")
    print(f"  pred.eval()  mean NLL = {mean_eval:.4f}")
    print(f"  diff (train - eval)   = {mean_train - mean_eval:+.4f}")

    print("\n  Per-batch diffs (first 10):")
    for i in range(min(10, len(nll_train_list))):
        print(f"    batch {i}: train={nll_train_list[i]:.4f}  "
              f"eval={nll_eval_list[i]:.4f}  "
              f"diff={nll_train_list[i] - nll_eval_list[i]:+.4f}")

    print("\n  Interpretation:")
    if abs(mean_train - mean_eval) < 0.05:
        print("    ▸ Train/eval predictor mode produces similar NLL.")
        print("    ▸ The train/eval asymmetry does NOT explain the ~2.0 gap.")
        print("    ▸ Most of the gap is genuine overfit.")
    elif mean_train < mean_eval - 0.3:
        print("    ▸ pred.train() mode gives LOWER NLL than pred.eval() on REAL data.")
        print("    ▸ The train-mode predictor is doing something eval-mode isn't.")
        print("    ▸ THIS MIGHT BE THE BUG. Investigation warranted.")
    else:
        print("    ▸ Moderate difference. Worth investigating but not smoking gun.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
