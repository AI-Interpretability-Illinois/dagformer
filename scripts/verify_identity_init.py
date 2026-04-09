"""Verify that FourWayDAGFormer with identity-init predictor produces
numerically identical logits to the underlying dense OLMo model.

This directly tests Codex's Q1 finding: "Identity init is implemented by
construction but never proven numerically equivalent." If this test fails,
we've found a structural bug explaining the ~2.0 train/eval gap.

Usage:
    python scripts/verify_identity_init.py
    python scripts/verify_identity_init.py --config configs/fourway_corrected_joint_causal.yaml
    python scripts/verify_identity_init.py --dtype float32  # rule out bf16 noise
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


def load_config(path: str | None) -> dict:
    """Load a fourway yaml. If path is None, use a minimal default."""
    if path is None:
        return {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "vocab_size": 100352,
            "tie_word_embeddings": True,
            "max_position_embeddings": 4096,
            "predictor_encoder_dim": 256,
            "predictor_encoder_layers": 2,
            "predictor_encoder_heads": 4,
            "predictor_max_seq_len": 4096,
            "fourway_hidden": 512,
            "predictor_causal": True,
        }
    with open(path) as f:
        return yaml.safe_load(f)


def build_dense(config: dict, device: torch.device, dtype: torch.dtype) -> Olmo2ForCausalLM:
    """Build a randomly-initialized dense Olmo2 model (matches create_model)."""
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
    torch.manual_seed(0)
    model = Olmo2ForCausalLM(model_config).to(device=device, dtype=dtype)
    model.eval()
    return model


def build_predictor(config: dict, device: torch.device) -> FourWayPredictor:
    """Build FourWayPredictor with identity init (always float32)."""
    predictor = FourWayPredictor(
        vocab_size=config["vocab_size"],
        encoder_dim=config["predictor_encoder_dim"],
        encoder_layers=config["predictor_encoder_layers"],
        encoder_heads=config["predictor_encoder_heads"],
        max_seq_len=config["predictor_max_seq_len"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        hidden_dim=config["fourway_hidden"],
        causal=config.get("predictor_causal", True),
    ).to(device)
    predictor.eval()
    return predictor


def check_identity_routing(rw: dict[str, list[torch.Tensor]]) -> None:
    """Assert the predictor output is exactly identity."""
    for stream in ("q", "k", "v", "r"):
        for l_idx, α in enumerate(rw[stream]):
            n_src = α.shape[-1]
            # Expected identity: last source = 1, rest = 0
            expected = torch.zeros_like(α)
            expected[..., -1] = 1.0
            diff = (α - expected).abs().max().item()
            assert diff < 1e-5, (
                f"Predictor output at stream={stream} layer_idx={l_idx} is NOT identity. "
                f"Max deviation = {diff:.2e} (expected 0). "
                f"This means the identity init itself is broken."
            )
    print(f"  [OK] Predictor output is exactly identity for all streams/layers")


def run_dense_forward(model: Olmo2ForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard HF forward."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    return out.logits


def run_fourway_forward(
    fourway: FourWayDAGFormer,
    predictor: FourWayPredictor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """FourWay forward with identity-init predictor."""
    with torch.no_grad():
        rw = predictor(input_ids)
        # sanity check: is the predictor output actually identity?
        check_identity_routing(rw)
        # cast to model dtype (mirrors training path)
        model_dtype = next(fourway.olmo.parameters()).dtype
        rw_cast = {
            k: [a.to(dtype=model_dtype) for a in v]
            for k, v in rw.items()
        }
        logits = fourway(input_ids, rw_cast)
    return logits


def report_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute numerical diff stats and print."""
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    stats = {
        "shape": tuple(a.shape),
        "dtype": str(a.dtype),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "p99_abs_diff": diff.flatten().kthvalue(int(0.99 * diff.numel())).values.item(),
        "max_rel_diff": (diff / (b32.abs() + 1e-6)).max().item(),
        "a_norm": a32.norm().item(),
        "b_norm": b32.norm().item(),
        "diff_norm_ratio": (diff.norm() / (b32.norm() + 1e-12)).item(),
    }
    print(f"\n[{name}]")
    print(f"  shape={stats['shape']}, dtype={stats['dtype']}")
    print(f"  max_abs_diff    = {stats['max_abs_diff']:.6e}")
    print(f"  mean_abs_diff   = {stats['mean_abs_diff']:.6e}")
    print(f"  p99_abs_diff    = {stats['p99_abs_diff']:.6e}")
    print(f"  max_rel_diff    = {stats['max_rel_diff']:.6e}")
    print(f"  a_norm          = {stats['a_norm']:.4f}")
    print(f"  b_norm          = {stats['b_norm']:.4f}")
    print(f"  diff/b ratio    = {stats['diff_norm_ratio']:.6e}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="yaml config path")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tol_bf16",
        type=float,
        default=1e-2,
        help="max_abs_diff threshold for PASS in bf16",
    )
    parser.add_argument(
        "--tol_fp32",
        type=float,
        default=1e-5,
        help="max_abs_diff threshold for PASS in fp32",
    )
    parser.add_argument(
        "--use_v_norm",
        action="store_true",
        help="enable post-mix V RMSNorm in FourWayDAGFormer",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")
    print(f"Config: {args.config or '<default>'}")
    print(
        f"Model:  hidden={config['hidden_size']}, "
        f"layers={config['num_hidden_layers']}, heads={config['num_attention_heads']}"
    )

    # Build models (dense OLMo + FourWay wrapper sharing the same weights)
    print("\nBuilding dense OLMo...")
    dense_model = build_dense(config, device, dtype)

    print("Building FourWayDAGFormer wrapper (shares weights with dense)...")
    fourway = FourWayDAGFormer(
        model=dense_model,
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        use_local_correction=False,  # pure predictor path; correction starts at 0 anyway
        correction_hidden=128,
        use_triton_kernel=False,
        use_v_norm=args.use_v_norm,
    ).to(device=device, dtype=dtype)
    fourway.eval()

    print("Building FourWayPredictor (identity init)...")
    predictor = build_predictor(config, device)

    # Fixed random input
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, config["vocab_size"], (args.batch, args.seq_len), device=device
    )
    print(f"Input:  shape={tuple(input_ids.shape)}")

    # Dense forward
    print("\nRunning dense forward...")
    dense_logits = run_dense_forward(dense_model, input_ids)

    # FourWay forward
    print("Running FourWay forward with identity predictor...")
    fourway_logits = run_fourway_forward(fourway, predictor, input_ids)

    # Compare
    stats = report_diff("logits", fourway_logits, dense_logits)

    # NLL comparison on a fixed label sequence
    labels = torch.randint(0, config["vocab_size"], (args.batch, args.seq_len), device=device)
    with torch.no_grad():
        dense_nll = torch.nn.functional.cross_entropy(
            dense_logits.float().view(-1, config["vocab_size"]),
            labels.view(-1),
        ).item()
        fourway_nll = torch.nn.functional.cross_entropy(
            fourway_logits.float().view(-1, config["vocab_size"]),
            labels.view(-1),
        ).item()
    print("\n[NLL on random labels]")
    print(f"  dense_nll   = {dense_nll:.6f}")
    print(f"  fourway_nll = {fourway_nll:.6f}")
    print(f"  diff        = {abs(dense_nll - fourway_nll):.6e}")

    tol = args.tol_fp32 if dtype == torch.float32 else args.tol_bf16
    print(f"\nTolerance ({args.dtype}): max_abs_diff < {tol:.2e}")
    if stats["max_abs_diff"] < tol:
        print("RESULT: PASS — FourWay at identity init matches dense OLMo.")
        return 0
    else:
        print("RESULT: FAIL — FourWay diverges from dense at identity init.")
        print("This is a structural bug that explains the train/eval gap.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
