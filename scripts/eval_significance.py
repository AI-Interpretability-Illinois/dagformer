"""Significance test: paired t-test comparing nll_hard vs nll_baseline per sequence.

Loads a checkpoint, runs eval on a large set, computes per-sequence NLL
for both hard topology (from predictor) and baseline (A=1), then does
a paired t-test to determine if the difference is statistically significant.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dolma import build_eval_dataloader
from src.model.olmo_graph import DAGFormerOLMo, create_all_ones_A
from src.model.predictor import StructurePredictor
from src.training.checkpointing import load_checkpoint


def per_sequence_nll(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Compute NLL per sequence (not averaged across sequences).

    Args:
        logits: [batch, seq, vocab]
        labels: [batch, seq]
    Returns:
        nll: [batch] — mean NLL per token for each sequence
    """
    batch, seq = labels.shape
    # Per-token cross entropy
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).reshape(batch, seq)
    # Mean per sequence
    return per_token.mean(dim=1)  # [batch]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--eval-skip", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    step = state.get("step", "?")

    # Load models
    olmo_model_id = config.get("olmo_model_id", "allenai/OLMo-2-0425-1B")
    olmo_revision = config.get("olmo_revision", None)
    qwen_model_id = config.get("qwen_model_id", "Qwen/Qwen3-Embedding-0.6B")
    tau = config.get("tau_init", 5.0)

    print(f"Step: {step}, tau: {tau}")
    print(f"OLMo: {olmo_model_id} @ {olmo_revision or 'main'}")

    # Load OLMo
    olmo = AutoModelForCausalLM.from_pretrained(
        olmo_model_id,
        revision=olmo_revision,
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
    ).to(device).eval()

    # Load OLMo state from checkpoint (Phase 2 — weights changed)
    olmo_state_path = config.get("olmo_state_path", None)
    if "olmo_state_dict" in state:
        olmo.load_state_dict(state["olmo_state_dict"])
        print("Loaded OLMo state from checkpoint (inline)")
    else:
        # Try separate file
        olmo_path = args.checkpoint.replace(".pt", "_olmo.pt")
        if os.path.exists(olmo_path):
            olmo_state = torch.load(olmo_path, map_location=device, weights_only=True)
            olmo.load_state_dict(olmo_state)
            print(f"Loaded OLMo state from {olmo_path}")
            del olmo_state

    for p in olmo.parameters():
        p.requires_grad_(False)

    olmo_tokenizer = AutoTokenizer.from_pretrained(olmo_model_id)
    olmo_wrapper = DAGFormerOLMo(
        model=olmo,
        input_norm=config.get("input_norm", "none"),
    ).to(device)

    # Load predictor
    predictor = StructurePredictor(
        qwen_model_id=qwen_model_id,
        hidden_dim=config.get("predictor_hidden_dim", 1024),
        rank=config.get("predictor_rank", 32),
        cascading_gate_k=config.get("cascading_gate_k", 5.0),
        init_logit=config.get("init_logit", 15.0),
    )
    predictor.load_state_dict(state["predictor_state_dict"])
    predictor = predictor.to(device).eval()

    # Build eval data (large set)
    ds_name = config.get("dataset", "allenai/dolmino-mix-1124")
    ds_version = config.get("dataset_name", "dolmino_mix")
    cache_tag = ds_version.replace("/", "_")
    cache_path = os.path.join(
        "checkpoints/eval_baselines",
        f"eval_cache_{cache_tag}_skip{args.eval_skip}_size{args.eval_size}.pt",
    )
    os.makedirs("checkpoints/eval_baselines", exist_ok=True)

    print(f"Building eval set: skip={args.eval_skip}, size={args.eval_size}")
    eval_batches = build_eval_dataloader(
        olmo_tokenizer=olmo_tokenizer,
        seq_len=config.get("seq_len", 1024),
        batch_size=args.batch_size,
        dataset_name=ds_name,
        dataset_version=ds_version,
        eval_skip=args.eval_skip,
        eval_size=args.eval_size,
        cache_path=cache_path,
    )

    vocab_size = olmo.config.vocab_size
    all_nll_hard = []
    all_nll_baseline = []
    n_seq = 0

    print(f"\nRunning paired eval...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_batches):
            olmo_ids = batch["olmo_ids"].to(device)
            olmo_labels = batch["olmo_labels"].to(device)
            bs = olmo_ids.shape[0]

            raw_text = batch.get("raw_text", None)
            if raw_text is None:
                raw_text = [olmo_tokenizer.decode(ids) for ids in batch["olmo_ids"]]

            # --- Hard topology from predictor ---
            A_hard = predictor(raw_text, mode="eval_hard", tau=tau)
            logits_hard = olmo_wrapper(olmo_ids, A_hard)
            nll_hard = per_sequence_nll(logits_hard, olmo_labels, vocab_size)

            # --- Baseline (A=1) ---
            A_ones = create_all_ones_A(bs).to(device)
            logits_base = olmo_wrapper(olmo_ids, A_ones)
            nll_base = per_sequence_nll(logits_base, olmo_labels, vocab_size)

            all_nll_hard.extend(nll_hard.cpu().tolist())
            all_nll_baseline.extend(nll_base.cpu().tolist())
            n_seq += bs

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {n_seq} sequences...")

    # Convert to tensors
    nll_hard_t = torch.tensor(all_nll_hard)
    nll_base_t = torch.tensor(all_nll_baseline)
    diff = nll_hard_t - nll_base_t  # negative = topology is better

    n = len(diff)
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    se_diff = std_diff / math.sqrt(n)
    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0

    # Two-sided p-value approximation (normal for large n)
    # For one-sided (topology better): p/2 if mean_diff < 0
    import scipy.stats as stats
    t_result = stats.ttest_rel(all_nll_hard, all_nll_baseline)

    print(f"\n{'='*60}")
    print(f"PAIRED T-TEST RESULTS (n={n} sequences, step={step})")
    print(f"{'='*60}")
    print(f"\n  Mean NLL (hard topology): {nll_hard_t.mean().item():.6f}")
    print(f"  Mean NLL (baseline A=1):  {nll_base_t.mean().item():.6f}")
    print(f"  Mean difference (hard-base): {mean_diff:.6f}")
    print(f"  Std of differences:          {std_diff:.6f}")
    print(f"  Standard error:              {se_diff:.6f}")
    print(f"  95% CI: [{mean_diff - 1.96*se_diff:.6f}, {mean_diff + 1.96*se_diff:.6f}]")
    print(f"\n  t-statistic: {t_result.statistic:.4f}")
    print(f"  p-value (two-sided): {t_result.pvalue:.6f}")
    print(f"  p-value (one-sided, hard < base): {t_result.pvalue/2:.6f}")

    print(f"\n  Sequences where hard < base: {(diff < 0).sum().item()}/{n} ({100*(diff < 0).float().mean().item():.1f}%)")
    print(f"  Sequences where hard > base: {(diff > 0).sum().item()}/{n} ({100*(diff > 0).float().mean().item():.1f}%)")
    print(f"  Sequences where hard = base: {(diff == 0).sum().item()}/{n}")

    # Effect size (Cohen's d)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    print(f"\n  Cohen's d: {cohens_d:.4f}")

    alpha = 0.05
    if t_result.pvalue < alpha:
        if mean_diff < 0:
            print(f"\n  CONCLUSION: Hard topology is SIGNIFICANTLY BETTER than baseline (p={t_result.pvalue:.6f} < {alpha})")
        else:
            print(f"\n  CONCLUSION: Hard topology is SIGNIFICANTLY WORSE than baseline (p={t_result.pvalue:.6f} < {alpha})")
    else:
        print(f"\n  CONCLUSION: NO significant difference (p={t_result.pvalue:.6f} >= {alpha})")

    # Percentile analysis of differences
    print(f"\n  Difference percentiles:")
    for pct in [5, 25, 50, 75, 95]:
        val = torch.quantile(diff, pct / 100).item()
        print(f"    {pct}th: {val:.6f}")


if __name__ == "__main__":
    main()
