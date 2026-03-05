"""Evaluate baseline NLL for different OLMo checkpoints.

Usage:
    python scripts/eval_baseline.py --revision stage1-step1907359-tokens4001B
    python scripts/eval_baseline.py --revision stage2-ingredient3-step23852-tokens51B
    python scripts/eval_baseline.py  # default: main branch
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dolma import build_eval_dataloader
from src.model.olmo_graph import DAGFormerOLMo, create_all_ones_A


def main():
    parser = argparse.ArgumentParser(description="Evaluate OLMo baseline NLL")
    parser.add_argument("--model-id", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None,
                        help="HuggingFace branch/revision to load")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--eval-skip", type=int, default=10000)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-dataset", type=str, default="allenai/dolma",
                        help="HuggingFace dataset for eval")
    parser.add_argument("--eval-dataset-name", type=str, default="v1_7",
                        help="Dataset config/version (e.g. 'v1_7' or 'dolmino_mix')")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    rev_label = args.revision or "main"
    print(f"=== Evaluating {args.model_id} @ {rev_label} ===")

    # Load model
    print(f"Loading model (revision={rev_label})...")
    olmo = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
    ).to(device).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)

    # Tokenizer (same across revisions)
    olmo_tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Wrap with DAGFormerOLMo (A=1 gives vanilla forward)
    olmo_wrapper = DAGFormerOLMo(model=olmo, input_norm="none").to(device)

    # Build eval data
    cache_dir = "checkpoints/eval_baselines"
    os.makedirs(cache_dir, exist_ok=True)
    ds_tag = args.eval_dataset_name.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"eval_cache_{ds_tag}_skip{args.eval_skip}_size{args.eval_size}.pt")

    eval_batches = build_eval_dataloader(
        olmo_tokenizer=olmo_tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dataset_name=args.eval_dataset,
        dataset_version=args.eval_dataset_name,
        eval_skip=args.eval_skip,
        eval_size=args.eval_size,
        cache_path=cache_path,
    )

    vocab_size = olmo.config.vocab_size

    # Evaluate baseline NLL
    nll_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in eval_batches:
            olmo_ids = batch["olmo_ids"].to(device)
            olmo_labels = batch["olmo_labels"].to(device)

            A_ones = create_all_ones_A(olmo_ids.shape[0]).to(device)
            logits = olmo_wrapper(olmo_ids, A_ones)
            nll = F.cross_entropy(
                logits.contiguous().view(-1, vocab_size),
                olmo_labels.contiguous().view(-1),
            )
            nll_sum += nll.item()
            n += 1

    avg_nll = nll_sum / n
    print(f"\n{'='*50}")
    print(f"  Model:    {args.model_id}")
    print(f"  Revision: {rev_label}")
    print(f"  Eval data: {args.eval_dataset} / {args.eval_dataset_name}")
    print(f"  NLL:      {avg_nll:.4f}")
    print(f"  Batches:  {n}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
