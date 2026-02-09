"""Evaluate a trained DAGFormer checkpoint.

Usage:
    python scripts/eval.py --config configs/sanity_check.yaml --checkpoint checkpoints/checkpoint_step1000.pt
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dolma import build_eval_dataloader
from src.model.olmo_graph import DAGFormerOLMo, create_all_ones_A
from src.model.predictor import StructurePredictor
from src.training.checkpointing import load_checkpoint
from src.training.trainer import TrainConfig
from src.utils.topology import compute_topology_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAGFormer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    device = torch.device(args.device)

    # Load models
    print(f"Loading {config.olmo_model_id}...")
    olmo = AutoModelForCausalLM.from_pretrained(
        config.olmo_model_id, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in olmo.parameters():
        p.requires_grad_(False)

    olmo_tokenizer = AutoTokenizer.from_pretrained(config.olmo_model_id)

    olmo_wrapper = DAGFormerOLMo(model=olmo, input_norm=config.input_norm).to(device)

    print(f"Loading {config.qwen_model_id}...")
    predictor = StructurePredictor(
        qwen_model_id=config.qwen_model_id,
        hidden_dim=config.predictor_hidden_dim,
        rank=config.predictor_rank,
        cascading_gate_k=config.cascading_gate_k,
        qwen_input_prefix=config.qwen_input_prefix,
        device=device,
    )

    # Load checkpoint
    load_checkpoint(args.checkpoint, predictor, device=device)
    predictor.eval()

    # Build eval data
    cache_path = f"{config.save_dir}/eval_cache.pt"
    eval_batches = build_eval_dataloader(
        olmo_tokenizer=olmo_tokenizer,
        seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        dataset_name=config.dataset,
        dataset_version=config.dataset_name,
        eval_skip=config.eval_skip,
        eval_size=config.eval_size,
        cache_path=cache_path,
    )

    vocab_size = olmo.config.vocab_size
    tau = config.tau_final  # use final temperature for eval

    # Evaluate
    nll_soft_sum = 0.0
    nll_hard_sum = 0.0
    nll_baseline_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in eval_batches:
            olmo_ids = batch["olmo_ids"].to(device)
            olmo_labels = batch["olmo_labels"].to(device)
            raw_texts = batch["raw_text"]

            # Soft
            A_soft = predictor(raw_texts, tau=tau, mode="eval_soft")
            logits_soft = olmo_wrapper(olmo_ids, A_soft)
            nll_soft = F.cross_entropy(
                logits_soft[:, :-1].contiguous().view(-1, vocab_size),
                olmo_labels[:, 1:].contiguous().view(-1),
            )
            nll_soft_sum += nll_soft.item()

            # Hard
            A_hard = predictor(raw_texts, tau=tau, mode="eval_hard")
            logits_hard = olmo_wrapper(olmo_ids, A_hard)
            nll_hard = F.cross_entropy(
                logits_hard[:, :-1].contiguous().view(-1, vocab_size),
                olmo_labels[:, 1:].contiguous().view(-1),
            )
            nll_hard_sum += nll_hard.item()

            # Baseline
            A_ones = create_all_ones_A(olmo_ids.shape[0]).to(device)
            logits_base = olmo_wrapper(olmo_ids, A_ones)
            nll_base = F.cross_entropy(
                logits_base[:, :-1].contiguous().view(-1, vocab_size),
                olmo_labels[:, 1:].contiguous().view(-1),
            )
            nll_baseline_sum += nll_base.item()

            # Topology
            topo = compute_topology_metrics(A_soft)

            n += 1

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({n} batches)")
    print(f"{'='*50}")
    print(f"  eval/nll_soft:     {nll_soft_sum / n:.4f}")
    print(f"  eval/nll_hard:     {nll_hard_sum / n:.4f}")
    print(f"  eval/nll_baseline: {nll_baseline_sum / n:.4f}")
    print(f"  topology/mean_A:   {topo['topology/mean_A']:.4f}")
    print(f"  topology/seq_gate: {topo['topology/seq_gate_frac']:.4f}")
    print(f"  topology/hyp_gate: {topo['topology/hyp_gate_frac']:.4f}")


if __name__ == "__main__":
    main()
