"""Gradient analysis at A=1: is A=1 a local/global optimum for frozen OLMo?

Computes ∂L/∂A at A=1 across multiple eval windows.
- If ∂L/∂A[i,j] ≤ 0 for all valid (i,j): A=1 is optimal (can't improve by reducing any connection)
- If ∂L/∂A[i,j] > 0 for some (i,j): those connections should be reduced → A=1 is NOT optimal

Usage:
    python scripts/gradient_analysis.py [--revision stage1-step1907359-tokens4001B]
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dolma import build_eval_dataloader
from src.model.olmo_graph import (
    DAGFormerOLMo,
    create_all_ones_A,
    create_block_upper_triangular_mask,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--eval-skip", type=int, default=10000)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-dataset", type=str, default="allenai/dolma")
    parser.add_argument("--eval-dataset-name", type=str, default="v1_7")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load OLMo weights from a training checkpoint (e.g. checkpoints/p2_ddp_long/checkpoint_step5000)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    rev_label = args.revision or "main"
    ckpt_label = f" + checkpoint {args.checkpoint}" if args.checkpoint else ""
    print(f"=== Gradient Analysis at A=1 for {args.model_id} @ {rev_label}{ckpt_label} ===\n")

    # Load model
    print("Loading model...")
    olmo = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
    ).to(device).eval()

    # Optionally load fine-tuned OLMo weights from checkpoint
    if args.checkpoint:
        ckpt_base = args.checkpoint.rstrip("/")
        # Try _olmo.pt suffix first
        olmo_path = ckpt_base + "_olmo.pt"
        if not os.path.exists(olmo_path):
            olmo_path = ckpt_base + ".pt"
        if os.path.exists(olmo_path.replace(".pt", "_olmo.pt")):
            olmo_path = olmo_path.replace(".pt", "_olmo.pt")
        print(f"Loading OLMo weights from {olmo_path}...")
        olmo_state = torch.load(olmo_path, map_location=device)
        olmo.load_state_dict(olmo_state)
        del olmo_state
        print("OLMo weights loaded.")

    for p in olmo.parameters():
        p.requires_grad_(False)

    olmo_tokenizer = AutoTokenizer.from_pretrained(args.model_id)
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
    mask = create_block_upper_triangular_mask().to(device)  # [256, 256]
    valid_mask = mask.bool()  # True for valid entries (30720)
    num_valid = valid_mask.sum().item()

    # Layer indices for categorization
    layer_idx = torch.arange(256) // 16
    # adjacent: layer(j) = layer(i) + 1
    src_layer = layer_idx.unsqueeze(1).expand(256, 256)
    tgt_layer = layer_idx.unsqueeze(0).expand(256, 256)
    adjacent_mask = ((tgt_layer - src_layer) == 1) & valid_mask.cpu()
    skip_mask = ((tgt_layer - src_layer) > 1) & valid_mask.cpu()
    adjacent_mask = adjacent_mask.to(device)
    skip_mask = skip_mask.to(device)

    print(f"Valid entries: {num_valid}")
    print(f"Adjacent entries: {adjacent_mask.sum().item()}")
    print(f"Skip entries: {skip_mask.sum().item()}\n")

    # Collect gradients across all eval windows — process one sample at a time to avoid OOM
    all_grads = []  # list of [256, 256] tensors
    all_nll = []
    n_windows = 0

    for batch_idx, batch in enumerate(eval_batches):
        olmo_ids_batch = batch["olmo_ids"].to(device)
        olmo_labels_batch = batch["olmo_labels"].to(device)
        bs = olmo_ids_batch.shape[0]

        for s in range(bs):
            # Process one sample at a time to fit in GPU memory
            olmo_ids = olmo_ids_batch[s:s+1]
            olmo_labels = olmo_labels_batch[s:s+1]

            A = create_all_ones_A(1).to(device).requires_grad_(True)

            logits = olmo_wrapper(olmo_ids, A)
            nll = F.cross_entropy(
                logits.contiguous().view(-1, vocab_size),
                olmo_labels.contiguous().view(-1),
            )
            nll.backward()

            grad = A.grad.detach()[0]  # [256, 256]
            all_grads.append(grad.cpu())
            all_nll.append(nll.item())

            valid_g = grad[valid_mask]
            n_pos = (valid_g > 0).sum().item()
            n_neg = (valid_g < 0).sum().item()
            n_zero = (valid_g == 0).sum().item()
            print(f"  Window {n_windows}: NLL={nll.item():.4f}, "
                  f"grad>0: {n_pos}/{num_valid} ({100*n_pos/num_valid:.1f}%), "
                  f"grad<0: {n_neg}/{num_valid} ({100*n_neg/num_valid:.1f}%), "
                  f"grad=0: {n_zero}/{num_valid}")

            n_windows += 1

            # Free memory
            del A, logits, nll, grad
            torch.cuda.empty_cache()

    # Aggregate across all batches
    avg_grad = torch.stack(all_grads).mean(dim=0)  # [256, 256]
    avg_nll = sum(all_nll) / len(all_nll)

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({n_windows} windows, avg NLL={avg_nll:.4f})")
    print(f"{'='*60}\n")

    # Stats on valid entries of the averaged gradient
    valid_grad = avg_grad[valid_mask.cpu()]

    n_pos = (valid_grad > 0).sum().item()
    n_neg = (valid_grad < 0).sum().item()
    n_zero = (valid_grad == 0).sum().item()

    print(f"Averaged gradient ∂L/∂A at A=1 (over {n_windows} windows):")
    print(f"  Positive (should reduce A): {n_pos}/{num_valid} ({100*n_pos/num_valid:.1f}%)")
    print(f"  Negative (A=1 is optimal):  {n_neg}/{num_valid} ({100*n_neg/num_valid:.1f}%)")
    print(f"  Zero:                        {n_zero}/{num_valid}")
    print()
    print(f"  Mean gradient:    {valid_grad.mean().item():.6f}")
    print(f"  Std gradient:     {valid_grad.std().item():.6f}")
    print(f"  Max gradient:     {valid_grad.max().item():.6f}")
    print(f"  Min gradient:     {valid_grad.min().item():.6f}")
    print(f"  Mean |gradient|:  {valid_grad.abs().mean().item():.6f}")

    # Breakdown: adjacent vs skip
    adj_grad = avg_grad[adjacent_mask.cpu()]
    skip_grad = avg_grad[skip_mask.cpu()]

    print(f"\nAdjacent connections (layer diff = 1):")
    print(f"  Positive: {(adj_grad > 0).sum().item()}/{len(adj_grad)} ({100*(adj_grad > 0).float().mean().item():.1f}%)")
    print(f"  Mean grad: {adj_grad.mean().item():.6f}")
    print(f"  Mean |grad|: {adj_grad.abs().mean().item():.6f}")

    print(f"\nSkip connections (layer diff > 1):")
    print(f"  Positive: {(skip_grad > 0).sum().item()}/{len(skip_grad)} ({100*(skip_grad > 0).float().mean().item():.1f}%)")
    print(f"  Mean grad: {skip_grad.mean().item():.6f}")
    print(f"  Mean |grad|: {skip_grad.abs().mean().item():.6f}")

    # Per-layer-pair breakdown
    print(f"\nPer-layer-pair gradient (mean ∂L/∂A, averaged over head pairs & windows):")
    print(f"{'Src→Tgt':>10} {'Mean grad':>12} {'|grad|':>10} {'%pos':>8}")
    for sl in range(16):
        for tl in range(sl + 1, 16):
            pair_mask = (src_layer == sl) & (tgt_layer == tl)
            pair_grad = avg_grad[pair_mask.cpu()]
            if len(pair_grad) == 0:
                continue
            pct_pos = 100 * (pair_grad > 0).float().mean().item()
            # Only print interesting pairs (first few, last few, and extreme)
            if tl - sl <= 2 or tl >= 14 or sl == 0:
                print(f"  L{sl:>2}→L{tl:<2}  {pair_grad.mean().item():>12.6f} {pair_grad.abs().mean().item():>10.6f} {pct_pos:>7.1f}%")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print(f"{'='*60}")
    if n_pos == 0:
        print("∂L/∂A ≤ 0 for ALL valid entries → A=1 is PROVEN optimal (local, and global in [0,1]).")
        print("No connection can be reduced to improve NLL. Frozen OLMo truly needs A=1.")
    elif n_pos < num_valid * 0.01:
        print(f"Only {n_pos} entries ({100*n_pos/num_valid:.2f}%) have positive gradient.")
        print("A=1 is NEAR-optimal. Tiny improvements possible but practically insignificant.")
    elif n_pos < num_valid * 0.5:
        print(f"{n_pos} entries ({100*n_pos/num_valid:.1f}%) have positive gradient.")
        print("A=1 is NOT optimal! Some connections should be reduced.")
        print("The predictor should be able to learn this — investigate why it can't.")
    else:
        print(f"{n_pos} entries ({100*n_pos/num_valid:.1f}%) have positive gradient.")
        print("A=1 is far from optimal! Many connections should be reduced.")
        print("This contradicts our Phase 1 results — need deeper investigation.")


if __name__ == "__main__":
    main()
