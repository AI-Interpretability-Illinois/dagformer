"""Extract and analyze the learned topology A from a checkpoint.

Since jaccard_var=0 (context-independent), A is the same for all inputs.
We run the predictor on a few eval samples to confirm, then analyze the structure.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from src.model.predictor import StructurePredictor
from src.model.olmo_graph import create_block_upper_triangular_mask
from src.data.dolma import build_eval_dataloader
from src.training.checkpointing import load_checkpoint


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tau", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = state.get("config", {})
    step = state.get("step", "?")
    print(f"Step: {step}")

    # Build predictor
    qwen_model_id = config_dict.get("qwen_model_id", "Qwen/Qwen3-Embedding-0.6B")
    predictor = StructurePredictor(
        qwen_model_id=qwen_model_id,
        hidden_dim=config_dict.get("predictor_hidden_dim", 1024),
        rank=config_dict.get("predictor_rank", 32),
        cascading_gate_k=config_dict.get("cascading_gate_k", 5.0),
        init_logit=config_dict.get("init_logit", 15.0),
    )
    predictor.load_state_dict(state["predictor_state_dict"])
    predictor = predictor.to(device).eval()

    # Build eval data for Qwen input
    olmo_model_id = config_dict.get("olmo_model_id", "allenai/OLMo-2-0425-1B")
    olmo_tokenizer = AutoTokenizer.from_pretrained(olmo_model_id)
    ds_name = config_dict.get("dataset", "allenai/dolmino-mix-1124")
    ds_version = config_dict.get("dataset_name", "dolmino_mix")

    cache_path = os.path.join(
        config_dict.get("save_dir", "checkpoints/"),
        "eval_cache.pt",
    )
    eval_batches = build_eval_dataloader(
        olmo_tokenizer=olmo_tokenizer,
        seq_len=config_dict.get("seq_len", 1024),
        batch_size=1,
        dataset_name=ds_name,
        dataset_version=ds_version,
        eval_skip=config_dict.get("eval_skip", 10000),
        eval_size=5,  # only need a few
        cache_path=cache_path,
    )

    tau = args.tau
    mask = create_block_upper_triangular_mask().to(device)
    valid_mask = mask.bool()

    # Collect A from multiple samples
    all_A = []
    with torch.no_grad():
        for batch in eval_batches[:5]:
            raw_text = batch.get("raw_text", None)
            if raw_text is None:
                # Decode from olmo_ids
                raw_text = [olmo_tokenizer.decode(ids) for ids in batch["olmo_ids"]]

            A = predictor(raw_text, mode="eval_soft", tau=tau)  # [1, 256, 256]
            all_A.append(A[0].cpu())

    A_stack = torch.stack(all_A)  # [N, 256, 256]
    A_mean = A_stack.mean(dim=0)  # [256, 256]

    # Check variance across samples (should be ~0 if context-independent)
    A_var = A_stack.var(dim=0)
    print(f"\nContext independence check (N={len(all_A)} samples):")
    print(f"  Max var across samples: {A_var[valid_mask.cpu()].max().item():.6f}")
    print(f"  Mean var: {A_var[valid_mask.cpu()].mean().item():.6f}")

    # Use the mean A for analysis
    A = A_mean
    valid_A = A[valid_mask.cpu()]

    print(f"\n{'='*60}")
    print(f"TOPOLOGY ANALYSIS (step {step}, tau={tau})")
    print(f"{'='*60}")

    print(f"\nOverall stats (30720 valid entries):")
    print(f"  Mean A:   {valid_A.mean().item():.4f}")
    print(f"  Std A:    {valid_A.std().item():.4f}")
    print(f"  Min A:    {valid_A.min().item():.4f}")
    print(f"  Max A:    {valid_A.max().item():.4f}")
    print(f"  A < 0.5:  {(valid_A < 0.5).sum().item()} ({100*(valid_A < 0.5).float().mean().item():.1f}%)")
    print(f"  A < 0.9:  {(valid_A < 0.9).sum().item()} ({100*(valid_A < 0.9).float().mean().item():.1f}%)")
    print(f"  A < 0.95: {(valid_A < 0.95).sum().item()} ({100*(valid_A < 0.95).float().mean().item():.1f}%)")
    print(f"  A > 0.99: {(valid_A > 0.99).sum().item()} ({100*(valid_A > 0.99).float().mean().item():.1f}%)")

    # Hard threshold
    A_hard = (A > 0.5).float()
    hard_valid = A_hard[valid_mask.cpu()]
    n_off = (hard_valid < 0.5).sum().item()
    print(f"\nHard threshold (>0.5):")
    print(f"  ON:  {int(hard_valid.sum().item())} / 30720")
    print(f"  OFF: {n_off} / 30720 ({100*n_off/30720:.1f}%)")

    # Layer-pair analysis
    layer_idx = torch.arange(256) // 16
    src_layer = layer_idx.unsqueeze(1).expand(256, 256)
    tgt_layer = layer_idx.unsqueeze(0).expand(256, 256)

    print(f"\nPer-layer-pair mean A:")
    print(f"{'Src→Tgt':>10} {'Mean A':>8} {'Min A':>8} {'Max A':>8} {'<0.5':>6} {'<0.9':>6}")
    for sl in range(16):
        for tl in range(sl + 1, 16):
            pair_mask = (src_layer == sl) & (tgt_layer == tl)
            pair_A = A[pair_mask]
            if len(pair_A) == 0:
                continue
            n_low = (pair_A < 0.5).sum().item()
            n_below9 = (pair_A < 0.9).sum().item()
            print(f"  L{sl:>2}→L{tl:<2}  {pair_A.mean().item():>8.4f} {pair_A.min().item():>8.4f} {pair_A.max().item():>8.4f} {n_low:>5}/256 {n_below9:>5}/256")

    # Per-head analysis: which target heads receive the least?
    print(f"\nTarget heads with lowest mean incoming A (bottom 20):")
    incoming_mean = []
    for j in range(256):
        tgt_l = j // 16
        if tgt_l == 0:
            continue
        sources = A[:j // 16 * 16, j]  # all source nodes in earlier layers
        valid_sources = sources[valid_mask.cpu()[:j // 16 * 16, j]]
        if len(valid_sources) > 0:
            incoming_mean.append((j, valid_sources.mean().item(), valid_sources.min().item()))
    incoming_mean.sort(key=lambda x: x[1])
    for j, mean_a, min_a in incoming_mean[:20]:
        l, h = j // 16, j % 16
        print(f"  Node {j:>3} (L{l:>2}, H{h:>2}): mean_in={mean_a:.4f}, min_in={min_a:.4f}")

    # Per-head analysis: which source heads have lowest outgoing A?
    print(f"\nSource heads with lowest mean outgoing A (bottom 20):")
    outgoing_mean = []
    for i in range(256):
        src_l = i // 16
        if src_l == 15:
            continue
        targets = A[i, (src_l + 1) * 16:]  # all target nodes in later layers
        valid_targets = targets[valid_mask.cpu()[i, (src_l + 1) * 16:]]
        if len(valid_targets) > 0:
            outgoing_mean.append((i, valid_targets.mean().item(), valid_targets.min().item()))
    outgoing_mean.sort(key=lambda x: x[1])
    for i, mean_a, min_a in outgoing_mean[:20]:
        l, h = i // 16, i % 16
        print(f"  Node {i:>3} (L{l:>2}, H{h:>2}): mean_out={mean_a:.4f}, min_out={min_a:.4f}")

    # Most weakened individual connections (top 30 lowest A values)
    print(f"\nWeakest individual connections (30 lowest A):")
    flat_A = A.clone()
    flat_A[~valid_mask.cpu()] = 1.0  # exclude invalid
    flat_vals, flat_idxs = flat_A.view(-1).sort()
    for k in range(30):
        idx = flat_idxs[k].item()
        i, j = idx // 256, idx % 256
        sl, sh = i // 16, i % 16
        tl, th = j // 16, j % 16
        print(f"  A[{i},{j}] = {flat_vals[k].item():.4f}  (L{sl}H{sh} → L{tl}H{th})")

    # Adjacent vs skip
    adj_mask = ((tgt_layer - src_layer) == 1) & valid_mask.cpu()
    skip_mask_t = ((tgt_layer - src_layer) > 1) & valid_mask.cpu()
    adj_A = A[adj_mask]
    skip_A = A[skip_mask_t]
    print(f"\nAdjacent vs Skip:")
    print(f"  Adjacent (layer diff=1): mean={adj_A.mean().item():.4f}, <0.9: {(adj_A < 0.9).sum().item()}/3840")
    print(f"  Skip     (layer diff>1): mean={skip_A.mean().item():.4f}, <0.9: {(skip_A < 0.9).sum().item()}/26880")

    # Save raw A for further analysis
    out_path = args.checkpoint.replace(".pt", "_topology.pt")
    torch.save({"A": A, "step": step, "tau": tau}, out_path)
    print(f"\nRaw A matrix saved to: {out_path}")


if __name__ == "__main__":
    main()
