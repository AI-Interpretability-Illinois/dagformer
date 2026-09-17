"""Verify: does the predictor produce different routing weights in train vs eval mode?

The hypothesis: nn.TransformerEncoder uses a fused fastpath when
self.training=False AND torch.no_grad(), producing numerically different
results from the standard path used during training.
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from src.model.predictor import FourWayPredictor


def main():
    device = "cuda"
    dtype = torch.bfloat16

    # Create predictor matching the training config
    predictor = FourWayPredictor(
        vocab_size=100352,
        encoder_dim=256,
        encoder_layers=2,
        encoder_heads=4,
        max_seq_len=4096,
        num_layers=12,
        num_heads=16,
        hidden_dim=512,
        causal=True,
        dropout=0.0,
    ).to(device, dtype=dtype)

    # Load weights from checkpoint if available
    ckpt_path = "checkpoints/fourway_full_fix_1gpu/checkpoint_step1000.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        print(f"Loaded predictor from {ckpt_path}")
        del ckpt
    except Exception as e:
        print(f"No checkpoint loaded ({e}), using random init")

    input_ids = torch.randint(0, 1000, (2, 128), device=device)

    # === Path 1: TRAINING mode (standard kernel) ===
    predictor.train()
    with torch.enable_grad():
        rw_train = predictor(input_ids)

    # === Path 2: EVAL mode + no_grad (fused fastpath) ===
    predictor.eval()
    with torch.no_grad():
        rw_eval = predictor(input_ids)

    # === Path 3: EVAL mode but WITH grad (no fastpath) ===
    predictor.eval()
    with torch.enable_grad():
        rw_eval_grad = predictor(input_ids)

    # === Path 4: TRAIN mode + no_grad ===
    predictor.train()
    with torch.no_grad():
        rw_train_nograd = predictor(input_ids)

    # Compare
    print("\n=== Routing Weight Comparison ===")
    for stream in ('q', 'k', 'v', 'r'):
        for i in range(len(rw_train[stream])):
            t = rw_train[stream][i].detach().float()
            e = rw_eval[stream][i].detach().float()
            eg = rw_eval_grad[stream][i].detach().float()
            tng = rw_train_nograd[stream][i].detach().float()

            diff_te = (t - e).abs().max().item()
            diff_teg = (t - eg).abs().max().item()
            diff_ttng = (t - tng).abs().max().item()
            diff_etng = (e - tng).abs().max().item()

            if i == 0 or i == 5 or i == 10:  # Sample layers
                print(f"  {stream} layer {i+1}: "
                      f"train_vs_eval={diff_te:.4e}  "
                      f"train_vs_eval+grad={diff_teg:.4e}  "
                      f"train_vs_train+nograd={diff_ttng:.4e}  "
                      f"eval_vs_train+nograd={diff_etng:.4e}")

    # Overall summary
    max_diff = 0
    mean_diff_sum = 0
    count = 0
    for stream in ('q', 'k', 'v', 'r'):
        for i in range(len(rw_train[stream])):
            t = rw_train[stream][i].detach().float()
            e = rw_eval[stream][i].detach().float()
            diff = (t - e).abs()
            max_diff = max(max_diff, diff.max().item())
            mean_diff_sum += diff.mean().item()
            count += 1

    print(f"\n=== Overall train vs eval ===")
    print(f"Max diff:  {max_diff:.6e}")
    print(f"Mean diff: {mean_diff_sum / count:.6e}")

    if max_diff > 1e-6:
        print(f"\n*** CONFIRMED: Predictor produces DIFFERENT routing weights in train vs eval! ***")
        print(f"*** This is the train/eval asymmetry bug. ***")
    else:
        print(f"\n  Routing weights are identical — fastpath is not the issue.")

    # Also check: what's the magnitude of routing weights?
    print(f"\n=== Routing weight statistics (train mode) ===")
    for stream in ('q', 'r'):
        for i in [0, 5, 10]:
            if i < len(rw_train[stream]):
                w = rw_train[stream][i].detach().float()
                # How far from identity? Identity = [..., 0, 0, 1]
                identity_dev = w[..., :-1].abs().mean().item()
                last_val = w[..., -1].mean().item()
                print(f"  {stream} layer {i+1}: identity_dev={identity_dev:.4f} last_source_mean={last_val:.4f}")


if __name__ == "__main__":
    main()
