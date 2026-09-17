"""Test: does SDPA produce different results with grad vs no_grad on TRAINED weights?

Mini-trains a FourWay model for a few steps, then compares NLL
with torch.no_grad() vs torch.enable_grad() on the SAME data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM
import sys
sys.path.insert(0, '.')

from src.model.olmo_graph import FourWayDAGFormer
from src.model.predictor import FourWayPredictor
from scripts.pretrain_dagformer import replace_olmo_rmsnorm


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, T, V = 4, 256, 10000  # small for speed
    H, L = 16, 12

    cfg = Olmo2Config(
        hidden_size=1024, num_hidden_layers=L, num_attention_heads=H,
        intermediate_size=4096, vocab_size=V, tie_word_embeddings=True,
        max_position_embeddings=4096,
    )
    base = Olmo2ForCausalLM(cfg).to(device, dtype=dtype)
    replace_olmo_rmsnorm(base)

    fourway = FourWayDAGFormer(
        model=base, num_layers=L, num_heads=H,
        use_local_correction=True, correction_hidden=128,
    ).to(device)

    predictor = FourWayPredictor(
        vocab_size=V, encoder_dim=256, encoder_layers=2, encoder_heads=4,
        max_seq_len=4096, num_layers=L, num_heads=H, hidden_dim=512, causal=True,
    ).to(device, dtype=dtype)

    # Mini-train for 20 steps to get non-trivial weights
    optimizer = torch.optim.AdamW([
        {"params": base.parameters(), "lr": 1e-3},
        {"params": predictor.parameters(), "lr": 1e-3},
        {"params": fourway.get_routing_parameters(), "lr": 1e-3},
    ])

    print("Mini-training 20 steps...")
    fourway.train(); predictor.train()
    for step in range(20):
        ids = torch.randint(0, V, (B, T), device=device)
        labels = torch.randint(0, V, (B, T), device=device)

        rw = predictor(ids)
        logits = fourway(ids, rw)
        loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")

    # Now test: same data, 4 modes
    test_ids = torch.randint(0, V, (B, T), device=device)
    test_labels = torch.randint(0, V, (B, T), device=device)

    print("\n=== Comparing NLL across modes (TRAINED weights) ===")

    # Mode 1: train + grad (same as training)
    fourway.train(); predictor.train()
    rw1 = predictor(test_ids)
    logits1 = fourway(test_ids, rw1)
    nll1 = F.cross_entropy(logits1.view(-1, V), test_labels.view(-1))
    print(f"  TRAIN + grad:    NLL={nll1.item():.6f}")

    # Mode 2: eval + no_grad (same as eval loop)
    fourway.eval(); predictor.eval()
    with torch.no_grad():
        rw2 = predictor(test_ids)
        logits2 = fourway(test_ids, rw2)
        nll2 = F.cross_entropy(logits2.view(-1, V), test_labels.view(-1))
    print(f"  EVAL + no_grad:  NLL={nll2.item():.6f}")

    # Mode 3: eval + grad (isolate eval mode from no_grad)
    fourway.eval(); predictor.eval()
    rw3 = predictor(test_ids)
    logits3 = fourway(test_ids, rw3)
    nll3 = F.cross_entropy(logits3.view(-1, V), test_labels.view(-1))
    print(f"  EVAL + grad:     NLL={nll3.item():.6f}")

    # Mode 4: train + no_grad (isolate train mode from grad)
    fourway.train(); predictor.train()
    with torch.no_grad():
        rw4 = predictor(test_ids)
        logits4 = fourway(test_ids, rw4)
        nll4 = F.cross_entropy(logits4.view(-1, V), test_labels.view(-1))
    print(f"  TRAIN + no_grad: NLL={nll4.item():.6f}")

    # Mode 5: force MATH backend (no fused kernels)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    fourway.eval(); predictor.eval()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        rw5 = predictor(test_ids)
        logits5 = fourway(test_ids, rw5)
        nll5 = F.cross_entropy(logits5.view(-1, V), test_labels.view(-1))
    print(f"  EVAL + MATH:     NLL={nll5.item():.6f}")

    print(f"\n  Logits diff (train+grad vs eval+nograd): {(logits1.detach() - logits2).abs().max().item():.6e}")
    print(f"  Logits diff (train+grad vs eval+grad):   {(logits1.detach() - logits3.detach()).abs().max().item():.6e}")
    print(f"  Logits diff (train+grad vs train+nograd): {(logits1.detach() - logits4).abs().max().item():.6e}")
    print(f"  Logits diff (eval+nograd vs eval+MATH):   {(logits2 - logits5).abs().max().item():.6e}")


if __name__ == "__main__":
    main()
