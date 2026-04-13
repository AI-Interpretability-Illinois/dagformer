"""Use PyTorch's official torch.autograd.gradcheck to verify FourWay backward.
This is the gold standard — if gradcheck fails, backward is definitively wrong.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Very small model for gradcheck speed
    L, H, D, V = 3, 2, 64, 100
    B, T = 1, 8

    mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                     num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                     tie_word_embeddings=True, max_position_embeddings=64)
    base = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)
    fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.float64)
    fw.eval()

    input_ids = torch.randint(0, V, (B, T), device=device)
    labels = torch.randint(0, V, (B, T), device=device)

    # Build α as leaf tensors
    alphas = []
    rw = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, L):
        n_src = l + 1
        for stream in ("q", "k", "v"):
            α = torch.randn(B, T, H, n_src, device=device, dtype=torch.float64,
                            requires_grad=True) * 0.1
            # Add identity baseline
            with torch.no_grad():
                α[..., -1] += 1.0
            rw[stream].append(α)
            alphas.append(α)
        α_r = torch.randn(B, T, n_src, device=device, dtype=torch.float64,
                          requires_grad=True) * 0.1
        with torch.no_grad():
            α_r[..., -1] += 1.0
        rw["r"].append(α_r)
        alphas.append(α_r)

    # Define a function that takes α tensors and returns loss
    def loss_fn(*alpha_tensors):
        idx = 0
        rw_local = {"q": [], "k": [], "v": [], "r": []}
        for l in range(1, L):
            rw_local["q"].append(alpha_tensors[idx]); idx += 1
            rw_local["k"].append(alpha_tensors[idx]); idx += 1
            rw_local["v"].append(alpha_tensors[idx]); idx += 1
            rw_local["r"].append(alpha_tensors[idx]); idx += 1
        logits = fw(input_ids, rw_local)
        return F.cross_entropy(logits.view(-1, V), labels.view(-1))

    # Run PyTorch's official gradcheck
    print(f"Model: {L}L {H}H {D}D, vocab={V}, batch={B}, seq={T}")
    print(f"Checking {len(alphas)} α tensors...")
    print(f"Total α elements: {sum(a.numel() for a in alphas)}")
    print()

    try:
        result = gradcheck(
            loss_fn,
            tuple(alphas),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        )
        print(f"torch.autograd.gradcheck: {'PASS ✓' if result else 'FAIL ✗'}")
    except Exception as e:
        print(f"torch.autograd.gradcheck: FAIL ✗")
        print(f"Error: {e}")

    # Also test individual streams separately
    print("\nPer-stream gradcheck:")
    for stream_idx, stream_name in enumerate(["q", "k", "v", "r"]):
        for l in range(1, L):
            α = rw[stream_name][l - 1]

            def single_loss(a):
                rw_copy = {s: list(rw[s]) for s in rw}
                rw_copy[stream_name][l - 1] = a
                logits = fw(input_ids, rw_copy)
                return F.cross_entropy(logits.view(-1, V), labels.view(-1))

            try:
                ok = gradcheck(single_loss, (α,), eps=1e-6, atol=1e-4, rtol=1e-3,
                               raise_exception=True)
                status = "PASS ✓" if ok else "FAIL ✗"
            except Exception as e:
                status = f"FAIL ✗ ({e})"
            print(f"  {stream_name}[layer {l}]: {status}")


if __name__ == "__main__":
    main()
