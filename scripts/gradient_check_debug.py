"""Focused gradient check with detailed value printing.
Checks just one α tensor, prints actual values to understand the failure.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    L, H, D, V = 4, 4, 128, 1000
    B, T = 2, 32

    mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                     num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                     tie_word_embeddings=True, max_position_embeddings=256)
    base = Olmo2ForCausalLM(mc).to(device, dtype=torch.float32)
    fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.float32)
    fw.eval()

    input_ids = torch.randint(0, V, (B, T), device=device)
    labels = torch.randint(0, V, (B, T), device=device)

    # Build α for layer 1 (simplest: 2 sources)
    n_src = 2
    α_q = torch.tensor([[[[0.3, 0.7]]]], device=device).expand(B, T, H, n_src).clone()
    α_q.requires_grad_(True)
    α_k = torch.tensor([[[[0.2, 0.8]]]], device=device).expand(B, T, H, n_src).clone()
    α_k.requires_grad_(True)
    α_v = torch.tensor([[[[0.4, 0.6]]]], device=device).expand(B, T, H, n_src).clone()
    α_v.requires_grad_(True)
    α_r = torch.tensor([[[0.1, 0.9]]], device=device).expand(B, T, n_src).clone()
    α_r.requires_grad_(True)

    # Identity for remaining layers (so we isolate layer 1's gradient)
    rw = {"q": [α_q], "k": [α_k], "v": [α_v], "r": [α_r]}
    for l in range(2, L):
        ns = l + 1
        for s in ("q", "k", "v"):
            a = torch.zeros(B, T, H, ns, device=device)
            a[..., -1] = 1.0
            a.requires_grad_(True)
            rw[s].append(a)
        ar = torch.zeros(B, T, ns, device=device)
        ar[..., -1] = 1.0
        ar.requires_grad_(True)
        rw["r"].append(ar)

    # Forward + backward
    logits = fw(input_ids, rw)
    loss = F.cross_entropy(logits.float().view(-1, V), labels.view(-1))
    loss.backward()

    print("=" * 60)
    print("GRADIENT CHECK — Layer 1 α_q (shape [B,T,H,2])")
    print("=" * 60)
    print(f"Loss: {loss.item():.6f}")
    print(f"α_q.grad is None: {α_q.grad is None}")
    if α_q.grad is not None:
        print(f"α_q.grad shape: {α_q.grad.shape}")
        print(f"α_q.grad mean: {α_q.grad.mean().item():.6e}")
        print(f"α_q.grad abs max: {α_q.grad.abs().max().item():.6e}")
        print(f"α_q.grad abs min: {α_q.grad.abs().min().item():.6e}")
        print(f"α_q.grad std: {α_q.grad.std().item():.6e}")
        print(f"α_q.grad nonzero: {(α_q.grad != 0).sum().item()} / {α_q.grad.numel()}")

    # Finite difference on first few elements of α_q
    eps = 1e-4
    print(f"\nFinite difference (eps={eps}):")
    print(f"{'idx':>5} | {'autograd':>12} | {'finite_diff':>12} | {'abs_err':>12} | {'rel_err':>12}")
    print("-" * 70)

    for flat_idx in range(min(20, α_q.numel())):
        ag = α_q.grad.view(-1)[flat_idx].item()

        with torch.no_grad():
            orig = α_q.data.view(-1)[flat_idx].item()
            α_q.data.view(-1)[flat_idx] = orig + eps
        logits_p = fw(input_ids, rw)
        loss_p = F.cross_entropy(logits_p.float().view(-1, V), labels.view(-1)).item()

        with torch.no_grad():
            α_q.data.view(-1)[flat_idx] = orig - eps
        logits_m = fw(input_ids, rw)
        loss_m = F.cross_entropy(logits_m.float().view(-1, V), labels.view(-1)).item()

        with torch.no_grad():
            α_q.data.view(-1)[flat_idx] = orig

        fd = (loss_p - loss_m) / (2 * eps)
        abs_err = abs(ag - fd)
        rel_err = abs_err / (max(abs(ag), abs(fd), 1e-5))

        print(f"{flat_idx:5d} | {ag:12.6e} | {fd:12.6e} | {abs_err:12.6e} | {rel_err:12.6e}")

    # Also check α_r, α_k, α_v
    for name, α in [("α_k", α_k), ("α_v", α_v), ("α_r", α_r)]:
        print(f"\n{name}: grad abs max={α.grad.abs().max().item():.6e}, "
              f"nonzero={( α.grad != 0).sum().item()}/{α.grad.numel()}")
        # Check first element
        ag = α.grad.view(-1)[0].item()
        with torch.no_grad():
            orig = α.data.view(-1)[0].item()
            α.data.view(-1)[0] = orig + eps
        lp = F.cross_entropy(fw(input_ids, rw).float().view(-1, V), labels.view(-1)).item()
        with torch.no_grad():
            α.data.view(-1)[0] = orig - eps
        lm = F.cross_entropy(fw(input_ids, rw).float().view(-1, V), labels.view(-1)).item()
        with torch.no_grad():
            α.data.view(-1)[0] = orig
        fd = (lp - lm) / (2 * eps)
        print(f"  [0] autograd={ag:.6e} fd={fd:.6e} abs_err={abs(ag-fd):.6e}")


if __name__ == "__main__":
    main()
