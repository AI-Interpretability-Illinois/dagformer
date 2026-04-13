import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/projects/bfqt/users/yurenh2/ml-projects/DAGFormer")
from transformers import Olmo2Config, Olmo2ForCausalLM
from src.model.olmo_graph import FourWayDAGFormer

device = torch.device("cuda")
torch.manual_seed(42)
L, H, D, V, B, T = 4, 4, 128, 1000, 2, 32

mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=256)
base = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)  # FLOAT64
fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.float64)
fw.eval()

ids = torch.randint(0, V, (B, T), device=device)
labs = torch.randint(0, V, (B, T), device=device)

# α for layer 1 in float64
α_q = torch.tensor([[[[0.3, 0.7]]]], device=device, dtype=torch.float64).expand(B,T,H,2).clone()
α_q.requires_grad_(True)
α_k = torch.tensor([[[[0.2, 0.8]]]], device=device, dtype=torch.float64).expand(B,T,H,2).clone()
α_k.requires_grad_(True)
α_v = torch.tensor([[[[0.4, 0.6]]]], device=device, dtype=torch.float64).expand(B,T,H,2).clone()
α_v.requires_grad_(True)
α_r = torch.tensor([[[0.1, 0.9]]], device=device, dtype=torch.float64).expand(B,T,2).clone()
α_r.requires_grad_(True)

rw = {"q": [α_q], "k": [α_k], "v": [α_v], "r": [α_r]}
for l in range(2, L):
    ns = l + 1
    for s in ("q","k","v"):
        a = torch.zeros(B,T,H,ns,device=device,dtype=torch.float64); a[...,-1]=1; a.requires_grad_(True)
        rw[s].append(a)
    ar = torch.zeros(B,T,ns,device=device,dtype=torch.float64); ar[...,-1]=1; ar.requires_grad_(True)
    rw["r"].append(ar)

loss = F.cross_entropy(fw(ids, rw).view(-1, V), labs.view(-1))
loss.backward()

eps = 1e-6  # smaller eps OK in float64
print("FLOAT64 GRADIENT CHECK (eps=1e-6)")
print(f"{'idx':>5} | {'autograd':>14} | {'finite_diff':>14} | {'rel_err':>12}")
print("-" * 65)

for flat_idx in range(min(20, α_q.numel())):
    ag = α_q.grad.view(-1)[flat_idx].item()
    with torch.no_grad():
        orig = α_q.data.view(-1)[flat_idx].item()
        α_q.data.view(-1)[flat_idx] = orig + eps
    lp = F.cross_entropy(fw(ids, rw).view(-1, V), labs.view(-1)).item()
    with torch.no_grad():
        α_q.data.view(-1)[flat_idx] = orig - eps
    lm = F.cross_entropy(fw(ids, rw).view(-1, V), labs.view(-1)).item()
    with torch.no_grad():
        α_q.data.view(-1)[flat_idx] = orig
    fd = (lp - lm) / (2 * eps)
    rel = abs(ag - fd) / (max(abs(ag), abs(fd), 1e-10))
    print(f"{flat_idx:5d} | {ag:14.8e} | {fd:14.8e} | {rel:12.6e}")

print("\nα_v[0]:")
ag_v = α_v.grad.view(-1)[0].item()
with torch.no_grad():
    o = α_v.data.view(-1)[0].item(); α_v.data.view(-1)[0] = o + eps
lp = F.cross_entropy(fw(ids, rw).view(-1, V), labs.view(-1)).item()
with torch.no_grad():
    α_v.data.view(-1)[0] = o - eps
lm = F.cross_entropy(fw(ids, rw).view(-1, V), labs.view(-1)).item()
with torch.no_grad():
    α_v.data.view(-1)[0] = o
fd_v = (lp - lm) / (2 * eps)
print(f"  autograd={ag_v:.8e} fd={fd_v:.8e} rel_err={abs(ag_v-fd_v)/max(abs(ag_v),abs(fd_v),1e-10):.6e}")
