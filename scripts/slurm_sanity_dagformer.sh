#!/bin/bash
#SBATCH --job-name=dagf-sanity
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dagf_sanity_%j.out
#SBATCH --error=logs/dagf_sanity_%j.err

# Quick sanity check: baseline reproduction + gradient flow + 5 training steps

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

python -u -c "
import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM
from src.model.olmo_graph import DAGFormerOLMo, create_all_ones_A
from src.model.predictor import StaticPredictor, SelfEmbedPredictor

device = torch.device('cuda')
torch.manual_seed(42)

# Create 300M model
cfg = Olmo2Config(
    hidden_size=1024, num_hidden_layers=12, num_attention_heads=16,
    num_key_value_heads=16, intermediate_size=4096, vocab_size=100352,
    tie_word_embeddings=True,
)
model = Olmo2ForCausalLM(cfg).to(device, dtype=torch.bfloat16)
dagformer = DAGFormerOLMo(model, input_norm='none', num_layers=12, num_heads=16)

B, S = 4, 256
input_ids = torch.randint(0, 100352, (B, S), device=device)
labels = torch.randint(0, 100352, (B, S), device=device)

print('=== Test 1: Baseline reproduction (A=1) ===')
with torch.no_grad():
    base_logits = model(input_ids=input_ids).logits
    base_nll = F.cross_entropy(base_logits.float().view(-1, 100352), labels.view(-1))

    A_ones = create_all_ones_A(B, num_nodes=192, num_heads=16).to(device)
    dag_logits = dagformer(input_ids, A_ones)
    dag_nll = F.cross_entropy(dag_logits.float().view(-1, 100352), labels.view(-1))

nll_diff = abs(base_nll.item() - dag_nll.item())
print(f'  baseline NLL={base_nll.item():.4f}, dagformer(A=1) NLL={dag_nll.item():.4f}, diff={nll_diff:.6f}')
assert nll_diff < 0.01, f'FAILED: diff={nll_diff}'
print('  PASSED')

print()
print('=== Test 2: Gradient flow (static predictor) ===')
model.train()
for p in model.parameters():
    p.requires_grad_(True)
pred = StaticPredictor(num_nodes=192, heads_per_layer=16, rank=32, init_logit=15.0).to(device)
A = pred(B, tau=5.0, mode='train')
logits = dagformer(input_ids, A)
loss = F.cross_entropy(logits.float().view(-1, 100352), labels.view(-1))
loss.backward()

model_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
pred_grads = sum(1 for p in pred.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f'  model params w/ grad: {model_grads}/{sum(1 for _ in model.parameters())}')
print(f'  predictor params w/ grad: {pred_grads}/{sum(1 for _ in pred.parameters())}')
assert pred_grads == sum(1 for _ in pred.parameters()), 'FAILED: predictor missing grads'
assert model_grads > 0, 'FAILED: model has no grads'
print('  PASSED')

print()
print('=== Test 3: Gradient flow (self-embed predictor) ===')
model.zero_grad()
pred_se = SelfEmbedPredictor(
    embed_dim=1024, hidden_dim=1024, num_nodes=192, heads_per_layer=16, rank=32, init_logit=15.0
).to(device)
emb = model.model.embed_tokens(input_ids)
A_se = pred_se(emb.detach(), tau=5.0, mode='train')
logits_se = dagformer(input_ids, A_se)
loss_se = F.cross_entropy(logits_se.float().view(-1, 100352), labels.view(-1))
loss_se.backward()
se_grads = sum(1 for p in pred_se.mlp.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f'  self-embed MLP params w/ grad: {se_grads}/{sum(1 for _ in pred_se.mlp.parameters())}')
assert se_grads == sum(1 for _ in pred_se.mlp.parameters()), 'FAILED'
print('  PASSED')

print()
print('=== Test 4: 5 training steps (static) ===')
model.zero_grad()
for p in pred.parameters():
    if p.grad is not None: p.grad.zero_()
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': 5e-4},
    {'params': pred.parameters(), 'lr': 3e-4},
], betas=(0.9, 0.95), weight_decay=0.1)

losses = []
for step in range(5):
    optimizer.zero_grad()
    A = pred(B, tau=5.0, mode='train')
    logits = dagformer(input_ids, A)
    loss = F.cross_entropy(logits.float().view(-1, 100352), labels.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(pred.parameters()), 1.0)
    optimizer.step()
    losses.append(loss.item())
    print(f'  step {step}: loss={loss.item():.4f}, mean_A={A.mean().item():.4f}')

print(f'  loss delta: {losses[0]:.4f} -> {losses[-1]:.4f} (diff={losses[0]-losses[-1]:.4f})')
print('  PASSED' if losses[-1] < losses[0] else '  WARNING: loss did not decrease')

print()
print(f'GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
print('ALL SANITY CHECKS DONE')
"
