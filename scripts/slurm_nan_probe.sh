#!/bin/bash
#SBATCH --job-name=nan-probe
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=0:40:00
#SBATCH --output=logs/nan_probe_%j.out
#SBATCH --error=logs/nan_probe_%j.err

# Single-GPU diagnostic for the 600M FourWay NaN (whole 13h run was NaN from
# step 0's backward). Only architectural delta vs the working 300M run is
# use_v_norm=true, so this builds the 600M model with v_norm ON and OFF and
# reports where NaN first appears (forward activations vs backward grads).

set -uo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

/usr/bin/python3.9 - << 'PY'
import torch
from scripts.pretrain_dagformer import (
    DAGFormerPretrainConfig, predict_fourway_routing,
)
from src.model.olmo_graph import FourWayDAGFormer
from src.model.predictor import FourWayPredictor
from scripts.pretrain_dagformer import create_model

dev = "cuda"

def probe(use_v_norm: bool) -> None:
    cfg = DAGFormerPretrainConfig.from_yaml("configs/fourway_600m_chinchilla.yaml")
    cfg.use_v_norm = use_v_norm
    print(f"\n===== use_v_norm={use_v_norm}  (h={cfg.hidden_size}, L={cfg.num_hidden_layers}, "
          f"head_dim={cfg.hidden_size // cfg.num_attention_heads}) =====", flush=True)
    base = create_model(cfg).to(dev).to(torch.bfloat16)
    model = FourWayDAGFormer(
        model=base,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        use_local_correction=(cfg.routing_mode == "fourway_corrected"),
        correction_hidden=cfg.correction_hidden,
        use_triton_kernel=cfg.use_triton_kernel,
        use_v_norm=cfg.use_v_norm,
        correction_pool=cfg.correction_pool,
    ).to(dev).to(torch.bfloat16)
    pred = FourWayPredictor(
        vocab_size=cfg.vocab_size,
        encoder_dim=cfg.predictor_encoder_dim,
        encoder_layers=cfg.predictor_encoder_layers,
        encoder_heads=cfg.predictor_encoder_heads,
        max_seq_len=cfg.predictor_max_seq_len,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        hidden_dim=cfg.fourway_hidden,
        causal=cfg.predictor_causal,
        dropout=cfg.predictor_dropout,
    ).to(dev).to(torch.bfloat16)

    ids = torch.randint(0, cfg.vocab_size, (2, 1024), device=dev)
    rw = predict_fourway_routing(pred, ids, cfg, 0)

    def finite(x):
        return bool(torch.isfinite(x).all().item()) if torch.is_tensor(x) else all(
            finite(y) for y in x) if isinstance(x, (list, tuple)) else True
    if isinstance(rw, dict):
        for k, v in rw.items():
            print(f"  routing[{k}] finite={finite(v)}", flush=True)

    out = model(ids, rw)
    logits = out if torch.is_tensor(out) else out.logits
    print(f"  logits finite={finite(logits)} absmax={logits.float().abs().max().item():.3f}", flush=True)

    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), ids.reshape(-1))
    print(f"  loss={loss.item():.4f} finite={bool(torch.isfinite(loss).item())}", flush=True)
    loss.backward()

    bad = [n for n, p in list(model.named_parameters()) + list(pred.named_parameters())
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    tot = sum(1 for _, p in list(model.named_parameters()) + list(pred.named_parameters())
              if p.grad is not None)
    print(f"  params with non-finite grad: {len(bad)}/{tot}", flush=True)
    for n in bad[:12]:
        print(f"    {n}", flush=True)
    del base, model, pred
    torch.cuda.empty_cache()

probe(False)   # like the working 300M run
probe(True)    # the NaN 600M config
PY
