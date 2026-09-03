#!/bin/bash
# Submit lm-eval jobs for all available checkpoints (baseline + dagformer + muddformer)
set -eo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

submit() {
    local cfg=$1; local ckpt=$2; local name=$3; local tasks=${4:-default}; local batch=${5:-8}
    if [ ! -f "$ckpt" ]; then echo "SKIP $name: ckpt missing ($ckpt)"; return; fi
    local id=$(sbatch --parsable scripts/slurm_eval_lm_harness_preempt.sh "$cfg" "$ckpt" "$name" "$tasks" "$batch")
    echo "Submitted $id: $name"
}

# Baselines from HF
submit hf_checkpoints/baseline-75m/config.yaml  hf_checkpoints/baseline-75m/ckpts/checkpoint_step3000.pt  75m_baseline_step3000  default 16
submit hf_checkpoints/baseline-150m/config.yaml hf_checkpoints/baseline-150m/ckpts/checkpoint_step6000.pt 150m_baseline_step6000 default 16
submit hf_checkpoints/baseline-300m/config.yaml hf_checkpoints/baseline-300m/ckpts/checkpoint_step12000.pt 300m_baseline_chinchilla_step12000 default 8

# DAGFormer (smaller batch due to FourWay memory overhead, esp. for wikitext rolling LL)
submit hf_checkpoints/dagformer-75m/config.yaml  hf_checkpoints/dagformer-75m/ckpts/checkpoint_step3000.pt   75m_dagformer_step3000  default 2
submit hf_checkpoints/dagformer-150m/config.yaml hf_checkpoints/dagformer-150m/ckpts/checkpoint_step6000.pt 150m_dagformer_step6000 default 2
submit hf_checkpoints/dagformer-300m/config.yaml hf_checkpoints/dagformer-300m/ckpts/checkpoint_step12000.pt 300m_dagformer_chinchilla_step12000 default 1
submit hf_checkpoints/dagformer-1b/config.yaml   hf_checkpoints/dagformer-1b/ckpts/checkpoint_step10000.pt   1b_dagformer_step10000 default 1

# MUDDFormer skipped: no loader in our codebase. Ckpts on HF, but reconstruction needs
# MultiwayDynamicDenseBlock — port from docs/reference_papers/muddformer/ first.

# 1B baseline (xiaocong) — only step 9500 available
submit configs/pretrain_1b_baseline.yaml /projects/bfqt/xiaocong/dagformer_ckpts/pretrain_1b_baseline/checkpoint_step9500.pt 1b_baseline_step9500 default 4
