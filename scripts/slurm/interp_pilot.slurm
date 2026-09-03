#!/bin/bash
#SBATCH --job-name=interp_pilot
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=logs/interp_pilot_%j.out
#SBATCH --error=logs/interp_pilot_%j.err

set -u
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

# Tokenizer lives in the legacy TRANSFORMERS_CACHE layout under hf_cache/transformers.
# Override ALL cache vars: the login env points HF_HUB_CACHE/TRANSFORMERS_CACHE at
# /work (full, and missing OLMo-2) and sbatch inherits it.
export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CFG=configs/fourway_300m_dagformer_mmap.yaml
CKPT_DIR=checkpoints/fourway_300m_dagformer_mmap
OUT=experiments/results/interp

echo "=== STEP 1/3: alpha dump ($(date)) ==="
python3 scripts/interp_dump_alpha.py \
    --config "$CFG" --ckpt-dir "$CKPT_DIR" --out "$OUT" --main-steps 9000
rc=$?; if [ $rc -ne 0 ]; then echo "DUMP FAILED rc=$rc"; exit $rc; fi

echo "=== STEP 2/3: swap eval ($(date)) ==="
python3 scripts/interp_swap_eval.py \
    --config "$CFG" --ckpt "$CKPT_DIR/checkpoint_step9000.pt" \
    --dense-config configs/pretrain_300m_baseline_mmap.yaml \
    --dense-ckpt checkpoints/pretrain_300m_baseline_mmap/checkpoint_step9000.pt \
    --out "$OUT"
rc=$?; if [ $rc -ne 0 ]; then echo "SWAP FAILED rc=$rc"; exit $rc; fi

echo "=== STEP 3/3: probes ($(date)) ==="
python3 scripts/interp_probe.py \
    --config "$CFG" --dump-dir "$OUT/alpha_dumps" \
    --swap-nll "$OUT/per_token_nll_step9000.pt" --main-step 9000 --out "$OUT"
rc=$?; if [ $rc -ne 0 ]; then echo "PROBE FAILED rc=$rc"; exit $rc; fi

echo "=== INTERP PILOT COMPLETE ($(date)) ==="
