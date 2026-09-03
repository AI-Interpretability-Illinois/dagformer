#!/bin/bash
#SBATCH --job-name=eval-openbl
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_openbl_%j.out
#SBATCH --error=logs/eval_openbl_%j.err

# Evaluate open-weight dense LM suites (Pythia, Cerebras-GPT) on wikitext to get
# tokenizer-independent bits-per-byte, directly comparable to our own points'
# wikitext BPB, for the FLOPs x loss Pareto frontier. Standard HF architectures
# (GPTNeoX / GPT2) load straight into lm-eval --model hf. Downloads (~15GB total)
# go to HF_HOME on first use. Robust to a single model failing (continues).

set -uo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/results/lmeval/open_baselines

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1

OUT=experiments/results/lmeval/open_baselines
# Within/around our 75M-1B range; 1.4B / 1.3B are the just-above anchors.
MODELS=(
  EleutherAI/pythia-70m
  EleutherAI/pythia-160m
  EleutherAI/pythia-410m
  EleutherAI/pythia-1b
  EleutherAI/pythia-1.4b
  cerebras/Cerebras-GPT-111M
  cerebras/Cerebras-GPT-256M
  cerebras/Cerebras-GPT-590M
  cerebras/Cerebras-GPT-1.3B
)

echo "Started: $(date)   Job: $SLURM_JOB_ID"
for m in "${MODELS[@]}"; do
  safe=$(echo "$m" | tr '/' '_')
  echo "================ evaluating $m ================"
  /usr/bin/python3.9 -m lm_eval --model hf \
      --model_args "pretrained=$m,dtype=bfloat16" \
      --tasks wikitext --batch_size 4 \
      --output_path "$OUT/$safe" \
    && echo "OK: $m" \
    || echo "FAILED: $m"
  echo "---- done $m at $(date) ----"
done
echo "ALL EVALS DONE $(date)"
df -h /work | tail -1
