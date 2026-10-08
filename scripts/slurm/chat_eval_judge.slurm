#!/bin/bash
#SBATCH --job-name=chat_eval_judge
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --reservation=sup-30781
#SBATCH --nodelist=gpua047
#SBATCH --output=logs/chat_eval_judge_%j.out
# Judged chat eval, step 2 on Delta: vLLM judge (Qwen3-Coder-30B-A3B-Instruct, bf16, TP=2) over the
# MT-Bench answers in experiments/chat_eval/answers. Resumable: already-judged keys are skipped.
# Env: conda env "vllm" (pip install vllm); weights: /projects/biro/xiaocong/models (the /work biro project quota is at its 550 GB hard limit).
cd /u/xiaocong/dagformer
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
JUDGE=${JUDGE:-/projects/biro/xiaocong/models/Qwen3-Coder-30B-A3B-Instruct}
~/anaconda3/envs/vllm/bin/python scripts/chat_eval_judge.py --root experiments/chat_eval --judge "$JUDGE" \
    --tp 2 --max-tokens 2048 --gpu-mem 0.90 "$@"
echo "exit rc=$? at $(date)"
