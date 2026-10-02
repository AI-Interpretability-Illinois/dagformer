#!/bin/bash
#SBATCH --job-name=chat_eval_gen
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --reservation=sup-30781
#SBATCH --nodelist=gpua047
#SBATCH --output=logs/chat_eval_gen_%j.out
# Judged chat eval, step 1 (answers of every finished SFT export to the MT-Bench first turns).
# Resumable: models with a complete answers file are skipped. Step 2 (the judge) runs on timan1.
cd /u/xiaocong/dagformer
source ~/.bashrc; conda activate modularity 2>/dev/null || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python3 scripts/chat_eval_generate.py "$@"
echo "exit rc=$? at $(date)"
