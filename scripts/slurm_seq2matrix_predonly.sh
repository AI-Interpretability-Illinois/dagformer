#!/bin/bash
#SBATCH --job-name=s2m-predonly
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=30:00:00
#SBATCH --output=logs/seq2matrix_predonly_%j.out
#SBATCH --error=logs/seq2matrix_predonly_%j.err
#SBATCH --signal=B:USR1@120

# SeqToMatrix predictor-only: random frozen LLM, train predictor to find optimal topology

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

echo "Config: configs/pretrain_300m_seq2matrix_predonly.yaml"
echo "Starting at $(date)"

torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    scripts/pretrain_dagformer.py \
    --config configs/pretrain_300m_seq2matrix_predonly.yaml
