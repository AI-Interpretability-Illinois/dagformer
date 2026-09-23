#!/bin/bash
#SBATCH --job-name=locality
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200g
#SBATCH --time=08:00:00
#SBATCH --output=logs/locality_%x_%j.out
#SBATCH --error=logs/locality_%x_%j.err
#SBATCH --signal=B:USR1@180
# Experiment 1 (router locality), one arm per job on 4 GPUs:
#   ARM=global|local|both|per_layer|dense sbatch --dependency=afterok:<prep> scripts/slurm/locality_arm.slurm
# Resumes from the latest checkpoint in save_dir on resubmission.
set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
mkdir -p logs
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
ARM="${ARM:?set ARM=global|local|both|per_layer|dense}"
CONFIG=configs/locality/75m_${ARM}.yaml
SCRIPT=scripts/pretrain_dagformer.py; [ "$ARM" = dense ] && SCRIPT=scripts/pretrain_baseline.py
trap 'echo "SIGUSR1: saving; resubmitting"; sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL "$0"' SIGUSR1
echo "arm=$ARM config=$CONFIG job=$SLURM_JOB_ID node=$(hostname) start=$(date)"
torchrun --nproc_per_node=4 --master_port=$((29000 + RANDOM % 1000)) $SCRIPT --config "$CONFIG" &
wait $!
rc=$?; echo "exit rc=$rc at $(date)"; exit $rc
