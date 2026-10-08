#!/bin/bash
# Launch one pretraining run on timan108 (no scheduler) in the background, same environment as the timan1
# runs of the 75M/150M pairs (run_pretrain_75m_baseline_dolma12b.sh), from the rsync'd checkout.
#   bash scripts/timan/run_extras_timan108.sh <config.yaml> <gpu-list> <trainer.py> <logname>
#   e.g. bash scripts/timan/run_extras_timan108.sh configs/extras/75m_dense_w656_timan108_dolma12b.yaml 0,1,2 \
#            scripts/pretrain_baseline.py extras_75m_dense_w656
# Logs: /srv/local/xy51/logs/<logname>.log ; pid file next to it.
set -euo pipefail
CFG=$1; GPUS=$2; TRAINER=${3:-scripts/pretrain_baseline.py}; NAME=$4
REPO=/srv/local/xy51/scaling/dagformer
PY=${PY:-/srv/local/xy51/envs/modularity310/bin}   # on timan1 pass PY=/home/xy51/anaconda3/envs/modularity/bin
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/srv/local/xy51/hf_cache HF_HUB_CACHE=/srv/local/xy51/hf_cache/hub HF_DATASETS_CACHE=/srv/local/xy51/hf_cache/datasets
export HF_MODULES_CACHE=/srv/local/xy51/hf_cache/modules TRANSFORMERS_CACHE=/srv/local/xy51/hf_cache/hub HUGGINGFACE_HUB_CACHE=/srv/local/xy51/hf_cache/hub
export HF_TOKEN_PATH=/home/xy51/.cache/huggingface/token
export PYTHONPATH=$REPO WANDB_MODE=offline WANDB_DIR=/srv/local/xy51/wandb OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p "$WANDB_DIR" /srv/local/xy51/logs
cd "$REPO"
nohup "$PY/torchrun" --nproc_per_node="$NPROC" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    "$TRAINER" --config "$CFG" > "/srv/local/xy51/logs/$NAME.log" 2>&1 &
echo $! > "/srv/local/xy51/logs/$NAME.pid"
echo "$(date '+%F %T') started $NAME pid $! gpus=$GPUS cfg=$CFG trainer=$TRAINER" | tee -a /srv/local/xy51/logs/extras_launches.log
