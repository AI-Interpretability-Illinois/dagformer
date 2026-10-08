#!/bin/bash
# Sequential training queue for one timan1 GPU (no scheduler there). Waits until WAIT_FILE exists (the final checkpoint
# of the run currently on this GPU), then trains each item in order in the foreground, skipping items whose save_dir
# already has the final checkpoint. Launch detached:
#   nohup bash scripts/timan/run_queue_timan1.sh <gpu> <wait_file> <queue_file> > /srv/local/xy51/logs/queue_gpu<gpu>.log 2>&1 &
# Queue file lines: <config.yaml> <trainer.py> <logname>   ('#' comments allowed). Stop: touch <queue_file>.STOP
set -uo pipefail
GPU=$1; WAIT_FILE=$2; QUEUE=$3
REPO=/srv/local/xy51/scaling/dagformer
PY=${PY:-/home/xy51/anaconda3/envs/modularity/bin}
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/srv/local/xy51/hf_cache HF_HUB_CACHE=/srv/local/xy51/hf_cache/hub HF_DATASETS_CACHE=/srv/local/xy51/hf_cache/datasets
export HF_MODULES_CACHE=/srv/local/xy51/hf_cache/modules TRANSFORMERS_CACHE=/srv/local/xy51/hf_cache/hub HUGGINGFACE_HUB_CACHE=/srv/local/xy51/hf_cache/hub
export HF_TOKEN_PATH=/home/xy51/.cache/huggingface/token
export PYTHONPATH=$REPO WANDB_MODE=offline WANDB_DIR=/srv/local/xy51/wandb OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$GPU
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }
log "queue $QUEUE on GPU $GPU; waiting for $WAIT_FILE"
while [ ! -f "$WAIT_FILE" ]; do [ -f "$QUEUE.STOP" ] && { log "STOP"; exit 0; }; sleep 120; done
sleep 60   # let the previous trainer exit and free its memory
while read -r cfg trainer name; do
    [ -z "${cfg:-}" ] && continue; case "$cfg" in \#*) continue;; esac
    [ -f "$QUEUE.STOP" ] && { log "STOP"; break; }
    save=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['save_dir'])")
    total=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['total_steps'])")
    [ -f "$save/checkpoint_step${total}.pt" ] && { log "skip (done): $name"; continue; }
    log "start $name ($cfg, $trainer) -> $save"
    echo "$(date '+%F %T') started $name gpus=$GPU cfg=$cfg trainer=$trainer (queue)" >> /srv/local/xy51/logs/extras_launches.log
    "$PY/torchrun" --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 "$trainer" --config "$cfg" \
        > "/srv/local/xy51/logs/$name.log" 2>&1
    log "end $name rc=$? final=$( [ -f "$save/checkpoint_step${total}.pt" ] && echo yes || echo no)"
done < "$QUEUE"
log "queue done"
