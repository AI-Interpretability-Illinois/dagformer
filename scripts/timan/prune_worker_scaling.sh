#!/bin/bash
# Pruning worker for the scaling checkout on timan1 (Table 4 baselines at 75M/150M, owner 2026-10-07). Waits until the
# pretraining queue on its GPU has exited (WAIT_PID; 0 = start now), then claims runs from the runlist (atomic lock
# dir) whose model checkpoint exists, skipping finished ones (summary.json). Runs whose checkpoint is not there yet stay
# pending: the list is passed over again every 10 min until nothing is pending. A run that fails twice is skipped.
#   nohup bash scripts/timan/prune_worker_scaling.sh <gpu> <wait_pid|0> <runlist> > /srv/local/xy51/logs/prune/worker_<gpu>_<k>.log 2>&1 &
# Runlist lines: <prune config> key=value ... (save_dir=... required). Stop: touch <runlist>.STOP
set -u
GPU=$1; WAIT_PID=$2; RL=$3; STOP=$RL.STOP; Q=/srv/local/xy51/logs/prune/queue_scaling.log
REPO=/srv/local/xy51/scaling/dagformer; PY=${PY:-/home/xy51/anaconda3/envs/modularity/bin/python}   # timan108: PY=/srv/local/xy51/envs/modularity310/bin/python
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/srv/local/xy51/hf_cache HF_HUB_CACHE=/srv/local/xy51/hf_cache/hub HF_DATASETS_CACHE=/srv/local/xy51/hf_cache/datasets
export HF_MODULES_CACHE=/srv/local/xy51/hf_cache/modules TRANSFORMERS_CACHE=/srv/local/xy51/hf_cache/hub HUGGINGFACE_HUB_CACHE=/srv/local/xy51/hf_cache/hub
export HF_TOKEN_PATH=/home/xy51/.cache/huggingface/token PYTHONPATH=$REPO TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE=offline WANDB_DIR=/srv/local/xy51/wandb CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8
cd "$REPO"; mkdir -p /srv/local/xy51/logs/prune
log() { echo "$(date '+%F %T') gpu$GPU/$$ $*" >> "$Q"; }
while [ "$WAIT_PID" != 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do [ -f "$STOP" ] && exit 0; sleep 120; done
sleep $(( RANDOM % 30 )); log "worker start ($RL)"
while :; do
  pending=0
  while read -r cfg rest; do
    [ -z "${cfg:-}" ] && continue; case "$cfg" in \#*) continue;; esac
    [ -f "$STOP" ] && { log STOP; exit 0; }
    args=(); sd=""; md=""
    for ov in $rest; do args+=(--override "$ov"); case "$ov" in save_dir=*) sd="${ov#save_dir=}";; model_dir=*) md="${ov#model_dir=}";; esac; done
    [ -z "$md" ] && md=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['model_dir'])")
    name=$(basename "$sd")
    [ -f "$sd/summary.json" ] && continue
    fails=$(cat "$sd.fails" 2>/dev/null || echo 0); [ "$fails" -ge 2 ] && continue
    # the checkpoint must exist and be at least 5 min old (a pretraining run on the other GPU may still be writing it)
    [ -n "$(find -L "$md/checkpoint.pt" -mmin +5 2>/dev/null)" ] || { pending=1; continue; }
    mkdir -p "$(dirname "$sd")"; mkdir "$sd.lock" 2>/dev/null || { pending=1; continue; }
    log "START $name"
    "$PY" scripts/prune_finetune.py --config "$cfg" "${args[@]}" > "/srv/local/xy51/logs/prune/$name.log" 2>&1; rc=$?
    log "END $name exit=$rc"
    if [ $rc -ne 0 ] || [ ! -f "$sd/summary.json" ]; then echo $((fails + 1)) > "$sd.fails"; rmdir "$sd.lock" 2>/dev/null; pending=1; fi
  done < "$RL"
  [ $pending = 0 ] && break
  sleep 600
done
log "worker done"
