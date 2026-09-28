#!/bin/bash
#SBATCH --job-name=resv_train
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodelist=gpua046
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/resv_train_%j.out
#SBATCH --error=logs/resv_train_%j.err
#SBATCH --signal=B:USR1@900

# Sequential training queue on ONE reserved node, kept busy across the 48h
# per-job limit:
#   * at start, a successor job is queued (afterany on this one) so the next
#     48h slice begins the moment this one ends;
#   * 15 min before the limit SIGUSR1 arrives, the running trainer is told to
#     checkpoint (it traps USR1, saves, exits) and this job exits; the successor
#     resumes the same run from the latest checkpoint in its save_dir;
#   * items in the queue file are skipped once their save_dir holds DONE, so
#     the chain ends by itself when everything has finished.
# Tokenization of the 21B-token Dolma corpus runs in the background on the
# node's CPUs while the GPUs work through the 75M items; the 1B items wait
# for the corpus index.
#
#   sbatch --begin=2026-09-26T20:00:00 scripts/slurm/reserved_train.slurm            # 1B chain on gpua046
#   sbatch --nodelist=gpua047 --job-name=resv_train_300m \
#          --export=ALL,QUEUE=experiments/pretrain_300m/queue.txt,TOKENIZE=0 scripts/slurm/reserved_train.slurm
#   Queue: experiments/pretrain_1b/queue.txt   (one item per line:
#          config=<yaml> [script=<trainer.py>] [wait=<path>] [eval=<eval_cache.pt>] [fallback=<yaml>] [skipif=<path>]
#          task=<shell command> done=<marker path> [wait=<path>]     a non-training step, e.g. an eval)
#   Stop the chain: touch experiments/pretrain_1b/STOP  (finishes the current item's save, no successor)
set -uo pipefail
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
mkdir -p logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}" OMP_NUM_THREADS=4
export HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ENABLE_HF_TRANSFER=0 HF_DATASETS_TIMEOUT=300 HF_XET_NATIVE=0 HF_HUB_DISABLE_XET=1
QUEUE=${QUEUE:-experiments/pretrain_1b/queue.txt}
STOP=${STOP:-$(dirname "$QUEUE")/STOP}
DATA21=/work/hdd/bfqt/xiaocong/dagformer_pruning/data/dolma_v1_7_21b
N_TOK_WORKERS=${N_TOK_WORKERS:-8}
TOKENIZE=${TOKENIZE:-1}          # 0: another job builds the corpus; never touch its worker dirs
log() { echo "[$(date '+%F %T')] $*"; }

log "job $SLURM_JOB_ID on $(hostname), GPUs=${CUDA_VISIBLE_DEVICES:-?}, cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- successor: keep the node busy across the 48h limit -------------------
if [ ! -f "$STOP" ]; then
    # same node, same name, same env (QUEUE / STOP / TOKENIZE) as this job
    nxt=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID --nodelist="$(hostname -s)" \
          --job-name="$SLURM_JOB_NAME" --export=ALL,QUEUE="$QUEUE",STOP="$STOP",TOKENIZE="$TOKENIZE" "$0") \
        && log "queued successor $nxt"
fi

# ---- reservation idle guard (scripts/slurm/resv_watchdog.sh) --------------
# Runs here on the compute node every 60 s (NCSA kills login-node processes at
# 7 days / 30 min CPU): e-mails the owner if a reserved node has had no allocated
# GPU for 5 min and re-queues the GPU fillers. Killed before the final `wait`.
guard_pid=""
if [ -f scripts/slurm/resv_watchdog.sh ]; then
    ( while sleep 60; do bash scripts/slurm/resv_watchdog.sh >/dev/null 2>&1; done ) &
    guard_pid=$!
fi

# ---- SIGUSR1 (15 min before the limit): checkpoint and leave ---------------
trainer_pid=""
cur_save_dir=""
LEAVING=0
on_usr1() {
    log "SIGUSR1: asking the trainer to checkpoint and exit"
    LEAVING=1
    # primary mechanism: the trainer polls <save_dir>/STOP_REQUEST every step
    # (Python signal handlers do not fire under torch.compile)
    [ -n "$cur_save_dir" ] && touch "$cur_save_dir/STOP_REQUEST"
    # signal only the rank processes (torchrun's direct children); NOT torchrun
    # itself (default USR1 action is to die) and NOT the DataLoader worker
    # forks, which share the cmdline -- signalling them made a worker exit and
    # crashed rank 0 mid-save on 2026-09-27.
    if [ -n "$trainer_pid" ]; then
        for pid in $(pgrep -P "$trainer_pid"); do kill -USR1 "$pid" 2>/dev/null; done
    fi
}
trap on_usr1 SIGUSR1

# ---- background: 21B-token Dolma corpus (parallel workers) -----------------
tokenize_bg() {
    [ -f "$DATA21/index.json" ] && return 0
    log "tokenizing Dolma v1.7 -> $DATA21 with $N_TOK_WORKERS workers"
    mkdir -p "$DATA21"
    pids=()
    for i in $(seq 0 $((N_TOK_WORKERS - 1))); do
        w=$DATA21/w$i
        if [ -f "$w/index.json" ]; then continue; fi
        rm -rf "$w"
        python3 scripts/pretokenize.py --dataset allenai/dolma --dataset-version v1_7 \
            --out-dir "$w" --token-budget 21000000000 --seq-len 1024 --seed 0 \
            --num-shards "$N_TOK_WORKERS" --shard-index "$i" \
            --tokenizer-id /work/hdd/bfqt/shared/dagformer-models/tokenizer > "logs/pretok21b_w$i.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    python3 scripts/merge_pretok_shards.py --root "$DATA21" && \
    python3 scripts/split_eval_from_mmap.py --index-dir "$DATA21" --n-windows 200 --out "$DATA21/eval_cache.pt" && \
    log "corpus ready: $(python3 -c "import json;print(json.load(open('$DATA21/index.json'))['total_tokens'])") tokens"
}
if [ "$TOKENIZE" = 1 ]; then tokenize_bg > logs/pretok21b_driver_$SLURM_JOB_ID.log 2>&1 & fi

# ---- the queue ------------------------------------------------------------
run_task() {   # cmd done_marker wait
    local cmd=$1 marker=$2 waitfor=$3
    [ -f "$marker" ] && { log "skip (done): task $cmd"; return 0; }
    if [ -n "$waitfor" ]; then
        while [ ! -f "$waitfor" ]; do [ -f "$STOP" ] && return 0; sleep 60; done
    fi
    log "=== task: $cmd"
    if bash -c "$cmd"; then touch "$marker"; log "=== task done"; else log "TASK FAILED: $cmd"; fi
}

run_item() {   # config script wait eval fallback skipif
    local cfg=$1 script=$2 waitfor=$3 evalc=$4 fallback=$5 skipif=$6
    local save_dir total
    save_dir=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['save_dir'])")
    total=$(python3 -c "import yaml;print(yaml.safe_load(open('$cfg'))['total_steps'])")
    mkdir -p "$save_dir"
    [ -f "$save_dir/DONE" ] && { log "skip (done): $cfg"; return 0; }
    rm -f "$save_dir/STOP_REQUEST"
    if [ -n "$waitfor" ]; then
        while [ ! -f "$waitfor" ]; do [ -f "$STOP" ] && return 0; sleep 60; done
    fi
    [ -n "$skipif" ] && [ -f "$skipif" ] && { log "skip ($skipif exists): $cfg"; return 0; }
    [ -n "$evalc" ] && [ ! -f "$save_dir/eval_cache.pt" ] && cp "$evalc" "$save_dir/eval_cache.pt"
    local attempt=0
    while [ $attempt -lt 3 ]; do
        attempt=$((attempt + 1))
        log "=== start $cfg (attempt $attempt) -> $save_dir"
        cur_save_dir=$save_dir
        torchrun --nproc_per_node=4 --master_port=$((29000 + RANDOM % 1000)) "$script" --config "$cfg" &
        trainer_pid=$!
        wait "$trainer_pid"; rc=$?
        while [ "$LEAVING" = 1 ] && kill -0 "$trainer_pid" 2>/dev/null; do sleep 10; done   # wait was interrupted by the trap
        [ "$LEAVING" = 1 ] && { wait "$trainer_pid" 2>/dev/null; rc=$?; }
        trainer_pid=""
        if [ $rc -eq 0 ] && [ -f "$save_dir/checkpoint_step${total}.pt" ]; then
            touch "$save_dir/DONE"; log "=== done $cfg"; return 0
        fi
        if [ "$LEAVING" = 1 ]; then log "checkpointed; leaving for the successor"; exit 0; fi
        [ -f "$STOP" ] && return 0
        log "trainer exited rc=$rc without finishing (step file missing); retrying in 60s"
        sleep 60
    done
    if [ -n "$fallback" ] && [ ! -f "$save_dir/DONE" ]; then
        log "falling back to $fallback"
        run_item "$fallback" "$script" "" "$evalc" "" ""
    else
        log "GIVING UP on $cfg after 3 attempts"
    fi
}

while read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    [ -f "$STOP" ] && { log "STOP present, leaving queue"; break; }
    cfg=""; script=scripts/pretrain_dagformer.py; waitfor=""; evalc=""; fallback=""; skipif=""; task=""; marker=""
    case "$line" in task=*)
        task=${line#task=}; task=${task%% done=*}
        marker=$(sed -n 's/.* done=\([^ ]*\).*/\1/p' <<<"$line"); waitfor=$(sed -n 's/.* wait=\([^ ]*\).*/\1/p' <<<"$line")
        run_task "$task" "$marker" "$waitfor"; continue;;
    esac
    for kv in $line; do
        case "$kv" in
            config=*) cfg=${kv#config=};; script=*) script=${kv#script=};; wait=*) waitfor=${kv#wait=};;
            eval=*) evalc=${kv#eval=};; fallback=*) fallback=${kv#fallback=};; skipif=*) skipif=${kv#skipif=};;
        esac
    done
    run_item "$cfg" "$script" "$waitfor" "$evalc" "$fallback" "$skipif"
done < "$QUEUE"

log "queue exhausted; no further successors (STOP written); waiting for the tokenizer if still running"
touch "$STOP"
[ -n "$guard_pid" ] && kill "$guard_pid" 2>/dev/null   # the guard loop would block the wait below
wait
log "all done"
