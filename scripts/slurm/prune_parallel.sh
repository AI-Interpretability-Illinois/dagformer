#!/bin/bash
# Run a pruning runlist on the GPUs of the current node (a task of a reserved_train.slurm chain): one sequential worker
# per GPU, each claiming the next unclaimed run (lock dirs private to this invocation, so a killed job leaves no stale
# lock). Finished runs (summary.json) are skipped and interrupted ones resume from <save_dir>/latest, so a successor chain
# job can simply re-run the task. Exits non-zero if a run has no summary.
#   bash scripts/slurm/prune_parallel.sh <runlist> [ngpu]
# Runlist lines: <prune config> key=value ... (save_dir=... required; '#' comments allowed).
set -uo pipefail
RL=$1; NG=${2:-$(nvidia-smi -L | wc -l)}
cd "${REPO_DIR:-/u/xiaocong/dagformer}"; mkdir -p logs/prune
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE=disabled OMP_NUM_THREADS=4
mapfile -t LINES < <(grep -v '^[[:space:]]*#' "$RL" | grep -v '^[[:space:]]*$')
LOCKS=$(mktemp -d "${TMPDIR:-/tmp}/prune_parallel.XXXXXX"); trap 'rm -rf "$LOCKS"' EXIT
sd_of() { sed -n 's/.*save_dir=\([^ ]*\).*/\1/p' <<< "$1"; }
worker() {
    local g=$1 i cfg rest sd name ov rc
    for ((i = 0; i < ${#LINES[@]}; i++)); do
        read -r cfg rest <<< "${LINES[$i]}"
        local args=()
        for ov in $rest; do args+=(--override "$ov"); done
        sd=$(sd_of "${LINES[$i]}"); name=$(basename "$sd")
        [ -f "$sd/summary.json" ] && continue
        mkdir "$LOCKS/$i" 2>/dev/null || continue
        echo "[$(date '+%F %T')] gpu$g START $name"
        CUDA_VISIBLE_DEVICES=$g python3 scripts/prune_finetune.py --config "$cfg" "${args[@]}" > "logs/prune/$name.log" 2>&1
        rc=$?; echo "[$(date '+%F %T')] gpu$g END $name exit=$rc"
    done
}
echo "[$(date '+%F %T')] prune_parallel: ${#LINES[@]} runs on $NG GPUs from $RL"
for ((g = 0; g < NG; g++)); do worker "$g" & done
wait
missing=0
for l in "${LINES[@]}"; do sd=$(sd_of "$l"); [ -f "$sd/summary.json" ] || { echo "missing summary: $sd"; missing=1; }; done
exit $missing
