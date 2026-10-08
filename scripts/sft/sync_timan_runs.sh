#!/bin/bash
# Copy SFT runs that finished on a timan host back into the Delta run root, so the Delta task list,
# the eval trigger and the benchmark scripts see them. Only small files move (final/ export,
# log.csv, samples.txt, DONE); resumable ckpt.pt stays on timan. Removes the Delta "ext:" lock.
# Run from a Delta login node (coordinator tick) or by hand:
#   bash scripts/sft/sync_timan_runs.sh [host, default timan108]
set -uo pipefail
HOST=${1:-timan108}
TR=/srv/local/xy51/dagformer_sft_timan/runs
DR=/work/hdd/bfqt/xiaocong/dagformer_sft/runs
done_list=$(ssh -o BatchMode=yes "$HOST" "cd $TR 2>/dev/null && for d in */; do [ -f \"\$d/DONE\" ] && echo \${d%/}; done" 2>/dev/null | grep -v "module: command")
n=0
for t in $done_list; do
  [ -f "$DR/$t/DONE" ] && continue
  mkdir -p "$DR/$t"
  rsync -a --exclude ckpt.pt --exclude ckpt.pt.tmp --exclude LOCK "$HOST:$TR/$t/" "$DR/$t/" 2>&1 | grep -v "module: command"
  if [ -f "$DR/$t/final/checkpoint.pt" ] && [ -f "$DR/$t/DONE" ]; then
    rm -rf "$DR/$t/LOCK"; n=$((n+1)); echo "[$(date '+%F %T')] synced $t from $HOST"
  else
    rm -f "$DR/$t/DONE"; echo "[$(date '+%F %T')] incomplete copy of $t from $HOST, will retry" >&2
  fi
done
[ "$n" -gt 0 ] && bash "$(dirname "$0")/../slurm/submit_sft_eval.sh" 2>&1 | tail -n 1
exit 0
