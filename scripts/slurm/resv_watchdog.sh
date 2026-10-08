#!/bin/bash
# Last-line guard for reservation sup-30781: a node with NO allocated GPU for more than
# IDLE_ALERT_S seconds means every layer (real jobs, SFT fillers, dry-run fallback) has
# failed to occupy it. This script is stateless apart from ~/.resv_keepalive/idle_since_*:
#
#   exit 0  everything fine (prints one status line)
#   exit 2  ALERT: a node has been fully idle > IDLE_ALERT_S (prints ALERT lines first),
#           or the reservation has disappeared
#
# On alert it also self-heals: re-seeds the pending filler and dry-run for the node and
# prints why the pending holders are not starting (squeue reason).
# Called every 60 s from inside SLURM jobs (fillers, dry-runs, reserved_train chains) via
# guard_tick in resv_lib.sh, so it runs on compute nodes, not under the login-node limits.
# Alerts are appended to logs/resv_guard.log and e-mailed (once per node per hour).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/resv_lib.sh"
IDLE_ALERT_S=${IDLE_ALERT_S:-300}
ALERT_LOG=${ALERT_LOG:-/u/xiaocong/dagformer/logs/resv_guard.log}
WHERE="$(hostname -s)${SLURM_JOB_ID:+ job $SLURM_JOB_ID}"
ALERT_EMAIL=${ALERT_EMAIL:-xy51@illinois.edu}      # empty to disable; at most one mail per hour per node
mkdir -p "$STATE_DIR"

email_alert() {   # NODE MESSAGE
  [ -n "$ALERT_EMAIL" ] && command -v mail >/dev/null || return 0
  local f="$STATE_DIR/mailed_$1"
  [ -f "$f" ] && [ $(( $(date +%s) - $(cat "$f") )) -lt 3600 ] && return 0
  printf '%s\n(checked from %s)\n\n%s\n' "$2" "$WHERE" "$(squeue -R "$RESV" -o '%i %T %j %u %N %r' 2>/dev/null)" |
    mail -s "[sup-30781] $1 idle: $2" "$ALERT_EMAIL" && date +%s >"$f"
}
now=$(date +%s); rc=0; status=()

resv=$(scontrol show reservation "$RESV" 2>/dev/null)
if [ -z "$resv" ]; then
  echo "ALERT: reservation $RESV not found (purged or expired)"; exit 2
fi
sync_holder_holds 2>&1 | sed 's/^/  /'      # hold/release pending holders around multi-GPU real jobs
fix_quota_pending 2>&1 | sed 's/^/  /'       # blocked by an exhausted allocation -> next account
nodes=$(scontrol show hostnames "$(sed -n 's/.*Nodes=\([^ ]*\).*/\1/p' <<<"$resv" | head -1)")

for node in $nodes; do
  alloc=$(scontrol show node -o "$node" 2>/dev/null | grep -o "AllocTRES=[^ ]*")
  gpus=$(grep -o "gres/gpu=[0-9]*" <<<"$alloc" | cut -d= -f2); gpus=${gpus:-0}
  f="$STATE_DIR/idle_since_$node"
  if [ "$gpus" -gt 0 ]; then
    rm -f "$f"; status+=("$node:${gpus}gpu")
    continue
  fi
  [ -f "$f" ] || echo "$now" >"$f"
  idle=$(( now - $(cat "$f") ))
  status+=("$node:IDLE ${idle}s")
  if [ "$idle" -ge "$IDLE_ALERT_S" ]; then
    rc=2
    pend=$(squeue -h -u "$USER" -t PD -o "%j:%r" 2>/dev/null | grep -E "resv-(filler|dryrun)-$node" | tr '\n' ' ')
    msg="ALERT: $node has had no allocated GPU for $((idle/60)) min; pending holders: ${pend:-none}"
    echo "$msg"; email_alert "$node" "$msg"
    echo "[$(date '+%F %T')] ($WHERE) $msg" >>"$ALERT_LOG" 2>/dev/null
    bash "$HERE/resv_filler_submit.sh" "$node" 2>&1 | tail -n 1 | sed 's/^/  reseed filler: /'
    bash "$HERE/resv_dryrun_submit.sh" "$node" 2>&1 | tail -n 1 | sed 's/^/  reseed dryrun: /'
  fi
done
echo "watchdog $(date '+%F %T'): ${status[*]} | holders: $(squeue -h -u "$USER" -o "%j=%T" 2>/dev/null | grep -E "resv-(filler|dryrun)" | sed 's/resv-//' | tr '\n' ' ')"
exit $rc
