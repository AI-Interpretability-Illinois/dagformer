#!/bin/bash
# Login-node loop around resv_watchdog.sh: a temporary bridge only. NCSA kills login-node
# processes at 7 days wall / 30 min CPU, so this exits by itself after MAX_AGE_S (6d20h)
# and must NOT be relaunched to extend coverage; the guard inside the SLURM jobs
# (guard_tick) is the permanent mechanism. Also exits when the reservation is gone or the
# stop file exists.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-/u/xiaocong/dagformer}
LOG=$REPO_DIR/logs/watchdog_$(hostname -s).log
STOP_FILE=${STOP_FILE:-$HOME/.stop_resv_keepalive}
MAX_AGE_S=${MAX_AGE_S:-590400}
START_S=$(date +%s)
echo "$(date '+%F %T') watchdog daemon pid $$ started; exits by $(date -d @$((START_S + MAX_AGE_S)) '+%F %T')" >>"$LOG"
while [ ! -f "$STOP_FILE" ]; do
  [ $(( $(date +%s) - START_S )) -ge "$MAX_AGE_S" ] && { echo "$(date '+%F %T') max age reached, daemon exiting (do not relaunch)" >>"$LOG"; exit 0; }
  out=$(bash "$HERE/resv_watchdog.sh" 2>&1); rc=$?
  echo "$out" >>"$LOG"
  grep -q "reservation .* not found" <<<"$out" && { echo "$(date '+%F %T') reservation gone, daemon exiting" >>"$LOG"; exit 0; }
  sleep 300
done
echo "$(date '+%F %T') stop file present, daemon exiting" >>"$LOG"
