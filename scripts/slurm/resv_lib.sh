# Shared helpers for the reservation anchor scripts. Source, do not execute.
#
# "Real" job = any job (any user) requesting the reservation whose name does not
# start with $ANCHOR_PREFIX.
RESV=${RESV:-sup-30781}
ANCHOR_PREFIX=${ANCHOR_PREFIX:-resv-anchor}
STATE_DIR=${STATE_DIR:-$HOME/.resv_keepalive}
STOP_FILE=${STOP_FILE:-$HOME/.stop_resv_keepalive}
STUCK_AFTER_S=${STUCK_AFTER_S:-1800}   # a pending real job that still has not started this
                                       # long after we freed the GPU for it is ignored

# id|state|name|nodelist|reqnodes|reason  for every non-anchor job in the reservation
resv_real_jobs() {
  squeue -h -R "$RESV" -o "%i|%T|%j|%N|%n|%r" 2>/dev/null | awk -F'|' -v p="$ANCHOR_PREFIX" '$3 !~ "^"p'
}

# node_in_list NODE HOSTLIST -> true if NODE is in the (possibly bracketed) hostlist
node_in_list() {
  [ -n "$2" ] && scontrol show hostnames "$2" 2>/dev/null | grep -qx "$1"
}

# real_running_on NODE -> ids of real jobs running (or completing) on NODE
real_running_on() {
  local node=$1 id nl
  resv_real_jobs | awk -F'|' '$2=="RUNNING"||$2=="COMPLETING"{print $1"|"$4}' |
  while IFS='|' read -r id nl; do node_in_list "$node" "$nl" && echo "$id"; done
}

# real_pending_for NODE -> ids of real jobs that are pending for lack of resources or
# priority and could land on NODE (no --nodelist, or a --nodelist that includes NODE).
# Remembers when we first stepped aside for each id; after STUCK_AFTER_S it is ignored
# so that a job that cannot start anyway does not keep the anchor away forever.
real_pending_for() {
  local node=$1 now id rn first f="$STATE_DIR/yield_$1"
  now=$(date +%s); mkdir -p "$STATE_DIR"; touch "$f"
  resv_real_jobs |
  awk -F'|' '$2=="PENDING" && ($6=="Resources"||$6=="Priority"||$6=="None"){print $1"|"$5}' |
  while IFS='|' read -r id rn; do
    if [ -n "$rn" ] && ! node_in_list "$node" "$rn"; then continue; fi
    first=$(awk -v j="$id" '$1==j{print $2}' "$f")
    if [ -z "$first" ]; then
      echo "$id $now" >>"$f"; echo "$id"
    elif [ $((now - first)) -lt "$STUCK_AFTER_S" ]; then
      echo "$id"
    fi
  done
}

# anchors_for NODE STATES -> ids of our anchor jobs for NODE in the given squeue states
anchors_for() { squeue -h -u "$USER" -n "$ANCHOR_PREFIX-$1" -t "$2" -o %i 2>/dev/null; }

log() { echo "[$(date '+%F %T')] $*"; }

# resv_time_limit BEGIN_EPOCH -> prints HH:MM:SS for a job starting at BEGIN_EPOCH so that it
# ends END_MARGIN_S before the reservation does, capped at MAX_HOURS; prints nothing if less
# than MIN_LINK_S would remain. Also sets RESV_START_S / RESV_END_S.
resv_time_limit() {   # BEGIN_EPOCH [CAP_SECONDS]
  local begin_s=$1 cap_s=${2:-0} resv avail_s max_s limit_s
  resv=$(scontrol show reservation "$RESV" 2>/dev/null) || return 1
  [ -n "$resv" ] || return 1
  RESV_START_S=$(date -d "$(sed -n 's/.*StartTime=\([^ ]*\).*/\1/p' <<<"$resv" | head -1)" +%s)
  RESV_END_S=$(date -d "$(sed -n 's/.*EndTime=\([^ ]*\).*/\1/p' <<<"$resv" | head -1)" +%s)
  [ "$RESV_START_S" -gt "$begin_s" ] && begin_s=$RESV_START_S
  avail_s=$(( RESV_END_S - ${END_MARGIN_S:-120} - begin_s ))
  max_s=$(( ${MAX_HOURS:-48} * 3600 ))
  limit_s=$(( avail_s < max_s ? avail_s : max_s ))
  [ "$cap_s" -gt 0 ] && [ "$cap_s" -lt "$limit_s" ] && limit_s=$cap_s
  [ "$limit_s" -ge "${MIN_LINK_S:-600}" ] || return 2
  s_to_slurm "$limit_s"
}

s_to_slurm() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

# slurm_to_s "D-HH:MM:SS" | "HH:MM:SS" | "MM:SS" -> seconds
slurm_to_s() {
  local t=$1 d=0 a
  [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
  IFS=: read -ra a <<<"$t"
  case ${#a[@]} in
    3) echo $(( d*86400 + a[0]*3600 + a[1]*60 + a[2] ));;
    2) echo $(( d*86400 + a[0]*60 + a[1] ));;
    *) echo $(( d*86400 + a[0] ));;
  esac
}

# --- backfill gap awareness ----------------------------------------------------------
# Backfill only starts a low-priority job if it cannot delay a queued higher-priority
# job's projected start (squeue %S). So a holder (filler / dry-run) must end before the
# next projected real-job start on its node, or it will sit pending next to an idle GPU.
HOLDER_MIN_S=${HOLDER_MIN_S:-1200}      # never ask for less than 20 min
GAP_MARGIN_S=${GAP_MARGIN_S:-120}

# resv_gap_s NODE -> seconds until the earliest projected start of a pending real job that
# could land on NODE; prints nothing when backfill has no estimate for any of them.
resv_gap_s() {
  local node=$1 now id name req start best="" s
  now=$(date +%s)
  while IFS='|' read -r id name req start; do
    case $name in resv-filler*|resv-dryrun*|resv-anchor*) continue;; esac
    { [ -z "$start" ] || [ "$start" = "N/A" ]; } && continue
    if [ -n "$req" ] && ! node_in_list "$node" "$req"; then continue; fi
    s=$(date -d "$start" +%s 2>/dev/null) || continue
    { [ -z "$best" ] || [ "$s" -lt "$best" ]; } && best=$s
  done < <(squeue -h -R "$RESV" -t PD -o "%i|%j|%n|%S" 2>/dev/null)
  [ -n "$best" ] && echo $(( best - now ))
}

# holder_cap_s NODE BEGIN_IN -> cap (seconds) for a holder starting BEGIN_IN from now, or 0
holder_cap_s() {
  local gap cap
  gap=$(resv_gap_s "$1") || true
  [ -n "$gap" ] || { echo 0; return; }
  cap=$(( gap - GAP_MARGIN_S - ${2:-0} ))
  [ "$cap" -lt "$HOLDER_MIN_S" ] && cap=$HOLDER_MIN_S
  echo "$cap"
}

# node_can_host NODE CPUS MEM_MB -> true if NODE has >=1 free GPU and the given CPUs / memory
node_can_host() {
  scontrol show node -o "$1" 2>/dev/null | awk -v need_c="$2" -v need_m="$3" '
    { cfg=""; al=""; for (i=1;i<=NF;i++) { if ($i ~ /^CfgTRES=/) cfg=substr($i,9); if ($i ~ /^AllocTRES=/) al=substr($i,11) } }
    function tres(str, key,   n, a, i, kv) { n=split(str,a,","); for(i=1;i<=n;i++){ split(a[i],kv,"="); if (kv[1]==key) return kv[2] } return 0 }
    function mb(v) { if (v ~ /G$/) return v*1024; if (v ~ /T$/) return v*1048576; return v+0 }
    END { fg = tres(cfg,"gres/gpu") - tres(al,"gres/gpu"); fc = tres(cfg,"cpu") - tres(al,"cpu"); fm = mb(tres(cfg,"mem")) - mb(tres(al,"mem"));
          exit !(fg >= 1 && fc >= need_c && fm >= need_m) }'
}

# refresh_pending_holders: keep our PENDING fillers / dry-runs startable by backfill.
#  (a) cap each to the projected gap before the next real job on its node (squeue %S);
#  (b) if one has been pending for >3 min while its node has an idle GPU with room for it,
#      backfill's real projections conflict with its length: halve the limit (floor 20 min);
#  (c) if the projected gap grew >30 min beyond a shortened limit, re-queue it longer.
refresh_pending_holders() {
  local id name limit reason node kind cap cur now since f need_c need_m new
  now=$(date +%s); mkdir -p "$STATE_DIR"
  sync_holder_holds
  while IFS='|' read -r id name limit reason; do
    [ "$reason" = JobHeldUser ] && continue          # held for a multi-GPU real job (sync_holder_holds)
    node=${name##*-}; kind=${name%-*}; kind=${kind#resv-}
    case $kind in filler) need_c=8; need_m=24576;; *) need_c=2; need_m=4096;; esac
    cur=$(slurm_to_s "$limit"); cap=$(holder_cap_s "$node" 0); f="$STATE_DIR/pending_idle_$id"
    if [ "$cap" -gt 0 ] && [ "$cap" -lt "$cur" ]; then
      scontrol update JobId="$id" TimeLimit="$(s_to_slurm "$cap")" 2>/dev/null &&
        log "shrank pending $name $id to $(s_to_slurm "$cap") (gap before next real job)"
      continue
    fi
    if [ "$reason" != "BeginTime" ] && node_can_host "$node" "$need_c" "$need_m"; then
      since=$(cat "$f" 2>/dev/null || echo "$now"); [ -f "$f" ] || echo "$now" >"$f"
      if [ $(( now - since )) -ge 180 ] && [ "$cur" -gt "$HOLDER_MIN_S" ]; then
        new=$(( cur / 2 )); [ "$new" -lt "$HOLDER_MIN_S" ] && new=$HOLDER_MIN_S
        scontrol update JobId="$id" TimeLimit="$(s_to_slurm "$new")" 2>/dev/null &&
          log "pending $name $id idle-blocked $(( (now-since)/60 )) min: halved limit to $(s_to_slurm "$new")"
        echo "$now" >"$f"
      fi
    else
      rm -f "$f"
      if [ "$cur" -lt $(( ${MAX_HOURS:-1} * 3600 )) ] && { [ "$cap" -eq 0 ] || [ "$cap" -gt $(( cur + 1800 )) ]; }; then
        scancel "$id" 2>/dev/null && log "re-queuing $name $id: gap grew beyond $(s_to_slurm "$cur")"
        bash "$REPO_DIR/scripts/slurm/resv_${kind}_submit.sh" "$node" >/dev/null 2>&1
      fi
    fi
  done < <(squeue -h -u "$USER" -t PD -o "%i|%j|%l|%r" 2>/dev/null | grep -E "^[0-9]+\|resv-(filler|dryrun)-")
  find "$STATE_DIR" -name 'pending_idle_*' -mmin +120 -delete 2>/dev/null
}

# claimed_nodes -> nodes where a pending real job needs more than one holder to leave (resv_plan.py --claimed)
claimed_nodes() { "${RESV_PY:-/u/xiaocong/anaconda3/envs/modularity/bin/python}" "$RESV_SCRIPTS_DIR/resv_plan.py" --claimed 2>/dev/null; }

# sync_holder_holds: hold our PENDING holders on claimed nodes (else each GPU a yielding holder frees is
# backfilled by the next holder and a multi-GPU real job never fits), release held ones elsewhere.
sync_holder_holds() {
  local claimed id name reason node
  claimed=$(claimed_nodes) || return 0
  while IFS='|' read -r id name reason; do
    node=${name##*-}
    if grep -qx "$node" <<<"$claimed"; then
      [ "$reason" = JobHeldUser ] || { scontrol hold "$id" 2>/dev/null && log "held pending $name $id (a real job needs $node)"; }
    elif [ "$reason" = JobHeldUser ]; then
      scontrol release "$id" 2>/dev/null && log "released $name $id ($node no longer claimed)"
    fi
  done < <(squeue -h -u "$USER" -t PD -o "%i|%j|%r" 2>/dev/null | grep -E "^[0-9]+\|resv-(filler|dryrun)-")
}

# fillers_for NODE STATES -> ids of our filler jobs for NODE in the given squeue states
FILLER_PREFIX=${FILLER_PREFIX:-resv-filler}
fillers_for() { squeue -h -u "$USER" -n "$FILLER_PREFIX-$1" -t "$2" -o %i 2>/dev/null; }

# holders_for PREFIX NODE STATES -> ids of our jobs named PREFIX-NODE in the given states
holders_for() { squeue -h -u "$USER" -n "$1-$2" -t "$3" -o %i 2>/dev/null; }
DRYRUN_PREFIX=${DRYRUN_PREFIX:-resv-dryrun}

# guard_tick: one run of the reservation idle guard from inside a SLURM job; silent unless it alerts.
RESV_SCRIPTS_DIR=${RESV_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
guard_tick() {
  local out
  out=$(bash "$RESV_SCRIPTS_DIR/resv_watchdog.sh" 2>&1) || log "idle guard: $(grep -m2 ALERT <<<"$out" | tr '\n' ' ')"
}

# --- account fallback (2026-09-28: biro-delta-gpu added to the reservation) ------------
# Accounts in order of preference. A submission rejected for quota/policy reasons is retried
# on the next account; a pending job blocked by an exhausted allocation is moved to the next.
RESV_ACCOUNTS=${RESV_ACCOUNTS:-bfqt-delta-gpu biro-delta-gpu}
QUOTA_ERR_RE='accounting/QOS policy|AssocGrp|AssocMax|[Ii]nvalid account|[Bb]illing|[Ii]nsufficient|allocation'
QUOTA_REASON_RE='^(AssocGrp[A-Za-z]*(Minutes|Min|Billing)[A-Za-z]*|AssocMax[A-Za-z]*Minutes[A-Za-z]*|InvalidAccount|AccountNotAllowed)$'

# sbatch_with_fallback ARGS... -> prints the job id; tries each account in RESV_ACCOUNTS
sbatch_with_fallback() {
  local acct out rc
  for acct in $RESV_ACCOUNTS; do
    out=$(sbatch --account="$acct" "$@" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
      [ "$acct" != "${RESV_ACCOUNTS%% *}" ] && log "submitted on fallback account $acct" >&2
      echo "$out" | tail -n 1; return 0
    fi
    if grep -qE "$QUOTA_ERR_RE" <<<"$out"; then
      log "sbatch on $acct rejected ($(tr '\n' ' ' <<<"$out" | cut -c1-160)); trying next account" >&2
      continue
    fi
    echo "$out" >&2; return $rc
  done
  log "sbatch rejected on every account ($RESV_ACCOUNTS)" >&2; return 1
}

# fix_quota_pending: move this user's pending jobs in the reservation that are blocked by an
# exhausted allocation to the next account in RESV_ACCOUNTS (holders and the user's own runs).
fix_quota_pending() {
  local id acct reason name next
  while IFS='|' read -r id acct reason name; do
    grep -qE "$QUOTA_REASON_RE" <<<"$reason" || continue
    next=$(tr ' ' '\n' <<<"$RESV_ACCOUNTS" | grep -A1 -x "$acct" | sed -n 2p)
    [ -n "$next" ] || { log "pending $name $id blocked ($reason) on $acct; no further account" >&2; continue; }
    scontrol update JobId="$id" Account="$next" 2>&1 | grep -v '^$' >&2
    log "pending $name $id blocked ($reason) on $acct -> moved to $next" >&2
  done < <(squeue -h -u "$USER" -R "$RESV" -t PD -o "%i|%a|%r|%j" 2>/dev/null)
}
