# Opportunistic instruction tuning on the reserved GPUs

Reservation `sup-30781` (gpua046-047, 2×4 A100-40GB, 2026-09-26 20:00 → 2026-10-18 12:00)
is purged by SLURM after 30 idle minutes (`PURGE_COMP=00:30:00`). Instead of a
heartbeat, idle GPUs run instruction tuning (SFT) of the shared pretrained
checkpoints, and step aside the moment a real job needs them.

## Pieces

| File | Role |
|---|---|
| `scripts/slurm/resv_filler.slurm` | 1-GPU / 8-CPU / 24 GB **filler** job (1h links). Runs the SFT worker; falls back to the GPU heartbeat if the worker crashes twice or no task is left (re-checking the task list every 30 min). Every 60 s asks the planner whether to yield; on yield the worker checkpoints (≤ 240 s) and the job exits, re-queuing itself with `--begin=now+5min`. |
| `scripts/slurm/resv_dryrun.slurm` | 1-GPU / 2-CPU / 4 GB **dry-run** job: the independent fallback that holds an idle GPU with the matmul heartbeat (`keepalive_gpu.py`, 4096² fp16 matmuls ~10 % of the time, so the GPU is visibly busy) when no filler is using it (fillers failed to submit, schedule or run). Yields to real jobs *and* to pending fillers. Depends only on bash + squeue; if the planner is broken it uses a crude rule (leave when any non-dry-run job is pending for resources). |
| `scripts/slurm/resv_filler_submit.sh NODE`, `resv_dryrun_submit.sh NODE` | Keep exactly one PENDING filler and one PENDING dry-run per node (`--nice=200` / `300`, so real ≈ 800 > filler ≈ 600 > dry-run ≈ 500 in priority). Called by every holder at start and exit. |
| `scripts/slurm/resv_plan.py` | Yield planner. Reads `squeue -R sup-30781` and node `AllocTRES`. Pass 1: each real job pending for Resources/Priority (priority order) gets the nodes where it fits once the fewest holders leave (GPU, CPU and memory), dry-runs before fillers, youngest first. Pass 2: pending fillers may displace dry-runs. Running real jobs are never touched. A job still pending 30 min after holders yielded for it is ignored (stuck for another reason). `--status` prints the plan. `--claimed` lists nodes where a pending real job needs more than one holder to leave; `sync_holder_holds` (resv_lib.sh, run every minute by holders and the watchdog) keeps pending holders there on `scontrol hold` and releases them once the job has started, and holders queued for such a node are submitted with `--hold`. Without this a 4-GPU job never fits: holders leave one at a time and backfill restarts a pending holder in each freed GPU (seen 2026-09-30). |
| `scripts/sft/sft_train.py` | Resumable SFT worker: claims a task via an atomic-mkdir lock on /work (stale if the owning job is gone), prepares data once, trains, checkpoints every `save_every` steps and on SIGUSR1/SIGTERM, exports the result, marks DONE, takes the next task. Exit 3 = nothing left. |
| `scripts/sft/sft_data.py` | ChatML rendering with the tokenizer's `<|im_start|>/<|im_end|>`, assistant-only loss mask, packing into 1025-token windows (`*_ids.npy`, `*_mask.npy`). Built once per dataset by the first filler that needs it (under a lock), in resumable parts of 20k conversations, so an interrupted preparation continues instead of restarting. |
| `configs/sft/tasks.yaml` | Task list (claimed top to bottom). `configs/sft/tasks_smoke.yaml` is a 60-step smoke test. |
| `scripts/slurm/resv_watchdog.sh` | Last guard: a reserved node with no allocated GPU for > 5 min (every layer failed) → alert line in `logs/resv_guard.log`, one e-mail per node per hour to the owner, holders re-queued. It runs **inside SLURM jobs** (NCSA kills login-node processes at 7 days wall or 30 min CPU): every filler and dry-run calls it each minute (`guard_tick` in `resv_lib.sh`), and `reserved_train.slurm` runs it in a background loop, so some job always watches both nodes. `resv_watchdog_daemon.sh` is a temporary login-node bridge that exits by itself after 6 d 20 h and must not be relaunched. |
| `scripts/slurm/resv_lib.sh` | Shared shell helpers: time-limit trimming to the reservation end, backfill-gap handling (see below), job listing. |

Paths on /work: raw data `/work/hdd/bfqt/xiaocong/dagformer_sft/hf`, packed windows
`.../data/<dataset>`, runs `.../runs/<task>/` (`ckpt.pt` while running, `final/` in the
shared-model layout, `log.csv`, `samples.txt`, `DONE`).

## Tasks and recipe

Datasets: `smol_smoltalk` = HuggingFaceTB/smol-smoltalk (the SFT mix used for
SmolLM2-135M/360M, 460k conversations) and `alpaca_dolly` = yahma/alpaca-cleaned +
databricks-dolly-15k (67k). Global batch 64 × 1024 tokens, AdamW (0.9, 0.95), wd 0.1,
3 % warmup, cosine to 10 %, bf16 weights (as in `scripts/prune_finetune.py`). For
DAGFormer the predictor and routing modules are trained with the same lr.

Two task sources, in this order:

1. **Explicit** (`tasks:` in `configs/sft/tasks.yaml`): the shared final checkpoints in
   `/work/hdd/bfqt/shared/dagformer-models` (75M/150M/300M × baseline/DAGFormer), 2 epochs
   smol-smoltalk then 3 epochs alpaca-dolly.
2. **Discovered** (`discover:`): every `config=`/`fallback=` entry of the pretraining queue
   files `experiments/pretrain_1b/queue.txt` and `experiments/pretrain_300m/queue.txt` whose
   `save_dir` holds the `DONE` marker (written by `reserved_train.slurm` only after the run's
   last step) **and** `checkpoint_step{total_steps}.pt`. A run that is still training, or
   has only intermediate checkpoints, is never used. The list is re-scanned every time a
   filler looks for work, so runs finishing during the reservation (75M locality variants,
   300M and 1B DAGFormer variants) are picked up automatically, oldest-finished first.
   Batch shape and lr follow the model size: <200M `mb16×4, 2e-4`; <500M `mb8×8, 2e-4`;
   <2B `mb4×16, 1e-4`. `python scripts/sft/sft_train.py --list` shows the resolved list
   with each task's status.

## Accounts

The reservation accepts two accounts, `bfqt-delta-gpu` (preferred) and `biro-delta-gpu`
(added 2026-09-28). Delta enforces allocations in "safe" mode, so a job its account cannot
pay for is rejected at submission, and a queued job can later block with an `AssocGrp…`
reason. `sbatch_with_fallback` (resv_lib.sh) retries a rejected holder submission on the
next account, `reserved_train.slurm` does the same for its successor, and
`fix_quota_pending` (run by every watchdog pass) moves any of the owner's pending jobs in
the reservation that are blocked by an exhausted allocation to the next account. Order
and list: `RESV_ACCOUNTS` in resv_lib.sh.

## Why holders have short, adaptive time limits

Fillers and dry-runs start only through backfill (the partition always has
higher-priority pending jobs). Backfill starts a low-priority job only if it cannot
delay any higher-priority job's projected start, so a 4h holder next to an idle GPU
stays pending when a real job is projected to need that GPU within 4h (observed: a
4h filler sat pending, a 1h30 one started in 30 s; a 1h reload costs ~1 min). Holders therefore get at most 1h,
capped to the gap before the next projected real-job start on their node (`squeue %S`),
and every running holder re-checks pending siblings each minute: one that has waited
3 min next to an idle GPU with room for it gets its limit halved (floor 20 min); one
whose gap grew again is re-queued longer. Holders resume from checkpoints, so short
links cost only a reload. Priorities: `--nice=200` (fillers) and `300` (dry-runs) keep
them below the group's real jobs (~800) but inside backfill's `bf_max_job_test=5000`
window (the cluster queue holds ~6000 jobs; at nice 100000 they were never tested).

## Operating

```bash
for n in gpua046 gpua047; do scripts/slurm/resv_filler_submit.sh $n; scripts/slurm/resv_dryrun_submit.sh $n; done   # start
python scripts/slurm/resv_plan.py --status                                  # what would yield now
squeue -R sup-30781 -o "%i %T %j %N %l %r"                                  # queue in the reservation
tail -n 5 logs/filler_gpua04*_*.out                                          # filler logs
ls /work/hdd/bfqt/xiaocong/dagformer_sft/runs/*/DONE                        # finished tasks
scancel --signal=USR1 --batch <filler jobid>   # graceful: checkpoint, exit, replacement stays queued
touch ~/.stop_resv_keepalive; squeue -h -u $USER -o "%i %j" | grep -E "resv-(filler|dryrun)" | awk '{print $1}' | xargs -r scancel   # stop all
```

Add tasks by appending to `configs/sft/tasks.yaml` (fillers re-read it when they
claim); a finished task is skipped because of its `DONE` file. Evaluate an export with
`scripts/eval_lm_harness.py --config runs/<task>/final/config.yaml --ckpt runs/<task>/final/checkpoint.pt`.

## Caveats

- Holders only fill idle GPUs; they never touch GPUs that real jobs already hold.
- The pending real jobs of the owner (1 GPU / 16 CPU / 64 GB each) do not fit next to
  three of their siblings on one node (3×64 + 64 > 251 GB); the planner sees that and does
  not evict holders when evicting would not make the job fit.
- Need a GPU that a filler holds? Just submit your job: a real job pending for resources
  makes the planner evict the filler within ~1-2 min, after a checkpoint. Killing fillers by
  hand also cancels their queued replacements, so re-seed with the submit scripts afterwards.
- A yield loses nothing (checkpoint first). A plain `scancel` gives the trainer only SLURM's
  KillWait (30 s) to save, so it can lose up to `save_every` steps (100 steps ≈ 12 min for
  300M); `scancel --signal=USR1 --batch <id>` is the graceful way to stop one filler. During the first minutes of a brand-new dataset the GPU shows 0 %
  because the job is tokenizing on the CPU; that happens once per dataset.
- Verified live on 2026-09-26: smoke tasks trained/exported; three fillers killed mid-preparation
  were recovered via stale-lock reclaim; a real yield to a queued 1-GPU job completed in ~4 min
  (checkpoint at step 0, replacement queued, real job started); CPU test of interrupt/resume.
- Fillers read the code from the working tree of `/u/xiaocong/dagformer`; switching
  branches or deleting these files breaks the next link.

## DAGFormer 300M / 1B instruction tuning (owner priority, 2026-09-29)

- Order and recipe live in `configs/sft/tasks.yaml`: the three finished 300M runs (priority 10),
  then the 1B 10B pair (20), the 1B 5B pair (25), other explicit tasks (50), other discovered (60).
  `discover.large_model`: models >= 500M params get SmolTalk 1 epoch and `min_gpus: 4`, so 1-GPU
  fillers skip them. `discover.ddp_hold` keeps the 5B pair from being submitted before
  `1b_dense_10b/DONE` exists.
- Multi-GPU mode: `torchrun --nproc_per_node=4 scripts/sft/sft_train.py --only TASK`; identical
  global batch, data order and loss as 1 GPU (CPU check: bitwise-equal weights), resumable across
  GPU counts. `scripts/slurm/submit_sft_ddp.sh` queues `scripts/slurm/sft_ddp.slurm` (name
  `sft-ddp-<task>`, 4 GPUs, SLURM nice from priority) for every eligible 1B task; fillers call it at
  start and after each task.
- Second SFT seed (2026-10-01, for error bars): explicit tasks `pt_dagformer_300m_seed2-*` (seed 2, priority 30; benchmark group `pt_dagformer_300m_seed2`); the 1B 5B seed-2 pair is added after the 10B pair finishes.
- Benchmarks: `scripts/slurm/submit_sft_evals.sh` submits `scripts/sft/eval_sft_exports.sh` once per
  group (300M six; each 1B pair) as soon as the group is DONE.
- The 300M six run on timan108 (`/srv/local/xy51/dagformer_sft_timan`, launcher `run_gpu.sh`,
  logs `logs/gpuN.log`), locked on Delta as `ext:timan108`; `scripts/sft/sync_timan_runs.sh`
  (run by the GPU coordinator) copies finished runs back and removes the lock.

- External (off-Delta) runs: lock the Delta run dir with `LOCK/owner.json` = `{"job": "ext:<host>", ...}` (ext: owners are never stale). A lock without owner.json counts as stale and gets reclaimed by fillers (2026-09-30: 7 locality tasks were duplicated on Delta because hand-made locks wrote LOCK/LOCK). `scripts/sft/sync_timan_runs.sh` removes the lock when it copies the finished run back.
