"""Opportunistic instruction-tuning worker for the reservation filler jobs.

Runs inside one 1-GPU SLURM filler job. It repeatedly claims a task from the
task list (a pretrained checkpoint x an SFT dataset), prepares the dataset if
needed, trains with frequent checkpoints, exports the result, and moves on. It
can be interrupted at any moment (SIGUSR1 from the filler wrapper when a real
job needs the GPU, SIGTERM at the time limit): it finishes the current
optimizer step, saves a resumable checkpoint, releases the task lock and
exits 0. The next filler that claims the task resumes exactly where it stopped.

Coordination across fillers (up to 8 concurrent, on two nodes) goes through
the shared run root on /work:

    <run_root>/<task>/LOCK/owner.json  atomic mkdir lock; stale if the owner's
                                       SLURM job is no longer running
    <run_root>/<task>/ckpt.pt          resumable state (weights + optimizer + step)
    <run_root>/<task>/final/           export in the shared-model layout
    <run_root>/<task>/DONE             finished marker (json summary)
    <data_root>/<dataset>/             packed windows, built once under a lock

Exit codes: 0 = stopped cleanly or finished a batch of work, 3 = nothing left
to do (all tasks done or locked by live jobs), 1 = error.

    python scripts/sft/sft_train.py --tasks configs/sft/tasks.yaml --job-id $SLURM_JOB_ID
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from scripts.eval_lm_harness import load_dense, load_fourway  # noqa: E402
from scripts.prune_finetune import PruneModule  # noqa: E402
from scripts.sft import sft_data  # noqa: E402

STOP = {"flag": False, "why": ""}


def _on_signal(signum, _frame):
    STOP["flag"] = True
    STOP["why"] = f"signal {signum}"
    log(f"received signal {signum}: will checkpoint and exit after this step")


def log(msg: str) -> None:
    if not IS_MAIN:
        return
    print(f"[sft {time.strftime('%F %T')}] {msg}", flush=True)


# ─── Distributed (torchrun) ──────────────────────────────────────────────────
# One task can run on several GPUs of one node: `torchrun --nproc_per_node=N sft_train.py --only T`.
# Global batch, data order and the token-weighted loss are identical to the 1-GPU run (each rank
# takes every N-th micro-batch of the step), so ckpt.pt resumes interchangeably at 1 or N GPUs.
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_MAIN = RANK == 0


def dist_init() -> torch.device:
    if WORLD > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        from datetime import timedelta
        # rank 0 exports + samples a 1B model while the others wait in a barrier: allow an hour
        dist.init_process_group(backend, timeout=timedelta(minutes=60))
        if torch.cuda.is_available():
            torch.cuda.set_device(LOCAL_RANK)
    if torch.cuda.is_available():
        return torch.device("cuda", LOCAL_RANK if WORLD > 1 else torch.cuda.current_device())
    return torch.device("cpu")


def barrier() -> None:
    if WORLD > 1:
        dist.barrier()


def any_rank(flag: bool, device) -> bool:
    """True if the flag is set on any rank (keeps every rank's control flow in lockstep)."""
    if WORLD == 1:
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def atomic_torch_save(obj, path: str, **kw) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f, **kw)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ─── Task config ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    name: str
    dataset: str
    model_dir: str = ""              # shared-model layout: <dir>/config.yaml + <dir>/checkpoint.pt
    model_config: str = ""           # ...or an explicit training config yaml
    checkpoint: str = ""             # ...and checkpoint file (final step of a finished pretraining)
    source: str = "explicit"
    epochs: float = 1.0
    seq_len: int = 1024
    micro_batch: int = 16
    grad_accum: int = 4
    lr: float = 2e-4
    predictor_lr: float = 2e-4
    weight_decay: float = 0.1
    warmup_frac: float = 0.03
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    save_every: int = 100
    eval_every: int = 500
    max_steps: int = 0           # debug cap (0 = full schedule)
    seed: int = 1234
    priority: int = 50           # lower = claimed first (explicit default 50, discovered default 60)
    min_gpus: int = 1            # 1-GPU fillers skip tasks that need more (run them via torchrun jobs)

    @property
    def global_batch(self) -> int:
        return self.micro_batch * self.grad_accum

    @property
    def config_path(self) -> str:
        return self.model_config or os.path.join(self.model_dir, "config.yaml")

    @property
    def checkpoint_path(self) -> str:
        return self.checkpoint or os.path.join(self.model_dir, "checkpoint.pt")


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def approx_params(cfg: dict) -> int:
    h, L, inter, v = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["vocab_size"]
    return v * h + L * (4 * h * h + 3 * h * inter)


def size_defaults(cfg: dict) -> dict:
    """Per-GPU batch shape / lr by model size (A100-40GB, seq 1024, bf16 AdamW)."""
    n = approx_params(cfg)
    if n < 200e6:
        return {"micro_batch": 16, "grad_accum": 4, "lr": 2e-4, "predictor_lr": 2e-4}
    if n < 500e6:
        return {"micro_batch": 8, "grad_accum": 8, "lr": 2e-4, "predictor_lr": 2e-4}
    if n < 2e9:
        return {"micro_batch": 4, "grad_accum": 16, "lr": 1e-4, "predictor_lr": 1e-4}
    return {"micro_batch": 2, "grad_accum": 32, "lr": 5e-5, "predictor_lr": 5e-5}


def discover_finished_pretraining(disc: dict, defaults: dict, notes: list[str]) -> list[Task]:
    """One task per (finished pretraining run x dataset).

    A run is finished only when its save_dir holds the DONE marker written by
    scripts/slurm/reserved_train.slurm *and* checkpoint_step{total_steps}.pt exists.
    Intermediate checkpoints of a run that is still training are never used.
    Runs come from the pretraining queue files (config=... / fallback=... entries).
    """
    datasets: dict = disc.get("datasets", {"smol_smoltalk": 2, "alpaca_dolly": 3})
    runs: dict[str, tuple[float, str, dict]] = {}          # save_dir -> (done_mtime, cfg_path, cfg)
    for q in disc.get("queues", []):
        qp = _abs(q)
        if not os.path.exists(qp):
            notes.append(f"queue file missing: {q}")
            continue
        with open(qp) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for kv in line.split():
                    if not (kv.startswith("config=") or kv.startswith("fallback=")):
                        continue
                    cp = _abs(kv.split("=", 1)[1])
                    try:
                        with open(cp) as cf:
                            cfg = yaml.safe_load(cf)
                        save_dir, total = cfg["save_dir"], int(cfg["total_steps"])
                    except Exception as e:
                        notes.append(f"unreadable config {cp}: {e}")
                        continue
                    done, final = os.path.join(save_dir, "DONE"), os.path.join(save_dir, f"checkpoint_step{total}.pt")
                    if os.path.exists(done) and os.path.exists(final):
                        if save_dir not in runs:
                            runs[save_dir] = (os.path.getmtime(done), cp, cfg)
                    elif os.path.exists(done):
                        notes.append(f"{save_dir}: DONE but final checkpoint step{total} missing, skipped")
    for root in disc.get("dirs", []):                        # shared-model layout: final exports
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            d = os.path.join(root, sub)
            cp, ck = os.path.join(d, "config.yaml"), os.path.join(d, "checkpoint.pt")
            if os.path.exists(cp) and os.path.exists(ck) and d not in runs:
                with open(cp) as cf:
                    cfg = yaml.safe_load(cf)
                runs[d] = (os.path.getmtime(ck), cp, cfg)
    large = disc.get("large_model", {})          # e.g. {min_params: 5e8, min_gpus: 4, epochs: {smol_smoltalk: 1}}
    rules = disc.get("priorities", [])           # [{match: regex on task name, priority: int}], first match wins
    default_prio = int(disc.get("default_priority", 60))
    tasks: list[Task] = []
    for save_dir, (mtime, cp, cfg) in sorted(runs.items(), key=lambda kv: kv[1][0]):   # first finished, first tuned
        if cfg.get("model_type", "") == "muddformer":
            notes.append(f"{save_dir}: muddformer not supported by the SFT worker, skipped")
            continue
        total = int(cfg.get("total_steps", 0))
        ck = os.path.join(save_dir, f"checkpoint_step{total}.pt")
        if not os.path.exists(ck):
            ck = os.path.join(save_dir, "checkpoint.pt")
        base = f"{os.path.basename(os.path.dirname(save_dir.rstrip('/')))}-{os.path.basename(save_dir.rstrip('/'))}"
        is_large = large and approx_params(cfg) >= float(large.get("min_params", 5e8))
        for ds, epochs in datasets.items():
            name = f"pt_{base}_{ds}"
            prio = next((int(r["priority"]) for r in rules if re.search(r["match"], name)), default_prio)
            extra = {}
            if is_large:
                epochs = large.get("epochs", {}).get(ds, epochs)
                extra["min_gpus"] = int(large.get("min_gpus", 1))
                if "save_every" in large:        # a 1B ckpt.pt is ~7.7 GB (~30 s on /work); every 100 steps cost ~25 %
                    extra["save_every"] = int(large["save_every"])
            tasks.append(Task(**{**defaults, **size_defaults(cfg), "name": name, "dataset": ds,
                                 "model_config": cp, "checkpoint": ck, "epochs": float(epochs),
                                 "source": f"discovered:{save_dir}", "priority": prio, **extra}))
    return tasks


def load_tasks(path: str, notes: Optional[list[str]] = None) -> tuple[list[Task], dict]:
    """Explicit tasks (in file order) followed by discovered finished pretraining runs."""
    notes = notes if notes is not None else []
    with open(path) as f:
        doc = yaml.safe_load(f)
    defaults = doc.get("defaults", {})
    tasks = [Task(**{**defaults, **t}) for t in doc.get("tasks", [])]
    if doc.get("discover"):
        explicit_ckpts = {os.path.realpath(t.checkpoint_path) for t in tasks}
        for t in discover_finished_pretraining(doc["discover"], defaults, notes):
            if os.path.realpath(t.checkpoint_path) in explicit_ckpts and t.dataset in {x.dataset for x in tasks}:
                continue                                    # already listed explicitly
            tasks.append(t)
    names = [t.name for t in tasks]
    assert len(names) == len(set(names)), f"duplicate task names: {[n for n in names if names.count(n) > 1]}"
    tasks.sort(key=lambda t: t.priority)          # stable: file / discovery order within a priority
    return tasks, doc


# ─── Locks ───────────────────────────────────────────────────────────────────

def job_is_running(job_id: str) -> bool:
    if not job_id or job_id == "local":
        return False
    if str(job_id).startswith("ext:"):
        # task runs outside this cluster (e.g. "ext:timan108"); the coordinator removes the
        # lock when it syncs the finished run back, so it is never treated as stale here
        return True
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id, "-t", "R,CG,PD", "-o", "%i"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # squeue hiccup: be conservative, treat as running
        log(f"squeue failed ({e}); assuming job {job_id} alive")
        return True
    return job_id in out.split()


class Lock:
    """mkdir-based lock on a shared filesystem, owned by a SLURM job."""

    def __init__(self, path: str, job_id: str):
        self.path, self.job_id, self.held = path, job_id, False

    def _owner(self) -> Optional[dict]:
        try:
            with open(os.path.join(self.path, "owner.json")) as f:
                return json.load(f)
        except Exception:
            return None

    def try_acquire(self) -> bool:
        for _ in range(2):
            try:
                os.mkdir(self.path)
            except FileExistsError:
                owner = self._owner()
                oj = owner.get("job") if owner else None
                if oj == self.job_id:
                    pass                                   # ours (restart within the job)
                elif oj and job_is_running(oj):
                    return False                           # live owner
                elif owner is None and time.time() - os.path.getmtime(self.path) < 120:
                    return False                           # being created right now
                else:
                    log(f"reclaiming stale lock {self.path} (owner {owner})")
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
            self.touch()
            self.held = True
            return True
        return False

    def touch(self) -> None:
        if not IS_MAIN:
            return
        os.makedirs(self.path, exist_ok=True)       # survive a lock dir removed underneath us
        atomic_json({"job": self.job_id, "host": socket.gethostname(), "pid": os.getpid(),
                     "time": time.time(), "when": time.strftime("%F %T")},
                    os.path.join(self.path, "owner.json"))

    def release(self) -> None:
        if self.held:
            shutil.rmtree(self.path, ignore_errors=True)
            self.held = False


# ─── Data ────────────────────────────────────────────────────────────────────

def ensure_dataset(data_root: str, dataset: str, seq_len: int, job_id: str) -> str:
    d = os.path.join(data_root, dataset)
    meta = os.path.join(d, "meta.json")
    os.makedirs(data_root, exist_ok=True)
    while not os.path.exists(meta):
        if not IS_MAIN:                                   # multi-GPU: rank 0 builds, the others wait
            time.sleep(10)
            continue
        lock = Lock(d + ".PREPARING", job_id)
        if lock.try_acquire():
            try:
                if not os.path.exists(meta):
                    log(f"preparing dataset {dataset} -> {d}")
                    sft_data.build(dataset, d, seq_len, stop=lambda: STOP["flag"])
            finally:
                lock.release()
        else:
            log(f"dataset {dataset} is being prepared by another job; waiting")
            time.sleep(30)
            if STOP["flag"]:
                raise InterruptedError
    return d


def _prefetch(path: str, chunk: int = 64 << 20) -> float:
    """Read a file once, sequentially, so its pages sit in the node's page cache (shared by all ranks).
    The windows are memory-mapped and read in random order; on /work (Lustre on HDD) every cold page is a
    network round trip, which starved a 4xA100 1B SFT job to ~4k sup-tok/s (2026-09-30, ranks stuck in
    cl_sync_io_wait) until the files were cached. Costs ~20 s for the 1.8 GB SmolTalk arrays."""
    t0 = time.time()
    try:
        with open(path, "rb", buffering=0) as f:
            while f.read(chunk):
                pass
    except OSError:
        pass
    return time.time() - t0


class Windows:
    def __init__(self, d: str, split: str):
        paths = [os.path.join(d, f"{split}_{k}.npy") for k in ("ids", "mask")]
        size_gb = sum(os.path.getsize(p) for p in paths) / 2**30
        # Load into RAM when small enough (SmolTalk: 1.8 GB per process). Page-cache warming alone does not
        # last on /work: every 7.7 GB 1B ckpt.pt written through the Lustre client cache evicted the
        # dataset again and the job fell back to ~4k sup-tok/s (2026-09-30). SFT_DATA_IN_RAM_GB caps it.
        if size_gb <= float(os.environ.get("SFT_DATA_IN_RAM_GB", "8")):
            t0 = time.time()
            self.ids = np.load(paths[0])
            self.mask = np.load(paths[1])
            if split == "train":
                log(f"loaded {split} windows of {os.path.basename(d)} into RAM ({size_gb:.1f} GB, {time.time()-t0:.0f}s)")
        else:
            if os.environ.get("SFT_PREFETCH", "1") == "1":
                dt = sum(_prefetch(p) for p in paths)
                if dt > 2:
                    log(f"prefetched {split} windows of {os.path.basename(d)} into the page cache in {dt:.0f}s")
            self.ids = np.load(paths[0], mmap_mode="r")
            self.mask = np.load(paths[1], mmap_mode="r")
        self.n = len(self.ids)

    def batch(self, idx: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.sort(idx)
        ids = torch.from_numpy(self.ids[idx].astype(np.int64))
        mask = torch.from_numpy(self.mask[idx].astype(bool))
        x = ids[:, :-1]
        y = ids[:, 1:].clone()
        y[~mask[:, 1:]] = -100
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def order_for_step(step: int, n: int, gb: int, seed: int) -> np.ndarray:
    """Window indices of global step `step` (epoch-wise seeded permutations)."""
    per_epoch = n // gb
    epoch, k = divmod(step, per_epoch)
    perm = np.random.default_rng(seed + epoch).permutation(n)
    return perm[k * gb:(k + 1) * gb]


# ─── Model ───────────────────────────────────────────────────────────────────

def build_module(task: Task, device) -> tuple[PruneModule, dict]:
    with open(task.config_path) as f:
        model_cfg = yaml.safe_load(f)
    ckpt = task.checkpoint_path
    log(f"model config {task.config_path}, checkpoint {ckpt} (~{approx_params(model_cfg)/1e6:.0f}M params)")
    is_fourway = str(model_cfg.get("routing_mode", "")).startswith("fourway")
    if is_fourway:
        fourway, predictor = load_fourway(ckpt, model_cfg, device)
        module = PruneModule(model_cfg, fourway.olmo, fourway, predictor)
        for p in fourway.get_routing_parameters():
            p.requires_grad_(True)
        for p in predictor.parameters():
            p.requires_grad_(True)
    else:
        base = load_dense(ckpt, model_cfg, device)
        module = PruneModule(model_cfg, base, None, None)
    for p in module.base.parameters():
        p.requires_grad_(True)
    module.train()
    return module, model_cfg


def build_optimizer(module: PruneModule, task: Task) -> torch.optim.AdamW:
    groups = [{"params": [p for p in module.base.parameters() if p.requires_grad],
               "lr": task.lr, "weight_decay": task.weight_decay, "base_lr": task.lr}]
    if module.is_fourway:
        routing = list(module.fourway.get_routing_parameters())
        bias = [p for n, p in module.predictor.named_parameters() if "layer_biases" in n]
        other = [p for n, p in module.predictor.named_parameters() if "layer_biases" not in n] + routing
        groups.append({"params": other, "lr": task.predictor_lr, "weight_decay": task.weight_decay,
                       "base_lr": task.predictor_lr})
        if bias:
            groups.append({"params": bias, "lr": task.predictor_lr, "weight_decay": 0.0,
                           "base_lr": task.predictor_lr})
    for g in groups:
        log(f"optimizer group: {sum(p.numel() for p in g['params']):,} params lr={g['base_lr']} wd={g['weight_decay']}")
    return torch.optim.AdamW(groups, betas=(0.9, 0.95))


def lr_at(step: int, total: int, base: float, task: Task) -> float:
    warm = max(1, int(task.warmup_frac * total))
    if step < warm:
        return base * (step + 1) / warm
    frac = min(1.0, (step - warm) / max(1, total - warm))
    return base * (task.min_lr_ratio + (1 - task.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))


def rng_state() -> dict:
    st = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["cpu"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"])


def save_training_state(path: str, module: PruneModule, opt, step: int, task: Task, job_id: str) -> None:
    if not IS_MAIN:
        return
    st = {**model_state(module), "step": step, "task": asdict(task), "job": job_id,
          "host": socket.gethostname(), "time": time.time(), "rng": rng_state(),
          "optimizer_state_dict": opt.state_dict()}
    atomic_torch_save(st, path)


def load_training_state(path: str, module: PruneModule, opt) -> int:
    st = torch.load(path, map_location="cpu", weights_only=False)
    load_model_state(module, st)
    opt.load_state_dict(st["optimizer_state_dict"])
    set_rng_state(st["rng"])
    log(f"resumed at step {st['step']} (saved by job {st.get('job')} on {st.get('host')})")
    return int(st["step"])


def model_state(module: PruneModule) -> dict:
    if module.is_fourway:
        return {"model_state_dict": module.base.state_dict(),
                "predictor_state_dict": module.predictor.state_dict(),
                "routing_state_dict": {k: v for k, v in module.fourway.state_dict().items()
                                       if not k.startswith("olmo.")}}
    return {"model_state_dict": module.base.state_dict()}


def load_model_state(module: PruneModule, st: dict) -> None:
    module.base.load_state_dict(st["model_state_dict"])
    if module.is_fourway:
        module.predictor.load_state_dict(st["predictor_state_dict"])
        module.fourway.load_state_dict(st["routing_state_dict"], strict=False)


@torch.no_grad()
def evaluate(module: PruneModule, val: Windows, device, micro: int) -> float:
    module.eval()
    tot, n = 0.0, 0
    for s in range(0, val.n, micro):
        x, y = val.batch(np.arange(s, min(val.n, s + micro)), device)
        logits = module(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100, reduction="sum").item()
        n += int((y != -100).sum())
    module.train()
    return tot / max(n, 1)


@torch.no_grad()
def sample(module: PruneModule, tok, prompts: list[str], device, max_new: int = 64) -> list[str]:
    module.eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    outs = []
    for p in prompts:
        text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        for _ in range(max_new):
            nxt = module(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if nxt.item() in (im_end, tok.eos_token_id):
                break
        outs.append(tok.decode(ids[0, -(ids.size(1) - len(tok(text, add_special_tokens=False)["input_ids"])):].tolist()))
    module.train()
    return outs


# ─── One task ────────────────────────────────────────────────────────────────

def export_final(run_dir: str, module: PruneModule, model_cfg: dict, step: int) -> str:
    out = os.path.join(run_dir, "final")
    os.makedirs(out, exist_ok=True)
    st = model_state(module)
    if module.is_fourway:
        side = os.path.join(out, f"checkpoint_step{step}_model.pt")
        atomic_torch_save(st["model_state_dict"], side, _use_new_zipfile_serialization=False)
        # relative: the loader resolves it next to checkpoint.pt, so exports stay valid when copied
        ck = {"step": step, "model_state_path": os.path.basename(side),
              "predictor_state_dict": st["predictor_state_dict"],
              "routing_state_dict": st["routing_state_dict"]}
    else:
        ck = {"step": step, "model_state_dict": st["model_state_dict"]}
    atomic_torch_save(ck, os.path.join(out, "checkpoint.pt"))
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)
    return out


def run_task(task: Task, run_dir: str, data_root: str, job_id: str, device) -> str:
    """Train until done or interrupted. Returns 'done' | 'interrupted'."""
    os.makedirs(run_dir, exist_ok=True)
    d = ensure_dataset(data_root, task.dataset, task.seq_len, job_id)
    train, val = Windows(d, "train"), Windows(d, "val")
    gb = task.global_batch
    total = int(task.epochs * (train.n // gb))
    if task.max_steps:
        total = min(total, task.max_steps)
    log(f"task {task.name}: {train.n} windows x {task.epochs} epochs, global batch {gb} -> {total} steps")

    if task.grad_accum % WORLD:
        raise ValueError(f"grad_accum {task.grad_accum} not divisible by world size {WORLD}")
    module, model_cfg = build_module(task, device)
    opt = build_optimizer(module, task)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    step = 0
    if os.path.exists(ckpt_path):
        step = load_training_state(ckpt_path, module, opt)
    else:
        torch.manual_seed(task.seed)

    fwd = module
    if WORLD > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        fwd = DDP(module, device_ids=[LOCAL_RANK] if device.type == "cuda" else None,
                  find_unused_parameters=True)
    log_path = os.path.join(run_dir, "log.csv")
    if IS_MAIN and not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,lr,train_nll,grad_norm,val_nll,tok_per_s,job\n")

    def save() -> None:
        save_training_state(ckpt_path, module, opt, step, task, job_id)

    lock = Lock(os.path.join(run_dir, "LOCK"), job_id)   # already held by caller; used for touch()
    vocab = None
    t_last, tok_count = time.time(), 0
    while step < total and not any_rank(STOP["flag"], device):
        module.set_step(step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, g["base_lr"], task)
        idx = order_for_step(step, train.n, gb, task.seed)
        # exact token-weighted mean over the whole global batch
        n_tok = int(train.mask[np.sort(idx)][:, 1:].sum())
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        mine = list(range(RANK, task.grad_accum, WORLD))    # this rank's micro-batches of the global batch
        for j, m in enumerate(mine):
            x, y = train.batch(idx[m * task.micro_batch:(m + 1) * task.micro_batch], device)
            sync = contextlib.nullcontext() if (WORLD == 1 or j == len(mine) - 1) else fwd.no_sync()
            with sync:
                logits = fwd(x)
                vocab = vocab or logits.size(-1)
                loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), ignore_index=-100,
                                       reduction="sum")
                # DDP averages gradients over ranks; x WORLD turns that back into the global sum / n_tok
                (loss * WORLD / n_tok).backward()
            loss_sum += loss.item()
        if WORLD > 1:
            t = torch.tensor([loss_sum], device=device, dtype=torch.float64)
            dist.all_reduce(t)
            loss_sum = float(t.item())
        params = [p for g in opt.param_groups for p in g["params"]]
        if step == 0 or (step == int(os.environ.get("SFT_START_STEP_CHECK", "-1"))):
            # parameters without gradient: on 1 GPU AdamW skips them, under DDP
            # (find_unused_parameters) they may get zero grads; log them so the recipes can't diverge silently
            none = sum(p.grad is None for p in params)
            zero = sum(p.grad is not None and not bool(p.grad.any()) for p in params)
            log(f"{task.name}: first step: {len(params)} trainable tensors, {none} without grad, {zero} with all-zero grad"
                + ("" if none == 0 and zero == 0 else "  <-- check: some parameters are unused"))
        gn = torch.nn.utils.clip_grad_norm_(params, task.max_grad_norm)
        if not torch.isfinite(gn):
            log(f"non-finite grad norm at step {step}; skipping step")
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        step += 1
        tok_count += n_tok
        if (step % 10 == 0 or step == total) and IS_MAIN:
            dt = time.time() - t_last
            with open(log_path, "a") as f:
                f.write(f"{step},{opt.param_groups[0]['lr']:.3e},{loss_sum / n_tok:.4f},{gn.item():.3f},,"
                        f"{tok_count / dt:.0f},{job_id}\n")
            if step % 50 == 0:
                log(f"{task.name} step {step}/{total} nll {loss_sum / n_tok:.4f} gn {gn.item():.2f} "
                    f"{tok_count / dt:.0f} sup-tok/s lr {opt.param_groups[0]['lr']:.2e}")
            t_last, tok_count = time.time(), 0
        if step % task.eval_every == 0 and step < total:
            if IS_MAIN:
                v = evaluate(module, val, device, task.micro_batch)
                log(f"{task.name} step {step} val nll {v:.4f}")
                with open(log_path, "a") as f:
                    f.write(f"{step},,,,{v:.4f},,{job_id}\n")
            barrier()
        if step % task.save_every == 0 and step < total:
            save(); lock.touch()

    if step < total:
        save()
        barrier()
        log(f"{task.name} interrupted at step {step}/{total} ({STOP['why'] or 'another rank'}); checkpoint saved")
        return "interrupted"

    # finished (rank 0 evaluates and exports; the other ranks just wait)
    if not IS_MAIN:
        barrier()
        return "done"
    v = evaluate(module, val, device, task.micro_batch)
    log(f"{task.name} FINISHED {step} steps, final val nll {v:.4f}")
    out = export_final(run_dir, module, model_cfg, step)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(sft_data.TOKENIZER)
        prompts = ["What is the capital of France?",
                   "Write a haiku about the ocean.",
                   "Explain in two sentences why the sky is blue."]
        with open(os.path.join(run_dir, "samples.txt"), "w") as f:
            for p, o in zip(prompts, sample(module, tok, prompts, device)):
                f.write(f"### {p}\n{o}\n\n")
    except Exception as e:  # samples are a nicety only
        log(f"sampling failed: {e}")
    with open(log_path, "a") as f:
        f.write(f"{step},,,,{v:.4f},,{job_id}\n")
    atomic_json({"task": asdict(task), "steps": step, "val_nll": v, "export": out,
                 "finished": time.strftime("%F %T"), "job": job_id}, os.path.join(run_dir, "DONE"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)          # optimizer state no longer needed; keep /work lean
    barrier()
    return "done"


# ─── Main loop over tasks ────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=os.path.join(REPO, "configs/sft/tasks.yaml"))
    ap.add_argument("--run-root", default=os.environ.get("SFT_RUN_ROOT", "/work/hdd/bfqt/xiaocong/dagformer_sft/runs"))
    ap.add_argument("--data-root", default=os.environ.get("SFT_DATA_ROOT", "/work/hdd/bfqt/xiaocong/dagformer_sft/data"))
    ap.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--only", default="", help="run just this task name")
    ap.add_argument("--list", action="store_true", help="print the resolved task list with status and exit")
    ap.add_argument("--list-json", action="store_true", help="like --list, one JSON object per task")
    args = ap.parse_args()

    if args.list or args.list_json:
        notes: list[str] = []
        tasks, _ = load_tasks(args.tasks, notes)
        for t in tasks:
            rd = os.path.join(args.run_root, t.name)
            owner = Lock(os.path.join(rd, "LOCK"), "list")._owner() if os.path.isdir(os.path.join(rd, "LOCK")) else None
            st = ("DONE" if os.path.exists(os.path.join(rd, "DONE")) else
                  f"RUNNING@{owner.get('job')}" if owner and job_is_running(str(owner.get("job"))) else
                  "STALE-LOCK" if os.path.isdir(os.path.join(rd, "LOCK")) else
                  "RESUMABLE" if os.path.exists(os.path.join(rd, "ckpt.pt")) else "todo")
            if args.list_json:
                print(json.dumps({"name": t.name, "status": st, "min_gpus": t.min_gpus,
                                  "priority": t.priority, "dataset": t.dataset}))
                continue
            print(f"{st:9s} {t.name:48s} {t.dataset:14s} x{t.epochs:g}  mb{t.micro_batch}x{t.grad_accum} "
                  f"lr={t.lr:g} p{t.priority} g{t.min_gpus}  [{t.source}]")
        if not args.list_json:
            for n in notes:
                print("note:", n)
        return 0

    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    device = dist_init()
    torch.backends.cuda.matmul.allow_tf32 = True
    if WORLD > 1 and not args.only:
        raise SystemExit("multi-GPU mode needs --only TASK")
    os.makedirs(args.run_root, exist_ok=True)
    log(f"worker job {args.job_id} on {socket.gethostname()} device {device} world {WORLD}")

    did_work = False
    while not STOP["flag"]:
        tasks, _ = load_tasks(args.tasks)          # re-read + re-discover: new finished runs are picked up
        if args.only:
            tasks = [t for t in tasks if t.name == args.only]
            assert tasks, f"no task named {args.only}"
        log(f"{len(tasks)} tasks listed ({sum(t.source != 'explicit' for t in tasks)} discovered)")
        claimed = None
        for t in tasks:
            if t.min_gpus > WORLD:
                continue                                 # e.g. 1B tasks: run by 4-GPU torchrun jobs only
            run_dir = os.path.join(args.run_root, t.name)
            if os.path.exists(os.path.join(run_dir, "DONE")):
                continue
            os.makedirs(run_dir, exist_ok=True)
            lock = Lock(os.path.join(run_dir, "LOCK"), args.job_id)
            got = lock.try_acquire() if IS_MAIN else False
            if any_rank(got, device):                    # rank 0 decides; everyone follows
                claimed = (t, run_dir, lock)
                break
        if claimed is None:
            log("no claimable task left" + (" (all done or held by running jobs)"))
            return 0 if did_work else 3
        t, run_dir, lock = claimed
        log(f"claimed task {t.name}")
        try:
            status = run_task(t, run_dir, args.data_root, args.job_id, device)
            did_work = True
        except InterruptedError:
            status = "interrupted"
        finally:
            if IS_MAIN:
                lock.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if status == "interrupted" or WORLD > 1:
            break
    if WORLD > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
