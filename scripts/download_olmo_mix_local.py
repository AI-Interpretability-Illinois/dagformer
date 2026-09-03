"""Download a fixed subset of olmo-mix-1124 to local disk.

Streaming with 8 ranks from HF blasts the resolver endpoint (5000 req/5min cap).
We pre-download a deterministic file set once, single-rank, with explicit
rate limiting + retries. Training then reads local files (zero HF requests).

Files come down to:
    /work/hdd/bfqt/data/olmo-mix-1124/<subset>/<original-relative-path>

Selection (deterministic via fixed shuffle seed):
    - dclm:            first 200 files (~120GB, ~26B tokens potential)
    - starcoder:       first 50 files
    - other 5 subsets: ALL files (each <2GB)

Run via SLURM (NOT login node):
    sbatch scripts/slurm_download_olmo_mix.sh
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("[boot] importing huggingface_hub...", flush=True)
from huggingface_hub import HfApi, hf_hub_download
print("[boot] imports done", flush=True)


REPO_ID = "allenai/olmo-mix-1124"
LOCAL_ROOT = Path("/work/hdd/bfqt/data/olmo-mix-1124")

SUFFIXES = {
    "dclm":            ".jsonl.zstd",
    "starcoder":       ".json.gz",
    "pes2o":           ".json.gz",
    "arxiv":           ".json.gz",
    "open-web-math":   ".json.gz",
    "algebraic-stack": ".json.gz",
    "wiki":            ".json.gz",
}

# Same caps as src/data/dolma.py::OLMO_MIX_MAX_FILES (matched seed → same files)
# For 20B-token training at natural mix proportions:
#   dclm:      19B tok needed @ ~5B tok/file → 4 files min; 200 caps ~60GB & ~1000B tok
#   starcoder: 425M tok @ ~110MB/file (~30M tok)    → 15 files; cap 50
#   pes2o:     300M tok @ ~3.8GB/file (~700M tok)   → 1 file; cap 2
#   arxiv:    107M tok @ ~1GB/file (~250M tok)     → 1 file; cap 3
#   ow-math:   63M tok @ ~1GB/file (~250M tok)     → 1 file; cap 3
#   alg-stack: 61M tok @ ~0.7GB/file (~180M tok)   → 1 file; cap 3
#   wiki:      19M tok @ ~3GB/file (~750M tok)     → 1 file; cap 2 (full)
CAPS = {
    "dclm":            200,
    "starcoder":        50,
    "pes2o":             2,
    "arxiv":             3,
    "open-web-math":     3,
    "algebraic-stack":   3,
    "wiki":              2,
}

SHUFFLE_SEED = 0          # MUST match src/data/dolma.py
INTER_FILE_SLEEP_S = 0.2  # be nice to HF resolver
MAX_RETRIES_PER_FILE = 10


def list_files(api: HfApi, subset: str) -> list[str]:
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"data/{subset}/"
    suffix = SUFFIXES[subset]
    files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(suffix))
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(files)
    cap = CAPS.get(subset)
    if cap is not None:
        files = files[:cap]
    return files


def download_one(rel_path: str, attempt: int = 0) -> Path | None:
    """Download one file with retries. Returns None on permanent failure."""
    try:
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            repo_type="dataset",
            local_dir=str(LOCAL_ROOT),
        )
        return Path(local)
    except Exception as e:
        if attempt + 1 >= MAX_RETRIES_PER_FILE:
            print(f"    PERMANENT FAIL {rel_path}: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            return None
        # 429 / connection errors: exponential backoff
        wait = min(60, 2 ** attempt)
        print(f"    retry {attempt+1}/{MAX_RETRIES_PER_FILE} after {wait}s "
              f"({type(e).__name__}: {str(e)[:80]})", flush=True)
        time.sleep(wait)
        return download_one(rel_path, attempt + 1)


def main():
    api = HfApi()
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    # smallest -> largest for early feedback
    for subset in ["wiki", "algebraic-stack", "open-web-math", "arxiv",
                   "pes2o", "starcoder", "dclm"]:
        print(f"\n[{subset}] listing...", flush=True)
        files = list_files(api, subset)
        print(f"[{subset}] {len(files)} files to download", flush=True)
        t0 = time.time()
        done = 0
        bytes_total = 0
        for i, rel in enumerate(files):
            local = LOCAL_ROOT / rel
            if local.exists() and local.stat().st_size > 0:
                done += 1
                bytes_total += local.stat().st_size
                continue
            t_a = time.time()
            p = download_one(rel)
            t_b = time.time()
            if p is None:
                continue
            sz = p.stat().st_size
            bytes_total += sz
            done += 1
            if i < 3 or i % 25 == 0:
                print(f"[{subset}] [{i+1}/{len(files)}] "
                      f"+{t_b-t_a:.1f}s  {sz/1e6:.1f}MB  -> {rel}",
                      flush=True)
            time.sleep(INTER_FILE_SLEEP_S)
        elapsed = time.time() - t0
        print(f"[{subset}] DONE {done}/{len(files)} files, "
              f"{bytes_total/1e9:.2f}GB in {elapsed:.1f}s "
              f"({bytes_total/1e6/max(1,elapsed):.1f} MB/s)", flush=True)

    print(f"\nTotal elapsed: {(time.time()-t_global)/60:.1f} min", flush=True)
    print("ALL DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
