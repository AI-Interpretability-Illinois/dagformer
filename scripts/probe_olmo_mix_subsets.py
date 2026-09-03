"""Per-subset probe of olmo-mix-1124.

Tests each of the 7 subsets in isolation to find which (if any) hangs.
For each subset: time load_dataset() + first 3 docs.

If a subset is responsible for the hang, this will show exactly which.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("[boot] importing datasets + hub...", flush=True)
import signal
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi
print("[boot] imports done", flush=True)

# Same patterns as src/data/dolma.py::OLMO_MIX_PATTERNS
PATTERNS = {
    "wiki":            ["data/wiki/**/*.json.gz"],
    "algebraic-stack": ["data/algebraic-stack/**/*.json.gz"],
    "open-web-math":   ["data/open-web-math/**/*.json.gz"],
    "arxiv":           ["data/arxiv/**/*.json.gz"],
    "pes2o":           ["data/pes2o/**/*.json.gz"],
    "starcoder":       ["data/starcoder/**/*.json.gz"],
    "dclm":            ["data/dclm/**/*.jsonl.zstd"],
}
FEATURES = Features({"text": Value("string")})

SUBSETS = list(PATTERNS.keys())
PER_SUBSET_TIMEOUT_S = 180


def timeout_handler(signum, frame):
    raise TimeoutError(f"Hit {PER_SUBSET_TIMEOUT_S}s per-subset cap")


_API = HfApi()
_FILE_CACHE: dict[str, list[str]] = {}


def list_subset_files(name: str, max_files: int | None = None) -> list[str]:
    """Pre-resolve concrete file URLs (avoids slow hf:// glob over 28K files for dclm)."""
    if name in _FILE_CACHE:
        files = _FILE_CACHE[name]
    else:
        all_files = _API.list_repo_files("allenai/olmo-mix-1124", repo_type="dataset")
        prefix = f"data/{name}/"
        suffix = ".jsonl.zstd" if name == "dclm" else ".json.gz"
        files = sorted([f for f in all_files if f.startswith(prefix) and f.endswith(suffix)])
        _FILE_CACHE[name] = files
    if max_files is not None:
        files = files[:max_files]
    return [f"hf://datasets/allenai/olmo-mix-1124/{p}" for p in files]


def probe(name):
    print(f"\n[{name}] start", flush=True)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(PER_SUBSET_TIMEOUT_S)
    try:
        t0 = time.time()
        # For dclm (28K files), limit the file list to 50 shards for the probe
        # — that's still ~5M docs, plenty to verify streaming works.
        max_files = 50 if name == "dclm" else None
        hf_files = list_subset_files(name, max_files=max_files)
        t_list = time.time()
        print(f"[{name}]   list_files ({len(hf_files)} files): +{t_list-t0:.1f}s", flush=True)
        ds = load_dataset(
            "json",
            data_files=hf_files,
            split="train",
            streaming=True,
        )
        t1 = time.time()
        print(f"[{name}]   load_dataset(json): +{t1-t_list:.1f}s", flush=True)
        # Project to text-only via map() to standardize across subsets
        ds = ds.map(lambda x: {"text": x.get("text", "") or ""},
                    remove_columns=[c for c in ds.column_names if c != "text"]
                    if ds.column_names else None)
        t2 = time.time()
        print(f"[{name}]   map: +{t2-t1:.1f}s", flush=True)
        it = iter(ds)
        t3 = time.time()
        print(f"[{name}]   iter: +{t3-t2:.1f}s", flush=True)
        for i in range(3):
            t_a = time.time()
            doc = next(it)
            t_b = time.time()
            tx = doc.get("text", "")
            print(f"[{name}]   doc[{i}]: +{t_b-t_a:.1f}s  len={len(tx)}  preview={repr(tx[:60])}",
                  flush=True)
        total = time.time() - t0
        print(f"[{name}] OK in {total:.1f}s", flush=True)
        return True
    except TimeoutError as e:
        print(f"[{name}] HANG (>{PER_SUBSET_TIMEOUT_S}s) -- {e}", flush=True)
        return False
    except Exception as e:
        print(f"[{name}] FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
        return False
    finally:
        signal.alarm(0)


def main():
    results = {}
    for s in SUBSETS:
        results[s] = probe(s)
    print("\n=== Summary ===", flush=True)
    for s, ok in results.items():
        print(f"  {s}: {'OK' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
