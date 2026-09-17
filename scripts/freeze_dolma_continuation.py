"""Freeze the nominal eight-rank Dolma suffix for the paired 600M continuations.

Uses the original streaming iterator, including its document selection and
packing behavior. Both models then consume the same frozen suffix without
network access. Historical HTTP-error paths cannot be reconstructed exactly.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pretokenize import write_index
from src.data.dolma import DolmaPackedDataset


class PinnedDolma(DolmaPackedDataset):
    def __init__(self, *args, revision: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.revision = revision

    def _load_stream(self):
        dataset = load_dataset(self.dataset_name, name=self.dataset_version,
                               revision=self.revision, split="train",
                               streaming=True, trust_remote_code=True)
        return dataset.shard(num_shards=self.world_size, index=self.rank)


def write_suffix(dataset, path: Path, samples: int, seq_len: int, rank: int) -> None:
    temporary = path.with_suffix(".bin.tmp")
    count = 0
    start = time.monotonic()
    with temporary.open("wb") as file:
        for row in dataset:
            window = np.empty(seq_len + 1, dtype=np.uint32)
            window[:-1] = row["olmo_ids"].numpy()
            window[-1] = int(row["olmo_labels"][-1])
            window.tofile(file)
            count += 1
            if count % 25000 == 0:
                print(f"Rank {rank}: saved {count:,}/{samples:,} suffix sequences; "
                      f"{(time.monotonic() - start) / 60:.1f} min", flush=True)
            if count == samples:
                break
        if count != samples:
            raise RuntimeError(f"Rank {rank}: source ended after {count}/{samples} samples")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def freeze_rank(task: dict) -> dict:
    rank = task["rank"]
    out = Path(task["out_dir"]) / "rank_cache"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rank_{rank:02d}.bin"
    marker = path.with_suffix(".json")
    samples = (task["target_updates"] - task["origin_updates"]) * 64
    size = samples * 1025 * 4
    if (path.exists() and marker.exists() and path.stat().st_size == size
            and json.loads(marker.read_text())["settings"] == task):
        return {"rank": rank, "path": str(path), "samples": samples}
    tokenizer = AutoTokenizer.from_pretrained(task["tokenizer"])
    dataset = PinnedDolma(olmo_tokenizer=tokenizer, seq_len=1024,
                          dataset_name="allenai/dolma", dataset_version="v1_7",
                          rank=rank, world_size=8, revision=task["revision"],
                          skip_samples=task["origin_updates"] * 64, max_samples=samples)
    write_suffix(dataset, path, samples, 1024, rank)
    marker.write_text(json.dumps({"settings": task, "samples": samples}, indent=2) + "\n")
    return {"rank": rank, "path": str(path), "samples": samples}


def merge_ranks(records: list[dict], target: Path, seq_len: int) -> int:
    """Arrange position p so the existing mmap loader assigns p % world_size to its rank."""
    records = sorted(records, key=lambda r: r["rank"])
    samples = records[0]["samples"]
    if any(r["samples"] != samples for r in records):
        raise ValueError("Rank lengths differ")
    arrays = [np.memmap(r["path"], mode="r", dtype=np.uint32,
                        shape=(samples, seq_len + 1)) for r in records]
    temporary = target.with_suffix(".bin.tmp")
    with temporary.open("wb") as file:
        for start in range(0, samples, 1024):
            # [local sample, rank, token]; no shuffle or padding between ranks.
            np.stack([a[start:start + 1024] for a in arrays], axis=1).tofile(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)
    return samples * len(records) * (seq_len + 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--origin-updates", type=int, default=14001)
    p.add_argument("--target-updates", type=int, default=22900)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info("allenai/dolma")
    task = {"out_dir": str(args.out_dir), "tokenizer": args.tokenizer,
            "origin_updates": args.origin_updates, "target_updates": args.target_updates,
            "revision": info.sha}
    (args.out_dir / "source.json").write_text(json.dumps({
        **task, "source_last_modified": str(info.last_modified),
    }, indent=2) + "\n")
    print(json.dumps(task, indent=2), flush=True)
    with ProcessPoolExecutor(max_workers=8,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(freeze_rank, [{**task, "rank": rank} for rank in range(8)]))
    target = args.out_dir / "continuation.bin"
    total = merge_ranks(records, target, 1024)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    provenance = {**task, "world_size": 8, "samples_per_update_per_rank": 64,
                  "rank_records": records,
                  "packing": "original DolmaPackedDataset, then deterministic rank interleave",
                  "training_loader": "num_workers=0, single block (no shuffle), per-rank offset relative to origin_updates",
                  "limitation": "Nominal restored stream suffix; historical HTTP errors make exact original document continuity unverifiable. Both continuations use the same frozen suffix.",
                  "versions": {p: importlib.metadata.version(p) for p in
                               ("datasets", "transformers", "tokenizers")}}
    write_index(str(args.out_dir), 1024, tokenizer.eos_token_id, args.tokenizer,
                "allenai/dolma", "v1_7", 42, total,
                [{"name": target.name, "n_tokens": total}], provenance)


if __name__ == "__main__":
    main()
