"""Leave out complete documents from every Dolma tokenization worker.

The input token files are read-only. A new index shortens the final shard of
each worker at EOS, and evaluation windows are drawn only from the excluded
suffix. This split is held out for the new six-axis runs, not for old models
that used the original index. Worker strata approximate the original token mix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def build_split(source_index, output, reserve_tokens=2_000_000,
                windows_per_worker=64, seed=730):
    source_index, output = Path(source_index), Path(output)
    if source_index.is_dir():
        source_index = source_index / "index.json"
    original = json.loads(source_index.read_text())
    length, eos = original["seq_len"], original["eos_id"]
    groups = {}
    for record in original["shards"]:
        worker = Path(record["name"]).parts[0]
        groups.setdefault(worker, []).append(record)
    if len(groups) != 8:
        raise ValueError(f"Expected the eight-worker corpus, found {list(groups)}")
    rng = np.random.default_rng(seed)
    shards, eval_batches, monitor_batches, strata = [], [], [], []
    for worker, records in groups.items():
        for record in records[:-1]:
            shards.append({**record, "name": str((source_index.parent / record["name"]).resolve())})
        last = records[-1]
        path = (source_index.parent / last["name"]).resolve()
        n = last["n_tokens"]
        cut = n - reserve_tokens
        if cut < 1:
            raise ValueError(f"Final {worker} shard too small: {n}")
        mmap = np.memmap(path, dtype=np.uint32, mode="r", shape=(n,))
        # End training immediately after an EOS before the reserved suffix.
        end = cut
        while end > 0:
            start = max(0, end - 1_000_000)
            hits = np.flatnonzero(mmap[start:end] == eos)
            if hits.size:
                cut = start + int(hits[-1]) + 1
                break
            end = start
        else:
            raise ValueError(f"No document boundary in final {worker} shard")
        shards.append({"name": str(path), "n_tokens": cut})
        n_windows = (n - cut) // (length + 1)
        selected = sorted(rng.choice(n_windows, windows_per_worker, replace=False).tolist())
        offsets = []
        for i, index in enumerate(selected):
            offset = cut + index * (length + 1)
            tokens = torch.from_numpy(np.array(mmap[offset:offset + length + 1], dtype=np.int64))
            batch = {"olmo_ids": tokens[:-1].unsqueeze(0),
                     "olmo_labels": tokens[1:].unsqueeze(0)}
            eval_batches.append(batch)
            if i % 4 == 0:
                monitor_batches.append(batch)
            offsets.append(offset)
        strata.append({"worker": worker, "final_shard": last["name"],
                       "original_tokens": n, "training_prefix_tokens": cut,
                       "excluded_tokens": n - cut, "eval_window_offsets": offsets})
        del mmap
    index = {**original, "shards": shards, "total_tokens": sum(s["n_tokens"] for s in shards)}
    index.pop("eval_split_off", None)
    index["provenance"] = {"source_index": str(source_index.resolve()),
                           "split": "eight_worker_document_suffix_holdout", "seed": seed}
    summary = {"source_index": str(source_index.resolve()), "seed": seed,
               "seq_len": length, "eos_id": eos, "strata": strata,
               "training_tokens": index["total_tokens"], "eval_windows": len(eval_batches),
               "monitor_windows": len(monitor_batches),
               "scope": "Held out from new split only; original-index checkpoints may have consumed these documents."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(eval_batches, output / "dolma_heldout.pt")
    torch.save(monitor_batches, output / "monitor.pt")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    summary = build_split(args.source_index, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "strata"}, indent=2))


if __name__ == "__main__":
    main()
