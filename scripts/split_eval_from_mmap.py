"""Carve a held-out eval cache out of the tail of a pretokenized mmap corpus.

The trainer expects ``<save_dir>/eval_cache.pt`` (list of collated batches);
building it by streaming Dolma takes ~30 min per run. Instead take the LAST
``--n-windows`` windows of the last shard, write them as the eval cache, and
truncate that shard (and index.json) so training never sees them.

Usage:
    python scripts/split_eval_from_mmap.py --index-dir <dir> --n-windows 200 \
        --out <dir>/eval_cache.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", required=True)
    p.add_argument("--n-windows", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    idx_path = os.path.join(args.index_dir, "index.json")
    index = json.load(open(idx_path))
    assert not index.get("eval_split_off"), "eval windows were already split off this corpus"
    seq_len = int(index["seq_len"])
    window = seq_len + 1
    need = args.n_windows * window
    last = index["shards"][-1]
    path = os.path.join(args.index_dir, last["name"])
    n_tok = int(last["n_tokens"])
    assert n_tok > need + window, f"last shard too small ({n_tok} tokens) for {args.n_windows} windows"
    # keep the shard a whole number of windows so sample boundaries do not shift
    keep = ((n_tok - need) // window) * window
    arr = np.memmap(path, dtype=np.uint32, mode="r")
    tail = np.array(arr[keep:keep + need]).reshape(args.n_windows, window).astype(np.int64)
    samples = [{"olmo_ids": torch.from_numpy(w[:seq_len]), "olmo_labels": torch.from_numpy(w[1:]),
                "raw_text": ""} for w in tail]
    batches = []
    for i in range(0, len(samples), args.batch_size):
        items = samples[i:i + args.batch_size]
        batches.append({"olmo_ids": torch.stack([s["olmo_ids"] for s in items]),
                        "olmo_labels": torch.stack([s["olmo_labels"] for s in items]),
                        "raw_text": [""] * len(items)})
    torch.save(batches, args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)

    # truncate the shard atomically, then the index
    head = np.array(arr[:keep])
    del arr
    tmp = path + ".tmp"
    head.astype(np.uint32).tofile(tmp)
    os.replace(tmp, path)
    removed = n_tok - keep
    last["n_tokens"] = int(keep)
    index["total_tokens"] = int(index["total_tokens"]) - removed
    index["eval_split_off"] = {"n_windows": args.n_windows, "eval_cache": os.path.abspath(args.out),
                               "tokens_removed": int(removed)}
    with open(idx_path + ".tmp", "w") as f:
        json.dump(index, f, indent=2)
    os.replace(idx_path + ".tmp", idx_path)
    print(f"eval: {args.n_windows} windows -> {args.out}; train shard {last['name']} now {keep:,} tokens "
          f"(removed {removed:,}); total_tokens={index['total_tokens']:,}")


if __name__ == "__main__":
    main()
