"""Merge the worker directories of a parallel pretokenize run into one corpus.

    <root>/w0/index.json + shard_*.bin, <root>/w1/..., ...  ->  <root>/index.json

MmapPackedDataset joins shard names onto the index directory, so the merged
index simply lists "w0/shard_00000.bin", ... and no data is moved.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    workers = sorted(glob.glob(os.path.join(args.root, "w*", "index.json")))
    assert workers, f"no worker index.json under {args.root}"
    shards, total, first = [], 0, None
    for wi in workers:
        idx = json.load(open(wi))
        first = first or idx
        assert idx["seq_len"] == first["seq_len"] and idx["eos_id"] == first["eos_id"]
        wdir = os.path.basename(os.path.dirname(wi))
        for s in idx["shards"]:
            assert os.path.exists(os.path.join(args.root, wdir, s["name"]))
            shards.append({"name": f"{wdir}/{s['name']}", "n_tokens": int(s["n_tokens"])})
            total += int(s["n_tokens"])
    merged = {k: v for k, v in first.items() if k not in ("shards", "total_tokens", "provenance")}
    merged.update({"shards": shards, "total_tokens": total,
                   "provenance": {"merged_from": [os.path.dirname(w) for w in workers],
                                  "worker_provenance": first.get("provenance")}})
    out = os.path.join(args.root, "index.json")
    with open(out + ".tmp", "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(out + ".tmp", out)
    print(f"merged {len(workers)} workers, {len(shards)} shards, {total:,} tokens -> {out}")


if __name__ == "__main__":
    main()
