"""Merge per-file pretokenize outputs into one mmap corpus dir (stream order, whole-window budget).

Companion to ``scripts/pretokenize.py --local-files``: tokenize each source file in its own
process (one out-dir per file), then merge the parts back into a single corpus that is
byte-for-byte equivalent to a sequential run except at file seams (each part drops the
<1 window tail of its last document, i.e. <= seq_len tokens per source file).

Shards are hard-linked (same filesystem) rather than copied; the shard that crosses the
budget is physically truncated so that sum(n_tokens) == total_tokens == the first
whole-window count >= --token-budget (same rounding as the sequential writer).

Usage:
    python scripts/merge_pretok_dirs.py --parts parts/books-0000 parts/books-0001 ... \
        --out-dir /path/corpus --token-budget 12000000000
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", required=True, help="part dirs (each with index.json) in stream order")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--token-budget", type=int, required=True)
    ap.add_argument("--copy", action="store_true", help="copy shards instead of hard-linking")
    a = ap.parse_args()

    parts = []
    for p in a.parts:
        with open(os.path.join(p, "index.json")) as f:
            parts.append(json.load(f))
    base = parts[0]
    window = int(base["seq_len"]) + 1
    for p, idx in zip(a.parts, parts):
        for k in ("format_version", "dtype", "seq_len", "eos_id", "tokenizer_id"):
            assert idx[k] == base[k], (p, k, idx[k], base[k])
        assert idx["total_tokens"] % window == 0, (p, idx["total_tokens"])
    target = -(-a.token_budget // window) * window  # ceil to whole windows

    os.makedirs(a.out_dir, exist_ok=True)
    assert not os.path.exists(os.path.join(a.out_dir, "index.json")), "out-dir already has an index.json"
    shards, total, k, used = [], 0, 0, []
    for pdir, idx in zip(a.parts, parts):
        for s in idx["shards"]:
            src = os.path.join(pdir, s["name"])
            n = int(s["n_tokens"])
            assert os.path.getsize(src) == n * 4, (src, os.path.getsize(src), n)
            assert n % window == 0, (src, n)
            take = min(n, target - total)
            if take <= 0:
                break
            name = f"shard_{k:05d}.bin"
            dst = os.path.join(a.out_dir, name)
            if os.path.lexists(dst):
                os.remove(dst)
            if take == n:
                if a.copy:
                    shutil.copyfile(src, dst)
                else:
                    os.link(src, dst)
            else:  # truncated final shard: copy the first `take` tokens
                with open(src, "rb") as fi, open(dst + ".tmp", "wb") as fo:
                    remaining = take * 4
                    while remaining:
                        chunk = fi.read(min(remaining, 64 << 20))
                        assert chunk, "short read"
                        fo.write(chunk)
                        remaining -= len(chunk)
                    fo.flush(); os.fsync(fo.fileno())
                os.replace(dst + ".tmp", dst)
            shards.append({"name": name, "n_tokens": take})
            total += take
            k += 1
            if total >= target:
                break
        used.append({"part_dir": os.path.abspath(pdir), "part_total_tokens": idx["total_tokens"],
                     "local_files": (idx.get("provenance") or {}).get("local_files")})
        if total >= target:
            break
    assert total == target, f"parts hold only {total:,} tokens < target {target:,}"

    index = {k2: base[k2] for k2 in ("format_version", "dtype", "seq_len", "eos_id", "tokenizer_id",
                                      "dataset", "dataset_version", "seed")}
    index.update({"total_tokens": total, "shards": shards, "created_utc": None,
                  "provenance": {**(base.get("provenance") or {}), "token_budget": a.token_budget,
                                 "merged_from": used}})
    path = os.path.join(a.out_dir, "index.json")
    with open(path + ".tmp", "w") as f:
        json.dump(index, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
    print(f"[merge] wrote {path}: {total:,} tokens in {len(shards)} shards from {len(used)} parts")


if __name__ == "__main__":
    main()
