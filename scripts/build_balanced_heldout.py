#!/usr/bin/env python3
"""D3 (post hoc, 2026-10-04): a source-balanced Dolma v1.7 held-out set built only from files that NO training
tokenization consumed, with the same number of 1024-token windows per Dolma source.

Why: every training-time held-out cache is the tail of its own corpus, and the 21B corpus's tail is flan/wiki-only
(experiments/scaling/README.md), so the paper's cross-corpus numbers come from WikiText-2 / MathInstruct / GSM8K.

Which files were consumed (all corpora use the HF stream order of `urls/v1_7.txt`, 2419 files, seed 0):
  * 21B corpus: 8 workers, each `IterableDataset.shard(num_shards=8, index=i)` (contiguous chunks of the file list,
    datasets 3.6.0), each consuming a PREFIX of its chunk worth 2.625B tokens (a few files of 1-3 GB).
  * 12B' (timan rebuild): the first 26 files (books-0000..2, c4-0000..22), listed in its index.json provenance.
  * the original 12B slice and the 1.7B locality slice: single-stream prefixes of the same list.
Rule: for every worker chunk, and for the whole list (the 12B prefixes), files are excluded from the start until the
cumulative compressed size covers the consumed token budget at a conservative 0.2 tokens/byte (measured: 0.23-0.32)
times a 1.5 safety margin; files whose size the server does not report count as small (0.5 GB), which excludes more.
Each source's windows then come from the median remaining file of that source; sources with no remaining file are
reported and skipped.

    python scripts/build_balanced_heldout.py --out /work/hdd/bfqt/xiaocong/dagformer_pruning/data/dolma_balanced \
        --consumed-index /work/hdd/biro/xiaocong/pretok/dolma_v1_7_12b/index.json --windows-per-source 64

Output: <out>/eval_cache.pt in the format of scripts/split_eval_from_mmap.py (batches of {olmo_ids, olmo_labels,
raw_text}), batches ordered by source, plus <out>/manifest.json (sources, files, rules, counts).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import os
import re
import time

import numpy as np
import requests
import torch
import zstandard
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

SEQ_LEN = 1024
EOS = 100257
UNKNOWN_SIZE = 500_000_000


def source_of(url: str) -> str:
    return re.sub(r"-\d+\.json\.gz$", "", os.path.basename(url))


def contiguous_chunks(n: int, k: int) -> list[range]:
    """datasets' contiguous shard split: the first n % k chunks get one extra file."""
    div, mod = divmod(n, k)
    out, start = [], 0
    for j in range(k):
        size = div + (1 if j < mod else 0)
        out.append(range(start, start + size))
        start += size
    return out


def consumed_from_index(path: str) -> set[str]:
    prov = json.load(open(path)).get("provenance", {}) or {}
    files = {os.path.basename(f) for f in prov.get("local_files") or []}
    for part in prov.get("merged_from") or []:
        files |= {os.path.basename(f) for f in part.get("local_files") or []}
    return files


def remote_size(url: str) -> int | None:
    """Compressed size via a byte-range request (206 + Content-Range); a few answers are transient, so retry."""
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}, timeout=60) as r:
                cr = r.headers.get("Content-Range", "")
                if r.status_code == 206 and "/" in cr:
                    return int(cr.rsplit("/", 1)[1])
                cl = r.headers.get("Content-Length")
                if r.status_code == 200 and cl and r.headers.get("Content-Encoding", "") in ("", "identity"):
                    return int(cl)
        except requests.RequestException:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


def excluded_prefix(positions: range, sizes: dict[int, int | None], budget_tokens: float, tok_per_byte: float, margin: float) -> list[int]:
    need_bytes = budget_tokens / tok_per_byte * margin
    out, acc = [], 0
    for i in positions:
        out.append(i)
        acc += sizes.get(i) or UNKNOWN_SIZE
        if acc >= need_bytes:
            break
    return out


def stream_docs(url: str, max_bytes: int):
    """Yield document texts from the (gzip or zstd) JSONL at `url`, reading at most ~max_bytes compressed bytes."""
    # The server applies transfer-level zstd (Content-Encoding) on top of the stored .gz unless asked not to.
    with requests.get(url, stream=True, timeout=120, headers={"Accept-Encoding": "identity"}) as r:
        r.raise_for_status()
        raw = r.raw
        raw.decode_content = False
        enc = r.headers.get("Content-Encoding", "").lower()
        stream = zstandard.ZstdDecompressor().stream_reader(raw) if enc == "zstd" else raw
        if enc in ("gzip", "deflate", "br"):
            raw.decode_content = True
        head = stream.read(4)
        rest = io.BufferedReader(_Chained(head, stream), buffer_size=1 << 20)
        if head[:2] == b"\x1f\x8b":
            dec = gzip.GzipFile(fileobj=rest)
        elif head == b"\x28\xb5\x2f\xfd":
            dec = zstandard.ZstdDecompressor().stream_reader(rest)
        else:
            raise RuntimeError(f"unknown compression magic {head.hex()} for {url}")
        read = 0
        for line in io.TextIOWrapper(dec, encoding="utf-8", errors="replace"):
            read += len(line)  # decompressed bytes; compressed is ~3-4x smaller, so this is a conservative cap
            if line.strip():
                try:
                    yield json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    pass
            if read > max_bytes * 4:
                return


class _Chained(io.RawIOBase):
    """Prepend already-read header bytes to a raw stream."""
    def __init__(self, head: bytes, raw):
        self.head, self.raw = head, raw
    def readable(self): return True
    def readinto(self, b):
        if self.head:
            n = min(len(b), len(self.head)); b[:n] = self.head[:n]; self.head = self.head[n:]; return n
        data = self.raw.read(len(b))
        b[:len(data)] = data
        return len(data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="/work/hdd/bfqt/shared/dagformer-models/tokenizer")
    p.add_argument("--consumed-index", action="append", default=[], help="index.json whose provenance lists consumed files")
    p.add_argument("--num-workers-21b", type=int, default=8)
    p.add_argument("--budget-21b", type=float, default=2.625e9, help="tokens consumed per 21B worker (prefix of its chunk)")
    p.add_argument("--budget-prefix", type=float, default=12e9, help="tokens of the longest single-stream prefix corpus (12B)")
    p.add_argument("--tok-per-byte", type=float, default=0.2, help="conservative tokens per compressed byte (measured 0.23-0.32)")
    p.add_argument("--margin", type=float, default=1.5)
    p.add_argument("--windows-per-source", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-mb-per-file", type=int, default=64, help="compressed bytes to read per file at most")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    urls = [l.strip() for l in open(hf_hub_download("allenai/dolma", "urls/v1_7.txt", repo_type="dataset")) if l.strip()]
    n = len(urls)
    chunks = contiguous_chunks(n, args.num_workers_21b)
    # sizes only where the walk may need them: the start of every chunk and the global prefix
    sizes: dict[int, int | None] = {}
    probe = set(range(0, 60))
    for ch in chunks:
        probe |= set(ch[:25])
    t0 = time.time()
    for i in sorted(probe):
        sizes[i] = remote_size(urls[i])
    print(f"probed {len(probe)} file sizes in {time.time()-t0:.0f}s ({sum(v is None for v in sizes.values())} unknown)", flush=True)
    excluded: dict[int, str] = {}
    for w, ch in enumerate(chunks):
        for i in excluded_prefix(ch, sizes, args.budget_21b, args.tok_per_byte, args.margin):
            excluded[i] = f"21B worker {w} prefix"
    for i in excluded_prefix(range(n), sizes, args.budget_prefix, args.tok_per_byte, args.margin):
        excluded.setdefault(i, "12B stream prefix")
    mirror = set()
    for ix in args.consumed_index:
        mirror |= consumed_from_index(ix)
    for i, u in enumerate(urls):
        if os.path.basename(u) in mirror:
            excluded.setdefault(i, "12B' mirror file")

    by_src: dict[str, list[int]] = collections.OrderedDict()
    for i, u in enumerate(urls):
        by_src.setdefault(source_of(u), []).append(i)
    chosen, skipped = {}, {}
    for src, idxs in by_src.items():
        ok = [i for i in idxs if i not in excluded]
        if not ok:
            skipped[src] = {"n_files": len(idxs), "reason": "every file consumed or inside an excluded prefix"}
            continue
        chosen[src] = ok[len(ok) // 2]
    print(f"{n} files, {len(by_src)} sources, {len(excluded)} excluded positions ({len(mirror)} mirror files); "
          f"{len(chosen)} sources usable, skipped: {list(skipped)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    window = SEQ_LEN + 1
    need = args.windows_per_source * window
    batches, manifest_sources = [], []
    for src, i in chosen.items():
        url = urls[i]
        t0 = time.time()
        # one window per long document (its first seq_len+1 tokens); short documents are packed with EOS
        # separators as in training, so every source contributes many distinct documents
        wins: list[list[int]] = []
        buf: list[int] = []
        n_docs = n_long = 0
        try:
            for text in stream_docs(url, args.max_mb_per_file << 20):
                t = tok(text, add_special_tokens=False)["input_ids"] + [EOS]
                n_docs += 1
                if len(t) >= window:
                    wins.append(t[:window]); n_long += 1
                else:
                    buf.extend(t)
                    while len(buf) >= window:
                        wins.append(buf[:window]); buf = buf[window:]
                if len(wins) >= args.windows_per_source:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"  {src}: read error {e!r}; skipping", flush=True)
            skipped[src] = {"n_files": len(by_src[src]), "reason": f"read error: {e!r}"}
            continue
        if len(wins) < args.windows_per_source:
            print(f"  {src}: only {len(wins)} windows from {url} ({n_docs} docs); skipping", flush=True)
            skipped[src] = {"n_files": len(by_src[src]), "reason": f"file gave only {len(wins)} windows"}
            continue
        arr = np.asarray(wins[:args.windows_per_source], dtype=np.int64)
        samples = [{"olmo_ids": torch.from_numpy(w[:SEQ_LEN].copy()), "olmo_labels": torch.from_numpy(w[1:].copy())} for w in arr]
        first_batch = len(batches)
        for j in range(0, len(samples), args.batch_size):
            items = samples[j:j + args.batch_size]
            batches.append({"olmo_ids": torch.stack([s["olmo_ids"] for s in items]),
                            "olmo_labels": torch.stack([s["olmo_labels"] for s in items]),
                            "raw_text": [""] * len(items)})
        manifest_sources.append({"source": src, "file_index": i, "url": url, "n_docs_read": n_docs, "n_long_docs": n_long,
                                 "n_windows": args.windows_per_source, "batches": [first_batch, len(batches)]})
        print(f"  {src}: {os.path.basename(url)} (#{i}), {n_docs} docs ({n_long} long) -> {args.windows_per_source} windows in {time.time()-t0:.0f}s", flush=True)

    out = os.path.join(args.out, "eval_cache.pt")
    torch.save(batches, out + ".tmp")
    os.replace(out + ".tmp", out)
    manifest = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url_list": "allenai/dolma urls/v1_7.txt",
                "n_files": n, "num_workers_21b": args.num_workers_21b, "budget_21b": args.budget_21b, "budget_prefix": args.budget_prefix,
                "tok_per_byte": args.tok_per_byte, "margin": args.margin,
                "excluded": {str(i): {"file": os.path.basename(urls[i]), "why": why} for i, why in sorted(excluded.items())},
                "mirror_files_excluded": sorted(mirror), "windows_per_source": args.windows_per_source, "seq_len": SEQ_LEN,
                "batch_size": args.batch_size, "tokenizer": args.tokenizer, "n_batches": len(batches),
                "n_windows": sum(s["n_windows"] for s in manifest_sources), "sources": manifest_sources, "skipped": skipped}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {out}: {len(batches)} batches, {manifest['n_windows']} windows from {len(manifest_sources)} sources")


if __name__ == "__main__":
    main()
