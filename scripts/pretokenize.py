"""Offline one-time tokenizer: stream a source, pack, and write uint32 shards.

This is the OFFLINE half of the mmap data pipeline that replaces train-time
HuggingFace streaming. Streaming at train time causes NCCL-timeout deaths from
silent network stalls and makes resume incorrect (the stream position cannot be
recovered exactly). Instead we tokenize+pack ONCE here, freeze the result to
disk as uint32 shard files plus an `index.json`, and let training read it via
`src/data/mmap_dataset.py` (zero network, zero per-step tokenization, exact
integer-offset resume).

Pipeline (matches CLAUDE.md §3.1.1 sequence packing):
    stream docs -> tokenize(add_special_tokens=False) -> extend buffer + EOS
    -> chunk buffer into windows of (seq_len + 1) tokens
    -> write flattened token stream to shard_*.bin (uint32) until token budget.

Token IDs use **uint32** (NOT uint16): OLMo-2 vocab_size = 100352 > 65535, so
uint16 would silently corrupt IDs 65536..100351.

Shards are written ATOMICALLY (write `.tmp`, fsync, os.replace) so a crash mid-
write never leaves a half-written shard that the reader would trust.

The mixture interleave order + proportions are FIXED by `--seed` and recorded in
`index.json` for provenance, so the training distribution is frozen/reproducible.

Usage (run as a SLURM job — NOT on the login node):
    python scripts/pretokenize.py \
        --dataset allenai/dolma --dataset-version olmo_mix \
        --out-dir /work/hdd/bfqt/data/pretok/olmo_mix_21b \
        --token-budget 21_000_000_000 --seq-len 1024 --seed 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from typing import Iterator, Optional

import numpy as np
from datasets import interleave_datasets, load_dataset
from transformers import AutoTokenizer

# Reuse the source-construction logic, mixture proportions, file lists, and
# retry constants from the existing streaming dataloader so the offline
# tokenizer produces exactly the same training distribution.
from src.data.dolma import (
    DOLMINO_MATH_PATTERNS,
    DOLMINO_MIX,
    MAX_RETRIES,
    OLMO_MIX,
    RETRY_WAIT,
    _list_olmo_mix_files,
)

# uint32 = 4 bytes/token. Default shard size 512M tokens -> ~2 GB per file.
DEFAULT_SHARD_TOKENS = 512_000_000

# numpy dtype used for on-disk token ids. MUST be uint32 (see module docstring).
SHARD_DTYPE = np.uint32


class LocalGzJsonlSource:
    """Re-iterable document source over local Dolma-format ``*.json.gz`` files.

    Used by ``--local-files`` to tokenize a mirrored subset of the corpus without
    HTTP streaming (e.g. when the CDN stops honouring range requests, which makes
    fsspec's HTTP reader fail). Yields the same ``{"text": ...}`` dicts as the HF
    streaming dataset, in file order, so ``packed_token_stream`` is unchanged.
    Deliberately re-iterable: the retry loop re-enters ``for doc in dataset`` and
    fast-forwards by document count.
    """

    def __init__(self, files: list[str]) -> None:
        self.files = list(files)

    def __iter__(self) -> Iterator[dict]:
        for fn in self.files:
            with gzip.open(fn, mode="rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield {"text": json.loads(line).get("text", "")}


def _iter_local_jsonl_text(path: str) -> Iterator[str]:
    """Yield non-empty 'text' strings from a local .json.gz or .jsonl.zstd file.

    Reads raw JSONL line-by-line and extracts the 'text' field directly, bypassing
    HF's Arrow-based json builder — which fails on olmo-mix dclm's heterogeneous
    nested `metadata` struct ("Couldn't cast struct<provenance> to {WARC...}").
    Malformed lines are skipped rather than crashing the whole pass.
    """
    if path.endswith(".zstd") or path.endswith(".zst"):
        import io
        import zstandard
        with open(path, "rb") as fh:
            # dclm .jsonl.zstd files are MULTI-FRAME (concatenated zstd frames);
            # read_across_frames=True is required or the reader stops/errors with
            # "Unknown frame descriptor" at the 2nd frame boundary.
            reader = zstandard.ZstdDecompressor().stream_reader(
                fh, read_across_frames=True
            )
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                t = doc.get("text")
                if t:
                    yield t
    else:  # .json.gz / .jsonl.gz — line-delimited JSON
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                t = doc.get("text")
                if t:
                    yield t


class _OlmoMixLocalSource:
    """Re-iterable source over the LOCAL olmo-mix-1124 mirror.

    Reads raw JSONL directly from the compressed shard files, bypassing HF's json
    builder (schema-inference failure on dclm's heterogeneous `metadata`). Subsets
    are emitted in natural-proportion order; the actual training mix is realized by
    (a) the mirror already being sized to natural proportions and (b) the
    block-shuffle in mmap_dataset.py at train time. Yields {"text": str}. Being a
    class with __iter__ (not a bare generator), it is re-iterable so the retry loop
    in packed_token_stream can restart it.
    """

    def __init__(self, mix_record: list[dict]) -> None:
        self.mix_record = mix_record
        for rec in mix_record:
            for p in rec["files"]:
                if str(p).startswith("hf://"):
                    raise RuntimeError(
                        f"olmo_mix local reader needs local files, got {p} — "
                        f"mirror incomplete for subset {rec['subset']}"
                    )

    def __iter__(self) -> Iterator[dict]:
        for rec in self.mix_record:
            for path in rec["files"]:
                # Skip a genuinely-corrupt file rather than aborting the whole run.
                try:
                    for text in _iter_local_jsonl_text(path):
                        yield {"text": text}
                except Exception as e:  # noqa: BLE001
                    print(f"[pretokenize] WARN skipping {path}: {e}", flush=True)
                    continue


def build_source_and_provenance(
    dataset: str,
    dataset_version: str,
    seed: int,
):
    """Construct the streaming HF dataset and a JSON-serializable provenance dict.

    Reuses `src.data.dolma`'s mixture proportions and file-list resolution so the
    offline tokenized corpus matches the online streaming distribution exactly.

    The mixture interleave ORDER and PROBABILITIES are fixed by `seed` (baked in
    below) and returned in `provenance` so `index.json` can record them.

    Args:
        dataset: HF dataset id (e.g. "allenai/dolma").
        dataset_version: "v1_7" | "olmo_mix" | "dolmino_mix".
        seed: fixed seed controlling per-subset file shuffling and mixture order.

    Returns:
        (hf_dataset, provenance_dict)
    """
    provenance: dict = {
        "dataset": dataset,
        "dataset_version": dataset_version,
        "seed": seed,
    }

    if dataset_version == "olmo_mix":
        # Stage-1 olmo-mix-1124 read from the LOCAL MIRROR file lists (or hf://
        # fallback resolved inside _list_olmo_mix_files). File lists are
        # deterministically shuffled by `seed`; mixture proportions are the
        # natural token-share weights renormalized to exactly 1.0.
        # Read the LOCAL mirror directly (see _OlmoMixLocalSource): HF's json
        # builder fails schema inference on dclm's heterogeneous `metadata` struct.
        _total = sum(OLMO_MIX.values())
        mix_record: list[dict] = []
        for name, weight in OLMO_MIX.items():
            files = _list_olmo_mix_files(name, seed=seed)
            mix_record.append(
                {"subset": name, "probability": weight / _total,
                 "n_files": len(files), "files": files}
            )
        dataset_obj = _OlmoMixLocalSource(mix_record)
        provenance["mixture"] = mix_record

    elif dataset_version == "dolmino_mix":
        subsets = []
        probs = []
        mix_record = []
        for name, weight in DOLMINO_MIX.items():
            if name == "math":
                ds = load_dataset(
                    dataset,
                    data_files=DOLMINO_MATH_PATTERNS,
                    split="train",
                    streaming=True,
                ).select_columns(["text"])
                files_used = list(DOLMINO_MATH_PATTERNS)
            else:
                ds = load_dataset(
                    dataset,
                    name=name,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                ).select_columns(["text"])
                files_used = None
            subsets.append(ds)
            probs.append(weight)
            mix_record.append(
                {"subset": name, "probability": weight, "files": files_used}
            )
        dataset_obj = interleave_datasets(
            subsets, probabilities=probs, stopping_strategy="all_exhausted", seed=seed
        )
        provenance["mixture"] = mix_record

    else:
        # Single-source HF streaming (e.g. Dolma v1_7).
        try:
            dataset_obj = load_dataset(
                dataset,
                name=dataset_version,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception:
            dataset_obj = load_dataset(
                dataset,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        provenance["mixture"] = None

    return dataset_obj, provenance


def packed_token_stream(
    tokenizer: AutoTokenizer,
    dataset,
    seq_len: int,
    eos_id: int,
) -> Iterator[np.ndarray]:
    """Yield flat uint32 numpy chunks of packed token ids, with robust retry.

    Packing: concatenate each doc's token ids, append EOS as a separator, then
    emit fixed windows of (seq_len + 1) tokens. Each window is exactly one
    training sample's [input | last-label] span; the reader re-derives
    olmo_ids = window[:seq_len], olmo_labels = window[1:seq_len+1].

    Transient HTTP / stream errors retry up to MAX_RETRIES times (RETRY_WAIT s
    between attempts), fast-forwarding past docs already consumed so a mid-stream
    network stall does not restart tokenization from the beginning.

    Yields:
        1-D np.ndarray[uint32] of length (seq_len + 1) per packed window.
    """
    window = seq_len + 1
    buffer: list[int] = []
    docs_consumed = 0
    retries = 0

    while retries <= MAX_RETRIES:
        try:
            docs_skipped_for_retry = 0
            for doc in dataset:
                # On retry, fast-forward past docs already folded into prior output.
                if docs_skipped_for_retry < docs_consumed:
                    docs_skipped_for_retry += 1
                    continue

                text = doc.get("text", "")
                if not text or not text.strip():
                    docs_consumed += 1
                    continue

                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                buffer.extend(ids)
                buffer.append(eos_id)
                docs_consumed += 1

                while len(buffer) >= window:
                    chunk = buffer[:window]
                    del buffer[:window]
                    arr = np.asarray(chunk, dtype=SHARD_DTYPE)
                    assert arr.shape == (window,), arr.shape
                    yield arr

            return  # stream exhausted normally

        except Exception as e:  # noqa: BLE001 — deliberately broad; never crash on transient error
            retries += 1
            if retries > MAX_RETRIES:
                raise RuntimeError(
                    f"pretokenize stream failed after {MAX_RETRIES} retries: {e}"
                ) from e
            print(
                f"[pretokenize] stream error (retry {retries}/{MAX_RETRIES}): {e}",
                flush=True,
            )
            print(
                f"[pretokenize] resuming from doc {docs_consumed} after {RETRY_WAIT}s...",
                flush=True,
            )
            time.sleep(RETRY_WAIT)
            # Re-fetch the stream; docs already consumed are skipped above. Do NOT
            # reset `buffer` — the partial-window tail is still valid and would be
            # lost otherwise (unlike the streaming loader, we cannot re-derive it).
            dataset, _ = _refetch_source_for_retry(dataset)


# Retry re-fetch is a no-op placeholder for single-iterable sources: HF streaming
# datasets are re-iterable, so re-entering the `for doc in dataset` loop restarts
# iteration and the `docs_consumed` fast-forward handles resumption. We keep a
# hook so this can be swapped for a fresh source construction if needed.
def _refetch_source_for_retry(dataset):
    return dataset, None


class ShardWriter:
    """Accumulate uint32 tokens and flush ATOMIC shard files up to a size cap.

    Each shard is written to `shard_XXXXX.bin.tmp`, fsync'd, then `os.replace`d to
    the final name so a crash mid-write never leaves a partial shard the reader
    would trust. Records (name, n_tokens) per shard for `index.json`.
    """

    def __init__(self, out_dir: str, shard_tokens: int) -> None:
        assert shard_tokens > 0, shard_tokens
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.shard_index = 0
        self.buf: list[np.ndarray] = []
        self.buf_len = 0
        self.shards: list[dict] = []
        self.total_tokens = 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, arr: np.ndarray) -> None:
        """Buffer one window; flush whenever the buffered token count hits the cap."""
        assert arr.dtype == SHARD_DTYPE, arr.dtype
        self.buf.append(arr)
        self.buf_len += arr.shape[0]
        self.total_tokens += arr.shape[0]
        # Windows are small (seq_len+1); flush when we cross the cap. Shards may
        # slightly exceed shard_tokens by up to one window — the reader treats the
        # concatenation as one flat array so exact per-shard boundaries don't matter.
        while self.buf_len >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        data = np.concatenate(self.buf) if len(self.buf) > 1 else self.buf[0]
        assert data.dtype == SHARD_DTYPE, data.dtype
        name = f"shard_{self.shard_index:05d}.bin"
        final_path = os.path.join(self.out_dir, name)
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        self.shards.append({"name": name, "n_tokens": int(data.shape[0])})
        print(
            f"[pretokenize] wrote {name} ({data.shape[0]:,} tokens, "
            f"total={self.total_tokens:,})",
            flush=True,
        )
        self.shard_index += 1
        self.buf = []
        self.buf_len = 0

    def close(self) -> None:
        """Flush any remaining buffered tokens as a final (possibly small) shard."""
        self._flush()


def write_index(
    out_dir: str,
    seq_len: int,
    eos_id: int,
    tokenizer_id: str,
    dataset: str,
    dataset_version: str,
    seed: int,
    total_tokens: int,
    shards: list[dict],
    provenance: dict,
) -> str:
    """Write index.json describing the shard set. Written atomically.

    created_utc is left null (no reliable wall-clock available per task spec).
    """
    index = {
        "format_version": 1,
        "dtype": "uint32",
        "seq_len": seq_len,
        "eos_id": int(eos_id),
        "tokenizer_id": tokenizer_id,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "seed": seed,
        "total_tokens": int(total_tokens),
        "shards": shards,
        "created_utc": None,
        "provenance": provenance,
    }
    path = os.path.join(out_dir, "index.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"[pretokenize] wrote {path} (total_tokens={total_tokens:,})", flush=True)
    return path


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "tokenizer must have an EOS token"
    vocab = getattr(tokenizer, "vocab_size", None)
    if vocab is not None:
        assert vocab <= np.iinfo(SHARD_DTYPE).max, (
            f"vocab_size {vocab} exceeds uint32 range — pick a wider dtype"
        )

    if args.local_files:
        dataset = LocalGzJsonlSource(args.local_files)
        provenance = {
            "dataset": args.dataset,
            "dataset_version": args.dataset_version,
            "seed": args.seed,
            "mixture": None,
            "local_files": [os.path.abspath(f) for f in args.local_files],
        }
    else:
        dataset, provenance = build_source_and_provenance(
            args.dataset, args.dataset_version, args.seed
        )
    provenance["tokenizer_id"] = args.tokenizer_id
    provenance["token_budget"] = int(args.token_budget)
    provenance["seq_len"] = args.seq_len

    writer = ShardWriter(args.out_dir, args.shard_tokens)

    budget = int(args.token_budget)
    t0 = time.time()
    last_log = t0
    tokens_at_last_log = 0

    for arr in packed_token_stream(tokenizer, dataset, args.seq_len, eos_id):
        writer.add(arr)
        if writer.total_tokens >= budget:
            break

        now = time.time()
        if now - last_log >= 30.0:
            dt = now - last_log
            rate = (writer.total_tokens - tokens_at_last_log) / max(dt, 1e-9)
            pct = 100.0 * writer.total_tokens / budget
            print(
                f"[pretokenize] {writer.total_tokens:,}/{budget:,} tokens "
                f"({pct:.2f}%) | {writer.shard_index} shards | {rate/1e6:.2f} Mtok/s",
                flush=True,
            )
            last_log = now
            tokens_at_last_log = writer.total_tokens

    writer.close()
    write_index(
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        eos_id=eos_id,
        tokenizer_id=args.tokenizer_id,
        dataset=args.dataset,
        dataset_version=args.dataset_version,
        seed=args.seed,
        total_tokens=writer.total_tokens,
        shards=writer.shards,
        provenance=provenance,
    )
    elapsed = time.time() - t0
    print(
        f"[pretokenize] DONE: {writer.total_tokens:,} tokens in "
        f"{len(writer.shards)} shards ({elapsed/60:.1f} min)",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline pretokenizer -> uint32 shards + index.json")
    p.add_argument("--dataset", default="allenai/dolma")
    p.add_argument(
        "--dataset-version",
        default="v1_7",
        help="v1_7 | olmo_mix | dolmino_mix",
    )
    p.add_argument("--out-dir", required=True, help="output dir under /work/hdd")
    p.add_argument("--token-budget", type=int, required=True, help="stop after this many tokens")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--tokenizer-id", default="allenai/OLMo-2-0425-1B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--local-files",
        nargs="+",
        default=None,
        help="tokenize these local Dolma *.json.gz files (in this order) instead of HF streaming",
    )
    p.add_argument(
        "--shard-tokens",
        type=int,
        default=DEFAULT_SHARD_TOKENS,
        help="tokens per shard file (default 512M -> ~2GB uint32)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
