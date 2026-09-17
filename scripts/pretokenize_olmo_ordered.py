"""Rebuild the historical OLMo local-file prefix with ordered parallel workers.

Files may finish tokenizing in any order. Packing always concatenates them in
the original subset/file order, with EOS after each nonblank document. This
preserves the serial pretokenizer's stream and its window-aligned shard sizes.
The historical reader does not interleave subsets by mixture probabilities.
Its retry counter also selects alternating documents on the initial pass; both
file-local parities are cached so global document parity survives file boundaries.
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
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pretokenize import _iter_local_jsonl_text, write_index  # noqa: E402

_TOKENIZER = None
_TOKENIZER_ID = None


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def init_worker(tokenizer_id: str) -> None:
    global _TOKENIZER, _TOKENIZER_ID
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER_ID = tokenizer_id
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_id)
    if _TOKENIZER.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")


def tokenize_file(task: tuple[str, str, int]) -> dict:
    source_name, target_name, batch_size = task
    source, target = Path(source_name), Path(target_name)
    meta_path = target.with_suffix(".json")
    identity = {"source": str(source), "source_bytes": source.stat().st_size,
                "source_mtime_ns": source.stat().st_mtime_ns,
                "tokenizer": _TOKENIZER_ID, "eos_id": _TOKENIZER.eos_token_id,
                "document_selection": "historical_alternating_documents_v1"}
    targets = [target.with_suffix(f".p{p}.bin") for p in range(2)]
    if meta_path.exists() and all(path.exists() for path in targets):
        old = json.loads(meta_path.read_text())
        if (old["identity"] == identity
                and all(targets[p].stat().st_size == old["tokens"][p] * 4
                        for p in range(2))):
            return old
    target.parent.mkdir(parents=True, exist_ok=True)
    temporaries = [path.with_suffix(".bin.tmp") for path in targets]
    token_count, document_count = [0, 0], [0, 0]
    reader_documents = 0
    batch = []
    with temporaries[0].open("wb") as even, temporaries[1].open("wb") as odd:
        streams = [even, odd]
        def flush() -> None:
            if not batch:
                return
            encoded = _TOKENIZER([text for _, text in batch],
                                 add_special_tokens=False)["input_ids"]
            for (parity, _), ids in zip(batch, encoded):
                ids.append(_TOKENIZER.eos_token_id)
                np.asarray(ids, dtype=np.uint32).tofile(streams[parity])
                token_count[parity] += len(ids)
                document_count[parity] += 1
            batch.clear()

        for text in _iter_local_jsonl_text(str(source)):
            parity = reader_documents % 2
            reader_documents += 1
            if text.strip():
                batch.append((parity, text))
                if len(batch) == batch_size:
                    flush()
        flush()
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
    for temporary, final in zip(temporaries, targets):
        os.replace(temporary, final)
    result = {"identity": identity, "paths": [str(p) for p in targets],
              "tokens": token_count, "documents": document_count,
              "reader_documents": reader_documents}
    atomic_json(meta_path, result)
    print(f"Tokenized {source.name}: {token_count} tokens by document parity", flush=True)
    return result


class OrderedShardWriter:
    """Copy flat streams into the same boundaries as serial window-at-a-time packing."""

    def __init__(self, out_dir: Path, seq_len: int, shard_tokens: int, budget: int):
        self.out_dir = out_dir
        self.window = seq_len + 1
        self.shard_limit = ((shard_tokens + self.window - 1) // self.window) * self.window
        self.target = ((budget + self.window - 1) // self.window) * self.window
        self.total_tokens = self.current_tokens = 0
        self.shards = []
        self.stream = None
        out_dir.mkdir(parents=True, exist_ok=True)

    def append(self, path: Path) -> int:
        consumed = 0
        with path.open("rb") as source:
            while self.total_tokens < self.target:
                if self.stream is None:
                    self.name = f"shard_{len(self.shards):05d}.bin"
                    self.temporary = self.out_dir / (self.name + ".tmp")
                    self.stream = self.temporary.open("wb")
                count = min(self.shard_limit - self.current_tokens,
                            self.target - self.total_tokens, 2_000_000)
                data = source.read(count * 4)
                if not data:
                    break
                if len(data) % 4:
                    raise ValueError(f"Incomplete uint32 stream: {path}")
                self.stream.write(data)
                tokens = len(data) // 4
                self.total_tokens += tokens
                self.current_tokens += tokens
                consumed += tokens
                if self.current_tokens == self.shard_limit:
                    self._finish()
        return consumed

    def _finish(self) -> None:
        if self.stream is None:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.temporary, self.out_dir / self.name)
        self.shards.append({"name": self.name, "n_tokens": self.current_tokens})
        self.stream = None
        self.current_tokens = 0
        print(f"Packed {self.total_tokens:,}/{self.target:,} tokens", flush=True)

    def close(self) -> None:
        if self.total_tokens != self.target:
            raise ValueError(f"Source exhausted at {self.total_tokens:,}, need {self.target:,}")
        self._finish()


def run(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("download_complete"):
        raise ValueError("Source download has not completed")
    source_root = args.manifest.parent
    # The list order is the old OLMO_MIX dictionary order, then the seeded file list.
    reconstruction = (json.loads(args.reconstruction.read_text())
                      if args.reconstruction else {})
    excluded = set(reconstruction.get("excluded_files", []))
    ordered = [(s["name"], p) for s in manifest["subsets"] for p in s["files"]
               if p not in excluded]
    tasks = [(str(source_root / p), str(args.cache_dir / f"file_{i:05d}.bin"),
              args.batch_size) for i, (_, p) in enumerate(ordered)]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    writer = OrderedShardWriter(args.out_dir, args.seq_len, args.shard_tokens,
                                args.token_budget)
    start = time.monotonic()
    contributions = []
    subset_tokens = {}
    # Bound lookahead so reaching the budget does not tokenize the unused corpus.
    pool = ProcessPoolExecutor(max_workers=args.workers,
                               mp_context=multiprocessing.get_context("spawn"),
                               initializer=init_worker, initargs=(args.tokenizer_id,))
    pending = {}
    next_task = 0
    reader_documents = 0
    try:
        for i, (subset, relative) in enumerate(ordered):
            while next_task < min(len(tasks), i + args.workers):
                pending[next_task] = pool.submit(tokenize_file, tasks[next_task])
                next_task += 1
            result = pending.pop(i).result()
            parity = reader_documents % 2
            consumed = writer.append(Path(result["paths"][parity]))
            reader_documents += result["reader_documents"]
            subset_tokens[subset] = subset_tokens.get(subset, 0) + consumed
            contributions.append({"subset": subset, "source": relative,
                                  "local_document_parity": parity,
                                  "reader_documents": result["reader_documents"],
                                  "available_tokens": result["tokens"][parity],
                                  "consumed_tokens": consumed})
            print(f"Consumed file {i + 1}: {writer.total_tokens:,} tokens; "
                  f"{(time.monotonic() - start) / 60:.1f} min", flush=True)
            if writer.total_tokens == writer.target:
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    writer.close()
    provenance = {
        "source_manifest": manifest,
        "historical_reconstruction": reconstruction,
        "ordering": "sequential subsets and files, matching the historical local reader",
        "document_selection": "alternating source documents, reproducing the original retry-counter behavior on its initial pass",
        "actual_subset_tokens": subset_tokens,
        "file_contributions": contributions,
        "token_budget": args.token_budget,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("transformers", "tokenizers", "numpy", "zstandard")},
    }
    write_index(str(args.out_dir), args.seq_len, tokenizer.eos_token_id,
                args.tokenizer_id, manifest["repository"], "olmo_mix",
                manifest["selection_seed"], writer.total_tokens, writer.shards, provenance)
    print(json.dumps({"total_tokens": writer.total_tokens, "subsets": subset_tokens,
                      "elapsed_minutes": (time.monotonic() - start) / 60}), flush=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--reconstruction", type=Path,
                        help="Historical missing/corrupt input file record")
    parser.add_argument("--token-budget", type=int, default=21_000_000_000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--shard-tokens", type=int, default=512_000_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.token_budget, args.seq_len,
           args.shard_tokens) < 1:
        parser.error("Numeric arguments must be positive")
    run(args)


if __name__ == "__main__":
    main()
