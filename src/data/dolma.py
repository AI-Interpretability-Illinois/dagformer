"""Streaming dataloader with sequence packing.

Supports Dolma v1.7 and Dolmino-Mix-1124 (multi-subset interleaving).
See CLAUDE.md §3.1.1 for sequence packing specification.
"""

from __future__ import annotations

import os
import time
from typing import Iterator, Optional

import torch
from datasets import interleave_datasets, load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

# Dolmino 50B mix approximate proportions (from OLMo-core dolmino50.txt)
# dclm:48%, flan:17%, tinygsm/math:21%, pes2o:6%, wiki:7%, stackexchange:2.5%
# math subset has schema issues, merge its weight into dclm
DOLMINO_MIX = {
    "dclm": 0.685,          # 48% + 21% math (redistributed)
    "flan": 0.17,
    "pes2o": 0.06,
    "wiki": 0.06,
    "stackexchange": 0.025,
}
# Sum = 1.000

MAX_RETRIES = 999999  # never crash from transient HTTP errors
RETRY_WAIT = 30  # seconds


class DolmaPackedDataset(IterableDataset):
    """Streaming Dolma dataset with sequence packing.

    Concatenates documents with EOS separators, then chunks into fixed-length
    sequences. No padding — every token contributes to NLL.

    Each sample yields:
        olmo_ids: [seq_len] — OLMo input token IDs
        olmo_labels: [seq_len] — shifted labels (next-token prediction)
        raw_text: str — decoded text for Qwen encoder
    """

    def __init__(
        self,
        olmo_tokenizer: AutoTokenizer,
        seq_len: int = 1024,
        dataset_name: str = "allenai/dolma",
        dataset_version: str = "v1_7",
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
        skip_samples: int = 0,
    ):
        super().__init__()
        self.olmo_tokenizer = olmo_tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.rank = rank
        self.world_size = world_size
        self.max_samples = max_samples
        self.skip_samples = skip_samples  # skip first N packed sequences on resume

        self.eos_id = olmo_tokenizer.eos_token_id
        assert self.eos_id is not None, "OLMo tokenizer must have an EOS token"

    def _load_local_stream(self, local_dir: str):
        """Iterate pre-downloaded *.json.gz files in local_dir (alphabetical order).
        Each rank takes every world_size-th doc (manual shard).
        Activated by env DOLMA_LOCAL_DIR — eliminates HTTP streaming instability.
        """
        import glob, gzip, json
        files = sorted(glob.glob(os.path.join(local_dir, "**", "*.json.gz"), recursive=True))
        if not files:
            raise RuntimeError(f"DOLMA_LOCAL_DIR={local_dir} has no *.json.gz files")
        if self.rank == 0:
            print(f"[DolmaDataset] Local mode: {len(files)} files in {local_dir}")

        def gen():
            doc_idx = 0
            for fpath in files:
                with gzip.open(fpath, "rt", encoding="utf-8") as f:
                    for line in f:
                        take = (self.world_size <= 1) or (doc_idx % self.world_size == self.rank)
                        doc_idx += 1
                        if not take:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue
        return gen()

    def _load_stream(self):
        """Load streaming dataset. Supports Dolmino multi-subset interleaving.

        For dolmino_mix: interleaved datasets don't support HF .shard(), and
        some subsets have too few parquet files for per-subset sharding.
        DDP sharding is handled via manual modulo in __iter__ instead.
        """
        local_dir = os.environ.get("DOLMA_LOCAL_DIR", "")
        if local_dir and os.path.isdir(local_dir) and self.dataset_version != "dolmino_mix":
            return self._load_local_stream(local_dir)

        if self.dataset_version == "dolmino_mix":
            # Interleave Dolmino subsets with approximate 50B mix proportions
            # Only keep 'text' column — metadata schemas differ across subsets
            subsets = []
            probs = []
            for name, weight in DOLMINO_MIX.items():
                ds = load_dataset(
                    self.dataset_name,
                    name=name,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                ).select_columns(["text"])
                subsets.append(ds)
                probs.append(weight)
            dataset = interleave_datasets(subsets, probabilities=probs, stopping_strategy="all_exhausted")
        else:
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    name=self.dataset_version,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
            except Exception:
                dataset = load_dataset(
                    self.dataset_name,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )

            if self.world_size > 1:
                dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

        return dataset

    def __iter__(self) -> Iterator[dict]:
        """Yield packed sequences from Dolma stream with retry on HTTP errors.

        On resume (skip_samples > 0), fast-forwards through the stream to avoid
        re-training on already-seen data. On HTTP retry, also fast-forwards to
        the last yielded position instead of restarting from the beginning.
        """
        buffer: list[int] = []
        sample_count = 0  # total samples produced (including skipped)
        yielded_count = 0  # samples actually yielded (after skip)
        retries = 0
        # For dolmino_mix, HF .shard() doesn't work on interleaved datasets,
        # so we do manual document-level modulo sharding here instead.
        manual_shard = (self.dataset_version == "dolmino_mix" and self.world_size > 1)

        # Track how many docs we've consumed for HTTP retry fast-forward
        docs_consumed = 0

        if self.skip_samples > 0:
            print(f"[DolmaDataset] Resuming: will skip first {self.skip_samples} packed sequences")

        while retries <= MAX_RETRIES:
            try:
                dataset = self._load_stream()
                doc_idx = 0
                docs_skipped_for_retry = 0

                for doc in dataset:
                    if self.max_samples is not None and yielded_count >= self.max_samples:
                        return

                    # Manual DDP sharding: each rank takes every world_size-th doc
                    if manual_shard:
                        if doc_idx % self.world_size != self.rank:
                            doc_idx += 1
                            continue
                        doc_idx += 1

                    # On HTTP retry, skip docs we already processed
                    if docs_skipped_for_retry < docs_consumed:
                        docs_skipped_for_retry += 1
                        continue

                    text = doc.get("text", "")
                    if not text.strip():
                        docs_consumed += 1
                        continue

                    tokens = self.olmo_tokenizer(text, add_special_tokens=False)["input_ids"]
                    buffer.extend(tokens)
                    buffer.append(self.eos_id)
                    docs_consumed += 1

                    # Yield packed sequences as buffer fills
                    while len(buffer) >= self.seq_len + 1:
                        chunk = buffer[:self.seq_len + 1]
                        buffer = buffer[self.seq_len + 1:]

                        sample_count += 1

                        # Fast-forward: skip already-seen samples on resume
                        if sample_count <= self.skip_samples:
                            continue

                        olmo_ids = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
                        olmo_labels = torch.tensor(chunk[1:self.seq_len + 1], dtype=torch.long)
                        raw_text = self.olmo_tokenizer.decode(chunk[:self.seq_len], skip_special_tokens=False)

                        yield {
                            "olmo_ids": olmo_ids,
                            "olmo_labels": olmo_labels,
                            "raw_text": raw_text,
                        }
                        yielded_count += 1

                        if self.max_samples is not None and yielded_count >= self.max_samples:
                            return

                # Stream exhausted normally
                return

            except Exception as e:
                retries += 1
                if retries > MAX_RETRIES:
                    raise RuntimeError(f"Dolma stream failed after {MAX_RETRIES} retries: {e}") from e
                print(f"[DolmaDataset] Stream error (retry {retries}/{MAX_RETRIES}): {e}")
                print(f"[DolmaDataset] Resuming from doc {docs_consumed} after {RETRY_WAIT}s...")
                time.sleep(RETRY_WAIT)
                buffer = []  # reset buffer on retry


def build_train_dataloader(
    olmo_tokenizer: AutoTokenizer,
    seq_len: int = 1024,
    batch_size: int = 4,
    dataset_name: str = "allenai/dolma",
    dataset_version: str = "v1_7",
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    skip_samples: int = 0,
) -> torch.utils.data.DataLoader:
    """Build training dataloader with sequence packing.

    Args:
        skip_samples: number of packed sequences to skip (for resume).
            On resume, pass step * accum_steps * batch_size to fast-forward
            past already-seen data instead of retraining on the same prefix.
    """
    dataset = DolmaPackedDataset(
        olmo_tokenizer=olmo_tokenizer,
        seq_len=seq_len,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        rank=rank,
        world_size=world_size,
        skip_samples=skip_samples,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_packed,
    )


def build_eval_dataloader(
    olmo_tokenizer: AutoTokenizer,
    seq_len: int = 1024,
    batch_size: int = 4,
    dataset_name: str = "allenai/dolma",
    dataset_version: str = "v1_7",
    eval_skip: int = 1_000_000,
    eval_size: int = 1_000,
    cache_path: Optional[str] = None,
) -> list[dict]:
    """Build eval batches (cached in memory).

    Skips eval_skip examples in the stream, then takes eval_size packed sequences.
    Caches to disk to avoid repeated skip on restart.
    """
    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading eval cache from {cache_path}")
        return torch.load(cache_path)

    print(f"Building eval set (skip={eval_skip}, size={eval_size})...")

    eos_id = olmo_tokenizer.eos_token_id
    eval_samples: list[dict] = []

    for attempt in range(MAX_RETRIES + 1):
        try:
            if dataset_version == "dolmino_mix":
                subsets = []
                probs = []
                for name, weight in DOLMINO_MIX.items():
                    ds = load_dataset(
                        dataset_name,
                        name=name,
                        split="train",
                        streaming=True,
                        trust_remote_code=True,
                    ).select_columns(["text"])
                    subsets.append(ds)
                    probs.append(weight)
                dataset = interleave_datasets(subsets, probabilities=probs, stopping_strategy="all_exhausted")
            else:
                try:
                    dataset = load_dataset(
                        dataset_name,
                        name=dataset_version,
                        split="train",
                        streaming=True,
                        trust_remote_code=True,
                    )
                except Exception:
                    dataset = load_dataset(
                        dataset_name,
                        split="train",
                        streaming=True,
                        trust_remote_code=True,
                    )

            # Skip to held-out region
            dataset = dataset.skip(eval_skip)

            buffer: list[int] = []
            eval_samples = []

            for doc in dataset:
                if len(eval_samples) >= eval_size:
                    break

                text = doc.get("text", "")
                if not text.strip():
                    continue

                tokens = olmo_tokenizer(text, add_special_tokens=False)["input_ids"]
                buffer.extend(tokens)
                buffer.append(eos_id)

                while len(buffer) >= seq_len + 1 and len(eval_samples) < eval_size:
                    chunk = buffer[:seq_len + 1]
                    buffer = buffer[seq_len + 1:]
                    eval_samples.append({
                        "olmo_ids": torch.tensor(chunk[:seq_len], dtype=torch.long),
                        "olmo_labels": torch.tensor(chunk[1:seq_len + 1], dtype=torch.long),
                        "raw_text": olmo_tokenizer.decode(chunk[:seq_len], skip_special_tokens=False),
                    })

            break  # success

        except Exception as e:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(f"Eval set build failed after {MAX_RETRIES} retries: {e}") from e
            print(f"[EvalBuild] Stream error (retry {attempt + 1}/{MAX_RETRIES}): {e}")
            print(f"[EvalBuild] Waiting {RETRY_WAIT}s before reconnecting...")
            time.sleep(RETRY_WAIT)

    print(f"Built {len(eval_samples)} eval sequences")

    # Batch the samples
    eval_batches = []
    for i in range(0, len(eval_samples), batch_size):
        batch_items = eval_samples[i:i + batch_size]
        eval_batches.append(_collate_packed(batch_items))

    # Cache to disk
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(eval_batches, cache_path)
        print(f"Eval cache saved to {cache_path}")

    return eval_batches


def _collate_packed(batch: list[dict]) -> dict:
    """Collate packed samples into a batch dict."""
    return {
        "olmo_ids": torch.stack([s["olmo_ids"] for s in batch]),
        "olmo_labels": torch.stack([s["olmo_labels"] for s in batch]),
        "raw_text": [s["raw_text"] for s in batch],
    }
