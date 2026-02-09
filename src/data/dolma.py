"""Streaming dataloader for Dolma v1.7 with sequence packing.

Produces packed sequences of fixed length for both OLMo and Qwen tokenizers.
See CLAUDE.md §3.1.1 for sequence packing specification.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


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
    ):
        super().__init__()
        self.olmo_tokenizer = olmo_tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.rank = rank
        self.world_size = world_size
        self.max_samples = max_samples

        self.eos_id = olmo_tokenizer.eos_token_id
        assert self.eos_id is not None, "OLMo tokenizer must have an EOS token"

    def __iter__(self) -> Iterator[dict]:
        """Yield packed sequences from Dolma stream."""
        try:
            dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_version,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback if specific version not available
            dataset = load_dataset(
                self.dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

        # Shard for multi-GPU
        if self.world_size > 1:
            dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

        buffer: list[int] = []
        sample_count = 0

        for doc in dataset:
            if self.max_samples is not None and sample_count >= self.max_samples:
                break

            text = doc.get("text", "")
            if not text.strip():
                continue

            tokens = self.olmo_tokenizer(text, add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            buffer.append(self.eos_id)

            # Yield packed sequences as buffer fills
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]

                olmo_ids = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
                olmo_labels = torch.tensor(chunk[1:self.seq_len + 1], dtype=torch.long)
                raw_text = self.olmo_tokenizer.decode(chunk[:self.seq_len], skip_special_tokens=False)

                yield {
                    "olmo_ids": olmo_ids,
                    "olmo_labels": olmo_labels,
                    "raw_text": raw_text,
                }
                sample_count += 1

                if self.max_samples is not None and sample_count >= self.max_samples:
                    break


def build_train_dataloader(
    olmo_tokenizer: AutoTokenizer,
    seq_len: int = 1024,
    batch_size: int = 4,
    dataset_name: str = "allenai/dolma",
    dataset_version: str = "v1_7",
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Build training dataloader with sequence packing."""
    dataset = DolmaPackedDataset(
        olmo_tokenizer=olmo_tokenizer,
        seq_len=seq_len,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        rank=rank,
        world_size=world_size,
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

    eos_id = olmo_tokenizer.eos_token_id
    buffer: list[int] = []
    eval_samples: list[dict] = []

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
