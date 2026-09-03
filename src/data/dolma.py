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

# Dolmino 50B mix exact proportions (Mix % column from allenai/dolmino-mix-1124 README)
# Source: https://huggingface.co/datasets/allenai/dolmino-mix-1124#mix-compositions
DOLMINO_MIX = {
    "dclm": 0.472,
    "flan": 0.166,
    "math": 0.208,          # "Stage 2 Math" — excludes codesearchnet (schema bug)
    "wiki": 0.071,
    "pes2o": 0.0585,
    "stackexchange": 0.0245,
}
# Sum = 1.000

# Math subset is loaded via direct file pattern (not config name) because the
# `math` HF config includes `codesearchnet-owmfilter` which has incompatible
# struct schema. We include all other math subdirs (gsm8k, synth, etc).
DOLMINO_MATH_PATTERNS = [
    "data/math/dolmino_math_synth/**/*.jsonl",
    "data/math/gsm8k/**/*.jsonl.zst",
    "data/math/mathcoder2-synthmath/**/*.jsonl",
    "data/math/metamath-owmfilter/**/*.jsonl.gz",
    "data/math/tinyGSM-MIND/**/*.jsonl.gz",
    "data/math/tulu_math/**/*.jsonl",
]

# OLMo-mix-1124 (stage 1) natural-proportion weights — token counts from the
# dataset README divided by the 3.90T total. Roughly equivalent to per-token
# uniform sampling across all stage-1 sources.
# Source: https://huggingface.co/datasets/allenai/olmo-mix-1124
OLMO_MIX = {
    "dclm":            0.9487,   # 3.70T / 3.90T
    "starcoder":       0.02128,  # 83.0B
    "pes2o":           0.01503,  # 58.6B
    "arxiv":           0.00533,  # 20.8B
    "open-web-math":   0.00313,  # 12.2B
    "algebraic-stack": 0.00303,  # 11.8B
    "wiki":            0.00094,  # 3.66B
}

# olmo-mix-1124 stores files at data/<subset>/**/. Most subdirs are *.json.gz
# (one JSON per file); dclm uses *.jsonl.zstd (line-delimited, zstd compressed).
#
# Loading via HF's `name=...` config triggers schema validation that fails
# because metadata fields (paloma_paragraphs, tokenizer_repetitions) are
# inconsistent structs across documents. Loading via the generic `json` builder
# with `data_files=hf://...` URLs sidesteps schema validation completely.
#
# For DCLM (28K files), passing a glob pattern hangs HF's file listing — so we
# pre-list the files via HfApi and pass concrete URLs.
OLMO_MIX_SUFFIXES = {
    "dclm":            ".jsonl.zstd",
    "starcoder":       ".json.gz",
    "pes2o":           ".json.gz",
    "arxiv":           ".json.gz",
    "open-web-math":   ".json.gz",
    "algebraic-stack": ".json.gz",
    "wiki":            ".json.gz",
}

# Cap files per subset. Sized for a 20B-token training budget at natural mix
# proportions with 3–10× safety margin (pes2o/arxiv files are huge — a single
# file already covers the budget for those minority sources).
#   dclm:      19B tok needed @ ~95% mix → 200 files (~60GB disk)
#   starcoder: 425M tok                  → 50 files
#   pes2o:     300M tok @ ~700M tok/file → 2 files (each ~4GB)
#   arxiv:     107M tok @ ~250M tok/file → 3 files
#   ow-math:    63M tok                  → 3 files
#   alg-stack:  61M tok                  → 3 files
#   wiki:       19M tok (entire subset)  → 2 files
# Total mirror size: ~99GB.
OLMO_MIX_MAX_FILES = {
    "dclm":            200,
    "starcoder":        50,
    "pes2o":             2,
    "arxiv":             3,
    "open-web-math":     3,
    "algebraic-stack":   3,
    "wiki":              2,
}

_OLMO_MIX_ALL_FILES_CACHE: list[str] | None = None

# Local mirror written by scripts/download_olmo_mix_local.py. If present, the
# olmo_mix dataloader reads files from here instead of streaming over HTTP —
# avoids HF's 5000-request/5min resolver rate-limit when running 8 DDP ranks.
OLMO_MIX_LOCAL_ROOT = "/work/hdd/bfqt/data/olmo-mix-1124"


def _list_olmo_mix_files(subset: str, seed: int = 0) -> list[str]:
    """Resolve file paths for an olmo-mix-1124 subset.

    Files are deterministically shuffled by `seed` (training order ≠ alphabetical)
    and capped via OLMO_MIX_MAX_FILES. If a local mirror exists at
    OLMO_MIX_LOCAL_ROOT, returns local filesystem paths; otherwise falls back
    to hf:// URLs (which hits the resolver rate-limit at 8-rank scale, so the
    local mirror is strongly preferred for any real training run).
    """
    import os as _os
    import random
    global _OLMO_MIX_ALL_FILES_CACHE
    if _OLMO_MIX_ALL_FILES_CACHE is None:
        from huggingface_hub import HfApi
        api = HfApi()
        _OLMO_MIX_ALL_FILES_CACHE = api.list_repo_files(
            "allenai/olmo-mix-1124", repo_type="dataset",
        )
    prefix = f"data/{subset}/"
    suffix = OLMO_MIX_SUFFIXES[subset]
    files = sorted(
        f for f in _OLMO_MIX_ALL_FILES_CACHE
        if f.startswith(prefix) and f.endswith(suffix)
    )
    rng = random.Random(seed)
    rng.shuffle(files)
    cap = OLMO_MIX_MAX_FILES.get(subset)
    if cap is not None:
        files = files[:cap]
    # Prefer local mirror if it exists for THIS subset's first file (cheap probe)
    if files and _os.path.exists(_os.path.join(OLMO_MIX_LOCAL_ROOT, files[0])):
        local_paths = [
            _os.path.join(OLMO_MIX_LOCAL_ROOT, p) for p in files
            if _os.path.exists(_os.path.join(OLMO_MIX_LOCAL_ROOT, p))
        ]
        if len(local_paths) == len(files):
            return local_paths
        # partial mirror — fall back to HF (warning printed once)
        print(
            f"[dolma] WARN olmo-mix/{subset}: only {len(local_paths)}/{len(files)} "
            f"local files; falling back to hf:// for completeness",
            flush=True,
        )
    return [f"hf://datasets/allenai/olmo-mix-1124/{p}" for p in files]



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
            # Interleave Dolmino subsets with official 50B-budget mix proportions
            # Only keep 'text' column — metadata schemas differ across subsets
            subsets = []
            probs = []
            for name, weight in DOLMINO_MIX.items():
                if name == "math":
                    # math via direct file patterns (HF `math` config has a broken subdir)
                    ds = load_dataset(
                        self.dataset_name,
                        data_files=DOLMINO_MATH_PATTERNS,
                        split="train",
                        streaming=True,
                    ).select_columns(["text"])
                else:
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
        elif self.dataset_version == "olmo_mix":
            # Stage-1 olmo-mix-1124. Natural-proportion sampling across 7 subsets
            # (dclm dominates at 95%, code/sci/wiki at small fractions).
            # We use the generic `json` builder with explicit hf:// file URLs
            # to bypass the broken schema validation in the named configs.
            # File lists are seeded-shuffled for determinism + non-alphabetical
            # training order.
            # OLMO_MIX values come from token-share rounding; renormalize to
            # exactly 1.0 so interleave_datasets stops complaining.
            _total = sum(OLMO_MIX.values())
            subsets = []
            probs = []
            for name, weight in OLMO_MIX.items():
                hf_files = _list_olmo_mix_files(name, seed=0)
                ds = load_dataset(
                    "json",
                    data_files=hf_files,
                    split="train",
                    streaming=True,
                )
                # Project to text-only so interleave_datasets can align subsets.
                ds = ds.map(
                    lambda x: {"text": x.get("text", "") or ""},
                    remove_columns=[c for c in ds.column_names if c != "text"]
                    if ds.column_names else None,
                )
                subsets.append(ds)
                probs.append(weight / _total)
            dataset = interleave_datasets(
                subsets, probabilities=probs, stopping_strategy="all_exhausted",
            )
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
        # For dolmino_mix / olmo_mix, HF .shard() doesn't work on interleaved
        # datasets, so we do manual document-level modulo sharding here instead.
        manual_shard = (
            self.dataset_version in ("dolmino_mix", "olmo_mix") and self.world_size > 1
        )

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

    # Eval-set construction is a fixed, bounded task: unlike the training stream
    # it must NOT retry forever. A deterministic failure (e.g. olmo_mix's
    # "Probabilities do not sum to 1") otherwise blocks the whole job before
    # training starts — this silently held 8xH200 idle for 12 minutes until
    # noticed. Fail loudly after a few attempts instead.
    EVAL_MAX_RETRIES = 5
    for attempt in range(EVAL_MAX_RETRIES + 1):
        try:
            if dataset_version == "dolmino_mix":
                subsets = []
                probs = []
                for name, weight in DOLMINO_MIX.items():
                    if name == "math":
                        ds = load_dataset(
                            dataset_name,
                            data_files=DOLMINO_MATH_PATTERNS,
                            split="train",
                            streaming=True,
                        ).select_columns(["text"])
                    else:
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
            elif dataset_version == "olmo_mix":
                subsets = []
                probs = []
                for name, weight in OLMO_MIX.items():
                    hf_files = _list_olmo_mix_files(name, seed=0)
                    ds = load_dataset(
                        "json",
                        data_files=hf_files,
                        split="train",
                        streaming=True,
                    )
                    ds = ds.map(
                        lambda x: {"text": x.get("text", "") or ""},
                        remove_columns=[c for c in ds.column_names if c != "text"]
                        if ds.column_names else None,
                    )
                    subsets.append(ds)
                    probs.append(weight)
                dataset = interleave_datasets(
                    subsets, probabilities=probs, stopping_strategy="all_exhausted",
                )
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
            if attempt >= EVAL_MAX_RETRIES:
                raise RuntimeError(
                    f"Eval set build failed after {EVAL_MAX_RETRIES} retries: {e}"
                ) from e
            print(f"[EvalBuild] Stream error (retry {attempt + 1}/{EVAL_MAX_RETRIES}): {e}")
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
