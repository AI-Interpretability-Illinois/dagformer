"""Frozen rank suffixes must replay the same samples at optimizer boundaries."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.freeze_dolma_continuation import merge_ranks, write_suffix
from src.data.dolma import DolmaPackedDataset
from src.data.mmap_dataset import MmapPackedDataset


class Tokenizer:
    eos_token_id = 100000

    def __call__(self, text, **kwargs):
        return {"input_ids": [70000 + int(x) for x in text.split()]}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class Source(DolmaPackedDataset):
    def _load_stream(self):
        return iter({"text": f"{i} {i + 1} {i + 2}"} for i in range(300))


def test_frozen_suffix_roundtrips_rank_order_and_resume(tmp_path):
    records, expected = [], []
    for rank in range(3):
        samples = list(Source(Tokenizer(), seq_len=7, skip_samples=5 + rank, max_samples=12))
        path = tmp_path / f"rank_{rank}.bin"
        write_suffix(samples, path, 12, 7, rank)
        records.append({"rank": rank, "samples": 12, "path": str(path)})
        expected.append(samples)
    total = merge_ranks(records[::-1], tmp_path / "continuation.bin", 7)
    index = {"format_version": 1, "dtype": "uint32", "seq_len": 7,
             "eos_id": 100000, "total_tokens": total,
             "shards": [{"name": "continuation.bin", "n_tokens": total}]}
    (tmp_path / "index.json").write_text(json.dumps(index))
    for rank in range(3):
        restored = MmapPackedDataset(str(tmp_path), seq_len=7, rank=rank, world_size=3,
                                     block_size=37, skip_samples=3 * 4, max_samples=8)
        for actual, original in zip(restored, expected[rank][4:]):
            np.testing.assert_array_equal(actual["olmo_ids"], original["olmo_ids"])
            np.testing.assert_array_equal(actual["olmo_labels"], original["olmo_labels"])
