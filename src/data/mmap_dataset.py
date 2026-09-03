"""Memory-mapped packed dataloader (zero network, zero per-step tokenization).

Training-time counterpart to `scripts/pretokenize.py`. Reads the frozen uint32
shard set produced offline, memory-maps it, and yields packed training samples.
This replaces train-time HuggingFace streaming, which caused NCCL-timeout deaths
from silent network stalls and made resume incorrect (the stream position could
not be recovered exactly). Here, resume is a single integer offset with exact
O(skip) fast-forward and no network at all.

Layout on disk (written by pretokenize.py):
    <index_dir>/index.json          # {seq_len, eos_id, total_tokens, shards:[...], ...}
    <index_dir>/shard_00000.bin     # flat uint32 token ids
    <index_dir>/shard_00001.bin
    ...

The concatenation of all shards is treated as ONE logical token array of length
`total_tokens`. Sample `k` occupies the window
    tokens[k*(seq_len+1) : (k+1)*(seq_len+1)]
and yields
    olmo_ids    = window[:seq_len]
    olmo_labels = window[1:seq_len+1]   # already shifted; trainer does NOT re-shift.
Total sample count  S = total_tokens // (seq_len + 1).

=== Per-rank + per-worker disjoint sharding ===
Within one epoch we build a DETERMINISTIC permutation of [0, S) (block-shuffle,
seed = base_seed + epoch). We then hand out permuted-position `p` to exactly one
(rank, worker) pair by a single flat modulo over the combined stride:
    global_worker_id   = rank * num_workers + worker_id      (0-indexed)
    total_workers      = world_size * num_workers
    (rank, worker) owns permuted positions p where  p % total_workers == global_worker_id
Because {global_worker_id} enumerates exactly [0, total_workers), the residue
classes partition [0, S): every sample is emitted by exactly one (rank, worker)
and none is emitted twice. This is the whole point — get it right.

=== Resume by a single integer ===
`skip_samples` is a GLOBAL count of samples already consumed across ALL ranks:
    skip_samples = global_step * accum * micro * world_size
(one optimizer step consumes accum * micro samples per rank, times world_size
ranks). Every rank/worker fast-forwards its own strided-permuted iterator past
the first `skip_samples` GLOBAL permuted positions, i.e. it drops the owned
positions whose permuted-position index p satisfies p < skip_samples. Since the
permutation order is identical across all ranks (same seed+epoch) and the stride
partition is fixed, dropping `p < skip_samples` on every stride reconstructs
exactly the global consumption prefix, so resume yields the identical suffix an
uninterrupted run would. See `_epoch_positions` for the arithmetic.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _load_index(index_path: str) -> dict:
    """Load index.json from a file path or a directory containing index.json."""
    if os.path.isdir(index_path):
        index_path = os.path.join(index_path, "index.json")
    with open(index_path) as f:
        index = json.load(f)
    assert index.get("format_version") == 1, index.get("format_version")
    assert index.get("dtype") == "uint32", f"expected uint32, got {index.get('dtype')}"
    return index, os.path.dirname(os.path.abspath(index_path))


def _block_permutation(
    n_samples: int, seed: int, block_size: int
) -> np.ndarray:
    """Deterministic block-shuffle permutation of [0, n_samples).

    Instead of a full Fisher-Yates permutation (which touches every index and
    scatters mmap reads randomly), we split [0, n_samples) into contiguous blocks
    of `block_size` samples, shuffle the BLOCK ORDER, and keep samples sequential
    within each block. This preserves mostly-sequential mmap access (fast) while
    still decorrelating the epoch order.

    We only materialize the block-order array (n_samples // block_size entries),
    not a full n_samples-length permutation, so memory stays O(n_samples/block).

    Returns an int64 array; consumers should iterate it lazily but for the sizes
    involved (S up to a few million) a single array is fine. For very large S the
    per-position iteration in `_epoch_positions` avoids building the full expansion.
    """
    assert n_samples >= 0, n_samples
    assert block_size >= 1, block_size
    n_blocks = (n_samples + block_size - 1) // block_size
    rng = np.random.default_rng(seed)
    block_order = rng.permutation(n_blocks)
    return block_order  # expanded lazily in _epoch_positions


def _epoch_positions(
    n_samples: int,
    seed: int,
    block_size: int,
    global_worker_id: int,
    total_workers: int,
    skip_positions: int,
) -> Iterator[int]:
    """Yield the sample indices this (rank, worker) owns for one epoch, in order.

    Walks the block-shuffled permutation lazily (never materializes a full
    S-length array) and applies BOTH the stride partition and the resume skip:

      * permuted-position counter `p` runs 0,1,2,... over the whole epoch in the
        block-shuffled order (identical on every rank for a given seed).
      * this worker keeps positions with  p % total_workers == global_worker_id.
      * resume drops any owned position with  p < skip_positions.

    `skip_positions` is the GLOBAL number of permuted positions already consumed
    (i.e. `skip_samples`), NOT a per-rank count — the p < skip_positions test is
    applied on the shared global position axis, so every stride is fast-forwarded
    consistently.

    Yields the actual token-array sample index (block_start + offset), not `p`.
    """
    assert 0 <= global_worker_id < total_workers, (global_worker_id, total_workers)
    block_order = _block_permutation(n_samples, seed, block_size)

    p = 0  # global permuted-position index within this epoch
    for blk in block_order:
        start = int(blk) * block_size
        end = min(start + block_size, n_samples)
        for sample_idx in range(start, end):
            if p >= skip_positions and (p % total_workers) == global_worker_id:
                yield sample_idx
            p += 1


class MmapPackedDataset(IterableDataset):
    """Iterable dataset over memory-mapped uint32 packed shards.

    Yields dicts:
        {"olmo_ids": LongTensor[seq_len], "olmo_labels": LongTensor[seq_len]}
    and optionally {"raw_text": str} when `emit_raw_text=True` (legacy qwen path).

    Per-rank + per-worker disjointness and integer-offset resume are documented in
    the module docstring. Loops epochs forever (re-seeding by epoch) unless
    `max_samples` bounds it.
    """

    def __init__(
        self,
        index_path: str,
        seq_len: int = 1024,
        rank: int = 0,
        world_size: int = 1,
        skip_samples: int = 0,
        seed: int = 0,
        block_size: int = 1024,
        max_samples: Optional[int] = None,
        emit_raw_text: bool = False,
        tokenizer=None,
    ) -> None:
        super().__init__()
        index, base_dir = _load_index(index_path)
        self.index = index
        self.base_dir = base_dir

        idx_seq_len = int(index["seq_len"])
        assert idx_seq_len == seq_len, (
            f"seq_len mismatch: caller={seq_len} index={idx_seq_len}"
        )
        self.seq_len = seq_len
        self.window = seq_len + 1
        self.eos_id = int(index["eos_id"])

        self.rank = rank
        self.world_size = world_size
        assert 0 <= rank < world_size, (rank, world_size)
        self.skip_samples = int(skip_samples)
        assert self.skip_samples >= 0, self.skip_samples
        self.seed = seed
        self.block_size = block_size
        self.max_samples = max_samples

        self.emit_raw_text = emit_raw_text
        self.tokenizer = tokenizer
        if emit_raw_text:
            assert tokenizer is not None, "emit_raw_text=True requires a tokenizer"

        # Memory-map every shard read-only; record cumulative token offsets so a
        # global token position maps to (shard, local offset). memmap is lazy —
        # opening it does not read the data.
        self.shard_paths: list[str] = []
        self.shard_lens: list[int] = []
        for s in index["shards"]:
            path = os.path.join(base_dir, s["name"])
            assert os.path.exists(path), f"missing shard: {path}"
            self.shard_paths.append(path)
            self.shard_lens.append(int(s["n_tokens"]))
        self.shard_lens_arr = np.asarray(self.shard_lens, dtype=np.int64)
        self.cum_offsets = np.concatenate(
            [[0], np.cumsum(self.shard_lens_arr)]
        )  # len = n_shards + 1; cum_offsets[i] = first global token idx of shard i
        self.total_tokens = int(self.cum_offsets[-1])

        recorded_total = int(index.get("total_tokens", self.total_tokens))
        assert recorded_total == self.total_tokens, (
            f"index total_tokens {recorded_total} != sum of shard lens {self.total_tokens}"
        )

        # Number of full training samples (windows of seq_len+1 tokens).
        self.n_samples = self.total_tokens // self.window
        assert self.n_samples > 0, (
            f"no full samples: total_tokens={self.total_tokens} window={self.window}"
        )

        self._mmaps: Optional[list[np.memmap]] = None  # opened lazily per worker

    # ---- token access ------------------------------------------------------
    def _ensure_mmaps(self) -> None:
        """Open memmaps lazily. Deferred so each DataLoader worker process (a
        fork) opens its own handles rather than sharing a parent's."""
        if self._mmaps is None:
            self._mmaps = [
                np.memmap(p, dtype=np.uint32, mode="r", shape=(n,))
                for p, n in zip(self.shard_paths, self.shard_lens)
            ]

    def _read_window(self, sample_idx: int) -> np.ndarray:
        """Return the (seq_len+1,) uint32 window for a given sample index.

        Handles windows that straddle a shard boundary by reading across shards.
        """
        assert 0 <= sample_idx < self.n_samples, (sample_idx, self.n_samples)
        self._ensure_mmaps()
        start = sample_idx * self.window
        end = start + self.window  # exclusive

        # Locate the shard containing `start`. cum_offsets is sorted ascending.
        shard_i = int(np.searchsorted(self.cum_offsets, start, side="right") - 1)
        assert 0 <= shard_i < len(self.shard_paths), (shard_i, start)

        out = np.empty(self.window, dtype=np.uint32)
        filled = 0
        pos = start
        while filled < self.window:
            shard_start = int(self.cum_offsets[shard_i])
            shard_end = int(self.cum_offsets[shard_i + 1])
            local_start = pos - shard_start
            take = min(shard_end, end) - pos
            assert take > 0, (take, pos, shard_end, end)
            out[filled : filled + take] = self._mmaps[shard_i][
                local_start : local_start + take
            ]
            filled += take
            pos += take
            shard_i += 1
        assert filled == self.window, (filled, self.window)
        return out

    # ---- iteration ---------------------------------------------------------
    def _worker_stride(self) -> tuple[int, int]:
        """Compute (global_worker_id, total_workers) for this process.

        Combines DDP rank and DataLoader worker id into a single flat stride so
        (rank, worker) pairs are disjoint AND exhaustive over [0, total_workers).
        """
        info = torch.utils.data.get_worker_info()
        if info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = info.num_workers
            worker_id = info.id
        total_workers = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id
        assert 0 <= global_worker_id < total_workers, (
            global_worker_id, total_workers, self.rank, worker_id
        )
        return global_worker_id, total_workers

    def __iter__(self) -> Iterator[dict]:
        global_worker_id, total_workers = self._worker_stride()

        yielded = 0
        epoch = 0
        # `skip_samples` only applies to epoch 0's global position axis; later
        # epochs start fresh (re-seeded). Consumed samples beyond epoch 0 are
        # counted globally too, but resume only ever lands inside a partial epoch
        # because checkpoints are frequent relative to an epoch — we still handle
        # skip that spans whole epochs by decrementing below.
        remaining_skip = self.skip_samples

        while True:
            epoch_seed = self.seed + epoch
            if remaining_skip >= self.n_samples:
                # This whole epoch was already consumed; skip it entirely.
                remaining_skip -= self.n_samples
                epoch += 1
                continue

            positions = _epoch_positions(
                n_samples=self.n_samples,
                seed=epoch_seed,
                block_size=self.block_size,
                global_worker_id=global_worker_id,
                total_workers=total_workers,
                skip_positions=remaining_skip,
            )
            for sample_idx in positions:
                window = self._read_window(sample_idx)
                # uint32 -> int64 for torch (torch has no uint32 tensor).
                olmo_ids = torch.from_numpy(
                    window[: self.seq_len].astype(np.int64)
                )
                olmo_labels = torch.from_numpy(
                    window[1 : self.seq_len + 1].astype(np.int64)
                )
                assert olmo_ids.shape == (self.seq_len,), olmo_ids.shape
                assert olmo_labels.shape == (self.seq_len,), olmo_labels.shape
                assert olmo_ids.dtype == torch.long, olmo_ids.dtype

                sample = {"olmo_ids": olmo_ids, "olmo_labels": olmo_labels}
                if self.emit_raw_text:
                    sample["raw_text"] = self.tokenizer.decode(
                        window[: self.seq_len].tolist(), skip_special_tokens=False
                    )
                yield sample

                yielded += 1
                if self.max_samples is not None and yielded >= self.max_samples:
                    return

            # Epoch exhausted; the resume skip only applies to the first (partial)
            # epoch, so it is fully spent now.
            remaining_skip = 0
            epoch += 1


def _collate_packed(batch: list[dict]) -> dict:
    """Collate packed samples into a batch dict (stacks ids/labels)."""
    out = {
        "olmo_ids": torch.stack([s["olmo_ids"] for s in batch]),
        "olmo_labels": torch.stack([s["olmo_labels"] for s in batch]),
    }
    if "raw_text" in batch[0]:
        out["raw_text"] = [s["raw_text"] for s in batch]
    return out


def build_mmap_train_dataloader(
    index_path: str,
    seq_len: int = 1024,
    batch_size: int = 4,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    skip_samples: int = 0,
    seed: int = 0,
    block_size: int = 1024,
    emit_raw_text: bool = False,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    """Build a training DataLoader over pretokenized mmap shards.

    Drop-in replacement for `src.data.dolma.build_train_dataloader`:
    same call-site shape, but reads pretokenized shards instead of streaming.

    Args:
        index_path: path to index.json (or the dir containing it).
        skip_samples: GLOBAL samples already consumed across all ranks for resume
            (= global_step * accum * micro * world_size). Each rank derives its own
            fast-forward internally; see module docstring.
        block_size: block-shuffle granularity (samples per contiguous block).
    """
    dataset = MmapPackedDataset(
        index_path=index_path,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        skip_samples=skip_samples,
        seed=seed,
        block_size=block_size,
        emit_raw_text=emit_raw_text,
        tokenizer=tokenizer,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_packed,
    )


# ---------------------------------------------------------------------------
# Self-test: synthetic shards only (numpy/torch), NO HuggingFace.
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import shutil
    import tempfile

    print("[selftest] starting", flush=True)
    tmp = tempfile.mkdtemp(prefix="mmap_selftest_")
    try:
        seq_len = 7
        window = seq_len + 1
        # Two shards. Use a strictly-increasing token stream so each sample's
        # first token uniquely identifies it (makes disjointness/order checks easy).
        # Keep tokens < uint32 max; include values > 65535 to exercise the uint32
        # requirement (would corrupt under uint16).
        shard0 = np.arange(0, 100, dtype=np.uint32) + 70000   # 100 tokens, ids 70000..70099
        shard1 = np.arange(100, 205, dtype=np.uint32) + 70000  # 105 tokens
        total_tokens = shard0.shape[0] + shard1.shape[0]  # 205
        for name, arr in [("shard_00000.bin", shard0), ("shard_00001.bin", shard1)]:
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(arr.tobytes())
        index = {
            "format_version": 1,
            "dtype": "uint32",
            "seq_len": seq_len,
            "eos_id": 0,
            "tokenizer_id": "synthetic",
            "dataset": "synthetic",
            "dataset_version": "synthetic",
            "seed": 0,
            "total_tokens": total_tokens,
            "shards": [
                {"name": "shard_00000.bin", "n_tokens": int(shard0.shape[0])},
                {"name": "shard_00001.bin", "n_tokens": int(shard1.shape[0])},
            ],
            "created_utc": None,
        }
        with open(os.path.join(tmp, "index.json"), "w") as f:
            json.dump(index, f)

        S = total_tokens // window  # 205 // 8 = 25 samples
        print(f"[selftest] total_tokens={total_tokens} window={window} S={S}", flush=True)

        # (a) shapes + straddle correctness (single rank, single worker).
        ds = MmapPackedDataset(tmp, seq_len=seq_len, rank=0, world_size=1, seed=0)
        assert ds.n_samples == S, (ds.n_samples, S)
        flat = np.concatenate([shard0, shard1])
        seen_a = []
        for i, sample in enumerate(ds):
            if i >= S:  # stop after one epoch's worth for this check
                break
            ids = sample["olmo_ids"].numpy()
            labels = sample["olmo_labels"].numpy()
            assert ids.shape == (seq_len,) and labels.shape == (seq_len,)
            # Verify labels are ids shifted by one within the window.
            first_tok = int(ids[0])
            k = first_tok - 70000  # since flat[j] = 70000 + j and sample start = p*window
            # Recover which sample index this is from its start token, then check
            # the window matches the flat array exactly (incl. straddle).
            start = None
            # find start by matching first token position in flat
            start = int(np.where(flat == first_tok)[0][0])
            expected = flat[start : start + window]
            assert np.array_equal(ids, expected[:seq_len]), (ids, expected[:seq_len])
            assert np.array_equal(labels, expected[1 : seq_len + 1])
            seen_a.append(start // window)
        assert sorted(seen_a) == list(range(S)), ("single-rank did not cover all samples", sorted(seen_a))
        print("[selftest] (a) shapes + shift + straddle OK", flush=True)

        # (b) 4-rank disjoint + exhaustive cover (one epoch each).
        world_size = 4
        collected: dict[int, list[int]] = {}
        for r in range(world_size):
            dr = MmapPackedDataset(tmp, seq_len=seq_len, rank=r, world_size=world_size, seed=0)
            idxs = []
            for i, sample in enumerate(dr):
                start = int(np.where(flat == int(sample["olmo_ids"].numpy()[0]))[0][0])
                idxs.append(start // window)
                # one epoch = ceil(S / world_size) at most; stop generously
                if len(idxs) >= S:  # safety
                    break
                # detect epoch wrap: block-shuffle repeats after S/world positions
                if len(idxs) > 1 and len(set(idxs)) < len(idxs):
                    idxs.pop()  # drop the wrap-around duplicate
                    break
            collected[r] = idxs

        union = []
        for r in range(world_size):
            union.extend(collected[r])
        # disjoint: no sample appears for two ranks
        assert len(union) == len(set(union)), "ranks overlap!"
        # exhaustive: union covers exactly [0, S)
        assert sorted(union) == list(range(S)), (
            "4 ranks did not tile all samples", sorted(union)
        )
        # per-rank counts differ by at most 1 (S not divisible by 4)
        counts = [len(collected[r]) for r in range(world_size)]
        assert max(counts) - min(counts) <= 1, counts
        print(f"[selftest] (b) 4-rank disjoint+exhaustive OK (counts={counts})", flush=True)

        # (b2) 2-rank x 2-worker (total_workers=4) also disjoint+exhaustive.
        world_size = 2
        num_workers = 2
        union_w = []
        for r in range(world_size):
            loader = build_mmap_train_dataloader(
                tmp, seq_len=seq_len, batch_size=1, rank=r,
                world_size=world_size, num_workers=num_workers, seed=0,
            )
            per_rank = []
            for bi, batch in enumerate(loader):
                start = int(np.where(flat == int(batch["olmo_ids"].numpy()[0, 0]))[0][0])
                per_rank.append(start // window)
                if len(per_rank) >= S:
                    break
                if len(per_rank) > 1 and len(set(per_rank)) < len(per_rank):
                    per_rank.pop()
                    break
            union_w.extend(per_rank)
        assert len(union_w) == len(set(union_w)), "rank×worker overlap!"
        assert sorted(union_w) == list(range(S)), (
            "2rank×2worker did not tile all samples", sorted(union_w)
        )
        print("[selftest] (b2) 2-rank x 2-worker disjoint+exhaustive OK", flush=True)

        # (c) resume reproduces the uninterrupted suffix.
        # Uninterrupted run for rank 0: collect all epoch-0 samples in order.
        world_size = 4
        ds_full = MmapPackedDataset(tmp, seq_len=seq_len, rank=0, world_size=world_size, seed=0)
        full_seq = []
        for i, sample in enumerate(ds_full):
            start = int(np.where(flat == int(sample["olmo_ids"].numpy()[0]))[0][0])
            full_seq.append(start // window)
            if len(full_seq) > 1 and len(set(full_seq)) < len(full_seq):
                full_seq.pop()
                break
            if len(full_seq) >= S:
                break

        # Simulate: after some global steps, skip_samples GLOBAL positions consumed.
        # Pick skip so it lands mid-stride. Global positions consumed = skip_samples.
        skip_samples = 2 * world_size  # e.g. 2 samples/rank already done -> 8 global
        ds_resume = MmapPackedDataset(
            tmp, seq_len=seq_len, rank=0, world_size=world_size,
            skip_samples=skip_samples, seed=0,
        )
        resume_seq = []
        for i, sample in enumerate(ds_resume):
            start = int(np.where(flat == int(sample["olmo_ids"].numpy()[0]))[0][0])
            resume_seq.append(start // window)
            if len(resume_seq) >= len(full_seq):
                break

        # rank 0 owns permuted positions p with p%4==0: p=0,4,8,12,...
        # skip_samples=8 drops p<8 (p=0,4) -> resume should start at p=8 which is
        # rank0's 3rd owned sample (index 2 in full_seq).
        n_owned_skipped = len([1 for j in range(skip_samples) if j % world_size == 0])
        expected_suffix = full_seq[n_owned_skipped:]
        assert resume_seq[: len(expected_suffix)] == expected_suffix, (
            "resume mismatch", resume_seq[: len(expected_suffix)], expected_suffix
        )
        print(
            f"[selftest] (c) resume OK (skipped {n_owned_skipped} owned samples; "
            f"suffix matches)",
            flush=True,
        )

        print("[selftest] ALL CHECKS PASSED", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
