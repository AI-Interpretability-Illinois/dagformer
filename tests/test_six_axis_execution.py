"""Execution checks for data exclusion, token alignment, and checkpoint progress."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_six_axis_data import build_split
from scripts.eval_scaling_checkpoint import progress
from src.data.mmap_dataset import MmapPackedDataset


def test_worker_holdout_excludes_complete_documents_and_preserves_batch_stream(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shards = []
    for worker in range(8):
        directory = source / f"w{worker}"
        directory.mkdir()
        tokens = np.arange(1, 2049, dtype=np.uint32) + worker * 10000
        tokens[15::16] = 999999
        tokens.tofile(directory / "data.bin")
        shards.append({"name": f"w{worker}/data.bin", "n_tokens": len(tokens)})
    index = {"format_version": 1, "dtype": "uint32", "seq_len": 7,
             "eos_id": 999999, "total_tokens": 8 * 2048, "shards": shards}
    (source / "index.json").write_text(json.dumps(index))
    out = tmp_path / "split"
    summary = build_split(source, out, reserve_tokens=512, windows_per_worker=8)
    train = MmapPackedDataset(str(out), seq_len=7, max_samples=8 * 192, block_size=17, seed=42)
    train_tokens = set()
    samples = list(train)
    for item in samples:
        train_tokens.update(item["olmo_ids"].tolist())
        train_tokens.update(item["olmo_labels"].tolist())
    for item in torch.load(out / "dolma_heldout.pt", weights_only=False):
        heldout = set(item["olmo_ids"].flatten().tolist() + item["olmo_labels"].flatten().tolist())
        assert (heldout & train_tokens) <= {999999}
    for worker in summary["strata"]:
        raw = np.fromfile(source / worker["final_shard"], dtype=np.uint32)
        assert raw[worker["training_prefix_tokens"] - 1] == 999999
        assert min(worker["eval_window_offsets"]) >= worker["training_prefix_tokens"]
    # Splitting/restarting at an optimizer boundary must preserve the next batch.
    resumed = list(MmapPackedDataset(str(out), seq_len=7, skip_samples=64,
                                     max_samples=32, block_size=17, seed=42))
    for actual, expected in zip(resumed, samples[64:96]):
        assert torch.equal(actual["olmo_ids"], expected["olmo_ids"])


@pytest.mark.parametrize("family", ["dense", "dag"])
def test_checkpoint_final_and_intermediate_have_unambiguous_update_counts(tmp_path, family):
    script = "pretrain_baseline" if family == "dense" else "pretrain_dagformer"
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(script, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[script] = module
    spec.loader.exec_module(module)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
    args = [str(tmp_path), 1, model]
    if family == "dag":
        args.append(None)
    args += [optimizer, 3.0]
    mid = module.save_checkpoint(*args)
    assert progress(mid)["completed_updates"] == 2
    args[1] = 2  # Final filenames use number of completed updates, not zero-based loop step.
    final = module.save_checkpoint(*args, completed_updates=2)
    assert progress(final)["completed_updates"] == 2
