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


@pytest.mark.parametrize("before,after,expected,last", [
    ("JobState=RUNNING", "", False, "show"),
    ("JobState=PENDING Restarts=1 RunTime=00:00:00", "", False, "show"),
    ("JobState=PENDING Restarts=0 RunTime=00:00:00", "JobState=PENDING Restarts=0 RunTime=00:00:00 Reason=JobHeldUser ArrayTaskId=2-27", False, "release"),
    ("JobState=PENDING Restarts=0 RunTime=00:00:00", "JobState=PENDING Restarts=0 RunTime=00:00:00 Reason=JobHeldUser ArrayTaskId=2 ", True, "scancel"),
])
def test_local_dispatch_only_cancels_an_individually_held_pending_task(monkeypatch, before, after, expected, last):
    from scripts import dispatch_six_axis_local as dispatcher
    calls = []
    states = iter([before, after])

    def remote(host, *args):
        calls.append(args)
        return next(states) if args[:2] == ("scontrol", "show") else ""

    monkeypatch.setattr(dispatcher, "remote", remote)
    assert dispatcher.take_pending("delta", 12345, 2) is expected
    final_action = calls[-1][0] if calls[-1][0] == "scancel" else calls[-1][1]
    assert final_action == last
    assert all("12345_2" in call for call in calls)


@pytest.mark.parametrize("cached_remote", [False, True])
def test_local_report_refreshes_when_remote_authentication_fails(tmp_path, monkeypatch, cached_remote):
    from types import SimpleNamespace
    from scripts import dispatch_six_axis_local as dispatcher

    snapshot = tmp_path / "delta_snapshot"
    snapshot.mkdir()
    if cached_remote:
        (snapshot / "manifest.json").write_text("{}")
    report = tmp_path / "report"
    report.mkdir()
    (report / "remote_sync.json").write_text(json.dumps({"last_successful_pull_unix": 123}))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "rsync":
            raise dispatcher.subprocess.CalledProcessError(255, command)
        (report / "results.json").write_text('{"updated": true}')

    monkeypatch.setattr(dispatcher.subprocess, "run", run)
    assert dispatcher.sync_report(
        SimpleNamespace(host="delta", remote_root="/study", remote_checkpoints="/models"), tmp_path) is False
    assert json.loads((report / "results.json").read_text())["updated"]
    sync = json.loads((report / "remote_sync.json").read_text())
    assert sync["remote_available"] is False
    assert sync["last_successful_pull_unix"] == 123
    assert sync["error"]
    assert sum(command[0] == "rsync" for command in calls) == 1
    assert (str(snapshot / "manifest.json") in calls[-1]) == cached_remote


@pytest.mark.parametrize("stage_fails", [False, True])
def test_requeue_is_copied_before_cancellation_and_released_if_copy_fails(monkeypatch, stage_fails):
    from scripts import dispatch_six_axis_local as dispatcher
    calls = []
    states = iter(["JobState=PENDING Restarts=1 RunTime=00:00:00",
                   "JobState=PENDING Restarts=1 RunTime=00:00:00 Reason=JobHeldUser ArrayTaskId=2 "])

    def remote(host, *args):
        calls.append(args[0] if args[0] == "scancel" else args[1])
        return next(states) if args[:2] == ("scontrol", "show") else ""

    def stage():
        calls.append("copy_and_validate")
        if stage_fails:
            raise ValueError("Missing companion checkpoint")

    monkeypatch.setattr(dispatcher, "remote", remote)
    if stage_fails:
        with pytest.raises(ValueError, match="Missing companion"):
            dispatcher.take_pending("delta", 12345, 2, stage)
    else:
        assert dispatcher.take_pending("delta", 12345, 2, stage)
    assert calls == ["show", "hold", "show", "copy_and_validate", "release" if stage_fails else "scancel"]


def test_running_job_is_not_transferred_even_when_checkpoint_migration_is_enabled(monkeypatch):
    from scripts import dispatch_six_axis_local as dispatcher
    monkeypatch.setattr(dispatcher, "remote", lambda *args: "JobState=RUNNING Restarts=1 RunTime=00:02:00")
    assert not dispatcher.take_pending("delta", 12345, 2, lambda: pytest.fail("Running job was staged"))


@pytest.mark.parametrize("bad_progress", [False, True])
def test_checkpoint_relocation_preserves_optimizer_and_resolves_companion(tmp_path, bad_progress):
    from scripts.stage_six_axis_checkpoint import relocate_resume_checkpoint
    companion = tmp_path / "checkpoint_step6_model.pt"
    torch.save({"weight": torch.arange(4, dtype=torch.bfloat16)}, companion)
    checkpoint = tmp_path / "checkpoint_step6.pt"
    moment = torch.tensor([.5, .25, .125, .0625], dtype=torch.bfloat16)
    state = {"step": 6, "completed_updates": 7,
             "optimizer_state_dict": {"state": {0: {"step": torch.tensor(8. if bad_progress else 7.),
                                                        "exp_avg": moment, "exp_avg_sq": moment.square()}},
                                      "param_groups": [{"params": [0], "lr": .0001}]},
             "model_state_path": "/remote/models/checkpoint_step6_model.pt",
             "predictor_state_dict": {"weight": torch.ones(2)},
             "routing_state_dict": {"weight": torch.ones(3)}}
    torch.save(state, checkpoint)
    if bad_progress:
        with pytest.raises(ValueError, match="Invalid resume progress"):
            relocate_resume_checkpoint(checkpoint, 10, "fourway_corrected")
        return
    assert relocate_resume_checkpoint(checkpoint, 10, "fourway_corrected") == 7
    loaded = torch.load(checkpoint, weights_only=False)
    assert loaded["model_state_path"] == str(companion)
    assert loaded["completed_updates"] == 7
    assert loaded["optimizer_state_dict"]["param_groups"] == state["optimizer_state_dict"]["param_groups"]
    assert torch.equal(loaded["optimizer_state_dict"]["state"][0]["exp_avg"], moment)
    assert torch.equal(torch.load(loaded["model_state_path"], weights_only=False)["weight"], torch.arange(4, dtype=torch.bfloat16))
