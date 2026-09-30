"""End-to-end CPU smoke test of scripts/prune_finetune.py on a tiny random
model, for both the dense and the FourWay path. Builds its own synthetic mmap
corpus + eval cache so it needs no checkpoint and no HF access.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from scripts.pretokenize import SHARD_DTYPE, ShardWriter, write_index  # noqa: E402

V, SEQ = 512, 32


def make_data(root: str, n_windows: int = 64) -> tuple[str, str]:
    rng = np.random.default_rng(0)
    train_dir = os.path.join(root, "train")
    w = ShardWriter(train_dir, shard_tokens=10_000)
    for _ in range(n_windows):
        w.add(rng.integers(0, V, size=SEQ + 1, dtype=SHARD_DTYPE))
    w.close()
    write_index(train_dir, SEQ, eos_id=0, tokenizer_id="none", dataset="synthetic",
                dataset_version="", seed=0, total_tokens=w.total_tokens, shards=w.shards,
                provenance={})
    batches = []
    for _ in range(2):
        ids = torch.from_numpy(rng.integers(0, V, size=(4, SEQ + 1)).astype(np.int64))
        batches.append({"olmo_ids": ids[:, :SEQ], "olmo_labels": ids[:, 1:], "raw_text": [""] * 4})
    eval_path = os.path.join(root, "eval_cache.pt")
    torch.save(batches, eval_path)
    return train_dir, eval_path


def model_yaml(path: str, fourway: bool, modular: bool = False) -> None:
    cfg = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
               intermediate_size=64, vocab_size=V, tie_word_embeddings=True,
               max_position_embeddings=64, replace_rmsnorm=True)
    if fourway:
        cfg.update(routing_mode="fourway_modular_corrected" if modular else "fourway_corrected",
                   predictor_encoder_dim=16, predictor_encoder_layers=1, predictor_encoder_heads=2,
                   predictor_max_seq_len=64, fourway_hidden=16, correction_hidden=8)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


@pytest.mark.parametrize("fourway", [False, True])
def test_prune_finetune_end_to_end(tmp_path, fourway):
    train_dir, eval_path = make_data(str(tmp_path / "data"))
    mcfg = str(tmp_path / "model.yaml")
    model_yaml(mcfg, fourway)
    save_dir = str(tmp_path / "run")
    cfg = dict(
        model_config=mcfg, checkpoint="", train_index_path=train_dir,
        eval_cache_path=eval_path, general_eval_cache_path=eval_path, seq_len=SEQ,
        micro_batch_size=2, gradient_accumulation_steps=2, total_steps=12,
        lr=1e-3, predictor_lr=1e-3, warmup_steps=2, data_num_workers=0,
        prune_units=["head", "neuron", "mlp"], target_sparsity=0.5,
        target_sparsity_by_type={"mlp": 0.5}, prune_start_step=3, prune_end_step=9,
        prune_every=3, importance="taylor", min_alive_per_layer={"head": 1},
        eval_every=4, log_every=1, save_dir=save_dir, save_every=5,
    )
    pcfg = str(tmp_path / "prune.yaml")
    with open(pcfg, "w") as f:
        yaml.safe_dump(cfg, f)
    env = dict(os.environ, PYTHONPATH=REPO, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "scripts/prune_finetune.py", "--config", pcfg,
                           "--override", "label_smoothing=0.0"],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    summary = json.load(open(os.path.join(save_dir, "summary.json")))
    assert summary["model"] == ("fourway" if fourway else "dense")
    assert summary["steps"] == 12
    counts = summary["unit_counts"]
    assert counts["head"] == [4, 8]           # 50% of 2x4 heads (min_alive keeps >=1 per layer)
    assert counts["neuron"] == [64, 128]
    assert counts["mlp"] == [1, 2]
    assert 0 < summary["base_params_remaining"] < summary["base_params_total"]

    traj = json.load(open(os.path.join(save_dir, "trajectory.json")))
    tags = [t["tag"] for t in traj]
    assert tags[0] == "initial" and tags[-1] == "final"
    assert tags.count("post_prune") == 3      # steps 3, 6, 9
    sparsities = [t["prune/base_param_sparsity"] for t in traj]
    assert all(b >= a for a, b in zip(sparsities, sparsities[1:]))
    assert sparsities[-1] > 0.25

    final = os.path.join(save_dir, "final")
    assert os.path.exists(os.path.join(final, "checkpoint.pt"))
    assert os.path.exists(os.path.join(final, "masks.pt"))
    assert os.path.exists(os.path.join(final, "config.yaml"))
    assert not os.path.exists(os.path.join(save_dir, "latest"))
    ck = torch.load(os.path.join(final, "checkpoint.pt"), map_location="cpu", weights_only=False)
    if fourway:
        assert "predictor_state_dict" in ck and os.path.exists(ck["model_state_path"])
    else:
        assert "model_state_dict" in ck
    # metrics.csv has the eval and prune columns
    header = open(os.path.join(save_dir, "metrics.csv")).readline()
    assert "eval/domain_nll" in header and "prune/base_param_sparsity" in header


def test_prune_finetune_modular_routing_column(tmp_path):
    """Modular DAGFormer with whole-module units scored by routing column mass."""
    train_dir, eval_path = make_data(str(tmp_path / "data"))
    mcfg = str(tmp_path / "model.yaml")
    model_yaml(mcfg, fourway=True, modular=True)
    save_dir = str(tmp_path / "run")
    cfg = dict(
        model_config=mcfg, checkpoint="", train_index_path=train_dir,
        eval_cache_path=eval_path, seq_len=SEQ, micro_batch_size=2,
        gradient_accumulation_steps=1, total_steps=8, lr=1e-3, predictor_lr=1e-3,
        warmup_steps=1, data_num_workers=0, prune_units=["attn", "mlp", "head"],
        target_sparsity=0.5, prune_start_step=2, prune_end_step=6, prune_every=2,
        importance="routing_column", eval_every=4, log_every=1, save_dir=save_dir,
    )
    pcfg = str(tmp_path / "prune.yaml")
    with open(pcfg, "w") as f:
        yaml.safe_dump(cfg, f)
    env = dict(os.environ, PYTHONPATH=REPO, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "scripts/prune_finetune.py", "--config", pcfg],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    summary = json.load(open(os.path.join(save_dir, "summary.json")))
    assert summary["model"] == "fourway"
    assert summary["unit_counts"]["attn"] == [1, 2]
    assert summary["unit_counts"]["mlp"] == [1, 2]
    assert summary["unit_counts"]["head"] == [4, 8]
    # the final checkpoint round-trips through the project's verified loader
    sys.path.insert(0, REPO)
    from scripts.eval_lm_harness import load_fourway
    fw, pred = load_fourway(os.path.join(save_dir, "final", "checkpoint.pt"),
                            yaml.safe_load(open(os.path.join(save_dir, "final", "config.yaml"))),
                            torch.device("cpu"))
    ids = torch.randint(0, V, (1, SEQ))
    with torch.no_grad():
        logits = fw(ids, pred(ids))
    assert logits.shape == (1, SEQ, V) and torch.isfinite(logits.float()).all()
