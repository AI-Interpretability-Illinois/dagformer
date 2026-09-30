"""Materialize the approved first-seed sweep, with metadata outside HDD storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default=str(ROOT / "experiments/scaling_ebt_20260930/six_axis_plan.json"))
    p.add_argument("--metadata-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--data-index", required=True)
    p.add_argument("--monitor-cache", required=True)
    p.add_argument("--eval", action="append", required=True, help="name=cache.pt")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", default="")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    meta = Path(args.metadata_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    meta.mkdir(parents=True, exist_ok=True)
    config_dir = meta / "configs"
    config_dir.mkdir(exist_ok=True)
    evals = dict(item.split("=", 1) for item in args.eval)
    revision = args.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    settings = list(plan["new_small_sweep"])
    # Start with the shared center, followed by the depth series.
    settings.sort(key=lambda x: (0 if len(x["panels"]) == 4 else 1 if "depth" in x["panels"] else 2))
    runs = []
    for setting in settings:
        for family in ("dense", "fourway_corrected"):
            name = f"{family}_{setting['id']}_s{args.seed}"
            run_meta = meta / "runs" / name
            run_meta.mkdir(parents=True, exist_ok=True)
            cache = run_meta / "eval_cache.pt"
            if not cache.exists():
                cache.symlink_to(Path(args.monitor_cache).resolve())
            denom = args.world_size * args.micro_batch
            if setting["batch_sequences"] % denom:
                raise ValueError(f"Global batch is not divisible by world*micro: {setting['id']}")
            warmup = plan["recipe"]["warmup_tokens"] // setting["batch_tokens"]
            cfg = dict(hidden_size=setting["width"], num_hidden_layers=setting["layers"],
                       num_attention_heads=setting["heads"], intermediate_size=setting["ffn_width"],
                       vocab_size=100352, tie_word_embeddings=True, max_position_embeddings=4096,
                       tokenizer_id=args.tokenizer, dataset="allenai/dolma", dataset_name="v1_7",
                       data_source="mmap", mmap_index_path=str(Path(args.data_index).resolve()),
                       data_num_workers=0, data_block_size=1024, seq_len=1024, seed=args.seed,
                       micro_batch_size=args.micro_batch,
                       gradient_accumulation_steps=setting["batch_sequences"] // denom,
                       total_steps=setting["optimizer_updates"], lr=5e-4, beta1=0.9, beta2=0.95,
                       weight_decay=0.1, warmup_steps=warmup, max_grad_norm=1.0, lr_schedule="linear",
                       eval_size=128, eval_every=max(1, 262144000 // setting["batch_tokens"]),
                       log_every=max(1, 5242880 // setting["batch_tokens"]),
                       save_every=max(1, 131072000 // setting["batch_tokens"]), keep_last_n=1,
                       save_dir=str(checkpoint_root / name), metadata_dir=str(run_meta),
                       wandb_project="dagformer", wandb_run_name=name)
            if family == "fourway_corrected":
                cfg.update({k: v for k, v in plan["architecture"].items()
                            if k.startswith("predictor_") or k in ("fourway_hidden", "correction_hidden", "use_v_norm")})
                cfg.update(routing_mode=family, replace_rmsnorm=True, use_torch_compile=False,
                           predictor_lr=3e-4, resume_require_optimizer=True)
            config = config_dir / f"{name}.yaml"
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            (run_meta / "config.yaml").write_text(config.read_text())
            runs.append({"name": name, "family": family, "config": str(config),
                         "metadata_dir": str(run_meta), "checkpoint_dir": cfg["save_dir"],
                         "world_size": args.world_size, "seed": args.seed, "revision": revision,
                         **setting})
    manifest = {"revision": revision, "metadata_root": str(meta),
                "evals": evals, "runs": runs,
                "training_data_index": str(Path(args.data_index).resolve())}
    (meta / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(runs)} runs to {meta / 'manifest.json'}")


if __name__ == "__main__":
    main()
