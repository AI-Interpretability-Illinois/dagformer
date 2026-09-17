"""Match a dense checkpoint to an existing routing-dependence evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from eval_context_fidelity import paired_summary
from interp_common import load_elh, load_eval_ids, synthetic_induction_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--routing-result", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    routed = json.loads(args.routing_result.read_text())
    settings = routed["args"]
    if settings.get("upcast_fp32", False):
        raise ValueError("Use the BF16 routing reference for this dense comparison")
    device = torch.device(args.device)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    model = elh.load_dense(args.ckpt, cfg, device)
    calibration, _ = load_eval_ids(settings["calibration_cache"])
    ids, labels = load_eval_ids(settings["eval_cache"])
    calibration = calibration[:settings["n_calibration"]]
    ids, labels = ids[:settings["n_eval"]], labels[:settings["n_eval"]]
    reference = routed["arms"]["pred_dynamic/corr_dynamic"]
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "git_commit": commit, "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
              "num_parameters": sum(p.numel() for p in model.parameters()),
              "protocol": "same natural-text windows and periodic token sequences as the routed reference; copy is teacher-forced next-token accuracy over the second half of each sequence",
              "nll": [], "copy": {}, "paired_dagformer_minus_dense": {"copy": {}}}
    with torch.inference_mode():
        for row, target in zip(ids, labels):
            logits = model(input_ids=row[None].to(device)).logits.float()
            result["nll"].append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.to(device))))
        result["mean_nll"] = float(np.mean(result["nll"]))
        result["paired_dagformer_minus_dense"]["natural_nll"] = paired_summary(reference["nll"], result["nll"])
        length = ids.shape[1]
        for period in settings["periods"]:
            copy_ids, _ = synthetic_induction_ids(calibration, settings["synthetic_sequences"], length,
                                                  period, settings["seed"] + period)
            accuracy, losses = [], []
            for row in copy_ids:
                logits = model(input_ids=row[None].to(device)).logits.float()[0, length // 2:-1]
                target = row[length // 2 + 1:].to(device)
                accuracy.append(float((logits.argmax(-1) == target).float().mean()))
                losses.append(float(F.cross_entropy(logits, target)))
            result["copy"][str(period)] = {"accuracy": accuracy, "nll": losses,
                                          "mean_accuracy": float(np.mean(accuracy)), "mean_nll": float(np.mean(losses))}
            result["paired_dagformer_minus_dense"]["copy"][str(period)] = {
                "accuracy": paired_summary(reference["copy"][str(period)]["accuracy"], accuracy),
                "nll": paired_summary(reference["copy"][str(period)]["nll"], losses)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("natural NLL", result["mean_nll"], "copy accuracy",
          {p: r["mean_accuracy"] for p, r in result["copy"].items()}, flush=True)


if __name__ == "__main__":
    main()
