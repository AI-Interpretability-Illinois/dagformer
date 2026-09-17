"""Check checkpoint optimizer compatibility and finite gradients without taking a step."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from interp_common import load_elh, load_eval_ids


def audit(root: Path, name: str, ids: torch.Tensor, labels: torch.Tensor) -> dict:
    elh = load_elh()
    directory = root / "checkpoints/source" / name
    cfg = elh.load_config(str(directory / "config.yaml"))
    model, predictor = elh.load_fourway(str(directory / "checkpoint.pt"), cfg, "cuda")
    model.train().requires_grad_(True)
    predictor.train().requires_grad_(True)
    # Match the trainer's ordering, including the no-weight-decay routing biases.
    base = list(model.olmo.parameters())
    biases = [p for n, p in predictor.named_parameters() if "layer_biases" in n]
    others = [p for n, p in predictor.named_parameters() if "layer_biases" not in n]
    others += list(model.get_routing_parameters())
    groups = [{"params": p} for p in (base, others, biases) if p]
    optimizer = torch.optim.AdamW(groups)
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu",
                            mmap=True, weights_only=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    updates = sorted({int(s["step"]) for s in optimizer.state.values() if "step" in s})
    del checkpoint
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids, predictor(ids))
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
    loss.backward()
    checks = {}
    for label, parameters in (("backbone", base), ("predictor_and_correction", others),
                              ("routing_biases", biases)):
        if not parameters:
            continue
        gradients = [p.grad for p in parameters if p.grad is not None]
        finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
        norm = sum(float(g.detach().float().square().sum()) for g in gradients) ** 0.5
        checks[label] = {"finite": finite, "gradient_tensors": len(gradients), "norm": norm}
        if not finite or norm == 0:
            raise RuntimeError(f"Invalid gradients for {name}/{label}: {checks[label]}")
    result = {"model": name, "optimizer_update_counts": updates,
              "optimizer_state_tensors": len(optimizer.state),
              "loss": float(loss.detach()), "gradients": checks,
              "optimizer_steps_taken": 0}
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ids, labels = load_eval_ids(str(args.root / "data/eval_corpora/wikitext_train.pt"))
    ids, labels = ids[:1].cuda(), labels[:1].cuda()
    torch.manual_seed(20260917)
    result = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "shape": list(ids.shape), "models": []}
    for name in ("1b-dagformer", "600m-dagformer"):
        result["models"].append(audit(args.root, name, ids, labels))
        gc.collect()
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
