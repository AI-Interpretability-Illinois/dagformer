"""Read original FourWay checkpoints to check predictor optimizer participation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch


def clean_keys(state):
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def inspect_checkpoint(record):
    source = Path(record["source"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    predictor = clean_keys(checkpoint["predictor_state_dict"])
    routing = clean_keys(checkpoint["routing_state_dict"])
    optimizer = checkpoint["optimizer_state_dict"]
    groups = optimizer["param_groups"]
    if len(groups) != 3 or "embed.weight" not in predictor:
        raise ValueError("This audit expects the full encoder's three optimizer groups")
    other = [(f"predictor.{key}", value) for key, value in predictor.items()
             if not key.startswith("layer_biases.")]
    corrections = [(key, value) for key, value in routing.items()
                   if key.startswith("correction_mlps.")]
    norms = [(key, value) for key, value in routing.items() if key.startswith("v_norms.")]
    biases = [(f"predictor.{key}", value) for key, value in predictor.items()
              if key.startswith("layer_biases.")]
    mapped = [(1, other + corrections + norms), (2, biases)]
    parameters = []
    components = defaultdict(list)
    for index, named in mapped:
        group = groups[index]
        if len(group["params"]) != len(named):
            raise ValueError(f"{record['model']}: group {index} parameter count does not match")
        for parameter_id, (name, weight) in zip(group["params"], named):
            state = optimizer["state"].get(parameter_id)
            if state is None:
                raise ValueError(f"{name}: optimizer state is absent")
            moments = [state[key] for key in ("exp_avg", "exp_avg_sq")]
            if any(moment.shape != weight.shape for moment in moments):
                raise ValueError(f"{name}: optimizer moment shape does not match the parameter")
            step = int(state["step"])
            row = {"name": name, "optimizer_group": index, "shape": list(weight.shape),
                   "num_parameters": weight.numel(), "updates": step,
                   "moment_dtypes": [str(value.dtype) for value in moments],
                   "moments_finite": all(bool(torch.isfinite(value).all()) for value in moments),
                   "nonzero_first_moment": int(torch.count_nonzero(moments[0])),
                   "nonzero_second_moment": int(torch.count_nonzero(moments[1]))}
            if name.startswith("predictor.layer_heads."):
                row["nonzero_weight_from_zero_init"] = int(torch.count_nonzero(weight))
            parameters.append(row)
            component = name.split(".")[1] if name.startswith("predictor.") else name.split(".")[0]
            components[component].append(row)
    aggregate = {}
    for name, rows in components.items():
        aggregate[name] = {
            "parameter_tensors": len(rows), "num_parameters": sum(r["num_parameters"] for r in rows),
            "updates_min": min(r["updates"] for r in rows),
            "updates_max": max(r["updates"] for r in rows),
            "all_moments_finite": all(r["moments_finite"] for r in rows),
            "tensors_with_nonzero_first_moment": sum(r["nonzero_first_moment"] > 0 for r in rows),
            "tensors_with_nonzero_second_moment": sum(r["nonzero_second_moment"] > 0 for r in rows),
        }
    return {"model": record["model"], "source": str(source), "checkpoint_step": checkpoint["step"],
            "mapping": "training optimizer order: predictor except layer_biases, then correction/v_norms; separate layer_biases group; all counts and shapes checked",
            "groups": [{"index": i, "tensors": len(g["params"]), "saved_lr": g["lr"],
                        "weight_decay": g["weight_decay"]} for i, g in enumerate(groups)],
            "components": aggregate, "parameters": parameters}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    records = {r["model"]: r for r in json.loads(args.manifest.read_text())["records"]}
    result = {"notes": [
        "Read-only CPU audit of original optimizer states; no training or checkpoint writes.",
        "Nonzero moments establish gradient/update history, not that optimization converged or every training detail was correct.",
        "Sparse or unused embedding coordinates may have zero moments even when their containing tensor is updated.",
    ], "records": []}
    for name in args.models:
        result["records"].append(inspect_checkpoint(records[name]))
        print(f"Checked {name}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
