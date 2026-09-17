"""Measure dependence on external routing and local corrections after training.

Estimate content-free position tables on a separate calibration split. Compare
the same evaluation sequences under frozen, exchanged and dynamic routing.
These are inference interventions on fixed weights, not retrained ablations.
"""
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
from interp_common import (flatten_alpha, identity_alpha, load_elh, load_eval_ids,
                           synthetic_induction_ids, unflatten_alpha)
from interp_editing import layer_chunks


ARMS = [(p, c) for p, c in (
    ("dynamic", "dynamic"), ("position", "dynamic"), ("global", "dynamic"),
    ("other_sequence", "dynamic"), ("shuffle_position", "dynamic"),
    ("identity", "dynamic"), ("dynamic", "zero"), ("position", "zero"),
    ("dynamic", "position"), ("position", "position"),
    ("dynamic", "global"), ("position", "global"))]


def table_batch(table, rows):
    batch, length = rows.shape
    if table.shape[0] not in (1, length) and table.shape[0] < length:
        raise ValueError("Routing table is shorter than the evaluated context")
    return table[:length].unsqueeze(0).expand(batch, length, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calibration-cache", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table-out", type=Path, required=True)
    ap.add_argument("--n-calibration", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--synthetic-sequences", type=int, default=16)
    ap.add_argument("--periods", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--upcast-fp32", action="store_true",
                    help="repeat with FP32 arithmetic on the loaded weights")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    model, predictor = elh.load_fourway(args.ckpt, cfg, device)
    if args.upcast_fp32:
        torch.set_float32_matmul_precision("highest")
        model.float()
        predictor.float()
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    calibration, _ = load_eval_ids(args.calibration_cache)
    evaluation, labels = load_eval_ids(args.eval_cache)
    if len(calibration) < args.n_calibration or len(evaluation) < args.n_eval:
        raise ValueError("Requested more sequences than the cache contains")
    calibration = calibration[:args.n_calibration]
    evaluation, labels = evaluation[:args.n_eval], labels[:args.n_eval]
    if calibration.shape[1] != evaluation.shape[1]:
        raise ValueError("Calibration and evaluation context lengths must match")
    # The split must be separate, including when both are slices of one cache.
    calibration_rows = {tuple(row.tolist()) for row in calibration}
    if any(tuple(row.tolist()) in calibration_rows for row in evaluation):
        raise ValueError("Calibration and evaluation contain identical sequences")
    length = evaluation.shape[1]
    state = {"calibrating": True, "corr": "dynamic"}
    corr_sums = [None] * (L - 1)
    handles = []
    tables = {}
    for index, mlp in enumerate(model.correction_mlps):
        def hook(module, inputs, output, index=index):
            if state["calibrating"]:
                value = output.detach().double().sum(0)
                corr_sums[index] = value if corr_sums[index] is None else corr_sums[index] + value
            elif state["corr"] != "dynamic":
                if state["corr"] == "zero":
                    return torch.zeros_like(output)
                table = tables["corr_" + state["corr"]]
                start, end = chunks[index]
                return table_batch(table[..., start:end], output[..., 0]).to(output.dtype)
            return output
        handles.append(mlp.register_forward_hook(hook))

    @torch.no_grad()
    def calibrate():
        total = None
        for row in calibration:
            ids = row[None].to(device)
            routing = predictor(ids)
            flat = flatten_alpha(routing).double().sum(0)
            total = flat if total is None else total + flat
            model(ids, routing)
        tables["pred_position"] = (total / len(calibration)).float()
        tables["corr_position"] = (torch.cat(corr_sums, -1) / len(calibration)).float()
        for channel in ("pred", "corr"):
            tables[channel + "_global"] = tables[channel + "_position"].mean(0, keepdim=True)

    @torch.no_grad()
    def forward(rows, pred_mode, donor=None, permutation=None):
        rows = rows.to(device)
        if pred_mode == "identity":
            routing = identity_alpha(*rows.shape, L, H, device)
        elif pred_mode in ("position", "global"):
            routing = unflatten_alpha(table_batch(tables["pred_" + pred_mode], rows), L, H)
        else:
            routing = predictor(rows if donor is None else donor.to(device))
            if pred_mode == "shuffle_position":
                flat = flatten_alpha(routing)[:, permutation]
                routing = unflatten_alpha(flat, L, H)
        return model(rows, routing).float()

    try:
        calibrate()
        state["calibrating"] = False
        # Keep CPU copies for later ordinary benchmark interventions.
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        table_metadata = {"checkpoint": args.ckpt, "config": args.config,
                          "calibration_cache": args.calibration_cache,
                          "n_calibration": len(calibration), "L": L, "H": H,
                          "upcast_fp32": args.upcast_fp32,
                          "float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
        torch.save({**table_metadata, "tables": {k: v.cpu() for k, v in tables.items()}}, args.table_out)
        generator = torch.Generator().manual_seed(args.seed)
        permutations = [torch.randperm(length, generator=generator).to(device) for _ in evaluation]
        synthetic = {}
        for period in args.periods:
            ids, _ = synthetic_induction_ids(calibration, args.synthetic_sequences, length,
                                              period, args.seed + period)
            synthetic[period] = ids
        result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
                  "protocol": "fixed-weight inference interventions; calibration/eval sequences disjoint",
                  "calibration": table_metadata, "arms": {}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for pred_mode, corr_mode in ARMS:
            state["corr"] = corr_mode
            nll = []
            for index, (row, target) in enumerate(zip(evaluation, labels)):
                donor = evaluation[(index + 1) % len(evaluation)][None] if pred_mode == "other_sequence" else None
                logits = forward(row[None], pred_mode, donor, permutations[index])
                nll.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.to(device)).item())
            name = f"pred_{pred_mode}/corr_{corr_mode}"
            if not result["arms"]:
                reference = nll
            record = {"nll": nll, "summary": paired_summary(nll, reference), "copy": {}}
            for period, ids in synthetic.items():
                accuracy, copy_nll = [], []
                for index, row in enumerate(ids):
                    donor = ids[(index + 1) % len(ids)][None] if pred_mode == "other_sequence" else None
                    logits = forward(row[None], pred_mode, donor, permutations[index % len(permutations)])
                    scores = logits[0, length // 2:-1]
                    target = row[length // 2 + 1:].to(device)
                    accuracy.append((scores.argmax(-1) == target).float().mean().item())
                    copy_nll.append(F.cross_entropy(scores, target).item())
                record["copy"][str(period)] = {"accuracy": accuracy, "nll": copy_nll,
                                               "mean_accuracy": float(np.mean(accuracy)),
                                               "mean_nll": float(np.mean(copy_nll))}
            result["arms"][name] = record
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(name, json.dumps(record["summary"]),
                  "copy", {p: round(v["mean_accuracy"], 4) for p, v in record["copy"].items()}, flush=True)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
