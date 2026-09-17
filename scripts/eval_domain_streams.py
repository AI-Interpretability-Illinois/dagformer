"""Exploratory stream decomposition of the fixed PR #2 domain-code directions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "interp"))
from behaviors import spec_from_dict
from circuit_common import (Edit, RoutingLayout, RoutingRunner, build_prompt_set,
                            load_models, load_tokenizer, score_behavior)
from interp_common import load_eval_ids


def item_means(prompts, scores):
    """Instruction paraphrases share a content item and are not independent draws."""
    groups = {}
    for prompt, value in zip(prompts, scores, strict=True):
        groups.setdefault(prompt.item, []).append(value)
    return np.array([np.mean(groups[key]) for key in sorted(groups)])


def permute_eligible(direction, eligible, layout, seed):
    """Keep each layer/stream's exact coefficient multiset on eligible edges."""
    rng = np.random.default_rng(seed)
    result = np.zeros_like(direction)
    for layer in range(1, layout.L):
        for stream in ("q", "k", "v", "r"):
            idx = np.flatnonzero(eligible & layout.mask(layers=[layer], streams=[stream]))
            result[idx] = rng.permutation(direction[idx])
    return result


def paired_bootstrap(values, reference, draws, seed):
    delta = np.asarray(values) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(1)
    return {"n": len(delta), "delta": float(delta.mean()),
            "paired_bootstrap_95ci": np.quantile(means, [.025, .975]).tolist(),
            "per_unit_delta": delta.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--transfer-items", required=True, type=Path)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--channels", nargs="+", default=["pred", "corr"])
    ap.add_argument("--streams", nargs="+", default=["all", "q", "k", "v", "r"])
    ap.add_argument("--lams", nargs="+", type=float, default=[.1, .25, .5, 1., 2., 4.])
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cap-batch-size", type=int, default=2)
    ap.add_argument("--cap-windows", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    contrast = torch.load(args.in_dir / "contrast_domain_code.pt", map_location="cpu", weights_only=False)
    old_spec = spec_from_dict(contrast["behavior"])
    new_raw = dict(contrast["behavior"])
    new_raw["items"] = json.loads(args.transfer_items.read_text())["items"]
    new_spec = spec_from_dict(new_raw)
    if {i.content for i in old_spec.items} & {i.content for i in new_spec.items}:
        raise ValueError("Transfer content must not repeat any original item")
    cfg, model, predictor = load_models(args.config, args.ckpt, torch.device(args.device), need_base=True)
    tokenizer = load_tokenizer(cfg, args.tokenizer)
    layout = RoutingLayout(cfg["num_hidden_layers"], cfg["num_attention_heads"])
    runner = RoutingRunner(layout, predictor, model, torch.device(args.device))
    specs = {"original_test": old_spec, "new_content": new_spec}
    sets = {name: build_prompt_set(spec, tokenizer) for name, spec in specs.items()}
    ids, labels = load_eval_ids(args.eval_cache)
    ids, labels = ids[:args.cap_windows, :args.seq_len], labels[:args.cap_windows, :args.seq_len]

    @torch.inference_mode()
    def nlls():
        values = []
        for start in range(0, len(ids), args.cap_batch_size):
            logits = runner.forward(ids[start:start + args.cap_batch_size]).float()
            targets = labels[start:start + args.cap_batch_size].to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
            values.extend(loss.reshape(targets.shape).mean(1).cpu().tolist())
        return values

    baseline_nll = nlls()
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "git_commit": commit, "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
              "complete": False, "protocol": {
                  "status": "exploratory follow-up motivated by the previously inspected stream arms",
                  "direction": "unchanged training-half contrast and circuit masks from PR #2 reproduction",
                  "new_content": "32 fixed new continuation pairs; original instruction paraphrases reused",
                  "metric": "mean per-token log probability of code minus prose continuation; not code correctness",
                  "controls": "same layer/stream coefficient multiset and norm, permuted over eligible hyperconnections; fixed across doses",
                  "uncertainty": "paired bootstrap over content items after averaging instruction paraphrases, or over natural-text windows; unadjusted intervals",
                  "edit_positions": "content and continuation for behavior; all positions for natural-text NLL"},
              "reference_nll": baseline_nll, "channels": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    for channel in args.channels:
        stats = np.load(args.in_dir / f"circuit_domain_code_{channel}_stats.npz")
        chosen, eligible = stats["chosen"].astype(bool), stats["eligible"].astype(bool)
        direction = stats["delta_train"] * chosen
        keep = set(stats["test_items"].tolist())
        prompts = {name: {pol: [p for p in ps.by(pol) if name == "new_content" or p.item in keep]
                          for pol in ("neutral", "pos", "neg")}
                   for name, ps in sets.items()}
        references = {}
        runner.clear_edits()
        for name, spec in specs.items():
            references[name] = {}
            for pol in ("neutral", "pos", "neg"):
                ps = sets[name]
                scores = score_behavior(runner, spec, prompts[name][pol], tokenizer, ps.filler_id, args.batch_size)
                references[name][pol] = item_means(prompts[name][pol], scores["per_item"]).tolist()
        output = {"reference": references, "arms": {}}
        result["channels"][channel] = output
        for stream in args.streams:
            selected = chosen if stream == "all" else chosen & layout.mask(streams=[stream])
            base = direction * selected
            if not selected.any():
                continue
            directions = {"circuit": base}
            directions.update({f"permuted_{i}": permute_eligible(base, eligible, layout, args.seed + i)
                               for i in range(args.controls)})
            for tag, vector in directions.items():
                for lam in args.lams:
                    edit = Edit("add", torch.from_numpy(vector != 0), lam,
                                torch.from_numpy(vector.astype(np.float32)))
                    arm = {"stream": stream, "control": tag, "lam": lam,
                           "n_edges": int(np.count_nonzero(vector)),
                           "direction_norm": float(np.linalg.norm(vector)), "behavior": {}}
                    for name, spec in specs.items():
                        ps = sets[name]
                        mask = torch.zeros(args.seq_len, dtype=torch.bool)
                        mask[ps.span[0]:] = True
                        runner.set_edit(channel, replace(edit, pos_mask=mask))
                        sc = score_behavior(runner, spec, prompts[name]["neutral"], tokenizer, ps.filler_id, args.batch_size)
                        values = item_means(prompts[name]["neutral"], sc["per_item"])
                        arm["behavior"][name] = {"score": float(values.mean()),
                            **paired_bootstrap(values, references[name]["neutral"], args.draws, args.seed)}
                    runner.set_edit(channel, edit)
                    losses = nlls()
                    arm["nll"] = {"mean": float(np.mean(losses)),
                                  **paired_bootstrap(losses, baseline_nll, args.draws, args.seed)}
                    key = f"{stream}/{tag}@{lam:g}"
                    output["arms"][key] = arm
                    print(channel, key, "new shift", arm["behavior"]["new_content"]["delta"],
                          "NLL rise", arm["nll"]["delta"], flush=True)
            save()
        runner.clear_edits()
    result["complete"] = True
    save()
    runner.close()


if __name__ == "__main__":
    main()
