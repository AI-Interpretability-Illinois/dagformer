"""Retest the six historical copy heads on new repetition and pattern-break inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

import torch
import torch.nn.functional as F

from eval_context_fidelity import apply_edges, norm_matched_edit, paired_summary
from interp_common import (flatten_alpha, load_elh, load_eval_ids,
                           synthetic_induction_ids, unflatten_alpha)
from interp_editing import layer_chunks


def matched_heads(named, num_heads, seed):
    groups = defaultdict(list)
    for layer, head in named:
        groups[layer].append(head)
    rng = random.Random(seed)
    return [(layer, head) for layer, selected in sorted(groups.items())
            for head in rng.sample([h for h in range(num_heads) if h not in selected], len(selected))]


def head_edges(named):
    return [(layer, stream, head, source) for layer, head in named
            for stream in ("q", "k") for source in range(layer + 1)]


def next_token_metrics(logits, rows, first_target):
    """Target at index j is predicted by logits at j-1, including the block boundary."""
    scores = logits[:, first_target - 1:-1].float()
    target = rows[:, first_target:].to(scores.device)
    losses = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1),
                             reduction="none").reshape(target.shape)
    return {"accuracy": (scores.argmax(-1) == target).float().mean(1).cpu().tolist(),
            "nll": losses.mean(1).cpu().tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--selection", default="experiments/results/interp/localize_step9000.json")
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-sequences", type=int, default=32)
    ap.add_argument("--n-nll", type=int, default=50)
    ap.add_argument("--periods", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0., .5, 1.5, 2.])
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if any(args.seq_len % p for p in args.periods):
        raise ValueError("Repetition periods must divide the sequence length")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = json.loads(Path(args.selection).read_text())
    names = source["stageC_gamma_sweep"]["named_heads"]
    named = [(int(n.split("/")[0][1:]), int(n.split("/h")[1])) for n in names]
    device = torch.device(args.device)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    model, predictor = elh.load_fourway(args.ckpt, cfg, device)
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    if any(layer >= L or head >= H for layer, head in named):
        raise ValueError("Historical heads do not fit the checkpoint")
    chunks = layer_chunks(L, H)
    token_ids, _ = load_eval_ids(args.token_cache)
    natural_ids, natural_labels = load_eval_ids(args.eval_cache)
    natural_ids = natural_ids[:args.n_nll, :args.seq_len]
    natural_labels = natural_labels[:args.n_nll, :args.seq_len]
    repeat = {p: synthetic_induction_ids(token_ids, args.n_sequences, args.seq_len,
                                         p, args.seed + p)[0] for p in args.periods}
    break_period, break_at = 128, 384
    if args.seq_len <= break_at:
        raise ValueError("Pattern-break inputs require a sequence longer than 384")
    break_repeat, break_random = synthetic_induction_ids(
        token_ids, args.n_sequences, args.seq_len, break_period, args.seed + 10000)
    broken = torch.cat([break_repeat[:, :break_at], break_random[:, break_at:]], 1)
    controls = {f"random{i}": matched_heads(named, H, args.seed + i) for i in range(args.controls)}
    reference_edges = head_edges(named)
    state = {"channel": "both", "gamma": 1., "edges": reference_edges, "control": False}

    def edit(chunk, layer):
        if state["control"]:
            return norm_matched_edit(chunk, layer, H, state["edges"], reference_edges, state["gamma"])
        return apply_edges(chunk, layer, H, state["edges"], state["gamma"])

    handles = []
    for index, mlp in enumerate(model.correction_mlps):
        def hook(module, inputs, output, layer=index + 1):
            return edit(output, layer) if state["channel"] in ("corr", "both") else output
        handles.append(mlp.register_forward_hook(hook))

    @torch.inference_mode()
    def forward(rows):
        rows = rows.to(device)
        routing = predictor(rows)
        if state["channel"] in ("pred", "both") and state["gamma"] != 1.:
            flat = flatten_alpha(routing)
            routing = unflatten_alpha(torch.cat([edit(flat[..., lo:hi], i + 1)
                                                for i, (lo, hi) in enumerate(chunks)], -1), L, H)
        return model(rows, routing).float()

    def evaluate():
        values = defaultdict(list)
        for start in range(0, len(natural_ids), args.batch_size):
            logits = forward(natural_ids[start:start + args.batch_size])
            targets = natural_labels[start:start + args.batch_size].to(device)
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                     reduction="none").reshape(targets.shape)
            values["natural_nll"].extend(losses.mean(1).cpu().tolist())
        for period, rows in repeat.items():
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                logits = forward(batch)
                for label, first in (("after_first_block", period), ("second_half", args.seq_len // 2)):
                    for metric, vals in next_token_metrics(logits, batch, first).items():
                        values[f"period{period}/{label}/{metric}"].extend(vals)
        for start in range(0, len(broken), args.batch_size):
            batch = broken[start:start + args.batch_size]
            logits = forward(batch)
            for metric, vals in next_token_metrics(logits, batch, break_at).items():
                values[f"pattern_break/{metric}"].extend(vals)
            scores = logits[:, break_at - 1:-1]
            targets = batch[:, break_at:].to(device)
            lagged = batch[:, break_at - break_period:args.seq_len - break_period].to(device)
            different = lagged != targets
            probability = (scores.gather(-1, lagged[..., None]).squeeze(-1)
                           - scores.logsumexp(-1)).exp()
            for name, matrix in (("false_lag_probability", probability),
                                 ("false_lag_argmax", (scores.argmax(-1) == lagged).float())):
                means = (matrix * different).sum(1) / different.sum(1)
                values[f"pattern_break/{name}"].extend(means.cpu().tolist())
        return dict(values)

    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "git_commit": commit, "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
              "complete": False, "named_heads": names, "control_heads": controls,
              "protocol": {
                  "selection": "six heads fixed by the historical localization JSON; no reselection on these inputs",
                  "edit": "scale Q and K deviations from the original head mean at all sources and positions",
                  "control": "same number of heads per layer, excluding selected heads; each layer/token edit norm matched to the named set on incoming activations",
                  "synthetic": "new blocks sampled from the empirical marginal of WikiText training tokens, not contiguous natural text",
                  "copy": "teacher-forced next-token prediction after the first block and separately over the second half",
                  "pattern_break": "three 128-token repetitions, then independent marginal tokens; false-lag metrics omit positions whose true token equals the lagged token",
                  "uncertainty": "paired normal intervals over independent synthetic sequences or natural-text windows; unadjusted; not training-seed variation"},
              "arms": {}}
    specs = [("reference", "both", "named", named, 1.)]
    for channel in ("pred", "corr", "both"):
        for label, heads in {"named": named, **controls}.items():
            for gamma in args.gammas:
                specs.append((f"{channel}/{label}@{gamma:g}", channel, label, heads, gamma))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, channel, label, heads, gamma in specs:
            state.update(channel=channel, gamma=gamma, edges=head_edges(heads), control=label != "named")
            values = evaluate()
            if name == "reference":
                reference = values
            result["arms"][name] = {"values": values,
                                    "summary": {k: paired_summary(v, reference[k]) for k, v in values.items()}}
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(name, "NLL", result["arms"][name]["summary"]["natural_nll"], flush=True)
        result["complete"] = True
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
