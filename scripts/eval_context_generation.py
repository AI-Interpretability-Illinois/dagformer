"""Test fixed context-fidelity edits with unconstrained greedy generation.

Count the first candidate value mentioned in the generated continuation. A
continuation mentioning no value scores zero on this inclusion endpoint; it
can still name the correct object while omitting an attribute. Retain whole-vocab
first-token accuracy and true-token probability, avoiding candidate-only
normalization as the sole endpoint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch
from transformers import AutoTokenizer

from eval_context_fidelity import (apply_edges, evaluation_prompt, heldout_items,
                                   matched_edges, norm_matched_edit, paired_summary,
                                   read_circuit)
from interp_common import flatten_alpha, load_elh, unflatten_alpha
from interp_editing import layer_chunks


def first_value(text, candidates):
    pattern = r"\b(" + "|".join(re.escape(word) for word in candidates) + r")\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0).lower() if match else None


def write_json(path, value, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=indent) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--circuit", default="experiments/results/interp/liar_cloze/deception_conns.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    ap.add_argument("--n-items", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--exclude-items", nargs="*", default=[])
    ap.add_argument("--styles", nargs="+", default=["original", "qa", "dialogue"])
    ap.add_argument("--cues", nargs="+", default=["neutral", "deceptive"])
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--controls", type=int, default=2)
    ap.add_argument("--pred-gamma", type=float, default=1.5)
    ap.add_argument("--corr-gamma", type=float, default=1.25)
    ap.add_argument("--both-gamma", type=float, default=1.25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    model, predictor = elh.load_fourway(args.ckpt, cfg, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    edges = read_circuit(args.circuit, 10)
    excluded = []
    for path in args.exclude_items:
        excluded.extend(json.loads(Path(path).read_text())["items"])
    items = heldout_items(args.n_items, args.seed, excluded)
    candidates = []
    for item in items:
        encoded = [tokenizer(" " + word, add_special_tokens=False)["input_ids"]
                   for word in [item["true"], *item["alts"]]]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("First-token endpoint requires single-token values")
        candidates.append([ids[0] for ids in encoded])
    buckets = {}
    for style in args.styles:
        for cue in args.cues:
            groups = defaultdict(list)
            for index, item in enumerate(items):
                tokens = tokenizer(evaluation_prompt(item, cue, style), add_special_tokens=False)["input_ids"]
                groups[len(tokens)].append((index, tokens))
            buckets[f"{style}/{cue}"] = groups
    state = {"channel": "both", "edges": edges, "gamma": 1., "control": False}

    def edit(chunk, layer):
        if state["control"]:
            return norm_matched_edit(chunk, layer, H, state["edges"], edges, state["gamma"])
        return apply_edges(chunk, layer, H, state["edges"], state["gamma"])

    handles = []
    for index, mlp in enumerate(model.correction_mlps):
        def hook(module, inputs, output, layer=index + 1):
            return edit(output, layer) if state["channel"] in ("corr", "both") else output
        handles.append(mlp.register_forward_hook(hook))

    @torch.inference_mode()
    def next_logits(ids):
        routing = predictor(ids)
        if state["channel"] in ("pred", "both") and state["gamma"] != 1.:
            flat = flatten_alpha(routing)
            parts = [edit(flat[..., start:end], i + 1) for i, (start, end) in enumerate(chunks)]
            routing = unflatten_alpha(torch.cat(parts, -1), L, H)
        return model(ids, routing)[:, -1].float()

    def evaluate():
        results = {}
        for name, groups in buckets.items():
            texts = [None] * len(items)
            metrics = {key: np.zeros(len(items)) for key in (
                "first_value_true", "any_value", "first_token_true", "first_token_candidate",
                "true_token_probability", "candidate_p_true")}
            for group in groups.values():
                for start in range(0, len(group), args.batch_size):
                    batch = group[start:start + args.batch_size]
                    ids = torch.tensor([row[1] for row in batch], device=device)
                    finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
                    generated = []
                    for step in range(args.max_new_tokens):
                        logits = next_logits(ids)
                        chosen = logits.argmax(-1)
                        if step == 0:
                            for row, (index, _) in enumerate(batch):
                                values = candidates[index]
                                metrics["first_token_true"][index] = int(chosen[row]) == values[0]
                                metrics["first_token_candidate"][index] = int(chosen[row]) in values
                                metrics["true_token_probability"][index] = torch.exp(
                                    logits[row, values[0]] - logits[row].logsumexp(-1)).item()
                                metrics["candidate_p_true"][index] = logits[row, values].softmax(-1)[0].item()
                        chosen = chosen.masked_fill(finished, tokenizer.eos_token_id)
                        generated.append(chosen)
                        finished |= chosen == tokenizer.eos_token_id
                        ids = torch.cat([ids, chosen[:, None]], dim=1)
                        if finished.all():
                            break
                    sequences = torch.stack(generated, 1).cpu().tolist()
                    for (index, _), sequence in zip(batch, sequences):
                        if tokenizer.eos_token_id in sequence:
                            sequence = sequence[:sequence.index(tokenizer.eos_token_id)]
                        text = tokenizer.decode(sequence, skip_special_tokens=True)
                        texts[index] = text
                        value = first_value(text, [items[index]["true"], *items[index]["alts"]])
                        metrics["any_value"][index] = value is not None
                        metrics["first_value_true"][index] = value == items[index]["true"]
            results[name] = {"generations": texts, **{k: v.tolist() for k, v in metrics.items()}}
        return results

    raw = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "checkpoint_step": torch.load(args.ckpt, map_location="cpu", weights_only=False,
                                         mmap=True).get("step"),
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
           "items": items, "circuit": edges, "arms": {}}
    summary = {k: v for k, v in raw.items() if k not in ("items", "arms")}
    summary.update(n_items=len(items), raw_artifact=str(args.out), arms={},
                   protocol="greedy unconstrained continuations; first candidate value mentioned; omission scores zero on the explicit-inclusion endpoint; random controls norm matched per layer/token")
    arms = [("reference", "both", edges, 1., False)]
    for channel in ("pred", "corr", "both"):
        gamma = getattr(args, channel + "_gamma")
        arms.append((channel + "/circuit", channel, edges, gamma, False))
        for seed in range(args.controls):
            arms.append((f"{channel}/random{seed}", channel,
                         matched_edges(edges, H, 1000 + seed), gamma, True))
    try:
        for name, channel, mask, gamma, control in arms:
            state.update(channel=channel, edges=mask, gamma=gamma, control=control)
            values = evaluate()
            if name == "reference":
                reference = values
            rec = {}
            for condition, scores in values.items():
                rec[condition] = {metric: paired_summary(vector, reference[condition][metric])
                                  for metric, vector in scores.items() if metric != "generations"}
            raw["arms"][name] = {"edges": mask, "gamma": gamma, "values": values}
            summary["arms"][name] = {"edges": mask, "gamma": gamma, "conditions": rec}
            write_json(args.out, raw)
            write_json(args.summary_out, summary, indent=2)
            print(name, {k: {m: round(v[m]["mean"], 4) for m in ("first_value_true", "any_value")}
                         for k, v in rec.items()}, flush=True)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
