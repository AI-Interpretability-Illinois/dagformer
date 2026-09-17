"""Evaluate a fixed context-fidelity circuit separately on pred/corr/both.

The circuit is selected from the existing discovery JSON, never from these
evaluation items. Report paired per-item effects and natural-text NLL alongside
head-randomized controls that preserve each edge's layer, stream and source.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from interp_common import flatten_alpha, load_elh, load_eval_ids, unflatten_alpha
from interp_editing import layer_chunks, stream_slices
from interp_liar_cloze import CUES, build_items, prompt_for


def apply_edges(chunk, layer, heads, edges, gamma):
    """Scale deviations from the ORIGINAL head mean, simultaneously.

    Reading means from the input makes the edit independent of edge ordering;
    the older apply_conn_set recomputes means after each preceding edit.
    """
    if gamma == 1.0:
        return chunk
    out = chunk.clone()
    slices, n_src = stream_slices(layer, heads)
    for l, stream, head, source in edges:
        if l != layer:
            continue
        lo, hi = slices[stream]
        original = chunk[..., lo:hi].reshape(*chunk.shape[:2], heads, n_src)
        target = out[..., lo:hi].view(*chunk.shape[:2], heads, n_src)
        mean = original[..., source].mean(dim=2)
        target[..., head, source] = mean + gamma * (original[..., head, source] - mean)
    return out


def norm_matched_edit(chunk, layer, heads, edges, reference_edges, gamma):
    """Match the reference edit's L2 norm separately at each layer and token.

    Both hypothetical edits use the same incoming activations. This removes
    local edit magnitude as an explanation of a control difference; it does
    not match downstream hidden states after earlier interventions.
    """
    if gamma == 1.0:
        return chunk
    change = apply_edges(chunk, layer, heads, edges, gamma) - chunk
    reference = apply_edges(chunk, layer, heads, reference_edges, gamma) - chunk
    norm = change.float().norm(dim=-1, keepdim=True)
    scale = reference.float().norm(dim=-1, keepdim=True) / norm.clamp_min(1e-12)
    return chunk + (change.float() * scale).to(chunk.dtype)


def evaluation_prompt(item, cue, style):
    if style == "original":
        return prompt_for(item, cue)
    cue_text = CUES[cue].format(a=item["a"], b=item["b"])
    if style == "qa":
        return (f"Fact: {item['fact']}{cue_text}\n"
                f"Question: What did {item['a']} see?\nAnswer:")
    if style == "dialogue":
        return (f"{item['fact']}{cue_text}\n"
                f"{item['b']}: What did you see?\n{item['a']}: I saw a")
    raise ValueError(style)


def read_circuit(path, topk):
    scores = json.loads(Path(path).read_text())["conn_scores"]
    edges = []
    for name in sorted(scores, key=scores.get)[:topk]:
        layer, head, connection = name.split("/")
        stream, source = connection.split("<-src")
        edges.append((int(layer[1:]), stream, int(head[1:]), int(source)))
    return edges


def matched_edges(edges, heads, seed):
    rng = random.Random(seed)
    used = set(edges)
    out = []
    for layer, stream, _, source in edges:
        choices = [(layer, stream, h, source) for h in range(heads)
                   if (layer, stream, h, source) not in used]
        edge = rng.choice(choices)
        used.add(edge)
        out.append(edge)
    return out


def heldout_items(n, seed, exclude=()):
    """Exclude all original 240 discovery items and repeated evaluation items."""
    key = lambda item: (item["fact"], item["ask"], item["a"], item["b"])
    used = {key(it) for it in build_items(240, seed=0)}
    used.update(key(it) for it in exclude)
    items = []
    for item in build_items(n * 10, seed=seed):
        if key(item) not in used:
            used.add(key(item))
            items.append(item)
        if len(items) == n:
            return items
    raise ValueError("Could not construct enough distinct held-out items")


def paired_summary(values, reference):
    values, reference = np.asarray(values), np.asarray(reference)
    delta = values - reference
    se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
    return {"mean": float(values.mean()), "delta": float(delta.mean()),
            "paired_se": se,
            "normal_95ci": [float(delta.mean() - 1.96 * se),
                            float(delta.mean() + 1.96 * se)] if se is not None else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--circuit", default="experiments/results/interp/liar_cloze/deception_conns.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--exclude-items", nargs="*", default=[],
                    help="prior raw evaluation JSON files whose content items must be excluded")
    ap.add_argument("--n-items", type=int, default=240)
    ap.add_argument("--n-nll", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--edge-scope", choices=["all", "hyper", "sequential"], default="all")
    ap.add_argument("--gammas", nargs="+", type=float, default=[0., 0.5, 1., 2., 4.])
    ap.add_argument("--channels", nargs="+", choices=["pred", "corr", "both"],
                    default=["pred", "corr", "both"])
    ap.add_argument("--controls", type=int, default=5)
    ap.add_argument("--norm-match-controls", action="store_true")
    ap.add_argument("--prompt-style", choices=["original", "qa", "dialogue"],
                    default="original")
    ap.add_argument("--cues", nargs="+", choices=["neutral", "honest", "deceptive"],
                    default=["neutral", "deceptive"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    model, predictor = elh.load_fourway(args.ckpt, cfg, device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    edges = read_circuit(args.circuit, args.topk)
    if args.edge_scope != "all":
        edges = [edge for edge in edges if
                 (edge[3] < edge[0]) == (args.edge_scope == "hyper")]
    previous_items = []
    for path in args.exclude_items:
        previous_items.extend(json.loads(Path(path).read_text())["items"])
    items = heldout_items(args.n_items, args.seed, previous_items)
    buckets = {}
    for cue in args.cues:
        grouped = defaultdict(list)
        for index, item in enumerate(items):
            ids = tok(evaluation_prompt(item, cue, args.prompt_style),
                      add_special_tokens=False)["input_ids"]
            candidates = [tok(" " + word, add_special_tokens=False)["input_ids"]
                          for word in [item["true"], *item["alts"]]]
            if any(len(c) != 1 for c in candidates):
                raise ValueError("Cloze metric requires single-token candidates")
            grouped[len(ids)].append((index, ids, [c[0] for c in candidates]))
        buckets[cue] = grouped
    ids, labels = load_eval_ids(args.eval_cache)
    if not 0 < args.n_nll <= len(ids):
        raise ValueError(f"n-nll must be in [1, {len(ids)}]")
    ids, labels = ids[-args.n_nll:], labels[-args.n_nll:]
    state = {"channel": "both", "edges": [], "gamma": 1.0, "norm_match": False}
    def edit(chunk, layer):
        if state["norm_match"]:
            return norm_matched_edit(chunk, layer, H, state["edges"], edges,
                                     state["gamma"])
        return apply_edges(chunk, layer, H, state["edges"], state["gamma"])

    hooks = []
    for index, mlp in enumerate(model.correction_mlps):
        def hook(module, inputs, output, layer=index + 1):
            if state["channel"] in ("corr", "both"):
                return edit(output, layer)
            return output
        hooks.append(mlp.register_forward_hook(hook))

    @torch.no_grad()
    def forward(rows):
        rows = rows.to(device)
        routing = predictor(rows)
        if state["channel"] in ("pred", "both") and state["gamma"] != 1.0:
            flat = flatten_alpha(routing)
            parts = [edit(flat[..., start:end], i + 1)
                     for i, (start, end) in enumerate(chunks)]
            routing = unflatten_alpha(torch.cat(parts, dim=-1), L, H)
        return model(rows, routing)

    def measure():
        cloze = {}
        for cue, grouped in buckets.items():
            probability = np.zeros(len(items))
            correct = np.zeros(len(items))
            logp_true = np.zeros(len(items))
            for group in grouped.values():
                for start in range(0, len(group), args.batch_size):
                    batch = group[start:start + args.batch_size]
                    logits = forward(torch.tensor([row[1] for row in batch]))[:, -1].float()
                    for row, (index, _, candidates) in enumerate(batch):
                        scores = logits[row, candidates]
                        probability[index] = scores.softmax(-1)[0].item()
                        correct[index] = (scores.argmax().item() == 0)
                        logp_true[index] = (logits[row, candidates[0]]
                                            - logits[row].logsumexp(-1)).item()
            cloze[cue] = {"p_true": probability.tolist(), "correct": correct.tolist(),
                          "logp_true": logp_true.tolist()}
        nll = []
        for row, target in zip(ids, labels):
            logits = forward(row[None]).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.to(device))
            nll.append(loss.item())
        return {"cloze": cloze, "nll": nll}

    try:
        reference = measure()
        print("reference", {c: np.mean(v["p_true"]) for c, v in reference["cloze"].items()},
              "nll", np.mean(reference["nll"]), flush=True)
        result = {
            "args": vars(args), "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "numpy")},
            "checkpoint_step": torch.load(args.ckpt, map_location="cpu",
                weights_only=False, mmap=True).get("step"),
            "circuit": edges, "items": items, "reference": reference, "arms": {},
            "protocol": {
                "selection": "fixed top connections from discovery JSON; no reselection",
                "holdout": "new item combinations; original 240 seed-0 items excluded",
                "edit": "simultaneous head-mean deviation scaling; pred/corr/both separated",
                "controls": ("random heads; same layer/stream/source counts; " +
                             ("L2 norm matched per layer/token on incoming activations"
                              if args.norm_match_controls else "not norm-matched")),
                "prompt_style": args.prompt_style,
                "intervals": "normal approximation to paired item/sequence mean differences",
                "damage_threshold_nats": 0.05,
            },
        }
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        masks = [("circuit", edges)] + [(f"random{seed}", matched_edges(edges, H, 1000 + seed))
                                        for seed in range(args.controls)]
        for channel in args.channels:
            for mask_name, mask in masks:
                for gamma in args.gammas:
                    state.update(channel=channel, edges=mask, gamma=gamma,
                                 norm_match=args.norm_match_controls and mask_name != "circuit")
                    values = reference if gamma == 1 else measure()
                    summary = {cue: paired_summary(values["cloze"][cue]["p_true"],
                                                    reference["cloze"][cue]["p_true"])
                               for cue in args.cues}
                    summary["nll"] = paired_summary(values["nll"], reference["nll"])
                    summary["damaged"] = summary["nll"]["delta"] > 0.05
                    name = f"{channel}/{mask_name}/gamma{gamma:g}"
                    result["arms"][name] = {"edges": mask, "summary": summary, **values}
                    output.write_text(json.dumps(result, indent=2) + "\n")
                    print(name, json.dumps(summary), flush=True)
    finally:
        for handle in hooks:
            handle.remove()


if __name__ == "__main__":
    main()
