"""Diagnose a frozen soft-prefix/routing search without refitting any parameters.

Separates predicting a color from retrieving the correct context color. Paired
color-swap examples also supply counterfactual donors at identical positions.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from semantic_routing_pilot import Pilot


METRICS = (
    "true_logp", "color_logmass", "p_true_given_context_colors",
    "true_vs_other_margin", "two_color_accuracy", "other_context_logp",
    "p_other_given_context_colors",
)


def summary(rows, reference=None):
    result = {}
    generator = np.random.default_rng(20260928)
    indices = generator.integers(len(rows) // 2, size=(4000, len(rows) // 2))
    for key in METRICS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        pair_values = values.reshape(-1, 2).mean(1)
        record = {"mean": float(values.mean()),
                  "paired_bootstrap_95ci": np.quantile(pair_values[indices].mean(1), [.025, .975]).tolist()}
        if reference is not None:
            delta = values - np.asarray([row[key] for row in reference])
            pair_delta = delta.reshape(-1, 2).mean(1)
            record["delta"] = float(delta.mean())
            record["delta_95ci"] = np.quantile(pair_delta[indices].mean(1), [.025, .975]).tolist()
        result[key] = record
    return result


def validate_pairs(items):
    assert len(items) % 2 == 0
    for start in range(0, len(items), 2):
        left, right = items[start:start + 2]
        assert left["pair"] == right["pair"]
        assert left["target"] != right["target"]
        assert len(left["ids"]) == len(right["ids"]), "Counterfactual donor positions differ"
        changed = [(a, b) for a, b in zip(left["ids"], right["ids"]) if a != b]
        assert len(changed) == 2, "Expected only two exchanged color tokens"
        assert sorted(changed) == sorted([(left["target"], right["target"]),
                                         (right["target"], left["target"])])
        left["other_context_target"] = right["target"]
        right["other_context_target"] = left["target"]


@torch.no_grad()
def measure(pilot, items, *, soft=False, capture=False, mask=None):
    rows = []
    for start in range(0, len(items), pilot.args.batch_size):
        batch = items[start:start + pilot.args.batch_size]
        logits = pilot.forward(batch, soft=soft, capture=capture, mask=mask, scope="content")
        logps = logits.log_softmax(-1)
        for index, item in enumerate(batch):
            true, other = item["target"], item["other_context_target"]
            margin = logits[index, true] - logits[index, other]
            prob = margin.sigmoid()
            rows.append({"pair": item["pair"], "target": true, "other_context_target": other,
                         "true_logp": float(logps[index, true]),
                         "color_logmass": float(logps[index, item["candidates"]].logsumexp(0)),
                         "p_true_given_context_colors": float(prob),
                         "true_vs_other_margin": float(margin),
                         "two_color_accuracy": float(margin > 0) + .5 * float(margin == 0),
                         "other_context_logp": float(logps[index, other]),
                         "p_other_given_context_colors": float(1 - prob)})
    return rows


def edge_key(label):
    return tuple(label[name] for name in ("channel", "stream", "layer", "head", "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = json.loads((args.run / "results.json").read_text())
    out = args.out or args.run / "diagnostics.json"
    pilot_args = copy.deepcopy(source["args"])
    for name in ("model", "corpora", "out"):
        pilot_args[name] = Path(pilot_args[name])
    torch.manual_seed(pilot_args["seed"])
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    pilot = Pilot(argparse.Namespace(**pilot_args))
    result = {"protocol": "Frozen first-round soft prefix and saved edge masks; no fitting or selection. "
              "Seven-color log mass separates category bias from retrieval between the two context colors. "
              "Counterfactual donors exchange only the two context-color tokens, with identical lengths/positions. "
              "Edits affect content positions; unpatched correction outputs are recomputed dynamically. "
              "Intervals bootstrap 48 counterfactual item pairs per split, 4000 draws. Ties score one half.",
              "source": str(args.run), "source_code_revision": source["code_revision"],
              "started": time.time(), "splits": {}}
    try:
        pilot.soft.load_state_dict(torch.load(args.run / "soft_prefix.pt", map_location=pilot.device,
                                            weights_only=True))
        label_to_index = {edge_key(label): index for index, label in enumerate(pilot.labels)}
        masks = {"both_content": torch.ones(2 * pilot.E, device=pilot.device)}
        for name in ("learned_256", "random0_256", "random1_256", "random2_256"):
            indices = [label_to_index[edge_key(label)] for label in source["sparse_arms"][name]["edges"]]
            assert len(indices) == len(set(indices)) == 256
            masks[name] = torch.zeros(2 * pilot.E, device=pilot.device)
            masks[name][indices] = 1
        for split in ("test", "transfer"):
            items = copy.deepcopy(source["items"][split])
            validate_pairs(items)
            baseline = measure(pilot, items)
            soft = measure(pilot, items, soft=True, capture=True)
            # Confirm the restored model/prefix reproduce the saved first-round run.
            base_error = max(abs(row["true_logp"] - old["logp"])
                             for row, old in zip(baseline, source["baseline"][split]["rows"]))
            soft_error = max(abs(row["true_logp"] - old["logp"])
                             for row, old in zip(soft, source["stages"]["soft_prompt"][split]["rows"]))
            assert max(base_error, soft_error) < .03, (split, base_error, soft_error)
            records = {"baseline": {"summary": summary(baseline), "rows": baseline},
                       "soft": {"summary": summary(soft, baseline), "rows": soft}}
            for name, mask in masks.items():
                rows = measure(pilot, items, mask=mask)
                records[name] = {"summary": summary(rows, baseline), "rows": rows}
            own_donors = [(item["donor_pred"], item["donor_corr"]) for item in items]
            try:
                for index, item in enumerate(items):
                    item["donor_pred"], item["donor_corr"] = own_donors[index ^ 1]
                for name in ("both_content", "learned_256"):
                    rows = measure(pilot, items, mask=masks[name])
                    records[name + "_swapped_donor"] = {
                        "summary": summary(rows, baseline), "rows": rows,
                        "paired_vs_own_donor": summary(rows, records[name]["rows"]),
                        "direction": "Positive p_other_given_context_colors change or negative true_vs_other_margin "
                                     "change is movement toward the counterfactual donor's target."}
            finally:
                for item, (pred, corr) in zip(items, own_donors):
                    item["donor_pred"], item["donor_corr"] = pred, corr
            result["splits"][split] = {"n_items": len(items), "n_pairs": len(items) // 2,
                                        "reproduction_max_abs_logp_error": {"baseline": base_error, "soft": soft_error},
                                        "arms": records}
            out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(split, json.dumps({name: {key: data["summary"][key]["mean"] for key in METRICS}
                                     for name, data in records.items()}), flush=True)
        result["finished"] = time.time()
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        for handle in pilot.handles:
            handle.remove()


if __name__ == "__main__":
    main()
