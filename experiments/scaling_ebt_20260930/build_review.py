"""Read PR #3 at a fixed revision and enumerate a proposed EBT-style sweep.

CPU only; does not check out, merge, train, submit jobs, or modify PR files.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REV = "c9757183c3695ff2d688922d3a38fdc0dc70f7dd"


def read_pr(path):
    return json.loads(subprocess.check_output(
        ["git", "show", f"{REV}:{path}"], cwd=ROOT, text=True))


def counts(layers, width, heads=8):
    # Current tied-embedding OLMo-2 + complete FourWay encoder/corrections.
    # V=100352; predictor E=256, encoder layers=2, heads=4, trunk=512;
    # predictor positions=4096; correction hidden=128; use_v_norm=False.
    backbone = 100352 * width + layers * (16 * width**2 + 4 * width) + width
    coefficients = (3 * heads + 1) * (layers * (layers + 1) // 2 - 1)
    router = 28450304 + 641 * coefficients + 128 * width * (layers - 1)
    return backbone, router


def main():
    runs, common = {}, {}
    for machine in ("delta", "timan1", "timan108"):
        for run in read_pr(f"experiments/scaling/collected_{machine}.json")["runs"]:
            runs[run["name"]] = run
        common.update(read_pr(f"experiments/scaling/common_eval_{machine}.json"))
    fits = read_pr("experiments/scaling/scaling_fits.json")
    pairs = [
        ("shared12b", "75m", "shared_75m_baseline", "shared_75m_dagformer"),
        ("timan12b", "75m", "timan_75m_dense", "timan_75m_corrected"),
        ("shared12b", "150m", "shared_150m_baseline", "shared_150m_dagformer"),
        ("timan12b", "150m", "timan_150m_dense", "timan_150m_corrected"),
        ("delta21b", "300m", "300m_dense", "300m_fourway_corrected"),
    ]
    rows = []
    for corpus, size, dn, rn in pairs:
        dense, routed = runs[dn], runs[rn]
        de, re = common[dn], common[rn]
        total = routed["params"]["total"] + routed["routing_params"]
        equiv = fits["effective_params"][rn]["N_eff"]
        gap = de["wikitext2"] - re["wikitext2"]
        rows.append({
            "corpus": corpus, "nominal_size": size,
            "dense_run": dn, "dag_run": rn,
            "dense_eval_step": de["step"], "dag_eval_step": re["step"],
            "dense_total_params": dense["params"]["total"],
            "dag_backbone_params": routed["params"]["total"],
            "dag_routing_params": routed["routing_params"],
            "dag_total_params": total,
            "dense_recorded_tokens": dense["total_tokens"],
            "dag_recorded_tokens": routed["total_tokens"],
            "dense_wikitext_token_nll": de["wikitext2"],
            "dag_wikitext_token_nll": re["wikitext2"],
            "nll_improvement": gap,
            "relative_token_ppl_reduction": 1 - math.exp(-gap),
            "fit_dense_equivalent_params": equiv,
            "fit_multiplier_backbone": equiv / routed["params"]["total"],
            "fit_multiplier_total": equiv / total,
        })
    with (OUT / "pr_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    snapshot = {
        "pr": "https://github.com/AI-Interpretability-Illinois/dagformer/pull/3",
        "revision": REV,
        "metric": "PR common wikitext2 cache, mean nats/token",
        "scope": "Committed PR results; no new evaluation performed.",
        "pairs": rows,
        "parameter_fit_coverage": {
            family: {
                "runs": len(value["points"]),
                "distinct_backbone_sizes": len({p["N"] for p in value["points"]}),
                "fit": value["fit"],
            }
            for family, value in fits["L_of_N"].items()
        },
        "equivalent_size_interpretation": (
            "Derived from the PR dense fit, with both model size and token budget "
            "varying across the ladder. Not a directly trained matched-size dense model "
            "or a fixed-data parameter-efficiency experiment."
        ),
    }
    (OUT / "pr_review.json").write_text(json.dumps(snapshot, indent=2) + "\n")

    seq_len = 1024
    base_tokens = 1572864000
    base_batch_sequences = 512
    # One common data corpus for all new runs. The existing small pairs used
    # different corpora, so the central pair is counted as new here.
    specs = {}

    def add(axis, layers=6, width=512, tokens=base_tokens, batch=base_batch_sequences):
        key = (layers, width, tokens, batch)
        if key not in specs:
            backbone, router = counts(layers, width)
            specs[key] = {
                "id": f"L{layers}_d{width}_D{tokens}_B{batch}",
                "panels": [], "layers": layers, "width": width,
                "heads": 8, "ffn_width": 4 * width,
                "tokens": tokens, "batch_sequences": batch,
                "batch_tokens": batch * seq_len,
                "optimizer_updates": tokens // (batch * seq_len),
                "dense_total_params": backbone,
                "dag_total_params": backbone + router,
                "dag_routing_params": router,
            }
        specs[key]["panels"].append(axis)

    for layers in (4, 6, 8, 12, 16):
        add("depth", layers=layers)
    for width in (384, 512, 640, 768):
        add("width", width=width)
    for tokens in (base_tokens // 2, base_tokens, base_tokens * 2, base_tokens * 4):
        add("data", tokens=tokens)
    for batch in (64, 128, 256, 512):
        add("batch", batch=batch)

    plan = {
        "status": "execution authorized; runtime status is recorded separately in execution.json",
        "interpretation": "Six scaling axes from EBT Figures 4 and 5.",
        "families": ["dense", "fourway_corrected"],
        "architecture": {
            "vocab_size": 100352, "tie_word_embeddings": True,
            "max_position_embeddings": 4096,
            "predictor_encoder_dim": 256, "predictor_encoder_layers": 2,
            "predictor_encoder_heads": 4, "predictor_max_seq_len": 4096,
            "fourway_hidden": 512, "predictor_causal": True,
            "correction_hidden": 128, "use_v_norm": False,
        },
        "recipe": {
            "training_corpus": "One fixed Dolma-v1.7 21B corpus manifest for all new runs",
            "seq_len": seq_len, "first_seed": 42,
            "backbone_lr": 5e-4, "predictor_lr": 3e-4,
            "betas": [0.9, 0.95], "weight_decay": 0.1,
            "max_grad_norm": 1.0, "lr_schedule": "linear",
            "warmup_tokens": 104857600,
            "schedule_rule": "Express warmup and decay in consumed tokens; each data budget ends its own schedule.",
            "data_order": "Same sample stream per seed, regrouped into global batches.",
            "primary_eval": "One worker-stratified Dolma cache from excluded document suffixes (512 windows, eight equal worker strata)",
            "secondary_eval": ["fixed wikitext2 cache", "MathInstruct NLL", "GSM8K gold-answer NLL"],
        },
        "existing_parameter_axis": {
            "source_revision": REV,
            "reuse": "PR ladder as descriptive evidence; common evaluation required for additional 600M/1B checkpoints.",
            "views": ["all trainable parameters", "all non-embedding parameters", "backbone parameters"],
            "fit_rule": "Group by training corpus and token budget policy; report distinct sizes, not just run count.",
        },
        "compute_axis": {
            "reuse": "Same evaluated checkpoints as parameter and data panels",
            "accounting": "Include repeated source QKV projections, predictor and corrections; distinguish analytic FLOPs and measured GPU-hours.",
            "claim": "Interpolated loss-vs-compute initially; direct matched-compute endpoints before claiming iso-compute superiority.",
        },
        "new_small_sweep": list(specs.values()),
        "unique_settings_per_family": len(specs),
        "first_seed_training_runs": 2 * len(specs),
        "seed_followup": "Two additional seeds on the shared center and any endpoint contrast used to claim a better scaling rate.",
        "depthbench_followup": (
            "Run existing depth diagnostics on the new depth checkpoints. "
            "A separate fixed-total-parameter depth/width scan answers a different question."
        ),
    }
    (OUT / "six_axis_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Read {len(rows)} matched pairs from PR revision {REV[:7]}.")
    print(f"Proposed small sweep: {len(specs)} settings/family, {2 * len(specs)} first-seed runs.")
    print("No PR merge, evaluation, or training performed.")


if __name__ == "__main__":
    main()
