"""Summarize the paired frozen-model attention-head support experiment.

Only reads completed JSON results. Bootstrap units are matched sequences/windows;
random-mask ranges are references, not confidence intervals over optimization seeds.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SYNTHETIC = ["test_p128_l512", "test_p128_l1024", "test_p256_l1024"]
NATURAL = "natural_wikitext"
DATASETS = SYNTHETIC + [NATURAL]
LABELS = {"test_p128_l512": "Copy p128 / 512", "test_p128_l1024": "Copy p128 / 1024",
          "test_p256_l1024": "Copy p256 / 1024", NATURAL: "WikiText sampled positions"}
FAMILIES = ["baseline", "dagformer"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Bootstrap:
    def __init__(self, draws, seed):
        self.draws, self.seed, self.indices = draws, seed, {}

    def mean_samples(self, values):
        a = np.asarray(values, dtype=float)
        assert a.ndim == 1 and len(a) > 0 and np.isfinite(a).all()
        if len(a) not in self.indices:
            rng = np.random.default_rng(self.seed + len(a))
            self.indices[len(a)] = rng.integers(len(a), size=(self.draws, len(a)))
        return np.r_[a.mean(), a[self.indices[len(a)]].mean(1)]


def summarize(samples):
    s = np.asarray(samples)
    if not np.isfinite(s).all():
        return {"mean": float(s[0]) if np.isfinite(s[0]) else None, "ci95": None,
                "reason": "Reference accuracy is zero in at least one resample; normalized loss undefined"}
    return {"mean": float(s[0]), "ci95": np.quantile(s[1:], [.025, .975]).tolist()}


def normalized_loss(reference, masked):
    return np.divide(reference - masked, reference, out=np.full_like(reference, np.nan), where=reference > 0)


def validate_pair(results):
    a, b = [results[f] for f in FAMILIES]
    assert a["dataset_metadata"] == b["dataset_metadata"], "Dataset shapes or scored positions differ"
    for key in ["seed", "n_train", "n_valid", "n_test", "n_natural", "score_positions", "budgets", "controls"]:
        assert a["args"][key] == b["args"][key], f"Paired generation/selection settings differ: {key}"
    for family, j in results.items():
        assert j["complete"] is True
        assert j["args"]["kind"] == family
        full = np.asarray(j["arms"]["full"]["mask"])
        assert full.shape == (12, 16) and full.sum() == 192, "This report expects the 300M 192-head pair"
        assert j["checks"]["all_one_max_logit_error"] == 0
        for k in j["args"]["budgets"]:
            mask = np.asarray(j["arms"][f"learned_{k}"]["mask"])
            assert mask.sum() == k and np.isin(mask, [0, 1]).all()
            assert np.array_equal(j["arms"][f"remove_learned_{k}"]["mask"], 1-mask)
            for index in range(j["args"]["controls"]):
                rand = np.asarray(j["arms"][f"random_{k}_{index}"]["mask"])
                assert np.array_equal(rand.sum(1), mask.sum(1)), "Random control lacks layer-count matching"
                assert np.array_equal(j["arms"][f"remove_random_{k}_{index}"]["mask"], 1-rand)
        if family == "dagformer":
            for k in j["args"]["budgets"]:
                assert np.array_equal(j["arms"][f"fixed_routes_learned_{k}"]["mask"], j["arms"][f"learned_{k}"]["mask"])
            assert "fixed_routes_full" in j["arms"] and "fixed_routes_zero" in j["arms"]
        for arm, data in j["arms"].items():
            assert data["retained_heads"] == int(np.asarray(data["mask"]).sum())
            for dataset in DATASETS:
                vals = data["datasets"][dataset]["values"]
                assert all(len(v) == j["dataset_metadata"][dataset]["n"] for v in vals.values())


def analyze(results, args):
    boot = Bootstrap(args.bootstrap_draws, args.seed)
    samples, per_model = {}, {}
    for family, j in results.items():
        samples[family], per_model[family] = {}, {}
        for arm, data in j["arms"].items():
            reference = "fixed_routes_full" if arm.startswith("fixed_routes_") else "full"
            samples[family][arm] = {}
            out = {"retained_heads": data["retained_heads"], "reference_arm": reference, "datasets": {}}
            for dataset in DATASETS:
                raw = data["datasets"][dataset]["values"]
                ref = j["arms"][reference]["datasets"][dataset]["values"]
                m = {k: boot.mean_samples(v) for k, v in raw.items()}
                r = {k: boot.mean_samples(v) for k, v in ref.items()}
                m.update({
                    "accuracy_loss": r["accuracy"] - m["accuracy"],
                    "normalized_accuracy_loss": normalized_loss(r["accuracy"], m["accuracy"]),
                    "target_nll_damage": m["target_nll"] - r["target_nll"],
                    "teacher_kl_increment": m["teacher_kl"] - r["teacher_kl"],
                })
                samples[family][arm][dataset] = m
                out["datasets"][dataset] = {name: summarize(v) for name, v in m.items()}
            per_model[family][arm] = out

    contrasts = {}
    for arm in results["baseline"]["arms"]:
        assert arm in results["dagformer"]["arms"]
        contrasts[arm] = {}
        for ds in DATASETS:
            b, d = [samples[f][arm][ds] for f in FAMILIES]
            contrasts[arm][ds] = {
                key: summarize(d[key] - b[key]) for key in ["normalized_accuracy_loss", "accuracy_loss",
                    "target_nll_damage", "teacher_kl", "teacher_kl_increment"]
            }

    random_references = {}
    for family, j in results.items():
        random_references[family] = {}
        for k in j["args"]["budgets"]:
            for prefix in ["", "remove_"]:
                learned_name = f"{prefix}learned_{k}"
                random_names = [f"{prefix}random_{k}_{i}" for i in range(j["args"]["controls"])]
                out = {}
                for ds in DATASETS:
                    out[ds] = {}
                    for metric in ["accuracy", "normalized_accuracy_loss", "target_nll_damage", "teacher_kl"]:
                        vals = [samples[family][name][ds][metric][0] for name in random_names]
                        out[ds][metric] = {
                            "random_means": [float(v) for v in vals],
                            "random_mean_range": [float(min(vals)), float(max(vals))],
                            "learned_minus_each_random": {
                                name: summarize(samples[family][learned_name][ds][metric] - samples[family][name][ds][metric])
                                for name in random_names},
                        }
                random_references[family][learned_name] = out

    fixed = {"full_change_from_dynamic": {}, "mask_damage_change_from_dynamic": {}}
    j = results["dagformer"]
    for ds in DATASETS:
        dynamic = samples["dagformer"]["full"][ds]
        fixed_full = samples["dagformer"]["fixed_routes_full"][ds]
        fixed["full_change_from_dynamic"][ds] = {
            k: summarize(fixed_full[k] - dynamic[k]) for k in ["accuracy", "target_nll", "teacher_kl"]}
    for k in j["args"]["budgets"]:
        fixed["mask_damage_change_from_dynamic"][str(k)] = {}
        for ds in DATASETS:
            dynamic = samples["dagformer"][f"learned_{k}"][ds]
            fixed_mask = samples["dagformer"][f"fixed_routes_learned_{k}"][ds]
            fixed["mask_damage_change_from_dynamic"][str(k)][ds] = {
                metric: summarize(fixed_mask[metric] - dynamic[metric]) for metric in [
                    "normalized_accuracy_loss", "target_nll_damage", "teacher_kl_increment"]}

    return {
        "status": "complete", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {f: str(args.input_dir / f"{f}.json") for f in FAMILIES},
        "protocol": {
            "scope": "Support over 192 backbone attention-head outputs; all MLPs and routing computations remain active; not a complete circuit or physical pruning",
            "bootstrap": f"{args.bootstrap_draws} paired bootstrap draws; same sequence/window indices resampled for both models and all arms; seed={args.seed}; intervals unadjusted for multiple comparisons",
            "paired_alignment": "Verified matching deterministic data-generation seed, split sizes and scored positions in the two runs",
            "normalized_accuracy_loss": "(mean reference accuracy - mean masked accuracy) / mean reference accuracy; numerator and denominator re-estimated jointly in each bootstrap",
            "reference": "Dynamic arms: each model's own full arm. DAG fixed_routes arms: fixed_routes_full. Loss can be negative if masking improves accuracy.",
            "cross_model_sign": "DAG minus dense; negative normalized loss / NLL-damage contrast favors DAG retention at the given head count",
            "teacher_kl": "Each model has its own unmasked dynamic teacher. Between-model KL contrasts compare drift from different distributions, not KL to a common target.",
            "fixed_route_kl": "Teacher cache remains dynamic. The fixed-mask KL increment is KL(dynamic teacher || fixed masked model) minus KL(dynamic teacher || fixed full model), not KL(fixed full || fixed masked).",
            "random_controls": "Three random masks preserve each learned mask's per-layer counts. Ranges span these masks only; not intervals across mask-optimization seeds. Individual learned-minus-random CIs resample evaluation sequences.",
            "natural": "24 WikiText windows x 512 input tokens; 64 sampled output positions per window. Damage is sampled token NLL, not full-corpus PPL.",
        },
        "model_metadata": {f: {"args": j["args"], "git_commit_at_start": j["git_commit_at_start"],
                               "elapsed_seconds": j["elapsed_seconds"], "dataset_metadata": j["dataset_metadata"],
                               "optimization": j["optimization"]} for f, j in results.items()},
        "per_model": per_model, "dag_minus_dense": contrasts,
        "random_mask_references": random_references, "fixed_routes": fixed,
    }


def fmt(summary, percent=False, ci=False):
    factor = 100 if percent else 1
    if summary["mean"] is None:
        return "undefined"
    s = f'{factor * summary["mean"]:.2f}' if percent else f'{summary["mean"]:.4f}'
    if ci and summary["ci95"] is not None:
        lo, hi = np.array(summary["ci95"]) * factor
        s += f" [{lo:.2f}, {hi:.2f}]" if percent else f" [{lo:.4f}, {hi:.4f}]"
    return s


def report(out, result):
    budgets = result["model_metadata"]["baseline"]["args"]["budgets"]
    pm, contrasts = result["per_model"], result["dag_minus_dense"]
    lines = ["# Frozen attention-head supports", "",
        "Both model runs are complete. Masks retain 64, 96, 128 or 160 of 192 backbone attention-head outputs; "
        "each budget has its own optimization and validation selection. Model weights are frozen. "
        "All MLPs, predictor and correction computations remain active, so these counts describe attention-head support, not complete circuit size.", "",
        "![Learned head support](head_support_learned.png)", "",
        "The [control comparison figure](head_support.png) also includes the random masks. "
        "Error bars bootstrap matched evaluation sequences/windows. Control bands span three per-layer-count-matched random masks; "
        "they are not optimization-seed confidence intervals. The natural-text endpoint scores 64 positions in each of 24 WikiText windows of length 512.", "",
        "## Full-model references", "", "| Dataset | Dense accuracy | DAG accuracy | Dense NLL | DAG NLL |",
        "|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        b, d = [pm[f]["full"]["datasets"][ds] for f in FAMILIES]
        lines.append(f'| {LABELS[ds]} | {fmt(b["accuracy"], True)}% | {fmt(d["accuracy"], True)}% | {fmt(b["target_nll"])} | {fmt(d["target_nll"])} |')
    lines += ["", "## Retention on new synthetic sequences", "",
        "Normalized accuracy loss is `(full accuracy − masked accuracy) / full accuracy`, reported as percent of each model's full accuracy. "
        "Bootstrap draws recompute both numerator and denominator. Negative DAG-minus-dense contrasts favor DAG retention; "
        "these are fixed-checkpoint, single-mask-optimization-seed comparisons. Intervals are unadjusted for multiple comparisons.", "",
        "| Dataset | Heads kept | Dense loss, % | DAG loss, % | DAG − dense, 95% CI |",
        "|---|---:|---:|---:|---:|"]
    for ds in SYNTHETIC:
        for k in budgets:
            name = f"learned_{k}"
            b, d = [pm[f][name]["datasets"][ds]["normalized_accuracy_loss"] for f in FAMILIES]
            lines.append(f'| {LABELS[ds]} | {k} | {fmt(b, True)} | {fmt(d, True)} | {fmt(contrasts[name][ds]["normalized_accuracy_loss"], True, True)} |')
    lines += ["", "## Natural-text capability cost", "", "NLL increments use each model's own full-model reference and matched windows.", "",
        "| Heads kept | Dense ΔNLL | DAG ΔNLL | DAG − dense damage, 95% CI |", "|---|---:|---:|---:|"]
    for k in budgets:
        name = f"learned_{k}"
        b, d = [pm[f][name]["datasets"][NATURAL]["target_nll_damage"] for f in FAMILIES]
        lines.append(f'| {k} | {fmt(b)} | {fmt(d)} | {fmt(contrasts[name][NATURAL]["target_nll_damage"], ci=True)} |')
    lines += ["", "## Teacher distribution retention", "",
        "KL is measured against each model's own unmasked dynamic teacher. Its cross-model difference compares "
        "drift from different distributions; it is not a comparison against one shared teacher.", "",
        "| Dataset | Heads kept | Dense KL | DAG KL | DAG − dense KL, 95% CI |", "|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        for k in budgets:
            name = f"learned_{k}"
            b, d = [pm[f][name]["datasets"][ds]["teacher_kl"] for f in FAMILIES]
            lines.append(f'| {LABELS[ds]} | {k} | {fmt(b)} | {fmt(d)} | {fmt(contrasts[name][ds]["teacher_kl"], ci=True)} |')
    for prefix, heading, count_label in [("", "Learned versus random retained supports", "Heads kept"),
                                           ("remove_", "Removing the learned support", "Heads removed")]:
        lines += ["", f"## {heading}", "",
            "Synthetic entries are normalized accuracy loss (%); WikiText entries are ΔNLL. "
            "Random ranges are the minimum and maximum means of the three layer-matched masks. "
            "All individual learned-minus-random paired contrasts are preserved in summary.json.", "",
            f"| Dataset | {count_label} | Dense learned / random range | DAG learned / random range |", "|---|---:|---:|---:|"]
        for ds in DATASETS:
            metric = "target_nll_damage" if ds == NATURAL else "normalized_accuracy_loss"
            for k in budgets:
                name = f"{prefix}learned_{k}"
                cols = []
                for family in FAMILIES:
                    s = fmt(pm[family][name]["datasets"][ds][metric], percent=ds != NATURAL)
                    lo, hi = result["random_mask_references"][family][name][ds][metric]["random_mean_range"]
                    if ds != NATURAL:
                        s += f" / [{100*lo:.2f}, {100*hi:.2f}]"
                    else:
                        s += f" / [{lo:.4f}, {hi:.4f}]"
                    cols.append(s)
                lines.append(f'| {LABELS[ds]} | {k} | {cols[0]} | {cols[1]} |')

    lines += ["", "## Fixed predictor position-table diagnostic", "",
        "Only the external predictor is replaced by position means from separate discovery-copy sequences; local correction remains active. "
        "The learned masks are unchanged. All fixed-mask damage below is relative to **fixed_routes_full**, not dynamic full.", "",
        "| Dataset | Fixed full − dynamic full accuracy, pp | Fixed full − dynamic full NLL | KL to dynamic teacher |",
        "|---|---:|---:|---:|"]
    for ds in DATASETS:
        s = result["fixed_routes"]["full_change_from_dynamic"][ds]
        kl = pm["dagformer"]["fixed_routes_full"]["datasets"][ds]["teacher_kl"]
        lines.append(f'| {LABELS[ds]} | {fmt(s["accuracy"], True, True)} | {fmt(s["target_nll"], ci=True)} | {fmt(kl)} |')
    lines += ["", "Synthetic damage is normalized accuracy loss (%); WikiText damage is ΔNLL. "
        "The last column is a paired contrast of damage, using the correct full reference for each routing condition.", "",
        "| Dataset | Heads kept | Dynamic damage | Fixed-route damage | Fixed − dynamic damage, 95% CI |",
        "|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        metric = "target_nll_damage" if ds == NATURAL else "normalized_accuracy_loss"
        for k in budgets:
            a = pm["dagformer"][f"learned_{k}"]["datasets"][ds][metric]
            b = pm["dagformer"][f"fixed_routes_learned_{k}"]["datasets"][ds][metric]
            c = result["fixed_routes"]["mask_damage_change_from_dynamic"][str(k)][ds][metric]
            lines.append(f'| {LABELS[ds]} | {k} | {fmt(a, ds != NATURAL)} | {fmt(b, ds != NATURAL)} | {fmt(c, ds != NATURAL, True)} |')
    lines += ["", "The fixed-route KL cache still uses the dynamic full teacher. JSON therefore reports fixed-mask KL "
        "increments over fixed full as `KL(dynamic teacher || fixed masked) − KL(dynamic teacher || fixed full)`; "
        "this is not `KL(fixed full || fixed masked)`.", "", "## Reproduction", "",
        "`python scripts/summarize_circuit_head_support.py --input-dir experiments/results/circuit_interpretability_20260928/head_support`", "",
        "Sources: [dense](baseline.json), [DAG](dagformer.json). [All paired statistics](summary.json); [learned-support vector figure](head_support_learned.pdf); [control vector figure](head_support.pdf). "
        "The [96-head mask-seed follow-up](mask_seed_replication.md) reports the original seed and two additional optimization seeds on the same data.", ""]
    (out / "summary.md").write_text("\n".join(lines))


def plot(out, result, controls=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    budgets = result["model_metadata"]["baseline"]["args"]["budgets"]
    pm = result["per_model"]
    colors, labels = {"baseline": "#4376A8", "dagformer": "#D57931"}, {"baseline": "Dense", "dagformer": "FourWay"}
    for ax, ds in zip(axes.flat, DATASETS):
        metric = "target_nll_damage" if ds == NATURAL else "normalized_accuracy_loss"
        for family in FAMILIES:
            records = [pm[family][f"learned_{k}"]["datasets"][ds][metric] for k in budgets]
            means = np.array([s["mean"] for s in records] + [0.])
            cis = np.array([s["ci95"] for s in records] + [[0., 0.]])
            x = budgets + [192]
            ax.errorbar(x, means, yerr=[means-cis[:, 0], cis[:, 1]-means], marker="o", capsize=3,
                        lw=1.6, ms=4.5, color=colors[family], label=f'{labels[family]} learned')
            if controls:
                ranges = np.array([result["random_mask_references"][family][f"learned_{k}"][ds][metric]["random_mean_range"]
                                   for k in budgets] + [[0., 0.]])
                random_means = [np.mean(result["random_mask_references"][family][f"learned_{k}"][ds][metric]["random_means"])
                                for k in budgets] + [0.]
                ax.fill_between(x, ranges[:, 0], ranges[:, 1], color=colors[family], alpha=.13)
                ax.plot(x, random_means, linestyle="--", color=colors[family], alpha=.6, lw=.9,
                        label=f'{labels[family]} random span')
        ax.axhline(0, color="gray", lw=.7)
        ax.set_xticks(budgets + [192])
        ax.set_xlabel("Backbone attention heads retained (of 192)")
        ax.set_title(LABELS[ds])
        ax.set_ylabel("Sampled target ΔNLL (nats/token)" if ds == NATURAL else "Loss relative to full-model accuracy")
        if ds != NATURAL:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.16)
    axes[0, 0].legend(frameon=False, fontsize=8)
    title = "Learned attention-head support and matched random masks" if controls else "Retention with learned attention-head supports"
    fig.suptitle(title + "\nAll MLPs and routing computations remain active", fontsize=13)
    stem = "head_support" if controls else "head_support_learned"
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)


def summarize_seed_replication(args, primary):
    """Summarize mask-initialization repeats with data and model weights fixed."""
    out = args.input_dir
    primary_seed = primary["baseline"]["args"].get("mask_seed") or primary["baseline"]["args"]["seed"]
    seeds = [primary_seed] + args.replication_mask_seeds
    runs = {primary_seed: primary}
    source_files = {str(primary_seed): {f: str(out / f"{f}.json") for f in FAMILIES}}
    statuses = {}
    for seed in args.replication_mask_seeds:
        runs[seed] = {}
        source_files[str(seed)] = {}
        for family in FAMILIES:
            path = out / f"{family}_maskseed{seed}.json"
            source_files[str(seed)][family] = str(path)
            key = f"{family}:{seed}"
            if not path.exists():
                statuses[key] = "missing"
                continue
            try:
                j = json.loads(path.read_text())
            except json.JSONDecodeError:
                statuses[key] = "incomplete JSON write"
                continue
            statuses[key] = "complete" if j.get("complete") else "incomplete"
            if j.get("complete"):
                runs[seed][family] = j
    if any(len(runs[seed]) != 2 for seed in seeds):
        dump(out / "mask_seed_replication.json", {"status": "incomplete", "models": statuses, "sources": source_files})
        (out / "mask_seed_replication.md").write_text("# Mask-seed replication: incomplete\n\n" +
            "\n".join(f"- {k}: {v}" for k, v in statuses.items()) + "\n\nFinal three-seed comparison awaits all completed runs.\n")
        return

    boot = Bootstrap(args.bootstrap_draws, args.seed)
    budget = args.replication_budget
    per_seed, contrasts, overlap = {}, {}, {}
    for seed, pair in runs.items():
        validate_pair(pair)
        per_seed[str(seed)], contrasts[str(seed)] = {}, {}
        seed_samples = {}
        for family, j in pair.items():
            ref_run = primary[family]
            for key in ["seed", "n_train", "n_valid", "n_test", "n_natural", "score_positions", "steps", "lr", "validate_every", "batch_size"]:
                assert j["args"][key] == ref_run["args"][key], f"Mask repeat changed {key}"
            assert j["dataset_metadata"] == ref_run["dataset_metadata"]
            actual_seed = j["args"].get("mask_seed") or j["args"]["seed"]
            assert actual_seed == seed
            for ds in DATASETS:
                for key, values in j["arms"]["full"]["datasets"][ds]["values"].items():
                    assert np.allclose(values, ref_run["arms"]["full"]["datasets"][ds]["values"][key], atol=1e-7, rtol=0), "Full-model endpoints changed across mask seeds"
            names = [f"learned_{budget}", f"remove_learned_{budget}"]
            if family == "dagformer":
                names.append(f"fixed_routes_learned_{budget}")
            per_seed[str(seed)][family] = {"arms": {}, "selected_validation": j["optimization"][str(budget)],
                                         "mask": j["arms"][f"learned_{budget}"]["mask"]}
            seed_samples[family] = {}
            for arm in names:
                reference = "fixed_routes_full" if arm.startswith("fixed_routes") else "full"
                seed_samples[family][arm] = {}
                per_seed[str(seed)][family]["arms"][arm] = {}
                for ds in DATASETS:
                    m = {k: boot.mean_samples(v) for k, v in j["arms"][arm]["datasets"][ds]["values"].items()}
                    r = {k: boot.mean_samples(v) for k, v in j["arms"][reference]["datasets"][ds]["values"].items()}
                    m.update({"normalized_accuracy_loss": normalized_loss(r["accuracy"], m["accuracy"]),
                              "target_nll_damage": m["target_nll"] - r["target_nll"],
                              "teacher_kl_increment": m["teacher_kl"] - r["teacher_kl"]})
                    seed_samples[family][arm][ds] = m
                    per_seed[str(seed)][family]["arms"][arm][ds] = {key: summarize(vals) for key, vals in m.items()}
        for arm in [f"learned_{budget}", f"remove_learned_{budget}"]:
            contrasts[str(seed)][arm] = {}
            for ds in DATASETS:
                b, d = [seed_samples[f][arm][ds] for f in FAMILIES]
                contrasts[str(seed)][arm][ds] = {key: summarize(d[key] - b[key]) for key in
                    ["normalized_accuracy_loss", "target_nll_damage", "teacher_kl"]}

    ranges = {}
    for family in FAMILIES:
        ranges[family], overlap[family] = {}, []
        names = list(per_seed[str(primary_seed)][family]["arms"])
        for arm in names:
            ranges[family][arm] = {}
            for ds in DATASETS:
                ranges[family][arm][ds] = {}
                for metric in ["accuracy", "normalized_accuracy_loss", "target_nll_damage", "teacher_kl", "teacher_kl_increment"]:
                    vals = [per_seed[str(s)][family]["arms"][arm][ds][metric]["mean"] for s in seeds]
                    ranges[family][arm][ds][metric] = {
                        "seed_means": dict(zip(map(str, seeds), vals)),
                        "mean_across_mask_seeds": float(np.mean(vals)), "range_across_mask_seeds": [float(min(vals)), float(max(vals))]}
        for i, a in enumerate(seeds):
            ma = np.asarray(per_seed[str(a)][family]["mask"], dtype=bool)
            for b in seeds[i + 1:]:
                mb = np.asarray(per_seed[str(b)][family]["mask"], dtype=bool)
                overlap[family].append({"seeds": [a, b], "intersection_heads": int((ma & mb).sum()),
                                        "union_heads": int((ma | mb).sum()), "jaccard": float((ma & mb).sum() / (ma | mb).sum())})
    result = {"status": "complete", "budget": budget, "mask_seeds": seeds, "sources": source_files,
              "protocol": "Same frozen model pair and fixed discovery/validation/test data. Only mask initialization and minibatch order vary. CIs resample matched evaluation sequences within each seed. Across-seed ranges are descriptive over three mask seeds, not confidence intervals across model training seeds.",
              "full_model_endpoints_identical_across_mask_seeds": True,
              "teacher_kl": "Own dynamic teacher per model; fixed-route damage uses fixed_routes_full and the KL-increment convention documented in summary.json.",
              "per_seed": per_seed, "dag_minus_dense_per_seed": contrasts,
              "across_seed_ranges": ranges, "support_overlap": overlap}
    dump(out / "mask_seed_replication.json", result)
    lines = [f"# {budget}-head support: three mask seeds", "",
        "Both frozen models were rerun with the same discovery, validation and test sequences. "
        "Only mask initialization and optimization minibatch order change. Full-model per-sequence endpoints match across all runs. "
        "These are mask-optimization repeats, not independently pretrained models.", "",
        "## Copy retention", "", "Normalized accuracy loss is relative to each run's full model. "
        "Paired 95% intervals resample sequences within that mask seed and are unadjusted for multiple comparisons.", "",
        "| Dataset | Mask seed | Dense loss, % | DAG loss, % | DAG − dense, 95% CI |", "|---|---:|---:|---:|---:|"]
    learned = f"learned_{budget}"
    for ds in SYNTHETIC:
        for seed in seeds:
            b, d = [per_seed[str(seed)][f]["arms"][learned][ds]["normalized_accuracy_loss"] for f in FAMILIES]
            delta = contrasts[str(seed)][learned][ds]["normalized_accuracy_loss"]
            lines.append(f"| {LABELS[ds]} | {seed} | {fmt(b, True)} | {fmt(d, True)} | {fmt(delta, True, True)} |")
    lines += ["", "## Natural-text NLL cost", "", "| Mask seed | Dense ΔNLL | DAG ΔNLL | DAG − dense, 95% CI |", "|---|---:|---:|---:|"]
    for seed in seeds:
        b, d = [per_seed[str(seed)][f]["arms"][learned][NATURAL]["target_nll_damage"] for f in FAMILIES]
        delta = contrasts[str(seed)][learned][NATURAL]["target_nll_damage"]
        lines.append(f"| {seed} | {fmt(b)} | {fmt(d)} | {fmt(delta, ci=True)} |")
    lines += ["", "## Across-mask-seed ranges", "",
        "Each entry is the mean across three mask seeds followed by their minimum and maximum. "
        "These ranges are not training-seed confidence intervals. Synthetic damage is normalized accuracy loss (%); natural-text damage is ΔNLL.", "",
        "| Arm | Dataset | Dense damage mean [range] | DAG damage mean [range] |", "|---|---|---:|---:|"]
    for arm in [learned, f"remove_learned_{budget}"]:
        for ds in DATASETS:
            metric = "target_nll_damage" if ds == NATURAL else "normalized_accuracy_loss"
            factor = 1 if ds == NATURAL else 100
            cols = []
            for family in FAMILIES:
                s = ranges[family][arm][ds][metric]
                lo, hi = np.asarray(s["range_across_mask_seeds"]) * factor
                cols.append(f'{s["mean_across_mask_seeds"]*factor:.4f} [{lo:.4f}, {hi:.4f}]')
            lines.append(f"| {arm} | {LABELS[ds]} | {cols[0]} | {cols[1]} |")
    lines += ["", "## KL across mask seeds", "", "Each model uses its own dynamic full-model teacher.", "",
        "| Dataset | Dense KL mean [range] | DAG KL mean [range] |", "|---|---:|---:|"]
    for ds in DATASETS:
        cols = []
        for family in FAMILIES:
            s = ranges[family][learned][ds]["teacher_kl"]
            lo, hi = s["range_across_mask_seeds"]
            cols.append(f'{s["mean_across_mask_seeds"]:.4f} [{lo:.4f}, {hi:.4f}]')
        lines.append(f"| {LABELS[ds]} | {cols[0]} | {cols[1]} |")
    lines += ["", "Raw per-sequence arrays remain in the source JSONs. [Replication statistics](mask_seed_replication.json) "
        "retain every seed, every fixed-route reference, validation histories and pairwise support overlap. "
        "[Main experiment](summary.md).", ""]
    (out / "mask_seed_replication.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("experiments/results/circuit_interpretability_20260928/head_support"))
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--replication-mask-seeds", nargs="+", type=int, default=[20261001, 20261002])
    ap.add_argument("--replication-budget", type=int, default=96)
    args = ap.parse_args()
    args.input_dir.mkdir(parents=True, exist_ok=True)
    results, status = {}, {}
    for family in FAMILIES:
        path = args.input_dir / f"{family}.json"
        if not path.exists():
            status[family] = "missing"
            continue
        try:
            j = json.loads(path.read_text())
        except json.JSONDecodeError:
            status[family] = "incomplete JSON write"
            continue
        status[family] = "complete" if j.get("complete") is True else "incomplete"
        if status[family] == "complete":
            results[family] = j
    if len(results) != 2:
        partial = {"status": "incomplete", "models": status,
                   "note": "Final paired comparison is produced only after both runs report complete=true."}
        dump(args.input_dir / "summary.json", partial)
        (args.input_dir / "summary.md").write_text("# Attention-head support: incomplete\n\n" +
            "\n".join(f"- {f}: {s}" for f, s in status.items()) +
            "\n\nFinal paired comparison awaits both completed runs.\n")
        print(json.dumps(partial))
        return
    validate_pair(results)
    summary = analyze(results, args)
    dump(args.input_dir / "summary.json", summary)
    report(args.input_dir, summary)
    plot(args.input_dir, summary)
    plot(args.input_dir, summary, controls=False)
    summarize_seed_replication(args, results)
    print(json.dumps({"status": "complete", "summary": str(args.input_dir / "summary.md")}))


if __name__ == "__main__":
    main()
