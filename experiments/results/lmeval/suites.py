"""Task suites, reasoning first.

Why these tasks, at this scale
------------------------------
The checkpoints here are 75M-600M params trained on 12B Dolma tokens.  Two
consequences drive the suite design:

1. **Generative math is near the floor.**  ``gsm8k`` exact-match at this scale
   is ~0-2% for every model in the comparison; it is reported because it is the
   benchmark that gets asked about, and because ``flexible-extract`` does pick
   up the occasional correct answer, but a 0.0-vs-0.4 gap is noise, not
   evidence.  Read it together with ``gsm8k_bpb``.

2. **Likelihood-scored reasoning still discriminates.**  ``gsm8k_bpb`` (defined
   in ``tasks/gsm8k_bpb.yaml``) scores bits-per-byte of the *gold* chain of
   thought, i.e. how well the model models step-by-step arithmetic prose.  It
   is smooth, has no floor effect, and separates models that are all at 0% EM.
   It is a custom task in this directory, not an upstream lm-eval task, so it
   is comparable across the models evaluated here but not to published numbers.

The multiple-choice reasoning tasks (arc_challenge, mathqa, commonsense_qa,
social_iqa, openbookqa, winogrande) are all above chance-ish at 300M+ and are
the load-bearing part of the comparison.

Not in the default suite: ``logiqa`` and ``logiqa2``.  Both hub repos ship a
dataset loading script, so they only load with ``HF_DATASETS_TRUST_REMOTE_CODE=1``
(``python prefetch_data.py --tasks logiqa --trust-remote-code`` first); keeping
them out means the suite runs without executing code from the hub.
"""
from __future__ import annotations

SUITES: dict[str, list[str]] = {
    # Headline suite: math/CoT reasoning first, then the multiple-choice
    # reasoning benchmarks that are still informative below 1B params.
    "reasoning": [
        "gsm8k",
        "gsm8k_bpb",
        "arc_challenge",
        "arc_easy",
        "mathqa",
        "commonsense_qa",
        "social_iqa",
        "openbookqa",
        "winogrande",
    ],
    # Math only — cheapest way to iterate on the generative path.
    "math": ["gsm8k", "gsm8k_bpb", "mathqa", "asdiv"],
    # Just the generative task, for smoke tests and timing.
    "gen": ["gsm8k"],
    # The suite the existing experiments/results/lmeval/*.json were produced
    # with (scripts/eval_lm_harness.py DEFAULT_TASKS) — keep for continuity.
    "core": [
        "lambada_openai",
        "hellaswag",
        "piqa",
        "arc_easy",
        "winogrande",
        "openbookqa",
        "sciq",
        "boolq",
        "wikitext",
    ],
}
SUITES["all"] = SUITES["reasoning"] + [t for t in SUITES["core"] if t not in SUITES["reasoning"]]

# Metric to headline per task in the comparison table, in preference order.
# acc_norm is the convention for length-varying multiple choice; gsm8k's
# flexible-extract is the more forgiving of its two filters.
PRIMARY_METRIC: dict[str, str] = {
    "gsm8k": "exact_match,flexible-extract",
    "gsm8k_bpb": "bits_per_byte,none",
    "wikitext": "bits_per_byte,none",
    "lambada_openai": "acc,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "sciq": "acc,none",
    "commonsense_qa": "acc,none",
    "social_iqa": "acc,none",
    "logiqa": "acc_norm,none",
    "logiqa2": "acc_norm,none",
    "asdiv": "acc,none",
}
DEFAULT_METRIC_ORDER = ("acc_norm,none", "acc,none", "exact_match,flexible-extract")

# Lower is better for these; everything else is an accuracy.
LOWER_IS_BETTER_METRICS = ("bits_per_byte", "perplexity", "byte_perplexity", "word_perplexity")


def resolve_tasks(suite: str | None, tasks: str | None) -> list[str]:
    """``--tasks`` wins over ``--suite``; both accept comma-separated names."""
    if tasks:
        return [t.strip() for t in tasks.split(",") if t.strip()]
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {', '.join(SUITES)}")
    return list(SUITES[suite])


def primary_metric(task: str, metrics: dict) -> str | None:
    """Pick the metric key to headline for ``task`` out of an lm-eval result dict."""
    preferred = PRIMARY_METRIC.get(task)
    if preferred and preferred in metrics:
        return preferred
    for key in DEFAULT_METRIC_ORDER:
        if key in metrics:
            return key
    for key in metrics:
        if not key.endswith("_stderr,none") and key != "alias":
            return key
    return None


def higher_is_better(metric_key: str) -> bool:
    return not metric_key.split(",")[0].endswith(LOWER_IS_BETTER_METRICS)
