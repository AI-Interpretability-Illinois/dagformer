"""Task suites, reasoning first.

Why these tasks, at this scale
------------------------------
The checkpoints have 75M-600M backbone scales and were trained from a 12B-token
Dolma mmap slice. Tokens actually consumed depend on checkpoint step and global
batch size; 12B is the available corpus size, not each checkpoint's budget.
Two observations motivate the suites:

1. **Generative math is near the floor.**  ``gsm8k`` exact-match at this scale
   was 0.5-2.5% in the original 200-item comparison. Full-test paired intervals
   help assess small differences. Read generated accuracy separately from
   ``gsm8k_bpb``.

2. **Likelihood-scored reasoning still discriminates.**  ``gsm8k_bpb`` (defined
   in ``tasks/gsm8k_bpb.yaml``) scores the entire question plus gold answer,
   including the answer's arithmetic prose. It is not answer-conditional
   likelihood or generated reasoning accuracy. It
   is smooth, has no floor effect, and separates models that are all at 0% EM.
   It is a custom task in this directory, not an upstream lm-eval task, so it
   is comparable across the models evaluated here but not to published numbers.

Several multiple-choice scores are close to label chance, including at 300M.
Interpret the per-task scores, paired differences and trivial-label baselines
before drawing a broad reasoning claim.

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
    "gsm8k_answer_bpb": "bits_per_byte,none",
    "gsm8k_question_bpb": "bits_per_byte,none",
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
