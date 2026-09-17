import itertools
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_context_prompt_clusters import cluster_stats


def test_repeating_the_same_prompts_does_not_narrow_cluster_uncertainty():
    reference = np.array([1., 0., 0.])
    variant = np.array([0., 0., 1.])
    draws = np.array(list(itertools.product(range(3), repeat=3)))
    counts = np.ones(3, dtype=int)
    original = cluster_stats(variant, reference, np.arange(3), counts, draws, counts[draws].sum(1))
    repeated_counts = counts * 4
    repeated = cluster_stats(np.repeat(variant, 4), np.repeat(reference, 4),
                             np.repeat(np.arange(3), 4), repeated_counts, draws,
                             repeated_counts[draws].sum(1))
    assert original == repeated
    lo, hi = original['prompt_cluster_bootstrap_95ci']
    assert lo < 0 < hi


def test_item_weighting_is_distinct_from_equal_prompt_weighting():
    reference = np.array([0., 0., 1.])
    variant = np.array([1., 1., 0.])
    groups = np.array([0, 0, 1])
    counts = np.array([2, 1])
    draws = np.array(list(itertools.product(range(2), repeat=2)))
    result = cluster_stats(variant, reference, groups, counts, draws, counts[draws].sum(1))
    assert result['delta'] == 1 / 3
    assert result['equal_unique_prompt_delta'] == 0
