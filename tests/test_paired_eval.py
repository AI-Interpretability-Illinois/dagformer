"""The reported uncertainty must follow paired document aggregation."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_paired_eval import paired_difference


def test_identical_accuracy_has_zero_paired_uncertainty():
    rows = {i: {"acc": i % 2, "doc_hash": str(i)} for i in range(20)}
    result = paired_difference(rows, rows, "acc,none", draws=100)
    assert result["paired_bootstrap_95ci"] == [0., 0.]
    assert result["mcnemar_exact_p"] == 1.


def test_bpb_bootstrap_preserves_byte_weighting():
    # Every document improves by exactly one bit per byte, despite differing
    # lengths. Averaging per-document total likelihoods would be incorrect.
    b = {0: {"bits_per_byte": [-4 * math.log(2), 2]},
         1: {"bits_per_byte": [-20 * math.log(2), 10]}}
    d = {0: {"bits_per_byte": [-2 * math.log(2), 2]},
         1: {"bits_per_byte": [-10 * math.log(2), 10]}}
    result = paired_difference(b, d, "bits_per_byte,none", draws=100)
    assert result["baseline"] == pytest.approx(2.)
    assert result["dagformer"] == pytest.approx(1.)
    assert result["paired_bootstrap_95ci"] == pytest.approx([1., 1.])


def test_mismatched_document_content_is_rejected():
    with pytest.raises(ValueError, match="content differs"):
        paired_difference({0: {"acc": 0, "doc_hash": "a"}},
                          {0: {"acc": 1, "doc_hash": "b"}}, "acc,none")
