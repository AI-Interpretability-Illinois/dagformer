"""Regression checks for the recorded PR #1 comparison and its task groups."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/results/lmeval"
spec = importlib.util.spec_from_file_location("reasoning_comparison", HERE / "compare.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_recorded_multiple_choice_counts_exclude_bpb_and_generation():
    payloads = comparison.load_results(HERE / "reasoning", "reasoning")
    _, meta = comparison.build_tables(payloads)
    assert meta["wins"] == {"dagformer": 18, "baseline": 7, "tie": 2}
    assert comparison.multiple_choice_wins(meta["rows"]) == {
        "dagformer": 14, "baseline": 5, "tie": 2,
    }
    assert abs(comparison.sign_test_p(14, 5) - 0.063568115234375) < 1e-12


def test_recorded_steps_survive_off_cluster_report_generation():
    payloads = comparison.load_results(HERE / "reasoning", "reasoning")
    steps = {p["model"]["name"]: comparison.recorded_step(p) for p in payloads}
    assert steps["300m-baseline"] == 12000
    assert steps["300m-dagformer"] == 9000
    markdown, _ = comparison.render(payloads)
    assert "14 wins, 5 losses, 2 ties" in markdown
    assert "lower bound" not in markdown
