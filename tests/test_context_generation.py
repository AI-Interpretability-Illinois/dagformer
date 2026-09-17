import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_context_generation import first_value


def test_first_mentioned_value_uses_word_boundaries_and_abstains():
    assert first_value("Blue, although the real color was red", ["red", "blue"]) == "blue"
    assert first_value("a reddish object", ["red", "blue"]) is None
    assert first_value("I do not know.", ["red", "blue"]) is None
