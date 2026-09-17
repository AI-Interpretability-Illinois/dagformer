"""Stream controls retain their support constraints and independent sampling units."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_domain_streams import RoutingLayout, item_means, permute_eligible


def test_paraphrases_are_averaged_within_item():
    prompts = [SimpleNamespace(item=i) for i in [4, 2, 4, 4, 2]]
    np.testing.assert_allclose(item_means(prompts, [1., 5., 3., 8., 7.]), [6., 4.])


def test_controls_preserve_per_layer_stream_values_and_hyper_support():
    layout = RoutingLayout(4, 2)
    eligible = layout.hyper_arr
    direction = np.arange(layout.D, dtype=float) * eligible
    direction[::3] = 0
    shuffled = permute_eligible(direction, eligible, layout, 42)
    assert not np.array_equal(shuffled, direction)
    assert np.all(shuffled[~eligible] == 0)
    for layer in range(1, 4):
        for stream in ("q", "k", "v", "r"):
            mask = eligible & layout.mask(layers=[layer], streams=[stream])
            np.testing.assert_array_equal(np.sort(shuffled[mask]), np.sort(direction[mask]))
