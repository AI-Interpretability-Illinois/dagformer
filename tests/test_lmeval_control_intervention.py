import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/results/lmeval"))
from interventions import install_context_edit, intervention_label
from interp_common import flatten_alpha, unflatten_alpha
from eval_context_fidelity import apply_edges


def test_random_control_hooks_match_reference_norm_and_record_coordinates(tmp_path):
    class Predictor(torch.nn.Module):
        def forward(self, flat):
            return unflatten_alpha(flat, 2, 4)

    circuit = tmp_path / "circuit.json"
    circuit.write_text(json.dumps({"conn_scores": {"L1/h1/q<-src0": -1.}}))
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2, num_attention_heads=4),
                            fourway_predictor=Predictor(),
                            fourway_model=SimpleNamespace(correction_mlps=[torch.nn.Identity()]))
    record = install_context_edit(model, "context_both", str(circuit), 1.25, control_seed=1000)
    assert record["control_seed"] == 1000
    assert record["edges"] != record["reference_edges"]
    flat = torch.arange(26).float().reshape(1, 1, 26).expand(2, 3, -1).clone()
    reference_change = apply_edges(flat, 1, 4, record["reference_edges"], 1.25) - flat
    predictor_change = flatten_alpha(model.fourway_predictor(flat)) - flat
    correction_change = model.fourway_model.correction_mlps[0](flat) - flat
    torch.testing.assert_close(predictor_change, correction_change)
    torch.testing.assert_close(predictor_change.norm(dim=-1), reference_change.norm(dim=-1))
    assert predictor_change.abs().sum() > 0
    assert "random1000_normmatched" in intervention_label("context_both", 1.25, 1000)
