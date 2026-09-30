"""Evaluation must retain the trained predictor variant and all routing weights."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_lm_harness import build_base_model, load_fourway
from src.model.olmo_graph import FourWayDAGFormer
from src.model.predictor import (FourWayPredictor, FourWayPerLayerPredictor,
                                 FourWayPositionalPredictor, FourWayStaticPredictor)


CFG = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
           intermediate_size=32, vocab_size=32, max_position_embeddings=32,
           seq_len=8, routing_mode="fourway_corrected", correction_hidden=8,
           predictor_encoder_dim=8, predictor_encoder_heads=2,
           predictor_encoder_layers=1, predictor_max_seq_len=32, fourway_hidden=8)


@pytest.mark.parametrize("variant", ["encoder", "static", "pos_table", "per_layer"])
def test_checkpoint_roundtrip_preserves_variant_routing_and_logits(tmp_path, variant):
    torch.manual_seed(17)
    cfg = {**CFG, "fourway_predictor_variant": variant}
    base = build_base_model(cfg, "cpu")
    model = FourWayDAGFormer(base, num_layers=2, num_heads=2,
                            use_local_correction=True, correction_hidden=8).eval()
    if variant in ("encoder", "per_layer"):
        predictor_class = FourWayPerLayerPredictor if variant == "per_layer" else FourWayPredictor
        predictor = predictor_class(vocab_size=32, encoder_dim=8, encoder_layers=1,
                                     encoder_heads=2, max_seq_len=32, num_layers=2,
                                     num_heads=2, hidden_dim=8)
    elif variant == "static":
        predictor = FourWayStaticPredictor(num_layers=2, num_heads=2)
    else:
        predictor = FourWayPositionalPredictor(max_seq_len=8, num_layers=2, num_heads=2)
    predictor.eval()
    with torch.no_grad():
        # Distinguish this checkpoint from each variant's identity initialization.
        for parameter in predictor.parameters():
            parameter.add_(torch.randn_like(parameter) * .03)
        model.correction_mlps[0][-1].weight.normal_(std=.03)
    path = tmp_path / "checkpoint.pt"
    torch.save({"predictor_state_dict": predictor.state_dict(),
                "model_state_dict": base.state_dict(),
                "routing_state_dict": model.state_dict()}, path)
    loaded, loaded_predictor = load_fourway(str(path), cfg, "cpu")
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = predictor(ids)
        actual = loaded_predictor(ids)
        for stream in expected:
            for a, b in zip(expected[stream], actual[stream]):
                assert torch.equal(a, b)
        assert torch.equal(model(ids, expected), loaded(ids, actual))


def test_missing_backbone_is_rejected(tmp_path):
    predictor = FourWayStaticPredictor(num_layers=2, num_heads=2)
    path = tmp_path / "checkpoint.pt"
    torch.save({"predictor_state_dict": predictor.state_dict()}, path)
    with pytest.raises(ValueError, match="no backbone weights"):
        load_fourway(str(path), {**CFG, "fourway_predictor_variant": "static"}, "cpu")
