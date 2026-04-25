"""HF-compatible wrapper around FourWayDAGFormer for use with lm-eval-harness.

Usage:
    python scripts/dagformer_lm_eval.py \
        --yaml configs/pretrain_75m_dagformer.yaml \
        --ckpt checkpoints/pretrain_75m_dagformer/checkpoint_step3000.pt \
        --tasks piqa,hellaswag,arc_easy,arc_challenge,winogrande,boolq \
        --device cuda:0 --batch_size 8
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import yaml
from transformers import AutoConfig, AutoTokenizer, Olmo2Config, Olmo2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, '/workspace/dagformer')

from src.model.olmo_graph import FourWayDAGFormer
from src.model.predictor import FourWayPredictor
from scripts.pretrain_dagformer import (
    DAGFormerPretrainConfig,
    predict_fourway_routing,
    apply_fourway_stream_mask,
    apply_deterministic_routing_transforms,
)


class DAGFormerForCausalLM(nn.Module):
    """Wraps FourWayDAGFormer + predictor with an HF-compatible forward().
    Satisfies the subset of the HF interface lm-eval uses for log-likelihood:
      - forward(input_ids, attention_mask=None, labels=None) -> CausalLMOutputWithPast
      - .config (has vocab_size, hidden_size, ...)
    """
    def __init__(self, cfg: DAGFormerPretrainConfig, base: Olmo2ForCausalLM,
                 predictor: FourWayPredictor, fourway: FourWayDAGFormer):
        super().__init__()
        self._pretrain_cfg = cfg
        self.base_model = base
        self.predictor = predictor
        self.fourway = fourway
        # HF-ish config
        self.config = base.config
        self.generation_config = base.generation_config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def tie_weights(self):
        return self.base_model.tie_weights()

    def can_generate(self):
        return True

    def __getattr__(self, name):
        # Proxy missing attributes to base_model (e.g. .name_or_path, .num_parameters)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, return_dict=True, **kw):
        # Full-seq forward. lm-eval only needs logits at each position.
        with torch.no_grad():
            rw = predict_fourway_routing(self.predictor, self.base_model,
                                         input_ids, self._pretrain_cfg)
            rw = apply_fourway_stream_mask(rw, self._pretrain_cfg)
            rw = apply_deterministic_routing_transforms(rw, self._pretrain_cfg, global_step=10**9)
            logits = self.fourway(input_ids, rw)
        loss = None
        if labels is not None:
            lv = logits.view(-1, logits.size(-1))
            lb = labels.view(-1)
            loss = nn.functional.cross_entropy(lv, lb, ignore_index=-100)
        if return_dict:
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
        return (loss, logits) if loss is not None else (logits,)


def build_dagformer(yaml_path: str, ckpt_path: str, device: str = "cuda:0",
                    dtype: torch.dtype = torch.bfloat16) -> DAGFormerForCausalLM:
    cfg_dict = yaml.safe_load(open(yaml_path))
    cfg = DAGFormerPretrainConfig(**{k: v for k, v in cfg_dict.items()
                                     if k in DAGFormerPretrainConfig.__dataclass_fields__})
    # Build base Olmo2
    olmo_cfg = Olmo2Config(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=getattr(cfg, "tie_word_embeddings", True),
        max_position_embeddings=cfg.max_position_embeddings,
    )
    base = Olmo2ForCausalLM(olmo_cfg)
    # Build predictor (mini-encoder variant)
    predictor = FourWayPredictor(
        vocab_size=cfg.vocab_size,
        encoder_dim=cfg.predictor_encoder_dim,
        encoder_layers=cfg.predictor_encoder_layers,
        encoder_heads=cfg.predictor_encoder_heads,
        max_seq_len=cfg.predictor_max_seq_len,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        hidden_dim=cfg.fourway_hidden,
        causal=getattr(cfg, "predictor_causal", True),
    )
    # Replace RMSNorms before wrapping (matches training path)
    if getattr(cfg, "replace_rmsnorm", False):
        from scripts.pretrain_dagformer import replace_olmo_rmsnorm
        replace_olmo_rmsnorm(base)
    # Build FourWayDAGFormer
    fourway = FourWayDAGFormer(
        model=base,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        use_local_correction=(cfg.routing_mode == "fourway_corrected"),
        use_v_norm=getattr(cfg, "use_v_norm", False),
        correction_hidden=getattr(cfg, "correction_hidden", 128),
    )

    # Load ckpt
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_path" in ck:
        base_sd = torch.load(ck["model_state_path"], map_location="cpu", weights_only=False)
    else:
        base_sd = ck.get("model_state_dict", {})
    base.load_state_dict(base_sd, strict=False)
    predictor.load_state_dict(ck["predictor_state_dict"], strict=False)
    if "routing_state_dict" in ck:
        # routing_state contains correction_mlps + v_norms (belong to fourway model)
        fourway.load_state_dict(ck["routing_state_dict"], strict=False)
    model = DAGFormerForCausalLM(cfg, base, predictor, fourway)
    model = model.to(device=device, dtype=dtype).eval()
    # Keep correction_mlps in fp32: their forward does hidden.float() first
    if hasattr(fourway, 'correction_mlps'):
        for mlp in fourway.correction_mlps:
            mlp.to(torch.float32)
    # Keep v_norms in fp32 too (rms_norm with bf16 input + float weight mismatch)
    if hasattr(fourway, 'v_norms'):
        for vn in fourway.v_norms:
            vn.to(torch.float32)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--tasks", default="piqa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output_path", default=None)
    a = ap.parse_args()

    dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[a.dtype]
    model = build_dagformer(a.yaml, a.ckpt, device=a.device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    wrapped = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size, device=a.device)
    res = simple_evaluate(model=wrapped, tasks=a.tasks.split(","), limit=a.limit)
    # strip non-serializable
    results = res.get("results", {})
    print(json.dumps(results, indent=2, default=str))
    if a.output_path:
        os.makedirs(os.path.dirname(a.output_path) or ".", exist_ok=True)
        json.dump(results, open(a.output_path, "w"), indent=2, default=str)
        print(f"saved: {a.output_path}")
