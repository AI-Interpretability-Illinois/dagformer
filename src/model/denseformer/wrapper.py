"""HF-compatible DenseFormer baseline for training + lm-eval.

FAIRNESS: DenseFormer here is the SAME base as our dense OLMo-2 baseline
(`scripts/pretrain_baseline.py::create_model`) — identical
hidden/layers/heads/vocab/tokenizer, POST-norm, q_norm/k_norm — with the ONLY
difference being a Depth-Weighted-Average (DWA) module inserted after each
decoder layer (Pagliardini et al., NeurIPS 2024, arXiv:2402.02622). The DWA
adds ~L²/2 scalar parameters (negligible), so the param count matches the
dense baseline to within a handful of scalars.

Integration contract (mirrors src/model/muddformer/wrapper.py::build_muddformer):
  build_denseformer(config) -> an HF-CausalLM-like object whose
  forward(input_ids=...) returns CausalLMOutputWithPast with .logits [B,T,vocab].
  This slots into scripts/pretrain_baseline.py's training loop unchanged (the
  loop only ever calls model(input_ids=...) and reads outputs.logits).

Implementation:
  * `DenseFormerOlmo2Model` subclasses HF `Olmo2Model` and overrides `forward`
    to (a) seed the DWA accumulators with the token embeddings (h_0) and
    (b) apply `dwa_modules(hidden_states, block_idx=i)` after each decoder
    layer, so each layer's output is replaced by the depth weighted average of
    {embeddings, all prior layer outputs, current layer output}.
  * `DenseFormerForCausalLM` subclasses HF `Olmo2ForCausalLM` and swaps in the
    DWA-augmented backbone. `lm_head` / weight tying / loss are inherited.

Only the training forward path is reimplemented (no kv-cache / generate). That
is all pretrain_baseline.py exercises; at init the DWA is identity so the model
is behaviourally a plain OLMo-2 until the alphas move.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2Model, create_causal_mask
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    from .dwa import DWAModules
except ImportError:  # allow `python src/model/denseformer/wrapper.py` (no package parent)
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from dwa import DWAModules  # type: ignore


class DenseFormerOlmo2Model(Olmo2Model):
    """Olmo2 backbone with a DWA tap after every decoder layer."""

    def __init__(self, config: Olmo2Config) -> None:
        super().__init__(config)
        # DWA knobs are stashed on the config by build_denseformer (defaults: dense).
        dilation = int(getattr(config, "dwa_dilation", 1))
        period = int(getattr(config, "dwa_period", 1))
        self.dwa_modules = DWAModules(
            n_blocks=config.num_hidden_layers, dilation=dilation, period=period
        )
        # Re-run DWA identity init: HF's post_init() may have re-inited nn.Linear
        # submodules (DWA alphas are Linear layers), which would clobber the
        # plain-residual init. Restore it so training starts from vanilla OLMo-2.
        self.dwa_modules._init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Training-mode forward: standard Olmo2 loop with DWA after each layer.

        Reimplements the (short) parent loop body directly rather than wrapping
        the decorated parent forward, so the DWA can be inserted cleanly. No
        kv-cache / generation path — matches what pretrain_baseline.py uses.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # h_0 = token embeddings seeds the depth accumulators.
        self.dwa_modules.init_accumulators(hidden_states)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # Depth-weighted average of {h_0, ..., h_i} replaces the layer output.
            hidden_states = self.dwa_modules(hidden_states, block_idx=i)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class DenseFormerForCausalLM(Olmo2ForCausalLM):
    """Olmo2ForCausalLM with the DWA-augmented backbone.

    Inherits lm_head, weight tying, and loss from the HF class; only the
    backbone (`self.model`) is swapped for the DWA version.
    """

    def __init__(self, config: Olmo2Config) -> None:
        super().__init__(config)
        self.model = DenseFormerOlmo2Model(config)
        # Re-tie / re-init to keep lm_head consistent with the new backbone's
        # embeddings (post_init handles tying when tie_word_embeddings=True).
        self.post_init()
        # CRITICAL: post_init()/_init_weights re-inits every nn.Linear — including
        # the DWA alpha layers — with the generic std-init, clobbering the
        # plain-residual identity init. Restore it LAST so training starts as
        # vanilla OLMo-2 (each block ≈ standard residual until the alphas move).
        self.model.dwa_modules._init_weights()


def build_denseformer(cfg) -> DenseFormerForCausalLM:
    """Build a DenseFormerForCausalLM from a PretrainConfig-shaped object.

    Maps our standard scaling-study fields onto an Olmo2Config exactly as the
    dense baseline's create_model does (so the only architectural difference is
    the DWA), plus optional DWA knobs:
      dwa_dilation (default 1 = dense), dwa_period (default 1 = every block).
    """
    model_config = Olmo2Config(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_attention_heads,  # MHA (not GQA), like the baseline
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=getattr(cfg, "tie_word_embeddings", True),
        max_position_embeddings=cfg.max_position_embeddings,
    )
    # Stash DWA knobs on the config so the model + any reload see them.
    model_config.dwa_dilation = int(getattr(cfg, "dwa_dilation", 1))
    model_config.dwa_period = int(getattr(cfg, "dwa_period", 1))
    return DenseFormerForCausalLM(model_config)


if __name__ == "__main__":
    # CPU sanity test: build tiny, check param count, forward shape, and that
    # the DWA is identity at init (output ≈ plain-residual OLMo-2 forward).
    import torch.nn.functional as F

    torch.manual_seed(0)

    class _TinyCfg:
        hidden_size = 128
        num_hidden_layers = 4
        num_attention_heads = 4
        intermediate_size = 256
        vocab_size = 100352
        tie_word_embeddings = True
        max_position_embeddings = 4096
        dwa_dilation = 1
        dwa_period = 1

    cfg = _TinyCfg()
    model = build_denseformer(cfg).eval()

    # --- param count vs a plain OLMo-2 baseline at identical dims ---
    dense_cfg = Olmo2Config(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=cfg.tie_word_embeddings,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    dense = Olmo2ForCausalLM(dense_cfg)

    def _unique(m):
        seen, tot = set(), 0
        for p in m.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                tot += p.numel()
        return tot

    df_params = _unique(model)
    dense_params = _unique(dense)
    dwa_params = sum(p.numel() for p in model.model.dwa_modules.parameters())
    print(f"DenseFormer unique params: {df_params:,}")
    print(f"Dense OLMo-2 unique params: {dense_params:,}")
    print(f"DWA-only params: {dwa_params}  (delta = {df_params - dense_params})")

    # --- forward shape ---
    ids = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size), out.logits.shape
    print(f"logits shape OK: {tuple(out.logits.shape)}")

    # --- identity-init check: DWA output ≈ plain-residual forward ---
    # Build a plain Olmo2 backbone that SHARES this model's weights, so the only
    # possible difference is the DWA taps. At init the DWA is identity, so the
    # two forwards must match to numerical precision.
    plain_backbone = Olmo2Model(model.config)
    plain_backbone.load_state_dict(model.model.state_dict(), strict=False)
    plain_backbone.eval()
    with torch.no_grad():
        dwa_hidden = model.model(input_ids=ids).last_hidden_state
        plain_hidden = plain_backbone(input_ids=ids).last_hidden_state
    max_diff = (dwa_hidden - plain_hidden).abs().max().item()
    print(f"max|DWA_hidden - plain_hidden| at init = {max_diff:.3e}")
    assert max_diff < 1e-4, f"DWA is not identity at init (diff={max_diff})"
    print("IDENTITY-INIT CHECK: PASS")
    print("ALL SANITY CHECKS PASSED")
