"""HF-compatible wrapper around MUDDFormer for training + lm-eval.

MUDDFormer's native forward(idx, input_pos) is gpt-fast style; this wrapper:
  - Accepts forward(input_ids=, attention_mask=, labels=) per HF convention
  - Sets up freqs_cis / causal_mask lazily on first call (training mode)
  - Returns CausalLMOutputWithPast so DDP + standard training loops just work
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_muddformer import MUDDFormerConfig
from .modeling_muddformer import MUDDFormer, find_multiple


class MUDDFormerForCausalLM(nn.Module):
    """HF-compatible wrapper exposing MUDDFormer with input_ids / labels."""

    def __init__(self, mudd_config: MUDDFormerConfig):
        super().__init__()
        # Force training mode: skip kv cache / layer cache to allow grads
        mudd_config.is_training = True
        mudd_config.use_layer_cache = False
        self.model = MUDDFormer(mudd_config)
        self._caches_ready = False
        self.config = mudd_config  # passthrough for HF callers
        # Tied weights
        if getattr(mudd_config, "tie_word_embeddings", False):
            self.model.output.weight = self.model.tok_embeddings.weight
        # PyTorch's nn.Embedding default is N(0, 1) — way too large. With tied output,
        # logits std = sqrt(dim) ≈ 22 → init loss ~470 instead of ~ln(vocab)=11.5.
        # Use GPT-style std=0.02 so logits start near 0.
        with torch.no_grad():
            self.model.tok_embeddings.weight.normal_(mean=0.0, std=0.02)
            # Identity-like init for dense routing — matches DAGFormer style (last source = current layer)
            for db in self.model.dense_bs:
                db.zero_(); db[..., -1] = 1.0
            if hasattr(self.model, "dynamic_dense"):
                for blk in self.model.dynamic_dense:
                    nn.init.zeros_(blk.w2.weight)
                    if blk.w2.bias is not None:
                        nn.init.zeros_(blk.w2.bias)

    def _ensure_caches(self, ids: torch.Tensor):
        if self._caches_ready:
            return
        B, T = ids.shape
        self.model.setup_caches(max_batch_size=max(B, 64),
                                max_seq_length=max(T, 1024),
                                dtype=ids.new_zeros(()).float().dtype)
        self._caches_ready = True

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", False):
            self.model.output.weight = self.model.tok_embeddings.weight

    def can_generate(self):
        return True

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask=None,  # ignored: causal mask only
                labels=None,
                past_key_values=None,
                use_cache=False,
                return_dict=True,
                **kw):
        self._ensure_caches(input_ids)
        out = self.model(input_ids, return_tensor=False)
        # MUDDFormer returns namedtuple with .logits (or tensor if return_tensor=True)
        logits = out.logits if hasattr(out, "logits") else out

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous() if labels.shape[-1] == logits.shape[-2] else logits.contiguous()
            shift_labels = labels[..., 1:].contiguous() if labels.shape[-1] == logits.shape[-2] else labels.contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)

        if return_dict:
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
        return (loss, logits) if loss is not None else (logits,)


def build_muddformer(cfg) -> MUDDFormerForCausalLM:
    """Build MUDDFormerForCausalLM from a PretrainConfig-shaped object.

    Maps our standard scaling-study fields → MUDDFormer config:
      hidden_size → dim, num_hidden_layers → n_layer, num_attention_heads → n_head,
      intermediate_size → intermediate_size, vocab_size → vocab_size,
      max_position_embeddings → block_size.
    """
    mudd_cfg = MUDDFormerConfig(
        block_size=cfg.max_position_embeddings,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.num_hidden_layers,
        n_head=cfg.num_attention_heads,
        dim=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        tie_word_embeddings=getattr(cfg, "tie_word_embeddings", True),
        is_training=True,
        use_layer_cache=False,
        # Architecture knobs (defaults: full multiway dynamic dense routing)
        dense=True,
        dynamic_dense=True,
        sepln=True,
        dense_type="qkvr",
        use_qk_norm=True,  # match Olmo2/our other scaling baselines
    )
    return MUDDFormerForCausalLM(mudd_cfg)
