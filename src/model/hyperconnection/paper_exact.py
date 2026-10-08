"""Paper-exact hyper-connections (HC) and manifold-constrained hyper-connections (mHC) on the OLMo-2 decoder.

Both papers place one connection module around every sub-layer, the attention block and the FFN block separately
(HC: arXiv:2409.19606, Algorithm 3; mHC: arXiv:2512.24880, Eq. 3), unlike ``wrapper.py``, whose static baseline wraps a
whole decoder layer. Everything else is the dense OLMo-2 of the scaling study (same embeddings, attention, MLP, the
post-sub-layer RMSNorms, QK-norm, final norm, tied head), so a branch here is ``post_attention_layernorm(self_attn(u))``
or ``post_feedforward_layernorm(mlp(u))`` and the connection module adds it back to the residual streams.

HC ("hc_paper"): the paper releases no code; we use lucidrains' ``hyper-connections`` (vendored verbatim in
``hyper_connections.py``), the most prominent community implementation, with the paper's recommended DHCx4 setting:
dynamic routing on, expansion rate n = 4, tanh, dynamic scales initialized to 0.01 and the Eq. 14 static initialization
(all as vendored). The one change is the normalization before the dynamic projections, which Algorithm 2 specifies as
LayerNorm (lucidrains substitutes RMSNorm). Streams are summed before the final norm (Algorithm 1).

mHC ("mhc"): the mixing math is DeepSeek's official PyTorch reference (``mhc_ref.py``, from deepseek-ai/TileKernels): an
RMS-normed projection of the flattened n*C stream to 2n + n^2 mixes, scaled by alpha^pre, alpha^post, alpha^res and
shifted by the static bias b; H^pre = sigmoid + pre_eps, H^post = 2 sigmoid, H^res = Sinkhorn-Knopp; the update
x_{l+1} = H^res-mix(x_l) + H^post f(H^pre-mix(x_l)); and a sigmoid-gated mix of the streams before the final norm.
Paper settings (Table 5): n = 4, gating factors alpha initialized to 0.01, Sinkhorn t_max = 20. Constants from the
official tests: pre_eps 1e-2, Sinkhorn eps 1e-6, norm eps 1e-6, post multiplier 2. Neither the paper nor the official
code fixes the initial static biases or projections; we follow lucidrains' mHC (the most prominent community
implementation of the dynamic variant): projections fn = 0, b^pre = e_{k mod n}, b^post = 1, b^res = I (as logits,
k = sub-layer index), head-mix bias 0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.olmo2.modeling_olmo2 import Olmo2Model

try:
    from .hyper_connections import get_init_and_expand_reduce_stream_functions
    from . import mhc_ref as M
except ImportError:  # run directly as a script
    from hyper_connections import get_init_and_expand_reduce_stream_functions
    import mhc_ref as M


class MHCConnection(nn.Module):
    """One mHC module around one sub-layer, in the official reference's parameterization."""

    def __init__(self, dim: int, n: int, layer_index: int, alpha_init: float = 0.01, sinkhorn_iters: int = 20,
                 pre_eps: float = 1e-2, sinkhorn_eps: float = 1e-6, norm_eps: float = 1e-6, post_mult: float = 2.0):
        super().__init__()
        self.n, self.sinkhorn_iters = n, sinkhorn_iters
        self.pre_eps, self.sinkhorn_eps, self.norm_eps, self.post_mult = pre_eps, sinkhorn_eps, norm_eps, post_mult
        self.fn = nn.Parameter(torch.zeros(2 * n + n * n, n * dim))            # phi^pre, phi^post, phi^res (dynamic)
        self.scale = nn.Parameter(torch.full((3,), float(alpha_init)))         # alpha^pre, alpha^post, alpha^res
        base = torch.zeros(2 * n + n * n)                                      # b^pre, b^post, b^res (static)
        base[layer_index % n] = 1.0
        base[n:2 * n] = 1.0
        base[2 * n:] = torch.eye(n).flatten()
        self.base = nn.Parameter(base)

    def width(self, x: torch.Tensor):
        """x: [B, T, n, C] streams -> (branch input [B, T, C], H^post, H^res)."""
        mixes = M.mhc_pre_norm_fn_ref(x, self.fn.float(), None, self.norm_eps)
        pre, post, comb = M.mhc_pre_split_mixes_ref(mixes, self.scale.float(), self.base.float(), self.n,
                                                    self.post_mult, self.pre_eps)
        comb = M.sinkhorn_normalize_ref(comb, repeat=self.sinkhorn_iters, eps=self.sinkhorn_eps)
        return M.mhc_pre_apply_mix_ref(x.float(), pre, out_dtype=x.dtype), post, comb

    def depth(self, branch_out: torch.Tensor, x: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return M.mhc_post_ref(branch_out, x, post, comb, out_dtype=x.dtype)


class MHCHead(nn.Module):
    """The sigmoid-gated mix of the n streams into one before the final norm (official mhc_head_compute_mix_ref)."""

    def __init__(self, dim: int, n: int, alpha_init: float = 0.01, pre_eps: float = 1e-2, norm_eps: float = 1e-6):
        super().__init__()
        self.pre_eps, self.norm_eps = pre_eps, norm_eps
        self.fn = nn.Parameter(torch.zeros(n, n * dim))
        self.scale = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.base = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mix = M.mhc_pre_norm_fn_ref(x, self.fn.float(), None, self.norm_eps)
        w = M.mhc_head_compute_mix_ref(mix, self.scale.float(), self.base.float(), self.pre_eps)
        return M.mhc_pre_apply_mix_ref(x.float(), w.unsqueeze(-1), out_dtype=x.dtype)


class PaperHCOlmo2Model(Olmo2Model):
    """OLMo-2 decoder with paper-exact HC (mode='hc_paper') or mHC (mode='mhc'), one module per sub-layer."""

    def __init__(self, config: Olmo2Config, mode: str = "hc_paper", num_streams: int = 4):
        super().__init__(config)
        assert mode in ("hc_paper", "mhc"), mode
        self.mode, self.n = mode, int(num_streams)
        n_sub = 2 * config.num_hidden_layers
        if mode == "hc_paper":
            init_hc, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(
                self.n, dim=config.hidden_size, disable=False)
            conns = [init_hc(layer_index=k) for k in range(n_sub)]
            for c in conns:   # Algorithm 2 normalizes with LayerNorm before the dynamic projections
                c.norm = nn.LayerNorm(config.hidden_size)
            self.connections = nn.ModuleList(conns)
        else:
            self.connections = nn.ModuleList([MHCConnection(config.hidden_size, self.n, k) for k in range(n_sub)])
            self.mhc_head = MHCHead(config.hidden_size, self.n)
        self.post_init()
        self._reset_connection_params()

    def _reset_connection_params(self) -> None:
        """post_init() may re-initialize LayerNorm/Linear-like modules; restore the papers' connection init."""
        for k, c in enumerate(self.connections):
            if self.mode == "hc_paper":
                nn.init.ones_(c.norm.weight); nn.init.zeros_(c.norm.bias)
            else:
                with torch.no_grad():
                    c.fn.zero_(); c.scale.fill_(0.01)
                    base = torch.zeros_like(c.base); base[k % self.n] = 1.0; base[self.n:2 * self.n] = 1.0
                    base[2 * self.n:] = torch.eye(self.n, device=base.device).flatten(); c.base.copy_(base)
        if self.mode == "mhc":
            with torch.no_grad():
                self.mhc_head.fn.zero_(); self.mhc_head.scale.fill_(0.01); self.mhc_head.base.zero_()

    def _sublayer(self, conn, branch, hidden):
        if self.mode == "hc_paper":
            branch_input, add_residual = conn(hidden)
            return add_residual(branch(branch_input))
        u, post, comb = conn.width(hidden)
        return conn.depth(branch(u), hidden, post, comb)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, cache_position: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None, **kwargs) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past, past + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask,
                                         cache_position=cache_position, past_key_values=past_key_values,
                                         position_ids=position_ids)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        if self.mode == "hc_paper":
            hidden = self.expand_stream(inputs_embeds)            # [B*n, T, C], streams folded into the batch
        else:
            hidden = M.expand_to_mhc_ref(inputs_embeds, self.n)   # [B, T, n, C]

        for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            def attn_branch(u, layer=layer):
                h, _ = layer.self_attn(hidden_states=u, attention_mask=causal_mask, position_ids=position_ids,
                                       past_key_values=past_key_values, use_cache=use_cache,
                                       cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
                return layer.post_attention_layernorm(h)

            def mlp_branch(u, layer=layer):
                return layer.post_feedforward_layernorm(layer.mlp(u))

            hidden = self._sublayer(self.connections[2 * i], attn_branch, hidden)
            hidden = self._sublayer(self.connections[2 * i + 1], mlp_branch, hidden)

        hidden = self.reduce_stream(hidden) if self.mode == "hc_paper" else self.mhc_head(hidden)
        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


class PaperHCOlmo2ForCausalLM(Olmo2ForCausalLM):
    def __init__(self, config: Olmo2Config, mode: str = "hc_paper", num_streams: int = 4):
        super().__init__(config)
        self.model = PaperHCOlmo2Model(config, mode=mode, num_streams=num_streams)
        self.post_init()
        self.model._reset_connection_params()


def _olmo2_config(cfg) -> Olmo2Config:
    return Olmo2Config(hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
                       num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_attention_heads,
                       intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
                       tie_word_embeddings=getattr(cfg, "tie_word_embeddings", True),
                       max_position_embeddings=cfg.max_position_embeddings)


def build_hc_paper(cfg) -> PaperHCOlmo2ForCausalLM:
    return PaperHCOlmo2ForCausalLM(_olmo2_config(cfg), mode="hc_paper", num_streams=4)


def build_mhc(cfg) -> PaperHCOlmo2ForCausalLM:
    return PaperHCOlmo2ForCausalLM(_olmo2_config(cfg), mode="mhc", num_streams=4)
