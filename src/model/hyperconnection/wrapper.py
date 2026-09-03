"""HF-CausalLM-compatible Hyper-Connections baseline for the scaling study.

Hyper-Connections (ByteDance Seed, arXiv:2409.19606) replace the single
residual stream of a transformer with ``n`` parallel residual streams.
Learnable *width*-connections mix the ``n`` streams into each layer's input,
and learnable *depth*-connections scatter each layer's output back across the
``n`` streams. At initialisation the routing reduces to the ordinary
single-residual transformer (one stream carries the identity, the dynamic
part is zero), so training is as stable as the dense baseline.

Fairness: this is built on the SAME base as our dense baseline — the HF
``Olmo2`` decoder (identical hidden/layers/heads/vocab/tokenizer, POST-norm,
``q_norm``/``k_norm``). We subclass ``Olmo2Model`` and inject one
``HyperConnections`` module around each decoder layer's residual; every other
weight (attention, MLP, norms, embeddings, lm_head) is byte-for-byte the dense
Olmo2. Hyper-connections add only a handful of tiny alpha/beta parameters per
layer, so the parameter count matches the dense baseline to <0.1%.

Variants (config knobs):
  * ``n`` = ``num_residual_streams`` (expansion rate, default 4).
  * ``dynamic`` (default False) — static-learnable variant. When False, the
    per-token dynamic routing parameters are frozen at their zero init so only
    the static alpha/beta matrices learn (simpler + more stable, per the paper
    for small scales). Set True to enable the full dynamic (token-conditioned)
    routing.

``build_hyperconnection(config)`` mirrors ``build_muddformer(config)``: it
returns an object whose ``forward(input_ids=...)`` yields ``.logits`` of shape
``[B, T, vocab]``, ready to drop into ``scripts/pretrain_baseline.py``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.olmo2.modeling_olmo2 import Olmo2Model

try:
    from .hyper_connections import get_init_and_expand_reduce_stream_functions
except ImportError:  # run directly as a script (python src/model/hyperconnection/wrapper.py)
    from hyper_connections import get_init_and_expand_reduce_stream_functions


class HyperConnectionOlmo2Model(Olmo2Model):
    """Olmo2 decoder stack with ``n`` parallel residual streams.

    Identical to :class:`~transformers.models.olmo2.modeling_olmo2.Olmo2Model`
    except that the residual stream is expanded to ``num_residual_streams``
    lanes on entry, each decoder layer is wrapped by a per-layer
    :class:`HyperConnections` module, and the lanes are summed back to a single
    stream before the final norm.
    """

    def __init__(
        self,
        config: Olmo2Config,
        num_residual_streams: int = 4,
        dynamic: bool = False,
    ) -> None:
        super().__init__(config)
        self.num_residual_streams = int(num_residual_streams)
        self.dynamic = bool(dynamic)

        # Factory returns (init_hyper_conn, expand_stream, reduce_stream).
        # expand/reduce operate on the (b s) stream-batched layout.
        init_hyper_conn, self.expand_stream, self.reduce_stream = (
            get_init_and_expand_reduce_stream_functions(
                self.num_residual_streams,
                dim=config.hidden_size,
                disable=False,
            )
        )

        # One hyper-connection per decoder layer. layer_index picks which stream
        # carries the identity at init (round-robin), matching the paper.
        self.hyper_connections = nn.ModuleList(
            [
                init_hyper_conn(layer_index=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )

        if not self.dynamic:
            self._freeze_dynamic_routing()

        # Re-run HF weight init over the newly added submodules so the
        # hyper-connection buffers land in a defined state (their own __init__
        # already zero-inits the dynamic parts; this is a no-op safety net).
        self.post_init()

    def _freeze_dynamic_routing(self) -> None:
        """Static-learnable variant: freeze per-token dynamic routing.

        The vendored ``HyperConnections`` zero-inits every dynamic parameter
        (``dynamic_alpha_fn``, ``dynamic_beta_fn`` are zeros; the ``*_scale``
        gains multiply them). Freezing them keeps the routing purely static —
        only ``static_alpha`` / ``static_beta`` train. This leaves the
        remaining alpha/beta math untouched, so the width/depth connections are
        the standard Hyper-Connections operations, just token-independent.
        """
        for hc in self.hyper_connections:
            for pname in (
                "dynamic_alpha_fn",
                "dynamic_alpha_scale",
                "dynamic_beta_fn",
                "dynamic_beta_scale",
            ):
                if hasattr(hc, pname):
                    getattr(hc, pname).requires_grad_(False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
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

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Expand single residual → n stream-batched lanes: [B, T, d] -> [B*n, T, d]
        hidden_states = self.expand_stream(inputs_embeds)

        for decoder_layer, hyper_conn in zip(
            self.layers[: self.config.num_hidden_layers], self.hyper_connections
        ):
            # width_connection: mix n lanes -> single branch input [B, T, d]
            branch_input, add_residual = hyper_conn(hidden_states)

            layer_out = decoder_layer(
                branch_input,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # HF decoder layers may return a tuple; take the hidden states.
            if isinstance(layer_out, tuple):
                layer_out = layer_out[0]

            # CRITICAL: Hyper-Connections expects the branch to return ONLY the
            # block transformation f(x) = attn(x) + mlp(x). Its depth_connection
            # re-adds the residual across the n streams. But an HF Olmo2 decoder
            # layer returns the *full* residual `x + attn + mlp` (it applies its
            # own `residual = hidden_states; hidden_states = residual + ...`
            # internally). Passing that straight to add_residual() double-counts
            # the input residual `x`: once carried in the streams, once inside
            # layer_out. Over 16 layers in bf16 the activations blow up to
            # Inf/NaN, which desynchronises DDP ranks and manifests as a NCCL
            # watchdog hang. Strip the layer's own input residual so the branch
            # contributes only the delta — this restores exact identity-at-init
            # (HC == dense Olmo2 within fp roundoff) and matches the standard
            # single-residual transformer when the routing is at its init value.
            branch_delta = layer_out - branch_input

            # depth_connection: scatter branch delta back across n lanes.
            hidden_states = add_residual(branch_delta)

        # Sum the n lanes back to a single residual: [B*n, T, d] -> [B, T, d]
        hidden_states = self.reduce_stream(hidden_states)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HyperConnectionOlmo2ForCausalLM(Olmo2ForCausalLM):
    """Olmo2 CausalLM whose backbone uses ``n`` hyper-connection streams."""

    def __init__(
        self,
        config: Olmo2Config,
        num_residual_streams: int = 4,
        dynamic: bool = False,
    ) -> None:
        # Olmo2ForCausalLM.__init__ builds self.model = Olmo2Model(config); we
        # swap in the hyper-connection backbone afterwards (weights re-init'd
        # inside its own post_init, then tied by our post_init below).
        super().__init__(config)
        self.model = HyperConnectionOlmo2Model(
            config,
            num_residual_streams=num_residual_streams,
            dynamic=dynamic,
        )
        # Re-tie lm_head <-> embeddings after swapping the backbone.
        self.post_init()


def build_hyperconnection(cfg) -> HyperConnectionOlmo2ForCausalLM:
    """Build a Hyper-Connections Olmo2 CausalLM from a ``PretrainConfig``.

    Mirrors :func:`src.model.muddformer.wrapper.build_muddformer`: consumes the
    standard scaling-study fields (``hidden_size``, ``num_hidden_layers``,
    ``num_attention_heads``, ``intermediate_size``, ``vocab_size``,
    ``max_position_embeddings``, ``tie_word_embeddings``) and returns an
    HF-CausalLM-compatible model.

    Two optional config fields control the hyper-connection topology; both fall
    back to sensible defaults if absent from the YAML:
      * ``hc_num_streams`` (int, default 4) — expansion rate ``n``.
      * ``hc_dynamic`` (bool, default False) — static-learnable (False) vs
        token-conditioned dynamic (True) routing.
    """
    model_config = Olmo2Config(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_attention_heads,  # MHA (not GQA), matches baseline
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=getattr(cfg, "tie_word_embeddings", True),
        max_position_embeddings=cfg.max_position_embeddings,
    )
    num_streams = int(getattr(cfg, "hc_num_streams", 4))
    dynamic = bool(getattr(cfg, "hc_dynamic", False))
    return HyperConnectionOlmo2ForCausalLM(
        model_config,
        num_residual_streams=num_streams,
        dynamic=dynamic,
    )


def _count_params(model: nn.Module) -> tuple[int, int]:
    """Return (unique, total) parameter counts, de-duplicating tied weights."""
    seen: set[int] = set()
    unique = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique += p.numel()
    return unique, total


def _identity_at_init_max_abs_diff(cfg, num_streams: int = 4) -> float:
    """Return ``max|HC_logits - dense_logits|`` for byte-identical weights.

    Builds a Hyper-Connections Olmo2 and a plain :class:`Olmo2ForCausalLM` at
    the given ``cfg`` dims, copies every shared Olmo2 weight from the HC model
    into the dense model (verified by ``strict=True`` load over matched names),
    and compares logits on the same input. Hyper-Connections is designed so
    that at initialisation the routing reduces to the ordinary single-residual
    transformer; this returns the max absolute logit discrepancy, which must be
    ~0 (fp roundoff) if the residual wiring is correct. A large value means the
    branch/residual layout is wrong (e.g. the input residual is double-counted).
    """
    from transformers import Olmo2Config as _Olmo2Config
    from transformers import Olmo2ForCausalLM as _Olmo2ForCausalLM

    model_config = _Olmo2Config(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=getattr(cfg, "tie_word_embeddings", True),
        max_position_embeddings=cfg.max_position_embeddings,
    )
    hc = HyperConnectionOlmo2ForCausalLM(
        model_config, num_residual_streams=num_streams, dynamic=False
    ).eval()
    dense = _Olmo2ForCausalLM(model_config).eval()

    hc_sd = hc.state_dict()
    dense_sd = dense.state_dict()
    merged = {
        k: (hc_sd[k].clone() if k in hc_sd and hc_sd[k].shape == dense_sd[k].shape
            else dense_sd[k])
        for k in dense_sd
    }
    dense.load_state_dict(merged, strict=True)

    ids = torch.randint(0, cfg.vocab_size, (2, 64), dtype=torch.long)
    with torch.no_grad():
        a = hc(input_ids=ids).logits
        b = dense(input_ids=ids).logits
    return (a - b).abs().max().item()


if __name__ == "__main__":
    # CPU-only validation that REPRODUCES the real training forward. The prior
    # sanity test used tiny dims / seq 16 / float32 and missed a residual-wiring
    # bug that only surfaced at scale (bf16 + seq 1024 + DDP → NCCL hang). This
    # version builds the model at the actual 75M config from the training YAML,
    # casts to bfloat16, and runs a forward at the real training seq_len (1024),
    # so it exercises the exact code path (and shapes) that crashed. No HF
    # dataset, no pretrained download — random-init model.
    import os
    import sys

    _REPO_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from scripts.pretrain_baseline import PretrainConfig  # noqa: E402

    _CFG_PATH = os.path.join(
        _REPO_ROOT, "configs", "pretrain_75m_hyperconnection.yaml"
    )

    torch.manual_seed(0)
    cfg = PretrainConfig.from_yaml(_CFG_PATH)
    num_streams = 4  # build_hyperconnection default (hc_num_streams not a YAML field)

    print("=" * 70)
    print("Hyper-Connections wrapper validation (real 75M config, bf16, seq 1024)")
    print("=" * 70)
    print(
        f"config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"heads={cfg.num_attention_heads}, vocab={cfg.vocab_size}, "
        f"streams={num_streams}, seq_len={cfg.seq_len}"
    )

    # ---- Check 1: identity-at-init (float32, small forward). This is the check
    # that would have caught the residual double-counting bug. ----
    id_diff = _identity_at_init_max_abs_diff(cfg, num_streams=num_streams)
    print(f"[1] identity-at-init  max|HC - dense| = {id_diff:.3e}")
    assert id_diff < 1e-2, (
        f"IDENTITY-AT-INIT FAILED (max|HC-dense|={id_diff:.4f}). At init the "
        f"Hyper-Connections routing must reduce to the dense Olmo2 forward. A "
        f"large diff means the branch/residual layout is wrong (e.g. the decoder "
        f"layer's own input residual is being double-counted)."
    )

    # ---- Check 2: the real training forward — 75M dims, bf16, seq 1024. ----
    model = build_hyperconnection(cfg)
    model = model.to(torch.bfloat16)
    model.eval()

    unique, total = _count_params(model)
    n_hc = sum(
        p.numel()
        for hc in model.model.hyper_connections
        for p in hc.parameters()
    )
    print(f"    model: {unique:,} unique params ({total:,} total)")
    print(f"    hyper-connection params (all {len(model.model.hyper_connections)} layers): {n_hc:,}")

    input_ids = torch.randint(0, cfg.vocab_size, (2, 1024), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits
    print(f"[2] bf16 forward: input_ids {tuple(input_ids.shape)} -> logits {tuple(logits.shape)} ({logits.dtype})")
    assert logits.shape == (2, 1024, cfg.vocab_size), (
        f"expected logits shape (2, 1024, {cfg.vocab_size}), got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), (
        "logits contain non-finite values (NaN/Inf) — activation blow-up. This "
        "is the failure mode that desynchronised DDP ranks in real training."
    )
    print(f"    logits finite: True | abs-max: {logits.abs().max().item():.3f}")

    print("=" * 70)
    print("VALIDATION PASSED: identity-at-init OK; bf16 seq-1024 forward shape + finiteness OK.")
    print("=" * 70)
