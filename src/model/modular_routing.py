"""Module-granular routing: the structure predictor sees every inter-module edge.

Motivation
----------
``FourWayDAGFormer`` routes each head's Q/K/V (and the shared residual R) over
*layer outputs* X_0..X_l. Inside a layer the wiring is fixed:

    X_{l+1} = R_l + a_l + m_l        a_l = attention block output
    mlp_in  = R_l + a_l              m_l = MLP block output

so the attention->MLP edge, the "module -> everything downstream" edges, and
the final read-out are all hard-wired at weight 1. The predictor can only add
*extra* hyperconnections; it cannot express "this module is not needed here",
and the routing matrix therefore cannot be used to prune modules.

This file replaces the source set with the *module outputs themselves*:

    sources for layer l:   S_l = [emb, a_0, m_0, a_1, m_1, ..., a_{l-1}, m_{l-1}]

and routes every read in the network over them:

    stream   reader                       sources          shape
    q,k,v    head h of layer l (l>=1)     S_l              [B, T, H, 2l+1]
    r        residual carrier of layer l  S_l              [B, T, 2l+1]
    m        MLP input of layer l (l>=0)  S_l + [a_l]      [B, T, 2l+2]
    o        final norm / lm_head         S_L              [B, T, 2L+1]

A standard pre-norm-residual transformer is exactly the all-ones routing
(X_l = emb + sum_{i<l}(a_i + m_i)), so **identity init is alpha = 1 everywhere**
(not one-hot on the last source as in FourWay). Every hard-wired edge of the
vanilla model -- including attention->MLP within a layer -- is now one entry
of alpha, on the same footing as a skip connection over ten layers.

What this buys for pruning
--------------------------
A module's whole downstream influence is one *column* of the routing matrix
(the entries alpha[.., s] over all readers of source s). If the column is
zero, the module can be deleted with no change to the function; column mass
is therefore a routing-native importance score for whole-module pruning
(``source_column_mass``), and a group penalty on columns
(``column_group_penalty``) lets the predictor learn to switch modules off.

Cost: the fused QKV projection runs over 2l+1 instead of l+1 sources
(~1.8x FourWay's routing overhead at L=12); the predictor emits
(3H+1)(2l+1) + (2l+2) values per routed layer plus 2L+1 for the read-out.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

MODULAR_STREAMS = ("q", "k", "v", "r", "m", "o")


def modular_n_sources(layer: int) -> int:
    """Number of module-level sources visible to layer ``layer`` (emb + 2 per prior layer)."""
    return 2 * layer + 1


def modular_source_labels(num_layers: int) -> list[str]:
    """Names of the 2L+1 sources in canonical order: emb, a0, m0, a1, m1, ..."""
    labels = ["emb"]
    for l in range(num_layers):
        labels += [f"a{l}", f"m{l}"]
    return labels


def identity_modular_routing(B: int, T: int, num_layers: int, num_heads: int,
                             device, dtype=torch.float32) -> dict:
    """All-ones routing == vanilla transformer."""
    out: dict = {s: [] for s in MODULAR_STREAMS}
    for l in range(num_layers):
        n = modular_n_sources(l)
        if l >= 1:
            for s in ("q", "k", "v"):
                out[s].append(torch.ones(B, T, num_heads, n, device=device, dtype=dtype))
            out["r"].append(torch.ones(B, T, n, device=device, dtype=dtype))
        out["m"].append(torch.ones(B, T, n + 1, device=device, dtype=dtype))
    out["o"] = torch.ones(B, T, modular_n_sources(num_layers), device=device, dtype=dtype)
    return out


# ─── Backbone ────────────────────────────────────────────────────────────────

class FourWayModularDAGFormer(nn.Module):
    """OLMo-2 backbone whose every inter-module edge is a routing weight.

    Args:
        model: base ``Olmo2ForCausalLM`` (weights shared, wrapper adds none unless
            ``use_local_correction`` / ``use_v_norm``).
        use_local_correction: per-layer correction MLPs (read the running
            residual X_l) adding a per-token delta to q/k/v/r/m routing of
            layers l >= 1, zero-init (Approach B of FourWay).
        use_v_norm: post-mix RMSNorm on V (same stabiliser as FourWay).
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        num_heads: int,
        use_local_correction: bool = False,
        correction_hidden: int = 128,
        use_v_norm: bool = False,
    ) -> None:
        super().__init__()
        self.olmo = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model.config.hidden_size
        self.head_dim = self.model_dim // num_heads
        self.rms_norm_eps = model.config.rms_norm_eps
        self.scaling = self.head_dim ** -0.5
        self.use_local_correction = use_local_correction
        self.use_v_norm = use_v_norm

        if use_v_norm:
            self.v_norms = nn.ModuleList([
                nn.RMSNorm(self.model_dim, eps=self.rms_norm_eps) for _ in range(num_layers - 1)
            ])
        if use_local_correction:
            self.correction_mlps = nn.ModuleList()
            for l in range(1, num_layers):
                n = modular_n_sources(l)
                out_dim = 3 * num_heads * n + n + (n + 1)        # q k v | r | m
                mlp = nn.Sequential(
                    nn.LayerNorm(self.model_dim, elementwise_affine=False),
                    nn.Linear(self.model_dim, correction_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(correction_hidden, out_dim, bias=False),
                )
                nn.init.zeros_(mlp[-1].weight)
                self.correction_mlps.append(mlp)

    def get_routing_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.use_local_correction:
            params.extend(self.correction_mlps.parameters())
        if self.use_v_norm:
            params.extend(self.v_norms.parameters())
        return params

    def forward(self, olmo_ids: torch.Tensor, routing_weights: dict) -> torch.Tensor:
        """Routed forward.

        Args:
            olmo_ids: [B, T]
            routing_weights: dict with 'q','k','v','r' (lists for l=1..L-1),
                'm' (list for l=0..L-1) and 'o' (tensor); see module docstring.
        Returns:
            logits [B, T, vocab]
        """
        B, T = olmo_ids.shape
        H, D = self.num_heads, self.model_dim
        rw = routing_weights
        assert len(rw["q"]) == self.num_layers - 1, (len(rw["q"]), self.num_layers)
        assert len(rw["m"]) == self.num_layers, (len(rw["m"]), self.num_layers)

        embedding = self.olmo.model.embed_tokens(olmo_ids)
        position_ids = torch.arange(T, device=olmo_ids.device).unsqueeze(0)
        cos, sin = self.olmo.model.rotary_emb(embedding, position_ids)

        sources: list[torch.Tensor] = [embedding]           # emb, a0, m0, a1, m1, ...
        running = embedding                                  # X_l = emb + sum(a_i + m_i), for corrections

        for l in range(self.num_layers):
            layer = self.olmo.model.layers[l]
            attn = layer.self_attn
            n_src = len(sources)
            assert n_src == modular_n_sources(l), (n_src, l)
            stacked = torch.stack(sources, dim=0)            # [n_src, B, T, D]

            if l == 0:
                q_all = attn.q_norm(attn.q_proj(embedding))
                k_all = attn.k_norm(attn.k_proj(embedding))
                v_all = attn.v_proj(embedding)
                q = q_all.view(B, T, H, self.head_dim).transpose(1, 2)
                k = k_all.view(B, T, H, self.head_dim).transpose(1, 2)
                v = v_all.view(B, T, H, self.head_dim).transpose(1, 2)
                R = embedding
                corr_m = None
            else:
                W_qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
                qkv = F.linear(stacked.reshape(-1, D), W_qkv).view(n_src, B, T, 3, H, self.head_dim)
                q_stack, k_stack, v_stack = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]

                a_q = rw["q"][l - 1].to(q_stack.dtype)          # [B, T, H, n_src]
                a_k = rw["k"][l - 1].to(q_stack.dtype)
                a_v = rw["v"][l - 1].to(q_stack.dtype)
                a_r = rw["r"][l - 1].to(q_stack.dtype)          # [B, T, n_src]
                corr_m = None
                if self.use_local_correction:
                    corr = self.correction_mlps[l - 1](running.float()).to(q_stack.dtype)
                    sz = H * n_src
                    a_q = a_q + corr[:, :, :sz].view(B, T, H, n_src)
                    a_k = a_k + corr[:, :, sz:2 * sz].view(B, T, H, n_src)
                    a_v = a_v + corr[:, :, 2 * sz:3 * sz].view(B, T, H, n_src)
                    a_r = a_r + corr[:, :, 3 * sz:3 * sz + n_src]
                    corr_m = corr[:, :, 3 * sz + n_src:]        # [B, T, n_src + 1]
                assert a_q.shape == (B, T, H, n_src), (a_q.shape, (B, T, H, n_src))
                assert a_r.shape == (B, T, n_src), (a_r.shape, (B, T, n_src))

                q = torch.einsum("lbthd,bthl->bhtd", q_stack, a_q)
                k = torch.einsum("lbthd,bthl->bhtd", k_stack, a_k)
                v = torch.einsum("lbthd,bthl->bhtd", v_stack, a_v)
                q = rearrange(attn.q_norm(rearrange(q, "b h t d -> b t (h d)")), "b t (h d) -> b h t d", h=H)
                k = rearrange(attn.k_norm(rearrange(k, "b h t d -> b t (h d)")), "b t (h d) -> b h t d", h=H)
                if self.use_v_norm:
                    v = rearrange(self.v_norms[l - 1](rearrange(v, "b h t d -> b t (h d)")),
                                  "b t (h d) -> b h t d", h=H)
                R = torch.einsum("lbtd,btl->btd", stacked, a_r)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            attn_values = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                                         is_causal=True, scale=self.scaling)
            attn_proj = attn.o_proj(rearrange(attn_values, "b h t d -> b t (h d)"))
            a_l = layer.post_attention_layernorm(attn_proj)                    # module output

            # MLP input: routed over S_l + [a_l]  (identity: R + a_l)
            a_m = rw["m"][l].to(a_l.dtype)                                      # [B, T, n_src + 1]
            if corr_m is not None:
                a_m = a_m + corr_m
            assert a_m.shape == (B, T, n_src + 1), (a_m.shape, (B, T, n_src + 1))
            stacked_m = torch.cat([stacked, a_l.unsqueeze(0)], dim=0)         # [n_src+1, B, T, D]
            mlp_in = torch.einsum("lbtd,btl->btd", stacked_m, a_m)
            m_l = layer.post_feedforward_layernorm(layer.mlp(mlp_in))          # module output

            sources.extend([a_l, m_l])
            running = running + a_l + m_l

        stacked = torch.stack(sources, dim=0)                                   # [2L+1, B, T, D]
        a_o = rw["o"].to(stacked.dtype)
        assert a_o.shape == (B, T, len(sources)), (a_o.shape, (B, T, len(sources)))
        final = torch.einsum("lbtd,btl->btd", stacked, a_o)
        return self.olmo.lm_head(self.olmo.model.norm(final))


# ─── Column statistics (module importance through routing) ───────────────────

def source_column_mass(rw: dict, num_layers: int, power: int = 2) -> torch.Tensor:
    """Per-source routing mass: for each of the 2L+1 sources, the mean over
    tokens of sum_readers |alpha[.., s]|^power. Returns [2L+1] (float32).

    A zero entry means no reader in the network uses the module's output, so
    the module can be removed without changing the function.
    """
    n_total = modular_n_sources(num_layers)
    device = rw["o"].device
    mass = torch.zeros(n_total, device=device, dtype=torch.float32)

    def add(alpha: torch.Tensor, n_src: int) -> None:
        a = alpha.float().abs().pow(power)
        # sum over all leading dims except the source dim, then mean over tokens
        B, T = a.shape[0], a.shape[1]
        col = a.reshape(B * T, -1, a.shape[-1]).sum(dim=1).mean(dim=0)        # [n_src(+1)]
        mass[:n_src] += col[:n_src]

    for l in range(1, num_layers):
        n = modular_n_sources(l)
        for s in ("q", "k", "v"):
            add(rw[s][l - 1], n)
        add(rw["r"][l - 1], n)
    for l in range(num_layers):
        n = modular_n_sources(l)
        a = rw["m"][l].float().abs().pow(power)
        col = a.reshape(-1, a.shape[-1]).mean(dim=0)                            # [n+1]
        mass[:n] += col[:n]
        mass[n] += col[n]                                                       # a_l read by its own MLP
    a = rw["o"].float().abs().pow(power)
    mass += a.reshape(-1, a.shape[-1]).mean(dim=0)
    return mass


def module_importance_from_mass(mass: torch.Tensor, num_layers: int) -> dict[str, torch.Tensor]:
    """Split a [2L+1] column-mass vector into masker-shaped scores
    {'attn': [L,1], 'mlp': [L,1]} (source 0 = embedding is not prunable)."""
    attn = mass[1::2][:num_layers].reshape(num_layers, 1)
    mlp = mass[2::2][:num_layers].reshape(num_layers, 1)
    return {"attn": attn.clone(), "mlp": mlp.clone()}


def column_group_penalty(rw: dict, num_layers: int, eps: float = 1e-8) -> torch.Tensor:
    """Group-lasso on source columns: sum_s sqrt(mass_s). Zero-inducing on whole
    columns, i.e. it asks the predictor to switch entire modules off rather
    than to thin edges uniformly. Differentiable in alpha."""
    mass = source_column_mass(rw, num_layers, power=2)
    return (mass[1:] + eps).sqrt().sum()          # skip the embedding column


# ─── Predictor ───────────────────────────────────────────────────────────────

class FourWayModularPredictor(nn.Module):
    """Per-token predictor for module-granular routing (all-ones identity init).

    Same encoder/trunk as ``FourWayPredictor``; the per-layer heads emit, for
    l = 1..L-1: [q (H*n), k (H*n), v (H*n), r (n), m (n+1)] with n = 2l+1,
    for l = 0: [m (2)], plus one read-out head 'o' of size 2L+1.
    W = 0 init and bias = 1 make step 0 exactly vanilla OLMo-2.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        max_seq_len: int = 4096,
        num_layers: int = 12,
        num_heads: int = 16,
        hidden_dim: int = 512,
        causal: bool = True,
        dropout: float = 0.0,
        init_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.causal = causal

        self.embed = nn.Embedding(vocab_size, encoder_dim)
        self.pos_embed = nn.Embedding(max_seq_len, encoder_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=encoder_heads, dim_feedforward=encoder_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.trunk = nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, hidden_dim),
                                   nn.GELU(), nn.Dropout(dropout))

        self.layer_heads = nn.ModuleList()
        self.layer_biases = nn.ParameterList()
        self.layer_out_dims: list[int] = []
        for l in range(num_layers):
            n = modular_n_sources(l)
            out_dim = (3 * num_heads * n + n + (n + 1)) if l >= 1 else (n + 1)
            head = nn.Linear(hidden_dim, out_dim, bias=False)
            nn.init.zeros_(head.weight)
            self.layer_heads.append(head)
            self.layer_biases.append(nn.Parameter(torch.full((out_dim,), float(init_value))))
            self.layer_out_dims.append(out_dim)
        n_out = modular_n_sources(num_layers)
        self.out_head = nn.Linear(hidden_dim, n_out, bias=False)
        nn.init.zeros_(self.out_head.weight)
        self.out_bias = nn.Parameter(torch.full((n_out,), float(init_value)))

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.embed(input_ids) + self.pos_embed(torch.arange(T, device=input_ids.device))
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            x = self.encoder(x, mask=mask)
        else:
            x = self.encoder(x)
        return self.trunk(x)

    def forward(self, input_ids: torch.Tensor) -> dict:
        B, T = input_ids.shape
        H = self.num_heads
        x = self.encode(input_ids)
        out: dict = {s: [] for s in ("q", "k", "v", "r", "m")}
        for l in range(self.num_layers):
            n = modular_n_sources(l)
            raw = self.layer_heads[l](x) + self.layer_biases[l]
            if l == 0:
                out["m"].append(raw)                                         # [B, T, 2]
                continue
            sz = H * n
            out["q"].append(raw[:, :, :sz].view(B, T, H, n))
            out["k"].append(raw[:, :, sz:2 * sz].view(B, T, H, n))
            out["v"].append(raw[:, :, 2 * sz:3 * sz].view(B, T, H, n))
            out["r"].append(raw[:, :, 3 * sz:3 * sz + n])
            out["m"].append(raw[:, :, 3 * sz + n:])                         # [B, T, n+1]
        out["o"] = self.out_head(x) + self.out_bias                          # [B, T, 2L+1]
        return out

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())


def build_modular_pair(model_cfg: dict, base: nn.Module, device) -> tuple[FourWayModularDAGFormer, FourWayModularPredictor]:
    """Construct backbone wrapper + predictor from a training config dict."""
    routing_mode = str(model_cfg.get("routing_mode", ""))
    fourway = FourWayModularDAGFormer(
        model=base,
        num_layers=model_cfg["num_hidden_layers"],
        num_heads=model_cfg["num_attention_heads"],
        use_local_correction=routing_mode.endswith("corrected"),
        correction_hidden=model_cfg.get("correction_hidden", 128),
        use_v_norm=model_cfg.get("use_v_norm", False),
    ).to(device)
    predictor = FourWayModularPredictor(
        vocab_size=model_cfg["vocab_size"],
        encoder_dim=model_cfg.get("predictor_encoder_dim", 256),
        encoder_layers=model_cfg.get("predictor_encoder_layers", 2),
        encoder_heads=model_cfg.get("predictor_encoder_heads", 4),
        max_seq_len=model_cfg.get("predictor_max_seq_len", 4096),
        num_layers=model_cfg["num_hidden_layers"],
        num_heads=model_cfg["num_attention_heads"],
        hidden_dim=model_cfg.get("fourway_hidden", 512),
        causal=model_cfg.get("predictor_causal", True),
        dropout=model_cfg.get("predictor_dropout", 0.0),
    ).to(device)
    return fourway, predictor
