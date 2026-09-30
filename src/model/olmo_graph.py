"""Modified OLMo2-1B forward pass with adjacency matrix A injection.

This module implements the core DAGFormer modification: per-head input
assembly controlled by a 256x256 adjacency matrix A. Each head receives
its own input (a gated combination of prior heads' outputs), rather than
the shared residual stream.

Key design decisions:
- Uses proportional attribution for post_attention_layernorm decomposition
  (OLMo2 is post-norm, not pre-norm as CLAUDE.md §2.1 assumes)
- Concatenate→q_norm→split pattern for per-head Q/K normalization
- Weight slices via .view() (not .clone()) for Phase 2 compatibility
- When A=all-ones and input_norm="none", output is identical to vanilla OLMo2
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2RMSNorm,
    apply_rotary_pos_emb,
)


def create_block_upper_triangular_mask(num_nodes: int = 256, heads_per_layer: int = 16) -> torch.Tensor:
    """Create block-upper-triangular mask based on LAYER indices.

    mask[i,j] = 1 iff layer(j) > layer(i), i.e. j//16 > i//16.
    Same-layer and backward connections are 0.
    Do NOT use torch.triu() — it allows same-layer connections.

    Returns:
        mask: [num_nodes, num_nodes] float tensor with 0s and 1s
    """
    layer_idx = torch.arange(num_nodes) // heads_per_layer
    mask = (layer_idx.unsqueeze(1) < layer_idx.unsqueeze(0)).float()  # [256, 256]
    return mask


class InputNormalizer(nn.Module):
    """Normalization methods for gated head output sums (CLAUDE.md §6.1).

    Applied ONLY to the gated_sum component, not the base (embedding + MLPs).
    """

    def __init__(self, method: str, model_dim: int = 2048, num_nodes: int = 256):
        super().__init__()
        self.method = method
        self.model_dim = model_dim

        if method == "none":
            pass
        elif method == "gate_mean":
            pass  # no learnable params
        elif method == "rms_post":
            self.norm = nn.RMSNorm(model_dim)
        elif method == "ln_post":
            self.norm = nn.LayerNorm(model_dim)
        elif method == "rms_pre":
            self.norms = nn.ModuleList([nn.RMSNorm(model_dim) for _ in range(num_nodes)])
        else:
            raise ValueError(f"Unknown input_norm method: {method}")

    def forward(
        self,
        gated_sum: torch.Tensor,
        A_slice: Optional[torch.Tensor] = None,
        prior_head_outs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize the gated sum of prior head outputs.

        Args:
            gated_sum: [batch, num_heads, seq, model_dim] — gated sum for this layer's heads
            A_slice: [batch, num_prior_nodes, num_heads] — gate values (for gate_mean)
            prior_head_outs: [batch, num_prior_nodes, seq, model_dim] — for rms_pre
        Returns:
            Normalized gated_sum, same shape
        """
        if self.method == "none":
            return gated_sum

        elif self.method == "gate_mean":
            assert A_slice is not None
            # Sum of gates per target head: [batch, num_heads]
            gate_sum = A_slice.sum(dim=1)  # [batch, num_heads]
            # Divide gated_sum by gate_sum (avoid div by zero)
            divisor = gate_sum.clamp(min=1e-8)  # [batch, num_heads]
            return gated_sum / divisor[:, :, None, None]  # broadcast over [seq, model_dim]

        elif self.method == "rms_post":
            return self.norm(gated_sum)

        elif self.method == "ln_post":
            return self.norm(gated_sum)

        elif self.method == "rms_pre":
            # Apply per-source-node RMSNorm before gating, then recompute gated sum
            # This requires prior_head_outs and A_slice
            assert prior_head_outs is not None and A_slice is not None
            num_prior = prior_head_outs.shape[1]
            # Normalize each source node's output
            normed_sources = []
            for i in range(num_prior):
                normed_sources.append(self.norms[i](prior_head_outs[:, i]))
            normed_sources = torch.stack(normed_sources, dim=1)  # [B, num_prior, S, D]
            # Recompute gated sum with normed sources
            return torch.einsum('bih,bisd->bhsd', A_slice, normed_sources)

        raise ValueError(f"Unknown method: {self.method}")


class DAGFormerOLMo(nn.Module):
    """Wraps OLMo2-1B with adjacency matrix A injection for per-head routing.

    When A is all-ones and input_norm is "none", this produces output
    identical to vanilla OLMo2-1B (baseline reproduction invariant).
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        input_norm: str = "none",
        num_layers: int = 16,
        num_heads: int = 16,
    ):
        super().__init__()
        self.olmo = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_nodes = num_layers * num_heads
        self.model_dim = model.config.hidden_size
        self.head_dim = self.model_dim // num_heads
        self.rms_norm_eps = model.config.rms_norm_eps

        # Runtime assertions
        assert model.config.num_attention_heads == num_heads, \
            f"Expected {num_heads} attention heads, got {model.config.num_attention_heads}"
        assert model.config.num_key_value_heads == num_heads, \
            f"Expected MHA ({num_heads} KV heads), got {model.config.num_key_value_heads} — GQA detected"

        # Verify no bias
        layer0_attn = model.model.layers[0].self_attn
        assert layer0_attn.o_proj.bias is None, \
            "Expected no bias in o_proj — update per-head splitting if bias exists"

        # Block-upper-triangular mask: [256, 256]
        self.register_buffer('dag_mask', create_block_upper_triangular_mask(self.num_nodes, num_heads))

        # Input normalization
        self.input_normalizer = InputNormalizer(input_norm, self.model_dim, self.num_nodes)

        # Attention scaling factor
        self.scaling = self.head_dim ** -0.5

    def _get_head_weight_views(self, layer_idx: int) -> dict:
        """Get per-head weight views for a given layer.

        Uses .view() which returns views of the same storage — no copy,
        gradients flow through for Phase 2 compatibility.
        """
        layer = self.olmo.model.layers[layer_idx]
        attn = layer.self_attn

        # Q, K, V projections: [model_dim, model_dim] → [num_heads, head_dim, model_dim]
        W_q = attn.q_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_k = attn.k_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_v = attn.v_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)

        # O projection: [model_dim, model_dim]
        # Split by INPUT dimension (columns): [model_dim, num_heads, head_dim]
        # Permute to [num_heads, model_dim, head_dim] for einsum
        W_o = attn.o_proj.weight.view(self.model_dim, self.num_heads, self.head_dim)
        W_o = W_o.permute(1, 0, 2)  # [num_heads, model_dim, head_dim]

        return {
            'W_q': W_q, 'W_k': W_k, 'W_v': W_v, 'W_o': W_o,
            'q_norm': attn.q_norm,
            'k_norm': attn.k_norm,
            'post_attn_norm': layer.post_attention_layernorm,
            'post_ff_norm': layer.post_feedforward_layernorm,
            'mlp': layer.mlp,
        }

    def forward(
        self,
        olmo_ids: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Modified OLMo2-1B forward pass with per-head routing via A.

        Args:
            olmo_ids: [batch, seq_len] — tokenized by OLMo's tokenizer
            A: [batch, N, N] (per-window) or [batch, seq_len, N, N] (per-token)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = olmo_ids.shape
        device = olmo_ids.device

        per_token_A = (A.dim() == 4)
        if per_token_A:
            assert A.shape == (batch, seq_len, self.num_nodes, self.num_nodes), \
                f"Per-token A shape: expected ({batch}, {seq_len}, {self.num_nodes}, {self.num_nodes}), got {A.shape}"
        else:
            assert A.shape == (batch, self.num_nodes, self.num_nodes), \
                f"A shape mismatch: expected ({batch}, {self.num_nodes}, {self.num_nodes}), got {A.shape}"

        # Cast A to model dtype (predictor outputs float32, OLMo uses bfloat16)
        model_dtype = self.olmo.model.embed_tokens.weight.dtype
        A = A.to(dtype=model_dtype)

        # Token embedding
        embedding = self.olmo.model.embed_tokens(olmo_ids)  # [B, S, D]

        # Position embeddings (computed once, shared across all layers)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]
        position_embeddings = self.olmo.model.rotary_emb(embedding, position_ids)
        cos, sin = position_embeddings

        # Causal attention mask: [1, 1, S, S]
        causal_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=embedding.dtype)
        causal_mask.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            float('-inf'),
        )

        # Storage for outputs across layers
        # We accumulate head_outputs as a list of [B, 16, S, D] tensors (one per layer)
        all_head_outputs: list[torch.Tensor] = []  # each: [B, 16, S, D]
        mlp_outputs: list[torch.Tensor] = []  # each: [B, S, D]

        # Running base: embedding + accumulated MLP outputs (for per-head assembly)
        base = embedding.clone()  # [B, S, D]
        # Accumulated ungated attention outputs (for MLP input)
        attn_accumulated = torch.zeros_like(embedding)  # [B, S, D]

        for l in range(self.num_layers):
            weights = self._get_head_weight_views(l)

            # === ASSEMBLE PER-HEAD INPUTS ===
            if l == 0:
                # Layer 0: all heads see only the embedding (no prior heads or MLPs)
                assembled = embedding.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                # assembled: [B, 16, S, D]
            else:
                # base_l = embedding + Σ_{l'<l} mlp_outputs[l']
                # (base is updated incrementally after each layer's MLP)

                # Stack all prior head outputs: [B, l*16, S, D]
                prior_head_outs = torch.cat(all_head_outputs, dim=1)

                # Slice A and compute gated sum
                src = l * self.num_heads
                tgt_start = l * self.num_heads
                tgt_end = (l + 1) * self.num_heads

                if per_token_A:
                    # A: [B, T, N, N] → A_slice: [B, T, src, 16]
                    A_slice = A[:, :, :src, tgt_start:tgt_end]
                    # prior_head_outs: [B, src, T, D]
                    # Want: result[b, h, t, d] = Σ_i A_slice[b, t, i, h] * prior[b, i, t, d]
                    gated_sum = torch.einsum('btih, bitd -> bhtd', A_slice, prior_head_outs)
                else:
                    # A: [B, N, N] → A_slice: [B, src, 16]
                    A_slice = A[:, :src, tgt_start:tgt_end]
                    gated_sum = torch.einsum('bih,bisd->bhsd', A_slice, prior_head_outs)
                # gated_sum: [B, 16, S, D]

                # Apply input normalization (only to gated_sum, not base)
                if self.input_normalizer.method == "rms_pre":
                    gated_sum = self.input_normalizer(
                        gated_sum, A_slice=A_slice if not per_token_A else None,
                        prior_head_outs=prior_head_outs if not per_token_A else None
                    )
                elif self.input_normalizer.method == "gate_mean":
                    gated_sum = self.input_normalizer(
                        gated_sum, A_slice=A_slice if not per_token_A else None
                    )
                else:
                    gated_sum = self.input_normalizer(gated_sum)

                # assembled = base + gated_sum
                assembled = base.unsqueeze(1) + gated_sum  # [B, 16, S, D]

            # === PER-HEAD Q/K/V PROJECTION ===
            W_q, W_k, W_v, W_o = weights['W_q'], weights['W_k'], weights['W_v'], weights['W_o']

            # Per-head projections via einsum
            # assembled: [B, H, S, D], W_q: [H, head_dim, D]
            q_per_head = torch.einsum('bhsd,hod->bhso', assembled, W_q)  # [B, H, S, head_dim]
            k_per_head = torch.einsum('bhsd,hod->bhso', assembled, W_k)
            v_per_head = torch.einsum('bhsd,hod->bhso', assembled, W_v)

            # === Q_NORM / K_NORM ===
            # OLMo2 applies RMSNorm to concatenated Q/K (2048-dim) AFTER projection.
            # Concat all heads → norm → split back.
            # When A=1 (all heads same input), this equals q_norm(q_proj(shared_input)).
            q_concat = rearrange(q_per_head, 'b h s d -> b s (h d)')  # [B, S, 2048]
            q_normed = weights['q_norm'](q_concat)
            q_per_head = rearrange(q_normed, 'b s (h d) -> b h s d', h=self.num_heads)

            k_concat = rearrange(k_per_head, 'b h s d -> b s (h d)')
            k_normed = weights['k_norm'](k_concat)
            k_per_head = rearrange(k_normed, 'b s (h d) -> b h s d', h=self.num_heads)

            # V has NO norm in OLMo2

            # === APPLY RoPE ===
            q_per_head, k_per_head = apply_rotary_pos_emb(q_per_head, k_per_head, cos, sin)

            # === ATTENTION COMPUTATION ===
            # q,k,v: [B, H, S, head_dim]
            attn_weights = torch.matmul(q_per_head, k_per_head.transpose(-2, -1)) * self.scaling
            # attn_weights: [B, H, S, S]
            attn_weights = attn_weights + causal_mask  # [1, 1, S, S] broadcasts
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_per_head.dtype)
            attn_values = torch.matmul(attn_weights, v_per_head)  # [B, H, S, head_dim]

            # === PER-HEAD O_PROJ ===
            # attn_values: [B, H, S, head_dim], W_o: [H, model_dim, head_dim]
            raw_head_outs = torch.einsum('bhsd,hod->bhso', attn_values, W_o)
            # raw_head_outs: [B, H, S, model_dim]

            # === PROPORTIONAL ATTRIBUTION WITH POST_ATTN_NORM ===
            # OLMo2 applies post_attention_layernorm to the COMBINED attention output.
            # RMSNorm(Σ_h x_h) = weight * (Σ_h x_h) / RMS(Σ_h x_h)
            #                   = Σ_h [weight * x_h / RMS(Σ_h x_h)]
            # We attribute each head's normed output proportionally.
            raw_sum = raw_head_outs.sum(dim=1)  # [B, S, D]
            # Compute RMS of the sum
            variance = raw_sum.to(torch.float32).pow(2).mean(-1, keepdim=True)
            rms = torch.sqrt(variance + self.rms_norm_eps)  # [B, S, 1]
            # Apply post_attn_norm weight and scale
            norm_weight = weights['post_attn_norm'].weight  # [D]
            # head_output[h] = norm_weight * raw_head_out[h] / rms
            scale = (norm_weight / rms).unsqueeze(1)  # [B, 1, S, D]
            head_outputs_l = raw_head_outs.float() * scale  # [B, H, S, D]
            head_outputs_l = head_outputs_l.to(raw_head_outs.dtype)

            # Store for routing to later layers
            all_head_outputs.append(head_outputs_l)

            # === MLP COMPUTATION (standard, ungated) ===
            # attn_normed = Σ_h head_output[l,h] = post_attn_norm(raw_sum)
            attn_normed = head_outputs_l.sum(dim=1)  # [B, S, D]

            # MLP input = full residual stream (embedding + all prior MLPs + all attn up to current)
            # In vanilla OLMo2: mlp_input = residual + post_attn_norm(attn_output)
            # where residual includes ALL prior components (embedding + prior MLPs + prior attns)
            mlp_in = base + attn_accumulated + attn_normed

            # Update accumulated attention for next layer
            attn_accumulated = attn_accumulated + attn_normed

            # MLP forward + post_feedforward_layernorm
            mlp_raw = weights['mlp'](mlp_in)
            mlp_output_l = weights['post_ff_norm'](mlp_raw)
            mlp_outputs.append(mlp_output_l)

            # Update running base for next layer
            # base_{l+1} = base_l + mlp_output_l = embedding + Σ_{l'<=l} mlp_output[l']
            base = base + mlp_output_l

        # === FINAL OUTPUT ===
        # final_state = embedding + Σ_l mlp_output[l] + Σ_l Σ_h head_output[l,h]
        # = embedding + Σ_l [post_attn_norm(attn_out_l) + post_ff_norm(mlp_out_l)]
        # 'base' = embedding + Σ_l mlp_output[l]
        # 'attn_accumulated' = Σ_l attn_output[l] (ungated sum of all attention outputs)
        final_state = base + attn_accumulated

        # Apply final norm and lm_head
        final_state = self.olmo.model.norm(final_state)
        logits = self.olmo.lm_head(final_state)

        return logits


def compute_vanilla_nll(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute NLL using vanilla OLMo2 forward pass (no A injection).

    Used for baseline comparison in sanity checks.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        # labels is already shifted (chunk[1:seq_len+1]), no additional shift needed
        nll = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            labels.contiguous().view(-1),
        )
    return nll


def create_all_ones_A(batch_size: int, num_nodes: int = 256, num_heads: int = 16) -> torch.Tensor:
    """Create A matrix with 1.0 for all valid (cross-layer) entries.

    When used with input_norm="none", this should reproduce vanilla OLMo2.
    """
    A = torch.zeros(batch_size, num_nodes, num_nodes)
    mask = create_block_upper_triangular_mask(num_nodes, num_heads)
    A = A + mask.unsqueeze(0)  # broadcast mask to batch
    return A


# ─── Approach 1: Per-head dynamic dense (MUDDFormer at head level) ────────────


class HeadRoutingMLP(nn.Module):
    """Per-layer MLP: hidden state → per-head routing weights (low-rank).

    Computes per-token, per-head routing weights from current hidden state.
    Identity init: W_out=0, bias=1 → starts as standard transformer (all connections ON).

    Args:
        model_dim: hidden size of the base model (e.g. 1024)
        n_targets: number of target heads in this layer (16)
        n_sources: number of source heads from all prior layers (l * 16)
        rank: low-rank factorization rank for UV^T
        hidden_dim: MLP hidden dimension
    """

    def __init__(self, model_dim: int, n_targets: int, n_sources: int,
                 rank: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.n_targets = n_targets
        self.n_sources = n_sources
        self.rank = rank

        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.proj = nn.Linear(model_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.head_u = nn.Linear(hidden_dim, n_targets * rank, bias=False)
        self.head_v = nn.Linear(hidden_dim, n_sources * rank, bias=False)

        # Identity init: W_out=0 → dynamic component outputs 0
        # Only static bias matters at init
        nn.init.zeros_(self.head_u.weight)
        nn.init.zeros_(self.head_v.weight)

        # Static bias: all ones (all connections active at init = standard transformer)
        self.bias = nn.Parameter(torch.ones(n_targets, n_sources))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-token per-head routing weights.

        Args:
            x: [B, T, D] hidden state

        Returns:
            alpha: [B, T, n_targets, n_sources] routing weights
        """
        h = self.act(self.proj(self.norm(x.float())))  # [B, T, hidden_dim]
        B, T, _ = h.shape
        U = self.head_u(h).view(B, T, self.n_targets, self.rank)
        V = self.head_v(h).view(B, T, self.n_sources, self.rank)
        dynamic = torch.einsum('btir, btjr -> btij', U, V)  # [B, T, n_tgt, n_src]
        return dynamic + self.bias  # bias broadcasts over [B, T]


class DynamicDenseHeadFormer(nn.Module):
    """MUDDFormer-style dynamic dense connections at HEAD level.

    Per-token, per-head routing weights computed from hidden state via small MLPs.
    Identity init: starts as standard transformer.
    Reuses DAGFormerOLMo's per-head QKV and attention logic.

    This has ~3.5x overhead vs standard transformer (per-head QKV splitting).
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        num_layers: int = 12,
        num_heads: int = 16,
        routing_rank: int = 16,
        routing_hidden: int = 256,
    ):
        super().__init__()
        self.olmo = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model.config.hidden_size
        self.head_dim = self.model_dim // num_heads
        self.rms_norm_eps = model.config.rms_norm_eps
        self.scaling = self.head_dim ** -0.5

        # Per-layer routing MLPs (layer 0 has no prior heads, so skip it)
        self.routing_mlps = nn.ModuleList()
        for l in range(1, num_layers):
            n_sources = l * num_heads
            self.routing_mlps.append(
                HeadRoutingMLP(self.model_dim, num_heads, n_sources,
                               rank=routing_rank, hidden_dim=routing_hidden)
            )

    def _get_head_weight_views(self, layer_idx: int) -> dict:
        """Same as DAGFormerOLMo._get_head_weight_views."""
        layer = self.olmo.model.layers[layer_idx]
        attn = layer.self_attn
        W_q = attn.q_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_k = attn.k_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_v = attn.v_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_o = attn.o_proj.weight.view(self.model_dim, self.num_heads, self.head_dim)
        W_o = W_o.permute(1, 0, 2)
        return {
            'W_q': W_q, 'W_k': W_k, 'W_v': W_v, 'W_o': W_o,
            'q_norm': attn.q_norm, 'k_norm': attn.k_norm,
            'post_attn_norm': layer.post_attention_layernorm,
            'post_ff_norm': layer.post_feedforward_layernorm,
            'mlp': layer.mlp,
        }

    def forward(self, olmo_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-token, per-head dynamic routing.

        Args:
            olmo_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = olmo_ids.shape
        device = olmo_ids.device

        embedding = self.olmo.model.embed_tokens(olmo_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = self.olmo.model.rotary_emb(embedding, position_ids)

        causal_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=embedding.dtype)
        causal_mask.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            float('-inf'),
        )

        all_head_outputs: list[torch.Tensor] = []  # each [B, H, T, D]
        mlp_outputs: list[torch.Tensor] = []
        base = embedding.clone()
        attn_accumulated = torch.zeros_like(embedding)

        for l in range(self.num_layers):
            weights = self._get_head_weight_views(l)

            # === PER-HEAD INPUT ASSEMBLY ===
            if l == 0:
                assembled = embedding.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            else:
                prior_head_outs = torch.cat(all_head_outputs, dim=1)  # [B, l*16, T, D]

                # Compute per-token routing weights from current hidden state
                alpha = self.routing_mlps[l - 1](base)  # [B, T, 16, l*16]
                alpha = alpha.to(dtype=prior_head_outs.dtype)

                # Per-token per-head gated sum
                # alpha: [B, T, H_tgt, H_src], prior: [B, H_src, T, D]
                gated_sum = torch.einsum('bthj, bjtd -> bhtd', alpha, prior_head_outs)
                # gated_sum: [B, H_tgt, T, D]

                assembled = base.unsqueeze(1) + gated_sum  # [B, H, T, D]

            # === PER-HEAD QKV + ATTENTION (same as DAGFormerOLMo) ===
            W_q, W_k, W_v, W_o = weights['W_q'], weights['W_k'], weights['W_v'], weights['W_o']

            q_per_head = torch.einsum('bhsd,hod->bhso', assembled, W_q)
            k_per_head = torch.einsum('bhsd,hod->bhso', assembled, W_k)
            v_per_head = torch.einsum('bhsd,hod->bhso', assembled, W_v)

            q_concat = rearrange(q_per_head, 'b h s d -> b s (h d)')
            q_normed = weights['q_norm'](q_concat)
            q_per_head = rearrange(q_normed, 'b s (h d) -> b h s d', h=self.num_heads)

            k_concat = rearrange(k_per_head, 'b h s d -> b s (h d)')
            k_normed = weights['k_norm'](k_concat)
            k_per_head = rearrange(k_normed, 'b s (h d) -> b h s d', h=self.num_heads)

            q_per_head, k_per_head = apply_rotary_pos_emb(q_per_head, k_per_head, cos, sin)

            attn_weights = torch.matmul(q_per_head, k_per_head.transpose(-2, -1)) * self.scaling
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_per_head.dtype)
            attn_values = torch.matmul(attn_weights, v_per_head)

            raw_head_outs = torch.einsum('bhsd,hod->bhso', attn_values, W_o)

            # Proportional attribution with post_attn_norm
            raw_sum = raw_head_outs.sum(dim=1)
            variance = raw_sum.to(torch.float32).pow(2).mean(-1, keepdim=True)
            rms = torch.sqrt(variance + self.rms_norm_eps)
            norm_weight = weights['post_attn_norm'].weight
            scale = (norm_weight / rms).unsqueeze(1)
            head_outputs_l = raw_head_outs.float() * scale
            head_outputs_l = head_outputs_l.to(raw_head_outs.dtype)

            all_head_outputs.append(head_outputs_l)

            # === MLP ===
            attn_normed = head_outputs_l.sum(dim=1)
            mlp_in = base + attn_accumulated + attn_normed
            attn_accumulated = attn_accumulated + attn_normed
            mlp_raw = weights['mlp'](mlp_in)
            mlp_output_l = weights['post_ff_norm'](mlp_raw)
            mlp_outputs.append(mlp_output_l)
            base = base + mlp_output_l

        final_state = base + attn_accumulated
        final_state = self.olmo.model.norm(final_state)
        logits = self.olmo.lm_head(final_state)
        return logits

    def get_routing_parameters(self) -> list[nn.Parameter]:
        """Return only the routing MLP parameters (for separate optimizer group)."""
        return list(self.routing_mlps.parameters())


# ─── Approach 2: Layer DWA + output gating ────────────────────────────────────


class FourWayDAGFormer(nn.Module):
    """4-way per-head per-token DAGFormer with soft continuous routing.

    MUDDFormer-style architecture at head level:
    - Q/K/V streams: per-head, per-token weighted sum of prior layer outputs
    - R stream: per-token (shared across heads), used as residual
    - Soft gates: continuous real values, no Gumbel-sigmoid
    - Identity init: starts as standard transformer

    Sources are LAYER outputs (not individual heads), targets are per-HEAD.
    Routing weights come from an external predictor (Approach A) and can
    optionally be refined by local correction MLPs (Approach B).

    Args:
        model: base OLMo model
        num_layers: number of transformer layers
        num_heads: number of attention heads per layer
        use_local_correction: if True, add per-layer correction MLPs (Approach B)
        correction_hidden: hidden dim for correction MLPs
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        num_layers: int = 12,
        num_heads: int = 16,
        use_local_correction: bool = False,
        correction_hidden: int = 128,
        use_triton_kernel: bool = False,
        use_v_norm: bool = False,
        correction_pool: str = "none",
    ):
        super().__init__()
        self.olmo = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model.config.hidden_size
        self.head_dim = self.model_dim // num_heads
        self.rms_norm_eps = model.config.rms_norm_eps
        self.scaling = self.head_dim ** -0.5
        self.use_local_correction = use_local_correction
        self.use_triton_kernel = use_triton_kernel
        self.use_v_norm = use_v_norm
        self.correction_pool = correction_pool  # "none" = per-token, "mean" = seq-avg
        self.max_L = num_layers + 1  # max sources (embedding + all layers)

        # Optional V post-mix RMSNorm (symmetric with Q/K norm).
        # Each layer l in 1..num_layers-1 gets its own v_norm over model_dim.
        # Layer 0 is not routed, so no v_norm there.
        if use_v_norm:
            # Use nn.RMSNorm (not Olmo2RMSNorm) to avoid f32 cast + torch.compile issues
            self.v_norms = nn.ModuleList([
                nn.RMSNorm(self.model_dim, eps=self.rms_norm_eps)
                for _ in range(num_layers - 1)
            ])

        # Optional local correction MLPs (Approach B)
        if use_local_correction:
            self.correction_mlps = nn.ModuleList()
            for l in range(1, num_layers):
                n_sources = l + 1  # embedding + l prior layer outputs
                # Output: 3 * H * n_sources (Q/K/V per-head) + n_sources (R shared)
                out_dim = 3 * num_heads * n_sources + n_sources
                mlp = nn.Sequential(
                    nn.LayerNorm(self.model_dim, elementwise_affine=False),
                    nn.Linear(self.model_dim, correction_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(correction_hidden, out_dim, bias=False),
                )
                # W=0 init: correction starts at zero (pure predictor at init)
                nn.init.zeros_(mlp[-1].weight)
                self.correction_mlps.append(mlp)

    def _get_head_weight_views(self, layer_idx: int) -> dict:
        """Get per-head weight views for a given layer."""
        layer = self.olmo.model.layers[layer_idx]
        attn = layer.self_attn
        W_q = attn.q_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_k = attn.k_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_v = attn.v_proj.weight.view(self.num_heads, self.head_dim, self.model_dim)
        W_o = attn.o_proj.weight.view(self.model_dim, self.num_heads, self.head_dim)
        W_o = W_o.permute(1, 0, 2)
        return {
            'W_q': W_q, 'W_k': W_k, 'W_v': W_v, 'W_o': W_o,
            'q_norm': attn.q_norm, 'k_norm': attn.k_norm,
            'post_attn_norm': layer.post_attention_layernorm,
            'post_ff_norm': layer.post_feedforward_layernorm,
            'mlp': layer.mlp,
        }

    def forward(
        self,
        olmo_ids: torch.Tensor,
        routing_weights: dict[str, list[torch.Tensor]],
        *,
        depth_probe=None,
    ) -> torch.Tensor:
        """4-way per-head forward pass with external routing weights.

        Args:
            olmo_ids: [batch, seq_len]
            routing_weights: dict with keys 'q', 'k', 'v', 'r', each a list
                of tensors (one per layer l=1..num_layers-1):
                - 'q'/'k'/'v': [B, T, H, l+1] — per-head per-source-layer
                - 'r': [B, T, l+1] — shared across heads
            depth_probe: optional evaluation observer/intervention implementing
                record_state(index, state, residual=None), should_skip(layer),
                and mix(layer, q, k, v, r). None preserves the normal forward.

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = olmo_ids.shape
        device = olmo_ids.device
        H = self.num_heads

        embedding = self.olmo.model.embed_tokens(olmo_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = self.olmo.model.rotary_emb(embedding, position_ids)

        causal_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=embedding.dtype)
        causal_mask.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            float('-inf'),
        )

        # Accumulate layer outputs for weighted sum
        layer_outputs: list[torch.Tensor] = [embedding]  # X_0 = embedding
        if depth_probe is not None:
            depth_probe.record_state(0, embedding)

        for l in range(self.num_layers):
            if depth_probe is not None and depth_probe.should_skip(l):
                # Keep source indices fixed. The probe decides whether future
                # readers may use this identity slot or must mask it out.
                layer_outputs.append(layer_outputs[-1])
                depth_probe.record_state(l + 1, layer_outputs[-1], layer_outputs[-2])
                continue
            weights = self._get_head_weight_views(l)
            W_q, W_k, W_v, W_o = weights['W_q'], weights['W_k'], weights['W_v'], weights['W_o']

            if l == 0:
                # Layer 0: all heads see embedding → standard QKV projection
                attn = self.olmo.model.layers[0].self_attn
                q_all = attn.q_proj(embedding)   # [B, T, H*hd]
                k_all = attn.k_proj(embedding)
                v_all = attn.v_proj(embedding)
                q_all = attn.q_norm(q_all)
                k_all = attn.k_norm(k_all)
                q_per_head = q_all.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
                k_per_head = k_all.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
                v_per_head = v_all.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
            else:
                # === OPTIMIZED: project-then-mix (not mix-then-project) ===
                # 1. Run standard QKV projection on each layer output (efficient batched matmul)
                # 2. Mix in head_dim space using routing weights
                # This avoids per-head model_dim projection (the 3.5x bottleneck)
                attn = self.olmo.model.layers[l].self_attn

                # Fused QKV projection: 1 matmul instead of 3*L separate matmuls.
                # Stack all layer outputs → fuse Q/K/V weights → single large matmul.
                # For layer 11 this replaces 36 small matmuls with 1 large one,
                # dramatically improving GPU utilization on bandwidth-bound workloads.
                L_cur = len(layer_outputs)
                stacked = torch.stack(layer_outputs, dim=0)  # [L, B, T, D]
                stacked_flat = stacked.reshape(-1, self.model_dim)  # [L*B*T, D]
                W_qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight,
                                   attn.v_proj.weight], dim=0)  # [3*D, D]
                qkv_flat = F.linear(stacked_flat, W_qkv)  # [L*B*T, 3*D]
                qkv = qkv_flat.view(L_cur, batch, seq_len, 3, H, self.head_dim)
                q_stack = qkv[:, :, :, 0]  # [L, B, T, H, hd]
                k_stack = qkv[:, :, :, 1]
                v_stack = qkv[:, :, :, 2]

                # Get routing weights
                α_q = routing_weights['q'][l - 1].to(dtype=q_stack.dtype)  # [B, T, H, L]
                α_k = routing_weights['k'][l - 1].to(dtype=q_stack.dtype)
                α_v = routing_weights['v'][l - 1].to(dtype=q_stack.dtype)
                α_r = routing_weights['r'][l - 1].to(dtype=q_stack.dtype)  # [B, T, L]

                # Approach B: local correction
                if self.use_local_correction:
                    hidden = layer_outputs[-1]
                    if self.correction_pool == "mean":
                        # Sequence-average pooling: reduces per-token memorization
                        # bandwidth from T×D to just D features. Each token in the
                        # sequence gets the SAME correction delta.
                        hidden = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
                    corr = self.correction_mlps[l - 1](hidden.float()).to(dtype=q_stack.dtype)
                    n_src = l + 1
                    qkv_size = H * n_src
                    cq = corr[:, :, :qkv_size].view(batch, seq_len, H, n_src)
                    ck = corr[:, :, qkv_size:2*qkv_size].view(batch, seq_len, H, n_src)
                    cv = corr[:, :, 2*qkv_size:3*qkv_size].view(batch, seq_len, H, n_src)
                    cr = corr[:, :, 3*qkv_size:]
                    α_q = α_q + cq
                    α_k = α_k + ck
                    α_v = α_v + cv
                    α_r = α_r + cr

                if depth_probe is not None:
                    α_q, α_k, α_v, α_r = depth_probe.mix(l, α_q, α_k, α_v, α_r)

                if self.use_triton_kernel:
                    # Fused Triton kernel: weighted-sum + projection in one pass
                    from src.model.triton_routing import fused_routing_proj

                    stacked = torch.stack(layer_outputs, dim=0)  # [L, B, T, D]
                    L_cur = stacked.shape[0]

                    # Pad to max_L
                    if L_cur < self.max_L:
                        pad = torch.zeros(self.max_L - L_cur, batch, seq_len, self.model_dim,
                                          device=device, dtype=stacked.dtype)
                        stacked_pad = torch.cat([stacked, pad], dim=0)
                    else:
                        stacked_pad = stacked

                    # Reshape: [max_L, B, T, D] → [max_L, B*T, D]
                    stacked_flat = stacked_pad.view(self.max_L, batch * seq_len, self.model_dim)

                    # Pad alpha to max_L: [B, T, H, L] → [B*T, H, max_L]
                    def pad_alpha(α, L_cur):
                        if α.shape[-1] < self.max_L:
                            pad = torch.zeros(*α.shape[:-1], self.max_L - α.shape[-1],
                                              device=α.device, dtype=α.dtype)
                            α = torch.cat([α, pad], dim=-1)
                        return α.view(batch * seq_len, -1, self.max_L)

                    α_q_flat = pad_alpha(α_q, L_cur)  # [N, H, max_L]
                    α_k_flat = pad_alpha(α_k, L_cur)
                    α_v_flat = pad_alpha(α_v, L_cur)

                    # Fused routing + projection
                    W_q_full = attn.q_proj.weight.t()  # [D, H*hd]
                    W_k_full = attn.k_proj.weight.t()
                    W_v_full = attn.v_proj.weight.t()

                    q_flat = fused_routing_proj(stacked_flat, α_q_flat, W_q_full)  # [N, H, hd]
                    k_flat = fused_routing_proj(stacked_flat, α_k_flat, W_k_full)
                    v_flat = fused_routing_proj(stacked_flat, α_v_flat, W_v_full)

                    # Reshape: [N, H, hd] → [B, T, H, hd] → [B, H, T, hd]
                    q_per_head = q_flat.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
                    k_per_head = k_flat.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
                    v_per_head = v_flat.view(batch, seq_len, H, self.head_dim).transpose(1, 2)

                    # Q/K norm
                    q_concat = rearrange(q_per_head, 'b h t d -> b t (h d)')
                    q_normed = attn.q_norm(q_concat)
                    q_per_head = rearrange(q_normed, 'b t (h d) -> b h t d', h=H)

                    k_concat = rearrange(k_per_head, 'b h t d -> b t (h d)')
                    k_normed = attn.k_norm(k_concat)
                    k_per_head = rearrange(k_normed, 'b t (h d) -> b h t d', h=H)

                    # R stream
                    α_r_flat = α_r.view(batch * seq_len, L_cur)
                    R = torch.einsum('nl, lnd -> nd', α_r_flat, stacked.view(L_cur, batch * seq_len, self.model_dim))
                    R = R.view(batch, seq_len, self.model_dim)

                else:
                    # Standard project-then-mix with einsum
                    # Mix in head_dim space: [L,B,T,H,hd] × [B,T,H,L] → [B,H,T,hd]
                    q_per_head = torch.einsum('lbthd, bthl -> bhtd', q_stack, α_q)
                    k_per_head = torch.einsum('lbthd, bthl -> bhtd', k_stack, α_k)
                    v_per_head = torch.einsum('lbthd, bthl -> bhtd', v_stack, α_v)

                    # Q/K norm (on mixed result)
                    q_concat = rearrange(q_per_head, 'b h t d -> b t (h d)')
                    q_normed = attn.q_norm(q_concat)
                    q_per_head = rearrange(q_normed, 'b t (h d) -> b h t d', h=H)

                    k_concat = rearrange(k_per_head, 'b h t d -> b t (h d)')
                    k_normed = attn.k_norm(k_concat)
                    k_per_head = rearrange(k_normed, 'b t (h d) -> b h t d', h=H)

                    # V norm (optional, symmetric with Q/K norm).
                    # Without this, V is the only unnormalized post-mix stream,
                    # and can develop large outliers (observed: max ±10 at
                    # step 5000). Since q_norm/k_norm already protect Q/K,
                    # V is the dominant runaway channel.
                    if self.use_v_norm:
                        v_concat = rearrange(v_per_head, 'b h t d -> b t (h d)')
                        v_normed = self.v_norms[l - 1](v_concat)
                        v_per_head = rearrange(v_normed, 'b t (h d) -> b h t d', h=H)

                    # R stream (reuse stacked from fused QKV — no redundant torch.stack)
                    R = torch.einsum('lbtd, btl -> btd', stacked, α_r)

            # RoPE (same for all approaches)
            q_per_head, k_per_head = apply_rotary_pos_emb(q_per_head, k_per_head, cos, sin)

            # Attention: use F.scaled_dot_product_attention with is_causal=True
            # to match HF's native OLMo2 SDPA dispatch. Using attn_mask=None
            # + is_causal=True enables flash attention kernels (same as HF).
            # Our old manual matmul+softmax(dtype=f32) had 7% gradient deviation.
            attn_values = F.scaled_dot_product_attention(
                q_per_head, k_per_head, v_per_head,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scaling,
            )  # [B, H, T, hd]

            # Standard O projection (concat heads → single matmul)
            attn_concat = rearrange(attn_values, 'b h t d -> b t (h d)')  # [B, T, H*hd]
            attn_proj = self.olmo.model.layers[l].self_attn.o_proj(attn_concat)  # [B, T, D]

            # Post-attention norm
            attn_out = weights['post_attn_norm'](attn_proj)  # [B, T, D]

            # MLP
            if l == 0:
                mlp_in = embedding + attn_out
            else:
                mlp_in = R + attn_out
            mlp_raw = weights['mlp'](mlp_in)
            mlp_out = weights['post_ff_norm'](mlp_raw)

            # New layer output = residual + attention + MLP
            if l == 0:
                X_next = embedding + attn_out + mlp_out
            else:
                X_next = R + attn_out + mlp_out
            layer_outputs.append(X_next)
            if depth_probe is not None:
                depth_probe.record_state(l + 1, X_next, embedding if l == 0 else R)

        # Final output
        final_state = self.olmo.model.norm(layer_outputs[-1])
        logits = self.olmo.lm_head(final_state)
        return logits

    def get_routing_parameters(self) -> list[nn.Parameter]:
        """Return all FourWayDAGFormer-owned trainable parameters that are
        NOT in the base OLMo model (correction MLPs + v_norms if enabled)."""
        params: list[nn.Parameter] = []
        if self.use_local_correction:
            params.extend(self.correction_mlps.parameters())
        if self.use_v_norm:
            params.extend(self.v_norms.parameters())
        return params


class LayerDWAGateFormer(nn.Module):
    """Layer-level dynamic weighted averaging + per-head output gating.

    - DWA: per-token dynamic weights aggregate all prior layer outputs (MUDDFormer-style)
    - Head gating: per-token gates on each head's output contribution
    - Uses standard batched MHA (no per-head splitting) → minimal overhead (~15%)
    - Identity init: DWA starts as identity, gates start at 1.0
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        num_layers: int = 12,
        num_heads: int = 16,
        dwa_hidden: int = 256,
    ):
        super().__init__()
        self.olmo = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model.config.hidden_size
        self.head_dim = self.model_dim // num_heads
        self.rms_norm_eps = model.config.rms_norm_eps

        # DWA weight MLPs: one per layer (layer l aggregates from l+1 sources)
        self.dwa_mlps = nn.ModuleList()
        self.dwa_biases = nn.ParameterList()
        for l in range(1, num_layers):  # layer 0 has no DWA
            n_sources = l + 1  # embedding + l prior layer outputs
            mlp = nn.Sequential(
                nn.LayerNorm(self.model_dim, elementwise_affine=False),
                nn.Linear(self.model_dim, dwa_hidden, bias=False),
                nn.GELU(),
                nn.Linear(dwa_hidden, n_sources, bias=False),
            )
            # W_out=0 → dynamic part outputs 0 at init
            nn.init.zeros_(mlp[-1].weight)
            self.dwa_mlps.append(mlp)
            # Static bias: [0, 0, ..., 0, 1] = current layer output only
            bias = torch.zeros(n_sources)
            bias[-1] = 1.0
            self.dwa_biases.append(nn.Parameter(bias))

        # Per-head output gate MLPs: one per layer
        self.gate_mlps = nn.ModuleList()
        for l in range(num_layers):
            gate = nn.Sequential(
                nn.LayerNorm(self.model_dim, elementwise_affine=False),
                nn.Linear(self.model_dim, num_heads, bias=True),
            )
            # Init bias positive → sigmoid(bias) ≈ 1 (all heads active)
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, 5.0)  # sigmoid(5) ≈ 0.993
            self.gate_mlps.append(gate)

    def forward(self, olmo_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with layer DWA + per-head output gating.

        Args:
            olmo_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = olmo_ids.shape
        device = olmo_ids.device

        embedding = self.olmo.model.embed_tokens(olmo_ids)

        # Collect all layer representations for DWA
        layer_outputs = [embedding]  # X_0 = embedding

        x = embedding  # current residual stream

        for l in range(self.num_layers):
            layer_module = self.olmo.model.layers[l]

            # === DWA: aggregate prior layer outputs ===
            if l > 0:
                stacked = torch.stack(layer_outputs, dim=0)  # [L, B, T, D]

                # Dynamic weights from current state (compute in float32)
                dwa_dynamic = self.dwa_mlps[l - 1](x.float())  # [B, T, n_sources]
                dwa_weights = dwa_dynamic + self.dwa_biases[l - 1]  # [B, T, n_sources]
                dwa_weights = dwa_weights.to(stacked.dtype)

                # Weighted sum: [B, T, D]
                x = torch.einsum('lbtd, btl -> btd', stacked, dwa_weights)

            # === Standard attention (all heads share same input) ===
            # Use the layer's own forward for attention
            residual = x
            normed = layer_module.input_layernorm(x) if hasattr(layer_module, 'input_layernorm') else x

            # Run self_attn to get attention output + per-head values
            attn = layer_module.self_attn
            q = attn.q_proj(normed)
            k = attn.k_proj(normed)
            v = attn.v_proj(normed)

            # Q/K norm
            q = attn.q_norm(q)
            k = attn.k_norm(k)

            # Reshape for attention
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # RoPE
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            cos, sin = self.olmo.model.rotary_emb(x, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Attention
            scaling = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
            causal = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=x.dtype)
            causal.masked_fill_(
                torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
                float('-inf'),
            )
            attn_weights = attn_weights + causal
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
            attn_values = torch.matmul(attn_weights, v)  # [B, H, T, head_dim]

            # === Per-head output gating ===
            gates = torch.sigmoid(self.gate_mlps[l](x.float())).to(x.dtype)  # [B, T, H]
            gates = gates.transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]
            gated_values = attn_values * gates  # [B, H, T, head_dim]

            # o_proj on gated values
            gated_concat = gated_values.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            attn_output = attn.o_proj(gated_concat)

            # Post-attention norm + residual
            attn_output = layer_module.post_attention_layernorm(attn_output)
            x = residual + attn_output

            # MLP
            residual = x
            mlp_output = layer_module.mlp(x)
            mlp_output = layer_module.post_feedforward_layernorm(mlp_output)
            x = residual + mlp_output

            # Store for DWA
            layer_outputs.append(x)

        # Final norm + lm_head
        x = self.olmo.model.norm(x)
        logits = self.olmo.lm_head(x)
        return logits

    def get_routing_parameters(self) -> list[nn.Parameter]:
        """Return DWA + gating parameters (for separate optimizer group)."""
        params = list(self.dwa_mlps.parameters()) + list(self.dwa_biases)
        params += list(self.gate_mlps.parameters())
        return params
