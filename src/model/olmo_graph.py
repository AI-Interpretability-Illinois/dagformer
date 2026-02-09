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
            A: [batch, 256, 256] — block-upper-triangular gate matrix

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = olmo_ids.shape
        device = olmo_ids.device

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

                # Slice A for connections into this layer's heads
                # A[:, source_nodes, target_nodes]
                # source: nodes 0..(l*16-1), target: nodes l*16..(l*16+15)
                A_slice = A[:, :l * self.num_heads, l * self.num_heads:(l + 1) * self.num_heads]
                # A_slice: [B, l*16, 16]

                # Batched gated sum via einsum
                gated_sum = torch.einsum('bih,bisd->bhsd', A_slice, prior_head_outs)
                # gated_sum: [B, 16, S, D]

                # Apply input normalization (only to gated_sum, not base)
                if self.input_normalizer.method == "rms_pre":
                    gated_sum = self.input_normalizer(
                        gated_sum, A_slice=A_slice, prior_head_outs=prior_head_outs
                    )
                elif self.input_normalizer.method == "gate_mean":
                    gated_sum = self.input_normalizer(gated_sum, A_slice=A_slice)
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
        nll = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
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
