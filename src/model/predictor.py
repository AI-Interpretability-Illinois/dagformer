"""Structure predictor: Qwen encoder + MLP decoder + Gumbel-Sigmoid + cascading gate.

Takes raw text, produces a 256x256 adjacency matrix A controlling per-head
routing in OLMo2-1B. See CLAUDE.md §2.3 for full specification.

Components:
- QwenEncoder: frozen Qwen3-Embedding-0.6B, mean-pooled to single vector
- PredictorMLP: trainable MLP with low-rank output heads (U, V → Z = UV^T)
- Gumbel-Sigmoid: differentiable relaxation of binary gates (3 modes)
- Cascading gate: kill outgoing edges from disconnected nodes
- Block-upper-triangular mask: enforce DAG constraint (layer(j) > layer(i))
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.model.olmo_graph import create_block_upper_triangular_mask


class QwenEncoder(nn.Module):
    """Frozen Qwen3-Embedding-0.6B encoder.

    Produces a single fixed-size vector per sequence via mean pooling.
    Uses its OWN tokenizer (separate from OLMo's).
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B", device: Optional[torch.device] = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embed_dim: int = self.model.config.hidden_size  # 1024 for Qwen3-Embedding-0.6B

        if device is not None:
            self.model = self.model.to(device)

    def encode(self, raw_texts: list[str], prefix: str = "") -> torch.Tensor:
        """Encode raw text strings to pooled embeddings.

        Args:
            raw_texts: list of raw text strings (one per sequence in batch)
            prefix: optional prefix for Qwen input (default: "" — no prefix)

        Returns:
            pooled: [batch, embed_dim] — mean-pooled embedding per sequence
        """
        if prefix:
            raw_texts = [prefix + t for t in raw_texts]

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over sequence dimension (masking padding tokens)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [B, S, 1]
        last_hidden = outputs.last_hidden_state  # [B, S, embed_dim]
        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-8)
        # pooled: [B, embed_dim]

        return pooled


class PredictorMLP(nn.Module):
    """Trainable MLP decoder with low-rank output heads.

    Maps Qwen embedding → logit matrix Z = UV^T ∈ R^{256×256}.
    See CLAUDE.md §2.3 for architecture.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024, rank: int = 32, num_nodes: int = 256,
                 init_logit: float = 15.0):
        super().__init__()
        self.rank = rank
        self.num_nodes = num_nodes

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_U = nn.Linear(hidden_dim, num_nodes * rank)
        self.head_V = nn.Linear(hidden_dim, num_nodes * rank)

        # Initialize head_U and head_V with small weights so UV^T ≈ 0 at init.
        # Default Kaiming init gives UV^T with std≈√rank≈5.7 which overwhelms
        # the logit_bias. Small init ensures Z ≈ logit_bias ± small noise.
        # std=0.01 gives UV^T std≈0.6 (with hidden_dim=1024, rank=32),
        # small vs logit_bias=15 but enough for input-dependent gradients.
        nn.init.normal_(self.head_U.weight, std=0.01)
        nn.init.normal_(self.head_V.weight, std=0.01)
        nn.init.zeros_(self.head_U.bias)
        nn.init.zeros_(self.head_V.bias)

        # Learnable bias added to Z logits. Initialized positive so that
        # σ(init_logit / τ_init) ≈ 1, reproducing dense connectivity (A≈1)
        # at init. With τ_init=5.0: σ(15/5) = σ(3) ≈ 0.95.
        # Training can decrease this to enable sparsity.
        self.logit_bias = nn.Parameter(torch.tensor(init_logit))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Map embedding to logit matrix.

        Args:
            e: [batch, input_dim] — pooled Qwen embedding

        Returns:
            Z: [batch, 256, 256] — raw logit matrix (before mask/Gumbel)
        """
        h = self.trunk(e)  # [batch, hidden_dim]
        U = self.head_U(h).view(-1, self.num_nodes, self.rank)  # [B, 256, r]
        V = self.head_V(h).view(-1, self.num_nodes, self.rank)  # [B, 256, r]
        Z = torch.bmm(U, V.transpose(-1, -2))  # [B, 256, 256]
        Z = Z + self.logit_bias  # shift logits positive → A≈1 at init
        return Z


def gumbel_sigmoid(
    Z_masked: torch.Tensor,
    tau: float,
    mode: str = "train",
) -> torch.Tensor:
    """Apply Gumbel-Sigmoid relaxation to masked logits.

    Three modes (CLAUDE.md §2.3):
    - "train": Gumbel noise + temperature → differentiable continuous relaxation
    - "eval_soft": σ(Z/τ) — deterministic soft gates, no noise
    - "eval_hard": (Z > 0).float() — deterministic binary 0/1

    Args:
        Z_masked: [batch, 256, 256] — logits with invalid positions at -1e9
        tau: temperature (τ > 0 for train/eval_soft)
        mode: one of "train", "eval_soft", "eval_hard"

    Returns:
        A: [batch, 256, 256] — gate values in [0, 1] (or {0, 1} for hard mode)
    """
    if mode == "train":
        # Sample from Logistic(0, 1): G = log(U) - log(1-U), U ~ Uniform(0,1)
        U = torch.rand_like(Z_masked).clamp(1e-8, 1 - 1e-8)
        G = torch.log(U) - torch.log(1 - U)
        return torch.sigmoid((Z_masked + G) / tau)
    elif mode == "eval_soft":
        return torch.sigmoid(Z_masked / tau)
    elif mode == "eval_hard":
        return (Z_masked > 0).float()
    else:
        raise ValueError(f"Unknown Gumbel-Sigmoid mode: {mode}. Expected: train, eval_soft, eval_hard")


def cascading_gate(
    A: torch.Tensor,
    k: float = 5.0,
    hard: bool = False,
    heads_per_layer: int = 16,
) -> torch.Tensor:
    """Apply cascading activation gate: kill outgoing edges from disconnected nodes.

    One-pass computation (not layer-by-layer):
    1. Compute incoming sums: inc_j = Σ_i A[i, j]
    2. Compute gates: g_j = σ(k * inc_j) (soft) or (inc_j > 0) (hard)
    3. Apply: A[j, :] *= g_j

    Layer 0 nodes are exempted: they have inc=0 structurally (no prior layers)
    but receive the embedding as input, so they are NOT disconnected.

    Supports both per-window A [B, N, N] and per-token A [B, T, N, N].

    Args:
        A: [batch, N, N] or [batch, T, N, N] — gate matrix
        k: steepness of sigmoid gate (default: 5.0)
        hard: if True, use binary gates (for eval_hard mode)
        heads_per_layer: number of heads per layer (default: 16)

    Returns:
        A_gated: same shape as A — with cascading gate applied
    """
    # Determine source dimension: dim=-2 is source (rows) for both 3D and 4D
    # For [B, N, N]: sum over dim=1 (source) → [B, N]
    # For [B, T, N, N]: sum over dim=2 (source) → [B, T, N]
    inc = A.sum(dim=-2)  # [..., N] — incoming sum per target node

    if hard:
        g = (inc > 0).float()
    else:
        g = torch.sigmoid(k * inc)

    # Exempt layer 0: always g=1
    N = g.shape[-1]
    exempt = torch.arange(N, device=g.device) < heads_per_layer
    # Reshape exempt for broadcasting: [N] → broadcastable with g's shape
    g = torch.where(exempt, torch.ones_like(g), g)

    # Gate outgoing edges: A[..., j, :] *= g[..., j]
    # g: [..., N] → [..., N, 1] to broadcast with A: [..., N, N]
    return A * g.unsqueeze(-1)


class StaticPredictor(nn.Module):
    """Static structure predictor: learnable global topology (same A for all inputs).

    No input dependency. Learnable low-rank parameters U, V produce a single
    global Z = UV^T + logit_bias. Gumbel noise still differs per batch item
    in training mode, but the underlying logits are shared.

    Total params: 2 × num_nodes × rank + 1 (logit_bias) ≈ 12K for default settings.
    """

    def __init__(
        self,
        num_nodes: int = 192,
        heads_per_layer: int = 16,
        rank: int = 32,
        cascading_gate_k: float = 5.0,
        init_logit: float = 15.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer
        self.rank = rank
        self.cascading_gate_k = cascading_gate_k

        # Learnable low-rank factors
        self.U = nn.Parameter(torch.randn(num_nodes, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(num_nodes, rank) * 0.01)
        self.logit_bias = nn.Parameter(torch.tensor(init_logit))

        # Block-upper-triangular mask
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

    def forward(
        self,
        batch_size: int,
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Produce adjacency matrix A (same logits for all batch items).

        Args:
            batch_size: number of items in batch (for expanding A)
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [batch, num_nodes, num_nodes] — block-upper-triangular gate matrix
        """
        # Z = UV^T + logit_bias: [num_nodes, num_nodes]
        Z = self.U @ self.V.t() + self.logit_bias

        # Apply mask
        mask = self.dag_mask  # [num_nodes, num_nodes]
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        # Expand to batch: [batch, num_nodes, num_nodes]
        Z_masked = Z_masked.unsqueeze(0).expand(batch_size, -1, -1)

        # Gumbel-Sigmoid (noise differs per batch item in train mode)
        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)

        # Cascading gate
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard,
                           heads_per_layer=self.heads_per_layer)

        assert A.shape == (batch_size, self.num_nodes, self.num_nodes), \
            f"A shape mismatch: expected ({batch_size}, {self.num_nodes}, {self.num_nodes}), got {A.shape}"

        return A

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return all parameters (all are trainable)."""
        return list(self.parameters())


class SelfEmbedPredictor(nn.Module):
    """Self-embed structure predictor: input-dependent A from model's own embeddings.

    Uses the base model's token embedding layer output (before any transformer
    computation) to produce input-dependent A. No circular dependency: embeddings
    are computed before A is used.

    Pipeline: embedding [B, S, D] → mean pool → [B, D] → PredictorMLP → Z → mask → Gumbel → cascade → A
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 1024,
        num_nodes: int = 192,
        heads_per_layer: int = 16,
        rank: int = 32,
        cascading_gate_k: float = 5.0,
        init_logit: float = 15.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer
        self.cascading_gate_k = cascading_gate_k

        # Reuse PredictorMLP (same architecture as Qwen-based predictor)
        self.mlp = PredictorMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            num_nodes=num_nodes,
            init_logit=init_logit,
        )

        # Block-upper-triangular mask
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Produce input-dependent adjacency matrix A from token embeddings.

        Args:
            embeddings: [batch, seq_len, embed_dim] — base model's embedding output
                        (should be .detach()'d by caller to break gradient back into embeddings)
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [batch, num_nodes, num_nodes] — block-upper-triangular gate matrix
        """
        # Mean pool over sequence dimension: [B, S, D] → [B, D]
        # Cast to float32 — embeddings may be bf16 from base model
        pooled = embeddings.float().mean(dim=1)

        # MLP → logits: [B, num_nodes, num_nodes]
        Z = self.mlp(pooled)
        assert Z.shape[1:] == (self.num_nodes, self.num_nodes), \
            f"Z shape mismatch: expected (*, {self.num_nodes}, {self.num_nodes}), got {Z.shape}"

        # Apply mask
        mask = self.dag_mask
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        # Gumbel-Sigmoid
        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)

        # Cascading gate
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard,
                           heads_per_layer=self.heads_per_layer)

        assert A.shape[1:] == (self.num_nodes, self.num_nodes), \
            f"A shape mismatch: expected (*, {self.num_nodes}, {self.num_nodes}), got {A.shape}"

        return A

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable MLP parameters."""
        return list(self.mlp.parameters())


class MiniEncoderPredictor(nn.Module):
    """Independent encoder predictor: separate embedding table + small transformer.

    Has its OWN embedding table (not shared with the base LLM) plus a lightweight
    transformer encoder to build contextual representations before pooling.
    The predictor learns its own text representation optimized for topology prediction.

    Pipeline: olmo_ids → own embed_table → small transformer → mean pool → PredictorMLP → A
    """

    def __init__(
        self,
        vocab_size: int = 100352,
        encoder_dim: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        hidden_dim: int = 1024,
        num_nodes: int = 192,
        heads_per_layer: int = 16,
        rank: int = 32,
        cascading_gate_k: float = 5.0,
        init_logit: float = 15.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer
        self.cascading_gate_k = cascading_gate_k

        # Independent embedding table
        self.embed = nn.Embedding(vocab_size, encoder_dim)

        # Small transformer encoder for contextual mixing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # MLP → A
        self.mlp = PredictorMLP(
            input_dim=encoder_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            num_nodes=num_nodes,
            init_logit=init_logit,
        )

        # Block-upper-triangular mask
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Produce A from token IDs using independent encoder.

        Args:
            input_ids: [batch, seq_len] — OLMo token IDs (shared vocab)
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [batch, num_nodes, num_nodes]
        """
        # Own embedding lookup
        x = self.embed(input_ids)  # [B, S, encoder_dim]

        # Causal mask for transformer (predictor sees full context, use None for bidirectional)
        x = self.encoder(x)  # [B, S, encoder_dim]

        # Mean pool → [B, encoder_dim]
        pooled = x.mean(dim=1)

        # MLP → logits
        Z = self.mlp(pooled)
        mask = self.dag_mask
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        # Gumbel-Sigmoid + cascading gate
        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard,
                           heads_per_layer=self.heads_per_layer)
        return A

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())


class SeqToMatrixPredictor(nn.Module):
    """Seq-to-matrix predictor: independent encoder with cross-attention node queries.

    Instead of mean-pooling to a single vector, uses learned node queries
    (one per attention head in the LLM) that cross-attend to the encoded
    sequence. Each node extracts context-specific information, then pairwise
    interactions via low-rank UV^T produce the adjacency matrix.

    Pipeline:
        olmo_ids [B, S]
            → own Embedding + positional embedding → [B, S, d]
            → Transformer Encoder (bidirectional) → [B, S, d]
            → Cross-Attention: 192 learned node queries attend to encoded
            → node_repr [B, 192, d]
            → head_U(node_repr) → U [B, 192, r]
              head_V(node_repr) → V [B, 192, r]
            → Z = UV^T + logit_bias → mask → Gumbel → cascade → A
    """

    def __init__(
        self,
        vocab_size: int = 100352,
        encoder_dim: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        cross_attn_heads: int = 4,
        max_seq_len: int = 4096,
        num_nodes: int = 192,
        heads_per_layer: int = 16,
        rank: int = 32,
        cascading_gate_k: float = 5.0,
        init_logit: float = 15.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer
        self.cascading_gate_k = cascading_gate_k
        self.rank = rank

        # Independent embedding table + positional encoding
        self.embed = nn.Embedding(vocab_size, encoder_dim)
        self.pos_embed = nn.Embedding(max_seq_len, encoder_dim)

        # Transformer encoder (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Learned node queries: each represents one attention head in the LLM
        self.node_queries = nn.Parameter(torch.randn(num_nodes, encoder_dim) * 0.02)

        # Cross-attention: node queries attend to encoded sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=cross_attn_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.cross_norm = nn.LayerNorm(encoder_dim)

        # Low-rank output heads (shared across nodes, node-specificity from cross-attn)
        self.head_U = nn.Linear(encoder_dim, rank)
        self.head_V = nn.Linear(encoder_dim, rank)

        # Small init so UV^T ≈ 0 at init → Z ≈ logit_bias → A ≈ 1
        nn.init.normal_(self.head_U.weight, std=0.01)
        nn.init.normal_(self.head_V.weight, std=0.01)
        nn.init.zeros_(self.head_U.bias)
        nn.init.zeros_(self.head_V.bias)

        # Logit bias: positive init for dense start (A ≈ 1)
        self.logit_bias = nn.Parameter(torch.tensor(init_logit))

        # Block-upper-triangular mask
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Produce A from token IDs using cross-attention node queries.

        Args:
            input_ids: [batch, seq_len] — OLMo token IDs
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [batch, num_nodes, num_nodes]
        """
        B, S = input_ids.shape

        # Token + positional embedding
        positions = torch.arange(S, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions)  # [B, S, d]

        # Encode (bidirectional)
        x = self.encoder(x)  # [B, S, d]

        # Cross-attention: node queries attend to encoded sequence
        queries = self.node_queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, d]
        node_repr, _ = self.cross_attn(queries, x, x)  # [B, N, d]
        node_repr = self.cross_norm(node_repr + queries)  # residual + norm

        # Low-rank factorization → logit matrix
        U = self.head_U(node_repr)  # [B, N, r]
        V = self.head_V(node_repr)  # [B, N, r]
        Z = torch.bmm(U, V.transpose(-1, -2))  # [B, N, N]
        Z = Z + self.logit_bias

        assert Z.shape == (B, self.num_nodes, self.num_nodes), \
            f"Z shape {Z.shape} != ({B}, {self.num_nodes}, {self.num_nodes})"

        # Mask → Gumbel-Sigmoid → cascading gate
        mask = self.dag_mask
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard,
                           heads_per_layer=self.heads_per_layer)
        return A

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())


class PerTokenSeq2MatrixPredictor(nn.Module):
    """Per-token global predictor: sees full sequence, outputs per-position A.

    Unlike SeqToMatrixPredictor which pools to one A per window, this outputs
    a separate 192×192 routing matrix for EACH token position. The predictor
    has a global view of the input (bidirectional encoder) and makes coordinated
    routing decisions across all positions.

    Pipeline:
        olmo_ids [B, T]
            → own Embedding + positional embedding → [B, T, d]
            → Transformer Encoder (bidirectional) → [B, T, d]
            → per-position low-rank projection:
                head_U(encoded) → U [B, T, N, r]
                head_V(encoded) → V [B, T, N, r]
            → Z = UV^T + logit_bias → [B, T, N, N]
            → mask → Gumbel-Sigmoid → cascading gate → A [B, T, N, N]
    """

    def __init__(
        self,
        vocab_size: int = 100352,
        encoder_dim: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        max_seq_len: int = 4096,
        num_nodes: int = 192,
        heads_per_layer: int = 16,
        rank: int = 16,
        cascading_gate_k: float = 5.0,
        init_logit: float = 15.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer
        self.cascading_gate_k = cascading_gate_k
        self.rank = rank

        # Independent embedding table + positional encoding
        self.embed = nn.Embedding(vocab_size, encoder_dim)
        self.pos_embed = nn.Embedding(max_seq_len, encoder_dim)

        # Transformer encoder (bidirectional — sees full context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Per-position projection to low-rank UV^T
        self.head_U = nn.Linear(encoder_dim, num_nodes * rank)
        self.head_V = nn.Linear(encoder_dim, num_nodes * rank)

        # Small init so UV^T ≈ 0 at init → Z ≈ logit_bias → A ≈ 1
        nn.init.normal_(self.head_U.weight, std=0.01)
        nn.init.normal_(self.head_V.weight, std=0.01)
        nn.init.zeros_(self.head_U.bias)
        nn.init.zeros_(self.head_V.bias)

        # Logit bias: positive init for dense start (A ≈ 1)
        self.logit_bias = nn.Parameter(torch.tensor(init_logit))

        # Block-upper-triangular mask
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Produce per-token A from token IDs.

        Args:
            input_ids: [batch, seq_len] — OLMo token IDs
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [batch, seq_len, num_nodes, num_nodes]
        """
        B, T = input_ids.shape

        # Token + positional embedding
        positions = torch.arange(T, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions)  # [B, T, d]

        # Encode (bidirectional — full context visibility)
        x = self.encoder(x)  # [B, T, d]

        # Per-position low-rank projection
        U = self.head_U(x).view(B, T, self.num_nodes, self.rank)  # [B, T, N, r]
        V = self.head_V(x).view(B, T, self.num_nodes, self.rank)  # [B, T, N, r]
        Z = torch.einsum('btnr, btmr -> btnm', U, V)  # [B, T, N, N]
        Z = Z + self.logit_bias

        # Mask → Gumbel-Sigmoid → cascading gate (applied per token position)
        mask = self.dag_mask  # [N, N]
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard,
                           heads_per_layer=self.heads_per_layer)
        return A

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())


class ContextEmbedPredictor(nn.Module):
    """Shared-embedding contextual predictor: reuses LLM's embed_tokens + small encoder.

    Shares the base model's embedding table (no extra vocab params), then applies
    a projection + lightweight transformer encoder to build contextual representations
    before pooling. Only the projection + encoder + MLP are trainable.

    Pipeline: LLM embeddings [B,S,D] → project → small transformer → mean pool → MLP → A
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        encoder_dim: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        hidden_dim: int = 1024,
        num_nodes: int = 192,
        heads_per_layer: int = 16,
        rank: int = 32,
        cascading_gate_k: float = 5.0,
        init_logit: float = 15.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer
        self.cascading_gate_k = cascading_gate_k

        # Project from LLM embed_dim to smaller encoder_dim
        self.project = nn.Linear(embed_dim, encoder_dim)

        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # MLP → A
        self.mlp = PredictorMLP(
            input_dim=encoder_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            num_nodes=num_nodes,
            init_logit=init_logit,
        )

        # Block-upper-triangular mask
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Produce A from LLM embeddings + contextual encoder.

        Args:
            embeddings: [batch, seq_len, embed_dim] — base model's embedding output
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [batch, num_nodes, num_nodes]
        """
        # Cast to float32 (embeddings may be bf16)
        x = embeddings.float()

        # Project down: [B, S, embed_dim] → [B, S, encoder_dim]
        x = self.project(x)

        # Contextual encoding
        x = self.encoder(x)  # [B, S, encoder_dim]

        # Mean pool → [B, encoder_dim]
        pooled = x.mean(dim=1)

        # MLP → logits
        Z = self.mlp(pooled)
        mask = self.dag_mask
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        # Gumbel-Sigmoid + cascading gate
        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard,
                           heads_per_layer=self.heads_per_layer)
        return A

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable (project + encoder + MLP)."""
        return list(self.parameters())


class StructurePredictor(nn.Module):
    """Full structure predictor: raw text → adjacency matrix A.

    Pipeline: raw_text → [Qwen encoder] → e → [MLP] → Z → [mask] → [Gumbel] → [cascade] → A

    The only trainable component is the PredictorMLP. Qwen is frozen.
    """

    def __init__(
        self,
        qwen_model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        hidden_dim: int = 1024,
        rank: int = 32,
        cascading_gate_k: float = 5.0,
        qwen_input_prefix: str = "",
        init_logit: float = 15.0,
        num_nodes: int = 256,
        heads_per_layer: int = 16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.cascading_gate_k = cascading_gate_k
        self.qwen_input_prefix = qwen_input_prefix
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer

        # Frozen Qwen encoder
        self.qwen_encoder = QwenEncoder(model_id=qwen_model_id, device=device)

        # Trainable MLP decoder
        self.mlp = PredictorMLP(
            input_dim=self.qwen_encoder.embed_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            init_logit=init_logit,
            num_nodes=num_nodes,
        )

        # Block-upper-triangular mask (registered as buffer — moves with .to(device))
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

        # Move all components to device (buffers + trainable MLP)
        if device is not None:
            self.to(device)

    def forward(
        self,
        raw_texts: list[str],
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Predict adjacency matrix A from raw text.

        Args:
            raw_texts: list of raw text strings (batch)
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [batch, 256, 256] — block-upper-triangular gate matrix
        """
        # Step 1: Qwen encoding (frozen, no grad)
        e = self.qwen_encoder.encode(raw_texts, prefix=self.qwen_input_prefix)
        # e: [batch, qwen_embed_dim]

        # Step 2: MLP decoder → logits
        Z = self.mlp(e)  # [batch, 256, 256]
        assert Z.shape[1:] == (self.num_nodes, self.num_nodes), \
            f"Z shape mismatch: expected (*, {self.num_nodes}, {self.num_nodes}), got {Z.shape}"

        # Step 3: Apply block-upper-triangular mask
        # Force invalid positions to -inf so sigmoid → 0
        mask = self.dag_mask  # [256, 256]
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        # Step 4: Gumbel-Sigmoid
        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)

        # Step 5: Cascading activation gate
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard)

        assert A.shape[1:] == (self.num_nodes, self.num_nodes), \
            f"A shape mismatch: expected (*, {self.num_nodes}, {self.num_nodes}), got {A.shape}"

        return A

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable MLP parameters (not Qwen)."""
        return list(self.mlp.parameters())


class FourWayPredictor(nn.Module):
    """4-way per-head per-token predictor: predicts Q/K/V/R routing weights.

    Independent encoder sees full sequence, outputs per-position routing weights
    for all layers and all 4 streams. Soft continuous values, identity init.

    Sources: layer outputs (layer-level granularity)
    Targets: per-head (Q/K/V) or shared (R)

    For each layer l (1..num_layers-1):
        Q/K/V: [B, T, H, l+1] — per head, per source layer
        R:     [B, T, l+1]    — shared across heads

    Identity init: bias = [0,...,0,1] (only most recent layer), W_out=0.
    At init, equivalent to standard transformer.
    """

    def __init__(
        self,
        vocab_size: int = 100352,
        encoder_dim: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        max_seq_len: int = 4096,
        num_layers: int = 12,
        num_heads: int = 16,
        hidden_dim: int = 512,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.causal = causal

        # Independent embedding + positional encoding
        self.embed = nn.Embedding(vocab_size, encoder_dim)
        self.pos_embed = nn.Embedding(max_seq_len, encoder_dim)

        # Transformer encoder (causal or bidirectional)
        # dropout > 0 enables classical regularization on attention + FFN.
        # In eval mode (.eval()) dropout is automatically disabled by nn.Module.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Shared trunk: encoder output → hidden
        # Dropout after GELU regularizes the input to per-layer routing heads.
        self.trunk = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-layer output heads + static biases
        # Each layer l (1..num_layers-1) needs:
        #   Q/K/V: H * (l+1) each → 3 * H * (l+1) total
        #   R: (l+1)
        #   Total: 3 * H * (l+1) + (l+1) = (3*H + 1) * (l+1)
        self.layer_heads = nn.ModuleList()
        self.layer_biases = nn.ParameterList()

        for l in range(1, num_layers):
            n_src = l + 1
            out_dim = (3 * num_heads + 1) * n_src
            head = nn.Linear(hidden_dim, out_dim, bias=False)
            # W=0 init: output starts at zero, only bias matters
            nn.init.zeros_(head.weight)
            self.layer_heads.append(head)

            # Static bias: identity init [0, 0, ..., 0, 1] for each stream
            # Q/K/V: H copies of [0,...,0,1], R: one [0,...,0,1]
            bias = torch.zeros(out_dim)
            # Set the last source weight to 1 for each stream
            for stream in range(3 * num_heads + 1):
                bias[stream * n_src + (n_src - 1)] = 1.0
            self.layer_biases.append(nn.Parameter(bias))

    def forward(self, input_ids: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Predict per-token 4-way routing weights for all layers.

        Args:
            input_ids: [B, T] — token IDs

        Returns:
            dict with 'q', 'k', 'v', 'r' keys, each a list of tensors
            (one per layer l=1..num_layers-1):
                'q'/'k'/'v': [B, T, H, l+1]
                'r': [B, T, l+1]
        """
        B, T = input_ids.shape
        H = self.num_heads

        # Encode
        positions = torch.arange(T, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions)
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            x = self.encoder(x, mask=causal_mask)  # [B, T, encoder_dim]
        else:
            x = self.encoder(x)  # [B, T, encoder_dim] — bidirectional
        x = self.trunk(x)    # [B, T, hidden_dim]

        # Per-layer routing weights
        result: dict[str, list[torch.Tensor]] = {'q': [], 'k': [], 'v': [], 'r': []}

        for l in range(1, self.num_layers):
            n_src = l + 1
            raw = self.layer_heads[l - 1](x) + self.layer_biases[l - 1]  # [B, T, out_dim]

            # Split into Q/K/V (per-head) and R (shared)
            qkv_size = H * n_src
            α_q = raw[:, :, :qkv_size].view(B, T, H, n_src)
            α_k = raw[:, :, qkv_size:2*qkv_size].view(B, T, H, n_src)
            α_v = raw[:, :, 2*qkv_size:3*qkv_size].view(B, T, H, n_src)
            α_r = raw[:, :, 3*qkv_size:]  # [B, T, n_src]

            result['q'].append(α_q)
            result['k'].append(α_k)
            result['v'].append(α_v)
            result['r'].append(α_r)

        return result

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())


class FourWayStaticPredictor(nn.Module):
    """Static FourWay routing shared across all inputs and token positions.

    This is the cleanest control for whether FourWay's layer-mixing math is
    itself problematic, or whether the train/eval pathology mainly comes from
    input-conditioned routing bandwidth.
    """

    def __init__(
        self,
        num_layers: int = 12,
        num_heads: int = 16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.q_biases = nn.ParameterList()
        self.k_biases = nn.ParameterList()
        self.v_biases = nn.ParameterList()
        self.r_biases = nn.ParameterList()

        for l in range(1, num_layers):
            n_src = l + 1

            q = torch.zeros(num_heads, n_src)
            k = torch.zeros(num_heads, n_src)
            v = torch.zeros(num_heads, n_src)
            r = torch.zeros(n_src)

            q[:, -1] = 1.0
            k[:, -1] = 1.0
            v[:, -1] = 1.0
            r[-1] = 1.0

            self.q_biases.append(nn.Parameter(q))
            self.k_biases.append(nn.Parameter(k))
            self.v_biases.append(nn.Parameter(v))
            self.r_biases.append(nn.Parameter(r))

    def forward(self, input_ids: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Broadcast static routing weights to every batch/time position."""
        B, T = input_ids.shape
        result: dict[str, list[torch.Tensor]] = {'q': [], 'k': [], 'v': [], 'r': []}

        for q, k, v, r in zip(self.q_biases, self.k_biases, self.v_biases, self.r_biases):
            n_src = r.shape[0]
            α_q = q.view(1, 1, self.num_heads, n_src).expand(B, T, self.num_heads, n_src)
            α_k = k.view(1, 1, self.num_heads, n_src).expand(B, T, self.num_heads, n_src)
            α_v = v.view(1, 1, self.num_heads, n_src).expand(B, T, self.num_heads, n_src)
            α_r = r.view(1, 1, n_src).expand(B, T, n_src)

            result['q'].append(α_q)
            result['k'].append(α_k)
            result['v'].append(α_v)
            result['r'].append(α_r)

        return result

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())


class FourWayPositionalPredictor(nn.Module):
    """Position-indexed FourWay routing: a learned per-position table with no
    content pathway at all.

    Motivated by the 2026-08-31 causal analysis of the trained 300M encoder
    predictor: replacing its per-token output with a per-POSITION mean matched
    full dynamic routing exactly, and cross-context alpha swaps were free —
    i.e. the encoder's causal contribution is a positional schedule. This
    variant tests the remaining hypothesis that content capacity matters as
    *training scaffolding*: if pos-table-from-scratch matches
    encoder-from-scratch, the external predictor is fully replaceable.
    """

    def __init__(
        self,
        max_seq_len: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.q_tables = nn.ParameterList()
        self.k_tables = nn.ParameterList()
        self.v_tables = nn.ParameterList()
        self.r_tables = nn.ParameterList()

        for l in range(1, num_layers):
            n_src = l + 1
            for tables, shape in (
                (self.q_tables, (max_seq_len, num_heads, n_src)),
                (self.k_tables, (max_seq_len, num_heads, n_src)),
                (self.v_tables, (max_seq_len, num_heads, n_src)),
                (self.r_tables, (max_seq_len, n_src)),
            ):
                t = torch.zeros(*shape)
                t[..., -1] = 1.0  # identity init, same as other variants
                tables.append(nn.Parameter(t))

    def forward(self, input_ids: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Broadcast per-position routing weights over the batch."""
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"T={T} > max_seq_len={self.max_seq_len}"
        result: dict[str, list[torch.Tensor]] = {'q': [], 'k': [], 'v': [], 'r': []}
        for i in range(self.num_layers - 1):
            n_src = i + 2
            result['q'].append(self.q_tables[i][:T].unsqueeze(0).expand(B, T, self.num_heads, n_src))
            result['k'].append(self.k_tables[i][:T].unsqueeze(0).expand(B, T, self.num_heads, n_src))
            result['v'].append(self.v_tables[i][:T].unsqueeze(0).expand(B, T, self.num_heads, n_src))
            result['r'].append(self.r_tables[i][:T].unsqueeze(0).expand(B, T, n_src))
        return result

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())


class FourWayAttentionBottleneckPredictor(nn.Module):
    """FourWay predictor with a single causal memory read per token.

    Query = current token layer-0 embedding from the base model.
    Key/Value = dense scout pass final hidden states from the same base model.

    Each token first reads a single summary vector from its causal prefix via
    cross-attention, then a small MLP predicts all Q/K/V/R routing weights.
    This deliberately bottlenecks conditional bandwidth compared with the
    independent token+pos encoder in ``FourWayPredictor``.
    """

    def __init__(
        self,
        model_dim: int = 1024,
        encoder_dim: int = 256,
        attn_heads: int = 4,
        num_layers: int = 12,
        num_heads: int = 16,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.query_proj = nn.Linear(model_dim, encoder_dim, bias=False)
        self.key_proj = nn.Linear(model_dim, encoder_dim, bias=False)
        self.value_proj = nn.Linear(model_dim, encoder_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.trunk = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layer_heads = nn.ModuleList()
        self.layer_biases = nn.ParameterList()

        for l in range(1, num_layers):
            n_src = l + 1
            out_dim = (3 * num_heads + 1) * n_src
            head = nn.Linear(hidden_dim, out_dim, bias=False)
            nn.init.zeros_(head.weight)
            self.layer_heads.append(head)

            bias = torch.zeros(out_dim)
            for stream in range(3 * num_heads + 1):
                bias[stream * n_src + (n_src - 1)] = 1.0
            self.layer_biases.append(nn.Parameter(bias))

    def forward(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
    ) -> dict[str, list[torch.Tensor]]:
        """Predict per-token 4-way routing weights from a bottleneck memory read.

        Args:
            query_states: [B, T, D] current-token layer-0 embeddings.
            memory_states: [B, T, D] dense scout final hidden states.

        Returns:
            dict with 'q', 'k', 'v', 'r' keys, each a list of tensors:
                'q'/'k'/'v': [B, T, H, l+1]
                'r': [B, T, l+1]
        """
        B, T, _ = query_states.shape
        H = self.num_heads
        dtype = self.query_norm.weight.dtype

        q_in = query_states.to(dtype=dtype)
        m_in = memory_states.to(dtype=dtype)

        q = self.query_proj(self.query_norm(q_in))
        k = self.key_proj(self.memory_norm(m_in))
        v = self.value_proj(self.memory_norm(m_in))

        causal_mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        x, _ = self.cross_attn(q, k, v, attn_mask=causal_mask, need_weights=False)
        x = x + q
        x = self.trunk(x)

        result: dict[str, list[torch.Tensor]] = {'q': [], 'k': [], 'v': [], 'r': []}

        for l in range(1, self.num_layers):
            n_src = l + 1
            raw = self.layer_heads[l - 1](x) + self.layer_biases[l - 1]

            qkv_size = H * n_src
            α_q = raw[:, :, :qkv_size].view(B, T, H, n_src)
            α_k = raw[:, :, qkv_size:2*qkv_size].view(B, T, H, n_src)
            α_v = raw[:, :, 2*qkv_size:3*qkv_size].view(B, T, H, n_src)
            α_r = raw[:, :, 3*qkv_size:]

            result['q'].append(α_q)
            result['k'].append(α_k)
            result['v'].append(α_v)
            result['r'].append(α_r)

        return result

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())
class AttentionPoolingPredictor(nn.Module):
    """Predictor for a single token position using cross-attention pooling.

    Predicts the topology (adjacency matrix A) for processing token t, given:
    - Query: raw embedding (layer 0) of token t                    [B, D]
    - Keys/Values: DAG-modified last-layer hidden states of
      tokens 0..t-1, accumulated from previous steps              [B, t, D]

    The forward is inherently single-step: topology for token t can only be
    computed after tokens 0..t-1 have been fully processed by DAGFormerOLMo,
    so the pipeline calls this once per token in a sequential loop.

    For t=0 (no previous tokens), a learnable null K/V token provides the
    only context, acting as a learned prior over the empty-history topology.

    Pipeline (single step at position t):
        embed_t         [B, D]
        hidden_prev     [B, t, D]   (t=0 allowed: empty history)

        Q      = q_proj(embed_t)          [B, d_attn]
        K_full = [null_kv, k_proj(hidden_prev)]  [B, t+1, d_attn]
        V_full = [null_kv, v_proj(hidden_prev)]  [B, t+1, d_attn]
        scores = Q · K_full^T / √d        [B, t+1]
        pooled = softmax(scores) @ V_full  [B, d_attn]  ← single pooled vector
        pooled = out_proj(pooled)          [B, d_attn]
        Z      = MLP(pooled)              [B, N, N]
        A      = mask → Gumbel → cascade  [B, N, N]
    """

    def __init__(
        self,
        model_dim: int = 2048,
        attn_dim: int = 256,
        mlp_hidden_dim: int = 1024,
        num_nodes: int = 256,
        heads_per_layer: int = 16,
        rank: int = 32,
        cascading_gate_k: float = 5.0,
        init_logit: float = 15.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.heads_per_layer = heads_per_layer
        self.cascading_gate_k = cascading_gate_k
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5

        # Single-head cross-attention: Q from embedding, K/V from past hidden states
        self.q_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, attn_dim, bias=False)

        # Learned prior for the empty-history case (t=0)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, attn_dim))

        # 2-layer MLP (PredictorMLP): pooled vector → logit matrix [N, N]
        self.mlp = PredictorMLP(
            input_dim=attn_dim,
            hidden_dim=mlp_hidden_dim,
            rank=rank,
            num_nodes=num_nodes,
            init_logit=init_logit,
        )

        # Block-upper-triangular mask
        self.register_buffer(
            'dag_mask',
            create_block_upper_triangular_mask(num_nodes, heads_per_layer),
        )

    def forward(
        self,
        embed_t: torch.Tensor,
        hidden_prev: torch.Tensor,
        tau: float,
        mode: str = "train",
    ) -> torch.Tensor:
        """Predict topology for token t from its embedding and past hidden states.

        Args:
            embed_t: [B, D] — raw embedding of the current token t
            hidden_prev: [B, t, D] — DAG-modified last-layer hidden states of
                tokens 0..t-1 (t=0 is valid: pass an empty [B, 0, D] tensor)
            tau: Gumbel-Sigmoid temperature
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            A: [B, N, N] — adjacency matrix for token t
        """
        B = embed_t.shape[0]

        # Cast to float32 (OLMo outputs may be bfloat16)
        embed_t = embed_t.float()
        hidden_prev = hidden_prev.float()

        # Single query vector from the current token's raw embedding
        Q = self.q_proj(embed_t)       # [B, d_attn]

        # Keys and values from all previous tokens' last-layer hidden states.
        # Prepend null_kv so softmax is well-defined when t=0 (empty history).
        null = self.null_kv.expand(B, 1, self.attn_dim)         # [B, 1, d_attn]
        K_full = torch.cat([null, self.k_proj(hidden_prev)], dim=1)  # [B, t+1, d_attn]
        V_full = torch.cat([null, self.v_proj(hidden_prev)], dim=1)  # [B, t+1, d_attn]

        # Attention: Q [B, 1, d_attn] × K_full^T [B, d_attn, t+1] → [B, 1, t+1]
        scores = torch.bmm(Q.unsqueeze(1), K_full.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)             # [B, 1, t+1]
        pooled = torch.bmm(attn_weights, V_full).squeeze(1)      # [B, d_attn]
        pooled = self.out_proj(pooled)                           # [B, d_attn]

        # MLP → logit matrix
        Z = self.mlp(pooled)  # [B, N, N]

        # Apply block-upper-triangular mask
        mask = self.dag_mask  # [N, N]
        Z_masked = Z * mask + (-1e9) * (1 - mask)

        # Gumbel-Sigmoid + cascading gate
        hard = (mode == "eval_hard")
        A = gumbel_sigmoid(Z_masked, tau=tau, mode=mode)
        A = cascading_gate(A, k=self.cascading_gate_k, hard=hard,
                           heads_per_layer=self.heads_per_layer)

        return A  # [B, N, N]

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters are trainable."""
        return list(self.parameters())
