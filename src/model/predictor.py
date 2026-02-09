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
) -> torch.Tensor:
    """Apply cascading activation gate: kill outgoing edges from disconnected nodes.

    One-pass computation (not layer-by-layer):
    1. Compute incoming sums: inc_j = Σ_i A[i, j]
    2. Compute gates: g_j = σ(k * inc_j) (soft) or (inc_j > 0) (hard)
    3. Apply: A[j, :] *= g_j

    Uses ORIGINAL A values for incoming sums (before any gates applied).
    See CLAUDE.md §2.3 cascading gate section.

    Args:
        A: [batch, 256, 256] — gate matrix
        k: steepness of sigmoid gate (default: 5.0)
        hard: if True, use binary gates (for eval_hard mode)

    Returns:
        A_gated: [batch, 256, 256] — A with cascading gate applied
    """
    # Incoming sum per node: [batch, 256]
    inc = A.sum(dim=1)  # sum over source dimension (rows)

    if hard:
        g = (inc > 0).float()  # [batch, 256]
    else:
        g = torch.sigmoid(k * inc)  # [batch, 256]

    # Gate outgoing edges: A[j, :] *= g[j]
    # g: [B, 256] → [B, 256, 1] to broadcast with A: [B, 256, 256]
    return A * g.unsqueeze(2)


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
