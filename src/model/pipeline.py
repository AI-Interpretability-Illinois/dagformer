"""End-to-end DAGFormer pipeline: raw text → predictor → A → OLMo → NLL.

Glues the structure predictor (Qwen + MLP) with the modified OLMo forward.
This is what the trainer calls. See CLAUDE.md §5 for file responsibilities.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.olmo_graph import DAGFormerOLMo, create_all_ones_A
from src.model.predictor import StructurePredictor


class DAGFormerPipeline(nn.Module):
    """Combines StructurePredictor + DAGFormerOLMo into a single forward pass.

    Forward: raw_text → predictor → A → modified OLMo → logits → NLL

    Only the predictor's MLP params are trainable. OLMo and Qwen are frozen.
    """

    def __init__(
        self,
        olmo_model_id: str = "allenai/OLMo-2-0425-1B",
        qwen_model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        predictor_hidden_dim: int = 1024,
        predictor_rank: int = 32,
        cascading_gate_k: float = 5.0,
        input_norm: str = "none",
        qwen_input_prefix: str = "",
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Load frozen OLMo2-1B
        olmo = AutoModelForCausalLM.from_pretrained(
            olmo_model_id,
            torch_dtype=torch.bfloat16,
        )
        olmo.eval()
        for p in olmo.parameters():
            p.requires_grad_(False)

        # Wrap OLMo with DAGFormer modification
        self.olmo_wrapper = DAGFormerOLMo(model=olmo, input_norm=input_norm)

        # Structure predictor (Qwen encoder + MLP decoder)
        self.predictor = StructurePredictor(
            qwen_model_id=qwen_model_id,
            hidden_dim=predictor_hidden_dim,
            rank=predictor_rank,
            cascading_gate_k=cascading_gate_k,
            qwen_input_prefix=qwen_input_prefix,
            device=device,
        )

        self.vocab_size = olmo.config.vocab_size

        if device is not None:
            self.to(device)

    def forward(
        self,
        raw_texts: list[str],
        olmo_ids: torch.Tensor,
        olmo_labels: torch.Tensor,
        tau: float,
        lambda_sparsity: float = 0.0,
        mode: str = "train",
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: text → A → logits → loss.

        Args:
            raw_texts: list of raw text strings (batch)
            olmo_ids: [batch, seq_len] — OLMo tokenized input
            olmo_labels: [batch, seq_len] — shifted labels for NLL
            tau: Gumbel-Sigmoid temperature
            lambda_sparsity: sparsity coefficient (λ_t)
            mode: "train", "eval_soft", or "eval_hard"

        Returns:
            dict with keys:
                "total_loss": nll + lambda * mean(A) — what the optimizer sees
                "nll": cross-entropy loss
                "sparsity_loss": lambda * mean(A)
                "A": [batch, 256, 256] adjacency matrix
        """
        # Step 1: Predict adjacency matrix
        A = self.predictor(raw_texts, tau=tau, mode=mode)
        # A: [batch, 256, 256]

        # Step 2: Modified OLMo forward with A
        logits = self.olmo_wrapper(olmo_ids, A)
        # logits: [batch, seq_len, vocab_size]

        # Step 3: Compute NLL (next-token prediction)
        # olmo_labels is already shifted (chunk[1:seq_len+1]), no additional shift needed
        nll = F.cross_entropy(
            logits.contiguous().view(-1, self.vocab_size),
            olmo_labels.contiguous().view(-1),
        )

        # Step 4: Sparsity regularization
        sparsity_loss = lambda_sparsity * A.mean()
        total_loss = nll + sparsity_loss

        return {
            "total_loss": total_loss,
            "nll": nll,
            "sparsity_loss": sparsity_loss,
            "A": A,
        }

    def forward_baseline(
        self,
        olmo_ids: torch.Tensor,
        olmo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with A=all-ones (baseline reproduction).

        Used for eval/nll_baseline metric.
        """
        batch = olmo_ids.shape[0]
        A = create_all_ones_A(batch).to(olmo_ids.device)
        with torch.no_grad():
            logits = self.olmo_wrapper(olmo_ids, A)
            # olmo_labels is already shifted, no additional shift needed
            nll = F.cross_entropy(
                logits.contiguous().view(-1, self.vocab_size),
                olmo_labels.contiguous().view(-1),
            )
        return nll

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable parameters (predictor MLP + any norm params)."""
        params = list(self.predictor.get_trainable_parameters())
        # Also include input normalizer params if they exist
        params.extend(self.olmo_wrapper.input_normalizer.parameters())
        return params
