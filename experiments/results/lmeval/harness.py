"""lm-eval ``LM`` wrapper for the DAGFormer / baseline checkpoints.

``HFLM`` accepts an already-initialised model object, which gives us its whole
request-batching, log-likelihood and few-shot machinery for free.  Only one
thing has to be replaced: generation.

The routed DAGFormer wrapper currently has no KV-cache implementation.
:meth:`CheckpointLM._generate_recompute` re-runs the full prefix for each new
token; cached decoding processes only the new token after the initial prompt.
Generation runs therefore have a separate sample limit — see ``run_eval.py``.

Dense baselines use HF's cached ``generate()`` by default. Cached and recomputed
BF16 execution can differ numerically and occasionally choose different greedy
tokens. ``--no-kv-cache`` uses the recomputation path for every model when
checking sensitivity to that implementation choice.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


class CheckpointLM(HFLM):
    """``HFLM`` over a locally-built checkpoint, with cache-free generation."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        batch_size: int = 8,
        max_length: int = 1024,
        max_gen_toks: int = 256,
        use_kv_cache: bool | None = None,
        softmax_dtype: str | None = "float32",
    ):
        # Resolved before super().__init__ so the property is safe to read during it.
        can_cache = hasattr(model, "generate") and hasattr(model, "prepare_inputs_for_generation")
        self._use_kv_cache = can_cache if use_kv_cache is None else (use_kv_cache and can_cache)
        self._max_gen_toks = int(max_gen_toks)
        self._truncated_generations = 0

        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            batch_size=batch_size,
            max_length=max_length,
            # These checkpoints were pretrained on packed documents with no BOS
            # prepended, so adding one at eval time is a distribution shift.
            add_bos_token=False,
            # bf16 weights: log_softmax in fp32 keeps the log-likelihood
            # differences we are comparing from being quantisation noise.
            softmax_dtype=softmax_dtype,
        )

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def uses_kv_cache(self) -> bool:
        return self._use_kv_cache

    @property
    def truncated_generations(self) -> int:
        """How many samples hit the context limit mid-generation."""
        return self._truncated_generations

    def _model_generate(self, context, max_length: int, stop: list[str], **generation_kwargs):
        if self._use_kv_cache:
            return super()._model_generate(
                context=context, max_length=max_length, stop=stop, **generation_kwargs
            )
        return self._generate_recompute(context, max_length, stop, **generation_kwargs)

    @torch.no_grad()
    def _generate_recompute(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: Sequence[str],
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Greedy/sampled decoding that re-runs the full prefix every step.

        Rows are decoded one at a time on purpose: the routed forward builds its
        own causal mask and takes no ``attention_mask``, so a left-padded batch
        would let pad tokens into attention and shift every RoPE position.  The
        per-row context is therefore un-padded via ``attention_mask`` first, and
        the returned tensor is re-assembled in the padded layout
        ``[B, ctx_len + gen_len]`` that ``HFLM.generate_until`` slices.
        """
        unsupported = set(generation_kwargs) - {"do_sample", "temperature", "max_new_tokens"}
        if unsupported:
            eval_logger.warning(
                "cache-free generation ignores unsupported generation kwargs: %s",
                sorted(unsupported),
            )
        do_sample = bool(generation_kwargs.get("do_sample", False))
        temperature = float(generation_kwargs.get("temperature") or 0.0)

        ctx_len = context.shape[1]
        max_new_tokens = max(int(max_length) - ctx_len, 0)
        eos_id = self.eot_token_id
        stop_strings = [s for s in stop if s]

        generations: list[list[int]] = []
        for row in range(context.shape[0]):
            ids = context[row]
            if attention_mask is not None:
                ids = ids[attention_mask[row].bool()]
            cur = ids.unsqueeze(0)

            produced: list[int] = []
            for _ in range(max_new_tokens):
                if cur.shape[1] >= self.max_length:
                    # Left-truncating mid-generation would shift RoPE positions
                    # and the predictor's positional encoding under the model,
                    # so stop here and report it instead.
                    self._truncated_generations += 1
                    break

                logits = self._model_call(cur)[:, -1, :].float()
                if do_sample and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)

                token = int(next_id)
                produced.append(token)
                cur = torch.cat([cur, next_id.to(cur.dtype)], dim=1)

                if token == eos_id:
                    break
                if stop_strings:
                    text = self.tok_decode(produced)
                    if any(s in text for s in stop_strings):
                        break
            generations.append(produced)

        # Pad with EOS (not <|pad|>) so that tok_decode(skip_special_tokens=True)
        # strips the filler regardless of how the tokenizer classifies its pad token.
        gen_len = max((len(g) for g in generations), default=0)
        out = torch.full(
            (context.shape[0], ctx_len + gen_len),
            eos_id,
            dtype=context.dtype,
            device=context.device,
        )
        out[:, :ctx_len] = context
        for row, produced in enumerate(generations):
            if produced:
                out[row, ctx_len : ctx_len + len(produced)] = torch.tensor(
                    produced, dtype=context.dtype, device=context.device
                )
        return out
