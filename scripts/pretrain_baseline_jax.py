"""JAX/Flax OLMo-2 baseline pretraining for TPU.

Minimal self-contained implementation matching PyTorch pretrain_baseline.py.
Architecture: OLMo-2 style (post-norm, SwiGLU, RoPE, q_norm/k_norm).

Usage (single host, multi-TPU):
    python scripts/pretrain_baseline_jax.py --config configs/pretrain_1b_baseline.yaml

Dependencies: jax, flax, optax, transformers (tokenizer only), datasets
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from jax import random
from transformers import AutoTokenizer
from datasets import load_dataset
import yaml


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    intermediate_size: int = 8192
    vocab_size: int = 100352
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    tokenizer_id: str = "allenai/OLMo-2-0425-1B"

    # Data
    dataset: str = "allenai/dolma"
    dataset_name: str = "v1_7"
    seq_len: int = 1024
    seed: int = 42

    # Training
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    total_steps: int = 10000
    lr: float = 4e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"

    # Logging
    log_every: int = 10
    save_every: int = 2000
    save_dir: str = "checkpoints/jax_1b_baseline"
    wandb_project: str = "dagformer"
    wandb_run_name: str = "jax-baseline-1b"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f)
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ─── Model ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (x.shape[-1],))
        variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        x_normed = x * jax.lax.rsqrt(variance + self.eps)
        return (weight * x_normed).astype(x.dtype)


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 500000.0):
    """Precompute RoPE cos/sin frequencies."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    t = jnp.arange(max_len).astype(jnp.float32)
    freqs = jnp.outer(t, freqs)  # [max_len, dim//2]
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """Apply RoPE to x: [batch, heads, seq, head_dim]."""
    seq_len = x.shape[2]
    cos = cos[:seq_len]  # [seq, head_dim//2]
    sin = sin[:seq_len]
    # Split into pairs
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # Rotate
    cos = cos[None, None, :, :]  # [1, 1, seq, dim//2]
    sin = sin[None, None, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    # Interleave back
    out = jnp.stack([out1, out2], axis=-1).reshape(x.shape)
    return out


class SwiGLUMLP(nn.Module):
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.intermediate_size, use_bias=False, name='gate_proj')(x)
        up = nn.Dense(self.intermediate_size, use_bias=False, name='up_proj')(x)
        down = nn.Dense(self.hidden_size, use_bias=False, name='down_proj')(
            nn.silu(gate) * up
        )
        return down


class Attention(nn.Module):
    num_heads: int
    head_dim: int
    rms_norm_eps: float

    @nn.compact
    def __call__(self, x, cos, sin, mask):
        B, T, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = nn.Dense(H * hd, use_bias=False, name='q_proj')(x)  # [B, T, H*hd]
        k = nn.Dense(H * hd, use_bias=False, name='k_proj')(x)
        v = nn.Dense(H * hd, use_bias=False, name='v_proj')(x)

        # Q/K norm (OLMo-2 specific)
        q = RMSNorm(eps=self.rms_norm_eps, name='q_norm')(q)
        k = RMSNorm(eps=self.rms_norm_eps, name='k_norm')(k)

        # Reshape to heads
        q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # [B, H, T, hd]
        k = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention
        scale = hd ** -0.5
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # [B, H, T, T]
        attn_weights = attn_weights + mask  # causal mask
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(x.dtype)
        attn_out = jnp.matmul(attn_weights, v)  # [B, H, T, hd]

        # Concat heads + output projection
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, H * hd)  # [B, T, D]
        out = nn.Dense(D, use_bias=False, name='o_proj')(attn_out)
        return out


class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    intermediate_size: int
    hidden_size: int
    rms_norm_eps: float

    @nn.compact
    def __call__(self, x, cos, sin, mask):
        # Attention + post-attention norm (OLMo-2 is POST-norm)
        attn_out = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            name='self_attn',
        )(x, cos, sin, mask)
        x = x + RMSNorm(eps=self.rms_norm_eps, name='post_attention_layernorm')(attn_out)

        # MLP + post-feedforward norm
        mlp_out = SwiGLUMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            name='mlp',
        )(x)
        x = x + RMSNorm(eps=self.rms_norm_eps, name='post_feedforward_layernorm')(mlp_out)
        return x


class OLMo2(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, input_ids):
        cfg = self.config
        B, T = input_ids.shape

        # Token embedding
        embed = nn.Embed(cfg.vocab_size, cfg.hidden_size, name='embed_tokens')
        x = embed(input_ids)  # [B, T, D]

        # Precompute RoPE
        cos, sin = precompute_freqs_cis(cfg.head_dim, cfg.max_position_embeddings)

        # Causal mask
        mask = jnp.triu(jnp.full((T, T), -1e9), k=1)  # [T, T]
        mask = mask[None, None, :, :]  # [1, 1, T, T]

        # Transformer layers
        for i in range(cfg.num_hidden_layers):
            x = TransformerBlock(
                num_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim,
                intermediate_size=cfg.intermediate_size,
                hidden_size=cfg.hidden_size,
                rms_norm_eps=cfg.rms_norm_eps,
                name=f'layers_{i}',
            )(x, cos, sin, mask)

        # Final norm
        x = RMSNorm(eps=cfg.rms_norm_eps, name='norm')(x)

        # LM head (tied with embedding)
        if cfg.tie_word_embeddings:
            logits = embed.attend(x)  # uses transposed embedding weights
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False, name='lm_head')(x)

        return logits


# ─── Data ────────────────────────────────────────────────────────────────────

def build_data_iterator(tokenizer, config: Config):
    """Streaming Dolma with sequence packing (same as PyTorch version)."""
    ds = load_dataset(config.dataset, name=config.dataset_name,
                      split="train", streaming=True, trust_remote_code=True)

    eos_id = tokenizer.eos_token_id
    buffer = []

    for doc in ds:
        text = doc.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        buffer.append(eos_id)

        while len(buffer) >= config.seq_len + 1:
            chunk = buffer[:config.seq_len + 1]
            buffer = buffer[config.seq_len + 1:]
            input_ids = jnp.array(chunk[:config.seq_len], dtype=jnp.int32)
            labels = jnp.array(chunk[1:config.seq_len + 1], dtype=jnp.int32)
            yield input_ids, labels


def get_batch(data_iter, batch_size):
    """Collect a batch from the data iterator."""
    inputs, targets = [], []
    for _ in range(batch_size):
        inp, tgt = next(data_iter)
        inputs.append(inp)
        targets.append(tgt)
    return jnp.stack(inputs), jnp.stack(targets)


# ─── Training ────────────────────────────────────────────────────────────────

def create_learning_rate_fn(config: Config):
    """Linear warmup + linear decay schedule."""
    def lr_fn(step):
        # Warmup
        warmup_lr = config.lr * step / max(config.warmup_steps, 1)
        # Linear decay
        decay_ratio = 1.0 - (step - config.warmup_steps) / max(
            config.total_steps - config.warmup_steps, 1)
        decay_lr = config.lr * jnp.maximum(decay_ratio, 0.0)
        return jnp.where(step < config.warmup_steps, warmup_lr, decay_lr)
    return lr_fn


def compute_loss(params, model, input_ids, labels):
    """Cross-entropy loss."""
    logits = model.apply(params, input_ids)  # [B, T, V]
    # Flatten
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    # Cross entropy
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    loss = -jnp.take_along_axis(log_probs, labels_flat[:, None], axis=-1).squeeze(-1)
    return loss.mean()


@partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
def train_step(state, input_ids, labels):
    """Single training step (pmap'd across devices)."""
    def loss_fn(params):
        return compute_loss(params, state.model, input_ids, labels)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # All-reduce gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    return state, loss


class TrainStateWithModel(train_state.TrainState):
    """TrainState that carries model reference for apply."""
    model: Any = None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    os.makedirs(config.save_dir, exist_ok=True)

    # Print setup
    num_devices = jax.device_count()
    print(f"JAX devices: {num_devices} ({jax.devices()[0].platform})")
    print(f"Model: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}")

    tokens_per_step = (config.micro_batch_size * num_devices *
                       config.gradient_accumulation_steps * config.seq_len)
    print(f"Tokens/step: {tokens_per_step:,} "
          f"({config.micro_batch_size} × {num_devices} devices × "
          f"{config.gradient_accumulation_steps} accum × {config.seq_len} seq)")
    print(f"Total tokens: {tokens_per_step * config.total_steps / 1e9:.2f}B")

    # Initialize model
    model = OLMo2(config=config)
    rng = random.PRNGKey(config.seed)
    dummy_input = jnp.ones((1, config.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count:,}")

    # Optimizer
    lr_fn = create_learning_rate_fn(config)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=lr_fn,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay,
        ),
    )

    state = TrainStateWithModel.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        model=model,
    )

    # Replicate across devices
    state = jax.device_put_replicated(state, jax.devices())

    # Data
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    data_iter = build_data_iterator(tokenizer, config)

    # Training loop
    print(f"\nTraining starts (total_steps={config.total_steps})")
    t0 = time.time()
    tokens_seen = 0

    for step in range(1, config.total_steps + 1):
        # Gradient accumulation
        for _ in range(config.gradient_accumulation_steps):
            # Get batch: [num_devices, micro_batch, seq_len]
            batch_input, batch_labels = get_batch(
                data_iter, config.micro_batch_size * num_devices)
            # Reshape for pmap: [num_devices, micro_batch, seq_len]
            batch_input = batch_input.reshape(num_devices, config.micro_batch_size, -1)
            batch_labels = batch_labels.reshape(num_devices, config.micro_batch_size, -1)
            state, loss = train_step(state, batch_input, batch_labels)

        tokens_seen += tokens_per_step

        # Logging
        if step % config.log_every == 0:
            loss_val = float(loss[0])  # take from first device
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            lr_val = float(lr_fn(step))
            print(f"[step {step}] loss={loss_val:.4f}, lr={lr_val:.6f}, "
                  f"tokens/sec={tok_per_sec:.0f}, tokens_seen={tokens_seen/1e9:.3f}B")

        # Checkpointing
        if step % config.save_every == 0:
            ckpt_path = os.path.join(config.save_dir, f"checkpoint_step{step}.pkl")
            # Save from first device
            params_cpu = jax.tree_util.tree_map(lambda x: x[0], state.params)
            import pickle
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'step': step, 'params': params_cpu}, f)
            print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\nTraining complete. Final loss: {float(loss[0]):.4f}")


if __name__ == "__main__":
    main()
