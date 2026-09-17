"""Compare eval NLL with old (fused kernel) vs fixed (standard kernel) predictor path.

Loads the step 1000 checkpoint and runs eval both ways.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM
import sys, os, yaml
sys.path.insert(0, '.')

from src.model.olmo_graph import FourWayDAGFormer
from src.model.predictor import FourWayPredictor
from scripts.pretrain_dagformer import (
    DAGFormerPretrainConfig, predict_fourway_routing,
    apply_fourway_stream_mask, apply_deterministic_routing_transforms,
    replace_olmo_rmsnorm,
)


def main():
    device = "cuda"
    config_path = "configs/fourway_full_fix_1gpu.yaml"
    ckpt_path = "checkpoints/fourway_full_fix_1gpu/checkpoint_step1000.pt"
    eval_cache_path = "checkpoints/fourway_full_fix_1gpu/eval_cache.pt"

    config = DAGFormerPretrainConfig.from_yaml(config_path)

    # Build model
    olmo_config = Olmo2Config(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_position_embeddings=config.max_position_embeddings,
    )
    base_model = Olmo2ForCausalLM(olmo_config).to(device, dtype=torch.bfloat16)
    if config.replace_rmsnorm:
        n = replace_olmo_rmsnorm(base_model)
        print(f"Replaced {n} Olmo2RMSNorm modules")

    fourway_model = FourWayDAGFormer(
        model=base_model,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        use_local_correction=(config.routing_mode == "fourway_corrected"),
        correction_hidden=config.correction_hidden,
        correction_pool=config.correction_pool,
    ).to(device)

    fourway_predictor = FourWayPredictor(
        vocab_size=config.vocab_size,
        encoder_dim=config.predictor_encoder_dim,
        encoder_layers=config.predictor_encoder_layers,
        encoder_heads=config.predictor_encoder_heads,
        max_seq_len=config.predictor_max_seq_len,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        hidden_dim=config.fourway_hidden,
        causal=config.predictor_causal,
    ).to(device, dtype=torch.bfloat16)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    fourway_predictor.load_state_dict(ckpt["predictor_state_dict"])
    model_state = torch.load(ckpt["model_state_path"], map_location=device)
    base_model.load_state_dict(model_state)
    if "routing_state_dict" in ckpt:
        # Load correction MLPs / v_norms
        fourway_model.load_state_dict(ckpt["routing_state_dict"], strict=False)
    step = ckpt["step"]
    del ckpt, model_state
    print(f"Loaded checkpoint at step {step}")

    # Load eval cache
    eval_batches = torch.load(eval_cache_path)
    print(f"Loaded {len(eval_batches)} eval batches")

    vocab_size = config.vocab_size

    def run_eval(predictor_train_mode: bool, label: str):
        fourway_model.eval()
        base_model.eval()
        if predictor_train_mode:
            fourway_predictor.train()
        else:
            fourway_predictor.eval()

        nll_routing_total = 0.0
        nll_baseline_total = 0.0
        n = 0

        with torch.no_grad():
            for eb in eval_batches:
                eids = eb["olmo_ids"].to(device)
                elabels = eb["olmo_labels"].to(device)

                rw = predict_fourway_routing(fourway_predictor, base_model, eids, config)
                rw = apply_fourway_stream_mask(rw, config)
                rw = apply_deterministic_routing_transforms(rw, config, step)

                logits_r = fourway_model(eids, rw)
                nll_r = F.cross_entropy(
                    logits_r.contiguous().view(-1, vocab_size),
                    elabels.contiguous().view(-1),
                )
                nll_routing_total += nll_r.item()

                eout = base_model(input_ids=eids)
                nll_b = F.cross_entropy(
                    eout.logits.contiguous().view(-1, vocab_size),
                    elabels.contiguous().view(-1),
                )
                nll_baseline_total += nll_b.item()
                n += 1

        nll_routing = nll_routing_total / n
        nll_baseline = nll_baseline_total / n
        print(f"  {label:30s}  routing={nll_routing:.4f}  baseline={nll_baseline:.4f}")
        return nll_routing, nll_baseline

    print(f"\n{'='*60}")
    print(f"Eval comparison at step {step}")
    print(f"{'='*60}")

    r_old, b_old = run_eval(predictor_train_mode=False, label="OLD (eval mode, fused kernel)")
    r_fix, b_fix = run_eval(predictor_train_mode=True,  label="FIX (train mode, std kernel)")

    print(f"\n{'='*60}")
    print(f"Routing NLL improvement: {r_old:.4f} -> {r_fix:.4f} (delta={r_fix - r_old:+.4f})")
    print(f"Baseline NLL (unchanged): {b_old:.4f} -> {b_fix:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
