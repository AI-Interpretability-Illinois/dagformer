"""Run lm-evaluation-harness against a MuDDFormer checkpoint.

Mirrors scripts/dagformer_lm_eval.py for the MuDDFormer baseline.
"""
from __future__ import annotations
import argparse, json, os, sys, types

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, '/workspace/dagformer')

from src.model.muddformer.wrapper import build_muddformer


def load_muddformer(yaml_path: str, ckpt_path: str, device: str, dtype: torch.dtype):
    cfg_dict = yaml.safe_load(open(yaml_path))
    cfg = types.SimpleNamespace(**cfg_dict)
    model = build_muddformer(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    m, u = model.load_state_dict(sd, strict=False)
    print(f"  load: missing={len(m)} unexpected={len(u)}")
    model = model.to(device=device, dtype=dtype).eval()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--tasks", default="piqa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output_path", default=None)
    a = ap.parse_args()

    dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[a.dtype]
    model = load_muddformer(a.yaml, a.ckpt, device=a.device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    wrapped = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size, device=a.device)
    res = simple_evaluate(model=wrapped, tasks=a.tasks.split(","), limit=a.limit)
    results = res.get("results", {})
    print(json.dumps(results, indent=2, default=str))
    if a.output_path:
        os.makedirs(os.path.dirname(a.output_path) or ".", exist_ok=True)
        json.dump(results, open(a.output_path, "w"), indent=2, default=str)
        print(f"saved: {a.output_path}")
