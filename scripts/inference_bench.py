"""Measured inference and training cost of every Table 1 model at every size, on one GPU.

Inference: forward pass over a batch of 8 sequences of 1024 tokens (prefill; how every evaluation in the paper runs),
no grad, bf16 autocast. Training: forward + backward over 2 x 1024 tokens (no optimizer step). Median of 15 (inference)
and 9 (training) timed iterations after warm-up. Weights do not change the speed, so the baselines are built at random
initialization from the scaling-study configs; DAGFormer is loaded from a checkpoint of the same architecture. Every model
is cast to bf16 weights, as in pretraining (pretrain_baseline.py / pretrain_dagformer.py train in pure bf16), so no
family pays autocast weight casts that another does not.
Families: dense OLMo-2, DenseFormer, hyper-connections (paper-exact DHCx4), mHC, MUDDFormer, DAGFormer (corrected).

    python scripts/inference_bench.py --out experiments/scaling/inference_cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from scripts.eval_lm_harness import _cfg_to_namespace, load_config, load_fourway  # noqa: E402

SIZES = {  # dense config (shapes) and a DAGFormer checkpoint of the same shapes
    "75m": ("configs/pretrain_75m_baseline_timan1_dolma12b.yaml",
            ("/work/hdd/bfqt/shared/dagformer-models/75m-dagformer/config.yaml",
             "/work/hdd/bfqt/shared/dagformer-models/75m-dagformer/checkpoint.pt")),
    "150m": ("configs/pretrain_150m_baseline_timan1_dolma12b.yaml",
             ("/work/hdd/bfqt/shared/dagformer-models/150m-dagformer/config.yaml",
              "/work/hdd/bfqt/shared/dagformer-models/150m-dagformer/checkpoint.pt")),
    "300m": ("configs/pretrain_300m/300m_dense_mb4.yaml",
             ("configs/pretrain_300m/300m_fourway_corrected_mb4.yaml",
              "/work/hdd/bfqt/xiaocong/dagformer_300m/300m_fourway_corrected/checkpoint_step12000.pt")),
    "1b": ("configs/pretrain_1b/1b_dense_10b_mb2.yaml",
           ("configs/pretrain_1b/1b_fourway_corrected_10b_mb2.yaml",
            "/work/hdd/bfqt/xiaocong/dagformer_1b/1b_fourway_corrected_10b/checkpoint_step19080.pt")),
}
FAMILIES = ["dense", "denseformer", "hc_paper", "mhc", "muddformer", "dagformer"]


def timeit(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def build(family: str, size: str, device):
    dense_cfg = load_config(os.path.join(REPO, SIZES[size][0]))
    ns = _cfg_to_namespace(dense_cfg)
    if family == "dagformer":
        cpath, ckpt = SIZES[size][1]
        cfg = load_config(cpath if os.path.isabs(cpath) else os.path.join(REPO, cpath))
        fw, pred = load_fourway(ckpt, cfg, device)
        return (lambda x: fw(x, pred(x))), [fw, pred], dense_cfg["vocab_size"]
    if family == "dense":
        from transformers import Olmo2Config, Olmo2ForCausalLM
        m = Olmo2ForCausalLM(Olmo2Config(hidden_size=ns.hidden_size, num_hidden_layers=ns.num_hidden_layers,
                                         num_attention_heads=ns.num_attention_heads, num_key_value_heads=ns.num_attention_heads,
                                         intermediate_size=ns.intermediate_size, vocab_size=ns.vocab_size,
                                         tie_word_embeddings=True, max_position_embeddings=ns.max_position_embeddings))
    elif family == "denseformer":
        from src.model.denseformer.wrapper import build_denseformer
        m = build_denseformer(ns)
    elif family == "muddformer":
        from src.model.muddformer.wrapper import build_muddformer
        m = build_muddformer(ns)
    elif family in ("hc_paper", "mhc"):
        from src.model.hyperconnection.paper_exact import build_hc_paper, build_mhc
        m = (build_hc_paper if family == "hc_paper" else build_mhc)(ns)
    m = m.to(device)
    return (lambda x: m(input_ids=x).logits), [m], dense_cfg["vocab_size"]


def bench(family, size, device, bs_infer, bs_train, seq):
    torch.manual_seed(0)
    fwd, mods, vocab = build(family, size, device)
    for m in mods:
        m.to(torch.bfloat16)
    params = [p for m in mods for p in m.parameters()]
    x_inf = torch.randint(0, vocab, (bs_infer, seq), device=device)
    x_tr = torch.randint(0, vocab, (bs_train, seq), device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_inf = timeit(lambda: fwd(x_inf))
    mem_inf = torch.cuda.max_memory_allocated() / 1e9
    for p in params:
        p.requires_grad_(True)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(x_tr)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x_tr[:, 1:].reshape(-1))
        loss.backward()
        for p in params:
            p.grad = None
    t_tr = timeit(step, warmup=2, iters=9)
    out = {"family": family, "size": size, "infer_tok_per_s": bs_infer * seq / t_inf, "infer_ms": t_inf * 1e3,
           "infer_peak_gb": mem_inf, "train_tok_per_s": bs_train * seq / t_tr, "train_ms": t_tr * 1e3,
           "n_params": sum(p.numel() for p in params)}
    del fwd, params, mods
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/scaling/inference_cost.json"))
    ap.add_argument("--sizes", default="75m,150m,300m,1b")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs-infer", type=int, default=8)
    ap.add_argument("--bs-train", type=int, default=2)
    a = ap.parse_args()
    device = torch.device("cuda")
    res = {"gpu": torch.cuda.get_device_name(0), "seq": a.seq, "bs_infer": a.bs_infer, "bs_train": a.bs_train, "runs": []}
    for size in a.sizes.split(","):
        for fam in a.families.split(","):
            try:
                r = bench(fam, size, device, a.bs_infer, a.bs_train, a.seq)
            except Exception as e:  # keep going: one OOM must not lose the rest
                r = {"family": fam, "size": size, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                torch.cuda.empty_cache()
            res["runs"].append(r)
            print(json.dumps(r), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    dense = {r["size"]: r for r in res["runs"] if r.get("family") == "dense" and "error" not in r}
    for r in res["runs"]:
        if "error" in r or r["size"] not in dense:
            continue
        d = dense[r["size"]]
        print(f"{r['size']:5s} {r['family']:12s} infer {r['infer_tok_per_s']/1e3:7.1f}K tok/s ({d['infer_tok_per_s']/r['infer_tok_per_s']:.2f}x dense time)"
              f"  train {r['train_tok_per_s']/1e3:6.1f}K tok/s ({d['train_tok_per_s']/r['train_tok_per_s']:.2f}x)")


if __name__ == "__main__":
    main()
