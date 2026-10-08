"""J6: measured cost of routing. Training step time (forward + backward) and inference (prefill) throughput of
dense OLMo-2 vs FourWay DAGFormer on one GPU, same shapes, bf16, plus DAGFormer with the predictor replaced by a
constant table (what a static-topology deployment would cost).

    python scripts/throughput_bench.py --out experiments/topology/throughput.json
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

from scripts.eval_lm_harness import load_config, load_dense, load_fourway  # noqa: E402

MODELS = {
    "300m": {"dense": ("configs/pretrain_300m/300m_dense_mb4.yaml", "/work/hdd/bfqt/xiaocong/dagformer_300m/300m_dense/checkpoint_step12000.pt"),
             "dag": ("configs/pretrain_300m/300m_fourway_corrected_mb4.yaml", "/work/hdd/bfqt/xiaocong/dagformer_300m/300m_fourway_corrected/checkpoint_step12000.pt")},
    "1b": {"dense": ("configs/pretrain_1b/1b_dense_10b_mb2.yaml", "/work/hdd/bfqt/xiaocong/dagformer_1b/1b_dense_10b/checkpoint_step19080.pt"),
           "dag": ("configs/pretrain_1b/1b_fourway_corrected_10b_mb2.yaml", "/work/hdd/bfqt/xiaocong/dagformer_1b/1b_fourway_corrected_10b/checkpoint_step19080.pt")},
}


def timeit(fn, warmup=3, iters=15):
    """Median per-iteration time (robust to a brief co-tenant on the node)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    return times[len(times) // 2]


def bench(size: str, kind: str, device, bs_train: int, bs_infer: int, seq: int) -> dict:
    cfg_path, ckpt = MODELS[size][kind]
    cfg = load_config(os.path.join(REPO, cfg_path))
    out = {"size": size, "kind": kind, "seq_len": seq, "bs_train": bs_train, "bs_infer": bs_infer}
    x_tr = torch.randint(0, cfg["vocab_size"], (bs_train, seq), device=device)
    x_inf = torch.randint(0, cfg["vocab_size"], (bs_infer, seq), device=device)
    if kind == "dense":
        model = load_dense(ckpt, cfg, device)
        for p in model.parameters():
            p.requires_grad_(True)
        fwd = lambda x: model(input_ids=x).logits
        variants = {"dense": fwd}
    else:
        fw, pred = load_fourway(ckpt, cfg, device)
        for p in list(fw.parameters()) + list(pred.parameters()):
            p.requires_grad_(True)
        fwd = lambda x: fw(x, pred(x))
        # static-topology deployment: predictor replaced by a constant table (its mean on random input)
        with torch.no_grad():
            rw0 = pred(x_inf[:1])
            const = {k: [t.mean(dim=(0, 1), keepdim=True).detach() for t in rw0[k]] for k in rw0}
        def fwd_static(x):
            B, T = x.shape
            rw = {k: [t.expand(B, T, *t.shape[2:]) for t in const[k]] for k in const}
            return fw(x, rw)
        variants = {"dag": fwd, "dag_static_predictor": fwd_static}
    for name, f in variants.items():
        # inference (prefill), no grad
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_inf = timeit(lambda: f(x_inf))
        # training step: forward + backward (no optimizer step)
        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = f(x_tr)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x_tr[:, 1:].reshape(-1))
            loss.backward()
        t_tr = timeit(step, warmup=2, iters=9)
        out[name] = {"infer_tok_per_s": bs_infer * seq / t_inf, "infer_ms": t_inf * 1e3,
                     "train_tok_per_s": bs_train * seq / t_tr, "train_ms": t_tr * 1e3,
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(size, name, {k: round(v, 1) for k, v in out[name].items()}, flush=True)
        torch.cuda.reset_peak_memory_stats()
    del variants, fwd
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/topology/throughput.json"))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bs-train", type=int, default=2)
    ap.add_argument("--bs-infer", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    import subprocess
    env = {}
    for name, cmd in (("nvidia_smi", ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"]),
                      ("node_jobs", ["bash", "-c", "squeue -w $(hostname -s) -o '%i %j %u %b %M' 2>/dev/null"])):
        try:
            env[name] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            env[name] = f"unavailable: {e}"
    res = {"gpu": props.name, "torch": torch.__version__, "host": os.uname().nodename, "environment": env, "runs": []}
    for size in ("300m", "1b"):
        for kind in ("dense", "dag"):
            res["runs"].append(bench(size, kind, device, args.bs_train, args.bs_infer, args.seq))
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
