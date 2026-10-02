"""Judged chat eval, step 1: answers of the instruction-tuned exports to MT-Bench first turns.

Each model answers the 80 MT-Bench first-turn questions in the ChatML format the SFT
trainer used (scripts/sft/sft_data.py, same prompt as sft_train.sample), greedy, up to
--max-new tokens, total length capped at the model's seq_len. Generation is batched with
right padding and no KV cache: every model here is causal (dense OLMo-2, and the FourWay
predictors have predictor_causal: true), so padding after a sequence's last token never
changes its logits.

    python scripts/chat_eval_generate.py --models experiments/chat_eval/models.yaml \
        --out experiments/chat_eval/answers [--only 300m_dense_alpaca_dolly] [--limit 4]

Output: <out>/<model>.jsonl, one {question_id, category, answer, n_new, stopped} per line.
Already-complete files are skipped, so the job can be re-run as exports land.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from scripts.eval_lm_harness import load_dense, load_fourway  # noqa: E402
from scripts.prune_finetune import PruneModule  # noqa: E402

QUESTIONS = os.path.join(REPO, "experiments/chat_eval/data/mt_bench_question.jsonl")


def load_module(export_dir: str, device) -> tuple[PruneModule, dict]:
    with open(os.path.join(export_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ckpt = os.path.join(export_dir, "checkpoint.pt")
    if str(cfg.get("routing_mode", "")).startswith("fourway"):
        assert cfg.get("predictor_causal", True), f"{export_dir}: non-causal predictor, padding unsafe"
        fourway, predictor = load_fourway(ckpt, cfg, device)
        module = PruneModule(cfg, fourway.olmo, fourway, predictor)
    else:
        module = PruneModule(cfg, load_dense(ckpt, cfg, device), None, None)
    return module.eval(), cfg


@torch.no_grad()
def generate(module, tok, prompts: list[str], device, max_new: int, max_len: int,
             batch: int) -> list[tuple[list[int], bool]]:
    """Greedy continuation of each prompt; returns (new token ids, stopped on end token)."""
    stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    pad = tok.eos_token_id
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    results: list = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(enc[i]))   # similar lengths per batch
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        seqs = {i: list(enc[i]) for i in idx}
        new = {i: [] for i in idx}
        active = list(idx)
        for _ in range(max_new):
            active = [i for i in active if len(seqs[i]) < max_len]
            if not active:
                break
            L = max(len(seqs[i]) for i in active)
            x = torch.full((len(active), L), pad, dtype=torch.long, device=device)
            for r, i in enumerate(active):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i], device=device)
            logits = module(x)
            last = torch.tensor([len(seqs[i]) - 1 for i in active], device=device)
            nxt = logits[torch.arange(len(active), device=device), last].float().argmax(-1).tolist()
            still = []
            for i, t in zip(active, nxt):
                seqs[i].append(t)
                if t in stop_ids:
                    results[i] = (new[i], True)
                else:
                    new[i].append(t)
                    still.append(i)
            active = still
        for i in idx:
            if results[i] is None:
                results[i] = (new[i], False)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(REPO, "experiments/chat_eval/models.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/chat_eval/answers"))
    ap.add_argument("--only", action="append", default=[], help="model name(s) to run")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    spec = yaml.safe_load(open(args.models))
    qs = [json.loads(l) for l in open(QUESTIONS)]
    if args.limit:
        qs = qs[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    for m in spec["models"]:
        name, export = m["name"], m["export"]
        if args.only and name not in args.only:
            continue
        out = os.path.join(args.out, f"{name}.jsonl")
        if os.path.exists(out) and sum(1 for _ in open(out)) >= len(qs):
            print(f"skip (done): {name}", flush=True)
            continue
        if not os.path.exists(os.path.join(export, "checkpoint.pt")):
            print(f"skip (no export yet): {name} {export}", flush=True)
            continue
        t0 = time.time()
        module, cfg = load_module(export, device)
        tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer_id"))
        prompts = [f"<|im_start|>user\n{q['turns'][0]}<|im_end|>\n<|im_start|>assistant\n" for q in qs]
        res = generate(module, tok, prompts, device, args.max_new, int(cfg.get("seq_len", 1024)), args.batch)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            for q, (ids, stopped) in zip(qs, res):
                f.write(json.dumps({"question_id": q["question_id"], "category": q["category"],
                                    "answer": tok.decode(ids, skip_special_tokens=True).strip(),
                                    "n_new": len(ids), "stopped": stopped}) + "\n")
        os.replace(tmp, out)
        n_stop = sum(s for _, s in res)
        print(f"{name}: {len(qs)} answers in {time.time() - t0:.0f}s, {n_stop} ended on <|im_end|>, "
              f"mean {sum(len(i) for i, _ in res) / len(res):.0f} new tokens", flush=True)
        del module
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
