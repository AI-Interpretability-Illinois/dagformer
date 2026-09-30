"""Pretokenize a domain / task corpus into the project's mmap shard format.

Companion to ``scripts/pretokenize.py`` (which handles the Dolma / OLMo-mix
pretraining mixtures). This one turns a *small, cached* HF dataset -- the
finetuning domain for the pruning experiment -- into

    <out_dir>/train/shard_00000.bin + index.json    (read by MmapPackedDataset)
    <out_dir>/eval_cache.pt                          (list of collated batches,
                                                      same layout as the trainer's
                                                      eval cache: olmo_ids /
                                                      olmo_labels / raw_text)

Packing is identical to pretraining: documents are tokenized without special
tokens, joined with EOS, and cut into windows of (seq_len + 1) tokens.

Built-in recipes (all present in the local HF cache, so this runs offline):

    mathinstruct   TIGER-Lab/MathInstruct train; text = instruction + "\\n" + output.
                   A seeded held-out slice of documents becomes the eval cache.
    gsm8k          openai/gsm8k main: train split -> train shards, test split
                   -> eval cache (question + "\\n" + answer).
    wikitext2      EleutherAI/wikitext_document_level wikitext-2-raw-v1:
                   train pages -> train, test pages -> eval. General-domain
                   control for "did domain pruning destroy general ability".

Or a generic dataset via --dataset/--config/--split/--text-template.

Usage (SLURM cpu job, not the login node):
    python scripts/pretokenize_domain.py --recipe mathinstruct \\
        --out-dir /work/hdd/bfqt/xiaocong/dagformer_pruning/data/mathinstruct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Iterator

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from scripts.pretokenize import SHARD_DTYPE, ShardWriter, write_index  # noqa: E402

RECIPES: dict[str, dict] = {
    "mathinstruct": {
        "dataset": "TIGER-Lab/MathInstruct", "config": None,
        "train_split": "train", "eval_split": None,          # eval = held-out docs
        "template": "{instruction}\n{output}",
        "eval_docs": 2000,
    },
    "gsm8k": {
        "dataset": "openai/gsm8k", "config": "main",
        "train_split": "train", "eval_split": "test",
        "template": "{question}\n{answer}",
        "eval_docs": 0,
    },
    "wikitext2": {
        "dataset": "EleutherAI/wikitext_document_level", "config": "wikitext-2-raw-v1",
        "train_split": "train", "eval_split": "test",
        "template": "{page}",
        "eval_docs": 0,
    },
}


def render(template: str, row: dict) -> str:
    return template.format(**row).strip()


def pack(tokenizer, texts: Iterator[str], seq_len: int, eos_id: int) -> Iterator[np.ndarray]:
    """Concatenate docs with EOS separators; yield windows of seq_len + 1."""
    window = seq_len + 1
    buf: list[int] = []
    for text in texts:
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buf.append(eos_id)
        while len(buf) >= window:
            yield np.asarray(buf[:window], dtype=SHARD_DTYPE)
            buf = buf[window:]


def write_train(out_dir: str, tokenizer, texts: list[str], seq_len: int, eos_id: int,
                shard_tokens: int, tokenizer_id: str, dataset: str, version: str,
                seed: int, provenance: dict) -> int:
    os.makedirs(out_dir, exist_ok=True)
    writer = ShardWriter(out_dir, shard_tokens)
    n_windows = 0
    t0 = time.time()
    for arr in pack(tokenizer, iter(texts), seq_len, eos_id):
        writer.add(arr)
        n_windows += 1
        if n_windows % 5000 == 0:
            print(f"[train] {n_windows:,} windows, {writer.total_tokens:,} tokens "
                  f"({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    write_index(out_dir=out_dir, seq_len=seq_len, eos_id=eos_id, tokenizer_id=tokenizer_id,
                dataset=dataset, dataset_version=version, seed=seed,
                total_tokens=writer.total_tokens, shards=writer.shards, provenance=provenance)
    print(f"[train] DONE: {n_windows:,} windows / {writer.total_tokens:,} tokens -> {out_dir}")
    return n_windows


def write_eval(path: str, tokenizer, texts: list[str], seq_len: int, eos_id: int,
               batch_size: int, max_windows: int) -> int:
    samples = []
    for arr in pack(tokenizer, iter(texts), seq_len, eos_id):
        chunk = arr.astype(np.int64).tolist()
        samples.append({
            "olmo_ids": torch.tensor(chunk[:seq_len], dtype=torch.long),
            "olmo_labels": torch.tensor(chunk[1:seq_len + 1], dtype=torch.long),
            "raw_text": tokenizer.decode(chunk[:seq_len], skip_special_tokens=False),
        })
        if max_windows > 0 and len(samples) >= max_windows:
            break
    batches = []
    for i in range(0, len(samples), batch_size):
        items = samples[i:i + batch_size]
        batches.append({
            "olmo_ids": torch.stack([s["olmo_ids"] for s in items]),
            "olmo_labels": torch.stack([s["olmo_labels"] for s in items]),
            "raw_text": [s["raw_text"] for s in items],
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(batches, tmp)
    os.replace(tmp, path)
    print(f"[eval] {len(samples)} windows in {len(batches)} batches -> {path}")
    return len(samples)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", choices=sorted(RECIPES), default=None)
    p.add_argument("--dataset", default=None, help="HF dataset id (generic mode)")
    p.add_argument("--config", default=None, help="HF dataset config name (generic mode)")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default=None)
    p.add_argument("--text-template", default=None, help='e.g. "{instruction}\\n{output}"')
    p.add_argument("--eval-docs", type=int, default=None,
                   help="held-out train documents used as eval when there is no eval split")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--tokenizer-id", default="/work/hdd/bfqt/shared/dagformer-models/tokenizer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-docs", type=int, default=0, help="0 = all")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-eval-windows", type=int, default=0, help="0 = all")
    p.add_argument("--shard-tokens", type=int, default=256_000_000)
    args = p.parse_args()

    if args.recipe:
        r = RECIPES[args.recipe]
        dataset = args.dataset or r["dataset"]
        config = args.config or r["config"]
        train_split = r["train_split"]
        eval_split = args.eval_split or r["eval_split"]
        template = args.text_template or r["template"]
        eval_docs = r["eval_docs"] if args.eval_docs is None else args.eval_docs
    else:
        assert args.dataset and args.text_template, "generic mode needs --dataset and --text-template"
        dataset, config, template = args.dataset, args.config, args.text_template
        train_split, eval_split = args.train_split, args.eval_split
        eval_docs = args.eval_docs or 0

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None

    print(f"[data] {dataset} config={config} train={train_split} eval={eval_split}")
    ds_train = load_dataset(dataset, config, split=train_split) if config else \
        load_dataset(dataset, split=train_split)
    train_texts = [render(template, row) for row in ds_train]
    rng = random.Random(args.seed)
    rng.shuffle(train_texts)

    eval_texts: list[str]
    if eval_split:
        ds_eval = load_dataset(dataset, config, split=eval_split) if config else \
            load_dataset(dataset, split=eval_split)
        eval_texts = [render(template, row) for row in ds_eval]
    else:
        assert eval_docs > 0, "no eval split: set --eval-docs > 0 to hold out train documents"
        eval_texts, train_texts = train_texts[:eval_docs], train_texts[eval_docs:]
    if args.max_train_docs > 0:
        train_texts = train_texts[:args.max_train_docs]
    print(f"[data] {len(train_texts):,} train docs, {len(eval_texts):,} eval docs")

    provenance = {
        "recipe": args.recipe, "dataset": dataset, "config": config,
        "train_split": train_split, "eval_split": eval_split, "template": template,
        "eval_docs_heldout": 0 if eval_split else eval_docs,
        "n_train_docs": len(train_texts), "n_eval_docs": len(eval_texts),
        "seed": args.seed,
    }
    n_train = write_train(os.path.join(args.out_dir, "train"), tokenizer, train_texts,
                          args.seq_len, eos_id, args.shard_tokens, args.tokenizer_id,
                          dataset, config or "", args.seed, provenance)
    n_eval = write_eval(os.path.join(args.out_dir, "eval_cache.pt"), tokenizer, eval_texts,
                        args.seq_len, eos_id, args.eval_batch_size, args.max_eval_windows)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({**provenance, "train_windows": n_train, "eval_windows": n_eval,
                   "seq_len": args.seq_len}, f, indent=2)


if __name__ == "__main__":
    main()
