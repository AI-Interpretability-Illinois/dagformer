"""Build packed, loss-masked instruction-tuning windows for the SFT filler jobs.

Datasets (raw files fetched once on a login node into $SFT_HF_ROOT):
    smol_smoltalk   HuggingFaceTB/smol-smoltalk (SmolLM2-135M/360M SFT mix), train/test
    alpaca_dolly    yahma/alpaca-cleaned (52k) + databricks-dolly-15k, 1% held out

Every conversation is rendered in ChatML with the tokenizer's native
<|im_start|>/<|im_end|> tokens, tokenized per message so that the loss mask
covers exactly the assistant turns (content + <|im_end|>) and the trailing
<|endoftext|>. Conversations are concatenated and cut into windows of
seq_len+1 tokens (standard packing). Output per split:

    <out_dir>/<split>_ids.npy   uint32 [N, seq_len+1]
    <out_dir>/<split>_mask.npy  uint8  [N, seq_len+1]   1 = predict this token
    <out_dir>/meta.json

    python scripts/sft/sft_data.py --dataset smol_smoltalk --out /work/.../data/smol_smoltalk
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np

HF_ROOT = os.environ.get("SFT_HF_ROOT", "/work/hdd/bfqt/xiaocong/dagformer_sft/hf")
TOKENIZER = os.environ.get("SFT_TOKENIZER", "/work/hdd/bfqt/shared/dagformer-models/tokenizer")
MAX_CONV_TOKENS = 4096

# ─── raw loading → list[list[{"role","content"}]] ────────────────────────────

def _read_parquets(pattern: str) -> list[dict]:
    import pyarrow.parquet as pq
    rows: list[dict] = []
    for f in sorted(glob.glob(pattern)):
        rows.extend(pq.read_table(f).to_pylist())
    return rows


def _alpaca_to_messages(row: dict) -> list[dict]:
    user = row["instruction"].strip()
    ctx = (row.get("input") or row.get("context") or "").strip()
    if ctx:
        user = f"{user}\n\n{ctx}"
    answer = (row.get("output") or row.get("response") or "").strip()
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def load_conversations(dataset: str, seed: int = 0) -> dict[str, list[list[dict]]]:
    if dataset == "smol_smoltalk":
        d = os.path.join(HF_ROOT, "HuggingFaceTB__smol-smoltalk", "data")
        train = [r["messages"] for r in _read_parquets(os.path.join(d, "train-*.parquet"))]
        val = [r["messages"] for r in _read_parquets(os.path.join(d, "test-*.parquet"))]
        return {"train": train, "val": val}
    if dataset == "alpaca_dolly":
        with open(os.path.join(HF_ROOT, "yahma__alpaca-cleaned", "alpaca_data_cleaned.json")) as f:
            rows = json.load(f)
        with open(os.path.join(HF_ROOT, "databricks__databricks-dolly-15k", "databricks-dolly-15k.jsonl")) as f:
            rows += [json.loads(l) for l in f if l.strip()]
        convs = [_alpaca_to_messages(r) for r in rows]
        convs = [c for c in convs if c[0]["content"] and c[1]["content"]]
        random.Random(seed).shuffle(convs)
        n_val = max(200, len(convs) // 100)
        return {"train": convs[n_val:], "val": convs[:n_val]}
    raise ValueError(f"unknown dataset {dataset!r}")


# ─── ChatML tokenization with assistant-only loss mask ───────────────────────

def tokenize_conversations(tok, convs: list[list[dict]], chunk: int = 4000):
    """Yield (ids, mask) uint32/uint8 arrays for each chunk of conversations, already
    concatenated (conversations separated by <|endoftext|>)."""
    eos = tok.eos_token_id
    for start in range(0, len(convs), chunk):
        block = convs[start:start + chunk]
        segs: list[str] = []          # flattened text segments
        layout: list[list[tuple[int, int]]] = []   # per conv: (segment index, mask)
        for conv in block:
            lay = []
            for m in conv:
                role = m["role"]
                if role not in ("system", "user", "assistant"):
                    continue
                is_a = 1 if role == "assistant" else 0
                lay.append((len(segs), 0)); segs.append(f"<|im_start|>{role}\n")
                lay.append((len(segs), is_a)); segs.append(m["content"])
                lay.append((len(segs), is_a)); segs.append("<|im_end|>\n")
            layout.append(lay)
        enc = tok(segs, add_special_tokens=False)["input_ids"]
        enc = [np.asarray(e, dtype=np.uint32) for e in enc]
        ids_parts, mask_parts = [], []
        for lay in layout:
            if not any(mk for _, mk in lay):        # no assistant turn -> nothing to learn
                continue
            ids = np.concatenate([enc[si] for si, _ in lay] + [np.array([eos], np.uint32)])
            mask = np.concatenate([np.full(len(enc[si]), mk, np.uint8) for si, mk in lay] + [np.array([1], np.uint8)])
            if len(ids) > MAX_CONV_TOKENS:
                ids, mask = ids[:MAX_CONV_TOKENS], mask[:MAX_CONV_TOKENS]
            ids_parts.append(ids); mask_parts.append(mask)
        if ids_parts:
            yield np.concatenate(ids_parts), np.concatenate(mask_parts)


def pack(stream, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    W = seq_len + 1
    carry_ids = np.zeros(0, np.uint32); carry_mask = np.zeros(0, np.uint8)
    out_ids, out_mask = [], []
    for ids, mask in stream:
        ids = np.concatenate([carry_ids, ids]); mask = np.concatenate([carry_mask, mask])
        n = len(ids) // W
        if n:
            out_ids.append(ids[:n * W].reshape(n, W)); out_mask.append(mask[:n * W].reshape(n, W))
        carry_ids, carry_mask = ids[n * W:], mask[n * W:]
    if not out_ids:
        return np.zeros((0, W), np.uint32), np.zeros((0, W), np.uint8)
    return np.concatenate(out_ids), np.concatenate(out_mask)


PART_CONVS = 20000     # conversations per resumable part file


def _save_np(arr: np.ndarray, path: str) -> None:
    np.save(path + ".tmp.npy", arr)
    os.replace(path + ".tmp.npy", path)


def build(dataset: str, out_dir: str, seq_len: int = 1024, max_val_windows: int = 300,
          stop=None) -> dict:
    """Build the packed windows. Resumable: work is written in parts of PART_CONVS
    conversations (<split>_partNNN_{ids,mask}.npy); an interrupted run continues
    from the last complete part. `stop()` (optional) is polled between parts and
    raises InterruptedError when it returns True."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"   # no fork after this point; use all cores
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    assert tok.convert_tokens_to_ids("<|im_start|>") != tok.unk_token_id
    convs = load_conversations(dataset)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"dataset": dataset, "seq_len": seq_len, "tokenizer": TOKENIZER, "splits": {}}
    for split, cs in convs.items():
        n_parts = (len(cs) + PART_CONVS - 1) // PART_CONVS
        for k in range(n_parts):
            if stop is not None and stop():
                raise InterruptedError("data preparation interrupted between parts")
            p_ids = os.path.join(out_dir, f"{split}_part{k:03d}_ids.npy")
            p_mask = os.path.join(out_dir, f"{split}_part{k:03d}_mask.npy")
            if os.path.exists(p_ids) and os.path.exists(p_mask):
                continue
            part = cs[k * PART_CONVS:(k + 1) * PART_CONVS]
            ids, mask = pack(tokenize_conversations(tok, part), seq_len)
            _save_np(ids, p_ids); _save_np(mask, p_mask)
            print(f"[sft_data] {dataset}/{split} part {k+1}/{n_parts}: {len(part)} convs -> {len(ids)} windows "
                  f"({time.time()-t0:.0f}s)", flush=True)
        ids = np.concatenate([np.load(os.path.join(out_dir, f"{split}_part{k:03d}_ids.npy")) for k in range(n_parts)])
        mask = np.concatenate([np.load(os.path.join(out_dir, f"{split}_part{k:03d}_mask.npy")) for k in range(n_parts)])
        if split == "val" and len(ids) > max_val_windows:
            ids, mask = ids[:max_val_windows], mask[:max_val_windows]
        _save_np(ids, os.path.join(out_dir, f"{split}_ids.npy"))
        _save_np(mask, os.path.join(out_dir, f"{split}_mask.npy"))
        meta["splits"][split] = {"conversations": len(cs), "windows": int(len(ids)),
                                 "tokens": int(ids.size), "loss_tokens": int(mask.sum())}
        print(f"[sft_data] {dataset}/{split}: {len(cs)} convs -> {len(ids)} windows, "
              f"{ids.size/1e6:.1f}M tokens ({mask.sum()/max(ids.size,1):.0%} supervised)", flush=True)
    meta["seconds"] = round(time.time() - t0, 1)
    tmp = os.path.join(out_dir, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "meta.json"))
    for f in glob.glob(os.path.join(out_dir, "*_part*_*.npy")):   # parts are only needed for resume
        os.remove(f)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["smol_smoltalk", "alpaca_dolly"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many train conversations")
    args = ap.parse_args()
    if args.limit:
        orig = load_conversations
        globals()["load_conversations"] = lambda ds, seed=0: {k: v[:args.limit] for k, v in orig(ds, seed).items()}
    build(args.dataset, args.out, args.seq_len)


if __name__ == "__main__":
    main()
