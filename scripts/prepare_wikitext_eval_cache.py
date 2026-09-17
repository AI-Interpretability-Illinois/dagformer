"""Build disjoint calibration/evaluation token caches from WikiText splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
import torch
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--train-windows", type=int, default=64)
    ap.add_argument("--validation-windows", type=int, default=50)
    ap.add_argument("--test-windows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "EleutherAI/wikitext_document_level"
    metadata = {"dataset": dataset_name, "config": "wikitext-2-raw-v1",
                "tokenizer": args.tokenizer, "seq_len": args.seq_len,
                "seed": args.seed, "packing": "EOS between documents; nonoverlapping T+1 chunks",
                "splits": {}}
    for split in ("train", "validation", "test"):
        data = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
        target = getattr(args, split + "_windows")
        tokens, doc_ids = [], []
        # Randomize document order before packing, identically for every model.
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed)).tolist()
        for index in order:
            doc = data[index]
            text = doc.get("page", doc.get("text"))
            if not isinstance(text, str):
                raise ValueError(f"No text column in {list(doc)}")
            tokens.extend(tok(text, add_special_tokens=False)["input_ids"])
            tokens.append(tok.eos_token_id)
            doc_ids.append(index)
            if len(tokens) >= target * (args.seq_len + 1):
                break
        if len(tokens) < target * (args.seq_len + 1):
            raise ValueError(f"Split {split} has insufficient tokens for {target} windows")
        chunks = torch.tensor(tokens[:target * (args.seq_len + 1)]).reshape(target, -1)
        batches = [{"olmo_ids": chunks[:, :-1].clone(),
                    "olmo_labels": chunks[:, 1:].clone()}]
        path = args.out_dir / f"wikitext_{split}.pt"
        torch.save(batches, path)
        metadata["splits"][split] = {"path": str(path), "windows": target,
                                     "document_ids": doc_ids,
                                     "dataset_fingerprint": data._fingerprint}
        print(split, target, "windows from", len(doc_ids), "documents", flush=True)
    (args.out_dir / "wikitext_cache_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
