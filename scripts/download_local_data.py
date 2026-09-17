"""Download and pre-tokenize Dolma data locally.

Creates two .pt files:
  - train_sequences.pt: packed sequences from Dolma offset 0 (~600K sequences)
  - eval_sequences.pt:  packed sequences from Dolma offset 1M (50 sequences)

Eliminates all streaming issues during training.

Usage:
    PYTHONPATH=. python scripts/download_local_data.py
"""
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SAVE_DIR = "data/local_dolma"
TOKENIZER_ID = "allenai/OLMo-2-0425-1B"
SEQ_LEN = 1024
TRAIN_SEQUENCES = 600_000   # enough for 1000 steps × 128 accum × 4 batch = 512K
EVAL_SKIP = 1_000_000
EVAL_SEQUENCES = 50


def pack_from_stream(dataset, tokenizer, num_sequences, label=""):
    """Tokenize and pack documents into fixed-length sequences."""
    eos_id = tokenizer.eos_token_id
    buffer = []
    sequences = []
    doc_count = 0

    for doc in dataset:
        if len(sequences) >= num_sequences:
            break

        text = doc.get("text", "")
        if not text.strip():
            continue

        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(tokens)
        buffer.append(eos_id)
        doc_count += 1

        while len(buffer) >= SEQ_LEN + 1 and len(sequences) < num_sequences:
            chunk = buffer[:SEQ_LEN + 1]
            buffer = buffer[SEQ_LEN + 1:]
            sequences.append({
                "olmo_ids": torch.tensor(chunk[:SEQ_LEN], dtype=torch.long),
                "olmo_labels": torch.tensor(chunk[1:SEQ_LEN + 1], dtype=torch.long),
            })

        if doc_count % 50000 == 0:
            print(f"  [{label}] {doc_count:,} docs → {len(sequences):,} sequences")

    print(f"  [{label}] Done: {doc_count:,} docs → {len(sequences):,} sequences")
    return sequences


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # === Training data (offset 0) ===
    train_path = os.path.join(SAVE_DIR, "train_sequences.pt")
    if os.path.exists(train_path):
        print(f"Train data already exists: {train_path}")
    else:
        print(f"Downloading training data ({TRAIN_SEQUENCES:,} sequences)...")
        t0 = time.time()
        ds = load_dataset("allenai/dolma", name="v1_7", split="train",
                          streaming=True, trust_remote_code=True)
        train_seqs = pack_from_stream(ds, tokenizer, TRAIN_SEQUENCES, label="train")
        torch.save(train_seqs, train_path)
        print(f"Saved {len(train_seqs):,} train sequences to {train_path} "
              f"({os.path.getsize(train_path)/1e9:.1f} GB, {time.time()-t0:.0f}s)")

    # === Eval data (offset 1M) ===
    eval_path = os.path.join(SAVE_DIR, "eval_sequences.pt")
    if os.path.exists(eval_path):
        print(f"Eval data already exists: {eval_path}")
    else:
        print(f"Downloading eval data (skip {EVAL_SKIP:,}, take {EVAL_SEQUENCES})...")
        t0 = time.time()
        ds = load_dataset("allenai/dolma", name="v1_7", split="train",
                          streaming=True, trust_remote_code=True)
        ds = ds.skip(EVAL_SKIP)
        eval_seqs = pack_from_stream(ds, tokenizer, EVAL_SEQUENCES, label="eval")
        torch.save(eval_seqs, eval_path)
        print(f"Saved {len(eval_seqs)} eval sequences to {eval_path} "
              f"({os.path.getsize(eval_path)/1e6:.1f} MB, {time.time()-t0:.0f}s)")

    print("\nDone! Use these files with a local dataloader to avoid streaming issues.")


if __name__ == "__main__":
    main()
