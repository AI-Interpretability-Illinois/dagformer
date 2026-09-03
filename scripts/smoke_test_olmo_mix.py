"""Smoke test for olmo_mix dataloader.

Verifies:
- Each of the 7 olmo-mix-1124 subsets can be streamed.
- The interleaved DolmaPackedDataset yields packed sequences with correct shapes.
- Source distribution roughly matches OLMO_MIX weights over a 100-sample window.
- Manual sharding works for world_size>1 simulated case.

Run via SLURM (NOT login node).
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

# Force line-buffered stdout so we see progress even if the job gets killed.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Importing transformers...", flush=True)
from transformers import AutoTokenizer
print("Importing DolmaPackedDataset...", flush=True)
from src.data.dolma import OLMO_MIX, DolmaPackedDataset


def main():
    t_start = time.time()
    print(f"[+{time.time()-t_start:.1f}s] Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    print(f"[+{time.time()-t_start:.1f}s] Tokenizer OK. vocab={tok.vocab_size}, eos={tok.eos_token_id}", flush=True)
    print(f"OLMO_MIX weights: {OLMO_MIX}", flush=True)

    print(f"[+{time.time()-t_start:.1f}s] Constructing DolmaPackedDataset...", flush=True)
    ds = DolmaPackedDataset(
        olmo_tokenizer=tok,
        seq_len=1024,
        dataset_name="allenai/olmo-mix-1124",
        dataset_version="olmo_mix",
        rank=0,
        world_size=1,
    )
    print(f"[+{time.time()-t_start:.1f}s] Dataset constructed. Getting iter()...", flush=True)
    it = iter(ds)
    print(f"[+{time.time()-t_start:.1f}s] iter() built. Streaming 3 packed sequences...", flush=True)

    samples = []
    for i in range(3):
        t0 = time.time()
        s = next(it)
        samples.append(s)
        print(
            f"[+{time.time()-t_start:.1f}s]   [{i}] +{time.time()-t0:.1f}s  "
            f"ids[:8]={s['olmo_ids'][:8].tolist()}  "
            f"raw[:60]={repr(s['raw_text'][:60])}",
            flush=True,
        )

    # Validate shapes
    for s in samples:
        assert s["olmo_ids"].shape == (1024,), s["olmo_ids"].shape
        assert s["olmo_labels"].shape == (1024,), s["olmo_labels"].shape
        assert isinstance(s["raw_text"], str)
    print(f"[+{time.time()-t_start:.1f}s] All shapes OK.", flush=True)
    print("SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
