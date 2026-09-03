#!/bin/bash
#SBATCH --job-name=interp_moe
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=logs/interp_moe_%j.out
#SBATCH --error=logs/interp_moe_%j.err

set -u
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export TOKENIZERS_PARALLELISM=false

echo "=== STEP 0: verify shards, re-download corrupt/missing ==="
python3 - << 'PY'
import json, os, struct
from pathlib import Path
from huggingface_hub import snapshot_download

def shard_ok(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
        return os.path.getsize(path) - 8 - n >= end
    except Exception as e:
        print(f"header parse failed for {path.name}: {e}")
        return False

for attempt in range(3):
    d = Path(snapshot_download("allenai/OLMoE-1B-7B-0125-Instruct"))
    bad = []
    for link in sorted(d.glob("*.safetensors")):
        real = link.resolve()
        if shard_ok(real):
            print(f"OK {link.name} ({os.path.getsize(real)/1e9:.2f} GB)")
        else:
            print(f"CORRUPT {link.name} — deleting blob {real.name}")
            bad.append(real)
    if not bad:
        print("all shards verified:", d)
        break
    for b in set(bad):
        os.remove(b)
else:
    raise SystemExit("could not obtain clean shards after 3 attempts")
PY
rc=$?; if [ $rc -ne 0 ]; then echo "DOWNLOAD FAILED rc=$rc"; exit $rc; fi

echo "=== STEP 1: MoE router surgery ==="
export HF_HUB_OFFLINE=1
python3 scripts/interp_moe_swap.py
rc=$?
echo "moe_swap exited rc=$rc"
exit "$rc"
