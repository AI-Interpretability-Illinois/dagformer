#!/bin/bash
#SBATCH --job-name=pretok21b
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --reservation=sup-30781
#SBATCH --output=logs/pretok21b_%j.out
# Rebuild the 21B-token Dolma v1.7 corpus exactly as scripts/slurm/reserved_train.slurm's tokenize_bg did
# (8 file-level workers, seed 0, same tokenizer), merge, and split the 200-window held-out tail off the last
# shard. The previous index.json / eval_cache.pt are kept as *_prev for a determinism check.
cd /u/xiaocong/dagformer
source ~/.bashrc; conda activate modularity 2>/dev/null || true   # before set -u: /etc/bashrc reads an unset variable
set -u
export PYTHONPATH=/u/xiaocong/dagformer TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Dolma is streamed from the Hub: same network settings as reserved_train.slurm's tokenize_bg
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ENABLE_HF_TRANSFER=0 HF_DATASETS_TIMEOUT=300 HF_XET_NATIVE=0 HF_HUB_DISABLE_XET=1
DATA21=/work/hdd/bfqt/xiaocong/dagformer_pruning/data/dolma_v1_7_21b
N=8
mkdir -p "$DATA21" logs
[ -f "$DATA21/index.json" ] && mv -f "$DATA21/index.json" "$DATA21/index_prev.json"
[ -f "$DATA21/eval_cache.pt" ] && [ ! -f "$DATA21/eval_cache_prev.pt" ] && cp "$DATA21/eval_cache.pt" "$DATA21/eval_cache_prev.pt"
pids=()
for i in $(seq 0 $((N - 1))); do
  w=$DATA21/w$i
  [ -f "$w/index.json" ] && continue
  rm -rf "$w"
  python3 scripts/pretokenize.py --dataset allenai/dolma --dataset-version v1_7 \
      --out-dir "$w" --token-budget 21000000000 --seq-len 1024 --seed 0 \
      --num-shards "$N" --shard-index "$i" \
      --tokenizer-id /work/hdd/bfqt/shared/dagformer-models/tokenizer > "logs/pretok21b_w$i.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] || { echo "a worker failed"; exit 1; }
python3 scripts/merge_pretok_shards.py --root "$DATA21" && \
python3 scripts/split_eval_from_mmap.py --index-dir "$DATA21" --n-windows 200 --out "$DATA21/eval_cache.pt" || exit 1
python3 - <<'PY'
import json, torch
D="/work/hdd/bfqt/xiaocong/dagformer_pruning/data/dolma_v1_7_21b"
new=json.load(open(f"{D}/index.json")); old=json.load(open(f"{D}/index_prev.json"))
same_shards = [s["n_tokens"] for s in new["shards"]] == [s["n_tokens"] for s in old["shards"]]
a=torch.load(f"{D}/eval_cache.pt", map_location="cpu", weights_only=False); b=torch.load(f"{D}/eval_cache_prev.pt", map_location="cpu", weights_only=False)
same_cache = len(a)==len(b) and all(torch.equal(x["olmo_ids"], y["olmo_ids"]) for x, y in zip(a, b))
print(f"total_tokens new={new.get('total_tokens')} old={old.get('total_tokens')} shard_counts_identical={same_shards} eval_cache_identical={same_cache}")
PY
echo "exit rc=$? at $(date)"
