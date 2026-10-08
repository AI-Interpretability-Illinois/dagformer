#!/bin/bash
#SBATCH --job-name=locality_patch
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --reservation=sup-30781
#SBATCH --nodelist=gpua047
#SBATCH --output=logs/locality_patch_%j.out
# J2: cross-layer patching (experiments/coherence/cross_layer_patch.py) on the 75M router-locality arms.
cd /u/xiaocong/dagformer
source ~/.bashrc; conda activate modularity 2>/dev/null || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
R=/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints/locality
mkdir -p experiments/coherence/results/locality
for arm in global local per_layer both; do
  for ev in own wikitext2; do
    if [ "$ev" = own ]; then cache=$R/$arm/eval_cache.pt; else cache=/work/hdd/bfqt/xiaocong/dagformer_pruning/data/wikitext2/eval_cache.pt; fi
    out=experiments/coherence/results/locality/${arm}_${ev}.json
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "=== $arm $ev"
    python experiments/coherence/cross_layer_patch.py --config configs/locality/75m_$arm.yaml \
      --ckpt $R/$arm/checkpoint_step3000.pt --eval-cache $cache --out $out || echo "FAILED $arm $ev"
  done
done
echo "exit at $(date)"
