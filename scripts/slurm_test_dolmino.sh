#!/bin/bash
#SBATCH --job-name=test-dolmino
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_dolmino_%j.out
#SBATCH --error=logs/test_dolmino_%j.err

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

echo "=== Test Dolmino mix interleave ==="
python3 -u -c "
from transformers import AutoTokenizer
from src.data.dolma import DolmaPackedDataset
import time

tok = AutoTokenizer.from_pretrained('allenai/OLMo-2-0425-1B')

print('Testing dolmino_mix interleave...')
t0 = time.time()
ds = DolmaPackedDataset(
    olmo_tokenizer=tok,
    seq_len=1024,
    dataset_name='allenai/dolmino-mix-1124',
    dataset_version='dolmino_mix',
    max_samples=20,
)
for i, sample in enumerate(ds):
    elapsed = time.time() - t0
    print(f'Sample {i}: shape={sample[\"olmo_ids\"].shape}, text[:60]={sample[\"raw_text\"][:60]}, t={elapsed:.1f}s')
print(f'Done in {time.time()-t0:.1f}s')
"
