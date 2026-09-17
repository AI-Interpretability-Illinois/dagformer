#!/bin/bash
#SBATCH --job-name=interp_bcirc
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --output=logs/interp_bcirc_%j.out
#SBATCH --error=logs/interp_bcirc_%j.err

# Behavioural-circuit pipeline: contrast -> discovery -> causal verification.
# See experiments/interp/README.md.
#
#   sbatch scripts/slurm/interp_behavior_circuit.slurm
#   BEHAVIORS="honesty domain_code" CHANNELS="pred corr" sbatch ...

set -u
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_DIR="${MODEL_DIR:-/work/hdd/bfqt/shared/dagformer-models/300m-dagformer}"
TOKENIZER="${TOKENIZER:-/work/hdd/bfqt/shared/dagformer-models/tokenizer}"
OUT_DIR="${OUT_DIR:-experiments/results/interp/circuits}"

# domain_code runs alongside honesty as a positive control: this repo already
# has evidence the model's routing responds to domain, so if honesty comes back
# null it says whether the method or the behaviour is the problem.
BEHAVIORS="${BEHAVIORS:-honesty honesty_fewshot domain_code}"
CHANNELS="${CHANNELS:-pred corr}"

mkdir -p "$OUT_DIR" logs
rc=0

for b in $BEHAVIORS; do
    echo "=================== contrast: $b ==================="
    python3 experiments/interp/extract_contrast.py \
        --config "$MODEL_DIR/config.yaml" \
        --ckpt   "$MODEL_DIR/checkpoint.pt" \
        --tokenizer "$TOKENIZER" \
        --behavior "$b" --channel both --out-dir "$OUT_DIR" || { rc=$?; continue; }

    for c in $CHANNELS; do
        echo "=================== discover: $b / $c ==================="
        python3 experiments/interp/discover_circuit.py \
            --behavior "$b" --channel "$c" \
            --in-dir "$OUT_DIR" --out-dir "$OUT_DIR" \
            --select null --q 0.005 --hyper-only || { rc=$?; continue; }

        echo "=================== verify: $b / $c ==================="
        python3 experiments/interp/verify_circuit.py \
            --config "$MODEL_DIR/config.yaml" \
            --ckpt   "$MODEL_DIR/checkpoint.pt" \
            --tokenizer "$TOKENIZER" \
            --behavior "$b" --channel "$c" \
            --in-dir "$OUT_DIR" --out-dir "$OUT_DIR" || rc=$?

        python3 experiments/interp/plot_circuit.py \
            --behavior "$b" --channel "$c" --dir "$OUT_DIR" \
            || echo "[skip] figures (matplotlib unavailable)"
    done
done

echo "=================== verdicts ==================="
grep -H -A2 '^## Verdict' "$OUT_DIR"/verify_*.md 2>/dev/null || true

echo "interp_behavior_circuit exited rc=$rc"
exit "$rc"
