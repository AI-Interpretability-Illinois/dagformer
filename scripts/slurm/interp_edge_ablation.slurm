#!/bin/bash
#SBATCH --job-name=interp_ablate
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/interp_ablate_%j.out
#SBATCH --error=logs/interp_ablate_%j.err

# Routing-edge ablation: which pathways carry the computation?
# The description half of the interpretability claim -- it does not need alpha
# to be a good steering lever, only a good basis. See experiments/interp/README.md.
#
#   sbatch scripts/slurm/interp_edge_ablation.slurm
#   PARTS="channel" N_WINDOWS=32 sbatch scripts/slurm/interp_edge_ablation.slurm

set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/experiments/interp:${PYTHONPATH:-}"

MODEL_DIR="${MODEL_DIR:-/work/hdd/bfqt/shared/dagformer-models/300m-dagformer}"
TOKENIZER="${TOKENIZER:-/work/hdd/bfqt/shared/dagformer-models/tokenizer}"
OUT_DIR="${OUT_DIR:-experiments/results/interp/circuits}"
PARTS="${PARTS:-channel attribute circuit}"
N_WINDOWS="${N_WINDOWS:-16}"
SEQ_LEN="${SEQ_LEN:-256}"

mkdir -p "$OUT_DIR" logs
rc=0

common=(--config "$MODEL_DIR/config.yaml" --ckpt "$MODEL_DIR/checkpoint.pt"
        --tokenizer "$TOKENIZER" --n-windows "$N_WINDOWS" --seq-len "$SEQ_LEN")

for p in $PARTS; do
    echo "=================== edge_ablation: $p ==================="
    extra=()
    # The circuit part reuses the attribution the previous part saved, so the
    # expensive backward pass is not repeated.
    if [ "$p" = "circuit" ] && [ -f "$OUT_DIR/ablate_attribute.npz" ]; then
        extra=(--attribution "$OUT_DIR/ablate_attribute.npz")
    fi
    python3 experiments/interp/edge_ablation.py --part "$p" \
        "${common[@]}" "${extra[@]}" \
        --out "$OUT_DIR/ablate_$p" || rc=$?
done

echo "=================== summary ==================="
for p in $PARTS; do
    [ -f "$OUT_DIR/ablate_$p.md" ] && { echo "--- $p ---"; cat "$OUT_DIR/ablate_$p.md"; }
done

echo "interp_edge_ablation exited rc=$rc"
exit "$rc"
