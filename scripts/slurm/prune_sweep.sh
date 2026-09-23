#!/bin/bash
# Submit the matched baseline-vs-DAGFormer pruning sweep on the math domain.
#   bash scripts/slurm/prune_sweep.sh            # 75m + 150m, sparsity 0.3/0.5/0.7
#   SIZES="300m" SPARSITIES="0.5" bash scripts/slurm/prune_sweep.sh
#   UNITS=modules bash scripts/slurm/prune_sweep.sh   # whole attn/mlp blocks instead
set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
SIZES="${SIZES:-75m 150m}"
FAMILIES="${FAMILIES:-baseline dagformer}"
SPARSITIES="${SPARSITIES:-0.3 0.5 0.7}"
UNITS="${UNITS:-heads_neurons}"       # heads_neurons | modules
IMPORTANCE="${IMPORTANCE:-taylor}"
EXTRA="${EXTRA:-}"
CKPT_ROOT=/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints
for size in $SIZES; do for fam in $FAMILIES; do for s in $SPARSITIES; do
    tag="s$(python3 -c "print(int(round($s*100)))")"
    ov="target_sparsity=$s importance=$IMPORTANCE"
    if [ "$UNITS" = "modules" ]; then
        ov="$ov prune_units=[attn,mlp] min_alive_per_layer={} "
        tag="mod_$tag"
    fi
    [ "$IMPORTANCE" != "taylor" ] && tag="${tag}_$IMPORTANCE"
    run="${size}_${fam}_math_${tag}"
    ov="$ov save_dir=$CKPT_ROOT/$run wandb_run_name=prune-$run $EXTRA"
    echo "submit $run: $ov"
    CONFIG=configs/prune/${size}_${fam}_math.yaml OVERRIDES="$ov" \
        sbatch --job-name="prune_$run" scripts/slurm/prune_finetune.slurm
done; done; done
