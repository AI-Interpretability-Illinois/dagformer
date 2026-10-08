#!/bin/bash
# Second pretraining seed (43) of the 300M dense / fourway_corrected pair on reservation sup-30781.
# 1. rebuild the 21B-token Dolma corpus (deleted 2026-10-02): 1 GPU, 32 CPU, 96 GB, ~2.5-3 h
# 2. dense chain on gpua038 (~7 h) and fourway_corrected chain on gpua047 (~13 h), each starting
#    only after the corpus job succeeded (afterok), each ending with the common eval + refit.
# Run from the repo root:  bash scripts/slurm/submit_pseed2.sh
set -euo pipefail
cd /u/xiaocong/dagformer
mkdir -p logs
# a previous attempt whose corpus job failed leaves its chains pending forever (DependencyNeverSatisfied): cancel them
if [ -f experiments/pretrain_300m/pseed2_jobids.txt ]; then
  for j in $(cat experiments/pretrain_300m/pseed2_jobids.txt); do
    st=$(squeue -h -j "$j" -o %T 2>/dev/null || true)
    [ "$st" = PENDING ] && { echo "cancelling stale pending job $j"; scancel "$j"; }
  done
fi
for i in 0 1 2 3 4 5 6 7; do [ -f logs/pretok21b_w$i.log ] && mv logs/pretok21b_w$i.log logs/pretok21b_w${i}_sep26.log; done
tok=$(sbatch --parsable scripts/slurm/pretokenize_21b.slurm)
echo "pretok21b=$tok"
d=$(sbatch --parsable --reservation=sup-30781 --nodelist=gpua038 --dependency=afterok:$tok \
    --export=ALL,QUEUE=experiments/pretrain_300m/queue_pseed2_dense.txt,STOP=experiments/pretrain_300m/STOP_pseed2_dense,TOKENIZE=0 \
    --job-name=resv_train_pseed2_dense scripts/slurm/reserved_train.slurm)
echo "dense_chain=$d"
c=$(sbatch --parsable --reservation=sup-30781 --nodelist=gpua047 --dependency=afterok:$tok \
    --export=ALL,QUEUE=experiments/pretrain_300m/queue_pseed2_corrected.txt,STOP=experiments/pretrain_300m/STOP_pseed2_corrected,TOKENIZE=0 \
    --job-name=resv_train_pseed2_corrected scripts/slurm/reserved_train.slurm)
echo "corrected_chain=$c"
echo "$tok $d $c" > experiments/pretrain_300m/pseed2_jobids.txt
squeue -u "$USER" -o '%i %j %T %M %N %R' | grep -v -e filler -e dryrun
