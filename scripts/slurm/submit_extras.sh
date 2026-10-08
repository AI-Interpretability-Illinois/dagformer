#!/bin/bash
# Post-hoc NAACL extras on Delta (plan: /u/xiaocong/gpu_coord/naacl_extras_plan_2026-10-04.md, section "dagformer").
#   * MUDDFormer 300M on the 21B corpus: 4-GPU chain on gpua047, after the seed-43 corrected run (22647481).
#   * 150M D1/D2 on gpua038 (two GPUs at a time, pinned so gpua047 stays whole for the 300M chain):
#       MUDDFormer and width-matched dense first (after the seed-43 dense run 22647480), then DenseFormer and
#       hyper-connections as those two finish; a 1-GPU evaluation job runs the common eval + fits at the end.
# Run from the repo root:  bash scripts/slurm/submit_extras.sh
set -euo pipefail
cd /u/xiaocong/dagformer
mkdir -p logs experiments/extras
R="--reservation=sup-30781"
EV=/work/hdd/bfqt/xiaocong/dagformer_extras/eval_cache_150m_timan12b.pt
sub1() {  # name trainer config dependency
  sbatch --parsable $R --nodelist=gpua038 --dependency="afterany:$4" --job-name="extras_$1" \
    --export=ALL,CFG="$3",TRAINER="$2",EVAL="$EV" scripts/slurm/extras_1gpu.slurm
}
mudd300=$(sbatch --parsable $R --nodelist=gpua047 --dependency=afterany:22647481 \
  --export=ALL,QUEUE=experiments/extras/queue_mudd300m.txt,STOP=experiments/extras/STOP_mudd300m,TOKENIZE=0 \
  --job-name=resv_train_mudd300m scripts/slurm/reserved_train.slurm)
echo "mudd300m_chain=$mudd300"
j1=$(sub1 150m_muddformer scripts/pretrain_muddformer.py configs/extras/150m_muddformer_delta_dolma12bp.yaml 22647480)
j2=$(sub1 150m_dense_w864 scripts/pretrain_baseline.py configs/extras/150m_dense_w864_delta_dolma12bp.yaml 22647480)
j3=$(sub1 150m_denseformer scripts/pretrain_denseformer.py configs/extras/150m_denseformer_delta_dolma12bp.yaml "$j1")
j4=$(sub1 150m_hyperconnection scripts/pretrain_hyperconnection.py configs/extras/150m_hyperconnection_delta_dolma12bp.yaml "$j2")
ev=$(sbatch --parsable $R --dependency="afterany:$j3:$j4" --job-name=extras_eval150m scripts/slurm/extras_eval.slurm)
echo "150m_muddformer=$j1 150m_dense_w864=$j2 150m_denseformer=$j3 150m_hyperconnection=$j4 eval150m=$ev"
echo "$mudd300 $j1 $j2 $j3 $j4 $ev" > experiments/extras/jobids_delta.txt
squeue -u "$USER" -o '%i %j %T %M %N %R' | grep -e extras -e mudd300m
