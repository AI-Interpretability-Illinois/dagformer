#!/bin/bash
#SBATCH --job-name=topology
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --reservation=sup-30781
#SBATCH --nodelist=gpua047
#SBATCH --output=logs/topology_%j.out
# J1/J3/J4: static-topology substitution, topology dump and inference-time edge pruning for the
# trained FourWay models (scripts/topology_analysis.py). Outputs: experiments/topology/<name>_*.{npz,json}.
#   MODELS="300m_corrected 1b_corrected_5b" sbatch scripts/slurm/topology_analysis.slurm
cd /u/xiaocong/dagformer
source ~/.bashrc; conda activate modularity 2>/dev/null || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=/work/hdd/bfqt/xiaocong/dagformer_pruning/data
L=/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints/locality
EV="--eval wikitext2=$D/wikitext2/eval_cache.pt --eval mathinstruct=$D/mathinstruct/eval_cache.pt --eval gsm8k=$D/gsm8k/eval_cache.pt"
mkdir -p experiments/topology
for m in ${MODELS:-300m_corrected 1b_corrected_5b}; do
  case $m in
    300m_corrected) cfg=configs/pretrain_300m/300m_fourway_corrected_mb4.yaml; ck=/work/hdd/bfqt/xiaocong/dagformer_300m/300m_fourway_corrected/checkpoint_step12000.pt; own=$D/dolma_v1_7_21b/eval_cache.pt; extra="--eval dolma21b=$own";;
    1b_corrected_5b) cfg=configs/pretrain_1b/1b_fourway_corrected_5b_mb2.yaml; ck=/work/hdd/bfqt/xiaocong/dagformer_1b/1b_fourway_corrected_5b/checkpoint_step9540.pt; own=$D/dolma_v1_7_21b/eval_cache.pt; extra="--eval dolma21b=$own";;
    1b_corrected_10b) cfg=configs/pretrain_1b/1b_fourway_corrected_10b_mb2.yaml; ck=/work/hdd/bfqt/xiaocong/dagformer_1b/1b_fourway_corrected_10b/checkpoint_step19080.pt; own=$D/dolma_v1_7_21b/eval_cache.pt; extra="--eval dolma21b=$own";;
    shared_75m) cfg=/work/hdd/bfqt/shared/dagformer-models/75m-dagformer/config.yaml; ck=/work/hdd/bfqt/shared/dagformer-models/75m-dagformer/checkpoint.pt; own=$D/wikitext2/eval_cache.pt; extra="";;
    shared_150m) cfg=/work/hdd/bfqt/shared/dagformer-models/150m-dagformer/config.yaml; ck=/work/hdd/bfqt/shared/dagformer-models/150m-dagformer/checkpoint.pt; own=$D/wikitext2/eval_cache.pt; extra="";;
    loc_global|loc_both|loc_local|loc_per_layer) arm=${m#loc_}; cfg=configs/locality/75m_$arm.yaml; ck=$L/$arm/checkpoint_step3000.pt; own=$L/$arm/eval_cache.pt; extra="--eval own=$own";;
    *) echo "unknown model $m"; continue;;
  esac
  [ -f experiments/topology/${m}_analysis.json ] && { echo "skip (done): $m"; continue; }
  echo "=== $m"
  python scripts/topology_analysis.py --config $cfg --ckpt $ck --own-cache $own $EV $extra --out experiments/topology/$m || echo "FAILED $m"
done
echo "exit at $(date)"
