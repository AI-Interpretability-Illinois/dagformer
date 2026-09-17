#!/bin/bash
#SBATCH --job-name=dag-scale-eval
#SBATCH --account=biro-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/eval_%x_%j.log
#SBATCH --error=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/eval_%x_%j.log
set -euo pipefail
source "${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}/scripts/biro_job_common.sh"
DAG_MODEL=${DAG_MODEL:?Choose a completed model}
DAG_EVAL_ENV=$DAG_ROOT/envs/evaluation
(
    flock 8
    if [ ! -f "$DAG_EVAL_ENV/.ready" ]; then
        "$DAG_PYTHON" -m venv --system-site-packages "$DAG_EVAL_ENV"
        "$DAG_EVAL_ENV/bin/python" -m pip install 'lm_eval==0.4.13' 'transformers==4.57.1'
        touch "$DAG_EVAL_ENV/.ready"
    fi
) 8>"$DAG_ROOT/.eval-env.lock"
DAG_PYTHON=$DAG_EVAL_ENV/bin/python
DAG_OUT=$DAG_ROOT/results/completed_scaling
mkdir -p "$DAG_OUT"
dag_copy_compact_results() {
    mkdir -p "$DAG_SOURCE/experiments/results/completed_scaling_20260917"
    rsync -a --exclude samples/ "$DAG_OUT/" \
        "$DAG_SOURCE/experiments/results/completed_scaling_20260917/"
}
trap dag_copy_compact_results EXIT
DAG_SIZE=${DAG_MODEL%%-*}
DAG_MODELS=$DAG_ROOT/checkpoints/eval
DAG_TASKS=wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,sciq,boolq,mathqa,commonsense_qa,social_iqa,gsm8k_bpb
DAG_COMMON=(--models-root "$DAG_MODELS" --model "$DAG_MODEL" --tokenizer "$DAG_ROOT/tokenizer"
    --batch-size 4 --max-length 1024 --softmax-dtype float32 --fewshot-seed 1234 --save-samples)
"$DAG_PYTHON" experiments/results/lmeval/run_eval.py "${DAG_COMMON[@]}" \
    --tasks "$DAG_TASKS" --save-likelihood-samples --out-dir "$DAG_OUT/standard_$DAG_SIZE"
dag_copy_compact_results
if [[ "$DAG_MODEL" == *dagformer ]]; then
    "$DAG_PYTHON" scripts/eval_routing_dependence.py \
        --config "$DAG_MODELS/$DAG_MODEL/config.yaml" --ckpt "$DAG_MODELS/$DAG_MODEL/checkpoint.pt" \
        --calibration-cache "$DAG_ROOT/data/eval_corpora/wikitext_train.pt" \
        --eval-cache "$DAG_ROOT/data/eval_corpora/wikitext_test.pt" \
        --out "$DAG_OUT/routing_dependence/$DAG_SIZE.json" \
        --table-out "$DAG_OUT/routing_dependence/${DAG_SIZE}_tables.pt" \
        --n-calibration 64 --n-eval 128 --seed 20260917
    "$DAG_PYTHON" scripts/eval_dense_routing_reference.py \
        --config "$DAG_MODELS/$DAG_SIZE-baseline/config.yaml" \
        --ckpt "$DAG_MODELS/$DAG_SIZE-baseline/checkpoint.pt" \
        --routing-result "$DAG_OUT/routing_dependence/$DAG_SIZE.json" \
        --kind dense --out "$DAG_OUT/routing_dependence/dense_$DAG_SIZE.json"
    "$DAG_PYTHON" experiments/results/lmeval/run_eval.py "${DAG_COMMON[@]}" \
        --tasks "$DAG_TASKS" --save-likelihood-samples --routing-intervention pred_position \
        --routing-table "$DAG_OUT/routing_dependence/${DAG_SIZE}_tables.pt" \
        --out-dir "$DAG_OUT/frozen_predictor_$DAG_SIZE"
    dag_copy_compact_results
fi
"$DAG_PYTHON" experiments/results/lmeval/run_eval.py "${DAG_COMMON[@]}" \
    --tasks gsm8k --gen-batch-size 1 --gen-limit 1319 --out-dir "$DAG_OUT/gsm8k_full_$DAG_SIZE"
