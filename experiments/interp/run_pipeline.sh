#!/bin/bash
# End-to-end behavioural-circuit pipeline.
#
#   bash experiments/interp/run_pipeline.sh [behavior] [channel] [model_dir]
#
# Defaults to the honesty contrast on the predictor channel of the shared 300M
# DAGFormer checkpoint.  Override the model with a third argument or by setting
# MODEL_DIR; extra discovery flags go in DISCOVER_ARGS, extra verification
# flags in VERIFY_ARGS.
set -euo pipefail

BEHAVIOR="${1:-honesty}"
CHANNEL="${2:-pred}"
MODEL_DIR="${3:-${MODEL_DIR:-/work/hdd/bfqt/shared/dagformer-models/300m-dagformer}}"
TOKENIZER="${TOKENIZER:-/work/hdd/bfqt/shared/dagformer-models/tokenizer}"
OUT_DIR="${OUT_DIR:-experiments/results/interp/circuits}"

CONFIG="$MODEL_DIR/config.yaml"
CKPT="$MODEL_DIR/checkpoint.pt"
[ -f "$CONFIG" ] || { echo "no config at $CONFIG" >&2; exit 1; }
[ -f "$CKPT" ]   || { echo "no checkpoint at $CKPT" >&2; exit 1; }

# extract_contrast records whichever channels discovery might want; 'eff'
# needs both, and 'both' is what makes the eff channel available later.
EXTRACT_CHANNEL="both"
[ "$CHANNEL" = "pred" ] && EXTRACT_CHANNEL="${EXTRACT_CHANNEL_OVERRIDE:-both}"

: "${DISCOVER_ARGS:=--select null --q 0.005 --hyper-only}"
: "${VERIFY_ARGS:=}"

export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/../.."
mkdir -p "$OUT_DIR"

echo "=== [1/4] contrast: $BEHAVIOR on $MODEL_DIR ==="
python experiments/interp/extract_contrast.py \
    --config "$CONFIG" --ckpt "$CKPT" --tokenizer "$TOKENIZER" \
    --behavior "$BEHAVIOR" --channel "$EXTRACT_CHANNEL" --out-dir "$OUT_DIR"

echo "=== [2/4] discover: channel $CHANNEL ==="
# shellcheck disable=SC2086
python experiments/interp/discover_circuit.py \
    --behavior "$BEHAVIOR" --channel "$CHANNEL" \
    --in-dir "$OUT_DIR" --out-dir "$OUT_DIR" $DISCOVER_ARGS

echo "=== [3/4] verify ==="
# shellcheck disable=SC2086
python experiments/interp/verify_circuit.py \
    --config "$CONFIG" --ckpt "$CKPT" --tokenizer "$TOKENIZER" \
    --behavior "$BEHAVIOR" --channel "$CHANNEL" \
    --in-dir "$OUT_DIR" --out-dir "$OUT_DIR" $VERIFY_ARGS

echo "=== [4/4] figures (optional) ==="
python experiments/interp/plot_circuit.py \
    --behavior "$BEHAVIOR" --channel "$CHANNEL" --dir "$OUT_DIR" \
    || echo "[skip] no matplotlib; the .md and .dot reports are the primary output"

echo
echo "=== reports ==="
sed -n '1,40p' "$OUT_DIR/verify_${BEHAVIOR}_${CHANNEL}.md"
echo
echo "full output in $OUT_DIR"
