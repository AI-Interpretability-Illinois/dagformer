#!/bin/bash
# Benchmark triggers for all DAGFormer SFT groups (each submits its lm-eval job once, when its tasks are DONE).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$HERE/submit_sft_eval.sh" pt_dagformer_300m 6
bash "$HERE/submit_sft_eval.sh" pt_dagformer_300m_seed2 6   # second SFT seed of the 300M triple
for m in 1b_dense_5b 1b_fourway_corrected_5b; do   # second SFT seed of the 1B 5B pair
  bash "$HERE/submit_sft_eval.sh" "pt_dagformer_1b_seed2-${m}_" 2
done
for m in 1b_dense_10b 1b_fourway_corrected_10b 1b_dense_5b 1b_fourway_corrected_5b; do
  bash "$HERE/submit_sft_eval.sh" "pt_dagformer_1b-${m}_" 2
done
exit 0
