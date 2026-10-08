# 75M locality arms: SFT benchmark (lm-eval 0.4.13, reasoning + core suites)

Experiment 1 (configs/locality/75m_*.yaml): 75M models, 3000 steps x 524K tokens = 1.57B tokens of the 1.7B Dolma slice, one arm per routing design, plus the dense OLMo-2 baseline at the same budget. Each arm's final step-3000 checkpoint before SFT ("base", results in ../sft_locality_base/) and after instruction tuning on Alpaca-Dolly (3 epochs) and SmolTalk (2 epochs), same recipe as the other SFT tables (global batch 64x1024, lr 2e-4 cosine, assistant-only loss, mb16x4 for <200M). Single runs, no seeds; GSM8K is 3-shot generation on the first 200 problems. All benchmarks ran on Delta (A100) 2026-10-01; "SFT ran on" says where the fine-tuning itself ran (DONE marker's job: Delta reservation filler, or timan108/timan1), same code and recipe on both.

Accuracy: higher is better; bits-per-byte (gsm8k_bpb, wikitext): lower is better.
Metric per task: arc_challenge/arc_easy/mathqa/openbookqa/hellaswag/piqa = acc_norm; commonsense_qa/social_iqa/winogrande/lambada_openai/sciq/boolq = acc; gsm8k = exact_match flexible-extract.

At this scale most tasks sit near chance (lambada ~0, arc_easy ~0.27), so differences of 0.5-1 point are within single-run noise; boolq swings by up to 20 points between runs of the same arm (majority-class flips) and dominates some MC means, so the 11-task mean without boolq is given as well.

| arm | stage | MC mean (12) | arc_challenge | arc_easy | mathqa | commonsense_qa | social_iqa | openbookqa | winogrande | lambada_openai | hellaswag | piqa | sciq | boolq | gsm8k | gsm8k_bpb | wikitext | SFT ran on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense (OLMo-2 baseline) | base | 0.278 | 0.224 | 0.270 | 0.197 | 0.196 | 0.329 | 0.266 | 0.493 | 0.002 | 0.245 | 0.498 | 0.235 | 0.380 | 0.005 | 2.392 | 1.713 | - |
| dense (OLMo-2 baseline) | +Alpaca-Dolly | 0.288 | 0.235 | 0.274 | 0.195 | 0.196 | 0.330 | 0.246 | 0.490 | 0.012 | 0.253 | 0.522 | 0.301 | 0.409 | 0.010 | 2.091 | 1.649 | timan |
| dense (OLMo-2 baseline) | +SmolTalk | 0.306 | 0.227 | 0.290 | 0.215 | 0.198 | 0.352 | 0.240 | 0.497 | 0.034 | 0.256 | 0.523 | 0.386 | 0.452 | 0.020 | 1.802 | 1.684 | timan |
| both (global + local, shipped DAGFormer) | base | 0.285 | 0.225 | 0.269 | 0.198 | 0.196 | 0.331 | 0.260 | 0.478 | 0.011 | 0.249 | 0.509 | 0.312 | 0.386 | 0.020 | 2.282 | 1.634 | - |
| both (global + local, shipped DAGFormer) | +Alpaca-Dolly | 0.298 | 0.236 | 0.279 | 0.195 | 0.194 | 0.344 | 0.254 | 0.495 | 0.049 | 0.258 | 0.526 | 0.357 | 0.389 | 0.010 | 1.992 | 1.591 | Delta |
| both (global + local, shipped DAGFormer) | +SmolTalk | 0.309 | 0.234 | 0.298 | 0.217 | 0.192 | 0.347 | 0.260 | 0.501 | 0.058 | 0.260 | 0.532 | 0.417 | 0.395 | 0.035 | 1.641 | 1.640 | timan |
| global (shared predictor only) | base | 0.283 | 0.225 | 0.271 | 0.192 | 0.198 | 0.334 | 0.250 | 0.488 | 0.008 | 0.246 | 0.496 | 0.308 | 0.386 | 0.005 | 2.302 | 1.644 | - |
| global (shared predictor only) | +Alpaca-Dolly | 0.297 | 0.240 | 0.287 | 0.200 | 0.194 | 0.348 | 0.234 | 0.509 | 0.041 | 0.256 | 0.514 | 0.355 | 0.384 | 0.005 | 2.013 | 1.603 | Delta |
| global (shared predictor only) | +SmolTalk | 0.321 | 0.221 | 0.292 | 0.228 | 0.200 | 0.341 | 0.252 | 0.489 | 0.054 | 0.260 | 0.523 | 0.457 | 0.533 | 0.030 | 1.693 | 1.661 | Delta |
| per_layer (independent encoder per layer) | base | 0.281 | 0.230 | 0.272 | 0.186 | 0.197 | 0.331 | 0.248 | 0.482 | 0.006 | 0.244 | 0.504 | 0.281 | 0.391 | 0.010 | 2.300 | 1.652 | - |
| per_layer (independent encoder per layer) | +Alpaca-Dolly | 0.297 | 0.235 | 0.282 | 0.194 | 0.197 | 0.351 | 0.246 | 0.493 | 0.038 | 0.254 | 0.523 | 0.373 | 0.379 | 0.010 | 2.015 | 1.608 | Delta |
| per_layer (independent encoder per layer) | +SmolTalk | 0.314 | 0.242 | 0.306 | 0.219 | 0.211 | 0.353 | 0.256 | 0.522 | 0.057 | 0.258 | 0.524 | 0.395 | 0.423 | 0.035 | 1.704 | 1.664 | timan |
| local (per-layer correction only) | base | 0.285 | 0.232 | 0.269 | 0.188 | 0.196 | 0.322 | 0.266 | 0.495 | 0.014 | 0.248 | 0.504 | 0.296 | 0.396 | 0.010 | 2.277 | 1.638 | - |
| local (per-layer correction only) | +Alpaca-Dolly | 0.301 | 0.232 | 0.285 | 0.194 | 0.197 | 0.340 | 0.230 | 0.496 | 0.049 | 0.255 | 0.532 | 0.379 | 0.428 | 0.020 | 2.008 | 1.601 | Delta |
| local (per-layer correction only) | +SmolTalk | 0.321 | 0.217 | 0.283 | 0.217 | 0.196 | 0.347 | 0.248 | 0.481 | 0.060 | 0.262 | 0.524 | 0.402 | 0.621 | 0.015 | 1.668 | 1.640 | timan |
| modular (module-granular) | base | 0.290 | 0.224 | 0.278 | 0.196 | 0.197 | 0.332 | 0.266 | 0.506 | 0.012 | 0.247 | 0.509 | 0.323 | 0.391 | 0.000 | 2.246 | 1.611 | - |
| modular (module-granular) | +Alpaca-Dolly | 0.304 | 0.224 | 0.287 | 0.203 | 0.197 | 0.343 | 0.258 | 0.509 | 0.061 | 0.253 | 0.523 | 0.384 | 0.410 | 0.015 | 1.954 | 1.581 | Delta |
| modular (module-granular) | +SmolTalk | 0.307 | 0.230 | 0.298 | 0.231 | 0.193 | 0.347 | 0.244 | 0.484 | 0.065 | 0.264 | 0.517 | 0.424 | 0.390 | 0.010 | 1.643 | 1.640 | timan |
| modular + sparsity | base | 0.281 | 0.233 | 0.276 | 0.197 | 0.196 | 0.327 | 0.260 | 0.500 | 0.003 | 0.247 | 0.511 | 0.248 | 0.380 | 0.010 | 2.325 | 1.660 | - |
| modular + sparsity | +Alpaca-Dolly | 0.303 | 0.239 | 0.292 | 0.193 | 0.200 | 0.349 | 0.248 | 0.496 | 0.020 | 0.256 | 0.526 | 0.330 | 0.492 | 0.030 | 2.052 | 1.625 | Delta |
| modular + sparsity | +SmolTalk | 0.322 | 0.245 | 0.304 | 0.229 | 0.198 | 0.353 | 0.262 | 0.493 | 0.041 | 0.259 | 0.523 | 0.395 | 0.567 | 0.015 | 1.758 | 1.673 | timan |


## Summary

| arm | MC mean (12) base / +Alpaca / +SmolTalk | MC mean w/o boolq (11) base / +Alpaca / +SmolTalk | wikitext bpb base |
|---|---|---|---|
| dense | 0.278 / 0.288 / 0.306 | 0.269 / 0.278 / 0.293 | 1.713 |
| both | 0.285 / 0.298 / 0.309 | 0.276 / 0.290 / 0.302 | 1.634 |
| global | 0.283 / 0.297 / 0.321 | 0.274 / 0.289 / 0.301 | 1.644 |
| per_layer | 0.281 / 0.297 / 0.314 | 0.271 / 0.290 / 0.304 | 1.652 |
| local | 0.285 / 0.301 / 0.321 | 0.275 / 0.290 / 0.294 | 1.638 |
| modular | 0.290 / 0.304 / 0.307 | 0.281 / 0.295 / 0.300 | 1.611 |
| modular_sparse | 0.281 / 0.303 / 0.322 | 0.273 / 0.286 / 0.300 | 1.660 |

- Every routed arm beats the dense baseline at every stage on the boolq-free mean (base +0.2 to +1.2 points, after SFT +0.1 to +1.7) and on wikitext bits-per-byte before SFT (1.61-1.66 vs 1.71).
- Among the routed arms the spread is small (base 0.271-0.281 w/o boolq). modular is best before SFT and after Alpaca-Dolly and has the lowest wikitext bpb (1.611); after SmolTalk per_layer/both/global/modular are within 0.4 points.
- SFT adds 1-3.5 points of boolq-free MC mean for every arm (SmolTalk more than Alpaca-Dolly) and does not change the ordering against dense.
- 2026-10-01: four of these JSONs (and one 1B file) were written by two eval jobs at once (eval_sft_exports.sh evaluated every export under one shared links dir); the files were repaired by keeping the first complete JSON object, which is one full result of the identical evaluation. eval_sft_exports.sh now uses a per-filter link set.
