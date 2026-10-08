# 1B benchmark before and after SFT (lm-eval 0.4.13, reasoning + core suites)

Final checkpoints of the 1B pretraining runs (Delta 21B corpus; dense OLMo-2 baseline and DAGFormer four-way corrected, each at 5B and 10B tokens) after instruction tuning on Alpaca-Dolly (3 epochs) and SmolTalk (1 epoch, configs/sft/tasks.yaml large_model) with the 1B recipe (global batch 64x1024, lr 1e-4 cosine, mb4x16, 4-GPU DDP on the Delta reservation), assistant-only loss. Single SFT runs in the main table (a second SFT seed for the 5B pair is at the end); GSM8K is 3-shot generation on the first 200 problems. Benchmarks ran on Delta 2026-09-30/10-01 (10B DAGFormer rows 2026-10-03). The base (pre-SFT) rows are the same final pretraining checkpoints (steps 9540 for 5B, 19080 for 10B) evaluated with identical settings on 2026-10-01/03 (results in ../pt_1b_base/).

Accuracy: higher is better; bits-per-byte (gsm8k_bpb, wikitext): lower is better. Metrics per task as in SUMMARY_locality.md.

| model | tokens | stage | MC mean (12) | MC mean w/o boolq (11) | arc_challenge | arc_easy | mathqa | commonsense_qa | social_iqa | openbookqa | winogrande | lambada_openai | hellaswag | piqa | sciq | boolq | gsm8k | gsm8k_bpb | wikitext |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 5B | base (pre-SFT) | 0.385 | 0.365 | 0.222 | 0.397 | 0.222 | 0.205 | 0.374 | 0.260 | 0.515 | 0.241 | 0.292 | 0.616 | 0.675 | 0.601 | 0.020 | 1.375 | 1.058 |
| dense | 5B | +Alpaca-Dolly | 0.385 | 0.364 | 0.241 | 0.402 | 0.219 | 0.185 | 0.381 | 0.270 | 0.507 | 0.255 | 0.294 | 0.619 | 0.631 | 0.618 | 0.015 | 1.348 | 1.090 |
| dense | 5B | +SmolTalk | 0.377 | 0.355 | 0.225 | 0.382 | 0.237 | 0.208 | 0.378 | 0.266 | 0.504 | 0.188 | 0.293 | 0.621 | 0.604 | 0.621 | 0.010 | 1.202 | 1.150 |
| DAGFormer four-way corrected | 5B | base (pre-SFT) | 0.411 | 0.395 | 0.240 | 0.421 | 0.228 | 0.229 | 0.393 | 0.266 | 0.510 | 0.305 | 0.321 | 0.637 | 0.793 | 0.589 | 0.020 | 1.242 | 0.990 |
| DAGFormer four-way corrected | 5B | +Alpaca-Dolly | 0.410 | 0.397 | 0.241 | 0.420 | 0.224 | 0.224 | 0.400 | 0.294 | 0.506 | 0.338 | 0.322 | 0.633 | 0.764 | 0.560 | 0.015 | 1.200 | 1.021 |
| DAGFormer four-way corrected | 5B | +SmolTalk | 0.402 | 0.383 | 0.228 | 0.411 | 0.230 | 0.206 | 0.380 | 0.278 | 0.506 | 0.271 | 0.320 | 0.634 | 0.752 | 0.611 | 0.015 | 1.038 | 1.067 |
| dense | 10B | base (pre-SFT) | 0.395 | 0.377 | 0.235 | 0.415 | 0.220 | 0.190 | 0.384 | 0.264 | 0.506 | 0.264 | 0.305 | 0.638 | 0.726 | 0.595 | 0.010 | 1.323 | 1.022 |
| dense | 10B | +Alpaca-Dolly | 0.394 | 0.374 | 0.239 | 0.406 | 0.222 | 0.201 | 0.382 | 0.270 | 0.510 | 0.278 | 0.305 | 0.628 | 0.673 | 0.618 | 0.045 | 1.290 | 1.052 |
| dense | 10B | +SmolTalk | 0.388 | 0.369 | 0.224 | 0.400 | 0.239 | 0.215 | 0.385 | 0.276 | 0.500 | 0.214 | 0.306 | 0.625 | 0.670 | 0.596 | 0.035 | 1.153 | 1.106 |
| DAGFormer four-way corrected | 10B | base (pre-SFT) | 0.421 | 0.404 | 0.242 | 0.434 | 0.231 | 0.215 | 0.396 | 0.278 | 0.519 | 0.333 | 0.337 | 0.651 | 0.810 | 0.606 | 0.025 | 1.239 | 0.963 |
| DAGFormer four-way corrected | 10B | +Alpaca-Dolly | 0.421 | 0.404 | 0.255 | 0.434 | 0.225 | 0.222 | 0.388 | 0.286 | 0.513 | 0.374 | 0.341 | 0.656 | 0.753 | 0.603 | 0.010 | 1.182 | 0.993 |
| DAGFormer four-way corrected | 10B | +SmolTalk | 0.407 | 0.388 | 0.233 | 0.418 | 0.233 | 0.179 | 0.390 | 0.280 | 0.511 | 0.299 | 0.330 | 0.643 | 0.754 | 0.611 | 0.015 | 1.020 | 1.036 |

- The DAGFormer advantage is already there before SFT: at 5B tokens base DAGFormer vs base dense is MC mean 0.411 vs 0.385 (+2.6 points; +3.0 without boolq), lambada +6.4, sciq +11.8, hellaswag +2.9, wikitext bpb 0.990 vs 1.058. Base DAGFormer at 5B also beats base dense at 10B (MC mean 0.411 vs 0.395; wikitext bpb 0.990 vs 1.022).
- SFT does not raise the multiple-choice benchmarks at 1B: Alpaca-Dolly leaves the MC mean unchanged (within 0.001 for all four models) and SmolTalk lowers it by 0.8-0.9 points (1.4 for DAGFormer 10B). It costs raw-text modelling (wikitext bpb +0.03 Alpaca, +0.07-0.09 SmolTalk) and lowers sciq (-3 to -7 points), while gsm8k_bpb improves (answer-format bits; 1.375 -> 1.348/1.202 for dense 5B, 1.242 -> 1.200/1.038 for DAGFormer 5B, 1.239 -> 1.182/1.020 for DAGFormer 10B). The ranking DAGFormer 10B > DAGFormer 5B > dense 10B > dense 5B holds at every stage.
- At equal tokens (5B) DAGFormer beats dense after both SFT sets: MC mean +2.5 points (both sets), +3.3/+2.8 without boolq; lambada +8.3/+8.3, sciq +13.3/+14.8, hellaswag +2.8/+2.7; wikitext bpb 1.021 vs 1.090 (Alpaca) and 1.067 vs 1.150 (SmolTalk).
- DAGFormer at 5B tokens also beats the dense model trained on twice the tokens (10B): MC mean 0.410 vs 0.394 (Alpaca) and 0.402 vs 0.388 (SmolTalk); wikitext bpb 1.021 vs 1.052 and 1.067 vs 1.106.
- At 10B tokens the gap is the same as at 5B: base DAGFormer vs base dense MC mean 0.421 vs 0.395 (+2.6; +2.7 without boolq), lambada +6.9, sciq +8.4, hellaswag +3.2, wikitext bpb 0.963 vs 1.022. After SFT: +2.7 (Alpaca) and +1.9 (SmolTalk) MC points (+3.0/+1.9 without boolq); lambada +9.6/+8.5, sciq +8.0/+8.4, hellaswag +3.6/+2.4; wikitext bpb 0.993 vs 1.052 and 1.036 vs 1.106. Doubling the tokens adds 1.0 MC point to both base models (dense 0.385 -> 0.395, DAGFormer 0.411 -> 0.421), so the routed advantage does not shrink from 5B to 10B.
- GSM8K is at floor for all 1B models (1-4.5% on 200 problems).
- Judged chat quality (MT-Bench first turns, Qwen3-Coder-30B judge, pairwise both orders; ../../../chat_eval/RESULTS.md): DAGFormer 5B beats dense 5B with win rate 0.58 (Alpaca) / 0.59 (SmolTalk) and dense 10B with 0.56 / 0.60; the 95% CI excludes 0.5 except vs dense 10B after Alpaca (0.478-0.641). Absolute judge scores are low for all 1B models (about 2 of 10). At equal tokens (10B), DAGFormer vs dense has win rate 0.56 [0.48, 0.63] after Alpaca (W/T/L 19/47/14; CI includes 0.5) and 0.64 [0.57, 0.72] after SmolTalk (30/41/9); paired single-score difference +0.14 [-0.15, +0.41] and +0.40 [+0.23, +0.59] (2026-10-03).

## Second SFT seed, 5B pair (2026-10-03)

Same pretrained checkpoints (step 9540) and recipe, SFT repeated with seed 2 instead of 1234 (different data order and RNG); runs `pt_dagformer_1b_seed2-*`, benchmarked with the same settings (jobs 22639239, 22639664). Cells are mean ± half the difference between the two SFT seeds.

| model | SFT | MC mean (12) | MC w/o boolq (11) | lambada | sciq | hellaswag | boolq | gsm8k | wikitext bpb |
|---|---|---|---|---|---|---|---|---|---|
| dense 5B | +Alpaca-Dolly | 0.385 ± 0.000 | 0.364 ± 0.000 | 0.254 ± 0.002 | 0.630 ± 0.001 | 0.294 ± 0.000 | 0.618 ± 0.000 | 0.018 ± 0.003 | 1.090 ± 0.000 |
| dense 5B | +SmolTalk | 0.377 ± 0.001 | 0.355 ± 0.001 | 0.187 ± 0.001 | 0.601 ± 0.003 | 0.293 ± 0.000 | 0.619 ± 0.002 | 0.013 ± 0.002 | 1.150 ± 0.000 |
| DAGFormer four-way corrected 5B | +Alpaca-Dolly | 0.410 ± 0.001 | 0.396 ± 0.000 | 0.336 ± 0.002 | 0.764 ± 0.000 | 0.321 ± 0.001 | 0.557 ± 0.003 | 0.015 ± 0.000 | 1.021 ± 0.000 |
| DAGFormer four-way corrected 5B | +SmolTalk | 0.403 ± 0.001 | 0.384 ± 0.001 | 0.270 ± 0.001 | 0.760 ± 0.008 | 0.320 ± 0.000 | 0.609 ± 0.002 | 0.015 ± 0.000 | 1.067 ± 0.000 |

| routed − dense | SFT | ΔMC (12) seed 1 / seed 2 | ΔMC w/o boolq seed 1 / seed 2 | Δwikitext bpb seed 1 / seed 2 |
|---|---|---|---|---|
| DAGFormer four-way corrected 5B | +Alpaca-Dolly | +0.025 / +0.025 | +0.033 / +0.033 | -0.069 / -0.069 |
| DAGFormer four-way corrected 5B | +SmolTalk | +0.025 / +0.027 | +0.028 / +0.031 | -0.083 / -0.083 |

- SFT-seed variance is as small as at 300M: half the seed-to-seed difference is at most 0.001 for both MC means and under 0.001 for wikitext bpb; the noisiest tasks are sciq (up to ±0.008) and boolq (±0.003).
- The DAGFormer-vs-dense gaps at 5B tokens reproduce across seeds within 0.003: +2.5/+2.5 (Alpaca) and +2.5/+2.7 (SmolTalk) MC points, -0.069 and -0.083 wikitext bpb. The 10B rows were not repeated.
- As at 300M, this covers SFT-seed variance only; each model has a single pretraining run.
