# 300M DAGFormer SFT benchmark (lm-eval 0.4.13, reasoning + core suites)

Pretrained step-12000 checkpoints of the three 300M runs (6.3B tokens, Delta 21B corpus) before SFT and after instruction tuning on Alpaca-Dolly (3 epochs) and SmolTalk (2 epochs); same recipe for all (global batch 64x1024, lr 2e-4 cosine, assistant-only loss). Single SFT runs in the main table (a second SFT seed is at the end); GSM8K is 3-shot generation on the first 200 problems. Accuracy: higher is better; bits-per-byte (gsm8k_bpb, wikitext): lower is better. Produced 2026-09-30 on timan1 (same lm-eval/torch/transformers versions as Delta).

| model | stage | MC mean (12) | arc_challenge | arc_easy | mathqa | commonsense_qa | social_iqa | openbookqa | winogrande | lambada_openai | hellaswag | piqa | sciq | boolq | gsm8k | gsm8k_bpb | wikitext |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | base | 0.369 | 0.206 | 0.381 | 0.224 | 0.205 | 0.377 | 0.246 | 0.522 | 0.195 | 0.274 | 0.612 | 0.625 | 0.567 | 0.010 | 1.458 | 1.128 |
| dense | +Alpaca-Dolly | 0.371 | 0.218 | 0.371 | 0.226 | 0.200 | 0.374 | 0.256 | 0.537 | 0.208 | 0.277 | 0.606 | 0.593 | 0.591 | 0.025 | 1.452 | 1.184 |
| dense | +SmolTalk | 0.364 | 0.223 | 0.361 | 0.235 | 0.196 | 0.372 | 0.276 | 0.510 | 0.142 | 0.276 | 0.583 | 0.571 | 0.619 | 0.020 | 1.269 | 1.309 |
| four-way corrected | base | 0.391 | 0.221 | 0.388 | 0.231 | 0.196 | 0.378 | 0.244 | 0.523 | 0.265 | 0.282 | 0.617 | 0.737 | 0.604 | 0.025 | 1.344 | 1.057 |
| four-way corrected | +Alpaca-Dolly | 0.390 | 0.234 | 0.380 | 0.226 | 0.195 | 0.387 | 0.272 | 0.524 | 0.287 | 0.287 | 0.619 | 0.656 | 0.618 | 0.020 | 1.295 | 1.108 |
| four-way corrected | +SmolTalk | 0.377 | 0.231 | 0.375 | 0.232 | 0.197 | 0.375 | 0.276 | 0.526 | 0.223 | 0.290 | 0.598 | 0.682 | 0.524 | 0.005 | 1.097 | 1.207 |
| four-way modular | base | 0.389 | 0.214 | 0.397 | 0.223 | 0.197 | 0.372 | 0.274 | 0.519 | 0.254 | 0.285 | 0.624 | 0.697 | 0.612 | 0.015 | 1.393 | 1.078 |
| four-way modular | +Alpaca-Dolly | 0.391 | 0.216 | 0.390 | 0.227 | 0.214 | 0.389 | 0.290 | 0.523 | 0.272 | 0.290 | 0.625 | 0.651 | 0.605 | 0.030 | 1.374 | 1.136 |
| four-way modular | +SmolTalk | 0.378 | 0.221 | 0.375 | 0.242 | 0.193 | 0.371 | 0.276 | 0.525 | 0.198 | 0.287 | 0.606 | 0.657 | 0.584 | 0.015 | 1.149 | 1.242 |

Metric per task: arc_challenge=acc_norm,none, arc_easy=acc_norm,none, mathqa=acc_norm,none, commonsense_qa=acc,none, social_iqa=acc,none, openbookqa=acc_norm,none, winogrande=acc,none, lambada_openai=acc,none, hellaswag=acc_norm,none, piqa=acc_norm,none, sciq=acc,none, boolq=acc,none, gsm8k=exact_match,flexible-extract, gsm8k_bpb=bits_per_byte,none, wikitext=bits_per_byte,none

- Judged chat quality (MT-Bench first turns, Qwen3-Coder-30B judge, pairwise both orders; ../../../chat_eval/RESULTS.md): 300M corrected beats dense with win rate 0.57 (Alpaca) / 0.69 (SmolTalk), 300M modular 0.59 / 0.58; all four 95% CIs are at or above 0.5.

## Second SFT seed (2026-10-02)

Same pretrained checkpoints and recipe, SFT repeated with seed 2 instead of 1234 (different data order and RNG); runs `pt_dagformer_300m_seed2-*`, benchmarked with the same settings (job 22631594). Cells are mean ± half the difference between the two SFT seeds.

| model | SFT | MC mean (12) | MC w/o boolq (11) | lambada | sciq | hellaswag | boolq | gsm8k | wikitext bpb |
|---|---|---|---|---|---|---|---|---|---|
| dense | +Alpaca-Dolly | 0.371 ± 0.000 | 0.351 ± 0.001 | 0.208 ± 0.000 | 0.593 ± 0.001 | 0.277 ± 0.000 | 0.591 ± 0.000 | 0.022 ± 0.003 | 1.184 ± 0.000 |
| dense | +SmolTalk | 0.364 ± 0.000 | 0.341 ± 0.000 | 0.141 ± 0.001 | 0.571 ± 0.001 | 0.274 ± 0.002 | 0.620 ± 0.001 | 0.015 ± 0.005 | 1.309 ± 0.000 |
| four-way corrected | +Alpaca-Dolly | 0.389 ± 0.001 | 0.368 ± 0.001 | 0.287 ± 0.001 | 0.650 ± 0.006 | 0.287 ± 0.000 | 0.618 ± 0.000 | 0.025 ± 0.005 | 1.108 ± 0.000 |
| four-way corrected | +SmolTalk | 0.377 ± 0.000 | 0.366 ± 0.002 | 0.228 ± 0.004 | 0.687 ± 0.005 | 0.290 ± 0.001 | 0.503 ± 0.021 | 0.013 ± 0.007 | 1.208 ± 0.000 |
| four-way modular | +Alpaca-Dolly | 0.391 ± 0.000 | 0.371 ± 0.000 | 0.268 ± 0.004 | 0.649 ± 0.003 | 0.290 ± 0.001 | 0.607 ± 0.002 | 0.028 ± 0.002 | 1.136 ± 0.000 |
| four-way modular | +SmolTalk | 0.379 ± 0.001 | 0.359 ± 0.000 | 0.197 ± 0.000 | 0.655 ± 0.002 | 0.288 ± 0.001 | 0.597 ± 0.013 | 0.015 ± 0.000 | 1.241 ± 0.000 |

| routed − dense | SFT | ΔMC (12) seed 1 / seed 2 | ΔMC w/o boolq seed 1 / seed 2 | Δwikitext bpb seed 1 / seed 2 |
|---|---|---|---|---|
| four-way corrected | +Alpaca-Dolly | +0.019 / +0.017 | +0.018 / +0.017 | -0.077 / -0.077 |
| four-way corrected | +SmolTalk | +0.014 / +0.013 | +0.024 / +0.027 | -0.102 / -0.100 |
| four-way modular | +Alpaca-Dolly | +0.020 / +0.021 | +0.020 / +0.021 | -0.049 / -0.049 |
| four-way modular | +SmolTalk | +0.014 / +0.016 | +0.019 / +0.019 | -0.067 / -0.067 |

- SFT-seed variance is tiny: half the seed-to-seed difference is at most 0.001 for the 12-task MC mean, 0.002 for the 11-task mean and under 0.001 for wikitext bpb. boolq is the noisiest task (up to ±0.021), which is why the boolq-free mean is reported.
- The routed-vs-dense gaps reproduce across seeds within 0.003: +1.3 to +2.1 MC points and -0.05 to -0.10 wikitext bpb for both routed variants after both SFT sets. So the 300M SFT comparison is not an SFT-seed artifact.
- This covers SFT-seed variance only. All rows share one pretraining run per model, so pretraining-seed variance is still unmeasured.
