# Shared-model SFT benchmark (lm-eval 0.4.13, reasoning + core suites)

Shared pretrained checkpoints (/work/hdd/bfqt/shared/dagformer-models: 75M/150M/300M dense baseline and DAGFormer) before SFT and after instruction tuning on Alpaca-Dolly (3 epochs) and SmolTalk (2 epochs), same recipe for all (global batch 64x1024, lr 2e-4 cosine, assistant-only loss). Single runs. Pre-SFT reasoning numbers are the 2026-09-16 runs in lmeval/reasoning/ (identical settings: gen_limit 200, batch 16, 3-shot GSM8K); pre-SFT core and all SFT numbers were produced 2026-09-30/10-01 on timan1 (same lm-eval/torch/transformers versions as Delta). Note: the shared checkpoints are not step-matched (e.g. 300M DAGFormer step 9000 vs baseline step 12000; see models.py checkpoint_step). Accuracy higher is better; gsm8k_bpb and wikitext are bits per byte, lower is better.

| size | model | stage | MC mean (12) | arc_challenge | arc_easy | mathqa | commonsense_qa | social_iqa | openbookqa | winogrande | lambada_openai | hellaswag | piqa | sciq | boolq | gsm8k | gsm8k_bpb | wikitext |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 75M | dense baseline | base | 0.304 | 0.217 | 0.302 | 0.202 | 0.196 | 0.340 | 0.244 | 0.504 | 0.035 | 0.258 | 0.559 | 0.413 | 0.380 | 0.010 | 1.840 | 1.393 |
| 75M | dense baseline | +Alpaca-Dolly | 0.309 | 0.231 | 0.319 | 0.199 | 0.196 | 0.350 | 0.232 | 0.515 | 0.055 | 0.258 | 0.547 | 0.419 | 0.384 | 0.010 | 1.801 | 1.459 |
| 75M | dense baseline | +SmolTalk | 0.309 | 0.227 | 0.309 | 0.231 | 0.196 | 0.339 | 0.240 | 0.494 | 0.052 | 0.263 | 0.555 | 0.416 | 0.385 | 0.015 | 1.613 | 1.595 |
| 75M | DAGFormer | base | 0.316 | 0.205 | 0.302 | 0.209 | 0.196 | 0.341 | 0.250 | 0.511 | 0.071 | 0.262 | 0.558 | 0.464 | 0.426 | 0.025 | 1.759 | 1.347 |
| 75M | DAGFormer | +Alpaca-Dolly | 0.319 | 0.223 | 0.311 | 0.203 | 0.197 | 0.342 | 0.240 | 0.500 | 0.095 | 0.261 | 0.561 | 0.462 | 0.439 | 0.015 | 1.733 | 1.417 |
| 75M | DAGFormer | +SmolTalk | 0.336 | 0.231 | 0.297 | 0.222 | 0.193 | 0.355 | 0.240 | 0.512 | 0.074 | 0.264 | 0.546 | 0.475 | 0.617 | 0.010 | 1.502 | 1.550 |
| 150M | dense baseline | base | 0.336 | 0.214 | 0.336 | 0.201 | 0.197 | 0.343 | 0.276 | 0.503 | 0.157 | 0.262 | 0.592 | 0.549 | 0.406 | 0.010 | 1.593 | 1.196 |
| 150M | dense baseline | +Alpaca-Dolly | 0.337 | 0.225 | 0.330 | 0.203 | 0.197 | 0.343 | 0.254 | 0.510 | 0.183 | 0.267 | 0.592 | 0.532 | 0.402 | 0.015 | 1.560 | 1.260 |
| 150M | dense baseline | +SmolTalk | 0.340 | 0.217 | 0.328 | 0.235 | 0.216 | 0.356 | 0.244 | 0.506 | 0.117 | 0.265 | 0.579 | 0.532 | 0.484 | 0.020 | 1.337 | 1.399 |
| 150M | DAGFormer | base | 0.361 | 0.224 | 0.347 | 0.222 | 0.197 | 0.364 | 0.256 | 0.528 | 0.213 | 0.274 | 0.601 | 0.600 | 0.509 | 0.005 | 1.519 | 1.152 |
| 150M | DAGFormer | +Alpaca-Dolly | 0.366 | 0.240 | 0.341 | 0.224 | 0.208 | 0.360 | 0.252 | 0.537 | 0.254 | 0.275 | 0.598 | 0.620 | 0.487 | 0.005 | 1.491 | 1.211 |
| 150M | DAGFormer | +SmolTalk | 0.361 | 0.237 | 0.355 | 0.232 | 0.195 | 0.359 | 0.252 | 0.511 | 0.170 | 0.276 | 0.573 | 0.605 | 0.572 | 0.010 | 1.219 | 1.333 |
| 300M | dense baseline | base | 0.379 | 0.217 | 0.368 | 0.216 | 0.199 | 0.362 | 0.274 | 0.513 | 0.263 | 0.298 | 0.627 | 0.641 | 0.574 | 0.025 | 1.402 | 1.065 |
| 300M | dense baseline | +Alpaca-Dolly | 0.367 | 0.235 | 0.373 | 0.210 | 0.192 | 0.360 | 0.264 | 0.486 | 0.284 | 0.301 | 0.622 | 0.623 | 0.448 | 0.025 | 1.403 | 1.117 |
| 300M | dense baseline | +SmolTalk | 0.368 | 0.235 | 0.361 | 0.241 | 0.201 | 0.361 | 0.272 | 0.513 | 0.188 | 0.297 | 0.595 | 0.623 | 0.529 | 0.015 | 1.145 | 1.232 |
| 300M | DAGFormer | base | 0.397 | 0.233 | 0.383 | 0.226 | 0.196 | 0.361 | 0.276 | 0.522 | 0.301 | 0.314 | 0.643 | 0.700 | 0.609 | 0.015 | 1.331 | 1.022 |
| 300M | DAGFormer | +Alpaca-Dolly | 0.395 | 0.250 | 0.386 | 0.228 | 0.206 | 0.377 | 0.278 | 0.520 | 0.334 | 0.319 | 0.643 | 0.667 | 0.539 | 0.015 | 1.324 | 1.072 |
| 300M | DAGFormer | +SmolTalk | 0.393 | 0.247 | 0.380 | 0.226 | 0.203 | 0.378 | 0.280 | 0.521 | 0.285 | 0.308 | 0.606 | 0.698 | 0.587 | 0.010 | 1.055 | 1.165 |
