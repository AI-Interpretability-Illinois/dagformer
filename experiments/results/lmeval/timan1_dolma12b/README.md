# 75M three-way comparison on the timan1 12B Dolma corpus (2026-09-25)

Three 75M models pretrained with the identical pipeline on timan1 (3x RTX A6000):
same locally rebuilt 12B-token Dolma v1.7 prefix corpus (same `scripts/pretokenize.py`
recipe as `/work/hdd/bfqt/data/pretok/dolma_v1_7_12b`, built per-file and merged with
`scripts/merge_pretok_dirs.py`), 3000 steps x 528,384 tokens = 1.585B tokens
(micro 4 x accum 43 x 3 GPUs = 516-seq global batch), same held-out eval set.

| run | config | routing |
|---|---|---|
| `modular_75m` | `configs/pretrain_75m_dagformer_modular_timan1_dolma12b.yaml` | fourway_modular (every inter-module edge predicted) |
| `fourway_corrected_75m` | `configs/pretrain_75m_dagformer_fourway_timan1_dolma12b.yaml` | fourway_corrected (standard connections fixed) |
| `dense_75m` | `configs/pretrain_75m_baseline_timan1_dolma12b.yaml` | none (dense OLMo-2 75M) |

## Training-time held-out eval NLL (50 web-text sequences, `eval_skip=1M`)

| step | modular | fourway_corrected | dense |
|---|---|---|---|
| 500 | 5.255 | 5.368 | 5.464 |
| 1000 | 4.499 | 4.565 | 4.743 |
| 1500 | 4.355 | 4.370 | 4.546 |
| 2000 | 4.238 | 4.257 | 4.395 |
| 2500 | 4.195 | 4.212 | 4.352 |
| 3000 | **4.194** | 4.203 | 4.339 |

## lm-eval (0-shot, `scripts/eval_lm_harness.py --tasks default`, lm_eval 0.4.13, batch 8)

| task | metric | modular | fourway_corrected | dense |
|---|---|---|---|---|
| wikitext | word ppl | **138.2** | 141.3 | 174.6 |
| wikitext | bits/byte | **1.330** | 1.336 | 1.393 |
| lambada_openai | ppl | 1871 | **1720** | 4876 |
| lambada_openai | acc | 8.4 | **8.7** | 3.2 |
| sciq | acc_norm | 44.9 | **46.7** | 41.5 |
| boolq | acc | **48.8** | 42.6 | 38.1 |
| arc_easy | acc_norm | **30.7** | 29.6 | 29.7 |
| piqa | acc_norm | **56.9** | 56.1 | 56.1 |
| hellaswag | acc_norm | 26.2 | 26.2 | 25.9 |
| openbookqa | acc_norm | 23.8 | 23.2 | **24.8** |
| winogrande | acc | 50.4 | 49.8 | **50.6** |

hellaswag / openbookqa / winogrande are at chance for all three (75M, 1.6B tokens).

Caveats shared by all three runs: the corpus is the every-other-document subsample that
`packed_token_stream` produces (as for the original corpora), and about half of the 50
training-eval documents lie inside the 12B training span. Neither affects the ordering.
Raw checkpoints/logs live on timan1 under `/srv/local/xy51/checkpoints` and `/srv/local/xy51/logs`.
