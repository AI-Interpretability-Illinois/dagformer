# Completed harness evaluation inventory

A setting can reuse the same trained checkpoint with a different intervention or task.
The natural-language and interpretability reports explain which settings are directly comparable.
Sample counts below come from each saved harness result; per-document files are retained separately.

| Group | Completed settings | Tasks per setting | Recorded effective samples across settings/tasks |
|---|---:|---:|---:|
| [standard_matched](../standard_matched) | 7 | 14 | 239,113 |
| [standard_latest](../standard_latest) | 1 | 14 | 34,159 |
| [standard_ladder](../standard_ladder) | 5 | 14 | 170,795 |
| [standard_frozen_predictor](../standard_frozen_predictor) | 3 | 14 | 102,477 |
| [standard_context_pred](../standard_context_pred) | 1 | 14 | 34,159 |
| [standard_context_corr](../standard_context_corr) | 1 | 14 | 34,159 |
| [standard_context_both](../standard_context_both) | 1 | 14 | 34,159 |
| [standard_context_random1000](../standard_context_random1000) | 1 | 14 | 34,159 |
| [standard_context_random1001](../standard_context_random1001) | 1 | 14 | 34,159 |
| [commonsense_content](../commonsense_content) | 7 | 1 | 8,547 |
| [gsm8k_components](../gsm8k_components) | 7 | 2 | 18,466 |
| [gsm8k_full](../gsm8k_full) | 7 | 1 | 9,233 |
| [gsm8k_latest](../gsm8k_latest) | 1 | 1 | 1,319 |
| [gsm8k_no_cache](../gsm8k_no_cache) | 1 | 1 | 200 |

Total: 44 harness runs, 324 task endpoints.
Multiple task endpoints and interventions share checkpoints and documents; they are not independent training replications.
