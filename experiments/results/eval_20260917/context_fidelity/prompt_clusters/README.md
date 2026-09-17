# Rendered-prompt clusters and generation endpoints

The original item key includes the receiver. Neutral QA prompts omit that
field, so different sampled items can render as identical prompts. This
posthoc check resamples identical-prompt groups together, retaining the
original item-weighted point estimates. JSON also gives equally weighted
unique-prompt point estimates. Intervals are unadjusted and do not represent
variation across training seeds or all possible control circuits.

The cohorts exclude original discovery item keys, not every repeated fact.
New attribute questions also use a new template; differences between cohorts
are not a paired estimate of changing only the question wording.

| Source | Condition | Item rows | Unique prompts | Largest repeated group |
|---|---|---:|---:|---:|
| context_generation | original/neutral | 1024 | 1024 | 1 |
| context_generation | original/deceptive | 1024 | 1024 | 1 |
| context_generation | qa/neutral | 1024 | 819 | 4 |
| context_generation | qa/deceptive | 1024 | 1024 | 1 |
| context_generation | dialogue/neutral | 1024 | 1024 | 1 |
| context_generation | dialogue/deceptive | 1024 | 1024 | 1 |
| context_attribute_generation | attribute_qa/neutral | 1024 | 830 | 3 |
| context_attribute_generation | attribute_qa/deceptive | 1024 | 1024 | 1 |

[All named endpoints](named_endpoints.md) include losses, gains and uncertain differences.
[Direct control comparisons](control_comparisons.md) retain the two fixed controls.
The companion JSON records overlap with original discovery facts, all arms,
and the change from giving each unique prompt equal weight.
