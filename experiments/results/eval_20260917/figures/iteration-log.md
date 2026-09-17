# Rendering record

1. Generated `ordinary_paired`, `routing_dependence`, and `context_transfer`
   with `scripts/plot_eval_campaign.py` from the checked-in numeric summaries.
   Exported PNG, PDF, and editable SVG for each.
2. Inspected all three PNGs. No clipped labels, overlapping legends, missing
   uncertainty marks, or unreadable panels were observed. No visual revision
   was needed for this rendering.
3. After the five 150M trained variants completed, generated `trained_ladder`
   from the paired baseline comparisons and inspected its PNG. The two
   panels and all configuration/parameter labels were readable; no revision
   was required. Exported the same figure to PDF and SVG.

Future plots or changed result inputs require a new render and inspection;
subsequent renders are recorded below.

## Context generation

Added six panels for explicit target-value inclusion after all ten generation
arms finished. The plot distinguishes neutral/deceptive cues and three prompt
forms, labels each channel's gamma, and shows two fixed controls as individual
gray crosses. Reference rates and run sizes are read from metadata. Inspection
of the rendered PNG found no clipping or overlapping labels; SVG/PDF exports
use the same figure. The caption and accompanying diagnostics separate word
omission from contradictory values.

## SAE transfer

Added the eight-feature old/new comparison after all 96 feature/control arms
completed. The left panel displays the signed dose span with paired block-16
intervals on the new corpus; the right reports the maximum other-token loss
cost over the two doses. The target-empty feature remains visible and has no
new target-effect point. The rendered PNG was inspected: row labels, marker
types, intervals, the cost criterion, and the caption are readable without
clipping. PDF and SVG use the same figure.

## Attribute-QA follow-up

Expanded the context-generation figure from six to eight panels after the
additional ten arms completed. All intervals now use the prompt-cluster
bootstrap, which keeps repeated neutral-QA prompts together. The new cohort
and question change are labeled, and the negative primary-endpoint effects
appear on the same scale as the earlier positive results. The first expanded
render crowded the footer against the lower labels; reserved footer space
and explicit y ticks resolve this. The revised PNG was inspected, with the
same updated figure exported to PDF and SVG.

## SAE article sample labels

After the additional 64-window run, relabeled the existing SAE figure to name
the first 128-window sample explicitly. Numeric inputs are unchanged; the new
article sample and its sparse-token uncertainty remain in the complete comparison
table. The updated PNG was inspected: longer title and legend fit, and markers,
intervals and the footer remain clear. PDF/SVG exports use the same figure.
