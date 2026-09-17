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
this record applies to the four exports listed above.

## Context generation

Added six panels for explicit target-value inclusion after all ten generation
arms finished. The plot distinguishes neutral/deceptive cues and three prompt
forms, labels each channel's gamma, and shows two fixed controls as individual
gray crosses. Reference rates and run sizes are read from metadata. Inspection
of the rendered PNG found no clipping or overlapping labels; SVG/PDF exports
use the same figure. The caption and accompanying diagnostics separate word
omission from contradictory values.
