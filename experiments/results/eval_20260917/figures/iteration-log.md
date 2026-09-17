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
