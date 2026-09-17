# Figure QA

All six PNG exports were rendered and visually inspected at their saved
resolution on September 17, 2026. The corresponding PDF and SVG exports use
the same Matplotlib figures.

| Figure | Numeric and semantic checks | Visual inspection | Status |
|---|---|---|---|
| `ordinary_paired` | All 14 task endpoints and all three matched backbone sizes come directly from `paired_summary.json`; signs consistently favor DAGFormer. Accuracy and BPB use separate panels and units. The label-bias footnote accompanies BoolQ and CommonsenseQA. | Task labels, scale legend, interval bars and footnote are legible, without clipping or overlap. | Pass |
| `routing_dependence` | Four checkpoints, three substitution/removal arms, and paired intervals come directly from routing JSONs. The caption explicitly identifies the panels' different vertical ranges. | Scale labels, markers and legends remain distinct; zero and axis units are visible. | Pass |
| `context_transfer` | Three prompt styles use the same second item set. Candidate-normalized fidelity, WikiText NLL, and five norm-matched random-head controls retain their distinct meanings. Both improvements and the QA reversal are shown. | The four panels, channel markers, control band and NLL criterion are readable, without clipping. | Pass |
| `trained_ladder` | Six variants use paired comparisons to the same 150M baseline; full-model data come from the ordinary paired summary. Labels use measured total parameters and the budget is verified from original optimizer states. | Both endpoint panels, all six configuration labels, intervals and budget caption are legible without overlap or clipping. | Pass |
| `context_generation` | Both completed ten-arm cohorts are required. Eight panels read point estimates and prompt-cluster intervals; metadata supplies gamma, reference rates and controls. The new cohort and changed question are explicit. | Updated PNG inspected: eight panels share zero and scale; negative attribute-QA effects are visible. Footer was moved away from lower-axis labels. Titles, doses, intervals and controls remain readable. | Pass |
| `sae_transfer` | Eight fixed feature IDs, target-token counts, old dose spans, first-sample block-16 paired intervals, and maximum other-token costs are read from saved summaries. The later 64-window sample is reported separately. The zero-target feature has no invented effect or interval. | Relabeled PNG inspected: the title and legend explicitly identify the first 128 windows. All rows, interval caps, zero lines, cost criterion and caption remain readable without overlap. | Pass |

The plots describe fixed checkpoints. Intervals represent document, item, or
window variation as labeled, rather than training-seed uncertainty. The
five-control range is not a confidence interval. Editable vector outputs are
available beside the PNG previews; labels remain SVG text.
