# Figure QA

All five PNG exports were rendered and visually inspected at their saved
resolution on September 17, 2026. The corresponding PDF and SVG exports use
the same Matplotlib figures.

| Figure | Numeric and semantic checks | Visual inspection | Status |
|---|---|---|---|
| `ordinary_paired` | All 14 task endpoints and all three matched backbone sizes come directly from `paired_summary.json`; signs consistently favor DAGFormer. Accuracy and BPB use separate panels and units. The label-bias footnote accompanies BoolQ and CommonsenseQA. | Task labels, scale legend, interval bars and footnote are legible, without clipping or overlap. | Pass |
| `routing_dependence` | Four checkpoints, three substitution/removal arms, and paired intervals come directly from routing JSONs. The caption explicitly identifies the panels' different vertical ranges. | Scale labels, markers and legends remain distinct; zero and axis units are visible. | Pass |
| `context_transfer` | Three prompt styles use the same second item set. Candidate-normalized fidelity, WikiText NLL, and five norm-matched random-head controls retain their distinct meanings. Both improvements and the QA reversal are shown. | The four panels, channel markers, control band and NLL criterion are readable, without clipping. | Pass |
| `trained_ladder` | Six variants use paired comparisons to the same 150M baseline; full-model data come from the ordinary paired summary. Labels use measured total parameters and the budget is verified from original optimizer states. | Both endpoint panels, all six configuration labels, intervals and budget caption are legible without overlap or clipping. | Pass |
| `context_generation` | All ten completed arms are required before plotting. The six cue/format panels read inclusion differences and paired intervals from the generation summary. Gamma values, reference rates, item count, token cap and control count come from metadata. | Six panels share a visible zero and y range; channel markers, control crosses, reference labels and endpoint caption are legible without clipping. | Pass |

The plots describe fixed checkpoints. Intervals represent document, item, or
window variation as labeled, rather than training-seed uncertainty. The
five-control range is not a confidence interval. Editable vector outputs are
available beside the PNG previews; labels remain SVG text.
