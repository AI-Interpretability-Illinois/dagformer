**Depth diagnostics: figure contract**

Question: how do the matched-step Dense and full DAG checkpoints distribute representation changes, predictive information and causal dependencies across their twelve backbone layers?

Evidence: native-checkpoint mechanism analysis on the existing WikiText test and Dolma evaluation caches. No figure asserts a measured effective number of layers or a fixed-budget depth-scaling advantage.

Sources: `dense_<corpus>.npz`, `dag_<corpus>.npz`, the two model metadata files, and `protocol.json`. Every line/heatmap comes from these arrays; `summary.json` and CSV exports preserve the plotted numbers. Uncertainty is a descriptive 95% window-bootstrap interval, not variation over training seeds or independent documents.

The overview uses four aligned line panels: adjacent-state angular distance; final-head readout NLL at each depth; single-layer bypass NLL increase; and DAG intervention contrasts. It keeps all layers, including unusually sensitive early layers. Dense is blue with circle markers; DAG is orange with square markers; different DAG interventions also differ in line style.

The dependence figure uses four causal matrices: Dense bypass, DAG bypass, DAG bypass with frozen effective routing, and DAG bypass with the source slot removed. All panels share a color scale, block indexing and triangular mask. Color encodes the normalized downstream update change, not probability or causal importance in isolation. Raw changes and denominator norms are retained in the source arrays.

Output: deterministic Matplotlib PNG previews plus editable SVG and vector PDF. Labels use one-based block numbers, while state 0 denotes the embedding. Results JSON retains the zero-based array indexing. Captions state the corpus, sample count, matched training step and the additional DAG parameters. Rendered previews are inspected for clipping and misleading scales before delivery.

Render review, 2026-09-30: inspected all four PNGs. Axes, legends and corpus titles are readable and unclipped; all causal panels use the same linear scale within each corpus. Full-depth plots retain the large early-layer effects rather than clipping them to emphasize late layers. Corpus display names were changed from internal identifiers to ordinary labels. Source-array checks confirm final readout CE agrees with model NLL within 5e-7, final KL is zero, and final top-5 overlap is one.
