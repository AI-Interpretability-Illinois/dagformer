# Scaling progress figure

The six-panel figure shows observed common WikiText2 NLL endpoints against
total parameters, analytic training FLOPs, tokens, layers, width and batch size.
Its current purpose is to track evidence as runs finish. Missing results remain
empty; planned x coordinates are shown without synthetic y values.

`scripts/collect_six_axis.py` reads the PR pair ledger, completed new-run
evaluations and completed 600M/1B exports. Its `results.json` and `endpoints.csv`
preserve each plotted value's source. Existing training corpora form separate
series. The full predictor counts toward the parameter axis. No scaling-rate
claim or fitted loss floor is inferred from this progress figure.

Dense uses gray dashed lines; DAGFormer uses blue solid lines. Marker shape
identifies the training corpus, preserving that distinction in grayscale.
New runs show the first seed; repeated-seed uncertainty is not available yet.
PNG previews and editable PDF/SVG are generated together.

Initial render QA: labels and legends are visible, axes specify units, pending
panels have no fabricated observations, and the figure uses token NLL throughout.
