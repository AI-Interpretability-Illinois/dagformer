"""Paper figures from the repo's result files (no training data or models needed).

    python paper/figures/make_figures.py      # runs gen_fig_scaling.py and gen_fig_pruning.py -> fig_*.pdf (+ .png)

Style (fonts, colors, sizes) lives in pubstyle.py; each figure has its own gen_fig_<name>.py.
"""
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ("gen_fig_scaling.py", "gen_fig_pruning.py"):
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
