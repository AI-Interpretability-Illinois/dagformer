# Evaluation figures

Format: project research report, quantitative plots with editable SVG/PDF and
PNG previews. No external image generation or architecture reconstruction.
Code: `scripts/plot_eval_campaign.py`.

| Artifact | Question and evidence role | Source data | Panel map and uncertainty |
|---|---|---|---|
| `ordinary_paired` | Which ordinary tasks improve, and how uncertain are the paired differences? Main result, including null/negative tasks. | `standard_matched/paired_summary.json` | All twelve accuracy endpoints, then both BPB endpoints; paired 10,000-draw document-bootstrap 95% intervals. Positive means DAGFormer improves. |
| `routing_dependence` | How much does prediction depend on external content routing versus local corrections? Mechanism. | `routing_dependence/{75m,150m,300m,300m_step10500}.json` | Predictor position-table substitution; local correction freeze/removal. Separate y ranges; paired normal intervals over 128 evaluated windows. |
| `context_transfer` | Do fixed edge edits transfer across prompt formats at modest language-model cost? Robustness and limitation. | `context_fidelity/context_validation_{original,qa,dialogue}.summary.json` | Three prompt formats under the deceptive narrative cue; WikiText NLL cost. Paired intervals over 1,024 items or 50 windows. Five norm-matched random-head controls for both-channel edits appear as a range. |

Channel colors are blue for predictor, orange for correction, and purple for
both. Markers redundantly encode the channels; benchmark scale markers remain
distinguishable in grayscale. Baselines and control ranges are neutral gray.

Labels name metrics and units, including candidate-normalized `p_true` and
NLL in nats/token. Captions state checkpoint steps, calibrated/test split roles,
interval units and the project-specific +0.05-NLL criterion. No training-seed
uncertainty or unrestricted honesty claim is implied.

Place each plot next to the corresponding result discussion in the campaign
README. Plot inputs are the checked-in numeric summaries; no values are
transcribed manually into plotting code. Generated files are deterministic
functions of those summaries and the script.
