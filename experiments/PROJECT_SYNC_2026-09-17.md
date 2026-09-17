# Code synchronization and PR evidence audit — 2026-09-17

## Sources recovered

- timan1 working tree: `e1c7806`, plus nine diagnostic/download scripts, a
  250-step config, April loss CSVs and four comparison plots.
- GitHub main: `f26c357`, including the author's September interpretability
  commit `7e5f05d`, [PR #1](https://github.com/AI-Interpretability-Illinois/dagformer/pull/1)
  and [PR #2](https://github.com/AI-Interpretability-Illinois/dagformer/pull/2).
- Delta: `/projects/bfqt/users/yurenh2/ml-projects/DAGFormer`, at `7e5f05d` with
  59 changed/new files (5.9 MB): September 3–13 analyses, result JSONs, figures,
  scripts and the appended experiment log. Recovered in commit `f6110e9`.

Tensor dumps and optimizer checkpoints stay on Delta. The configured historical
training path `/work/hdd/bfqt/data/pretok/dolma_v1_7_12b` was absent when checked
on September 17; current evaluation uses the available checkpoints, cached
natural-text windows, and benchmark datasets. The local
evaluation copy under `checkpoints/pr_sync_20260917` omits optimizer state and
duplicate backbone entries, preserving evaluation tensors. The April stash
`9901429` is retained, not blindly applied over the current data-resume and
training implementations. Raw logs and machine-specific links remain local.

## What the results actually test

| Analysis | Intervention / endpoint | Evidence and interpretation |
|---|---|---|
| Author's swap ladder | Replace external predictor with a position table; keep corrections | Dynamic NLL 3.217253 vs position table 3.217315, only +0.000062. The predictor's input-dependent variation has little marginal effect on this corpus. |
| Author's named copy heads | Scale Q/K head deviations, in both predictor and corrections | Copy accuracy 72.7% to 80.2% at gamma 1.5 with about +0.02 natural-text NLL. Tests head-specific wiring. |
| Author's context-fidelity circuit | Directly edit ten chosen Q/K/V coefficients in both channels | Original cloze p_true 0.719 to 0.522 at gamma 0 and 0.935 at gamma 4. Discovery and verification reused seed-0 items; this needs a new-item evaluation and NLL measurement. |
| Author's correction-space SAE | Add a learned feature direction; measure selected-token loss and free-generation token counts | The old screen has 34 of 187 directions with opposite-sign loss effects exceeding a 0.04-nat span and off-rule cost below 0.03. Large-sample generation was much less stable. These are conditional-token effects, not an instruction-contrast steering result. |
| PR #2 | Discover instruction-contrast differences, then edit pred or corr separately | Reports no useful domain-code steering in the tested directions under its +0.05 NLL threshold; honesty prompts have weak baseline instruction effects. Tests a different behavior and intervention; see the statistical-label audit below. |

Sources: `results/interp/swap2_step9000.json`, `localize_step9000.json`,
`liar_cloze/deception_{conns,steer,steer_neutral}.json`, and
`circuits/verify_domain_code_{pred,corr}.json`.
The recovered SAE sources are `routing_sae/screen.json` and
`routing_sae/screen_generation.json`. For the earlier kinship-direction
example f3583, the larger generation run has 15/13/9/7/12 hits across
alpha -4/-2/0/+2/+4, so the strong earlier monotonic-frequency claim did not
replicate. The author's own September 3–4 log already records this limitation.

The two sets of findings can coexist. Useful static head-specific wiring and
local correction dynamics do not require the external predictor's response to
instruction wording to be a usable control signal. Conversely, a weak response
to one instruction contrast does not establish that every predictor edit is
causally inert.

PR #2 restricts its tested coordinates to **hyperconnections**. Three of the
author's ten context-fidelity edges are sequential paths, including the
strongest discovery edge `L4/h1/v<-src4`; these are excluded from that search.
The new evaluation therefore supports all/hyper/sequential subsets as well as
pred/corr/both channels.

## Numeric and interpretation findings

1. **PR #1 multiple-choice count was wrong.** Original JSONs give 14 wins,
   5 losses and 2 ties across seven tasks and three sizes. The nominal sign-test
   p is 0.063568, not 0.043285. The latter counts BPB and generation too.
   `compare.py` now reports both groups and labels sign tests descriptive;
   repeated model pairs are not independent experimental units.
2. **GSM8K BPB includes the question and gold answer.** Its 4.38%, 4.65%, 5.03%
   improvements are likelihood results, not generated problem-solving accuracy.
   The reported 1.7x ratio uses absolute BPB differences; relative improvement
   ratios are 1.26–1.33x. The original 200-item generation scores were
   0.5–2.5%; the [new full-split matched evaluation](results/eval_20260917/README.md)
   reports all three pairs separately.
3. **The Wikitext discrepancy is traced to checkpoint metadata.** The old
   1.0714925 number belongs to dense step 9000, as recorded in
   `results/lmeval/matched_mmap/dense_300m_mmap_s9000.json`. PR #1 measured
   1.0648552 on the shared step-12000 checkpoint. The shared bundle README
   attached the step-9000 number to the step-12000 file.
4. **Some PR #2 prose refers to an earlier run.** The committed pred discovery
   JSON selects **513** edges, signal/null selection ratio **12.035**, not the
   README's 44 and 26.4. Its full eligible-coordinate patch shifts the score
   **+0.000991**, or **0.0296%** of 3.349919 headroom, not +0.0021. The negative
   direction remains the same, but the numbers must come from the JSON.
5. **A full patch is not an upper bound on every subset edit.** Opposing
   contributions can cancel. The patch result bounds the measured full-patch
   intervention, not all sparse circuits or all editing directions. Random
   sparse perturbations also do not constitute a causal positive-control
   circuit of known behavioral effect.
6. **The variance denominators differ.** PR #2's 0.119 residual fraction is
   measured *after* position means are removed; 0.047 is the fraction of total
   variance remaining after position removal. Calling both a single "88% of
   total variance" decomposition is inaccurate.
7. **The original connection editor is order-dependent.** It recomputes head
   means after each prior edge edit. The new evaluation computes every mean
   from the unedited input. Original scripts and raw results are preserved;
   a regression test covers simultaneous editing of heads sharing a source.
8. **PR #2's automatic NULL label is not a paired null test.** Its condition
   compares the mean change with twice the edited arm's *raw-score* SEM,
   rather than the SEM of paired changes. The displayed paired t statistic
   also treats instruction paraphrases as independent prompts. This can hide
   small consistent effects behind large between-item score differences.
   The current reporter labels that rule descriptive; archived outputs are
   retained. The new stream follow-up averages paraphrases within each item
   and reports paired item-bootstrap intervals, with 32 additional fixed
   content pairs. This is an exploratory follow-up, not a preregistered test.
9. **The stream arms lacked capability checks.** In the archived verifier,
   q/k/v/r restrictions ran only at the largest additive dose and omitted NLL.
   A large stream score shift therefore could not establish useful steering.
   Future verifier runs include their NLL, and the follow-up measures all
   four streams and the joint direction across six small doses with controls.
   The [completed stream results](results/eval_20260917/domain_streams/README.md)
   detect small correction Q/V effects on new content, alongside their
   natural-text NLL costs and separate code/prose likelihood changes.
   The [large-dose completion](results/eval_20260917/domain_streams_large/README.md)
   reproduces the original R-only dose-64 shift of +8.1373, but measures
   +13.2181 natural-text NLL. Both code and prose likelihoods worsen, with
   a larger prose decrease. This fills the missing capability measurement.
   Also, the verifier's reported magnitude ratio uses predictor alpha in its
   denominator even for correction edits; it is not a measurement of the
   correction channel's own relative amplitude.

For 12 layers and 16 heads, the canonical layout has 3,773 coordinates:
**3,234 hyperconnections and 539 sequential paths**. These are signed mixing
coefficients; absolute coefficient mass is not a causal contribution measure.

## Evaluation campaign

The user requested ordinary and interpretability evaluation and eight hours of
unattended work, starting around **2026-09-17 06:20 UTC**, through **14:20 UTC**.
Only timan1 GPUs 2 and 3 are used; existing processes on GPUs 0 and 1 are left
running. The isolated benchmark environment is
`/scratch/yurenh2/venvs/dagformer-eval-20260917` (`lm_eval==0.4.13`,
`transformers==4.57.1`; PyTorch 2.10.0 inherited from the host).

- Ordinary: matched 300M step-9000 baseline/DAGFormer, full core tasks plus
  likelihood-scored reasoning. Save per-document outputs for paired analysis.
- Interpretability: fixed ten-edge context-fidelity circuit, new distinct
  content items excluding the original 240 discovery items, pred/corr/both
  interventions, matched-coordinate random heads and natural-text NLL.
- Then separate hyperconnections from sequential paths and test intermediate
  doses around the original model. Repeat supported effects on a second item
  seed and expand ordinary evaluation to other available sizes.

Fresh results are written under `experiments/results/eval_20260917`; logs under
`logs/eval_20260917`. The initial 128-item / 8-NLL-sequence run is a pilot,
not the final uncertainty estimate. No new training is part of this campaign.
