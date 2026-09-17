# DAGFormer evaluation campaign — September 17, 2026

This campaign compares fixed checkpoints on ordinary language-model tasks and
tests what the external predictor and local corrections contribute. The
[code/PR audit](../../PROJECT_SYNC_2026-09-17.md) records recovered Delta work,
the merged PRs and corrections to their interpretations.

## Ordinary evaluation

The full 14-task likelihood suite is complete for the 75M, 150M and 300M
baseline/DAGFormer pairs. The 600M baseline is an unpaired reference. The 300M
pair uses **step 9000 for both models**, resolving PR #1's 9000-versus-12000
training-step mismatch. The nominal sizes refer to backbone scales;
DAGFormer includes additional predictor and correction parameters.

The [training-budget audit](provenance/training_budgets.json) reads optimizer
update counters from the original Delta checkpoints and tokens per update
from training logs. Both models consumed 1.573B tokens at 75M, 3.146B at 150M,
and 4.719B at 300M. All five trained 150M routing variants also consumed
3.146B tokens. The step-9000 periodic checkpoints contain 9,001 optimizer
updates in both families because that training-loop label is zero-based;
the final step-3000/6000 labels equal completed updates.

| Backbone scale | Baseline parameters | Full DAGFormer parameters |
|---|---:|---:|
| 75M | 76,558,848 | 105,657,332 |
| 150M | 152,593,152 | 182,561,679 |
| 300M | 304,137,216 | 336,447,805 |

Selected 300M results, with paired document-bootstrap intervals:

| Task / metric | Baseline | DAGFormer | Difference favoring DAGFormer | 95% interval |
|---|---:|---:|---:|---|
| WikiText / bits per byte | 1.0714 | 1.0225 | 0.0489 | [0.0461, 0.0521] |
| LAMBADA / accuracy | 25.54% | 30.18% | 4.64 percentage points | [3.65, 5.63] |
| HellaSwag / normalized accuracy | 29.81% | 31.45% | 1.63 percentage points | [1.08, 2.21] |
| SciQ / accuracy | 63.90% | 69.80% | 5.90 percentage points | [3.50, 8.30] |
| BoolQ / accuracy | 56.27% | 60.80% | 4.53 percentage points | [2.97, 6.09] |

[All paired results](standard_matched/paired_summary.md) include losses and
uncertain differences. Intervals describe document sampling for these trained
checkpoints, without multiple-testing adjustment or training-seed uncertainty.
GSM8K BPB scores the question and gold answer jointly; generated-answer
accuracy is being evaluated separately on the full 1,319-item test split.
Two further custom controls score the question prefix and the gold answer
conditioned on that prefix separately. These remain teacher-forced
likelihood endpoints; they do not count generated solutions as correct.
The [completed conditional-answer control](gsm8k_components/README.md)
improves from 2.2130 to 2.1047 BPB at 75M, 1.9205 to 1.8223 at 150M, and
1.6870 to 1.5684 at 300M. The gain is therefore present in the gold-answer
likelihood as well as the question likelihood. All three paired intervals
exclude zero. The earlier gold solution tokens are supplied in this task;
free generation below tests whether the model can produce the solution.

Full GSM8K generation is complete for the 75M, 150M and 300M pairs, with 1,319
documents per model, three few-shot examples and a 256-token generation cap.
All three pairs show no detected improvement in flexible numeric extraction:

| Scale | Baseline | DAGFormer | Difference | Paired 95% interval |
|---|---:|---:|---:|---|
| 75M | 1.82% | 1.67% | -0.15 percentage points | [-1.06, +0.76] |
| 150M | 1.59% | 1.21% | -0.38 percentage points | [-1.21, +0.45] |
| 300M | 1.59% | 1.52% | -0.08 percentage points | [-0.91, +0.76] |

Strict-format accuracy is 0% for all six. The
[paired results](gsm8k_full/paired_summary.md) and
[generation audit](gsm8k_full/generation_audit.md) retain the official scores
alongside deterministic output examples. A matching extracted number can
occur in repeated or irrelevant text: document 52, for example, generates
"15 pounds" where the requested answer is 15 toys. The best constant-number
diagnostic on this test split is 3.03% (always 5). Generation runs for
the 600M baseline and the later 300M DAGFormer checkpoint are still pending.
Dense baselines use KV caching; DAGFormer recomputes the prefix. In the
[200-item cache check](gsm8k_no_cache/README.md), the 300M baseline has the
same five flexible matches with caching enabled or disabled. Text changes
on 24 documents and extracted numbers on 11, but all correctness indicators
are unchanged. Repetition remains high with either setting. This checks
the evaluated subset; it does not establish numerical equivalence.

The [explicit-label audit](standard_matched/label_bias.md) shows two limitations:
all models choose A on at least 98.94% of standard CommonsenseQA questions,
and every BoolQ score remains below the 62.17% constant-yes baseline. The
relative BoolQ gain above therefore does not establish successful reading
comprehension. The [answer-text CommonsenseQA control](commonsense_content/README.md)
is now complete. Its length-normalized accuracies are 24.73% → 25.23% at 75M,
26.04% → 27.44% at 150M, and 28.99% → 29.65% at 300M. Only the 150M paired
interval excludes zero, narrowly; using raw answer likelihood, that pair
changes from 23.91% to 23.42% with an interval spanning zero. This custom
control's small 150M gain therefore depends on the scoring convention;
neither other pair has a detected gain with either score.

The [step-10500 checkpoint comparison](standard_latest/README.md) is also
complete for all 14 endpoints. With 5.506B processed tokens, its WikiText BPB
improves from step 9000's 1.02248 to 1.01830. Its twelve accuracy differences
all have paired intervals including zero, and GSM8K joint BPB is slightly
worse. This measures continued training of the same model. The comparison
at equal baseline/DAGFormer training budgets continues to use step 9000.

![All paired ordinary-task effects](figures/ordinary_paired.png)

All 14 endpoints at all three matched backbone scales. Positive values favor
DAGFormer; intervals are paired document-bootstrap 95% intervals. The
asterisked tasks have strong label biases, described above.

## Routing dependence

Position means were calibrated on 64 WikiText training windows, then evaluated
on 128 disjoint test windows, each 1024 tokens. The table replaces the external
predictor's content-dependent output; the trained model's weights are fixed.

| Checkpoint | Original NLL | Replace predictor with position table: ΔNLL | Disable corrections: ΔNLL |
|---|---:|---:|---:|
| 75M, step 3000 | 4.76458 | +0.00287 | +0.88061 |
| 150M, step 6000 | 4.08631 | +0.00103 | +1.38065 |
| 300M, step 9000 | 3.61780 | +0.00061 | +1.74929 |
| 300M, step 10500 | 3.60116 | +0.00085 | +1.77919 |

The external predictor's input-dependent component has a small marginal effect
in this test. Removing its learned wiring entirely is a different intervention
and causes substantial damage. Freezing the local corrections also hurts
performance. These inference edits do not establish what a retrained model
needs; the separately trained 150M ablation ladder below tests that question.

The predictor retains useful **position dependence**. At 300M, substituting
another sequence's predictor output at the same positions costs +0.00094 NLL.
A single mean over all positions costs +0.06746, and shuffling positions costs
+0.07240. Replacing its learned source/head pattern with identity wiring costs
+1.79541. Position-specific wiring and sensitivity to the current text are
therefore distinct contributions in the fixed checkpoint.

The two routing channels also interact. At 300M, the predictor's position-table
cost rises from +0.00061 NLL with dynamic corrections to +0.00525 with frozen
position-table corrections and +0.01318 with corrections disabled. Thus the
small predictor effect above is conditional on the rest of the trained model
remaining active; it is not a statement that the channels act independently.

The [routing results](routing_dependence/300m.json) retain all per-sequence
losses, intervals, constant/global and cross-sequence substitutions, plus
synthetic repetition tests at periods 64, 128 and 256.

The [dense reference on identical inputs](routing_dependence/dense_comparison.md)
shows copy-accuracy gains of 24.2–27.9 points at 75M and 6.0–11.6 points at
150M. At 300M, period 64 improves by 1.92 points [1.07, 2.77]; period 128
changes by -0.55 points [-4.66, 3.56] and period 256 by +0.50 points
[-5.30, 6.30]. The advantage depends on scale and repetition period. These
are teacher-forced predictions on periodic token sequences.

![Routing substitutions and removals](figures/routing_dependence.png)

Tables use 64 WikiText training windows and evaluation uses 128 test windows.
Intervals are paired normal intervals across windows. The two panels use
different vertical ranges to show the small external-predictor effect and
the larger correction effects. Substitution hooks still execute the original
predictor before replacing its output; these runs measure dependence, not
an inference-speed improvement.

A [moving-block bootstrap check](routing_dependence/block_bootstrap.md) keeps
groups of adjacent windows together because windows can share source articles.
The conclusions persist with blocks of 4, 8 and 16 windows. At 300M, the
position-table substitution costs +0.000614 NLL; the 16-window-block interval
is [0.000274, 0.000893]. This supports a small positive cost even when nearby
windows are resampled together.

The [FP32 arithmetic repeat](routing_dependence/300m_fp32.json) gives the same
pattern at 300M: the position table costs +0.000550 NLL, with paired normal
interval [0.000322, 0.000779], while removing corrections costs +1.74925.
This run upcasts the loaded checkpoint weights and disables TF32; it does
not recover precision lost when the checkpoint was stored in BF16. The weak
external content dependence persists under this arithmetic change.

The [MHA fastpath check](routing_dependence/fastpath.md) also exercises the
April stash's proposed numerical change. It confirms 64 fused encoder calls
when enabled and zero when disabled at each scale. Mean NLL differences on
32 windows have paired intervals spanning zero for all three models. The
default remains unchanged in this evaluation environment.

The same position-table substitution was also evaluated on all 14 ordinary
tasks at all three scales. The table below reports *cost* for BPB and signed
accuracy change for LAMBADA; the full
[paired report](standard_frozen_predictor/paired_vs_dagformer.md) uses a common
positive-is-better convention and includes every task's intervals.

| Scale | WikiText BPB increase | GSM8K joint BPB increase | LAMBADA accuracy change |
|---|---:|---:|---:|
| 75M | +0.000472 | +0.002540 | +0.233 percentage points |
| 150M | +0.000205 | +0.003791 | -0.136 percentage points |
| 300M | +0.000129 | +0.002993 | -0.213 percentage points |

The likelihood penalties are small but distinguishable from zero under
paired document resampling. Most accuracy changes are small and uncertain.
External content dependence is therefore weak in these evaluations, rather
than identically absent. The calibration table came from WikiText training
windows and was reused unchanged on the other tasks.

## Separately trained routing variants

The [complete 150M ladder](standard_ladder/README.md) holds the backbone,
training updates and processed-token budget fixed across seven models.
"Static", "position table" and "identity" describe the external predictor.
In corrected variants, the local MLPs still read backbone hidden states and
make the effective per-head routing depend on the current input.
Static + correction has **153.45M total parameters**, 15.95% fewer than the
full encoder model's 182.56M, and improves WikiText BPB from 1.15211 to 1.14890.
The paired BPB reduction is 0.00320, with interval [0.00214, 0.00431]. LAMBADA
is similar: 21.48% versus 21.37%, with a difference interval of [-0.74, +1.01]
percentage points. Even frozen identity routing plus correction reaches
1.15252 BPB. The static and position-table models without corrections are
substantially worse on WikiText, LAMBADA and SciQ.

![Trained routing ladder](figures/trained_ladder.png)

These results support local corrections as the main contributor to the
ordinary-task gains at this scale and budget. Simpler routing is not uniformly
best: position table + correction improves WikiText and LAMBADA but is 1.21
points lower on MathQA than the full encoder. The
[full paired table](standard_ladder/paired_vs_dagformer.md) includes all 14
endpoints and their unadjusted intervals. Each configuration has one training
run, so these intervals do not measure training-seed variation.

## Direct head edits and the PR #2 contrast

The fixed ten-edge circuit selected in the author's earlier experiment was
tested on 1,024 new content combinations, excluding all 240 discovery items.
At gamma 1.5, predictor-only edits raise candidate-normalized context fidelity
by 2.10 percentage points under the deceptive cue, with +0.00613 natural-text
NLL; correction-only edits raise it by 4.77 points, with +0.01519 NLL.
The same directions also affect neutral prompts. This is evidence about
context retrieval, not intent to deceive.

The original gamma 4 edit raises NLL by +1.05066 when applied to both channels,
so its large fidelity gain comes with a substantial language-model cost.
[The full dose table](context_fidelity/context_fidelity_full.md) records this
alongside random-head controls.

A second set of 1,024 content combinations excludes both the 240 original
discovery items and the first 1,024 evaluation items. The same second set was
tested in three prompt formats, with five random-head controls whose edit
norms match the circuit per layer and token. Language-model cost now uses 50
WikiText validation windows. At gamma 1.25, the both-channel edit changes
candidate-normalized fidelity under the deceptive cue by **+3.84 points in
the narrative format, +3.65 in dialogue, and -1.61 in question–answer format**.
Its WikiText NLL cost is +0.01806. At gamma 1.5 the narrative gain is +6.33
points, but the NLL cost rises to +0.07179, exceeding the project's +0.05
criterion. The original natural-text cache had understated this cost.

![Prompt-format transfer and language-model cost](figures/context_transfer.png)

The gray range shows five norm-matched random-head controls for both-channel
edits; it is not a confidence interval. Colored intervals are paired normal
intervals across the 1,024 content combinations or 50 language-model windows.
The [metric diagnostics](context_fidelity/metric_diagnostics.md) motivate the
generation check below: the unchanged model assigns the true value
only 0.0000149 mean whole-vocabulary probability at the first answer position
under the deceptive cue, where an article or another opening word may come
first. In narrative and dialogue, the mean probabilities are 0.16665 and
0.30551, increasing by 0.02187 and 0.05512 with the edit. Thus the cloze gains
also exist in whole-vocabulary probability, while the QA candidate ratio is
an incomplete answer metric.

Unconstrained greedy generation on a third disjoint set of 1,024 content
combinations is now complete. With a 16-token cap, the both-channel edit at
gamma 1.25 increases explicit target-value inclusion under the deceptive cue:

| Prompt | Unedited inclusion | Edited inclusion | Difference | Paired 95% interval |
|---|---:|---:|---:|---|
| Narrative | 45.90% | 49.12% | +3.22 percentage points | [2.14, 4.30] |
| Dialogue | 87.01% | 89.75% | +2.73 percentage points | [1.74, 3.73] |
| Question–answer | 90.33% | 91.11% | +0.78 percentage points | [0.24, 1.32] |

![Target-value inclusion in unconstrained generation](figures/context_generation.png)

Neutral cues also show gains. The both-channel circuit exceeds each of the
two fixed controls in all six conditions under the unadjusted
[paired comparisons](context_fidelity/context_generation_control_pairs.md).
This compares the measured controls; two draws do not characterize every
possible random circuit.

The [generation diagnostics](context_fidelity/context_generation_diagnostics.md)
substantially narrow the interpretation. Across all 61,440 continuations,
none mentions a listed alternative candidate value. Most narrative misses
omit an attribute while naming the object, such as saying "ring" for "iron
ring". The observed gain is therefore about **explicit target-word inclusion**;
these counts do not measure lying or semantic factual-error rates. The QA
prompt asks about the observed fact, while narrative/dialogue prompts continue
a character's utterance, changing the query as well as its presentation.
The QA generation gain also reverses its first-token candidate-ratio decline,
confirming that the latter was an inadequate answer endpoint here. A further
test explicitly asks for the color, metal or animal on 1,024 additional
content combinations so that the target property is required by the question.

The [seven hyperconnection edges](context_fidelity/context_validation_hyper.md)
and [three sequential-path edges](context_fidelity/context_validation_sequential.md)
each contribute: at gamma 1.25, both-channel edits give +2.26 and +1.80 points
respectively in the narrative/deceptive condition. These effects need not add
linearly. The unchanged ten-edge selection also transfers to the
[step-10500 checkpoint](context_fidelity/context_validation_step10500.md), with
+3.79 points and +0.02037 WikiText NLL at gamma 1.25. These are content-item
and nearby-checkpoint replications, not evidence of a general honesty circuit.

All three fixed edits were also evaluated on the 14 ordinary tasks:

| Edit | Gamma | LAMBADA accuracy change | WikiText BPB cost | GSM8K joint BPB cost |
|---|---:|---:|---:|---:|
| Predictor | 1.5 | +1.22 points [0.82, 1.63] | +0.0029 | +0.0053 |
| Correction | 1.25 | +1.79 points [1.32, 2.25] | +0.0018 | +0.0045 |
| Both | 1.25 | +2.33 points [1.80, 2.85] | +0.0043 | +0.0085 |

LAMBADA rises from 30.18% to 32.51% with the both-channel edit. Its ARC-easy
accuracy falls by 0.55 points, with paired interval [-1.09, -0.04]. The
[predictor](standard_context_pred/paired_vs_dagformer.md),
[correction](standard_context_corr/paired_vs_dagformer.md) and
[both-channel](standard_context_both/paired_vs_dagformer.md) tables retain every
endpoint. The edits have task-dependent benefits and costs. Two matched
random-head ordinary-task controls are queued at the both-channel dose.

A [posthoc LAMBADA stratification](context_fidelity/lambada_answer_occurrence.md)
finds that 114 of the both-channel edit's 120 net additional correct predictions
come from the 3,791 prompts whose exact answer-token sequence already appears
in context. Accuracy in that group rises by 3.01 points [2.32, 3.67], versus
0.44 points [-0.00, 0.95] on the other 1,362 prompts. This association supports
context reuse as a working interpretation; it does not identify a causal
mediation mechanism. Exact token occurrence also misses paraphrases and
capitalization changes.

PR #2's domain-code experiment was rerun with the same 50 natural-text windows
used for the first context-fidelity test and a denser small-dose grid. The
[predictor](domain_code/verify_domain_code_pred.md) and
[correction](domain_code/verify_domain_code_corr.md) whole-circuit edits give
small shifts at acceptable NLL cost. Among the tested additive/scaling arms
below +0.05 NLL, the largest neutral-prompt shifts are +0.05257 for predictor
edits and +0.02823 for correction edits: 1.57% and 0.84% of the instruction
gap. These are descriptive maxima over the tested grid, not confirmed
optima or adjusted significance tests. Large whole-circuit changes incur
substantial NLL costs.
The default eight null draws select 1,598 correction edges; reproducing the
PR's six-draw setting selects 813 (original: 829). The selection count is
sensitive to this Monte Carlo setting even though the measured channel-level
instruction effect is nearly unchanged.

The archived verifier's automatic NULL label uses the raw-score SEM rather
than paired-change uncertainty. Its per-stream arms also omitted NLL. These
labels alone do not establish an absence of small effects. The completed
[small-dose stream follow-up](domain_streams/README.md) tests all four streams
and the joint direction at six doses, with three fixed norm-preserving
controls, natural-text NLL, and 32 new content pairs. It averages prompt
paraphrases within content items before calculating paired bootstrap intervals.
Predictor-direction effects remain small: the largest new-content shift among
tested doses below +0.05 NLL is +0.0033, or 0.09% of the instruction gap.

The correction Q direction at dose 2 gives +0.0902 [0.0626, 0.1167], 2.42% of
the new-content instruction gap, at +0.0203 NLL. V at dose 2 gives +0.0664
[0.0263, 0.1034], 1.79% of the gap, at +0.0266 NLL. Most of each score change
comes from lowering prose likelihood: code/prose log-probability changes are
+0.0186/-0.0717 for Q and +0.0099/-0.0565 for V. None of these circuit arms
changes which candidate wins. Q does not consistently beat the three fixed
controls; V's comparison depends on dose and control. The controls often
damage natural-text likelihood much more, so the comparison does not isolate
semantic selectivity at matched capability cost. The larger-dose follow-up,
including the PR's dose 64, is still pending.

## SAE direction transfer

The [eight previously selected SAE directions](sae_transfer/README.md) were
retested without reselection on 128 WikiText test windows. Three reproduce
opposite-sign target-token loss changes at alpha -4 and +4, with each dose's
paired interval excluding zero. Their other-token NLL costs remain below
0.018 in both directions:

| Feature | Target tokens / target-bearing windows | Target ΔNLL at -4 | Target ΔNLL at +4 | Dose span: +4 minus -4 |
|---|---:|---:|---:|---:|
| 3560 | 1,139 / 94 | -0.0582 | +0.0833 | +0.1415 |
| 7068 | 4,134 / 127 | -0.0521 | +0.0773 | +0.1294 |
| 4222 | 84 / 38 | -0.0563 | +0.0818 | +0.1381 |

![Historical and new-corpus SAE effects](figures/sae_transfer.png)

A [block-bootstrap check](sae_transfer/block_bootstrap.md) preserves these
three patterns with adjacent-window blocks up to length 16. Four other
directions have dose-span intervals including zero, including the earlier
kinship-associated feature 3583; feature 7019 has no matching target tokens
in this corpus and cannot be assessed. The feature token sets mix words,
fragments, punctuation and whitespace, so this supports transfer of some
fixed token-conditioned loss effects, not clean semantic labels or reliable
free-generation control.

The [target-token composition](sae_transfer/target_token_counts.json) makes
this distinction concrete: 4,032 of feature 7068's 4,134 target occurrences
are the whitespace piece `Ġ`; 998 of feature 3560's 1,139 are the
space-prefixed quote piece. The earlier kinship-associated set for 3583
contains 186 occurrences of `Ġtwo` among 246 target tokens. Aggregate rule
effects should not be assigned equally to every word in those sets.

The five coordinate-permuted controls per feature preserve direction norms
but often cause much larger other-token NLL increases. The full report shows
their costs alongside target effects. Those controls test the importance of
coordinate alignment; they do not isolate semantic specificity at equal
language-model damage.

## Reproduction and artifacts

- Evaluation environment: `lm_eval==0.4.13`, Transformers 4.57.1, PyTorch
  2.10.0+cu128; timan1 GPUs 2 and 3.
- Ordinary entry point: `experiments/results/lmeval/run_eval.py`; use
  `--save-samples --save-likelihood-samples` for paired document analysis.
- Context-fidelity entry point: `scripts/eval_context_fidelity.py`.
- Routing-dependence entry point: `scripts/eval_routing_dependence.py`;
  `scripts/prepare_wikitext_eval_cache.py` creates the disjoint split caches.
- JSONs record arguments, checkpoint steps and code versions. The
  [checkpoint manifest](provenance/checkpoint_locations.json) and
  [corpus metadata](provenance/wikitext_cache_metadata.json) identify inputs.
  Early benchmark JSONs record the Git commit at result-write time; later
  runs explicitly mark `git_commit_recorded_at: process_start`. Commits were
  made during this campaign, so the earlier field is not a process-start
  timestamp. Evaluation settings and samples are retained in either case.
- Per-document benchmark samples and raw context-fidelity arrays remain under
  this directory's ignored `samples/` and `raw/` paths. Compact summaries are
  checked into Git. Logs are under `logs/eval_20260917`.
- The current code mirror on Delta is
  `/work/hdd/bfqt/yurenh2/dagformer-eval-20260917`. The original project under
  `/projects/bfqt/users/yurenh2/ml-projects/DAGFormer` was preserved because its
  filesystem quota is full. Its uncommitted source/results were recovered and
  pushed before running these experiments.
