# Validation record

- Full unit suite: **68 passed**, one PyTorch nested-tensor warning, in 10.43
  seconds. Command: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
  OPENBLAS_NUM_THREADS=4 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python
  -m pytest tests -q`. Log: `logs/eval_20260917/full_pytest.log`.
- The suite covers the existing routing components plus checkpoint loading
  for encoder/static/position-table variants, simultaneous head editing,
  matched random-head harness hooks, paired statistics, result grouping,
  generated-value extraction and copy-task boundary scoring.
- CPU end-to-end checks exercised the custom answer-text CommonsenseQA task,
  fixed-head ordinary-harness interventions, and the unconstrained context
  generation runner before GPU evaluation.
- An eight-document CPU check exercised the GSM8K question and conditional
  answer likelihood tasks. Saved per-document likelihood/byte pairs reproduce
  both aggregate BPB values exactly; answer byte counts include only the
  gold-answer suffix, including its leading space. All 1,319 complete
  question/answer strings fit in 418 tokenizer tokens or fewer.
- Future-token prefix checks passed at all three backbone scales. The
  position-table harness hook and direct routing-dependence evaluator agreed
  exactly on the checked NLL. Results are in
  [causality_and_hook_checks.json](causality_and_hook_checks.json).
- Six result figures were rendered and inspected; see
  `figures/qa-ledger.md`. Plotting reads the numeric result files directly.
- The SAE transfer runner passed a one-window CPU execution check
  plus three focused tests: token-weighted conditional loss aggregation,
  exact preservation of each layer/stream's coefficient multiset in controls,
  and paired uncertainty for the positive-minus-negative dose span.
  A fourth test checks the additional whole-head control: source positions,
  Q/K/V alignment and shared R coefficients remain fixed.
  A fifth checks that feature-minus-control dose spans retain all four arms'
  pairing, including cancellation of shared window variation.
  The CPU execution check is an implementation check, not a scientific result.
- Two domain-stream tests passed: paraphrases are averaged within
  content items, and permuted controls preserve each layer/stream's coefficient
  multiset while remaining on eligible hyperconnections.
- The MHA fastpath control confirms actual fused encoder execution changes
  from 64 calls to zero on each 32-window run; see
  [the numerical comparison](routing_dependence/fastpath.md).
- A direct verifier reporting check confirms that reference prompts with
  unequal counts report an unavailable paired t statistic as JSON `null`
  and table `n/a`. Four legacy `NaN` placeholders in this campaign's two
  whole-circuit JSONs were normalized; scores, NLLs and shifts are unchanged.
- Two prompt-cluster tests verify that duplicating identical prompts cannot
  narrow the clustered uncertainty, and that item-weighted and equally
  weighted unique-prompt effects remain distinct estimands.
- The original four encoder-model optimizer states match every inspected
  predictor/correction tensor by group order, count and shape; see
  [optimizer participation](provenance/optimizer_activity.md).
- All 80 feature-minus-control block-bootstrap point estimates across both
  article samples agree with
  the independent-window paired estimates. Only the resampling unit changes.
- The additional 64-window cache uses 14 test documents and excludes all 31
  earlier test documents, with matching dataset fingerprints. Raw records
  confirm the same 128 intervention names, doses, direction norms and control
  seeds across the two samples.
- Delta readback verified all 324 harness endpoints and 765,856 saved rows,
  plus the completed 49/128/160/128-arm follow-ups and all 13 checkpoint bundles.
  The additional cache loads on Delta with shape 64 × 1,024, correct next-token
  label shifts, and zero overlap with the earlier article set.
- All five routing-dependence runs contain the expected 12 arms, with 128
  natural-text windows and 16 synthetic sequences per repetition period in
  each arm. The three literal harness commands in [REPRODUCE.md](REPRODUCE.md)
  pass CPU `--dry-run` resolution for their stated models and tasks.

These checks validate implementation and artifact consistency. Evaluation
uncertainty is reported separately in the paired result tables.
