# Validation record

- Full unit suite: **56 passed**, one PyTorch nested-tensor warning, in 11.39
  seconds. Command: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
  OPENBLAS_NUM_THREADS=4 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python
  -m pytest tests -q`. Log: `logs/eval_20260917/full_pytest.log`.
- The suite covers the existing routing components plus checkpoint loading
  for encoder/static/position-table variants, simultaneous head editing,
  paired statistics, result grouping and generated-value extraction.
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
  `routing_dependence/causality_and_hook_checks.json`.
- Four result figures were rendered and inspected; see
  `figures/qa-ledger.md`. Plotting reads the numeric result files directly.
- The subsequent SAE transfer runner passed a one-window CPU execution check
  plus two focused tests: token-weighted conditional loss aggregation and
  exact preservation of each layer/stream's coefficient multiset in controls.
  The CPU execution check is an implementation check, not a scientific result.
- Two domain-stream tests subsequently passed: paraphrases are averaged within
  content items, and permuted controls preserve each layer/stream's coefficient
  multiset while remaining on eligible hyperconnections.

These checks validate implementation and artifact consistency. Evaluation
uncertainty is reported separately in the paired result tables.
