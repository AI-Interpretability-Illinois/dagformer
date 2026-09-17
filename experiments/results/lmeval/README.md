# lm-eval: DAGFormer vs. baseline, reasoning first

Evaluation harness for the pretrained checkpoints in
`/work/hdd/bfqt/shared/dagformer-models` — baseline OLMo-2 and DAGFormer at
75M / 150M / 300M (plus a 600M baseline), using the same 12B-token Dolma v1.7
data source. Training budgets differ by size; the shared 300M pair uses
baseline step 12000 and DAGFormer step 9000. A matched step-9000 baseline is
retained in the project checkpoint directory.

```
prefetch_data.py   download every task's dataset (run on a login node)
run_eval.py        run lm-eval on one/many checkpoints -> reasoning/<model>__<suite>.json
compare.py         pair the JSONs into a baseline-vs-DAGFormer table
models.py          checkpoint discovery + loading (wraps scripts/eval_lm_harness.py)
harness.py         lm-eval HFLM subclass; adds cache-free generation
suites.py          task suites and which metric each task headlines
tasks/             custom lm-eval task YAMLs (gsm8k_bpb)
slurm_lmeval_reasoning.slurm   batch submission
```

## Quick start

```bash
conda activate modularity                # lm-eval 0.4.13 lives here

# once, from a login node (compute nodes may have no outbound network):
python prefetch_data.py --suite all

# on a GPU (interactive node or via sbatch):
python run_eval.py --model 300m --suite reasoning     # both families at 300M
python run_eval.py --model all  --suite reasoning --save-samples
python compare.py --write                             # -> reasoning/comparison.md
```

`--model` accepts a directory name (`300m-dagformer`), a size (`300m` — runs
both families), a family (`dagformer`), or `all`. Existing result files are
skipped unless `--overwrite`.

Batch: `sbatch slurm_lmeval_reasoning.slurm`, or e.g.
`MODELS="300m" SUITE=math sbatch slurm_lmeval_reasoning.slurm`.

## Suites

| suite | tasks |
|---|---|
| `reasoning` (default) | gsm8k, gsm8k_bpb, arc_challenge, arc_easy, mathqa, commonsense_qa, social_iqa, openbookqa, winogrande |
| `math` | gsm8k, gsm8k_bpb, mathqa, asdiv |
| `gen` | gsm8k only (smoke test / timing) |
| `core` | the 9 tasks the existing `experiments/results/lmeval/*.json` were produced with |
| `all` | `reasoning` + `core` |

## Reading gsm8k at this scale

**gsm8k exact-match will be ~0 for every model here, and that is the expected
result, not a bug.** These are 75M-600M models trained on 12B tokens; chain-of-
thought arithmetic does not emerge anywhere near this scale. The generations
are fluent and entirely wrong:

> Question: Janet's ducks lay 16 eggs per day… Answer: *The bear has been in
> the market for about 2 years. She has been in the market for about 2 years…*

So the suite reports gsm8k two ways:

- **`gsm8k`** — standard exact-match (`strict-match` needs the `#### N` format,
  `flexible-extract` takes the last number in the output). Reported because it
  is the number people ask for. A 0.000-vs-0.005 gap here is noise.
- **`gsm8k_bpb`** — *custom task* (`tasks/gsm8k_bpb.yaml`): bits-per-byte of the
  **gold** chain of thought under the model. Same rolling-loglikelihood
  machinery as the project's `wikitext` BPB, on math-reasoning text instead of
  encyclopedia text. It has no floor effect, so it still separates models that
  are all at 0% EM, and it is the number to look at for "did routing help
  reasoning?". Lower is better. It is not comparable to published gsm8k
  numbers — only across the models evaluated here.

`logiqa`/`logiqa2` are deliberately out of the default suite: both hub repos
ship a dataset loading script and need `HF_DATASETS_TRUST_REMOTE_CODE=1`. To
include one: `python prefetch_data.py --tasks logiqa --trust-remote-code`, then
`run_eval.py --tasks logiqa,...`.

## Things that matter for correctness

**Context length is 1024.** Every checkpoint was trained at `seq_len: 1024`;
RoPE is configured for 4096 but no run ever saw a longer context, so
`--max-length` defaults to the training length rather than the architectural
one. A standard 5-shot gsm8k prompt averages 872 OLMo tokens and does not fit
alongside 256 generated tokens, so the generative pass defaults to **3-shot**
(`--gen-num-fewshot`, mean 563 tokens) instead of silently letting lm-eval
left-truncate the prompt. Any truncation that does happen is counted in the
result JSON (`prompts_left_truncated_batches`,
`generations_hitting_context_limit`).

**DAGFormer has no KV cache.** Its routed attention rebuilds every layer's
per-head Q/K/V inputs from a per-token weighted sum over all prior layer
outputs, so there is no per-step state to carry. `harness.py` therefore decodes
by re-running the whole prefix per token — correct, but O(T) forwards per
sample. That is why the run is split into two passes:

| pass | what | default limit |
|---|---|---|
| log-likelihood | one forward per request, batched | full split |
| generative | one forward per *token*, rows decoded serially | `--gen-limit 200` |

The limit takes the first N documents, so every model is scored on the same
subset. Dense baselines use HF's cached `generate()` (identical greedy output,
~100× faster); `--no-kv-cache` forces them down the same recompute path when
you want the decode arithmetic identical across architectures.

**Padding.** The routed forward builds its own causal mask and takes no
attention mask, so a left-padded batch would let pad tokens into attention and
shift every RoPE position. Cache-free generation un-pads each row, decodes it
alone, and re-pads the result into the layout lm-eval expects.

**fp32 log-softmax.** Weights are bf16; log-likelihoods are accumulated in fp32
(`--softmax-dtype`) so the differences being compared are not quantisation
noise.

**Determinism/fairness.** Both families share the tokenizer, few-shot seed
(`--fewshot-seed 1234`), context length, and document subset. Loading goes
through `scripts/eval_lm_harness.py` — the project's verified loader — so the
DAGFormer predictor and routing state are loaded, not silently dropped.

## Output

`reasoning/<model>__<suite>.json` holds lm-eval's `results` / `n-shot` /
`n-samples` plus a `model` and `eval` block recording checkpoint path, context
length, few-shot counts, batch sizes, truncation counts, wall time and peak GPU
memory. `--save-samples` additionally writes per-document generations for the
generative tasks to `reasoning/samples/` (log-likelihood samples are not kept:
~47k rows per model and nothing readable in them).

### Results in this directory

- `reasoning/comparison.md` / `.csv` — the paired table (regenerate with `compare.py --write`)
- `reasoning/FINDINGS.md` — what the numbers say, and what not to quote from them
- `wikitext_control/` — the same harness on `wikitext`, as a control for
  `gsm8k_bpb`: it reproduces the BPB in the shared checkpoint README to four
  decimals for six of seven checkpoints, which is the evidence that loading and
  scoring are right. Regenerate with
  `python run_eval.py --model all --tasks wikitext --out-dir wikitext_control`.

`compare.py` pairs them per size and prints Δ = DAGFormer − baseline,
sign-corrected so **+ always means DAGFormer is better** (including for
bits-per-byte, where lower is better). `*` marks a delta larger than twice the
quadrature-combined standard error — a strict flag, since the per-document
scores needed for a proper paired test are not retained for log-likelihood
tasks.
