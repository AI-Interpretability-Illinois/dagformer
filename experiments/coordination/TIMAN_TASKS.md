# Coordination: timan1 / timan108 sessions and the Delta chains

Updated 2026-09-27 17:10 CDT by the Delta session (`dagformer-53`), after
reading the timan state over ssh. Home (`/home/xy51`, shared by both timan
hosts) is at 100% of its 10T quota with ~43G free: keep every checkpoint,
cache and log under `/srv/local/xy51` as you already do, and keep commits
small.

## State (what already exists, do not redo)

| where | done | running |
|---|---|---|
| timan1 (4x A6000; GPU0 holds another user's 34 GB vLLM, GPU3 7.6 GB of another user) | 75M + 150M triples dense / fourway_corrected / fourway_modular on the local 12B Dolma corpus; lm-eval + held-out NLL (`experiments/results/lmeval/timan1_dolma12b/`); the full 75M prune-during-finetune sweep incl. `routing_column` (`experiments/results/pruning/timan1_75m_math/`) | nothing on GPUs 1-3 |
| timan108 (4x A5000 24 GB) | | 300M `fourway_modular`, 12B corpus, step ~5460/12000 (~2.4 days left) |
| Delta gpua046 (4x A100) | 75M locality arms (global / local / both / per_layer / dense / modular / modular_sparse) on a 1.7B slice | 1B dense baseline, 5B tokens (~Sep 28 05:00), then the 300M-selected routed 1B at 5B tokens |
| Delta gpua047 (4x A100) | | 300M `fourway_modular` -> `fourway_corrected` on a 21B corpus, then an automatic comparison that picks the 1B architecture, then `modular_corrected` |

Both places now have a 300M modular run, on different corpora (12B local vs
21B Delta). That is a replication at two data budgets, not waste, provided
the *pair* exists on each corpus (see task 2).

## Task 1 (timan1, GPUs 1-2 now): 150M prune-during-finetune sweep

Exactly the 75M protocol (`experiments/results/pruning/timan1_75m_math/README.md`,
its `runlist.txt` and `configs/prune/75m_*_math_timan1.yaml`) applied to the
150M triple in `/srv/local/xy51/checkpoints/pretrain_150m_*_dolma12b`:
s0 / s30 / s50 / s70 heads+neurons for the three families, whole blocks at
33% / 50% for the three families, plus `routing_column` for the modular ones.
Use `configs/prune/150m_*_math.yaml` as the hyperparameter base (micro 8 x
accum 4; drop to micro 4 x accum 8 if the A6000 is short on memory), 1 GPU
per run. Commit under `experiments/results/pruning/timan1_150m_math/` with
the same README layout as the 75M one. The 75M result to test at 150M: the
modular advantage appearing at 70% (2.551 vs 2.669 vs 2.815) and the
whole-block picture (modular Taylor best at 50%, routing_column best at 33%).

## Task 2 (timan108, after its 300M modular finishes): the 300M pair on the 12B corpus

Run 300M `fourway_corrected` with the same config apart from `routing_mode`
(and, if there is time, the 300M dense baseline) so the 12B-corpus 300M
comparison is a matched pair like Delta's 21B-corpus one. The Delta
comparison (`experiments/pretrain_300m/eval_300m.md`, expected ~Sep 30)
decides the 1B architecture by held-out Dolma NLL; a second corpus agreeing
or disagreeing is the most useful thing timan108 can add. Owner's call on
whether to spend the ~3 days.

## Task 3 (timan1, when a GPU is free, cheap): evaluate the Delta 75M locality arms

The arms are on Delta (`/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints/locality/*`,
150-300 MB each); Delta cannot spare a GPU until ~Sep 28. If you can pull
them (scp from dt-login01), run `scripts/eval_pretrained.py` on the four
caches and the lm-eval `default` suite, and put the table under
`experiments/results/timan1_locality_75m/`. Otherwise skip; Delta will do it.

## Rules

- Only commit under `experiments/results/pruning/timan1_*`,
  `experiments/results/lmeval/timan*`, `experiments/results/timan1_*` and
  `configs/prune/*_timan*.yaml`; `git pull --rebase` before every push.
  Delta commits touch `experiments/pruning`, `experiments/coherence`,
  `experiments/pretrain_*`, `configs/pretrain_1b`, `configs/pretrain_300m`,
  `configs/locality`, `scripts/slurm`.
- The Delta session can reach both timan hosts by ssh (`ssh timan1`) and
  will read results from the repo; it will not start GPU jobs on timan
  without saying so in this file first.
