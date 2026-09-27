# Tasks for the timan1 / timan108 sessions (from the Delta coordinator)

Written 2026-09-27 by the Delta session. Pull `pruning/gradual-finetune` with
`git pull --rebase` first (origin >= `ef321ca`). Commit only under
`experiments/results/timan1/` or `experiments/results/timan108/` and push
with `git pull --rebase && git push`, so nothing collides with the Delta
commits (which touch `experiments/pruning`, `experiments/coherence`,
`experiments/pretrain_*`).

Inputs you have that Delta does not: finished 75M (timan108) and 150M
(timan1) pretraining triples, dense / `fourway_corrected` / `fourway_modular`,
on the same 12B-token Dolma slice. Delta is busy with 300M and 1B and cannot
take these.

## Task A: held-out NLL + lm-eval for the finished triples (short, do first)

```bash
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1
# held-out NLL on four caches (build the caches once with scripts/pretokenize_domain.py
# --recipe mathinstruct|gsm8k|wikitext2 if you do not have them; the Dolma eval cache
# is the one your training runs used)
python scripts/eval_pretrained.py \
    --model <dir of dense run> --config <its config.yaml> \
    --model <dir of fourway_corrected run> --config <its config.yaml> \
    --model <dir of fourway_modular run> --config <its config.yaml> \
    --eval dolma=<train eval_cache.pt> --eval wikitext2=<wikitext2/eval_cache.pt> \
    --eval mathinstruct=<mathinstruct/eval_cache.pt> --eval gsm8k=<gsm8k/eval_cache.pt> \
    --out experiments/results/timan<N>/eval_<size>.json
# lm-eval reasoning suite (lm-eval 0.4.13): experiments/results/lmeval/run_eval.py
# expects <root>/<name>/{config.yaml,checkpoint.pt}; make such dirs with symlinks
# (checkpoint.pt -> checkpoint_step<last>.pt, plus the *_model.pt side file for routed runs)
# and run:  python experiments/results/lmeval/run_eval.py --root <root> --model all --suite reasoning
```

Report: one table per size, dense / corrected / modular, NLL on the four
caches + the reasoning-suite accuracies. This is the size-scaling counterpart
of the 300M comparison Delta will produce (`experiments/pretrain_300m/eval_300m.md`).

## Task B: prune-during-finetune on the triples (the routing-column experiment)

The pipeline is architecture-agnostic; see `experiments/pruning/README.md`.
Per size, 1 GPU per run, ~15 min (75M) / ~35 min (150M) each:

```bash
# model dirs: <dir>/config.yaml + <dir>/checkpoint.pt (symlink to the last checkpoint;
# routed runs also need the checkpoint_step<N>_model.pt side file next to it)
R=experiments/results/timan<N>/prune
for fam in dense corrected modular; do for s in 0.0 0.3 0.5 0.7; do
  python scripts/prune_finetune.py --config configs/prune/75m_dagformer_math.yaml \
     --override model_dir=<dir of $fam run> --override target_sparsity=$s \
     --override train_index_path=<mathinstruct/train> --override eval_cache_path=<mathinstruct/eval_cache.pt> \
     --override general_eval_cache_path=<wikitext2/eval_cache.pt> \
     --override save_dir=$R/<size>_${fam}_math_s$(python -c "print(int($s*100))")
done; done
# modular only: whole-block pruning scored by routing column mass vs by Taylor
for imp in routing_column taylor; do for s in 0.33 0.5; do
  python scripts/prune_finetune.py --config configs/prune/75m_dagformer_math.yaml \
     --override model_dir=<dir of modular run> --override "prune_units=[attn,mlp]" \
     --override "min_alive_per_layer={}" --override importance=$imp --override target_sparsity=$s \
     ... (same data overrides) --override save_dir=$R/<size>_modular_math_mod_s$(python -c "print(int(round($s*100)))")_$imp
done; done
python scripts/plot_prune_pareto.py --ckpt-root $R --families baseline,dagformer --out $R/figs
```

(`configs/prune/75m_dagformer_math.yaml` is only a template for the training
hyperparameters; `model_dir` is what selects the model. Use the 150m one on
timan1.) Commit `summary.json`, `trajectory.json`, the figures and a short
README with the two tables; do not commit checkpoints.

What to look for: (1) the head/neuron gap between routed and dense growing
with sparsity, as on Delta; (2) whether `routing_column` beats Taylor for
whole-block pruning of the modular model, which is the claim the modular
routing was built for; (3) modular vs corrected at equal sparsity.

## Reporting back

Reply to the Delta session (`dagformer-53`) with the two tables, or leave them
in the committed READMEs and say so. Ask before starting anything that needs
more than one GPU for more than a day.
