"""Localize damage-induced routing compensation in a frozen FourWay model.

Replay selected groups of post-lesion effective routing coefficients while all
other groups remain at their clean values. No source content is transplanted.
Selection uses discovery only; confirmation inputs have independent seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from circuit_damage_mechanism import Runner, spec, pair_stats
from circuit_message_patching import make_data
from interp_common import load_eval_ids


def metric(logits, a, b):
    lp = logits.log_softmax(-1)
    return {
        'nll': -lp.gather(1, a[:, None])[:, 0],
        'correct_margin': logits.gather(1, a[:, None])[:, 0] - logits.gather(1, b[:, None])[:, 0],
        'accuracy': (logits.argmax(-1) == a).float(),
    }


def evaluate(runner, data, choices, primary, batch_size):
    values = {name: {} for name in ['clean', 'A_dynamic', 'A_fixed', 'A_full_replay', *choices]}
    replay_error = 0.0
    for start in range(0, len(data['ids']), batch_size):
        ids = data['ids'][start:start + batch_size].cuda()
        a = data['a'][start:start + batch_size].cuda()
        b = data['b'][start:start + batch_size].cuda()
        baseline, clean = runner.forward(ids, data['positions'], capture=True)
        dynamic, damaged = runner.forward(ids, data['positions'], spec('A_dynamic', [primary]),
                                          clean=clean, capture=True)
        fixed, _ = runner.forward(ids, data['positions'], spec('A_fixed', [primary], fixed=True), clean=clean)
        replay, _ = runner.forward(ids, data['positions'], spec('A_full_replay', [primary], fixed=True), clean=damaged)
        replay_error = max(replay_error, (replay - dynamic).abs().max().item())

        def record(name, logits):
            for field, tensor in metric(logits, a, b).items():
                values[name].setdefault(field, []).extend(tensor.cpu().tolist())

        for name, logits in [('clean', baseline), ('A_dynamic', dynamic), ('A_fixed', fixed), ('A_full_replay', replay)]:
            record(name, logits)
        for name, choice in choices.items():
            groups = {tuple(key) for key in choice['groups']}
            hybrid = {key: (z, damaged[key][1] if key in groups else alpha)
                      for key, (z, alpha) in clean.items()}
            heads = [primary] if choice.get('lesion', True) else []
            logits, _ = runner.forward(ids, data['positions'], spec(name, heads, fixed=True), clean=hybrid)
            record(name, logits)
        if start == 0 or (start + len(ids)) % 32 == 0:
            print('evaluated', start + len(ids), '/', len(data['ids']), flush=True)
    assert replay_error == 0, replay_error
    arms = {name: {'values': fields, 'summary': {k: pair_stats(v) for k, v in fields.items()}}
            for name, fields in values.items()}
    fixed_nll = np.asarray(values['A_fixed']['nll'])
    dynamic_nll = np.asarray(values['A_dynamic']['nll'])
    total_repair = pair_stats(fixed_nll - dynamic_nll)
    for name, choice in choices.items():
        ref = fixed_nll if choice.get('lesion', True) else np.asarray(values['clean']['nll'])
        arms[name]['nll_rescue_vs_fixed_or_clean'] = pair_stats(ref - np.asarray(values[name]['nll']))
    return dict(seed=data['seed'], n_pairs=len(data['ids']) // 2, period=data['period'], query=data['query'],
                complete_replay_max_logit_error=replay_error, normal_routing_repair=total_repair, arms=arms)


def report(result, out):
    lines = ['# Localization of damage-induced routing compensation', '',
             'Frozen 300M FourWay + local correction. Zero one attention head output at all token positions. Capture the complete clean and damaged effective coefficients (predictor + correction). Keep the lesion and all clean coefficients, then restore selected layer/stream groups to their damaged-run values. Source activations are always computed by the recipient.', '',
             'A group is all head/source coefficients of Q, K, V, or R in one layer at all tokens. This is a coarse response-localization experiment, not a sparse-edge or full-circuit claim. The donor is the same input and checkpoint under the lesion; no answer or other-example content is introduced.', '',
             'Replaying every damaged-run coefficient must reproduce the normal damaged model exactly. Global predictor inputs are unchanged, so coefficient changes arise from local correction. Candidate groups are only downstream of the lesion. Discovery ranks their individual NLL rescue; top-k unions are not asserted to be optimal.', '',
             'This is offline causal replay of damage-induced coefficients: it tests the sufficiency of recorded changes under other clean-coefficient clamps, rather than the necessity of an online feedback loop. Random controls match layer-group counts; their selected-set overlaps and any identical sets are recorded in final_arms.', '',
             f"Primary head (zero-based): {result['primary']}. Selection: {result.get('selection', {})}", '']
    for split in ('heldout', 'transfer', 'unchanged_query'):
        if split not in result['stages']:
            continue
        stage = result['stages'][split]
        lines += [f'## {split}', '', f"Pairs: {stage['n_pairs']}. Full replay logit error: {stage['complete_replay_max_logit_error']}.", '',
                  '| Arm | NLL | Accuracy | NLL rescue [paired 95% CI] |', '|---|---:|---:|---:|']
        for name, arm in stage['arms'].items():
            s = arm['summary']
            rescue = arm.get('nll_rescue_vs_fixed_or_clean')
            text = '—' if rescue is None else f"{rescue['mean']:+.4f} [{rescue['paired_95ci'][0]:+.4f}, {rescue['paired_95ci'][1]:+.4f}]"
            lines.append(f"| {name} | {s['nll']['mean']:.4f} | {100*s['accuracy']['mean']:.2f}% | {text} |")
        lines += ['', 'Positive rescue is lower NLL than the lesion with clean coefficients held fixed. Uninjured-response controls instead compare against the uninjured model. Paired bootstrap groups both value-swap directions; uncertainty is over input pairs, not model seeds.', '']
        if 'contrasts' in stage:
            lines += ['| Contrast | NLL benefit [paired 95% CI] |', '|---|---:|']
            for name, estimate in stage['contrasts'].items():
                lo, hi = estimate['paired_95ci']
                lines.append(f"| {name} | {estimate['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
            lines += ['', 'Recovered fractions divide the mean NLL rescue by mean total routing compensation; they do not describe a fraction of the whole task circuit.', '',
                      str(stage['recovered_fraction']), '']
    (out / 'README.md').write_text('\n'.join(lines).rstrip() + '\n')


def add_contrasts(result):
    for split in ('heldout', 'transfer', 'unchanged_query'):
        if split not in result['stages']:
            continue
        stage = result['stages'][split]
        values = {name: np.asarray(arm['values']['nll']) for name, arm in stage['arms'].items()}
        denominator = (values['A_fixed'] - values['A_dynamic']).reshape(-1, 2).mean(1)
        rng = np.random.default_rng(2026092944)
        indices = rng.integers(len(denominator), size=(4000, len(denominator)))
        stage['contrasts'], stage['recovered_fraction'] = {}, {}
        for k in (1, 2, 4, 8):
            name = f'top{k}'
            rescue = values['A_fixed'] - values[name]
            uninjured_rescue = values['clean'] - values[f'uninjured_top{k}']
            stage['contrasts'][f'{name}_damage_specific_rescue'] = pair_stats(rescue - uninjured_rescue)
            numerator = rescue.reshape(-1, 2).mean(1)
            ratio = numerator[indices].mean(1) / denominator[indices].mean(1)
            stage['recovered_fraction'][name] = dict(mean=float(numerator.mean() / denominator.mean()),
                                                    paired_95ci=np.quantile(ratio, [.025, .975]).tolist())
            for i in range(3):
                control = f'random{k}_{i}'
                stage['contrasts'][f'{name}_minus_{control}'] = pair_stats(values[control] - values[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('checkpoints/pr_sync_20260917/300m-dagformer'))
    parser.add_argument('--out', type=Path, default=Path('experiments/results/circuit_followup_20260929/route_repair'))
    parser.add_argument('--cache', default='checkpoints/pr_sync_20260917/eval_corpora/wikitext_train.pt')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--discovery-pairs', type=int, default=32)
    parser.add_argument('--test-pairs', type=int, default=64)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        result = json.loads((args.out / 'results.json').read_text())
        add_contrasts(result)
        (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        report(result, args.out)
        return
    assert args.batch_size % 2 == 0
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    args.out.mkdir(parents=True, exist_ok=True)
    corpus, _ = load_eval_ids(args.cache)
    datasets = {
        'discovery': make_data(corpus, args.discovery_pairs, 64, 2026092940),
        'heldout': make_data(corpus, args.test_pairs, 64, 2026092941),
        'transfer': make_data(corpus, args.test_pairs, 128, 2026092942),
        'unchanged_query': make_data(corpus, args.test_pairs, 64, 2026092941, query=15),
    }
    primary = (6, 11)  # fixed by the preceding, independent copy-localization study
    runner = Runner('dag', args.model)
    started = time.time()
    result = dict(complete=False, primary=primary, started=started,
                  git_commit_at_start=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, stages={})
    def save():
        result['elapsed_seconds'] = time.time() - started
        (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        report(result, args.out)
    with runner.installed():
        candidates = [(layer, stream) for layer in range(primary[0] + 1, runner.L) for stream in 'qkvr']
        choices = {f'group_{l}_{s}': dict(groups=[(l, s)]) for l, s in candidates}
        discovery = evaluate(runner, datasets['discovery'], choices, primary, args.batch_size)
        result['stages']['discovery'] = discovery
        ordered = sorted(candidates, key=lambda group: discovery['arms'][f'group_{group[0]}_{group[1]}']['nll_rescue_vs_fixed_or_clean']['mean'], reverse=True)
        result['selection'] = dict(ordered_groups=ordered, discovery_pairs=args.discovery_pairs,
                                   individual_nll_rescue=[discovery['arms'][f'group_{l}_{s}']['nll_rescue_vs_fixed_or_clean']['mean'] for l, s in ordered])
        final = {}
        rng = np.random.default_rng(2026092943)
        for k in (1, 2, 4, 8):
            chosen = ordered[:k]
            final[f'top{k}'] = dict(groups=chosen)
            final[f'uninjured_top{k}'] = dict(groups=chosen, lesion=False)
            for i in range(3):
                for attempt in range(100):
                    shuffled = []
                    for layer in sorted({l for l, _ in chosen}):
                        n = sum(l == layer for l, _ in chosen)
                        shuffled += [(layer, str(s)) for s in rng.choice(list('qkvr'), n, replace=False)]
                    if set(shuffled) != set(chosen):
                        break
                final[f'random{k}_{i}'] = dict(groups=shuffled,
                                              selected_overlap=len(set(shuffled) & set(chosen)),
                                              identical_to_selected=set(shuffled) == set(chosen))
        result['final_arms'] = final
        save()
        for split in ('heldout', 'transfer', 'unchanged_query'):
            result['stages'][split] = evaluate(runner, datasets[split], final, primary, args.batch_size)
            save()
            print('DONE', split, result['stages'][split]['normal_routing_repair'], flush=True)
    result['complete'] = True
    add_contrasts(result)
    save()
    print('COMPLETE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
