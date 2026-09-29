"""Plot observed damage compensation and composition capability results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def paired_estimate(x):
    x = np.asarray(x).reshape(-1, 2).mean(1)
    rng = np.random.default_rng(20260929)
    means = x[rng.integers(len(x), size=(4000, len(x)))].mean(1)
    return x.mean(), *np.quantile(means, [.025, .975])


def save(fig, root, name):
    fig.savefig(root / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(root / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/results/circuit_followup_20260929'))
    args = parser.parse_args()
    root = args.root
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    damage = json.loads((root / 'damage_mechanism/results.json').read_text())
    repair = json.loads((root / 'route_repair/results.json').read_text())
    assert damage['complete'] and repair['complete']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), layout='constrained')
    colors = ['#237a99', '#cc7940']
    splits = ['heldout', 'transfer']
    for offset, (arm, label) in enumerate([('A_dynamic', 'Normal routing response'), ('A_fixed', 'Clean routing held fixed')]):
        estimates = []
        for split in splits:
            stage = damage['models']['dag']['stages'][split]
            vals = np.asarray(stage['arms'][arm]['values']['target_nll'])
            clean = np.asarray(stage['arms']['clean']['values']['target_nll'])
            estimates.append(paired_estimate(vals - clean))
        mean, lo, hi = np.asarray(estimates).T
        axes[0].bar(np.arange(2) + (offset - .5) * .32, mean, width=.30, color=colors[offset], label=label,
                    yerr=[mean - lo, hi - mean], capsize=3)
    axes[0].set_xticks(range(2), ['Heldout', 'Longer distance'])
    axes[0].set_ylabel('Added next-token NLL after damage (nats)')
    axes[0].set_title('A. Routing responses reduce damage')
    axes[0].legend(frameon=False, fontsize=9)
    for split, color, label in zip(splits, colors, ['Heldout', 'Longer distance']):
        s = repair['stages'][split]['recovered_fraction']
        estimates = np.asarray([[s[f'top{k}']['mean'], *s[f'top{k}']['paired_95ci']] for k in [1, 2, 4, 8]]) * 100
        mean, lo, hi = estimates.T
        axes[1].errorbar([1, 2, 4, 8], mean, yerr=[mean - lo, hi - mean], color=color, marker='o', capsize=3, label=label)
    axes[1].axhline(100, color='0.6', linestyle='--', linewidth=1)
    axes[1].set_xticks([1, 2, 4, 8])
    axes[1].set_xlabel('Restored layer / stream groups')
    axes[1].set_ylabel('Total routing compensation recovered (%)')
    axes[1].set_title('B. Recorded coefficient changes are sufficient')
    axes[1].legend(frameon=False, fontsize=9, loc='lower right')
    save(fig, root, 'damage_and_route_repair')

    names = ['dense300m', 'dag300m', 'dense1b', 'dag1b']
    paths = [root / 'composition' / name / 'results.json' for name in names]
    if all(p.exists() for p in paths):
        results = [json.loads(p.read_text()) for p in paths]
        if all(d.get('complete') for d in results):
            fig, ax = plt.subplots(figsize=(10.5, 4.5), layout='constrained')
            tasks = [('first_hop', 'First lookup'), ('second_hop', 'Second lookup'),
                     ('two_hop', 'Implicit two lookups'), ('supplied_middle', 'Correct middle key supplied')]
            palette = ['#7198ae', '#237a99', '#cc7940', '#7a9e53']
            for offset, ((task, label), color) in enumerate(zip(tasks, palette)):
                scores = []
                for d in results:
                    key = f"boxes/4/{task}"
                    s = d['stages']['natural_fourshot']['summary'][key]
                    scores.append([s['candidate_accuracy'], *s['candidate_accuracy_wilson95']])
                mean, lo, hi = np.asarray(scores).T * 100
                ax.bar(np.arange(4) + (offset - 1.5) * .19, mean, width=.18, color=color, label=label,
                       yerr=[mean - lo, hi - mean], capsize=2)
            ax.axhline(100 / 3, color='0.5', linestyle='--', linewidth=1, label='Three-choice chance')
            ax.set_xticks(range(4), ['Dense 300M', 'DAG 300M', 'Dense 1B', 'DAG 1B'])
            ax.set_ylim(0, 105)
            ax.set_ylabel('Heldout accuracy among three candidates (%)')
            ax.set_title('Natural-language lookup task: same four-shot format')
            ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper left')
            save(fig, root, 'composition_capability')


if __name__ == '__main__':
    main()
