"""Build training curves from the unmodified CSVs in this handoff directory."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'manifest.json').read_text())
out = ROOT / 'curves'
out.mkdir(exist_ok=True)
series = {}
summary = []
for name, rule in manifest['curve_rules'].items():
    if not isinstance(rule, dict):
        continue
    loss_key = 'train/nll' if 'dagformer' in name else 'train/loss'
    chunks = []
    for phase in ['historical', 'continuation']:
        path = ROOT / 'raw' / f'{name}_{phase}_metrics.csv'
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        data = pd.DataFrame({'step': pd.to_numeric(raw['step']),
                             'tokens_B': pd.to_numeric(raw['train/tokens_seen_B']),
                             'train_nll': pd.to_numeric(raw[loss_key]),
                             'lr': pd.to_numeric(raw['train/lr'])}).dropna()
        if phase == 'historical' and 'keep_historical_step_max_inclusive' in rule:
            data = data[data.step <= rule['keep_historical_step_max_inclusive']]
        data['phase'] = phase
        data['source_file'] = str(path.relative_to(ROOT))
        # Smoothing is for display only and never spans the restart boundary.
        data['display_rolling21_nll'] = data.train_nll.rolling(21, min_periods=1).mean()
        chunks.append(data)
    data = pd.concat(chunks, ignore_index=True).sort_values('step')
    assert not data.step.duplicated().any()
    assert np.isfinite(data[['train_nll', 'tokens_B', 'lr']]).all().all()
    assert np.allclose(data.tokens_B, (data.step + 1) * 524288 / 1e9)
    expected = rule['final_optimizer_updates'] // 10
    assert len(data) == expected, (name, len(data), expected)
    data.to_csv(out / f'{name}_train.csv', index=False)
    series[name] = data
    summary.append({'run': name, 'logged_rows': len(data),
                    'last_logged_step': int(data.iloc[-1].step),
                    'last_logged_tokens_B': float(data.iloc[-1].tokens_B),
                    'last_logged_nll': float(data.iloc[-1].train_nll),
                    'last21_mean_nll': float(data.tail(21).train_nll.mean()),
                    'final_optimizer_updates': rule['final_optimizer_updates'],
                    'final_checkpoint_tokens_B': manifest['completion_proofs'][name]['processed_tokens']/1e9})
pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)

colors = {'baseline': '#2864a0', 'dagformer': '#be532f'}
for zoom in [False, True]:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), layout='constrained')
    for ax, size, dataset, resume in zip(axes, ['1b','600m'], ['OLMo-mix', 'Dolma'], [31001,14001]):
        for kind in ['baseline', 'dagformer']:
            data = series[f'{size}-{kind}']
            if zoom:
                data = data[data.tokens_B >= .5]
            for phase, part in data.groupby('phase', sort=False):
                ax.plot(part.tokens_B, part.train_nll, color=colors[kind], alpha=.13, lw=.6)
                ax.plot(part.tokens_B, part.display_rolling21_nll, color=colors[kind], lw=1.6,
                        label=('Dense' if kind=='baseline' else 'FourWay + correction') if phase=='historical' else None)
        boundary = resume*524288/1e9
        ax.axvline(boundary, color='#666666', ls='--', lw=1)
        ax.text(boundary-.15, .97, 'Sep continuation', transform=ax.get_xaxis_transform(),
                rotation=90, va='top', ha='right', color='#555555', fontsize=8)
        ax.set(title=f'{size.upper()} — {dataset}', xlabel='Processed training tokens (billions)',
               ylabel='Training minibatch NLL (nats/token)')
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Training loss — raw batches (faint), trailing 21 logged batches (solid)' +
                 ('\nAfter 0.5B tokens' if zoom else ''), fontsize=12)
    name = 'loss_curves_after_warmup' if zoom else 'loss_curves'
    fig.savefig(out/f'{name}.png', dpi=180)
    fig.savefig(out/f'{name}.pdf')
    plt.close(fig)
print(pd.DataFrame(summary).to_string(index=False))
