"""Plot dense baseline vs DAGFormer post-fix at 300M (first 1K steps)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_train_nll(csv_path):
    """Load (step, train_nll) from metrics csv, dedup steps (keep first)."""
    seen = set()
    steps, losses = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0]:
                continue
            try:
                s = int(row[0])
                nll_str = row[1]
                if not nll_str:
                    continue
                nll = float(nll_str)
            except (ValueError, IndexError):
                continue
            if s in seen:
                continue
            seen.add(s)
            steps.append(s)
            losses.append(nll)
    # Sort by step
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(losses)[order]


def smooth(x, window=5):
    """Simple moving average."""
    pad = np.concatenate([np.full(window - 1, x[0]), x])
    return np.convolve(pad, np.ones(window) / window, mode='valid')


def main():
    dense_csv = "checkpoints/pretrain_300m_baseline_5k/metrics.csv"
    dag_csv = "checkpoints/fourway_full_fix_quick/metrics.csv"

    d_step, d_nll = load_train_nll(dense_csv)
    g_step, g_nll = load_train_nll(dag_csv)

    # Restrict to first 1000 steps for fair comparison
    dmask = d_step <= 1000
    gmask = g_step <= 1000
    d_step, d_nll = d_step[dmask], d_nll[dmask]
    g_step, g_nll = g_step[gmask], g_nll[gmask]

    # Convert steps to tokens (524K per step)
    tokens_per_step = 524_288
    d_tok = d_step * tokens_per_step / 1e6  # millions
    g_tok = g_step * tokens_per_step / 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: raw train NLL trajectories ---
    ax1.plot(d_tok, d_nll, color='#888888', alpha=0.35, linewidth=0.9, label=None)
    ax1.plot(g_tok, g_nll, color='#C23B3B', alpha=0.35, linewidth=0.9, label=None)
    # Smoothed
    ax1.plot(d_tok, smooth(d_nll, 5), color='#333333', linewidth=2.2,
             label='Dense baseline (300M)')
    ax1.plot(g_tok, smooth(g_nll, 5), color='#C23B3B', linewidth=2.2,
             label='DAGFormer post-fix (300M)')

    ax1.set_xlabel('Tokens seen (M)')
    ax1.set_ylabel('Training NLL')
    ax1.set_title('300M at same token budget (first 0.52B tokens)')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(3.8, 8.0)

    # --- Right: gap (dense - DAGFormer), positive = DAG wins ---
    # Interpolate to common steps
    common_steps = np.intersect1d(d_step, g_step)
    d_interp = dict(zip(d_step, d_nll))
    g_interp = dict(zip(g_step, g_nll))
    gap_steps = np.array(sorted(common_steps))
    gaps = np.array([d_interp[s] - g_interp[s] for s in gap_steps])
    gap_tokens = gap_steps * tokens_per_step / 1e6

    ax2.axhline(0, color='#888888', linewidth=1, linestyle='--')
    ax2.fill_between(gap_tokens, 0, gaps, where=(gaps > 0),
                     color='#C23B3B', alpha=0.3, label='DAGFormer ahead')
    ax2.fill_between(gap_tokens, 0, gaps, where=(gaps <= 0),
                     color='#888888', alpha=0.3, label='Dense ahead')
    ax2.plot(gap_tokens, gaps, color='#333333', linewidth=0.8)
    ax2.plot(gap_tokens, smooth(gaps, 5), color='#C23B3B', linewidth=2.2,
             label='Gap (smoothed)')

    avg_gap = gaps[(gap_steps >= 100) & (gap_steps <= 750)].mean()
    ax2.set_xlabel('Tokens seen (M)')
    ax2.set_ylabel('NLL gap (Dense − DAGFormer)')
    ax2.set_title(f'DAGFormer advantage\n(avg +{avg_gap:.2f} nats over steps 100–750)')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    out_dir = Path("experiments/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "300m_quick_dense_vs_dagformer.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Also print summary
    print(f"\nSummary (first 1000 steps, same tokens):")
    print(f"  Dense baseline final train NLL:     {d_nll[-1]:.3f}")
    print(f"  DAGFormer post-fix final train NLL: {g_nll[-1]:.3f}")
    print(f"  Mean gap (steps 100-750):           +{avg_gap:.3f} nats (DAGFormer ahead)")
    wins = (gaps > 0).sum()
    print(f"  DAGFormer wins: {wins}/{len(gaps)} comparison points")


if __name__ == "__main__":
    main()
