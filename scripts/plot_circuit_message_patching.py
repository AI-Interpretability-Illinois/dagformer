"""Plot completed native-edge and common-head message interchange experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "experiments/results/circuit_interpretability_20260928/message_patching"


def main():
    result = json.loads((OUT / "results.json").read_text())
    assert result["complete"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    colors = ["#4376A8", "#D57931"]
    ks = result["selection"]["topk_sizes"]
    for j, (stage, label, color) in enumerate(zip(
            ["heldout", "transfer"], ["Held-out: period 64", "Distance transfer: period 128"], colors)):
        arms = result["stages"][stage]["arms"]
        modes = ["alpha", "content", "message"]
        metrics = [arms[f"all/{mode}/dynamic"]["summary"]["margin_recovery"] for mode in modes]
        mean = np.array([m["mean"] for m in metrics]) * 100
        ci = np.array([m["paired_95ci"] for m in metrics]) * 100
        axes[0].bar(np.arange(3) + (j - .5) * .34, mean, width=.32, color=color,
                    label=label, yerr=np.stack([mean-ci[:, 0], ci[:, 1]-mean]), capsize=3)
        metrics = [arms[f"top{k}/message"]["summary"]["margin_recovery"] for k in ks]
        mean = np.array([m["mean"] for m in metrics]) * 100
        ci = np.array([m["paired_95ci"] for m in metrics]) * 100
        axes[1].errorbar(ks, mean, yerr=np.stack([mean-ci[:, 0], ci[:, 1]-mean]),
                         marker="o", color=color, capsize=3, label=label)
        random = np.array([[arms[f"top{k}/random{i}"]["summary"]["margin_recovery"]["mean"]
                            for i in range(3)] for k in ks]) * 100
        axes[1].plot(ks, random.mean(1), "--", color=color, alpha=.7)
        axes[1].fill_between(ks, random.min(1), random.max(1), color=color, alpha=.14)
    axes[0].set_xticks(range(3), ["Coefficients", "Projected content", "Full message"])
    axes[0].set_title("Exchange at all historical-value reads")
    axes[1].set_title("Discovery-selected source edges")
    axes[1].set_xticks(ks)
    axes[1].set_xlabel("Source-edge types edited at each of 3 known positions")
    for ax in axes:
        ax.set_ylabel("Counterfactual logit-margin recovery (%)")
        ax.set_ylim(-3, 106)
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", framealpha=.95, fontsize=9)
    axes[1].text(.98, .07, "Dashed / shaded: 3 matched random masks", ha="right", transform=axes[1].transAxes, fontsize=8)
    fig.suptitle("300M FourWay: copied values travel through source content\nFrozen model; known historical token positions; upstream computation remains present", fontsize=11)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"message_interchange.{ext}")
    plt.close(fig)

    common = OUT.parent / "common_message_patching"
    if not (common / "results.json").exists():
        return
    result = json.loads((common / "results.json").read_text())
    assert result["complete"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ks = [1, 2, 4, 8, 16]
    for ax, stage, title in zip(axes, ["heldout", "transfer"],
                               ["Held-out: period 64", "Distance transfer: period 128"]):
        for kind, label, color in zip(["dense", "dag"], ["Dense", "FourWay"], colors):
            arms = result["models"][kind]["stages"][stage]["arms"]
            metrics = [arms[f"top{k}"]["summary"]["margin_recovery"] for k in ks]
            mean = np.array([m["mean"] for m in metrics]) * 100
            ci = np.array([m["paired_95ci"] for m in metrics]) * 100
            ax.errorbar(ks, mean, yerr=np.stack([mean-ci[:, 0], ci[:, 1]-mean]),
                        marker="o", color=color, capsize=3, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, ks)
        ax.set_ylim(0, 104)
        ax.set_xlabel("Head/stream sites edited at each of 3 known positions")
        ax.set_ylabel("Counterfactual logit-margin recovery (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False)
    fig.suptitle("Matched local message interventions\nSame discovery budget; sites selected separately for each model", fontsize=11)
    for ext in ["png", "pdf"]:
        fig.savefig(common / f"common_interchange.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
