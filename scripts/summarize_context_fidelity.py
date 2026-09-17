"""Make compact reports while keeping paired item data in the raw artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_context_fidelity import paired_summary


def summarize(source, out_dir):
    raw = json.loads(source.read_text())
    ref = raw["reference"]
    result = {k: raw[k] for k in ("args", "git_commit", "versions", "checkpoint_step", "circuit", "protocol")}
    result["raw_artifact"] = str(source)
    result["n_items"] = len(raw["items"])
    result["reference"] = {"cloze": {cue: {m: float(np.mean(v)) for m, v in values.items()}
                                      for cue, values in ref["cloze"].items()},
                           "nll": float(np.mean(ref["nll"])), "n_sequences": len(ref["nll"])}
    result["arms"] = {}
    lines = [f"# Context fidelity — {source.stem}", "",
             f"Fixed {len(raw['circuit'])}-edge circuit; {len(raw['items'])} new content items; "
             f"{len(ref['nll'])} natural-text sequences.",
             "Effects are paired against the unchanged checkpoint. p_true is normalized "
             "over the candidate values, not the entire vocabulary.", "",
             "| Channel | Gamma | Neutral Δp | Deceptive Δp | ΔNLL (nats) | NLL rise > 0.05 |",
             "|---|---:|---:|---:|---:|---|"]
    for name, arm in raw["arms"].items():
        rec = {"edges": arm["edges"], "summary": arm["summary"], "cloze": {}}
        for cue, metrics in arm["cloze"].items():
            rec["cloze"][cue] = {metric: paired_summary(values, ref["cloze"][cue][metric])
                                   for metric, values in metrics.items()}
        if {"neutral", "deceptive"}.issubset(arm["cloze"]):
            arm_gap = np.array(arm["cloze"]["deceptive"]["p_true"]) - arm["cloze"]["neutral"]["p_true"]
            ref_gap = np.array(ref["cloze"]["deceptive"]["p_true"]) - ref["cloze"]["neutral"]["p_true"]
            rec["deceptive_minus_neutral_effect"] = paired_summary(arm_gap, ref_gap)
        result["arms"][name] = rec
        if "/circuit/" in name:
            channel, _, gamma = name.split("/")
            s = arm["summary"]
            effect = lambda cue: f"{s[cue]['delta']:+.4f}" if cue in s else "—"
            lines.append(f"| {channel} | {gamma.removeprefix('gamma')} | {effect('neutral')} | "
                         f"{effect('deceptive')} | {s['nll']['delta']:+.5f} | "
                         f"{'yes' if s['damaged'] else 'no'} |")
    lines += ["", "Controls: " + raw["protocol"]["controls"] + ".",
              "Paired intervals and all control arms are in the adjacent summary JSON. "
              "They describe these content combinations, not unseen training runs.", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / (source.stem + ".summary.json")).write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / (source.stem + ".md")).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    for source in args.inputs:
        summarize(source, args.out_dir)
        print(source, "summarized")


if __name__ == "__main__":
    main()
