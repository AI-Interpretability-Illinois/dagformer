"""Audit explicit-label tasks for constant-answer collapse from saved samples."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    output = {"scope": "explicit letter/yes-no prediction only; not hidden answer-array indices",
              "tasks": {}}
    lines = ["# Explicit-label bias", "",
             "The best constant-label score uses evaluation label frequencies as a diagnostic, "
             "not as a fitted predictive model.", "",
             "| Model | Task | Accuracy | Most predicted label | Share | Best constant label accuracy |",
             "|---|---|---:|---|---:|---:|"]
    for task in ("commonsense_qa", "boolq"):
        output["tasks"][task] = {}
        for path in sorted((args.dir / "samples").glob(f"*__{task}.jsonl")):
            predictions, targets, correct = [], [], []
            for text in path.open():
                row = json.loads(text)
                if row["filter"] != "none":
                    continue
                choices = [pair[1].strip() for pair in row["arguments"]]
                prediction = int(np.argmax([pair[0] for pair in row["filtered_resps"]]))
                target = row["doc"]["answerKey"] if task == "commonsense_qa" else choices[int(row["target"])]
                predictions.append(choices[prediction])
                targets.append(target)
                correct.append(row["acc"])
            predicted_counts, target_counts = Counter(predictions), Counter(targets)
            model = path.name.removesuffix(f"__{task}.jsonl")
            dominant, count = predicted_counts.most_common(1)[0]
            rec = {"n": len(targets), "accuracy": float(np.mean(correct)),
                   "prediction_counts": dict(predicted_counts), "target_counts": dict(target_counts),
                   "dominant_prediction": dominant, "dominant_prediction_share": count / len(targets),
                   "best_constant_label_accuracy": max(target_counts.values()) / len(targets)}
            output["tasks"][task][model] = rec
            lines.append(f"| {model} | {task} | {100 * rec['accuracy']:.2f}% | {dominant} | "
                         f"{100 * rec['dominant_prediction_share']:.2f}% | "
                         f"{100 * rec['best_constant_label_accuracy']:.2f}% |")
    (args.dir / "label_bias.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.dir / "label_bias.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
