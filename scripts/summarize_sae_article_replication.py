"""Compare fixed SAE effects across two samples without pairing different articles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def bidirectional(minus, plus):
    if minus["delta"] is None or plus["delta"] is None:
        return None
    a, b = minus["paired_bootstrap_95ci"], plus["paired_bootstrap_95ci"]
    return (a[1] < 0 < b[0]) or (b[1] < 0 < a[0])


def display(value):
    if value["delta"] is None:
        return "no target tokens"
    lo, hi = value["paired_bootstrap_95ci"]
    return f"{value['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    args = ap.parse_args()
    runs = [json.loads((folder / "300m.json").read_text()) for folder in (args.first, args.second)]
    if not all(run["complete"] for run in runs):
        raise ValueError("Both runs must be complete")
    for key in ("ckpt", "config", "directions", "control_mode", "controls", "alphas", "seed"):
        if runs[0]["args"][key] != runs[1]["args"][key]:
            raise ValueError(f"The fixed protocol changed: {key}")
    summaries = [json.loads((folder / "summary.json").read_text())["features"]
                 for folder in (args.first, args.second)]
    if set(summaries[0]) != set(summaries[1]):
        raise ValueError("Feature sets differ")
    output = {"first": str(args.first), "second": str(args.second),
              "comparison": "different articles, not paired observations across samples",
              "features": {}}
    lines = ["# Fixed SAE directions on additional articles", "",
             "The first sample has 128 windows from 31 WikiText test documents. The additional",
             "sample has 64 windows from 14 other documents, excluding all first-sample documents.",
             "Each intervention is paired with its own unedited reference; the two article",
             "samples are not paired with each other. All directions and controls were fixed",
             "before running the additional sample.", "",
             "The criterion below requires the alpha -4 and +4 intervals to lie on opposite",
             "sides of zero. Failure to meet it does not establish a zero effect. All intervals",
             "are unadjusted window-bootstrap intervals. The separate block tables show",
             "sensitivity to grouping adjacent windows.", "",
             "| Feature | First / additional target counts | First -4 / +4 ΔNLL | Additional -4 ΔNLL [95% CI] | Additional +4 ΔNLL [95% CI] | Opposite signs supported, first / additional |",
             "|---|---:|---:|---|---|---|"]
    costs = ["", "## Dose spans and costs", "",
             "The signed span is +4 minus -4. Other-token NLL costs are listed at both doses,",
             "without selecting the more favorable direction.", "",
             "| Feature | First span | Additional span [95% CI] | Additional other-token ΔNLL -4 / +4 |",
             "|---|---:|---|---:|"]
    for fid, first in summaries[0].items():
        second = summaries[1][fid]
        if first["tokens"] != second["tokens"]:
            raise ValueError("Target-token sets changed")
        status = [bidirectional(r["on_minus"], r["on_plus"]) for r in (first, second)]
        labels = ["no targets" if value is None else "yes" if value else "not established" for value in status]
        def doses(record):
            return (f"{record['on_minus']['delta']:+.4f} / {record['on_plus']['delta']:+.4f}"
                    if record['on_minus']['delta'] is not None else "no targets")
        output["features"][fid] = {"tokens": first["tokens"],
            "first_count": first["on_plus"]["n_tokens"], "additional_count": second["on_plus"]["n_tokens"],
            "bidirectional_window_intervals": {"first": status[0], "additional": status[1]},
            "first_span": first["span"], "additional_span": second["span"],
            "additional_minus": {k: second["on_minus"][k] for k in ("delta", "paired_bootstrap_95ci")},
            "additional_plus": {k: second["on_plus"][k] for k in ("delta", "paired_bootstrap_95ci")}}
        lines.append(f"| {fid} | {first['on_plus']['n_tokens']} / {second['on_plus']['n_tokens']} | "
                     f"{doses(first)} | {display(second['on_minus'])} | {display(second['on_plus'])} | {' / '.join(labels)} |")
        original_span = f"{first['span']['delta']:+.4f}" if first['span']['delta'] is not None else "no targets"
        costs.append(f"| {fid} | {original_span} | {display(second['span'])} | "
                     f"{second['off_minus']['delta']:+.4f} / {second['off_plus']['delta']:+.4f} |")
    lines += costs + ["", "[All additional-sample results](README.md), [individual tokens](individual_tokens.md),",
                       "[whole-head control contrasts](control_comparisons.md), and",
                       "[block-bootstrap components](block_bootstrap.md) retain the full evidence.",
                       "The [pre-run protocol](../provenance/sae_replication_protocol.md) records sample selection."]
    (args.second / "replication_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.second / "replication_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
