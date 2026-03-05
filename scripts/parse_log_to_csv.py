"""Parse pretrain_baseline.py stdout logs into CSV.

Usage:
    python scripts/parse_log_to_csv.py logs/pretrain_300m_16110921.out -o checkpoints/pretrain_300m_baseline/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys


def parse_log(log_path: str) -> list[dict[str, str]]:
    """Parse [step N] key=value lines from stdout log."""
    # Pattern: [step 500] key1=val1, key2=val2, ...
    line_re = re.compile(r"^\[step (\d+)\] (.+)$")
    kv_re = re.compile(r"([\w/]+)=([\d.eE+\-]+)")

    rows: list[dict[str, str]] = []
    for line in open(log_path):
        line = line.strip()
        m = line_re.match(line)
        if not m:
            continue
        step = m.group(1)
        kvs = kv_re.findall(m.group(2))
        if not kvs:
            continue
        row = {"step": step}
        for k, v in kvs:
            row[k] = v
        rows.append(row)
    return rows


def merge_same_step(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge rows with the same step (e.g. train + eval on same step)."""
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        step = row["step"]
        if step not in merged:
            merged[step] = {}
            order.append(step)
        merged[step].update(row)
    return [merged[s] for s in order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse pretrain log to CSV")
    parser.add_argument("log_file", help="Path to stdout log file")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    rows = parse_log(args.log_file)
    if not rows:
        print("No metric lines found.", file=sys.stderr)
        sys.exit(1)

    rows = merge_same_step(rows)

    # Collect all columns
    columns = ["step"]
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

    if args.output:
        out.close()
        print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
