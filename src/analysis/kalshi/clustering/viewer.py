"""Generate an offline title-review browser from clustering assignments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

FIELDS = ("ticker", "event_ticker", "title", "yes_sub_title", "split", "cluster", "status", "similarity", "family")


def build_viewer(assignments: Path, output: Path) -> int:
    """Embed actual assignments, with script delimiters escaped for safe HTML."""
    with assignments.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Assignments CSV is missing required title-review columns")
        rows = [[row[field] for field in FIELDS] for row in reader]
    payload = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    identity = hashlib.sha256(payload.encode()).hexdigest()[:16]
    template = Path(__file__).with_name("viewer.html").read_text()
    html = template.replace("__DATA__", payload.replace("<", "\\u003c")).replace("__RUN__", identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=Path("output/clustering/assignments.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/clustering/title-review.html"))
    args = parser.parse_args()
    count = build_viewer(args.assignments, args.output)
    print(f"Embedded {count:,} contracts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
