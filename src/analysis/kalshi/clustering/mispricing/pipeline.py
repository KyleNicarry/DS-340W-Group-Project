"""Rerun paper-style ungrouped and cluster-level wedges on the recent-year CSVs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .data import extract
from .statistics import curve, estimate, positions


def fingerprint(paths: list[Path]) -> list[dict]:
    return [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/kalshi_recent_year_csv"))
    parser.add_argument("--assignments", type=Path, default=Path("output/clustering/assignments.csv"))
    parser.add_argument("--output", type=Path, default=Path("src/analysis/kalshi/clustering/mispricing/results"))
    parser.add_argument(
        "--reuse-stats", action="store_true", help="Reuse audited sufficient statistics only if input metadata match"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = fingerprint([args.data / "master/markets.csv", args.data / "master/trades.csv", args.assignments])
    audit_path = args.output / "data_audit.json"
    if args.reuse_stats and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["inputs"] != inputs:
            raise ValueError("Cached statistics input metadata do not match")
        cells = pd.read_csv(args.output / "sufficient_statistics.csv.gz", keep_default_na=False)
    else:
        print("Scanning CSVs and auditing trade IDs...", flush=True)
        cells, audit = extract(args.data, args.assignments, args.output)
        audit["inputs"] = inputs
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"{len(cells):,} contract-price sufficient-statistic rows", flush=True)
    manifest = {
        "completed": False,
        "inputs": inputs,
        "primary": "execution-weighted, pooled splits, costs 1-49",
        "config": vars(args).copy(),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
    }
    manifest["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest["config"].items()}
    manifest["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    assigned = cells[cells.cluster.ne("")].copy()
    pops = {
        "all_resolved": cells,
        "non_combination": cells[cells.non_combination.eq(True)],
        "assigned_matched": assigned,
    }
    coverage = []
    curves = []
    summaries = []
    tests = []
    for name, subset in pops.items():
        coverage.append(
            {
                "population": name,
                "contracts": subset.ticker.nunique(),
                "events": subset.event_id.nunique(),
                "executions": int(subset.n.sum()),
                "units": float(subset.units.sum()),
            }
        )
        for weighting, column in [("execution", "n"), ("unit", "units")]:
            c = curve(positions(subset, column))
            c["population"] = name
            c["weighting"] = weighting
            curves.append(c)
    pd.DataFrame(coverage).to_csv(args.output / "population_coverage.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(args.output / "pooled_curves.csv", index=False)
    with threadpool_limits(limits=2):
        for weighting, column in [("execution", "n"), ("unit", "units")]:
            print(f"Constructing {weighting}-weighted positions...", flush=True)
            pos = positions(assigned, column)
            c = curve(pos, True)
            c["weighting"] = weighting
            c.to_csv(args.output / f"cluster_curves_{weighting}.csv", index=False)
            for band, mincost in [
                ((1, 49), 10),
                ((1, 10), 6),
                ((11, 20), 6),
                ((21, 30), 6),
                ((31, 40), 6),
                ((41, 49), 6),
            ]:
                print(f"Inference: {weighting} costs {band}", flush=True)
                rows, test = estimate(pos, band, mincost, 20, 10, weighting)
                summaries.append(rows)
                tests.append(dict(weighting=weighting, reference="pooled", band=f"{band[0]}-{band[1]}", **test))
            print(f"Inference: {weighting} within structural families", flush=True)
            rows, test = estimate(pos, (1, 49), 10, 20, 10, weighting, reference="family")
            summaries.append(rows)
            tests.append(dict(weighting=weighting, reference="family", band="1-49", **test))
            # Descriptive split consistency uses frozen assignments, not model selection.
            split_curves = []
            for split, g in pos.groupby("split"):
                c = curve(g, True)
                c["split"] = split
                split_curves.append(c)
            pd.concat(split_curves).to_csv(args.output / f"split_curves_{weighting}.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(args.output / "cluster_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(args.output / "heterogeneity_tests.csv", index=False)
    from .report import write_report

    write_report(args.output, Path(__file__).parent / "FINDINGS.md")
    manifest["versions"] = {
        name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "duckdb", "matplotlib"]
    }
    manifest["fixed_support_equality_completed"] = True
    manifest["report_sha256"] = hashlib.sha256((Path(__file__).parent / "FINDINGS.md").read_bytes()).hexdigest()
    manifest["completed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed. Report: " + str(Path(__file__).parent / "FINDINGS.md"), flush=True)


if __name__ == "__main__":
    main()
