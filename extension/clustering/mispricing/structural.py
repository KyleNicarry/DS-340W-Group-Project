"""Compare rule-only structural families using the audited recent-year CSV trades."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from ..features import extract_attributes
from .heterogeneity import centered_max_test
from .pipeline import fingerprint
from .statistics import curve, estimate, positions, sparse_residuals


def structural_labels(markets: pd.DataFrame) -> pd.DataFrame:
    """Compute rule-only families afresh, including previously unassigned titles."""
    valid = (
        markets.market_type.eq("binary")
        & ~markets.ticker.str.startswith(("KXMVE", "MVE"))
        & markets.title.str.strip().ne("")
        & markets.event_ticker.str.strip().ne("")
    )
    selected = markets[valid]
    rows = []
    for row in selected.to_dict("records"):
        attrs = extract_attributes(row)
        rows.append(
            {
                "ticker": row["ticker"],
                "structural_group": attrs["family"],
                "domain": attrs["domain"],
                "underlying": attrs["underlying"],
                "structure": attrs["structure"],
                "attribute_source": attrs["attribute_source"],
                "text": attrs["text"],
            }
        )
    return pd.DataFrame(rows)


def equal_support_test(pos: pd.DataFrame, summary: pd.DataFrame) -> list[dict]:
    names = summary.loc[summary.n_costs.eq(49), "cluster"].tolist()
    if len(names) < 2:
        return [
            {"dependence_unit": unit, "tested_groups": len(names), "p_value": None} for unit in ["event_id", "ticker"]
        ]
    pos = pos[pos.cluster.isin(names) & pos.cost.between(1, 49)].copy()
    cs = curve(pos, True)
    index = {name: i for i, name in enumerate(names)}
    weights = sparse.coo_matrix(
        (np.where(cs.direction.eq("YES"), 100 / 49, -100 / 49), (cs.cluster.map(index), np.arange(len(cs)))),
        shape=(len(names), len(cs)),
    ).tocsr()
    values = np.asarray(weights @ cs.win_rate).ravel()
    rows = []
    for unit in ["event_id", "ticker"]:
        residuals = sparse_residuals(pos, cs, unit, True)
        influences = weights @ residuals
        n = residuals.shape[1]
        cov = (influences @ influences.T).toarray() * n / (n - 1)
        test = centered_max_test(values, cov, ["ALL"] * len(names), repetitions=100000)
        test["tested_groups"] = test.pop("tested_clusters")
        rows.append({"dependence_unit": unit, **test})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/kalshi_recent_year_csv"))
    parser.add_argument("--cache", type=Path, default=Path("extension/clustering/mispricing/results"))
    parser.add_argument("--output", type=Path, default=Path("extension/clustering/mispricing/results/structural"))
    args = parser.parse_args()
    audit = json.loads((args.cache / "data_audit.json").read_text())
    inputs = fingerprint([args.data / "master/markets.csv", args.data / "master/trades.csv"])
    if inputs != audit["inputs"][:2]:
        raise ValueError("CSV inputs do not match audited statistics; rerun the base mispricing extraction first")
    # Cached learned labels are used only to define the explicit same-population sensitivity.
    assignment_path = Path(audit["inputs"][2]["path"])
    if fingerprint([assignment_path])[0] != audit["inputs"][2]:
        raise ValueError("Original assignments changed; refresh the audited statistics before population matching")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "completed": False,
        "inputs": inputs,
        "cache": fingerprint([args.cache / "sufficient_statistics.csv.gz"]),
        "grouping": "rule-only domain/underlying/structure; no encoder or clustering model",
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Recomputing structural attributes from market CSV titles and YES subtitles...", flush=True)
    markets = pd.read_csv(
        args.data / "master/markets.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["ticker", "event_ticker", "market_type", "title", "yes_sub_title"],
    )
    labels = structural_labels(markets)
    labels.to_csv(args.output / "structural_assignments.csv", index=False)
    cells = pd.read_csv(args.cache / "sufficient_statistics.csv.gz", keep_default_na=False)
    cells["previously_ml_assigned"] = cells.cluster.ne("")
    cells = cells.drop(columns=["cluster", "family"]).merge(
        labels[["ticker", "structural_group"]], on="ticker", validate="many_to_one"
    )
    cells = cells.rename(columns={"structural_group": "cluster"})
    summaries = []
    tests = []
    equality = []
    pooled = []
    coverage = []
    with threadpool_limits(limits=2):
        for population in ["all_structural", "same_ml_population"]:
            subset = cells if population == "all_structural" else cells[cells.previously_ml_assigned]
            coverage.append(
                {
                    "population": population,
                    "contracts": subset.ticker.nunique(),
                    "events": subset.event_id.nunique(),
                    "executions": int(subset.n.sum()),
                    "units": float(subset.units.sum()),
                    "structural_groups": subset.cluster.nunique(),
                }
            )
            for weighting, column in [("execution", "n"), ("unit", "units")]:
                print(f"{population}, {weighting}: estimating wedges...", flush=True)
                pos = positions(subset, column)
                c = curve(pos, True)
                c["population"] = population
                c["weighting"] = weighting
                c.to_csv(args.output / f"curves_{population}_{weighting}.csv", index=False)
                p = curve(pos)
                p["population"] = population
                p["weighting"] = weighting
                pooled.append(p)
                bands = [((1, 49), 10)]
                if population == "all_structural":
                    bands += [((1, 10), 6), ((11, 20), 6), ((21, 30), 6), ((31, 40), 6), ((41, 49), 6)]
                for band, mincost in bands:
                    result, test = estimate(pos, band, mincost, 20, 10, weighting)
                    result["population"] = population
                    summaries.append(result)
                    tests.append(
                        {"population": population, "weighting": weighting, "band": f"{band[0]}-{band[1]}", **test}
                    )
                    if band == (1, 49):
                        for row in equal_support_test(pos, result):
                            equality.append({"population": population, "weighting": weighting, **row})
    pd.concat(summaries, ignore_index=True).to_csv(args.output / "structural_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(args.output / "heterogeneity_tests.csv", index=False)
    pd.DataFrame(equality).to_csv(args.output / "fixed_support_equality_tests.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output / "population_coverage.csv", index=False)
    pd.concat(pooled, ignore_index=True).to_csv(args.output / "pooled_curves.csv", index=False)
    from .structural_report import write_report

    target = Path(__file__).parent / "STRUCTURAL_FINDINGS.md"
    write_report(args.output, args.cache, target)
    manifest["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
    }
    manifest["features_sha256"] = hashlib.sha256(
        Path(__file__).parent.parent.joinpath("features.py").read_bytes()
    ).hexdigest()
    manifest["versions"] = {
        name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "duckdb", "matplotlib"]
    }
    manifest["completed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed: {target}", flush=True)


if __name__ == "__main__":
    main()
