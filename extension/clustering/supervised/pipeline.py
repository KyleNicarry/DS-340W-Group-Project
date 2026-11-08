"""Reproducible supervised comparisons on the supplied event-disjoint CSV splits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..mispricing.pipeline import fingerprint
from .evaluation import matched_curves, paired_test, subgroup_metrics, wedge_summary
from .historical import HISTORICAL_MODELS, historical_predict
from .models import FEATURES, KINDS, SCOPES, ResidualModel, combination_mask, design, text_features, validate_splits


def train_predict(frame: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every scope on training labels only; never use validation for fitting."""
    validate_splits(frame)
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    held = frame[frame.split.ne("train")].reset_index(drop=True).copy()
    xt, xh = design(train), design(held)
    y, w = train.outcome.to_numpy(), train.n.to_numpy(dtype=float)
    held["market_price"] = held.p / 100
    fits, coefficients = [], []
    rng = np.random.default_rng(340)
    for scope in SCOPES:
        groups = [("ALL", np.arange(len(train)))] if scope == "pooled" else train.groupby(scope).indices.items()
        for group, indices in groups:
            hi = np.arange(len(held)) if scope == "pooled" else np.flatnonzero(held[scope].eq(group).to_numpy())
            if len(hi) == 0:
                continue
            for kind in KINDS:
                ti = indices if kind == "ridge" or len(indices) <= 50000 else rng.choice(indices, 50000, replace=False)
                model = ResidualModel(kind).fit(xt[ti], y[ti], w[ti])
                name = f"{scope}_{kind}"
                if name not in held:
                    held[name] = np.nan
                held.loc[hi, name] = model.predict(xh[hi])
                fits.append(
                    {
                        "scope": scope,
                        "group": group,
                        "kind": kind,
                        "training_cells": len(indices),
                        "fitted_cells": len(ti),
                        "training_events": train.iloc[indices].event_id.nunique(),
                    }
                )
                if kind == "ridge":
                    for feature, coef, center, scale in zip(FEATURES, model.model.coef_, model.center, model.scale):
                        coefficients.append(
                            {
                                "scope": scope,
                                "group": group,
                                "feature": feature,
                                "coefficient": coef,
                                "center": center,
                                "scale": scale,
                                "intercept": model.model.intercept_,
                            }
                        )
        print(f"Fitted {scope}: {len(fits)} cumulative models", flush=True)
    pd.DataFrame(coefficients).to_csv(output / "ridge_coefficients.csv", index=False)
    return held, pd.DataFrame(fits)


def select_models(held: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict]:
    """Select algorithm and correction strength using validation Brier only."""
    v = held[held.split.eq("validation")]
    rows, selected = [], {}
    # Market-price shrinkage protects sparse local models against large corrections.
    # A zero strength is an explicit no-signal option, not a model failure.
    for scope in SCOPES:
        for kind in KINDS:
            for strength in [0, 0.1, 0.25, 0.5, 0.75, 1]:
                q = v.market_price + strength * (v[f"{scope}_{kind}"] - v.market_price)
                rows.append(
                    {
                        "scope": scope,
                        "kind": kind,
                        "strength": strength,
                        "validation_brier": np.average((v.outcome - q) ** 2, weights=v.n),
                    }
                )
        candidates = [r for r in rows if r["scope"] == scope]
        best = min(candidates, key=lambda r: r["validation_brier"])
        selected[scope] = best
        held[f"selected_{scope}"] = held.market_price + best["strength"] * (
            held[f"{scope}_{best['kind']}"] - held.market_price
        )
    # Partial pooling: tune the local correction relative to the selected pooled model.
    for scope in ["family", "cluster"]:
        candidates = []
        base = held.selected_pooled
        for kind in KINDS:
            for strength in [0, 0.1, 0.25, 0.5, 0.75, 1]:
                q = base + strength * (held[f"{scope}_{kind}"] - base)
                mask = held.split.eq("validation")
                row = {
                    "scope": f"partial_{scope}",
                    "kind": kind,
                    "strength": strength,
                    "validation_brier": np.average(
                        (held.loc[mask, "outcome"] - q[mask]) ** 2, weights=held.loc[mask, "n"]
                    ),
                }
                candidates.append(row)
                rows.append(row)
        best = min(candidates, key=lambda r: r["validation_brier"])
        selected[f"partial_{scope}"] = best
        held[f"partial_{scope}"] = base + best["strength"] * (held[f"{scope}_{best['kind']}"] - base)
    pd.DataFrame(rows).to_csv(output / "validation_search.csv", index=False)
    (output / "selected_models.json").write_text(json.dumps(selected, indent=2) + "\n")
    return held, selected


def evaluate(held: pd.DataFrame, output: Path, bootstrap: int) -> None:
    models = ["market_price"] + [f"{s}_{k}" for s in SCOPES for k in KINDS]
    models += [f"selected_{s}" for s in SCOPES] + ["partial_family", "partial_cluster"]
    models += HISTORICAL_MODELS
    metrics, comparisons, curves, wedges, coverage = [], [], [], [], []
    pairs = [
        ("cluster_ridge", "pooled_ridge"),
        ("cluster_ridge", "family_ridge"),
        ("cluster_ridge", "market_price"),
        ("selected_pooled", "market_price"),
        ("selected_family", "selected_pooled"),
        ("selected_cluster", "selected_pooled"),
        ("selected_cluster", "selected_family"),
        ("partial_family", "selected_pooled"),
        ("partial_cluster", "selected_pooled"),
        ("selected_family", "market_price"),
        ("selected_cluster", "market_price"),
        ("partial_family", "market_price"),
        ("partial_cluster", "market_price"),
        ("historical_pooled", "market_price"),
        ("historical_family", "historical_pooled"),
        ("historical_cluster", "historical_family"),
        ("historical_cluster", "historical_pooled"),
        ("historical_family", "market_price"),
        ("historical_cluster", "market_price"),
        ("selected_pooled", "historical_pooled"),
        ("selected_family", "historical_family"),
        ("selected_cluster", "historical_cluster"),
        ("pooled_ridge", "historical_pooled"),
        ("family_ridge", "historical_family"),
        ("cluster_ridge", "historical_cluster"),
    ]
    for split, frame in held.groupby("split"):
        print(f"Evaluating {split}: {len(frame):,} cells", flush=True)
        y = frame.outcome.to_numpy()
        contract_weights = 1 / frame.groupby("ticker").ticker.transform("size").to_numpy()
        coverage.append(
            {
                "split": split,
                "contracts": frame.ticker.nunique(),
                "events": frame.event_id.nunique(),
                "clusters": frame.cluster.nunique(),
                "executions": int(frame.n.sum()),
                "cells": len(frame),
            }
        )
        for name in models:
            q = frame[name].to_numpy()
            loss = (y - q) ** 2
            metrics.append(
                {
                    "split": split,
                    "model": name,
                    "brier": np.average(loss, weights=frame.n),
                    "residual_mae_pp": 100 * np.average(np.abs(y - q), weights=frame.n),
                    "residual_rmse_pp": 100 * np.sqrt(np.average(loss, weights=frame.n)),
                    "unit_brier": np.average(loss, weights=frame.units),
                    "contract_balanced_brier": np.average(loss, weights=contract_weights),
                    "improvement_vs_price": np.average((y - frame.market_price) ** 2 - loss, weights=frame.n),
                }
            )
            c = matched_curves(frame, q, name)
            curves.append(c)
            wedges.append(wedge_summary(c))
        for candidate, baseline in pairs:
            delta = (y - frame[baseline].to_numpy()) ** 2 - (y - frame[candidate].to_numpy()) ** 2
            for unit in ["event_id", "series"]:
                result = paired_test(frame, delta, bootstrap, unit)
                comparisons.append(
                    {"split": split, "candidate": candidate, "baseline": baseline, "dependence_unit": unit, **result}
                )
    comparisons = pd.DataFrame(comparisons)
    # Holm correction over all reported comparisons, separately by split/block unit.
    comparisons["holm_p"] = np.nan
    for _, group in comparisons.groupby(["split", "dependence_unit"]):
        order = group.p_value.sort_values().index
        comparisons.loc[order, "holm_p"] = np.minimum(
            1, np.maximum.accumulate(comparisons.loc[order, "p_value"].to_numpy() * np.arange(len(order), 0, -1))
        )
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    comparisons.to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "holdout_coverage.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output / "matched_cost_curves.csv.gz", index=False)
    wedge = pd.concat(wedges, ignore_index=True)
    wedge.to_csv(output / "cluster_wedges.csv", index=False)
    wedge.groupby(["split", "model"]).agg(
        clusters=("cluster", "size"),
        wedge_mae_pp=("absolute_error_pp", "mean"),
        sign_agreement=("sign_correct", "mean"),
        wedge_rmse_pp=("absolute_error_pp", lambda x: np.sqrt(np.mean(x**2))),
        yes_mae_pp=("yes_absolute_error_pp", "mean"),
        no_mae_pp=("no_absolute_error_pp", "mean"),
    ).reset_index().to_csv(output / "wedge_metrics.csv", index=False)
    subgroup_metrics(held, output)
    # Per-contract deviations describe the observed execution window, not an equal-cost wedge.
    base = held[["ticker", "split", "event_id", "cluster", "family", "n"]].copy()
    base["observed_yes_pp"] = (held.outcome - held.market_price) * 100 * held.n
    for name in models:
        base[name] = (held[name] - held.market_price) * 100 * held.n
    keys = ["ticker", "split", "event_id", "cluster", "family"]
    agg = base.groupby(keys).sum(numeric_only=True).reset_index()
    for column in ["observed_yes_pp"] + models:
        agg[column] /= agg.n
    agg.to_csv(output / "contract_predictions.csv.gz", index=False)


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    audit = json.loads((args.stats.parent / "data_audit.json").read_text())
    inputs = fingerprint([args.data / "master/markets.csv", args.data / "master/trades.csv", args.assignments])
    if inputs != audit["inputs"]:
        raise ValueError("Audited CSV/assignment inputs changed; rerun mispricing extraction")
    cells = pd.read_csv(args.stats, keep_default_na=False)
    cells = cells[cells.cluster.ne("") & cells.assignment_status.eq("assigned")].copy()
    assignments = pd.read_csv(
        args.assignments, usecols=["ticker", "event_ticker", "split", "cluster", "family"], keep_default_na=False
    )
    # Assert current labels equal cached labels, in addition to fingerprint checks.
    checked = (
        cells[["ticker", "event_id", "split", "cluster", "family"]]
        .drop_duplicates()
        .merge(assignments, on="ticker", validate="one_to_one", suffixes=("", "_current"))
    )
    if (
        len(checked) != cells.ticker.nunique()
        or not all(checked[c].eq(checked[c + "_current"]).all() for c in ["split", "cluster", "family"])
        or not checked.event_id.eq(checked.event_ticker).all()
    ):
        raise ValueError("Cached labels do not match frozen assignments")
    # Authoritative membership from actual split CSVs, not a new random split.
    split_tables = []
    for split in ["train", "validation", "test"]:
        m = pd.read_csv(
            args.data / f"split/{split}/markets.csv",
            usecols=["ticker", "event_ticker", "title", "yes_sub_title"],
            keep_default_na=False,
        )
        m["split"] = split
        split_tables.append(m)
    contracts = pd.concat(split_tables, ignore_index=True)
    if contracts.ticker.duplicated().any():
        raise ValueError("Duplicate contract in split CSVs")
    validate_splits(contracts.rename(columns={"event_ticker": "event_id"}))
    labels = (
        cells[["ticker", "split"]]
        .drop_duplicates()
        .merge(contracts[["ticker", "split"]], on="ticker", how="left", validate="one_to_one", suffixes=("", "_csv"))
    )
    if not labels.split.eq(labels.split_csv).all():
        raise ValueError("Assignments disagree with actual CSV split membership")
    validate_splits(cells)
    excluded = contracts.loc[combination_mask(contracts), "ticker"]
    removed = cells[cells.ticker.isin(excluded)]
    removed.groupby("split").agg(
        contracts=("ticker", "nunique"), events=("event_id", "nunique"), executions=("n", "sum")
    ).reset_index().to_csv(args.output / "combination_exclusions.csv", index=False)
    cells = cells[~cells.ticker.isin(excluded)].copy()
    audit["supervised_combination_screen"] = {
        "rule": "MVE prefixes plus explicit multi-award win category conjunctions",
        "excluded_traded_contracts": int(removed.ticker.nunique()),
        "excluded_executions": int(removed.n.sum()),
        "designed_after": "exploratory subgroup screen exposed shared-ceremony combinations",
    }
    support = (
        cells[cells.split.eq("train")]
        .groupby("cluster")
        .agg(contracts=("ticker", "nunique"), events=("event_id", "nunique"))
        .reset_index()
    )
    support["eligible"] = support.contracts.ge(100) & support.events.ge(20)
    support.to_csv(args.output / "training_support.csv", index=False)
    before = (
        cells.groupby("split")
        .agg(contracts=("ticker", "nunique"), events=("event_id", "nunique"), executions=("n", "sum"))
        .reset_index()
    )
    before.to_csv(args.output / "assigned_population.csv", index=False)
    cells = cells[cells.cluster.isin(support.loc[support.eligible, "cluster"])].copy()
    cells = cells.drop(columns=["title"]).merge(text_features(contracts), on="ticker", validate="many_to_one")
    cells["series"] = cells.ticker.str.split("-").str[0]
    return cells, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/kalshi_recent_year_csv"))
    parser.add_argument("--assignments", type=Path, default=Path("output/clustering/assignments.csv"))
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("src/analysis/kalshi/clustering/mispricing/results/sufficient_statistics.csv.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("extension/clustering/supervised/results"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "completed": False,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seed": 340,
        "features": FEATURES,
        "primary": "historical calibration transfer: pooled vs structural vs NLP clusters; identical execution observations",
        "revision": "historical baseline added after earlier exploratory outcomes were inspected",
        "secondary": "validation-selected algorithm and strength; partial pooling",
        "split": "existing CSV event holdout; retrospective and previously explored, not temporal",
        "minimum_training_contracts": 100,
        "minimum_training_events": 20,
        "tree_fit_cell_cap": 50000,
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Loading and validating audited CSV statistics and actual split membership...", flush=True)
    frame, audit = load_data(args)
    manifest["audit"] = audit
    manifest["inputs"] = fingerprint(
        [args.stats, args.assignments] + [args.data / f"split/{s}/markets.csv" for s in ["train", "validation", "test"]]
    )
    print(f"Matched population: {len(frame):,} cells, {frame.ticker.nunique():,} contracts", flush=True)
    with threadpool_limits(limits=2):
        held, fits = train_predict(frame, args.output)
        fits.to_csv(args.output / "model_fits.csv", index=False)
        held, selected = select_models(held, args.output)
        held, historical_selected = historical_predict(frame, held, args.output)
        held.to_parquet(args.output / "heldout_predictions.parquet", index=False)
        evaluate(held, args.output, args.bootstrap)
    manifest["historical_selected"] = historical_selected
    manifest["selected"] = selected
    manifest["fits"] = len(fits)
    manifest["versions"] = {
        name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "duckdb"]
    }
    manifest["source_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
    }
    from .report import write_report

    write_report(args.output, Path(__file__).parent / "REPORT.md")
    manifest["completed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed supervised analysis.", flush=True)


if __name__ == "__main__":
    main()
