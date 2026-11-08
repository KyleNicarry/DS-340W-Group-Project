"""Training-only, event-support-smoothed historical calibration transfer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import SCOPES, validate_splits

WIDTHS = (1, 5, 10)
STRENGTHS = (10, 50, 200, 1000, None)  # None means inherit the parent exactly.
HISTORICAL_MODELS = [f"historical_{scope}" for scope in SCOPES]


class HistoricalCalibration:
    """Execution-weighted residual means; shrink using effective event counts.

    Pooled estimates shrink to zero correction, families to pooled estimates,
    and learned clusters to their structural family. Estimates never use
    held-out labels, and missing group/price support inherits its parent.
    """

    def __init__(self, width: int):
        if width not in WIDTHS:
            raise ValueError("Unsupported price width")
        self.width = width

    def fit(self, train: pd.DataFrame) -> HistoricalCalibration:
        if not train.split.eq("train").all():
            raise ValueError("Historical estimates require training rows only")
        self.tables = {}
        for scope in SCOPES:
            frame = train[["p", "outcome", "n", "event_id"]].copy()
            frame["group"] = "ALL" if scope == "pooled" else train[scope].to_numpy()
            frame["bucket"] = (frame.p.astype(int) - 1) // self.width
            frame["numerator"] = frame.n * (frame.outcome - frame.p / 100)
            keys = ["group", "bucket"]
            stats = frame.groupby(keys).agg(
                weight=("n", "sum"), numerator=("numerator", "sum"), events=("event_id", "nunique")
            )
            event_weights = frame.groupby(keys + ["event_id"]).n.sum().pow(2).groupby(level=[0, 1]).sum()
            stats["effective_events"] = stats.weight**2 / event_weights
            stats["residual"] = stats.numerator / stats.weight
            self.tables[scope] = stats
        return self

    def lookup(self, frame: pd.DataFrame, scope: str) -> tuple[np.ndarray, np.ndarray]:
        group = np.repeat("ALL", len(frame)) if scope == "pooled" else frame[scope].to_numpy()
        index = pd.MultiIndex.from_arrays(
            [group, (frame.p.astype(int).to_numpy() - 1) // self.width], names=["group", "bucket"]
        )
        values = self.tables[scope].reindex(index)
        return values.residual.to_numpy(), values.effective_events.fillna(0).to_numpy()

    def correction(self, frame: pd.DataFrame, scope: str, strength, parent: np.ndarray) -> np.ndarray:
        if strength is None:
            return parent.copy()
        residual, support = self.lookup(frame, scope)
        fraction = support / (support + strength)
        return parent + fraction * (np.nan_to_num(residual) - parent)

    def export(self) -> pd.DataFrame:
        tables = []
        for scope, stats in self.tables.items():
            table = stats.reset_index()
            table["scope"], table["width"] = scope, self.width
            table["price_low"] = table.bucket * self.width + 1
            table["price_high"] = np.minimum(99, (table.bucket + 1) * self.width)
            tables.append(table)
        return pd.concat(tables, ignore_index=True)


def historical_predict(frame: pd.DataFrame, held: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict]:
    """Select price resolution on pooled validation; local scopes share that grid."""
    validate_splits(frame)
    train = frame[frame.split.eq("train")]
    validation = held.split.eq("validation").to_numpy()
    if not validation.any():
        raise ValueError("Validation rows required for historical selection")
    price = held.p.to_numpy() / 100
    y = held.loc[validation, "outcome"].to_numpy()
    weight = held.loc[validation, "n"].to_numpy()
    rows, estimates, selected = [], [], {}
    zero = np.zeros(len(held))

    def score(scope, width, strength, correction):
        q = np.clip(price + correction, 0, 1)
        row = {
            "scope": scope,
            "width": width,
            "strength": strength,
            "validation_brier": float(np.average((y - q[validation]) ** 2, weights=weight)),
        }
        rows.append(row)
        return row

    best = None
    fitted = {}
    for width in WIDTHS:
        model = HistoricalCalibration(width).fit(train)
        fitted[width] = model
        estimates.append(model.export())
        for strength in STRENGTHS:
            correction = model.correction(held, "pooled", strength, zero)
            row = score("pooled", width, strength, correction)
            if best is None or row["validation_brier"] < best["validation_brier"]:
                best, pooled = row, correction
    selected["pooled"] = best
    model = fitted[best["width"]]
    held["historical_pooled"] = np.clip(price + pooled, 0, 1)
    parent = pooled
    for scope in ["family", "cluster"]:
        candidates = []
        for strength in STRENGTHS:
            correction = model.correction(held, scope, strength, parent)
            row = score(scope, model.width, strength, correction)
            candidates.append((row, correction))
        row, correction = min(candidates, key=lambda item: item[0]["validation_brier"])
        selected[scope] = row
        held[f"historical_{scope}"] = np.clip(price + correction, 0, 1)
        parent = correction
    pd.concat(estimates, ignore_index=True).to_csv(output / "historical_training_estimates.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "historical_validation_search.csv", index=False)
    (output / "historical_selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    return held, selected
