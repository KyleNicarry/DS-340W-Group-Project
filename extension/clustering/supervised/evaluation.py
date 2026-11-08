"""Matched-cost wedge estimation and paired dependence-block inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def paired_interval(
    frame: pd.DataFrame, difference: np.ndarray, repetitions: int = 2000, unit: str = "event_id", weight: str = "n"
) -> tuple[float, float]:
    """Percentile bootstrap of paired weighted losses; resample whole events."""
    blocks = (
        pd.DataFrame(
            {
                "block": frame[unit].to_numpy(),
                "num": difference * frame[weight].to_numpy(),
                "den": frame[weight].to_numpy(),
            }
        )
        .groupby("block")[["num", "den"]]
        .sum()
        .to_numpy()
    )
    rng = np.random.default_rng(340)
    values = []
    for _ in range(repetitions):
        sample = blocks[rng.integers(len(blocks), size=len(blocks))].sum(axis=0)
        values.append(sample[0] / sample[1])
    return tuple(np.quantile(values, [0.025, 0.975]))


def paired_test(frame: pd.DataFrame, difference: np.ndarray, repetitions: int, unit: str = "event_id") -> dict:
    weight = frame.n.to_numpy()
    mean = np.average(difference, weights=weight)
    blocks = pd.DataFrame({"block": frame[unit].to_numpy(), "v": weight * (difference - mean)}).groupby("block").v.sum()
    se = (
        np.sqrt(np.sum(blocks.to_numpy() ** 2) * len(blocks) / (len(blocks) - 1)) / weight.sum()
        if len(blocks) > 1
        else np.nan
    )
    lo, hi = paired_interval(frame, difference, repetitions, unit)
    return {
        "improvement": mean,
        "ci_low": lo,
        "ci_high": hi,
        "se": se,
        "p_value": 2 * norm.sf(abs(mean / se)) if se > 0 else np.nan,
        "blocks": len(blocks),
    }


def matched_curves(frame: pd.DataFrame, prediction: np.ndarray, model: str, group: str = "cluster") -> pd.DataFrame:
    pieces = []
    for direction in ["YES", "NO"]:
        yes = direction == "YES"
        part = frame[["ticker", "event_id", group, "split", "n"]].copy().rename(columns={group: "cluster"})
        part["cost"] = frame.p.to_numpy() if yes else 100 - frame.p.to_numpy()
        part["direction"] = direction
        part["observed"] = frame.outcome.to_numpy() if yes else 1 - frame.outcome.to_numpy()
        part["predicted"] = prediction if yes else 1 - prediction
        part = part[part.cost.between(1, 49)].copy()
        part["obs_num"] = part.observed * part.n
        part["pred_num"] = part.predicted * part.n
        pieces.append(part)
    pos = pd.concat(pieces, ignore_index=True)
    keys = ["split", "cluster", "cost", "direction"]
    result = (
        pos.groupby(keys)
        .agg(
            n=("n", "sum"),
            contracts=("ticker", "nunique"),
            events=("event_id", "nunique"),
            obs_num=("obs_num", "sum"),
            pred_num=("pred_num", "sum"),
        )
        .reset_index()
    )
    result["observed_pp"] = 100 * result.obs_num / result.n - result.cost
    result["predicted_pp"] = 100 * result.pred_num / result.n - result.cost
    result["model"] = model
    return result


def wedge_summary(curves: pd.DataFrame) -> pd.DataFrame:
    supported = curves[curves.contracts.ge(5) & curves.events.ge(5)]
    keys = ["split", "cluster", "model", "cost"]
    wide = supported.pivot(index=keys, columns="direction", values=["observed_pp", "predicted_pp"]).dropna()
    if wide.empty or not all(d in wide.columns.get_level_values(1) for d in ["YES", "NO"]):
        return pd.DataFrame()
    rows = []
    for (split, cluster, model), group in wide.groupby(level=["split", "cluster", "model"]):
        if len(group) < 10:
            continue
        obs = group.observed_pp.mean()
        pred = group.predicted_pp.mean()
        ow, pw = obs.YES - obs.NO, pred.YES - pred.NO
        rows.append(
            {
                "split": split,
                "cluster": cluster,
                "model": model,
                "costs": len(group),
                "observed_yes_pp": obs.YES,
                "observed_no_pp": obs.NO,
                "predicted_yes_pp": pred.YES,
                "predicted_no_pp": pred.NO,
                "observed_wedge_pp": ow,
                "predicted_wedge_pp": pw,
                "absolute_error_pp": abs(ow - pw),
                "yes_absolute_error_pp": abs(obs.YES - pred.YES),
                "no_absolute_error_pp": abs(obs.NO - pred.NO),
                "sign_correct": np.sign(ow) == np.sign(pw) if abs(pw) > 1e-8 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(held: pd.DataFrame, output) -> None:
    """Exploratory complete subgroup screen; adjust across all groups and contrasts."""
    models = [
        "market_price",
        "pooled_ridge",
        "family_ridge",
        "cluster_ridge",
        "selected_pooled",
        "selected_family",
        "selected_cluster",
    ]
    pairs = [
        ("cluster_ridge", "market_price"),
        ("cluster_ridge", "pooled_ridge"),
        ("selected_cluster", "selected_pooled"),
        ("selected_cluster", "selected_family"),
        ("selected_cluster", "market_price"),
    ]
    historical = [f"historical_{scope}" for scope in ["pooled", "family", "cluster"]]
    if all(name in held for name in historical):
        models += historical
        pairs += [
            ("historical_family", "historical_pooled"),
            ("historical_cluster", "historical_family"),
            ("historical_cluster", "historical_pooled"),
            ("historical_cluster", "market_price"),
            ("selected_cluster", "historical_cluster"),
        ]
    scores, contrasts = [], []
    for (split, cluster), frame in held.groupby(["split", "cluster"]):
        y, w = frame.outcome.to_numpy(), frame.n.to_numpy()
        support = {
            "split": split,
            "cluster": cluster,
            "family": frame.family.iloc[0],
            "events": frame.event_id.nunique(),
            "contracts": frame.ticker.nunique(),
            "executions": int(w.sum()),
        }
        for name in models:
            scores.append({**support, "model": name, "brier": np.average((y - frame[name]) ** 2, weights=w)})
        for candidate, baseline in pairs:
            delta = (y - frame[baseline].to_numpy()) ** 2 - (y - frame[candidate].to_numpy()) ** 2
            mean = np.average(delta, weights=w)
            influence = (
                pd.DataFrame({"event": frame.event_id.to_numpy(), "v": w * (delta - mean)}).groupby("event").v.sum()
            )
            blocks = len(influence)
            se = np.sqrt(np.sum(influence.to_numpy() ** 2) * blocks / (blocks - 1)) / w.sum() if blocks > 1 else np.nan
            # Minimum event support is independent of outcome; untested groups get
            # p=1 in multiplicity correction so their presence cannot reduce it.
            eligible = blocks >= 30 and se > 0
            contrasts.append(
                {
                    **support,
                    "candidate": candidate,
                    "baseline": baseline,
                    "improvement": mean,
                    "se": se,
                    "normal_ci_low": mean - 1.96 * se,
                    "normal_ci_high": mean + 1.96 * se,
                    "inference_eligible": eligible,
                    "p_value": 2 * norm.sf(abs(mean / se)) if eligible else 1.0,
                    "max_event_weight_share": frame.groupby("event_id").n.sum().max() / w.sum(),
                }
            )
    scores = pd.DataFrame(scores)
    contrasts = pd.DataFrame(contrasts)
    contrasts["holm_p"] = np.nan
    for _, group in contrasts.groupby("split"):
        order = group.p_value.sort_values().index
        contrasts.loc[order, "holm_p"] = np.minimum(
            1, np.maximum.accumulate(contrasts.loc[order, "p_value"].to_numpy() * np.arange(len(order), 0, -1))
        )
    scores.to_csv(output / "cluster_metrics.csv", index=False)
    contrasts.to_csv(output / "cluster_comparisons.csv", index=False)
