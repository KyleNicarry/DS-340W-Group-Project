"""Exact-price estimators and market/event-cluster influence-function inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm


def fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values within the specified comparison family."""
    p = np.asarray(pvalues)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(q, 1)
    return result


def positions(cells: pd.DataFrame, weight: str) -> pd.DataFrame:
    cols = ["ticker", "event_id", "cluster", "split", "p", "outcome", weight]
    c = cells[cols].copy()
    yes = c.assign(cost=c.p, direction="YES", weight=c[weight], wins=c[weight] * c.outcome)
    no = c.assign(cost=100 - c.p, direction="NO", weight=c[weight], wins=c[weight] * (1 - c.outcome))
    return pd.concat([yes, no], ignore_index=True).drop(columns=["p", "outcome"])


def curve(pos: pd.DataFrame, group: bool = False, family: bool = False) -> pd.DataFrame:
    keys = (["family"] if family else ["cluster"] if group else []) + ["cost", "direction"]
    stats = (
        pos.groupby(keys)
        .agg(
            weight=("weight", "sum"),
            wins=("wins", "sum"),
            contracts=("ticker", "nunique"),
            events=("event_id", "nunique"),
        )
        .reset_index()
    )
    stats["win_rate"] = stats.wins / stats.weight
    stats["mispricing_pp"] = 100 * stats.win_rate - stats.cost
    stats["return_pct"] = 100 * stats.mispricing_pp / stats.cost
    return stats


def sparse_residuals(
    pos: pd.DataFrame, stats: pd.DataFrame, unit: str, group: bool, family: bool = False
) -> sparse.csr_matrix:
    """Rows are (group, price, direction) means; columns are dependence units."""
    keys = (["family"] if family else ["cluster"] if group else []) + ["cost", "direction"]
    indexed = stats.reset_index(drop=True).reset_index(names="row")
    merged = pos.merge(
        indexed[keys + ["row", "win_rate", "weight"]].rename(columns={"weight": "denominator"}),
        on=keys,
        validate="many_to_one",
    )
    codes = pd.Categorical(merged[unit], categories=sorted(pos[unit].unique())).codes
    values = (merged.wins - merged.win_rate * merged.weight) / merged.denominator
    return sparse.coo_matrix((values, (merged.row, codes)), shape=(len(stats), len(np.unique(codes)))).tocsr()


def estimate(
    pos: pd.DataFrame,
    band: tuple[int, int],
    min_costs: int,
    min_contracts: int,
    min_events: int,
    weight_name: str,
    reference: str = "pooled",
) -> tuple[pd.DataFrame, dict]:
    """Compare each cluster to the same assigned pooled population at identical costs.

    Fixed observed cost support; delta method conditions on learned groups and
    inclusion. Both event and market sandwiches preserve all repeated executions.
    Gaussian max-T calibration uses the joint event-cluster covariance.
    """
    pos = pos[pos.cost.between(*band)].copy()
    pos["family"] = pos.cluster.map({c: c.rsplit("/", 1)[0] for c in pos.cluster.unique()})
    by_family = reference == "family"
    cs, ps = curve(pos, True), curve(pos, family=by_family)
    eligible = cs[(cs.contracts >= min_contracts) & (cs.events >= min_events)]
    support = eligible.groupby(["cluster", "cost"]).direction.nunique()
    support = support[support.eq(2)].reset_index().groupby("cluster").cost.apply(list)
    support = support[support.map(len) >= min_costs]
    if support.empty:
        return pd.DataFrame(), {"clusters": 0}
    rows = []
    cweights = []
    pweights = []
    c_index = {(r.cluster, r.cost, r.direction): i for i, r in cs.iterrows()}
    p_index = {(r.family, r.cost, r.direction) if by_family else (r.cost, r.direction): i for i, r in ps.iterrows()}
    member_groups = dict(tuple(pos.groupby("cluster")))
    for cluster, costs in support.items():
        cw = np.zeros(len(cs))
        pw = np.zeros(len(ps))
        for cost in costs:
            for direction, sign in [("YES", 1), ("NO", -1)]:
                cw[c_index[cluster, cost, direction]] = 100 * sign / len(costs)
                key = (cluster.rsplit("/", 1)[0], cost, direction) if by_family else (cost, direction)
                pw[p_index[key]] = 100 * sign / len(costs)
        wedge = float(cw @ cs.win_rate)
        pooled = float(pw @ ps.win_rate)
        members = member_groups[cluster]
        members = members[members.cost.isin(costs)]
        rows.append(
            {
                "cluster": cluster,
                "weighting": weight_name,
                "reference": reference,
                "band": f"{band[0]}-{band[1]}",
                "costs": ",".join(map(str, costs)),
                "n_costs": len(costs),
                "contracts": members.ticker.nunique(),
                "events": members.event_id.nunique(),
                "wedge_pp": wedge,
                "pooled_matched_pp": pooled,
                "difference_pp": wedge - pooled,
            }
        )
        cweights.append(cw)
        pweights.append(pw)
    del member_groups
    result = pd.DataFrame(rows)
    cw = sparse.csr_matrix(np.vstack(cweights))
    pw = sparse.csr_matrix(np.vstack(pweights))
    joint = None
    for unit in ["event_id", "ticker"]:
        ci = sparse_residuals(pos, cs, unit, True)
        pi = sparse_residuals(pos, ps, unit, False, family=by_family)
        # All categories use the same sorted dependence-unit universe.
        group_if = cw @ ci
        # Expand (G - W P)(G - W P)' without constructing a dense
        # cluster-by-all-markets residual matrix. Shared pooled influences
        # otherwise turn sparse multiplication into a large dense expansion.
        group_cov = group_if @ group_if.T
        cross_cov = (group_if @ pi.T) @ pw.T
        contrast_cov = group_cov - cross_cov - cross_cov.T + (pw @ (pi @ pi.T)) @ pw.T
        n = ci.shape[1]
        factor = n / (n - 1) if n > 1 else np.nan
        se = np.sqrt(np.maximum(group_cov.diagonal(), 0) * factor)
        diffse = np.sqrt(np.maximum(contrast_cov.diagonal(), 0) * factor)
        label = "event" if unit == "event_id" else "market"
        result[f"se_{label}_pp"] = se
        result[f"ci_low_{label}_pp"] = result.wedge_pp - 1.96 * se
        result[f"ci_high_{label}_pp"] = result.wedge_pp + 1.96 * se
        result[f"difference_se_{label}_pp"] = diffse
        z = np.divide(result.difference_pp, diffse, out=np.zeros(len(result)), where=diffse > 0)
        result[f"p_{label}"] = 2 * norm.sf(np.abs(z))
        result[f"q_{label}"] = fdr(result[f"p_{label}"].to_numpy())
        if unit == "event_id":
            joint = contrast_cov.toarray() * factor
    sd = np.sqrt(np.maximum(np.diag(joint), 0))
    valid = sd > 1e-12
    if not valid.any():
        result["p_event_maxT"] = 1.0
        result["simultaneous_difference_low_pp"] = result.difference_pp
        result["simultaneous_difference_high_pp"] = result.difference_pp
        return result, {
            "clusters": len(result),
            "global_maxT_p": 1.0,
            "maxT_05": 0,
            "event_fdr_05": 0,
            "market_fdr_05": 0,
            "full_support_clusters": int(result.n_costs.eq(band[1] - band[0] + 1).sum()),
        }
    corr = joint[np.ix_(valid, valid)] / np.outer(sd[valid], sd[valid])
    eig, vec = np.linalg.eigh(corr)
    draws = np.random.default_rng(340).standard_normal((10000, len(eig))) @ (vec * np.sqrt(np.maximum(eig, 0))).T
    maxima = np.max(np.abs(draws), axis=1)
    observed = np.abs(result.difference_pp.to_numpy()[valid] / sd[valid])
    adjusted = np.ones(len(result))
    adjusted[valid] = [(1 + np.sum(maxima >= z)) / 10001 for z in observed]
    result["p_event_maxT"] = adjusted
    critical = float(np.quantile(maxima, 0.95))
    result["simultaneous_difference_low_pp"] = result.difference_pp - critical * result.difference_se_event_pp
    result["simultaneous_difference_high_pp"] = result.difference_pp + critical * result.difference_se_event_pp
    summary = {
        "clusters": len(result),
        "global_maxT_p": float((1 + np.sum(maxima >= max(observed))) / 10001),
        "maxT_critical": critical,
        "event_fdr_05": int(result.q_event.lt(0.05).sum()),
        "market_fdr_05": int(result.q_market.lt(0.05).sum()),
        "maxT_05": int(result.p_event_maxT.lt(0.05).sum()),
        "full_support_clusters": int(result.n_costs.eq(band[1] - band[0] + 1).sum()),
    }
    return result, summary
