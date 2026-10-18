"""Direct equality tests on a common full 1–49-cent support cohort."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from .statistics import curve, positions, sparse_residuals


def centered_max_test(values: np.ndarray, covariance: np.ndarray, blocks: list[str], repetitions: int = 10000) -> dict:
    """Test equal cluster means within each block, without a pooled price mixture.

    A cluster-versus-trade-weighted-pool test can also reflect differences in
    direction-specific cluster composition. Centering equal-support cluster
    wedges directly tests the user's heterogeneity question instead.
    """
    blocks = np.asarray(blocks)
    eligible = np.array([np.sum(blocks == b) > 1 for b in blocks])
    if eligible.sum() < 2:
        return {"tested_clusters": int(eligible.sum()), "p_value": None}
    values = values[eligible]
    covariance = covariance[np.ix_(eligible, eligible)]
    blocks = blocks[eligible]
    h = np.eye(len(values))
    for block in np.unique(blocks):
        ix = np.flatnonzero(blocks == block)
        h[np.ix_(ix, ix)] -= 1 / len(ix)
    centered = h @ values
    cov = h @ covariance @ h.T
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    good = se > 1e-10
    if not good.any():
        return {
            "tested_clusters": len(values),
            "p_value": 1.0 if np.max(np.abs(centered)) < 1e-8 else 1 / (repetitions + 1),
        }
    corr = cov[np.ix_(good, good)] / np.outer(se[good], se[good])
    eig, vec = np.linalg.eigh(corr)
    draws = np.random.default_rng(340).standard_normal((repetitions, len(eig))) @ (vec * np.sqrt(np.maximum(eig, 0))).T
    observed = float(np.max(np.abs(centered[good] / se[good])))
    return {
        "tested_clusters": len(values),
        "p_value": float((1 + np.sum(np.abs(draws).max(axis=1) >= observed)) / (repetitions + 1)),
        "observed_max_z": observed,
    }


def run_fixed_support(out: Path) -> None:
    summary = pd.read_csv(out / "cluster_summary.csv")
    columns = ["ticker", "event_id", "cluster", "split", "p", "outcome", "n", "units"]
    cells = pd.read_csv(out / "sufficient_statistics.csv.gz", usecols=columns, keep_default_na=False)
    cells = cells[cells.cluster.ne("")]
    rows = []
    with threadpool_limits(limits=2):
        for weighting, column in [("execution", "n"), ("unit", "units")]:
            cohort = summary.query("weighting==@weighting and reference=='pooled' and band=='1-49' and n_costs==49")
            names = cohort.cluster.tolist()
            if len(names) < 2:
                continue
            pos = positions(cells[cells.cluster.isin(names)], column)
            pos = pos[pos.cost.between(1, 49)].copy()
            cs = curve(pos, True)
            index = {name: i for i, name in enumerate(names)}
            weights = sparse.coo_matrix(
                (np.where(cs.direction.eq("YES"), 100 / 49, -100 / 49), (cs.cluster.map(index), np.arange(len(cs)))),
                shape=(len(names), len(cs)),
            ).tocsr()
            values = np.asarray(weights @ cs.win_rate).ravel()
            for unit in ["event_id", "ticker"]:
                residuals = sparse_residuals(pos, cs, unit, True)
                influences = weights @ residuals
                n = residuals.shape[1]
                cov = (influences @ influences.T).toarray() * n / (n - 1)
                for question, blocks in [
                    ("all_clusters", ["ALL"] * len(names)),
                    ("within_structural_family", [name.rsplit("/", 1)[0] for name in names]),
                ]:
                    rows.append(
                        {
                            "weighting": weighting,
                            "dependence_unit": unit,
                            "question": question,
                            **centered_max_test(values, cov, blocks),
                        }
                    )
    pd.DataFrame(rows).to_csv(out / "fixed_support_equality_tests.csv", index=False)
