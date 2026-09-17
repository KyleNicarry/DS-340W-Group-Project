"""Reproducible, retrospective MVP of GROUP-level equal-cost YES/NO wedges.

Run from the repository root: python -m mvp.pipeline
This is deliberately separate from the original paper's replication code. Read
mvp/README.md before interpreting results: snapshot text is not versioned, the
holdout is by series (not time), and a wedge is not a realizable trading return.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/optimism-tax-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/optimism-tax-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

SEED = 340
BANDS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 49)]
MARKET = ["mean_price", "price_sd", "log_trades", "log_units"]
LANGUAGE = [
    "words",
    "characters",
    "mean_word_length",
    "question_mark",
    "threshold",
    "negation",
    "positive_words",
    "negative_words",
    "political_words",
    "capitalized_tokens",
    "number_tokens",
    "punctuation",
]
# These small transparent lexicons are indicator counts, NOT validated sentiment,
# emotion, partisan affiliation, named-entity recognition or readability scores.
LEXICONS = {
    "negation": {"no", "not", "never", "without", "fail", "fails"},
    "positive_words": {"win", "wins", "gain", "gains", "success", "rise", "increase"},
    "negative_words": {"lose", "loses", "loss", "fall", "decrease", "fail", "fails"},
    "political_words": {"president", "election", "democrat", "republican", "senate", "congress"},
}


def sql_path(path):
    """Quote local paths as SQL literals; never interpolate unescaped CLI paths."""
    return "'" + str(path).replace("'", "''") + "'"


def split_series(series):
    """Stable, outcome-blind 75/25 split, grouping a whole recurring ticker series.

    Strip the optional KX prefix so older/newer identifiers for one series cannot
    trivially cross the split. Series is coarser than event_ticker, but is still
    an imperfect proxy for economic dependence across different products.
    """
    key = re.sub(r"^KX", "", str(series))
    return "test" if int(hashlib.sha256((str(SEED) + key).encode()).hexdigest()[:8], 16) % 4 == 0 else "train"


def linguistic_features(title):
    """Only explicit surface features; do not relabel capital letters as people."""
    raw = re.findall(r"\b[^\W\d_]+\b", str(title), flags=re.UNICODE)
    words = [w.lower() for w in raw]
    out = {
        "words": len(words),
        "characters": len(str(title)),
        "mean_word_length": np.mean([len(w) for w in words]) if words else 0,
        "question_mark": int("?" in str(title)),
        "threshold": int(
            bool(
                re.search(r"\b(over|under|above|below|between|at least|more than|less than)\b|[<>]", str(title).lower())
            )
        ),
        "capitalized_tokens": sum(w[0].isupper() for w in raw[1:]),
        "number_tokens": len(re.findall(r"\d+(?:\.\d+)?", str(title))),
        "punctuation": len(re.findall(r"[,;:()]", str(title))),
    }
    out.update({name: sum(w in lexicon for w in words) for name, lexicon in LEXICONS.items()})
    return out


def prepare(args, out):
    """Scan Parquet with DuckDB, retaining bounded metadata and early executions.

    Resolved/non-MVE selection is a retrospective research population restriction.
    MVE combination products overlap underlying legs and need a leg-level event
    graph, unavailable here. Do not silently treat their identifiers as independent.
    The sample is a deterministic ticker hash, not the first files or busiest markets.
    No full-lifetime volume or final quote enters screening or predictors.
    """
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory={sql_path(out / 'duckdb_tmp')}")
    market_paths = str(args.data / "markets" / "*.parquet")
    trade_paths = str(args.data / "trades" / "*.parquet")
    con.execute(f"CREATE VIEW raw_markets AS SELECT * FROM read_parquet({sql_path(market_paths)})")
    con.execute(f"CREATE VIEW raw_trades AS SELECT * FROM read_parquet({sql_path(trade_paths)})")
    audit = {}
    audit["market_rows"] = con.sql("SELECT count(*) FROM raw_markets").fetchone()[0]
    # Fail on ambiguous IDs rather than potentially multiply executions in a join.
    if con.sql("SELECT count(*) - count(DISTINCT ticker) FROM raw_markets").fetchone()[0]:
        raise ValueError("Duplicate market tickers: explicit snapshot reconciliation is required")
    con.sql("DESCRIBE raw_markets").df().to_csv(out / "market_schema.csv", index=False)
    con.sql("DESCRIBE raw_trades").df().to_csv(out / "trade_schema.csv", index=False)
    audit["trade_rows"] = con.sql("SELECT count(*) FROM raw_trades").fetchone()[0]
    predicates = [
        ("finalized_binary_outcome", "status='finalized' AND result IN ('yes','no') AND market_type='binary'"),
        ("non_combination", "NOT starts_with(ticker, 'KXMVE') AND NOT starts_with(ticker, 'MVE')"),
        (
            "usable_metadata",
            "title IS NOT NULL AND length(trim(title))>0 AND event_ticker IS NOT NULL AND length(event_ticker)>0 AND open_time IS NOT NULL AND close_time IS NOT NULL",
        ),
        ("open_past_landmark", f"close_time > open_time + INTERVAL '{args.window_hours} hours'"),
    ]
    clauses = []
    for label, clause in predicates:
        clauses.append(clause)
        audit[label] = con.sql("SELECT count(*) FROM raw_markets WHERE " + " AND ".join(clauses)).fetchone()[0]
    con.execute(f"""CREATE TEMP TABLE selected AS
        SELECT ticker,event_ticker,split_part(event_ticker,'-',1) AS series,title,yes_sub_title,
               open_time,close_time,open_time + INTERVAL '{args.window_hours} hours' AS prediction_time,
               CAST(result='yes' AS INTEGER) AS outcome
        FROM raw_markets WHERE {" AND ".join(clauses)}
        ORDER BY md5(ticker || '{SEED}'),ticker LIMIT {args.max_contracts}""")
    audit["sampled_metadata"] = con.sql("SELECT count(*) FROM selected").fetchone()[0]
    # Collect only execution-window records; late/final activity cannot enter even
    # if metadata have final volume/quotes. Inclusive landmark is explicitly fixed.
    con.execute("""CREATE TEMP TABLE window_raw AS
        SELECT t.* FROM raw_trades t JOIN selected m USING(ticker)
        WHERE t.created_time >= m.open_time AND t.created_time <= m.prediction_time""")
    audit["window_raw_trades"] = con.sql("SELECT count(*) FROM window_raw").fetchone()[0]
    invalid = "trade_id IS NULL OR yes_price NOT BETWEEN 1 AND 99 OR no_price NOT BETWEEN 1 AND 99 OR yes_price+no_price<>100 OR count<=0 OR count IS NULL OR yes_price IS NULL OR no_price IS NULL OR taker_side IS NULL OR taker_side NOT IN ('yes','no')"
    audit["invalid_window_trades"] = con.sql(f"SELECT count(*) FROM window_raw WHERE {invalid}").fetchone()[0]
    # Identical re-fetches are deduplicated, conflicting records stop the pipeline.
    conflict = con.sql("""SELECT count(*) FROM (SELECT trade_id FROM window_raw
        GROUP BY trade_id HAVING count(DISTINCT (ticker,yes_price,no_price,count,taker_side,created_time))>1)""").fetchone()[
        0
    ]
    if conflict:
        raise ValueError(f"{conflict} conflicting trade IDs in sampled window")
    con.execute(f"""CREATE TEMP TABLE early AS SELECT * EXCLUDE(rn) FROM
        (SELECT *,row_number() OVER(PARTITION BY trade_id ORDER BY _fetched_at) rn
         FROM window_raw WHERE NOT ({invalid})) WHERE rn=1""")
    audit["deduplicated_valid_window_trades"] = con.sql("SELECT count(*) FROM early").fetchone()[0]
    contracts = con.sql(f"""SELECT m.*,s.* EXCLUDE(ticker) FROM selected m JOIN
        (SELECT ticker,count(*) AS n_trades,sum(count) AS units,
         avg(yes_price)/100 AS mean_price,stddev_pop(yes_price)/100 AS price_sd,
         min(created_time) AS first_trade,max(created_time) AS last_feature_trade
         FROM early GROUP BY ticker HAVING count(*) >= {args.min_early_trades}) s USING(ticker)
         ORDER BY ticker""").df()
    audit["contracts_with_min_early_trades"] = len(contracts)
    # Sufficient statistics preserve execution weighting without loading 72M rows
    # into pandas. Volume is a feature only; target weights count executions, not units.
    con.register("included", contracts[["ticker"]])
    cells = con.sql("""SELECT ticker,yes_price AS price,count(*) AS n
        FROM early JOIN included USING(ticker) GROUP BY ticker,yes_price ORDER BY ticker,price""").df()
    if len(contracts) < args.clusters * 20:
        raise ValueError("Too few eligible contracts for requested cluster count")
    # A generic title may omit its strike or named alternative. Preserve the
    # observed affirmative subtitle rather than inventing it from ticker codes.
    contracts["similarity_text"] = contracts.title + " " + contracts.yes_sub_title.fillna("")
    contracts["split"] = contracts.series.map(split_series)
    # Full event identifiers must remain split-disjoint, regardless of series parsing.
    assert contracts.groupby("event_ticker").split.nunique().max() == 1
    assert (contracts.last_feature_trade <= contracts.prediction_time).all()
    contracts["log_trades"] = np.log1p(contracts.n_trades)
    contracts["log_units"] = np.log1p(contracts.units)
    contracts = pd.concat([contracts, pd.DataFrame(contracts.title.map(linguistic_features).tolist())], axis=1)
    audit["split_contracts"] = contracts.split.value_counts().to_dict()
    audit["split_series"] = contracts.groupby("split").series.nunique().to_dict()
    audit["window_hours"] = args.window_hours
    (out / "attrition.json").write_text(json.dumps(audit, indent=2))
    con.close()
    print(json.dumps(audit, indent=2), flush=True)
    return contracts, cells


def representations(contracts, args, out):
    """Fit all learned transforms on training series only, then assign held-out titles.

    Offline TF-IDF + latent semantic analysis is a clearly labelled alternative to
    pretrained sentence embeddings. It captures lexical similarity, not guaranteed
    economic substitutability. Its small supplementary feature block enables a
    reproducible ablation without downloading unpinned neural weights.
    """
    train = contracts.split.eq("train").to_numpy()
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.98, max_features=12000, sublinear_tf=True
    )
    tf_train = vectorizer.fit_transform(contracts.loc[train, "similarity_text"])
    tf_all = vectorizer.transform(contracts.similarity_text)
    svd = TruncatedSVD(n_components=min(32, tf_train.shape[1] - 1), random_state=SEED)
    svd.fit(tf_train)
    text = normalize(svd.transform(tf_all))
    for i in range(text.shape[1]):
        contracts[f"lsa_{i:02d}"] = text[:, i]
    # A normalized text block dominates distance. Linguistic characteristics add
    # only 0.15 total scale; the single clipped, standardized price adds at most
    # 0.30 in either direction. Raw cents never enter Euclidean distance.
    ling_scaler = StandardScaler().fit(contracts.loc[train, LANGUAGE])
    ling = normalize(np.clip(ling_scaler.transform(contracts[LANGUAGE]), -3, 3)) * 0.15
    base = np.column_stack([text, ling])
    price_scaler = StandardScaler().fit(contracts.loc[train, ["mean_price"]])
    price = np.clip(price_scaler.transform(contracts[["mean_price"]]), -2, 2) * 0.15
    combined = np.column_stack([base, price])
    summary, diagnostics = [], []
    for name, matrix in [("text", base), ("text_price", combined)]:
        # Euclidean k-means on unit text vectors approximates cosine grouping;
        # appended bounded blocks make the mixed-representation distance explicit.
        km = KMeans(n_clusters=args.clusters, random_state=SEED, n_init=10)
        km.fit(matrix[train])
        labels = km.predict(matrix)
        contracts[f"cluster_{name}"] = labels
        rng = np.random.default_rng(SEED)
        ids = rng.choice(np.flatnonzero(train), min(2000, train.sum()), replace=False)
        diagnostics.append(
            {
                "approach": name,
                "silhouette": float(silhouette_score(matrix[ids], labels[ids])),
                "price_bin_nmi": float(
                    normalized_mutual_info_score((contracts.loc[train, "mean_price"] * 10).astype(int), labels[train])
                ),
                "lsa_explained_variance": float(svd.explained_variance_ratio_.sum()),
                "zero_text_vectors": int((np.linalg.norm(text, axis=1) == 0).sum()),
            }
        )
        for split in ["train", "test"]:
            for cluster in range(args.clusters):
                ix = np.flatnonzero((labels == cluster) & contracts.split.eq(split).to_numpy())
                if not len(ix):
                    continue
                g = contracts.iloc[ix]
                # Titles nearest the trained centroid support human semantic review.
                closest = ix[np.argsort(np.linalg.norm(matrix[ix] - km.cluster_centers_[cluster], axis=1))[:5]]
                dominant = g.series.value_counts(normalize=True)
                centroid = normalize(text[ix].mean(axis=0).reshape(1, -1))[0]
                summary.append(
                    {
                        "approach": name,
                        "split": split,
                        "cluster": cluster,
                        "contracts": len(g),
                        "events": g.event_ticker.nunique(),
                        "series": g.series.nunique(),
                        "dominant_series": dominant.index[0],
                        "dominant_series_share": dominant.iloc[0],
                        "category": "unavailable",
                        "mean_price": g.mean_price.mean(),
                        "price_sd": g.mean_price.std(),
                        "price_p10": g.mean_price.quantile(0.1),
                        "price_p90": g.mean_price.quantile(0.9),
                        "mean_text_cosine": float((text[ix] @ centroid).mean()),
                        "representative_titles": " | ".join(contracts.iloc[closest].similarity_text),
                    }
                )
    ari = adjusted_rand_score(contracts.cluster_text, contracts.cluster_text_price)
    for d in diagnostics:
        d["price_removal_adjusted_rand"] = float(ari)
    pd.DataFrame(summary).to_csv(out / "cluster_profiles.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "clustering_diagnostics.csv", index=False)
    return contracts


def exact_estimate(group, costs):
    """Equal-weight exact-cost standardization of execution-weighted win rates.

    Each original execution contributes one YES and one NO position at complementary
    costs. For retained c, delta_Y=q_Y-c/100, delta_N=q_N-c/100. Identical cost
    weights for both directions remove within-band price-composition confounding.
    The result is a standardized GROUP estimate, never a contract-level label.
    """
    sums = group.groupby(["cost", "direction"])[["n", "wins"]].sum()
    idx = pd.MultiIndex.from_product([costs, ["Y", "N"]], names=["cost", "direction"])
    sums = sums.reindex(idx)
    if sums["n"].isna().any() or (sums["n"] <= 0).any():
        return np.array([np.nan, np.nan, np.nan])
    q = (sums.wins / sums.n).unstack()
    dy = float((q.Y - np.asarray(costs) / 100).mean())
    dn = float((q.N - np.asarray(costs) / 100).mean())
    return np.array([dy, dn, dy - dn]) * 100  # probability percentage points


def bootstrap_wedge(group, costs, repetitions, seed=SEED):
    """Resample whole events jointly across directions and prices, not executions.

    Contract/price sufficient statistics remain intact. Exact-cost support is fixed
    before resampling; replicates missing a direction are invalid, never zero-filled.
    Intervals are conditional on fitted clusters/support and omit cluster-learning
    uncertainty and dependence across different event identifiers.
    """
    keys = pd.MultiIndex.from_product([costs, ["Y", "N"]])
    agg = group.groupby(["event_ticker", "cost", "direction"])[["n", "wins"]].sum()
    n = agg.n.unstack(["cost", "direction"]).reindex(columns=keys).fillna(0).to_numpy()
    w = agg.wins.unstack(["cost", "direction"]).reindex(columns=keys).fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    # Multinomial event multiplicities are exactly an ordinary event bootstrap.
    for _ in range(repetitions):
        multiplicity = rng.multinomial(len(n), np.repeat(1 / len(n), len(n)))
        den = multiplicity @ n
        if (den <= 0).any():
            continue
        rates = (multiplicity @ w / den).reshape(len(costs), 2)
        draws.append(float((rates[:, 0] - rates[:, 1]).mean() * 100))
    return np.asarray(draws)


def build_labels(contracts, cells, approach, args, out):
    """Create separate train/test group populations using one frozen cluster map."""
    base = cells.merge(
        contracts[["ticker", "event_ticker", "outcome", "split", f"cluster_{approach}"]],
        on="ticker",
        validate="many_to_one",
    )
    base = base.rename(columns={f"cluster_{approach}": "cluster"})
    yes = base.assign(cost=base.price, direction="Y", wins=base.n * base.outcome)
    no = base.assign(cost=100 - base.price, direction="N", wins=base.n * (1 - base.outcome))
    pos = pd.concat([yes, no], ignore_index=True)
    # For all-side execution weighting Delta(c)=Delta(100-c); excluding 50-99
    # removes mechanically duplicated upper-half labels. At c=50 the contrast is
    # twice YES calibration rather than a distinct framing comparison; omit it too.
    pos = pos[pos.cost.between(1, 49)].copy()
    pos["band"] = ((pos.cost - 1) // 10).astype(int)
    rows, support_rows = [], []
    # Reject diffuse clusters using training-input coherence only. This numeric
    # floor is a safeguard, not a substitute for economic/manual validation.
    profiles = pd.read_csv(out / "cluster_profiles.csv")
    coherence = (
        profiles[(profiles.approach == approach) & (profiles.split == "train")].set_index("cluster").mean_text_cosine
    )
    predictor_cols = MARKET + LANGUAGE + [c for c in contracts if c.startswith("lsa_")]
    contract_index = contracts.set_index("ticker")
    for (split, cluster, band), g in pos.groupby(["split", "cluster", "band"], sort=True):
        support = g.groupby(["cost", "direction"]).agg(
            contracts=("ticker", "nunique"), events=("event_ticker", "nunique"), trades=("n", "sum")
        )
        valid = support[(support.contracts >= args.min_cell_contracts) & (support.events >= args.min_cell_events)]
        counts = valid.reset_index().groupby("cost").direction.nunique()
        costs = sorted(counts[counts == 2].index.tolist())
        h = g[g.cost.isin(costs)]
        row = {
            "approach": approach,
            "split": split,
            "cluster": int(cluster),
            "band": int(band),
            "cost_low": BANDS[band][0],
            "cost_high": BANDS[band][1],
            "costs": ",".join(map(str, costs)),
            "n_costs": len(costs),
            "contracts": h.ticker.nunique(),
            "events": h.event_ticker.nunique(),
            "trades_total": int(h.n.sum()),
        }
        for direction in ["Y", "N"]:
            d = h[h.direction == direction]
            row.update(
                {
                    f"contracts_{direction}": d.ticker.nunique(),
                    f"events_{direction}": d.event_ticker.nunique(),
                    f"trades_{direction}": int(d.n.sum()),
                }
            )
        # Thresholds refer to independent resolutions as well as execution counts.
        enough = len(costs) >= 3 and all(
            row[f"contracts_{d}"] >= args.min_group_contracts
            and row[f"events_{d}"] >= args.min_group_events
            and row[f"trades_{d}"] >= 50
            for d in ["Y", "N"]
        )
        row["training_text_coherence"] = float(coherence.loc[cluster])
        coherent = row["training_text_coherence"] >= args.min_coherence
        row["status"] = "eligible" if enough else "insufficient_support"
        if not coherent:
            row["status"] = "insufficient_coherence"
        row.update(
            delta_yes_pp=np.nan,
            delta_no_pp=np.nan,
            wedge_pp=np.nan,
            ci_low_pp=np.nan,
            ci_high_pp=np.nan,
            se_pp=np.nan,
            bootstrap_valid=0,
        )
        for (cost, direction), s in support.iterrows():
            support_rows.append(
                dict(
                    split=split,
                    cluster=cluster,
                    band=band,
                    cost=cost,
                    direction=direction,
                    retained=cost in costs,
                    **s.to_dict(),
                )
            )
        if len(h):
            # Equal-contract predictors describe the group, while labels deliberately
            # follow the paper's equal-execution estimand. No outcomes/SE/CI enter X.
            f = contract_index.loc[h.ticker.unique(), predictor_cols].mean().to_dict()
            row.update(f)
            row["cost_mean"] = np.mean(costs) / 100
        if enough and coherent:
            dy, dn, wedge = exact_estimate(h, costs)
            draws = bootstrap_wedge(h, costs, args.bootstrap, SEED + int(cluster) * 10 + int(band))
            row.update(delta_yes_pp=dy, delta_no_pp=dn, wedge_pp=wedge, bootstrap_valid=len(draws))
            if len(draws) >= 0.9 * args.bootstrap:
                row.update(
                    ci_low_pp=np.quantile(draws, 0.025), ci_high_pp=np.quantile(draws, 0.975), se_pp=draws.std(ddof=1)
                )
            else:
                row["status"] = "insufficient_bootstrap_support"
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / f"wedges_{approach}.csv", index=False)
    pd.DataFrame(support_rows).to_csv(out / f"exact_cost_support_{approach}.csv", index=False)
    return result


def block_metric_interval(frame, predictions, baseline, repetitions):
    """Cluster-block CI for paired MAE improvement on independent test populations.

    Keeps cost-band rows of the same semantic cluster together. This is conditional
    on the training fit and does not capture correlated events across clusters.
    """
    difference = np.abs(frame.wedge_pp.to_numpy() - baseline) - np.abs(frame.wedge_pp.to_numpy() - predictions)
    blocks = frame.cluster.to_numpy()
    unique = np.unique(blocks)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in sampled])
        vals.append(difference[idx].mean())
    return np.quantile(vals, [0.025, 0.975])


def model_labels(labels, approach, args, out):
    """Fixed models; no test-set hyperparameter tuning or winner selection.

    Sharing a centroid ID across split populations shares only a similarity rule,
    never contracts, event series, targets, or residuals. Cluster ID is not an input.
    Metrics measure transfer to unseen series under snapshot assumptions, NOT a
    verified forecast of future settlements. There are few effective group labels.
    """
    usable = labels[labels.status == "eligible"].copy()
    train = usable[usable.split == "train"]
    test = usable[usable.split == "test"]
    if len(train) < 15 or len(test) < 10 or test.cluster.nunique() < 3:
        raise ValueError(
            f"{approach}: insufficient independent grouped labels ({len(train)} train, {len(test)} test); do not relax thresholds automatically"
        )
    blocks = {
        "market": MARKET + ["cost_mean"],
        "interpretable": MARKET + LANGUAGE + ["cost_mean"],
        "interpretable_lsa": MARKET + LANGUAGE + ["cost_mean"] + [c for c in labels if c.startswith("lsa_")],
    }
    metrics, predictions, coefficients = [], [], []
    y, yt = train.wedge_pp.to_numpy(), test.wedge_pp.to_numpy()
    naive = np.repeat(y.mean(), len(test))
    for block, features in blocks.items():
        x, xt = train[features], test[features]
        assert np.isfinite(x.to_numpy()).all() and np.isfinite(xt.to_numpy()).all()
        models = {
            "mean": DummyRegressor(),
            "linear": make_pipeline(StandardScaler(), LinearRegression()),
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "forest": RandomForestRegressor(
                n_estimators=200, max_depth=3, min_samples_leaf=5, random_state=SEED, n_jobs=1
            ),
        }
        for name, model in models.items():
            if name == "mean" and block != "market":
                continue
            # Equal row weights target the average eligible group-cost cell. Labels
            # with smaller estimated SE are NOT promoted using realized outcomes.
            model.fit(x, y)
            pred = model.predict(xt)
            lo, hi = block_metric_interval(test, pred, naive, args.bootstrap)
            metrics.append(
                {
                    "approach": approach,
                    "block": block,
                    "model": name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_clusters": test.cluster.nunique(),
                    "mae_pp": mean_absolute_error(yt, pred),
                    "rmse_pp": np.sqrt(mean_squared_error(yt, pred)),
                    "r2": r2_score(yt, pred),
                    "mae_improvement_pp": mean_absolute_error(yt, naive) - mean_absolute_error(yt, pred),
                    "improvement_ci_low_pp": lo,
                    "improvement_ci_high_pp": hi,
                }
            )
            for (_, r), p in zip(test.iterrows(), pred):
                predictions.append(
                    {
                        "approach": approach,
                        "block": block,
                        "model": name,
                        "cluster": r.cluster,
                        "band": r.band,
                        "observed_pp": r.wedge_pp,
                        "predicted_pp": p,
                        "ci_low_pp": r.ci_low_pp,
                        "ci_high_pp": r.ci_high_pp,
                    }
                )
            if name in ["linear", "ridge"]:
                # Standardization is fitted on train groups. A coefficient is the
                # response per one training-SD change, conditional on other inputs.
                for f, coef in zip(features, model[-1].coef_):
                    coefficients.append(
                        {"approach": approach, "block": block, "model": name, "feature": f, "coefficient_pp": coef}
                    )
    pd.DataFrame(metrics).to_csv(out / f"metrics_{approach}.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / f"predictions_{approach}.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / f"coefficients_{approach}.csv", index=False)
    # Pre-specify text-only grouping / interpretable ridge as primary, irrespective
    # of test performance. Report all comparisons as exploratory, not selected wins.
    p = pd.DataFrame(predictions)
    p = p[(p.model == "ridge") & (p.block == "interpretable")].copy()
    p["predicted_tax_pp"] = -p.predicted_pp  # negative Delta means YES disadvantage
    p["realized_tax_pp"] = -p.observed_pp
    # Ties stay together: percentile thresholds never use test outcomes.
    cuts = p.predicted_tax_pp.quantile([1 / 3, 2 / 3]).to_numpy()
    p["quantile"] = np.searchsorted(cuts, p.predicted_tax_pp, side="left") + 1
    ranked = p.groupby("quantile").agg(
        rows=("observed_pp", "size"),
        clusters=("cluster", "nunique"),
        predicted_tax_pp=("predicted_tax_pp", "mean"),
        realized_tax_pp=("realized_tax_pp", "mean"),
        realized_wedge_pp=("observed_pp", "mean"),
    )
    ranked.to_csv(out / f"ranking_{approach}.csv")
    # CI for high-minus-low tax uses cluster blocks and fixed predicted membership.
    rng = np.random.default_rng(SEED)
    contrasts = []
    for _ in range(args.bootstrap):
        draw = rng.choice(p.cluster.unique(), p.cluster.nunique(), replace=True)
        b = pd.concat([p[p.cluster == c] for c in draw])
        if (b["quantile"] == 1).any() and (b["quantile"] == 3).any():
            contrasts.append(
                b.loc[b["quantile"] == 3, "realized_tax_pp"].mean()
                - b.loc[b["quantile"] == 1, "realized_tax_pp"].mean()
            )
    rank_summary = {
        "high_minus_low_tax_pp": float(ranked.loc[3, "realized_tax_pp"] - ranked.loc[1, "realized_tax_pp"])
        if 3 in ranked.index
        else None,
        "contrast_ci_pp": np.quantile(contrasts, [0.025, 0.975]).tolist() if contrasts else None,
        "bootstrap_valid": len(contrasts),
        "baseline_tax_pp": float(p.realized_tax_pp.mean()),
        "high_minus_baseline_tax_pp": float(ranked.loc[3, "realized_tax_pp"] - p.realized_tax_pp.mean())
        if 3 in ranked.index
        else None,
    }
    (out / f"ranking_summary_{approach}.json").write_text(json.dumps(rank_summary, indent=2))
    return pd.DataFrame(metrics), pd.DataFrame(predictions), pd.DataFrame(coefficients), ranked


def figures(labels, profiles, metrics, predictions, coefficients, rankings, out):
    """Static exportable figures; all wedge scales are probability percentage points."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in profiles[profiles.split == "train"].groupby("approach"):
        axes[0].plot(g.cluster, g.contracts, "o-", label=name)
        axes[1].plot(g.cluster, g.price_p90 - g.price_p10, "o-", label=name)
    axes[0].set(xlabel="Cluster ID (IDs differ across approaches)", ylabel="Training contracts", title="Cluster sizes")
    axes[1].set(xlabel="Cluster ID", ylabel="Within-cluster price P90 - P10", title="Price variation within clusters")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "cluster_quality.png", dpi=160)
    plt.close(fig)
    good = labels[labels.status == "eligible"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for split, offset in [("train", -0.13), ("test", 0.13)]:
        g = good[good.split == split]
        ax.errorbar(
            g.cluster + offset + (g.band - 2) * 0.025,
            g.wedge_pp,
            yerr=np.vstack([np.maximum(0, g.wedge_pp - g.ci_low_pp), np.maximum(0, g.ci_high_pp - g.wedge_pp)]),
            fmt="o",
            markersize=3,
            alpha=0.65,
            label=split,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="Text cluster (five cost bands offset slightly)",
        ylabel="YES - NO wedge (pp)",
        title="Grouped wedge estimates and event-bootstrap 95% intervals",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "wedge_variation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    m = metrics[metrics.approach == "text"]
    axes[0].barh(m.block + " / " + m.model, m.mae_pp)
    axes[0].set(xlabel="Held-out MAE (pp; lower is better)", title="Fixed model comparisons")
    p = predictions[
        (predictions.approach == "text") & (predictions.block == "interpretable") & (predictions.model == "ridge")
    ]
    axes[1].scatter(p.predicted_pp, p.observed_pp, alpha=0.7)
    limits = [min(p.predicted_pp.min(), p.observed_pp.min()), max(p.predicted_pp.max(), p.observed_pp.max())]
    axes[1].plot(limits, limits, "--", color="gray")
    axes[1].set(
        xlabel="Predicted wedge (pp)", ylabel="Realized group wedge (pp)", title="Primary model: interpretable ridge"
    )
    fig.tight_layout()
    fig.savefig(out / "prediction_performance.png", dpi=160)
    plt.close(fig)
    co = coefficients[
        (coefficients.approach == "text") & (coefficients.block == "interpretable") & (coefficients.model == "ridge")
    ].copy()
    co = co.loc[co.coefficient_pp.abs().sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(co.feature, co.coefficient_pp, color=np.where(co.coefficient_pp < 0, "#bf5545", "#287e9e"))
    ax.set(
        xlabel="Wedge change per training SD (pp)",
        title="Ridge coefficients: predictive associations, not causal effects",
    )
    fig.tight_layout()
    fig.savefig(out / "feature_coefficients.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rankings.index - 0.15, rankings.predicted_tax_pp, width=0.3, label="Predicted")
    ax.bar(rankings.index + 0.15, rankings.realized_tax_pp, width=0.3, label="Realized")
    ax.set(
        xticks=[1, 2, 3],
        xticklabels=["Low", "Middle", "High"],
        xlabel="Predicted optimism-tax tertile",
        ylabel="Tax = minus wedge (pp)",
        title="Held-out ranking diagnostic (not trading profit)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "signal_ranking.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/kalshi"))
    parser.add_argument("--output", type=Path, default=Path("mvp/results"))
    parser.add_argument("--max-contracts", type=int, default=60000)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--min-early-trades", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--min-cell-contracts", type=int, default=5)
    parser.add_argument("--min-cell-events", type=int, default=3)
    parser.add_argument("--min-group-contracts", type=int, default=20)
    parser.add_argument("--min-group-events", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--min-coherence", type=float, default=0.65)
    args = parser.parse_args()
    if not 0 <= args.min_coherence <= 1:
        parser.error("Coherence must be between zero and one")
    if (
        args.bootstrap < 100
        or min(
            args.window_hours,
            args.max_contracts,
            args.min_early_trades,
            args.clusters,
            args.min_cell_contracts,
            args.min_cell_events,
            args.min_group_contracts,
            args.min_group_events,
        )
        <= 0
    ):
        parser.error("Require positive settings and at least 100 bootstrap repetitions")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    manifest = {
        "seed": SEED,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "duckdb", "scipy", "scikit-learn", "matplotlib"]
        },
        "input_files": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for sub in ["markets", "trades"]
            for p in sorted((args.data / sub).glob("*.parquet"))
        ],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_source_sha256": hashlib.sha256(Path(__file__).with_name("report.py").read_bytes()).hexdigest(),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    with threadpool_limits(limits=2):
        contracts, cells = prepare(args, out)
        contracts = representations(contracts, args, out)
        contracts.to_parquet(out / "contract_features.parquet", index=False)
        cells.to_parquet(out / "contract_price_counts.parquet", index=False)
        all_metrics, all_preds, all_coefs = [], [], []
        for approach in ["text", "text_price"]:
            print(f"Estimating {approach} grouped labels...", flush=True)
            labels = build_labels(contracts, cells, approach, args, out)
            print(labels.groupby(["split", "status"]).size(), flush=True)
            met, pred, coef, rank = model_labels(labels, approach, args, out)
            all_metrics.append(met)
            all_preds.append(pred)
            all_coefs.append(coef)
            if approach == "text":
                primary_labels, primary_rank = labels, rank
        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics.to_csv(out / "model_comparison.csv", index=False)
        figures(
            primary_labels,
            pd.read_csv(out / "cluster_profiles.csv"),
            metrics,
            pd.concat(all_preds),
            pd.concat(all_coefs),
            primary_rank,
            out,
        )
    from mvp.report import write_findings

    write_findings(out)
    manifest["elapsed_seconds"] = time.time() - start
    manifest["completed"] = True
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False), flush=True)
    print(f"Completed in {manifest['elapsed_seconds']:.1f}s. Results: {out}", flush=True)


if __name__ == "__main__":
    main()
