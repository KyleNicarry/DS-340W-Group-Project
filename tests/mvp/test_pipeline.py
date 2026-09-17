"""Tests of estimand algebra and leakage boundaries, not of model accuracy.

Synthetic fixtures are used only for validation; all delivered analysis outputs
are produced from the repository's real Kalshi Parquet files.
"""

from argparse import Namespace

import numpy as np
import pandas as pd

from mvp.pipeline import bootstrap_wedge, exact_estimate, linguistic_features, prepare, split_series


def test_exact_cost_matching_removes_composition_confounding():
    # Both directions are perfectly calibrated at each cost, but radically
    # different execution mixes would confound a pooled win-rate comparison.
    g = pd.DataFrame(
        [
            (10, "Y", 1000, 100),
            (10, "N", 10, 1),
            (20, "Y", 10, 2),
            (20, "N", 1000, 200),
        ],
        columns=["cost", "direction", "n", "wins"],
    )
    np.testing.assert_allclose(exact_estimate(g, [10, 20]), [0, 0, 0], atol=1e-12)


def test_wedge_sign_and_complementary_price_identity():
    # q_Y(10)=.05, q_Y(90)=.80; q_N(10)=.20 -> Delta=-.15.
    g = pd.DataFrame(
        [(10, "Y", 100, 5), (10, "N", 100, 20), (90, "Y", 100, 80), (90, "N", 100, 95)],
        columns=["cost", "direction", "n", "wins"],
    )
    np.testing.assert_allclose(exact_estimate(g, [10]), [-5, 10, -15])
    np.testing.assert_allclose(exact_estimate(g, [90])[2], -15)


def test_missing_direction_is_not_zero_filled():
    g = pd.DataFrame([(10, "Y", 100, 5)], columns=["cost", "direction", "n", "wins"])
    assert np.isnan(exact_estimate(g, [10])).all()


def test_event_bootstrap_does_not_treat_more_executions_as_more_outcomes():
    # Multiplying trade weights uniformly must not make event-level uncertainty
    # artificially shrink. Resampling individual executions would violate this.
    rows = []
    for event, win in enumerate([0, 0, 0, 1, 1, 1, 1, 1]):
        rows += [(str(event), 10, "Y", 1, win), (str(event), 10, "N", 1, 1 - win)]
    g = pd.DataFrame(rows, columns=["event_ticker", "cost", "direction", "n", "wins"])
    draws = bootstrap_wedge(g, [10], 200)
    g[["n", "wins"]] *= 1000
    np.testing.assert_array_equal(draws, bootstrap_wedge(g, [10], 200))
    assert draws.std() > 10


def test_old_new_series_prefixes_share_split_and_literal_features():
    assert split_series("CPI") == split_series("KXCPI")
    f = linguistic_features("Will Team A win over 10 games?")
    assert f["threshold"] == 1 and f["positive_words"] == 1
    assert "sentiment" not in f and "proper_nouns" not in f


def test_prepare_enforces_landmark_and_deduplicates(tmp_path):
    # A full extraction fixture verifies the actual SQL path. Extreme later trades
    # and final metadata volume must not alter first-day predictors or cells.
    data = tmp_path / "data"
    (data / "markets").mkdir(parents=True)
    (data / "trades").mkdir()
    out = tmp_path / "results"
    out.mkdir()
    opened = pd.Timestamp("2024-01-01", tz="UTC")
    markets, trades = [], []
    for i in range(45):
        markets.append(
            {
                "ticker": f"SERIES-{i}-A",
                "event_ticker": f"SERIES-{i}",
                "title": "Will value be over ten?",
                "yes_sub_title": "over ten",
                "market_type": "binary",
                "status": "finalized",
                "result": "yes",
                "open_time": opened,
                "close_time": opened + pd.Timedelta(days=3),
                "volume": 999999999,
            }
        )
        for suffix, hour, price in [("early", 1, 25), ("late", 25, 99), ("before", -1, 99)]:
            trades.append(
                {
                    "trade_id": f"{i}-{suffix}",
                    "ticker": f"SERIES-{i}-A",
                    "count": 10,
                    "yes_price": price,
                    "no_price": 100 - price,
                    "taker_side": "yes",
                    "created_time": opened + pd.Timedelta(hours=hour),
                    "_fetched_at": opened + pd.Timedelta(days=4),
                }
            )
    trades.append(trades[0].copy())
    pd.DataFrame(markets).to_parquet(data / "markets" / "part.parquet")
    pd.DataFrame(trades).to_parquet(data / "trades" / "part.parquet")
    args = Namespace(data=data, window_hours=24, max_contracts=100, min_early_trades=1, clusters=2)
    contracts, cells = prepare(args, out)
    assert len(contracts) == 45 and contracts.n_trades.eq(1).all()
    assert contracts.mean_price.eq(0.25).all() and contracts.units.eq(10).all()
    assert cells.price.eq(25).all() and cells.n.eq(1).all()
    assert "volume" not in contracts


def test_group_labels_never_pool_train_and_test_outcomes(tmp_path):
    from mvp.pipeline import LANGUAGE, MARKET, build_labels

    records, counts = [], []
    for split in ["train", "test"]:
        for i in range(24):
            ticker = f"{split}-{i}"
            # Two contracts per event with joint outcomes model event dependence.
            records.append(
                {
                    "ticker": ticker,
                    "event_ticker": f"{split}-{i // 2}",
                    "outcome": (i // 2) % 2 if split == "train" else 0,
                    "split": split,
                    "cluster_text": 0,
                    **dict.fromkeys(MARKET + LANGUAGE, 1.0),
                }
            )
            for price in [11, 12, 13, 87, 88, 89]:
                counts.append({"ticker": ticker, "price": price, "n": 100})
    contracts = pd.DataFrame(records)
    cells = pd.DataFrame(counts)
    pd.DataFrame([{"approach": "text", "split": "train", "cluster": 0, "mean_text_cosine": 1.0}]).to_csv(
        tmp_path / "cluster_profiles.csv", index=False
    )
    args = Namespace(
        min_cell_contracts=5,
        min_cell_events=3,
        min_group_contracts=20,
        min_group_events=10,
        min_coherence=0.65,
        bootstrap=100,
    )
    labels = build_labels(contracts, cells, "text", args, tmp_path).set_index("split")
    assert labels.status.eq("eligible").all()
    np.testing.assert_allclose(labels.loc["train", "wedge_pp"], 0, atol=1e-12)
    np.testing.assert_allclose(labels.loc["test", "wedge_pp"], -100, atol=1e-12)
    # Changing held-out outcomes must not change any training label or interval.
    contracts.loc[contracts.split.eq("test"), "outcome"] = 1
    changed = build_labels(contracts, cells, "text", args, tmp_path).set_index("split")
    pd.testing.assert_series_equal(labels.loc["train"], changed.loc["train"])
    np.testing.assert_allclose(changed.loc["test", "wedge_pp"], 100, atol=1e-12)
    # The coherence gate keeps population counts but never emits an unstable label.
    profiles = pd.read_csv(tmp_path / "cluster_profiles.csv")
    profiles["mean_text_cosine"] = 0.1
    profiles.to_csv(tmp_path / "cluster_profiles.csv", index=False)
    rejected = build_labels(contracts, cells, "text", args, tmp_path)
    assert rejected.status.eq("insufficient_coherence").all()
    assert rejected.wedge_pp.isna().all()
