"""Checks for paper-based weights, cost matching and dependent-position inference."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.kalshi.clustering.mispricing.statistics import curve, estimate, fdr, positions


def fixture_cells():
    rows = []
    for cluster in ["crypto/BTC/above/000", "crypto/BTC/above/001"]:
        for event in range(24):
            for price in [10, 90]:
                outcome = int(event < (18 if cluster.endswith("000") else 6))
                rows.append(
                    {
                        "ticker": f"{cluster}-{event}-{price}",
                        "event_id": f"event-{event}",
                        "cluster": cluster,
                        "split": "train",
                        "p": price,
                        "outcome": outcome,
                        "n": 1,
                        "units": event + 1,
                    }
                )
    return pd.DataFrame(rows)


def test_cost_basis_and_complement_identity():
    cells = fixture_cells()
    pos = positions(cells, "n")
    assert len(pos) == 2 * len(cells)
    cs = curve(pos, True).pivot(index=["cluster", "cost"], columns="direction", values="win_rate")
    wedge = cs.YES - cs.NO
    assert wedge.loc[("crypto/BTC/above/000", 10)] == pytest.approx(0.5)
    assert wedge.loc[("crypto/BTC/above/000", 90)] == pytest.approx(0.5)
    assert wedge.loc[("crypto/BTC/above/001", 10)] == pytest.approx(-0.5)
    assert positions(cells, "units").weight.sum() == 2 * cells.units.sum()


def test_paired_inference_handles_shared_events():
    pos = positions(fixture_cells(), "n")
    result, summary = estimate(pos, (10, 10), 1, 5, 5, "execution")
    assert len(result) == 2
    assert result.wedge_pp.tolist() == pytest.approx([50, -50])
    assert result.pooled_matched_pp.tolist() == pytest.approx([0, 0])
    assert np.isfinite(result.difference_se_event_pp).all()
    assert (result.difference_se_event_pp > 0).all()
    assert summary["global_maxT_p"] <= 0.05
    family, _ = estimate(pos, (10, 10), 1, 5, 5, "execution", reference="family")
    assert family.difference_pp.tolist() == pytest.approx([50, -50])


def test_sparse_support_is_not_zero_label():
    result, summary = estimate(positions(fixture_cells(), "n"), (1, 9), 1, 5, 5, "execution")
    assert result.empty and summary["clusters"] == 0


def test_single_cluster_comparison_has_zero_paired_variance():
    cells = fixture_cells()
    cells["cluster"] = "single/family/above/000"
    result, summary = estimate(positions(cells, "n"), (10, 10), 1, 5, 5, "execution")
    assert result.difference_pp.iloc[0] == pytest.approx(0)
    assert result.difference_se_event_pp.iloc[0] == pytest.approx(0)
    assert summary["global_maxT_p"] == 1


def test_fdr_adjustment():
    assert fdr(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.04, 0.04])


def test_paired_sandwich_matches_hand_calculation():
    result, _ = estimate(positions(fixture_cells(), "n"), (10, 10), 1, 5, 5, "execution")
    # 24 independent events, paired cluster-minus-pool IF is +/-100/48.
    expected = np.sqrt((24 / 23) * 24 * (100 / 48) ** 2)
    assert result.difference_se_event_pp.to_numpy() == pytest.approx([expected, expected])


def test_csv_extraction_uses_observed_units_for_volume_and_rejects_invalid(tmp_path):
    from src.analysis.kalshi.clustering.mispricing.data import extract

    data = tmp_path / "data" / "master"
    data.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": t,
                "event_ticker": t,
                "title": t,
                "result": "yes",
                "status": "finalized",
                "market_type": "binary",
            }
            for t in ["A", "B"]
        ]
    ).to_csv(data / "markets.csv", index=False)
    pd.DataFrame(
        [
            {"trade_id": "1", "ticker": "A", "taker_side": "yes", "yes_price": 10, "no_price": 90, "count": 100},
            {"trade_id": "2", "ticker": "B", "taker_side": "yes", "yes_price": 10, "no_price": 90, "count": 99},
            {"trade_id": "3", "ticker": "A", "taker_side": "yes", "yes_price": 10, "no_price": 80, "count": 1000},
        ]
    ).to_csv(data / "trades.csv", index=False)
    assignment = tmp_path / "assignments.csv"
    pd.DataFrame(
        [
            {
                "ticker": "A",
                "cluster": "crypto/BTC/above/000",
                "split": "train",
                "status": "assigned",
                "family": "crypto/BTC/above",
            }
        ]
    ).to_csv(assignment, index=False)
    out = tmp_path / "out"
    out.mkdir()
    cells, audit = extract(data.parent, assignment, out)
    assert cells.ticker.tolist() == ["A"]
    assert cells.n.tolist() == [1]
    assert cells.units.tolist() == [100]
    assert audit["invalid_trades"] == 1


def test_direct_equality_distinguishes_within_and_between_families():
    from src.analysis.kalshi.clustering.mispricing.heterogeneity import centered_max_test

    means = np.array([-20.0, -20.0, 20.0, 20.0])
    covariance = np.eye(4)
    global_test = centered_max_test(means, covariance, ["ALL"] * 4)
    family_test = centered_max_test(means, covariance, ["a", "a", "b", "b"])
    assert global_test["p_value"] < 0.05
    assert family_test["p_value"] == 1
    assert family_test["tested_clusters"] == 4


def test_structural_groups_ignore_learned_membership_and_exclude_combinations():
    from src.analysis.kalshi.clustering.mispricing.structural import structural_labels

    markets = pd.DataFrame(
        [
            {
                "ticker": "A",
                "event_ticker": "KXBTCD-DATE",
                "market_type": "binary",
                "title": "Bitcoin price?",
                "yes_sub_title": "100 or above",
                "cluster": "learned-1",
                "status": "distant",
            },
            {
                "ticker": "B",
                "event_ticker": "KXBTCD-DATE2",
                "market_type": "binary",
                "title": "Bitcoin price?",
                "yes_sub_title": "200 or above",
                "cluster": "learned-99",
                "status": "assigned",
            },
            {
                "ticker": "C",
                "event_ticker": "KXBTCD-DATE3",
                "market_type": "binary",
                "title": "Bitcoin price?",
                "yes_sub_title": "200 or below",
                "cluster": "learned-1",
                "status": "unseen_family",
            },
            {
                "ticker": "KXMVE-X",
                "event_ticker": "KXMVE-X",
                "market_type": "binary",
                "title": "Bitcoin price?",
                "yes_sub_title": "200 or below",
                "cluster": "",
                "status": "",
            },
        ]
    )
    result = structural_labels(markets).set_index("ticker")
    assert result.index.tolist() == ["A", "B", "C"]
    assert result.loc["A", "structural_group"] == result.loc["B", "structural_group"] == "crypto/BTC/above"
    assert result.loc["C", "structural_group"] == "crypto/BTC/below"
    assert result.structural_group.str.count("/").eq(2).all()
