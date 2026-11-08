"""Guard the forecasting boundary, weighting and equal-cost target."""

import numpy as np
import pandas as pd
import pytest

from extension.clustering.supervised.evaluation import matched_curves, paired_interval, wedge_summary
from extension.clustering.supervised.models import ResidualModel, design, text_features, validate_splits
from extension.clustering.supervised.pipeline import train_predict


def test_events_and_contracts_cannot_cross_splits():
    frame = pd.DataFrame({"ticker": ["a", "b"], "event_id": ["shared", "shared"], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="Event leakage"):
        validate_splits(frame)
    frame.event_id = ["first", "second"]
    validate_splits(frame)
    frame.ticker = "shared"
    with pytest.raises(ValueError, match="Contract leakage"):
        validate_splits(frame)


def test_features_ignore_outcomes_and_final_market_fields():
    contracts = pd.DataFrame(
        {
            "ticker": ["a"],
            "title": ["Will X win 10?"],
            "yes_sub_title": ["Not X"],
            "result": ["yes"],
            "last_price": [99],
        }
    )
    before = text_features(contracts)
    contracts["result"], contracts["last_price"] = "no", 1
    pd.testing.assert_frame_equal(before, text_features(contracts))
    assert before.negation.iloc[0] == 1
    assert before.numbers.iloc[0] == 1


def test_ridge_execution_weights_equal_expanded_executions():
    # Mean-normalized weights imply a different penalty after expansion; alpha
    # has negligible effect in this constant design, so the intercept is exact.
    x = np.zeros((2, 8))
    x[:, 0] = 0.2
    model = ResidualModel("ridge").fit(x, np.array([0, 1]), np.array([1, 3]))
    assert model.predict(x) == pytest.approx([0.75, 0.75])
    assert np.all((model.predict(x) >= 0) & (model.predict(x) <= 1))


def test_matched_costs_use_opposite_prices_not_same_contract_errors():
    rows = []
    for cost in range(1, 11):
        for event in range(5):
            for price in [cost, 100 - cost]:
                rows.append(
                    {
                        "ticker": f"{cost}-{event}-{price}",
                        "event_id": f"event-{event}",
                        "cluster": "group",
                        "split": "test",
                        "p": price,
                        "outcome": 1,
                        "n": 1,
                    }
                )
    frame = pd.DataFrame(rows)
    curves = matched_curves(frame, np.repeat(0.75, len(frame)), "model")
    summary = wedge_summary(curves).iloc[0]
    assert summary.costs == 10
    assert summary.observed_wedge_pp == pytest.approx(100)
    assert summary.predicted_wedge_pp == pytest.approx(50)
    assert summary.observed_yes_pp == pytest.approx(94.5)
    assert summary.observed_no_pp == pytest.approx(-5.5)
    sparse = curves[curves.cost < 10]
    assert wedge_summary(sparse).empty


def test_event_bootstrap_pairs_model_errors():
    frame = pd.DataFrame({"event_id": ["a", "a", "b", "b"], "n": [1, 9, 2, 8]})
    assert paired_interval(frame, np.repeat(0.1, 4), 100) == pytest.approx((0.1, 0.1))
    assert paired_interval(frame, np.zeros(4), 100) == pytest.approx((0, 0))


def test_prediction_does_not_use_test_outcomes():
    frame = pd.DataFrame(
        {
            "p": [10, 30, 70, 90],
            "words": [3] * 4,
            "characters": [12] * 4,
            "numbers": [1] * 4,
            "negation": [0] * 4,
            "question": [1] * 4,
        }
    )
    x = design(frame)
    model = ResidualModel("ridge").fit(x, np.array([0, 0, 1, 1]), np.ones(4))
    first = model.predict(x)
    frame["outcome"] = [1, 1, 0, 0]
    assert model.predict(design(frame)) == pytest.approx(first)


def test_all_scope_fits_ignore_both_holdout_outcome_sets(tmp_path):
    rows = []
    for cluster in ["family/000", "family/001"]:
        for split in ["train", "validation", "test"]:
            for index in range(4):
                rows.append(
                    {
                        "ticker": f"{cluster}-{split}-{index}",
                        "event_id": f"{split}-{index}",
                        "cluster": cluster,
                        "family": "family",
                        "split": split,
                        "p": 10 + 20 * index,
                        "outcome": index % 2,
                        "n": index + 1,
                        "words": 4,
                        "characters": 15,
                        "numbers": 1,
                        "negation": 0,
                        "question": 1,
                    }
                )
    frame = pd.DataFrame(rows)
    first, _ = train_predict(frame, tmp_path)
    held = frame.split.ne("train")
    frame.loc[held, "outcome"] = 1 - frame.loc[held, "outcome"]
    second, _ = train_predict(frame, tmp_path)
    prediction_columns = [
        f"{scope}_{kind}" for scope in ["pooled", "family", "cluster"] for kind in ["ridge", "forest", "boosting"]
    ]
    np.testing.assert_array_equal(first[prediction_columns], second[prediction_columns])


def test_price_baseline_has_no_directional_prediction():
    rows = []
    for cost in range(1, 11):
        for event in range(5):
            for price in [cost, 100 - cost]:
                rows.append(
                    {
                        "ticker": f"{price}-{event}",
                        "event_id": f"e{event}",
                        "cluster": "g",
                        "split": "test",
                        "n": 1,
                        "outcome": 1,
                        "p": price,
                    }
                )
    frame = pd.DataFrame(rows)
    result = wedge_summary(matched_curves(frame, frame.p.to_numpy() / 100, "market_price"))
    assert result.predicted_wedge_pp.iloc[0] == pytest.approx(0, abs=1e-8)
    assert result.sign_correct.isna().all()
