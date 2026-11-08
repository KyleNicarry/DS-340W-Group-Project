"""Guard historical calibration weighting, fallback and split boundaries."""

import numpy as np
import pandas as pd
import pytest

from extension.clustering.supervised.historical import HistoricalCalibration, historical_predict


def training():
    return pd.DataFrame(
        {
            "ticker": ["a", "b"],
            "event_id": ["a", "b"],
            "split": ["train"] * 2,
            "family": ["f"] * 2,
            "cluster": ["f/0"] * 2,
            "p": [30, 30],
            "outcome": [0, 1],
            "n": [9, 1],
        }
    )


def test_execution_mean_with_event_effective_support():
    frame = training()
    model = HistoricalCalibration(1).fit(frame)
    stats = model.tables["pooled"].iloc[0]
    assert stats.residual == pytest.approx(-0.2)
    assert stats.effective_events == pytest.approx(100 / 82)
    prediction = model.correction(frame, "pooled", 10, np.zeros(2))
    assert prediction == pytest.approx(np.repeat(-0.2 * (100 / 82) / (100 / 82 + 10), 2))
    # Splitting one observation into repeated rows of the same event adds no support.
    expanded = pd.concat([frame.iloc[[0]].assign(n=1)] * 9 + [frame.iloc[[1]]], ignore_index=True)
    assert HistoricalCalibration(1).fit(expanded).tables["pooled"].effective_events.iloc[0] == pytest.approx(100 / 82)


def test_missing_price_or_group_inherits_parent():
    model = HistoricalCalibration(1).fit(training())
    held = training().assign(p=[50, 30], cluster=["f/0", "unseen"])
    parent = np.array([0.03, -0.04])
    assert model.correction(held, "cluster", 10, parent) == pytest.approx(parent)
    assert model.correction(training(), "cluster", None, parent) == pytest.approx(parent)


def test_only_training_labels_can_fit():
    with pytest.raises(ValueError, match="training rows only"):
        HistoricalCalibration(1).fit(training().assign(split="validation"))


def test_test_labels_never_affect_estimates_or_selection(tmp_path):
    train = training()
    validation = train.assign(ticker=["c", "d"], event_id=["c", "d"], split="validation")
    test = train.assign(ticker=["e", "f"], event_id=["e", "f"], split="test")
    frame = pd.concat([train, validation, test], ignore_index=True)
    held = frame[frame.split.ne("train")].reset_index(drop=True)
    first, selected = historical_predict(frame, held.copy(), tmp_path)
    assert selected["pooled"]["strength"] is not None
    assert (first.historical_cluster < first.historical_family).all()
    assert (first.historical_family < first.historical_pooled).all()
    assert first.historical_pooled.lt(0.3).all()
    estimates = pd.read_csv(tmp_path / "historical_training_estimates.csv")
    frame.loc[frame.split.eq("test"), "outcome"] = 1 - frame.loc[frame.split.eq("test"), "outcome"]
    held.loc[held.split.eq("test"), "outcome"] = 1 - held.loc[held.split.eq("test"), "outcome"]
    second, changed = historical_predict(frame, held.copy(), tmp_path)
    assert selected == changed
    assert selected["family"]["width"] == selected["cluster"]["width"] == selected["pooled"]["width"]
    pd.testing.assert_frame_equal(first.filter(like="historical_"), second.filter(like="historical_"))
    pd.testing.assert_frame_equal(estimates, pd.read_csv(tmp_path / "historical_training_estimates.csv"))
