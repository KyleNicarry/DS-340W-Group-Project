"""Protect validation-only regularization and matched evaluation semantics."""

import numpy as np
import pandas as pd

from extension.clustering.supervised.evaluation import paired_test
from extension.clustering.supervised.pipeline import select_models


def test_validation_selection_never_uses_test_outcomes(tmp_path):
    frame = pd.DataFrame(
        {
            "split": ["validation"] * 4 + ["test"] * 4,
            "outcome": [0, 0, 1, 1] * 2,
            "market_price": [0.1, 0.2, 0.8, 0.9] * 2,
            "n": np.ones(8),
        }
    )
    for scope in ["pooled", "family", "cluster"]:
        for kind in ["ridge", "forest", "boosting"]:
            frame[f"{scope}_{kind}"] = [0.4, 0.4, 0.6, 0.6] * 2
    first, settings = select_models(frame.copy(), tmp_path)
    assert all(row["strength"] == 0 for row in settings.values())
    frame.loc[frame.split.eq("test"), "outcome"] = 1 - frame.loc[frame.split.eq("test"), "outcome"]
    second, changed = select_models(frame.copy(), tmp_path)
    assert settings == changed
    pd.testing.assert_frame_equal(first.drop(columns="outcome"), second.drop(columns="outcome"))


def test_event_inference_uses_blocks_instead_of_execution_count():
    frame = pd.DataFrame({"event_id": ["a", "a", "b", "b"], "n": [10000] * 4})
    result = paired_test(frame, np.array([0.1, 0.1, -0.1, -0.1]), 100)
    assert result["blocks"] == 2
    assert result["se"] == 0.1
    assert result["ci_low"] < 0 < result["ci_high"]


def test_subgroup_support_counts_events_and_exports_every_comparison(tmp_path):
    from extension.clustering.supervised.evaluation import subgroup_metrics

    frame = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "cluster": ["family/000"] * 4,
            "family": ["family"] * 4,
            "ticker": ["a", "a", "b", "b"],
            "event_id": ["a", "a", "b", "b"],
            "outcome": [0, 0, 1, 1],
            "n": [1000000] * 4,
        }
    )
    for model in [
        "market_price",
        "pooled_ridge",
        "family_ridge",
        "cluster_ridge",
        "selected_pooled",
        "selected_family",
        "selected_cluster",
    ]:
        frame[model] = [0.1, 0.2, 0.8, 0.9]
    subgroup_metrics(frame, tmp_path)
    comparisons = pd.read_csv(tmp_path / "cluster_comparisons.csv")
    assert len(comparisons) == 5
    assert comparisons.events.eq(2).all()
    assert not comparisons.inference_eligible.any()
    assert comparisons.holm_p.eq(1).all()


def test_combination_screen_catches_multi_award_outcomes_without_labels():
    from extension.clustering.supervised.models import combination_mask

    frame = pd.DataFrame(
        {
            "ticker": ["KXOSCARWINNERS-25-A", "KXOSCAR-25-A", "KXMVE-25-A", "OTHER-25-A"],
            "title": [
                "Will Anora,Sean Baker win Best Picture,Best Director at the Oscars?",
                "Will Anora win Best Picture at the Oscars?",
                "Combination",
                "Will X win?",
            ],
            "result": ["yes", "no", "yes", "no"],
        }
    )
    assert combination_mask(frame).tolist() == [True, False, True, False]
    frame["result"] = frame.result.map({"yes": "no", "no": "yes"})
    assert combination_mask(frame).tolist() == [True, False, True, False]
