"""Economic boundaries, rejection, and held-out isolation checks."""

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.kalshi.clustering.features import extract_attributes
from src.analysis.kalshi.clustering.pipeline import group_contracts, load_contracts


@pytest.mark.parametrize(
    "title,subtitle,structure",
    [
        ("Bitcoin price?", "$94,750 or above", "above"),
        ("Bitcoin price?", "$94,750 or below", "below"),
        ("Bitcoin price?", "$97,500 to 97,999.99", "interval"),
        ("Chicago high temp?", "53° to 54°", "interval"),
        ("Team total points", "Over 51.5 points scored", "total"),
    ],
)
def test_structure_preserves_proposition(title, subtitle, structure):
    attrs = extract_attributes({"title": title, "yes_sub_title": subtitle, "event_ticker": "KXBTC-DATE"})
    assert attrs["structure"] == structure
    assert attrs["threshold_tokens"]


def test_economic_boundaries():
    a = extract_attributes({"title": "Bitcoin price?", "yes_sub_title": "100 or above", "event_ticker": "KXBTCD-A"})
    b = extract_attributes({"title": "Ethereum price?", "yes_sub_title": "100 or above", "event_ticker": "KXETHD-A"})
    assert a["family"] != b["family"]
    assert a["settlement_source"] == "unknown"


@pytest.mark.parametrize("method", ["hierarchical", "kmeans"])
def test_assignment_rejects_and_does_not_fit_holdout(method):
    frame = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "event_ticker": ["a", "b", "c", "d", "e"],
            "family": ["crypto/BTC/above"] * 4 + ["unknown/new/unknown"],
            "split": ["train", "train", "test", "test", "test"],
        }
    )
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    args = Namespace(
        family_fit_limit=100,
        method=method,
        clusters=8,
        distance=0.35,
        min_similarity=0.75,
        min_contracts=2,
        min_events=2,
    )
    result = group_contracts(frame, vectors, args)
    assert result.status.tolist() == ["assigned", "assigned", "assigned", "distant", "unseen_family"]
    changed = vectors.copy()
    changed[2:] *= -1
    second = group_contracts(frame, changed, args)
    pd.testing.assert_frame_equal(result.iloc[:2], second.iloc[:2])
    # Many contracts sharing one outcome must not satisfy independent-event support.
    frame.loc[1, "event_ticker"] = "a"
    assert not group_contracts(frame, vectors, args).status.eq("assigned").any()


def test_csv_loader_detects_event_leakage(tmp_path):
    row = {
        "ticker": "A",
        "event_ticker": "shared",
        "market_type": "binary",
        "title": "Bitcoin price?",
        "yes_sub_title": "100 or above",
        "open_time": "",
        "close_time": "",
    }
    for split in ["train", "validation", "test"]:
        path = tmp_path / "split" / split
        path.mkdir(parents=True)
        pd.DataFrame([{**row, "ticker": split}]).to_csv(path / "markets.csv", index=False)
    with pytest.raises(ValueError, match="Event IDs overlap"):
        load_contracts(tmp_path)


def test_sentence_backend_requires_local_checkpoint(tmp_path):
    from src.analysis.kalshi.clustering.pipeline import embed

    args = Namespace(backend="sentence", model=tmp_path / "missing")
    with pytest.raises(ValueError, match="local sentence-transformers checkpoint"):
        embed(pd.DataFrame(), args)
