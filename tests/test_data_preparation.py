"""Fixture-based checks for local dataset alignment and event-grouped splits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def raw_data(tmp_path):
    root = tmp_path / "raw's"
    (root / "markets").mkdir(parents=True)
    (root / "trades").mkdir()
    markets = [{"ticker": f"M{i}", "event_ticker": f"E{i // 2}", "result": "yes"} for i in range(100)]
    markets += [
        {"ticker": "FALLBACK", "event_ticker": None, "result": "no"},
        {"ticker": "OLD", "event_ticker": "OLD_EVENT", "result": "no"},
    ]
    trades = [
        {"trade_id": f"T{i}", "ticker": m["ticker"], "created_time": "2025-11-25T12:00:00Z", "count": 1}
        for i, m in enumerate(markets[:-1])
    ]
    trades += [
        {"trade_id": "boundary", "ticker": "M0", "created_time": "2024-11-25T12:00:00Z", "count": 1},
        {"trade_id": "old", "ticker": "OLD", "created_time": "2024-11-25T11:59:59Z", "count": 1},
    ]
    trade_frame = pd.DataFrame(trades)
    trade_frame["created_time"] = pd.to_datetime(trade_frame.created_time, utc=True)
    pd.DataFrame(markets).to_parquet(root / "markets/part.parquet", index=False)
    trade_frame.to_parquet(root / "trades/part.parquet", index=False)
    return root


def run_script(source, output):
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/build_kalshi_recent_year_csv.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )


def test_csv_layout_alignment_window_and_reproducible_splits(raw_data, tmp_path):
    output = tmp_path / "output's"
    process = run_script(raw_data, output)
    assert process.returncode == 0, process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["duckdb_version"]
    groups = {}
    all_markets = set()
    all_trades = set()
    for split in ["train", "test", "validation"]:
        markets = pd.read_csv(output / "split" / split / "markets.csv")
        trades = pd.read_csv(output / "split" / split / "trades.csv")
        assert set(trades.ticker).issubset(set(markets.ticker))
        assert not all_markets.intersection(markets.ticker)
        assert not all_trades.intersection(trades.trade_id)
        all_markets.update(markets.ticker)
        all_trades.update(trades.trade_id)
        for row in markets.itertuples():
            event = row.event_ticker if pd.notna(row.event_ticker) else row.ticker
            assert event not in groups or groups[event] == split
            groups[event] = split
        assert manifest["splits"][split]["markets"] == len(markets)
        assert manifest["splits"][split]["trades"] == len(trades)
    master_markets = pd.read_csv(output / "master/markets.csv")
    master_trades = pd.read_csv(output / "master/trades.csv")
    assert set(master_markets.ticker) == all_markets
    assert set(master_trades.trade_id) == all_trades
    assert "old" not in all_trades and "boundary" in all_trades
    assert manifest["master"]["markets"] == 101 and manifest["master"]["trades"] == 102
    second = tmp_path / "second"
    assert run_script(raw_data, second).returncode == 0
    for split in ["train", "test", "validation"]:
        assert set(pd.read_csv(second / "split" / split / "markets.csv").ticker) == set(
            pd.read_csv(output / "split" / split / "markets.csv").ticker
        )


@pytest.mark.parametrize("problem", ["duplicate_market", "orphan_trade"])
def test_preparation_rejects_ambiguous_or_unaligned_data(problem, raw_data, tmp_path):
    if problem == "duplicate_market":
        path = raw_data / "markets/part.parquet"
        data = pd.read_parquet(path)
        pd.concat([data, data.iloc[:1]], ignore_index=True).to_parquet(path, index=False)
    else:
        path = raw_data / "trades/part.parquet"
        data = pd.read_parquet(path)
        data.loc[0, "ticker"] = "NO_METADATA"
        data.to_parquet(path, index=False)
    process = run_script(raw_data, tmp_path / "bad")
    assert process.returncode != 0
    assert ("unique" if problem == "duplicate_market" else "matching market") in process.stderr
    assert not (tmp_path / "bad/manifest.json").exists()


def test_preparation_refuses_existing_destination(raw_data, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing data")
    process = run_script(raw_data, output)
    assert process.returncode != 0 and "Refusing to overwrite" in process.stderr
    assert marker.read_text() == "existing data"
