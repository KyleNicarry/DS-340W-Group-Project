"""Build recent-year, event-grouped Kalshi CSV splits from raw Parquet data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

SPLITS = (("train", 0, 70), ("test", 70, 90), ("validation", 90, 100))


def sql_path(path: Path) -> str:
    """Escape a local path for use in a DuckDB SQL string literal."""
    return str(path).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/kalshi"))
    parser.add_argument("--output", type=Path, default=Path("data/kalshi_recent_year_csv"))
    parser.add_argument("--seed", default="ds340w-kalshi-v1")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.output}")
    market_files = args.input / "markets"
    trade_files = args.input / "trades"
    if not next(market_files.glob("*.parquet"), None) or not next(trade_files.glob("*.parquet"), None):
        raise FileNotFoundError(f"Expected raw market and trade Parquet files below {args.input}")

    markets = sql_path(market_files / "*.parquet")
    trades = sql_path(trade_files / "*.parquet")
    args.output.mkdir(parents=True)
    db = duckdb.connect()
    db.execute("PRAGMA threads=8")
    db.execute(f"CREATE VIEW markets AS SELECT * FROM read_parquet('{markets}')")
    db.execute(f"CREATE VIEW trades AS SELECT * FROM read_parquet('{trades}')")

    latest = db.execute("SELECT max(created_time) FROM trades").fetchone()[0]
    if latest is None:
        raise ValueError("No timestamped trades available for a recent-year subset")
    cutoff = db.execute("SELECT ?::TIMESTAMPTZ - INTERVAL '1 year'", [latest]).fetchone()[0]
    if db.execute("SELECT count(*)-count(DISTINCT ticker) FROM markets").fetchone()[0]:
        raise ValueError("Market tickers must be unique and non-null")
    orphans = db.execute(
        "SELECT count(*) FROM trades t ANTI JOIN markets m USING(ticker) WHERE created_time >= ? AND created_time <= ?",
        [cutoff, latest],
    ).fetchone()[0]
    if orphans:
        raise ValueError(f"{orphans} recent trades have no matching market")
    db.execute(
        """
        CREATE TEMP TABLE assignments AS
        SELECT m.ticker, m.event_ticker,
               hash(coalesce(m.event_ticker, m.ticker) || ?) % 100 AS bucket
        FROM markets m
        SEMI JOIN (SELECT DISTINCT ticker FROM trades WHERE created_time >= ? AND created_time <= ?) t USING (ticker)
        """,
        [args.seed, cutoff, latest],
    )

    manifest: dict[str, object] = {
        "source": str(args.input),
        "duckdb_version": duckdb.__version__,
        "window_start_inclusive": cutoff.isoformat(),
        "window_end_inclusive": latest.isoformat(),
        "seed": args.seed,
        "grouping_key": "event_ticker (ticker fallback)",
        "method": "seeded hash buckets; train 0-69, test 70-89, validation 90-99",
        "splits": {},
    }
    for name, lower, upper in SPLITS:
        split_dir = args.output / "split" / name
        split_dir.mkdir(parents=True)
        market_output = sql_path(split_dir / "markets.csv")
        trade_output = sql_path(split_dir / "trades.csv")
        db.execute(
            f"""
            COPY (
                SELECT m.* FROM markets m JOIN assignments a USING (ticker)
                WHERE a.bucket >= {lower} AND a.bucket < {upper}
            ) TO '{market_output}' (FORMAT CSV, HEADER)
            """
        )
        db.execute(
            f"""
            COPY (
                SELECT t.* FROM trades t JOIN assignments a USING (ticker)
                WHERE t.created_time >= ? AND t.created_time <= ?
                  AND a.bucket >= {lower} AND a.bucket < {upper}
            ) TO '{trade_output}' (FORMAT CSV, HEADER)
            """,
            [cutoff, latest],
        )
        stats = db.execute(
            f"""
            SELECT
                (SELECT count(*) FROM read_csv_auto('{market_output}', header=true)) AS markets,
                (SELECT count(*) FROM read_csv_auto('{trade_output}', header=true)) AS trades,
                count(DISTINCT coalesce(event_ticker, ticker)) AS events
            FROM read_csv_auto('{market_output}', header=true)
            """
        ).fetchone()
        manifest["splits"][name] = dict(zip(("markets", "trades", "events"), stats))

    total_trades = sum(split["trades"] for split in manifest["splits"].values())
    for split in manifest["splits"].values():
        split["trade_fraction"] = split["trades"] / total_trades

    master_dir = args.output / "master"
    master_dir.mkdir()
    master_markets = sql_path(master_dir / "markets.csv")
    master_trades = sql_path(master_dir / "trades.csv")
    db.execute(
        f"""
        COPY (
            SELECT m.* FROM markets m
            SEMI JOIN (
                SELECT DISTINCT ticker FROM trades
                WHERE created_time >= ? AND created_time <= ?
            ) t USING (ticker)
        ) TO '{master_markets}' (FORMAT CSV, HEADER)
        """,
        [cutoff, latest],
    )
    db.execute(
        f"""
        COPY (
            SELECT * FROM trades
            WHERE created_time >= ? AND created_time <= ?
        ) TO '{master_trades}' (FORMAT CSV, HEADER)
        """,
        [cutoff, latest],
    )
    manifest["master"] = {
        "source": "raw Kalshi Parquet tables with recent-year filter; not concatenated splits",
        "markets": sum(split["markets"] for split in manifest["splits"].values()),
        "trades": total_trades,
    }
    manifest["completed"] = True
    db.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
