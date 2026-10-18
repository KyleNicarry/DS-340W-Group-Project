"""Create an aligned Kalshi subset covering its latest year of trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def sql_path(path: Path) -> str:
    """Escape a local path for use in a DuckDB SQL string literal."""
    return str(path).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/kalshi"))
    parser.add_argument("--output", type=Path, default=Path("data/kalshi/recent_year"))
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.output}")
    if not list((args.input / "markets").glob("*.parquet")) or not list((args.input / "trades").glob("*.parquet")):
        raise FileNotFoundError(f"Expected market and trade Parquet files below {args.input}")

    market_glob = sql_path(args.input / "markets" / "*.parquet")
    trade_glob = sql_path(args.input / "trades" / "*.parquet")
    args.output.mkdir(parents=True)
    market_output = sql_path(args.output / "markets.parquet")
    trade_output = sql_path(args.output / "trades.parquet")

    db = duckdb.connect()
    db.execute("PRAGMA threads=8")
    db.execute(f"CREATE VIEW markets AS SELECT * FROM read_parquet('{market_glob}')")
    db.execute(f"CREATE VIEW trades AS SELECT * FROM read_parquet('{trade_glob}')")
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
        f"""
        COPY (
            SELECT * FROM trades
            WHERE created_time >= ? AND created_time <= ?
        ) TO '{trade_output}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """,
        [cutoff, latest],
    )
    db.execute(
        f"""
        COPY (
            SELECT m.* FROM markets m
            SEMI JOIN read_parquet('{trade_output}') t USING (ticker)
        ) TO '{market_output}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )

    stats = db.execute(
        f"""
        SELECT
            (SELECT count(*) FROM read_parquet('{market_output}')) AS markets,
            (SELECT count(*) FROM read_parquet('{trade_output}')) AS trades,
            (SELECT count(DISTINCT ticker) FROM read_parquet('{trade_output}')) AS traded_markets,
            (SELECT count(*) FROM read_parquet('{trade_output}') t
             ANTI JOIN read_parquet('{market_output}') m USING (ticker)) AS orphan_trades
        """
    ).fetchone()
    manifest = {
        "source": str(args.input),
        "duckdb_version": duckdb.__version__,
        "completed": True,
        "window_start_inclusive": cutoff.isoformat(),
        "window_end_inclusive": latest.isoformat(),
        **dict(zip(("markets", "trades", "traded_markets", "orphan_trades"), stats)),
    }
    db.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
