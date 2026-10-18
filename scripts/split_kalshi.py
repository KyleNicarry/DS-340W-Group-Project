"""Create deterministic, event-grouped Kalshi train/test/validation Parquet splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

SPLITS = (("train", 0, 70), ("test", 70, 90), ("validation", 90, 100))


def parquet_source(input_dir: Path, name: str) -> str:
    """Resolve either a consolidated file or a directory of chunk files."""
    consolidated = input_dir / f"{name}.parquet"
    chunks = input_dir / name
    if consolidated.is_file():
        source = consolidated
    elif chunks.is_dir() and next(chunks.glob("*.parquet"), None):
        source = chunks / "*.parquet"
    else:
        raise FileNotFoundError(f"No {name} Parquet data found below {input_dir}")
    return str(source).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/kalshi"))
    parser.add_argument("--output", type=Path, default=Path("data/kalshi/splits"))
    parser.add_argument("--seed", default="ds340w-kalshi-v1")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing split directory: {args.output}")
    market_glob = parquet_source(args.input, "markets")
    trade_glob = parquet_source(args.input, "trades")

    args.output.mkdir(parents=True)
    db = duckdb.connect()
    db.execute("PRAGMA threads=8")
    db.execute(f"CREATE VIEW markets AS SELECT * FROM read_parquet('{market_glob}')")
    db.execute(f"CREATE VIEW trades AS SELECT * FROM read_parquet('{trade_glob}')")

    if db.execute("SELECT count(*)-count(DISTINCT ticker) FROM markets").fetchone()[0]:
        raise ValueError("Market tickers must be unique and non-null")
    if db.execute("SELECT count(*) FROM trades").fetchone()[0] == 0:
        raise ValueError("No trades available to split")
    orphans = db.execute("SELECT count(*) FROM trades t ANTI JOIN markets m USING(ticker)").fetchone()[0]
    if orphans:
        raise ValueError(f"{orphans} trades have no matching market")

    # Every contract in one parent event receives the same seeded hash bucket.
    # This prevents closely related contracts—and all of their trades—from leaking
    # across splits without imposing an artificial chronological cutoff.
    db.execute(
        """
        CREATE TEMP TABLE assignments AS
        SELECT ticker, event_ticker,
               hash(coalesce(event_ticker, ticker) || ?) % 100 AS bucket
        FROM markets
        """,
        [args.seed],
    )

    manifest: dict[str, object] = {
        "seed": args.seed,
        "duckdb_version": duckdb.__version__,
        "grouping_key": "event_ticker (ticker fallback)",
        "method": "seeded hash buckets; train 0-69, test 70-89, validation 90-99",
        "splits": {},
    }
    for name, lower, upper in SPLITS:
        split_dir = args.output / name
        split_dir.mkdir()
        market_path = str(split_dir / "markets.parquet").replace("'", "''")
        trade_path = str(split_dir / "trades.parquet").replace("'", "''")
        db.execute(
            f"COPY (SELECT m.* FROM markets m JOIN assignments a USING (ticker) "
            f"WHERE a.bucket >= {lower} AND a.bucket < {upper}) "
            f"TO '{market_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        db.execute(
            f"COPY (SELECT t.* FROM trades t JOIN assignments a USING (ticker) "
            f"WHERE a.bucket >= {lower} AND a.bucket < {upper}) "
            f"TO '{trade_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        stats = db.execute(
            f"""
            SELECT count(DISTINCT coalesce(m.event_ticker, m.ticker)) AS events,
                   count(*) AS markets,
                   count(*) FILTER (WHERE m.result = 'yes') AS yes_markets,
                   count(*) FILTER (WHERE m.result = 'no') AS no_markets,
                   (SELECT count(*) FROM read_parquet('{trade_path}')) AS trades
            FROM read_parquet('{market_path}') m
            """
        ).fetchone()
        manifest["splits"][name] = dict(zip(("events", "markets", "yes_markets", "no_markets", "trades"), stats))

    totals = {key: sum(split[key] for split in manifest["splits"].values()) for key in ("events", "markets", "trades")}
    for split in manifest["splits"].values():
        split["market_fraction"] = split["markets"] / totals["markets"]
        split["trade_fraction"] = split["trades"] / totals["trades"]
    manifest["completed"] = True
    db.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
