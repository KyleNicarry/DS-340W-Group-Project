"""Bounded CSV scans producing execution sufficient statistics; no MVP imports."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def quote(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def extract(data: Path, assignments: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory={quote(output / 'spill')}")
    con.execute(
        f"CREATE VIEW markets AS SELECT * FROM read_csv({quote(data / 'master/markets.csv')}, all_varchar=true)"
    )
    con.execute(f"CREATE VIEW assignments AS SELECT * FROM read_csv({quote(assignments)}, all_varchar=true)")
    for table in ["markets", "assignments"]:
        if con.sql(f"SELECT count(*)-count(DISTINCT ticker) FROM {table}").fetchone()[0]:
            raise ValueError(f"Duplicate tickers in {table}")
    if con.sql(
        "SELECT count(*) FROM assignments a LEFT JOIN markets m USING(ticker) WHERE m.ticker IS NULL"
    ).fetchone()[0]:
        raise ValueError("Assignments contain tickers absent from master CSV")
    con.execute(f"""CREATE VIEW trades AS SELECT trade_id,ticker,taker_side,
        try_cast(yes_price AS BIGINT) p,try_cast(no_price AS BIGINT) np,
        try_cast(count AS BIGINT) units FROM read_csv({quote(data / "master/trades.csv")},all_varchar=true)""")
    audit = dict(
        zip(
            ["raw_trades", "distinct_trade_ids", "missing_trade_ids", "invalid_trades"],
            con.sql("""
        SELECT count(*),count(DISTINCT trade_id),count(*) FILTER(WHERE trade_id IS NULL OR trade_id=''),
        count(*) FILTER(WHERE p IS NULL OR np IS NULL OR units IS NULL OR p NOT BETWEEN 1 AND 99
        OR np NOT BETWEEN 1 AND 99 OR p+np<>100 OR units<=0 OR taker_side IS NULL OR taker_side NOT IN ('yes','no'))
        FROM trades""").fetchone(),
        )
    )
    if audit["missing_trade_ids"] or audit["raw_trades"] != audit["distinct_trade_ids"]:
        raise ValueError("Missing/duplicate trade IDs: reconcile explicitly before analysis")
    con.execute("""CREATE TEMP TABLE cells AS SELECT ticker,p,count(*)::BIGINT n,sum(units)::DOUBLE units
        FROM trades WHERE p BETWEEN 1 AND 99 AND np BETWEEN 1 AND 99 AND p+np=100
        AND units>0 AND taker_side IN ('yes','no') GROUP BY ticker,p""")
    con.execute("""CREATE TEMP TABLE eligible AS SELECT m.ticker,m.event_ticker,m.title,m.result,
        coalesce(nullif(m.event_ticker,''),m.ticker) event_id,
        CAST(m.result='yes' AS INTEGER) outcome, sum(c.units) AS notional_usd
        FROM markets m JOIN cells c USING(ticker)
        WHERE m.status='finalized' AND m.result IN ('yes','no') AND m.market_type='binary'
        GROUP BY ALL HAVING sum(c.units)>=100""")
    audit["raw_markets"] = con.sql("SELECT count(*) FROM markets").fetchone()[0]
    audit["eligible_markets"] = con.sql("SELECT count(*) FROM eligible").fetchone()[0]
    audit["notional_definition"] = (
        "sum(count) dollars: both complementary positions together cost $1 per unit; observed CSV window only"
    )
    frame = con.sql("""SELECT c.*,e.event_id,e.outcome,e.notional_usd,
        coalesce(a.cluster,'') AS "cluster",coalesce(a.split,'not_in_assignments') split,
        coalesce(a.status,'not_in_assignments') assignment_status,
        coalesce(a.family,'') AS "family",e.title,
        NOT starts_with(e.ticker,'KXMVE') AND NOT starts_with(e.ticker,'MVE') non_combination
        FROM cells c JOIN eligible e USING(ticker) LEFT JOIN assignments a USING(ticker)
        ORDER BY c.ticker,c.p""").df()
    frame.to_csv(output / "sufficient_statistics.csv.gz", index=False)
    con.close()
    return frame, audit
