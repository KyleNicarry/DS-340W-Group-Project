# Kalshi CSV preparation

`build_kalshi_recent_year_csv.py` creates the inputs for both research notebooks from downloaded Kalshi Parquet files. Run it from the repository root after `uv sync`:

```bash
uv run python scripts/build_kalshi_recent_year_csv.py
# Optional separate destination:
uv run python scripts/build_kalshi_recent_year_csv.py \
  --input data/kalshi --output data/kalshi_recent_year_csv_new
```

The input must contain `markets/*.parquet` and `trades/*.parquet`. The script selects trades between the latest observed trade timestamp minus one calendar year and that latest timestamp, inclusive, then retains matching market metadata. It does not screen finalization or outcomes; the notebooks do that.

```text
data/kalshi_recent_year_csv/
├── master/{markets,trades}.csv
├── split/
│   ├── train/{markets,trades}.csv
│   ├── validation/{markets,trades}.csv
│   └── test/{markets,trades}.csv
└── manifest.json
```

The master CSVs are exported directly from the filtered source. Event-grouped splits use `hash(coalesce(event_ticker, ticker) || seed) % 100`: buckets 0–69 train, 70–89 test, and 90–99 validation. NULL event IDs fall back to the market ticker. Contracts in the same event stay together; these are random event holdouts, not chronological splits.

The default seed is `ds340w-kalshi-v1`. The manifest records the window, seed, counts, and DuckDB version. Reproducing an extract requires the same raw data and DuckDB version because the latest timestamp and hash implementation affect the result. Existing CSVs remain the inputs for the delivered notebook results; rebuilding from another raw extract may change those results.

The script refuses an existing output directory and rejects duplicate/missing market IDs or in-window trades without matching metadata. A successful run records `completed: true`; a failure can leave partial files, so retry in a new destination. Allow sufficient disk space for the large CSVs. Inputs and outputs stay under ignored `data/`; no live API calls are made.
