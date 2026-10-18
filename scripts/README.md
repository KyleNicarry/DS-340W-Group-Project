# Kalshi data preparation

These utilities create reproducible local datasets from the downloaded Kalshi
Parquets. Run them from the repository root with the project environment installed.
The scripts belong in version control; generated data, manifests and archives stay
under ignored `data/`. None makes live API calls.

## Current CSV workflow

```bash
uv run python scripts/build_kalshi_recent_year_csv.py
# Optional explicit locations:
uv run python scripts/build_kalshi_recent_year_csv.py \
  --input data/kalshi --output data/kalshi_recent_year_csv_new
```

The builder selects executions from the latest observed trade timestamp minus one
calendar year through that latest timestamp, both inclusive. It retains metadata
for markets with trades in that window; it does not filter outcomes or finalization.
The input must contain `markets/*.parquet` and `trades/*.parquet`.

```text
data/kalshi_recent_year_csv/
├── master/
│   ├── markets.csv
│   └── trades.csv
├── split/
│   ├── train/{markets,trades}.csv
│   ├── test/{markets,trades}.csv
│   └── validation/{markets,trades}.csv
└── manifest.json
```

The master CSVs are exported directly from the raw source with the year filter;
they are not concatenated splits. This layout matches the clustering and mispricing
workflows. The script does not create ZIP archives.

Splits use `hash(coalesce(event_ticker, ticker) || seed) % 100`: buckets 0–69 are
train, 70–89 test, and 90–99 validation. Missing (NULL) event IDs fall back to ticker.
All contracts and trades sharing an event stay together. Percentages are approximate
by event hash, not guaranteed by trade count or category. These are **not chronological
holdouts**, do not separate recurring series, and do not establish historical label
availability. New out-of-sample research needs a separate chronological design.

The seed defaults to `ds340w-kalshi-v1`. DuckDB's hash is implementation-dependent;
keep the recorded DuckDB version and seed when reproducing assignments. A later raw
extract changes the latest-year window. The manifest records its boundaries, counts,
seed and DuckDB version; it does not cryptographically identify all raw files.

## Alternative Parquet utilities

```bash
# Create consolidated recent-year markets.parquet and trades.parquet.
uv run python scripts/subset_kalshi_recent_year.py

# Split a recent-year consolidated pair into event-grouped Parquet files.
uv run python scripts/split_kalshi.py \
  --input data/kalshi/recent_year --output data/kalshi/recent_year_splits

# Or split all available raw history (the splitter itself applies no date filter).
uv run python scripts/split_kalshi.py
```

`split_kalshi.py` accepts either consolidated `markets.parquet` / `trades.parquet`
or chunk directories `markets/*.parquet` / `trades/*.parquet`. Its splits are written
directly below its output directory, not under a second `split/` directory. The two
Parquet utilities are alternatives; they are not prerequisites for the CSV builder.

All three refuse to overwrite an existing output directory and fail on duplicate/
missing market IDs or in-window trades without matching metadata. A completed run
writes `manifest.json` with `completed: true`. A failed run may leave partial files;
use a new destination for a retry. Allow substantial disk space and time for the
full extract. These preparation steps do not alter the existing research CSVs.

## Repository history

The earlier `codex-split-data` branch was merged in `7a86994`; its source commit
`5141524` tracked `AGENTS.md`, but not these preparation scripts. They remained
untracked during the subsequent clustering work. This addition preserves the
preparation utilities and aligns the CSV builder with the existing `master/` and
`split/` dataset layout; it does not regenerate or replace the current extract.
