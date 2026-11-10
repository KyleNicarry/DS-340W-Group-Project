# Prediction Market Mispricing — DS 340W

We build on Jonathan Becker's *The Microstructure of Wealth Transfer in Prediction Markets*. Becker studies systematic mispricing across Kalshi. Our extension asks whether finer groups of similar contracts help predict mispricing on held-out events.

The project has two notebooks:

| Notebook | Purpose |
| --- | --- |
| [replication.ipynb](extension/notebooks/replication.ipynb) | Reproduce Becker's calibration, longshot, YES/NO, and maker/taker checks on our recent-year CSV subset. |
| [main.ipynb](extension/notebooks/main.ipynb) | Compare pooled predictions with predictions that include structural and text-based contract groups. |

Both include explanations, figures, and saved results. They run independently; reading the replication first provides context for the extension. Becker's original setup, framework documentation, and bibliography are preserved in [README_UPSTREAM.md](README_UPSTREAM.md).

## Setup

Use Python 3.9 and `uv`, matching the project's `.python-version` and locked environment. From the repository root:

```bash
uv sync
uv pip install --python .venv/bin/python -r extension/notebooks/requirements.txt
uv run --no-sync jupyter lab extension/notebooks/
```

Open either notebook, select the project's Python environment, and run all cells from top to bottom. In VS Code, select `.venv/bin/python` as the notebook kernel. Run from the repository or a subdirectory so the notebooks can locate the inputs.

The CSVs contain roughly 65 million trades. The notebooks use DuckDB to summarize them before loading smaller tables into memory; allow several GB of RAM and sufficient disk space for the input files and temporary data.

## Data

The delivered analysis uses trades from November 25, 2024 through November 25, 2025. Data files are ignored by Git and must be available locally:

```text
data/kalshi_recent_year_csv/
├── master/
│   ├── markets.csv
│   └── trades.csv
├── split/
│   ├── train/markets.csv
│   ├── validation/markets.csv
│   └── test/markets.csv
└── manifest.json
```

The replication uses the master CSVs and manifest. The extension also uses the market files defining the event splits. The builder exports trade CSVs for each split too, but the notebooks read trades from the master file.

If the CSVs are missing, obtain the original Kalshi Parquets using [Becker's setup instructions](README_UPSTREAM.md#installation--usage), then run:

```bash
uv run python scripts/build_kalshi_recent_year_csv.py
```

The builder selects the latest year in its input data and refuses to overwrite an existing output directory. A different raw extract can therefore produce a different sample. See [CSV preparation](scripts/README.md) for the window, event split rules, and reproducibility details.

## Our extension

We predict the mispricing residual: **resolved YES outcome − observed YES price**, with price expressed as a probability. This builds on Becker's calibration-based mispricing measure. Mispricing and return on capital are distinct measures; the replication labels both explicitly.

We compare one shared ridge model per feature set:

| Approach | Features |
| --- | --- |
| Pooled | Price, price squared, and price cubed |
| Structural | Price terms plus structural family membership |
| Structural + NLP | Price terms plus family and text-cluster membership |

Structural families group contracts by domain/category, series, and proposition structure. Within families, TF-IDF and LSA represent the wording, and hierarchical clustering forms finer groups. Text representations and clusters are learned from training contracts; validation and test contracts use the frozen assignments. Group membership enters the model as categorical features.

Pooled is the main modeling baseline. Uncorrected market price is a separate benchmark that makes no fitted adjustment. We add each model's predicted residual to price, clip the resulting probability to [0, 1], and evaluate execution-weighted Brier score on identical test observations. Lower scores are better. Event-bootstrap intervals compare models, and a top-K diagnostic checks performance in clusters with the most training-event support.

The splits keep events together but are not chronological. Results are exploratory: this dataset informed earlier methodology revisions, and the analysis does not test fee-adjusted trading profits. The replication is restricted to our sample and cannot reproduce Becker's full historical coverage or every reported estimate.

## Outputs and checks

Running the notebooks writes figures, tables, and run manifests to:

- `output/replication/`
- `output/extension_notebook_simple/`

Generated outputs are ignored by Git. The notebooks retain displayed results for review.

```bash
make test
make lint
```

The tests cover the inherited framework and CSV preparation. The notebooks also check data consistency during execution.

## Attribution

The inherited baseline is commit `2276382` of Jonathan Becker's [prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis). Its code remains under `src/`, with its original tooling, documentation, and tests. Our additions are the two notebooks, CSV preparation utility and tests, and supporting setup documentation.

See [Becker's original README and references](README_UPSTREAM.md) and the retained [MIT license](LICENSE).
