# Prediction Market Mispricing — DS 340W

We build on Jonathan Becker's *The Microstructure of Wealth Transfer in Prediction Markets*. Becker studies systematic mispricing across Kalshi. Our extension asks whether finer groups of similar contracts help predict mispricing on held-out events.

The project has two notebooks:

| Notebook | Purpose |
| --- | --- |
| [replication.ipynb](extension/notebooks/replication.ipynb) | Reproduce Becker's calibration, longshot, YES/NO, and maker/taker checks on our recent-year CSV subset. |
| [main.ipynb](extension/notebooks/main.ipynb) | Compare pooled predictions with predictions that include structural and text-based contract groups. |

Both include explanations, figures, and saved results. They run independently; reading the replication first provides context for the extension. Becker's original setup, framework documentation, and bibliography are preserved in [README_UPSTREAM.md](README_UPSTREAM.md).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use a copy of this repository's `main` branch. From the repository root:

```bash
uv sync --locked --group notebooks
uv run --locked --group notebooks python scripts/setup_data.py
uv run --locked --group notebooks jupyter lab extension/notebooks/
```

`uv` uses the project's Python 3.9 version and installs the locked dependencies. The setup script downloads the shared dataset and prepares it for both notebooks. Then open either notebook, select the project's Python environment, and run all cells from top to bottom. In VS Code, select the `.venv` kernel (`.venv/bin/python` on macOS/Linux, `.venv\Scripts\python.exe` on Windows). Run from the repository or a subdirectory so the notebooks can locate the inputs.

The dataset contains roughly 65 million trades. Allow at least 25 GB of free disk space for the downloads and extracted CSVs, plus room for notebook outputs and temporary data. The notebooks use DuckDB to summarize trades before loading smaller tables into memory; allow several GB of RAM.

## Data

The delivered sample covers November 25, 2024 through November 25, 2025. `scripts/setup_data.py` downloads `raw_data.zip`, `split.zip`, and `manifest.json` from the [shared Google Drive folder](https://drive.google.com/drive/folders/1WOUfs-T62UJGP6rafvvSuSlxwAPmMFc8?usp=share_link). It verifies the downloaded files against the supplied sample's SHA-256 hashes, extracts the CSVs, and checks their columns and row counts against the manifest. Downloads are retained in `data/.kalshi_downloads/` and reused on later attempts.

The script stages extraction before publishing the dataset. If `data/kalshi_recent_year_csv/` already exists, it checks its files and leaves them in place instead of overwriting them. To check an existing installation explicitly:

```bash
uv run --locked --group notebooks python scripts/setup_data.py --check
```

If automatic download fails, download the three files individually from the Drive folder into a separate directory and run the same verification and extraction steps locally:

```bash
uv run --locked --group notebooks python scripts/setup_data.py --from-dir /path/to/downloads
```

For a separate installation, add `--output /path/to/new/dataset`. The notebooks expect the default location shown below; the custom output is useful for checking or preparing a second copy. A partial or different existing dataset is never overwritten: use a new output location or move the existing directory yourself before retrying.

```text
data/kalshi_recent_year_csv/
├── master/{markets,trades}.csv
├── split/
│   ├── train/{markets,trades}.csv
│   ├── validation/{markets,trades}.csv
│   └── test/{markets,trades}.csv
└── manifest.json
```

The replication uses the master CSVs and manifest. The extension also uses the market files defining the event splits; it reads trades from the master file. Data and generated output are ignored by Git.

### Optional: rebuild from the original data

To create a new extract instead of downloading the prepared sample, obtain the original Kalshi Parquets using [Becker's setup instructions](README_UPSTREAM.md#installation--usage), then run:

```bash
uv run --locked --group notebooks python scripts/build_kalshi_recent_year_csv.py
```

The builder selects the latest year in its input data and refuses to overwrite an existing output directory. It writes its own manifest. A different raw extract can produce a different sample. See [CSV preparation](scripts/README.md) for the window, split rules, and reproducibility details.

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
