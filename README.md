# Prediction Market Research — DS 340W

This project extends Jonathan Becker’s [prediction-market-analysis repository](https://github.com/jon-becker/prediction-market-analysis) and research, *The Microstructure of Wealth Transfer in Prediction Markets*. Our research asks whether similarities in early market activity and question wording help predict group-level YES/NO mispricing differences (the “optimism-tax” wedge) in Kalshi markets.

The repository contains the inherited data collection and analysis framework, our article replication workflow, and our research extension. See [contributions and provenance](docs/contributions.md) for the boundary between upstream work and our additions.

## Start here

- **Our research:** [methodology and run instructions](extension/README.md), [findings and figures](extension/results/FINDINGS.md), and [data limitations](extension/DATA_AVAILABILITY.md).
- **Structured CSV analysis:** [contract clustering and title review](extension/clustering/README.md), [mispricing results](extension/clustering/mispricing/FINDINGS.md), and [data preparation](scripts/README.md).
- **Article replication:** [workflow and sources](replication/README.md).
- **Inherited framework:** [original documentation](docs/upstream.md), [data schemas](docs/SCHEMAS.md), and [writing analyses](docs/ANALYSIS.md).

The extension estimates constructed group-by-cost-band labels, not observed contract-level wedges. Its evaluation is a retrospective unseen-series holdout, not a chronological trading backtest. Read the methodology and limitations before interpreting the results.

## Run the extension

Run all commands from the repository root. Install the project and development dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run python -m extension.pipeline
```

Inputs must already exist at `data/kalshi/{markets,trades}/*.parquet`. The pipeline does not download data. For the inherited dataset download and collection commands, see [upstream setup](docs/upstream.md#installation--usage). A separate virtual-environment setup is documented in the [extension instructions](extension/README.md).

The default run writes to `extension/results/` and replaces outputs with the same names. Full runs can take several minutes and several GB of memory and disk. To check the extension without running the full research pipeline:

```bash
uv run pytest tests/extension -q
```

## Repository layout

```text
README.md                 Project overview and navigation
src/                      Inherited analysis, indexers, and shared utilities
main.py                   Inherited framework entry point
extension/                Our clustering, wedge estimation, models, and reports
    results/              Delivered extension findings, figures, and tables
replication/              Our workflow for reproducing the original article
tests/                   Framework and extension correctness tests
data/                     Shared input datasets
output/                   Framework, CSV clustering, and replication output
docs/                    Framework documentation and contribution history
```

The inherited framework writes analysis outputs to `output/`; article builds write to `output/pdf/`. The original extension results live in `extension/results/`; CSV clustering writes to `output/clustering/`, and its mispricing workflow writes to `extension/clustering/mispricing/results/`. The directory formerly named `mvp/` is now `extension/`; use `python -m extension.pipeline` for new runs.

## Attribution and license

The inherited framework and dataset originate from Jonathan Becker and upstream contributors. The research extension and replication workflow are this project team’s additions; contributions by Kyle Nicarry and GitHub user `c-kva` belong to the extension work. See the [provenance record](docs/contributions.md) and retained [MIT license](LICENSE). The original documentation’s research bibliography is preserved in [docs/upstream.md](docs/upstream.md#research--citations).
