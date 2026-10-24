# Contributions and provenance

## Inherited work

This repository builds on [jon-becker/prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis), Jonathan Becker’s prediction-market research, and contributions from the upstream community. The inherited baseline immediately preceding the local team’s first replication commit in the current branch is commit [`2276382cb616107db8c8647803bffa4a0d7091f8`](https://github.com/jon-becker/prediction-market-analysis/commit/2276382cb616107db8c8647803bffa4a0d7091f8).

The inherited framework includes `src/`, `main.py`, the original data collection scripts, the Makefile, framework documentation and tests, and project tooling. Shared configuration and dependencies may subsequently be extended by the team. The original README is preserved in [upstream.md](upstream.md), with documentation links adjusted for its new location. The upstream [MIT license](../LICENSE) remains in place.

## Team additions

Contributions by Kyle Nicarry and GitHub user `c-kva` are team extension work. Directories describe the purpose of the work; Git history records individual authorship.

| Location | Purpose |
| --- | --- |
| `extension/` (formerly `mvp/`) | Early-market and title features, contract clustering, exact-cost wedge estimates, event-bootstrap uncertainty, grouped model evaluation, and generated findings |
| `tests/extension/` (formerly `tests/mvp/`) | Correctness tests for estimation, data preparation, and leakage boundaries |
| `extension/clustering/` (formerly `src/analysis/kalshi/clustering/`) | Team CSV contract clustering, title-review browser, pooled/grouped mispricing comparisons, and structural-family analysis |
| `tests/extension/clustering/` | Correctness tests for the CSV clustering, viewer, and mispricing workflows |
| `scripts/{build_kalshi_recent_year_csv,subset_kalshi_recent_year,split_kalshi}.py` and `tests/test_data_preparation.py` | Team recent-year subset, CSV export, and event-preserving split preparation |
| `replication/` | Editable article reproduction and figure-generation workflow built on the original research |
| Root README and contribution documentation | Team research overview, navigation, and attribution |
| Shared project configuration | Dependencies and settings needed to run the team’s work alongside the inherited framework |

The reproduction of the original article is a team-created workflow; its underlying research remains attributed to Becker. Shared data are upstream inputs, not a new dataset created by this team.

## Existing results after the rename

The directory rename does not rerun the analysis. Existing result tables and figures are retained. Historical `run_manifest.json` and log files retain their original `mvp/` paths, invocation details, and source hashes as evidence of the delivered run; those hashes describe the source before this reorganization. The findings page’s reproduction command now uses `extension.pipeline`.

New runs use `extension/results/` by default and record the current source hashes. Use a separate `--output` directory when comparing a new run with the delivered results.

## CSV workflow relocation

The team clustering and mispricing code added in the newer remote commits now lives in `extension/clustering/`. Run it with `python -m extension.clustering.pipeline` and `python -m extension.clustering.mispricing.pipeline`. Existing ignored caches moved with the directory; historical manifests retain their original paths and source hashes. Tracked findings retain their reported results with reproduction commands updated. Shared data-preparation scripts stay in `scripts/` and are identified above as team contributions.
