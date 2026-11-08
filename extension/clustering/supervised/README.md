# Historical calibration transfer and supervised mispricing prediction

Read [REPORT.md](REPORT.md) for completed results, [METHODOLOGY.md](METHODOLOGY.md)
for equations, timing and inference, and [CHANGES.md](CHANGES.md) for the revision.

The primary comparison transfers training-set calibration at pooled,
structural-family and structural + NLP cluster levels. Sparse estimates shrink
through that hierarchy using effective event support. Price resolution and
smoothing are selected using validation outcomes only. Existing supervised
comparators remain, evaluated on identical held-out execution observations.
There is no fixed 72-hour horizon and no claim of a chronological backtest.

```bash
uv sync
.venv/bin/python -m extension.clustering.supervised.pipeline
```

Run from the repository root. Defaults read recent-year CSV markets/trades under
`data/kalshi_recent_year_csv/`, their actual split membership, frozen assignments
at `output/clustering/assignments.csv`, and the read-only legacy audited cache at
`src/analysis/kalshi/clustering/mispricing/results/sufficient_statistics.csv.gz`.
No full historical Parquet input is scanned. All new research code is in extension/.

To rebuild the sufficient-statistics cache under extension/ instead:

```bash
.venv/bin/python -m extension.clustering.mispricing.pipeline \
  --output extension/clustering/supervised/results/cache
.venv/bin/python -m extension.clustering.supervised.pipeline \
  --stats extension/clustering/supervised/results/cache/sufficient_statistics.csv.gz
```

Use `--output /path/to/results` for another output directory and `--bootstrap 2000`
for block-bootstrap repetitions. Runs replace named outputs and the adjacent
REPORT.md. Default generated results are ignored by Git. A valid manifest must
say `completed: true`. Inputs must match audit fingerprints; changed inputs require
fresh extraction. Do not commit raw datasets or generated predictions.

- historical.py: training means, event support, smoothing, validation selection.
- models.py: observable features, data-quality screen, residual regressors.
- pipeline.py: CSV split checks, common cohort, training and orchestration.
- evaluation.py: exact-cost direction errors, paired event/series inference.
- report.py: tables, figure and research report derived from saved results.

New outputs are `historical_training_estimates.csv`,
`historical_validation_search.csv`, and `historical_selected.json`.
`heldout_predictions.parquet` contains all methods' probabilities, including the
three historical models. `metrics.csv` includes Brier and residual MAE/RMSE;
`comparisons.csv` includes all corrected contrasts; `matched_cost_curves.csv.gz`,
`cluster_wedges.csv`, and `wedge_metrics.csv` retain the group-calibration targets.
Other outputs document eligibility, full model selection, subgroup screening and
provenance. The report describes every artifact and limitation.

```bash
make test
make lint
```

## Notebook walkthrough

The [completion notebooks](../../notebooks/README.md) explain the complete CSV
extension in four stages, with real-artifact review by default and isolated
recomputation of the prediction/evaluation functions.
