# Implementation revision: historical calibration transfer

## Motivation and scope

Following discussion of Becker's execution-based methodology, the research focus
was clarified: test whether historical calibration among comparable positions
transfers to held-out contracts before asking whether flexible ML adds value.
No fixed 72-hour forecast horizon or chronological split was introduced.

This revision follows inspection of earlier results. It is an exploratory extension
on the same CSV extract, not a fresh confirmatory test. Test labels do not enter
training or hyperparameter selection, but that does not erase prior exploration.

## Changes

- Added pooled -> structural-family -> NLP-cluster historical residual estimators.
- Smoothed local means using effective training-event support instead of treating
  repeated executions as independent evidence; unsupported cells inherit parents.
- Selected a shared price grid and scope-level smoothing with validation Brier only.
- Retained existing model features, model settings, eligibility screens and splits.
- Added direct comparisons of ML against historical estimators at the same scope,
  of grouping resolutions, and of historical estimators against raw market price.
- Expanded the aggregate Holm family from 13 to 25 contrasts and the subgroup
  family from five to ten contrasts per cluster. Previously reported adjusted
  p-values therefore change even where model predictions have not changed.
- Added individual-residual MAE/RMSE, mean-wedge RMSE, and direction-specific mean
  calibration-error MAE; exact-cost evaluation and signed errors remain exported.
- Reframed the generated report around historical calibration transfer, with full
  grids, support, uncertainty, negative results and explicit timing limitations.
- Added focused tests of event support, weighted estimates, parent fallback,
  shared price grids and exclusion of test outcomes from fit/selection.

## Files and preservation

New estimator code is in historical.py. Existing pipeline.py, evaluation.py,
report.py and README.md integrate and document the expanded experiment.
METHODOLOGY.md supplies the formulas and interpretation. Historical training
estimates, selection grids and chosen settings are generated under ignored results/.
The preceding session's tables, predictions, manifest and report were copied into
results/previous_supervised/ before rerunning. The earlier prefix-filter audit
remains in results/inherited_prefix_audit/. These are local generated artifacts.

The earlier Oscar candidate remains excluded: its many event IDs represented
multi-award conjunctions from the same ceremony, not independent outcome evidence.
No new eligibility screen or feature was selected to improve the new results.
