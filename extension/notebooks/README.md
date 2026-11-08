# Completed methodology: notebook walkthrough

These four notebooks explain the final CSV extension from inputs to conclusions.
They follow the staged, reusable-function style of the earlier
[experimental notebooks](../../notebooks/extension_prototype/README.md), and add
the completed historical-calibration and supervised-prediction stages. Existing
notebooks and analysis modules remain unchanged.

| Order | Notebook | What it explains |
| --- | --- | --- |
| 1 | [Data and clustering](01_data_and_clustering.ipynb) | Actual CSV split membership; structural rules; training-only TF-IDF/LSA and hierarchical grouping; frozen assignment; coverage and rejection |
| 2 | [Measuring mispricing](02_measuring_mispricing.ipynb) | Becker's calibration definition; executed positions; audit/cache; exact-cost YES/NO comparisons; descriptive versus predictive evidence; structural-only analysis |
| 3 | [Historical and supervised prediction](03_historical_and_supervised_prediction.ipynb) | Common cohort; combination exclusions; training-only historical means; event support and shrinkage; supervised fits; validation-only selection; saved predictions |
| 4 | [Evaluation and conclusions](04_evaluation_and_conclusions.ipynb) | Test Brier/MAE/RMSE; direction-specific calibration and wedge errors; event/series uncertainty; corrected subgroup comparisons; final negative findings and limitations |

## Open and run

From the repository root:

```bash
uv sync
uv pip install --python .venv/bin/python -r extension/notebooks/requirements.txt
.venv/bin/python -m jupyterlab extension/notebooks
```

Select the project `.venv` Python kernel. Start Jupyter inside the repository;
each notebook finds the root from its working directory. Run top to bottom in a
fresh kernel. Registering a named kernel is optional if your Jupyter installation
cannot select the project Python directly. Never rely on out-of-order hidden state.

Authored notebooks have no saved outputs. Executed copies are generated locally
under ignored `results/validation/`, so the data-heavy outputs are not committed.
This set is an explanatory completion walkthrough of the existing experiment,
not a new study or a replacement for the authoritative Python functions.

## Default: review the completed real-data experiment

`RECOMPUTE=False` is the default. The notebooks resolve actual CSV, assignment and
statistics paths from `extension/clustering/supervised/results/run_manifest.json`.
They require its completion marker and matching implementation hashes. They then
show real recorded coverage, estimates, settings, predictions and evaluation
results. Raw split membership is checked explicitly. Prediction previews are
bounded leading samples, never representative performance estimates. Small
hand-written arithmetic examples are labeled synthetic and do not enter findings.

Review mode does not fit text encoders/models again or rescan the full trade CSV.
It still needs the prepared CSVs and local generated artifacts; these are not
committed. It does not download data or silently substitute a demonstration dataset.
The earlier Parquet/first-day notebook is a separate exploratory route, not the
source of the final CSV experiment's results.

If the canonical artifacts are absent, first prepare the CSVs following
[scripts/README.md](../../scripts/README.md), then run the established workflows:

```bash
.venv/bin/python -m extension.clustering.pipeline
.venv/bin/python -m extension.clustering.mispricing.pipeline
.venv/bin/python -m extension.clustering.mispricing.structural
.venv/bin/python -m extension.clustering.supervised.pipeline \
  --stats extension/clustering/mispricing/results/sufficient_statistics.csv.gz
```

On this workspace the completed run references a read-only legacy cache under
`src/analysis/kalshi/clustering/mispricing/results/`; the notebooks use the actual
recorded path instead of copying labels from an incompatible cache. The setup
commands above create new caches under extension/ and record those paths instead.
The descriptive CLI commands can replace their named reports, as documented in
those workflows. Notebook recomputation below is isolated.

## Optional: reproduce stages in isolated directories

Configuration cells expose `RUN_NAME`, `RECOMPUTE`, and `USE_REBUILT_UPSTREAM`.
All notebook writes are restricted to `extension/notebooks/results/<RUN_NAME>/`.
Reruns replace files within their named stage directory; choose another run name
for a separate specification. The existing canonical run is the reference, so
these notebooks require that completed run even when reproducing its stages.

1. Set notebook 01 `RECOMPUTE=True` to rebuild text representations, grouping,
   profiles and review tables under `clustering/`.
2. To use those new assignments, set notebook 02 `RECOMPUTE=True` and
   `USE_REBUILT_UPSTREAM=True`. It rescans the CSVs into a fresh `measurement/`
   cache, exports pooled curves and overall pooled/family cluster contrasts.
   This notebook does not repeat every old band-specific or structural inferential
   report; the prior notebooks 03/04 remain the full interfaces for those analyses.
3. Set notebook 03 `RECOMPUTE=True` and `USE_REBUILT_UPSTREAM=True` to train on
   that new cache. To reproduce the existing canonical grouping/cache instead,
   leave `USE_REBUILT_UPSTREAM=False`. Fit artifacts go under `models/`.
4. Set notebook 04 `RECOMPUTE=True` and `USE_REBUILT_UPSTREAM=True` to evaluate
   the fitted notebook predictions and create an isolated report under `evaluation/`.
   Leaving `USE_REBUILT_UPSTREAM=False` evaluates canonical predictions instead.

New assignments must be paired with freshly extracted statistics; source paths,
sizes and modification times are verified. A different file path alone can invalidate
an audit even if labels look identical. Never silently relabel an old cache.

Each stage records completion, its input fingerprints, research source hashes,
notebook authored-cell hashes, versions and configuration. A completed prediction-fit
stage explicitly says evaluation is pending; it is not a completed inference run.
The evaluation manifest separately identifies the prediction artifact it evaluated.
Output-free authored-cell hashes ignore kernel metadata and saved output rendering.
For a programmatically overridden parameter cell, the verifier also records the
exact overrides and saves the executed notebook copy.

## Verification

Install the notebook requirements first, then:

```bash
# Schemas, Python compilation, and fresh-kernel real-artifact review of all four.
.venv/bin/python extension/notebooks/verify.py

# Also refit notebook 03 and reevaluate notebook 04 on the real CSV cohort,
# then compare selected settings and four numeric/count tables to CLI artifacts.
.venv/bin/python extension/notebooks/verify.py --recompute-prediction

make test
make lint
```

Real Jupyter kernels require permission to bind local connection ports. The optional
prediction rerun takes several minutes and several GB of RAM; extraction/grouping
can also be expensive. No recomputation is triggered by merely opening a notebook.
The four parity tables are metrics, coverage, wedge metrics and paired comparisons;
numeric tolerances are rtol=1e-10 and atol=1e-12 to allow floating-point accumulation
order differences. The same historical width/strength selections must match.
The verifier does not independently rerun all descriptive/clustering inference or
validate the economic assumptions. Results, executed copies and verification status
stay under ignored results/.

## Reading the outcome

The selected historical baselines inherited their parents at every level, ultimately
matching raw market prices. This is a validation selection result, not proof that
true mispricing is zero. Original unshrunk cluster ridge had worse test point
performance than pooled ridge, while the selected small correction had an
inconclusive tiny improvement against price. The notebooks distinguish these cases.

There is no fixed 72-hour horizon, chronological label-availability guarantee,
fresh confirmatory holdout or trading-profit claim. Related outcomes may occur
under different event IDs and test results informed later revisions. See the
[methodology](../clustering/supervised/METHODOLOGY.md),
[full report](../clustering/supervised/REPORT.md), and
[change log](../clustering/supervised/CHANGES.md).

## Verified local run

All four notebooks passed schema checks, code compilation and review-mode execution
in separate Jupyter kernels against the real artifacts. Notebook 03 was also refitted
on the actual CSV cohort and notebook 04 reevaluated those predictions; the four
parity tables and selected historical settings matched the canonical CLI run within
the documented tolerance. The full suite passed 171 tests, and repository lint and
formatting passed. Notebook 01/02 full recompute branches were not rerun as part of
this verification; their review branches were executed and all source cells compiled.

After verification, inspect these locally generated executed review copies:

- [Executed data and clustering](results/validation/01_data_and_clustering.ipynb)
- [Executed mispricing measurement](results/validation/02_measuring_mispricing.ipynb)
- [Executed prediction walkthrough](results/validation/03_historical_and_supervised_prediction.ipynb)
- [Executed evaluation and conclusions](results/validation/04_evaluation_and_conclusions.ipynb)

The `recomputed_03_...` and `recomputed_04_...` copies in that same directory contain
the full optional prediction/evaluation execution. `verification.json` records the
completion marker, notebook cell hashes, parameter overrides and parity checks.
These generated files remain ignored and need to be recreated on another checkout.
