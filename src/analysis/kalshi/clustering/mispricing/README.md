# Mispricing within contract clusters

See [next steps](NEXT_STEPS.md) for the persistence checks and chronological
out-of-sample validation plan.

Read [FINDINGS.md](FINDINGS.md) for the completed CSV analysis, figures, uncertainty,
and interpretation. This workflow follows the original paper's exact-price
YES/NO calibration contrast; it does not use the MVP.

```bash
.venv/bin/python -m src.analysis.kalshi.clustering.mispricing.pipeline
# Reuse audited sufficient statistics if input metadata match:
.venv/bin/python -m src.analysis.kalshi.clustering.mispricing.pipeline --reuse-stats
```

Inputs are the recent-year CSV master markets/trades and the frozen clustering
assignments in `output/clustering/assignments.csv`. Both execution and unit
weighting are calculated. The analysis preserves every valid execution in the
CSV window for eligible resolved markets; it does not use an opening-day window.
Expect several minutes for CSV scans and covariance calculations. DuckDB is
limited to 3 GB with spill; pandas/SciPy also require several GB of memory.

The comparison separates all resolved, non-combination, and assigned-only pooled
populations. Clusters are compared with the assigned-only pool and their own
structural family, matching exact purchase costs. Event/market sandwich errors,
FDR and correlated max-T checks address dependent outcomes and many comparisons.
Direct equality tests use a complete 1–49-cent support cohort. Thresholds and
additional standardization choices are documented in the findings report.

Implementation:

- `data.py`: validate and aggregate CSVs, resolved/binary/$100 screens.
- `statistics.py`: position-level weights, exact-price curves, paired inference.
- `heterogeneity.py`: common-support direct equal-wedge tests.
- `pipeline.py`: reproducible analysis orchestration and provenance.
- `report.py`: findings and figures derived from saved tables.

Generated tables and compressed sufficient statistics are under ignored `results/`.
The report and figures are in this folder for review. Running the workflow replaces
these named files; use distinct output directories for different datasets/settings.
The manifest must say `completed: true` before a run is treated as complete.
Cached input checks compare paths, sizes and modification times, not raw-file hashes.

The initial completed analysis is exploratory. A large observed wedge is not
necessarily precise; zero/one cell win rates and concentrated event weights can
make normal-approximation inference unreliable. No trading or causal claim follows.

## Rule-only structural comparison

[STRUCTURAL_FINDINGS.md](STRUCTURAL_FINDINGS.md) reports the separate rerun using
only domain/underlying/structure rules, without text encoding or learned clusters:

```bash
.venv/bin/python -m src.analysis.kalshi.clustering.mispricing.structural
```

It recomputes labels from the master CSV, reuses the validated execution sufficient
statistics after checking inputs, and compares both all rule-eligible contracts
and exactly the earlier ML-assigned population. New tables live under
`results/structural/`; prior learned-group results remain intact.
