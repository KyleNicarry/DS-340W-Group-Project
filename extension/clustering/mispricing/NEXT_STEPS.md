# Next steps: persistence and out-of-sample validation

The current analysis finds candidate differences between structural contract
families, but has not established that a family is *usually* overpriced or
underpriced. The priority is testing persistence, rather than adding clustering
complexity. See [structural findings](STRUCTURAL_FINDINGS.md) for the starting point.

## 1. Separate direction from absolute mispricing

For each sufficiently supported family, report YES calibration error, NO
calibration error, and their matched-cost wedge separately. A negative wedge
means YES underperformed NO; it does not establish that YES was absolutely
overpriced or NO underpriced. Keep probability-point errors distinct from returns.

## 2. Check whether the historical pattern repeats

Recalculate these estimates by calendar month or quarter, matching purchase
costs and retaining event-level uncertainty. Report event counts, price coverage,
weight concentration and missing estimates. Use both a fixed common-price cohort
and a clearly labeled available-support sensitivity so changing price support
does not masquerade as a changing bias.

Prioritize the touchdown and press-secretary-mention candidates, but report the
full prespecified eligible family set. Check whether signs and magnitudes persist
across periods and whether removing the largest contributing events changes the
conclusion. Retain execution weighting as primary, with unit and contract-balanced
weighting as sensitivities. Treat zero/one win-rate cells and zero empirical
variance as warnings about inference, not evidence of perfect precision.

## 3. Make chronological out-of-sample testing the main next experiment

**Yes: this deserves focused time.** Historical pooled differences alone cannot
show that a family bias transfers to later contracts.

- Freeze family rules, cost support, minimum sample requirements, weighting,
  candidate selection and evaluation criteria before evaluating a new period.
- Use earlier periods to estimate each family's direction-specific calibration
  errors and wedge; carry those estimates forward without refitting on later outcomes.
  Start with this simple structural estimator before testing learned subclusters.
- Compare later-period errors with a pooled historical calibration/wedge baseline
  using identical costs and contracts. Report MAE, sign agreement and uncertainty
  in the improvement; do not count noisy near-zero signs as confident successes.
- Keep entire underlying events together and purge events spanning cutoffs.
  Use event-block uncertainty and check dependence across recurring series.
- Use rolling or expanding training windows. Tune any smoothing or settings only
  inside the training period; do not select the best specification on the final test.

The existing train/validation/test files are event-hash splits, not chronological
holdouts. Moreover, outcomes across the current extract have already informed
this exploratory analysis. A time split of that extract is useful **retrospective
validation**, but cannot become a fresh confirmatory holdout by renaming it.
Reserve genuinely unexamined later data for the final test.

## 4. Establish what was knowable at each cutoff

A deployment-style test requires training outcomes to have been publicly known
before each prediction, and contract wording/rules to have been available then.
The current extract lacks verified outcome-publication timestamps and versioned
metadata. Acquire those records where possible; do not substitute close time or
final metadata fetch time for settlement publication. Until this is resolved,
label chronological results as retrospective persistence checks rather than a
verified historical forecasting backtest. Execution timestamps alone do not fix
label-availability leakage.

## Decision after validation

Continue developing a family signal if its later-period calibration errors and
wedge direction repeat, it improves on the pooled baseline with useful uncertainty,
and the result survives event-concentration and weighting checks. If differences
are unstable, report that limitation and consider partial pooling for sparse
families before increasing model complexity. Revisit learned subclusters only if
they add out-of-sample value over the structural baseline. A trading-profit test
would be a later, separate project requiring executable prices, fees and fills.
