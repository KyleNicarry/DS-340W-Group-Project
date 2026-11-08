# Methodology: transferring historical calibration

## Research question

Becker's section 3 equation (1) measures mispricing of a set of positions as its
realized win rate minus its mean purchase-implied probability. His equal-cost
YES/NO contrast is a difference between two such calibration errors. Our extension
tests whether training-set calibration transfers to held-out comparable positions,
and whether structural families or finer text clusters improve that transfer.
These shrinkage estimators and supervised prediction procedures are our additions.

The observation unit remains an execution, rather than a single chosen moment per
contract. The supplied CSV event-hash train/validation/test membership is preserved.
There is no 72-hour cutoff. Chronological forecasting is a separate design requiring
verified training-label availability; event disjointness alone does not establish it.

## Outcome, prediction and sufficient statistics

Let `p_i` be the YES execution price in cents, `y_i` its eventual binary YES
settlement, and `n_i` the number of executions in its contract-price cell. Define:

```
r_i = y_i - p_i/100
q_i = clip(p_i/100 + predicted_residual_i, 0, 1)
predicted_YES_error_pp = 100*q_i - p_i
observed_YES_error_pp  = 100*y_i - p_i
```

NO probabilities are `1-q_i`, outcomes `1-y_i`, and costs `100-p_i`.
Complementary NO deviations are the negatives of YES deviations for that execution.
This identity does not make the equal-cost YES/NO wedge zero: YES at cost c and NO
at cost c generally come from different executions with YES prices c and 100-c.

For these features and losses, aggregating repeated executions of one contract at
one price into a cell and retaining execution-count weights is lossless. Contracts
can appear at several prices, and all executions share the settlement outcome.
Inference therefore resamples blocks rather than treating executions as independent.

Example: a contract trades at 30 cents, and the historical estimator predicts a
+3 percentage-point correction. Its YES probability prediction is 0.33 and its
predicted deviation is +0.03. If it later settles YES, its observed deviation is
+0.70; if NO, -0.30. Individual deviations are noisy labels. Persistent mispricing
concerns conditional averages, not identifying every losing trade as overpriced.

## Historical estimates and support

For group g and price band b, training execution-weighted mean residual is:

```
W_gb = sum_i n_i
raw_delta_gb = sum_i n_i * (y_i - p_i/100) / W_gb
```

Within each cell, aggregate execution weights by training event e, `W_egb`, and
measure concentration-adjusted event support:

```
E_gb = W_gb^2 / sum_e W_egb^2
fraction_gb = E_gb / (E_gb + k)
shrunk_delta_gb = parent_delta_gb
                 + fraction_gb * (raw_delta_gb - parent_delta_gb)
```

`E` cannot exceed the number of contributing event IDs. Replicating executions in
the same event does not create new independent-event support. This effective-count
rule is a regularization heuristic, not an inferential sample-size guarantee or
Bayesian posterior. Event aliases and common outcomes can remain dependent.

The hierarchy is:

1. Pooled residual shrinks toward zero correction (raw market price).
2. Structural-family residual shrinks toward the pooled residual in that band.
3. Structural + NLP cluster residual shrinks toward its family's residual.

Missing group/price support inherits the parent exactly. An explicit `inherit`
configuration does the same at every price, even where local data exist. Final
probabilities are clipped; residual smoothing uses the underlying unclipped means.
All baseline estimates use all training cells; none uses test or validation labels.
Frozen upstream groups and the common eligible population are unchanged.

## Validation selection

The complete grid is defined in code before the new evaluation:

- Price-band widths: 1, 5, 10 cents. Width 5 means 1-5, 6-10, ..., 96-99.
- Smoothing k: 10, 50, 200, 1000 effective-event units, or `inherit`.
- Selection criterion: execution-weighted validation Brier, never test results.

Select the pooled width and k. Freeze that price grid for the other two scopes.
Select family k with its pooled parent fixed; select cluster k with its family
parent fixed. There is one k per scope, not one per subgroup. Ties use grid order.
Do not refit on train+validation. This yields 15 pooled plus 5 family plus 5
cluster candidates. Export the full grid, selected settings and all training
means/weights/effective event counts. Validation scores are selected scores and
must not be presented as independent confirmation.

This revision was motivated after earlier supervised results and this dataset's
outcomes were inspected. Label separation remains enforced, but the experiment is
retrospective and exploratory rather than a newly prespecified confirmatory test.

## Comparators and common population

Compare price, the three historical estimators, and all previously implemented
supervised models on identical held-out observations. The ML feature set, seed,
model settings, sample caps and selection grid are unchanged. Direct ML-versus-
historical contrasts are included at each scope, as are historical-versus-price,
structural-versus-pooled and NLP-versus-structural contrasts.

The inherited resolved/binary/$100-window-notional screens, MVE prefix exclusion,
explicit multi-award conjunction exclusion and cluster eligibility of 100 training
contracts/20 training events remain in force. These are retrospective selection
rules. Structural baselines use exactly the supported NLP-cluster cohort, so this
is a matched scope comparison, not an all-markets structural performance estimate.

## Individual and group-level evaluation

Execution-weighted Brier is the primary individual-prediction score. It equals
MSE for the predicted residual because subtracting the same execution price from
prediction and outcome cancels. Export residual RMSE `100*sqrt(Brier)` and residual
MAE `100*mean(abs(y-q))` in percentage points. MAE is descriptive because it is not
a proper binary probability scoring rule. Unit- and contract-balanced Brier are
sensitivities of evaluation weights, without refitting.

For every supported held-out NLP cluster, estimate observed and predicted YES and
NO calibration errors at the same exact purchase costs. Retain costs 1-49 with at
least 5 contracts and 5 events on each side, requiring at least 10 bilateral costs.
Average each direction's error equally over retained costs, then take YES minus NO
for the wedge. Training bands do not replace this exact-cost evaluation. All
models have the same held-out support. Support may differ between groups/splits.

Report MAE/RMSE of those cluster mean wedges, MAE of cluster mean YES and NO errors,
and descriptive wedge-sign agreement. Retain signed errors and exact-cost curves.
These group scores weight supported clusters equally and are distinct from Brier.
No significance claims about wedge accuracy are made without additional inference.

## Uncertainty and interpretation

Pair observation-level losses across models and bootstrap whole held-out events
(default 2,000 replicates). Also block by recurring series as a coarser sensitivity.
Intervals condition on fitted models, selected settings and held-out covariates.
Use block-sandwich normal p-values and Holm correction across all 25 aggregate
contrasts, separately within each split and block definition. Exact no-difference
comparisons have undefined inferential p-values and still occupy a conservative
multiplicity slot. Old comparison p-values can change as the comparison family grows.

The exploratory within-cluster table has ten contrasts per supported cluster,
with one Holm family per split across every cluster/contrast. Fewer than 30 held-out
events suppresses inference (p=1 for adjustment). No replication claim follows from
selected subgroup hits, especially with shared drivers or concentrated weights.

Positive evidence requires comparison against price and simple historical
baselines, not just a weak ML comparator. Distinguish structural improvement from
additional NLP value. Neither statistical significance nor probability corrections
establish trading profit. A later untouched sample with verified label availability
and outcome-group purging is required for prospective claims.
