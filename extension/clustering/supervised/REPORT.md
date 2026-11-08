# Historical mispricing transfer and supervised prediction in Kalshi

## Main results

Validation selected `inherit` at all three historical levels. Their held-out predictions therefore equal raw market prices exactly. This is a no-correction selection result within the evaluated grid, not proof that true calibration errors are zero. All pooled widths tie when inheriting price; width 1 is chosen by grid order and has no substantive meaning here.

**historical_family versus historical_pooled:** identical held-out probabilities after validation selection; Brier difference 0. No distinct correction was selected for this contrast, and a significance p-value is not applicable.

**historical_cluster versus historical_family:** identical held-out probabilities after validation selection; Brier difference 0. No distinct correction was selected for this contrast, and a significance p-value is not applicable.

**historical_cluster versus market_price:** identical held-out probabilities after validation selection; Brier difference 0. No distinct correction was selected for this contrast, and a significance p-value is not applicable.

Positive improvements favor the candidate; lower Brier scores are better. Improvements against a
weaker learned comparator do not by themselves show improvement over market prices. These are
retrospective results from an already explored extract, not a fresh confirmatory test.

| model | brier | residual_mae_pp | residual_rmse_pp | improvement_vs_price |
| --- | --- | --- | --- | --- |
| market_price | 0.171318 | 34.639380 | 41.390562 | 0.000000 |
| selected_pooled | 0.173611 | 34.694696 | 41.666659 | -0.002293 |
| selected_family | 0.171318 | 34.639380 | 41.390562 | 0.000000 |
| selected_cluster | 0.171301 | 34.669825 | 41.388544 | 0.000017 |
| historical_pooled | 0.171318 | 34.639380 | 41.390562 | 0.000000 |
| historical_family | 0.171318 | 34.639380 | 41.390562 | 0.000000 |
| historical_cluster | 0.171318 | 34.639380 | 41.390562 | 0.000000 |

## Question and relationship to Becker

Becker measures mispricing of a set of positions as realized win frequency minus mean purchase-
implied probability (section 3, equation 1), and compares YES and NO positions at equal costs
(sections 5.1 and 6.3). Our extension asks whether those historical calibration deviations transfer
to held-out comparable contracts, whether structural grouping helps, whether NLP refinement adds
value beyond structure, and whether supervised regressors outperform simple historical estimates.
The historical shrinkage estimators and supervised models are our additions, not estimators
specified by Becker. See [METHODOLOGY.md](METHODOLOGY.md) for equations and a prediction example.

The observation is an executed position, compressed losslessly for these models into a
contract-price cell with execution-count weight. For YES price p cents and eventual YES settlement
y, the observed deviation is y-p/100. Models predict conditional mean deviation d; predicted YES
probability is q=clip(p/100+d,0,1), predicted YES mispricing is 100q-p, and complementary NO
mispricing is its negative. One losing contract is not evidence of systematic overpricing.

**There is no fixed prediction horizon.** All valid observed-window executions are evaluated
against eventual settlement, as in the parent paper's execution-based measurement. A contract
can contribute at several prices/times; its outcome is shared, not independent across trades.
This tests generalization across held-out event IDs, not verified prediction of later unresolved
events. The 72-hour restriction in the draft abstract is removed. Choosing a fixed horizon would
not by itself establish that training labels were available at prediction time.

## Historical calibration baselines: new primary comparison

Estimate execution-weighted mean training residual within each price band and group. Pooled
estimates shrink toward zero residual (the market-price baseline); structural families shrink
toward the pooled estimate at the same band; NLP clusters shrink toward their structural-family
estimate. An unsupported group/price cell inherits its parent. No held-out outcomes enter these
training estimates. This explicitly transfers historical group calibration instead of assuming
that a flexible regressor will discover it from wording counts.

The local weight is E/(E+k), where E=(sum event execution weights)^2/sum squared event execution
weights. E is a concentration-adjusted count of contributing training events, not a count of
trades and not a proof of independent underlying outcomes. k is a regularization heuristic in
effective-event units, not a fitted Bayesian prior or a confidence interval. Repeated executions
within one event affect the execution-weighted mean but cannot inflate E beyond the event count.

Choose pooled band width from 1, 5 or 10 cents and k from 10, 50, 200, 1000 or `inherit` using
validation Brier only. `inherit` means exactly the parent prediction. Families and NLP clusters
use the **same selected price grid** and choose their own k on validation, sequentially with the
selected parent fixed. Pooled fallback is price; family fallback is the pool; cluster fallback
is the family. Group-specific k values are not tuned. All training cells enter every historical
estimate; there is no tree-style sampling cap. Models are not refitted on train+validation.
Final probabilities are clipped after the hierarchical residual correction.

Selected historical settings (`None`/`NA` means inherit the parent):

| scope | width | strength | validation_brier |
| --- | --- | --- | --- |
| pooled | 1 | None | 0.169457 |
| family | 1 | None | 0.169457 |
| cluster | 1 | None | 0.169457 |

## Data, matched population and eligibility

Only recent-year CSV inputs are used, via audited sufficient statistics. Actual market CSV split
membership is checked against frozen assignments; contract and event IDs must be disjoint.
Clustering representation/centroids were fitted on training metadata. Supervised/historical
estimators fit training labels only; validation chooses settings and test is used for evaluation.
The full historical Parquet dataset is not read.

Finalized binary YES/NO markets with at least $100 observed-window notional follow the inherited
screen. Notional is sum of traded unit counts because the complementary pair costs $1 per unit;
it is not lifetime volume. Final status and full-window activity are retrospective eligibility
conditions, not verified execution-time information. MVE-prefix combinations and explicit
multi-award conjunctions are excluded. The latter screen removed 562 traded contracts across
all splits after an earlier apparent Oscar subgroup hit exposed shared-ceremony outcomes under
different event IDs. That candidate was invalid and did not show NLP benefit. The screen is a
conservative wording heuristic, not an exhaustive semantic audit. See combination_exclusions.csv.

Require at least 100 training contracts and 20 training events per learned cluster:
**286 of 426** traded clusters pass. Every method uses exactly
the same resulting contracts, prices, outcomes, weights and holdouts. This isolates grouping
resolution; structural-only performance on all rule-eligible markets is a separate question.
Historical sparse price cells receive parent fallback; no additional evaluation rows are dropped.

| split | contracts | events | clusters | executions | cells |
| --- | --- | --- | --- | --- | --- |
| test | 45619 | 8545 | 286 | 8991005 | 859047 |
| validation | 22942 | 4369 | 286 | 4434490 | 429788 |

## Supervised comparators and prediction errors

Retain the earlier pooled, structural-family and NLP-cluster residual regressors: weighted ridge
(alpha=10), random forest (60 trees, depth 6, min leaf 20, bootstrap disabled), and histogram
boosting (80 iterations, 7 leaves, min leaf 30, L2=10, no early stopping). Features are price,
price squared/cubed, and title+YES-subtitle word, character, number, negation and question counts.
Final outcome/quotes/volume and close time are not predictors. NLP enters through frozen grouping,
not a supervised semantic encoder or validated sentiment feature. Training-only weighted scaling
is used; weights have mean one per fit. Ridge is uncapped; trees use at most 50,000 uniformly
sampled training cells with execution weights retained (seed 340). Sampling can affect results.

As before, validation selects an ML algorithm and correction strength in 0,.1,.25,.5,.75,1 toward
market price per scope, plus partial-pooling variants toward the selected pooled ML model.
ML settings and data screens were retained when adding the historical baselines. No new learner
or feature search was performed to obtain a positive result. Earlier outcomes were already
inspected, so this revision remains exploratory. See [CHANGES.md](CHANGES.md).

| scope | kind | strength | validation_brier |
| --- | --- | --- | --- |
| pooled | boosting | 0.500000 | 0.168521 |
| family | ridge | 0.000000 | 0.169457 |
| cluster | ridge | 0.100000 | 0.169453 |
| partial_family | ridge | 0.100000 | 0.168484 |
| partial_cluster | ridge | 0.100000 | 0.168461 |

Brier equals execution-weighted MSE of both settlement-probability predictions and residual
mispricing predictions. Residual RMSE in percentage points is 100*sqrt(Brier); residual MAE is
100*mean(abs(y-q)). Both refer to noisy individual realized deviations, not latent true mispricing.
MAE is descriptive: unlike Brier, minimizing binary-outcome MAE can favor extreme predictions.
Unit-weighted and equal-contract Brier (uniform across observed distinct prices within contract)
are sensitivities without retraining, not independent replications.

All test models:

| model | brier | residual_mae_pp | residual_rmse_pp | unit_brier | contract_balanced_brier | improvement_vs_price |
| --- | --- | --- | --- | --- | --- | --- |
| market_price | 0.171318 | 34.639380 | 41.390562 | 0.162020 | 0.114059 | 0.000000 |
| pooled_ridge | 0.171412 | 34.867578 | 41.401961 | 0.162192 | 0.114352 | -0.000094 |
| pooled_forest | 0.188719 | 35.462189 | 43.441750 | 0.179298 | 0.119426 | -0.017401 |
| pooled_boosting | 0.178456 | 34.750012 | 42.244015 | 0.170490 | 0.114548 | -0.007138 |
| family_ridge | 0.172936 | 34.995043 | 41.585609 | 0.163469 | 0.114605 | -0.001618 |
| family_forest | 0.200698 | 35.753619 | 44.799290 | 0.198699 | 0.119935 | -0.029380 |
| family_boosting | 0.184317 | 35.138993 | 42.932178 | 0.178005 | 0.117600 | -0.012999 |
| cluster_ridge | 0.172937 | 34.943827 | 41.585748 | 0.163333 | 0.115447 | -0.001620 |
| cluster_forest | 0.200536 | 35.410393 | 44.781248 | 0.197272 | 0.121587 | -0.029218 |
| cluster_boosting | 0.188975 | 35.461192 | 43.471239 | 0.183068 | 0.119265 | -0.017657 |
| selected_pooled | 0.173611 | 34.694696 | 41.666659 | 0.164810 | 0.114036 | -0.002293 |
| selected_family | 0.171318 | 34.639380 | 41.390562 | 0.162020 | 0.114059 | 0.000000 |
| selected_cluster | 0.171301 | 34.669825 | 41.388544 | 0.161992 | 0.113984 | 0.000017 |
| partial_family | 0.173251 | 34.724731 | 41.623376 | 0.164360 | 0.113960 | -0.001933 |
| partial_cluster | 0.173232 | 34.719609 | 41.621206 | 0.164349 | 0.113947 | -0.001915 |
| historical_pooled | 0.171318 | 34.639380 | 41.390562 | 0.162020 | 0.114059 | 0.000000 |
| historical_family | 0.171318 | 34.639380 | 41.390562 | 0.162020 | 0.114059 | 0.000000 |
| historical_cluster | 0.171318 | 34.639380 | 41.390562 | 0.162020 | 0.114059 | 0.000000 |

Validation (used for selection, not independent inferential evidence):

| model | brier | residual_mae_pp | residual_rmse_pp | improvement_vs_price |
| --- | --- | --- | --- | --- |
| market_price | 0.169457 | 34.659577 | 41.165112 | 0.000000 |
| pooled_ridge | 0.169891 | 34.920742 | 41.217883 | -0.000435 |
| pooled_forest | 0.185281 | 35.430979 | 43.044230 | -0.015824 |
| pooled_boosting | 0.170360 | 34.043588 | 41.274681 | -0.000903 |
| family_ridge | 0.172077 | 35.057385 | 41.482108 | -0.002620 |
| family_forest | 0.192261 | 34.960274 | 43.847625 | -0.022805 |
| family_boosting | 0.184499 | 34.924492 | 42.953357 | -0.015042 |
| cluster_ridge | 0.172402 | 34.982703 | 41.521353 | -0.002946 |
| cluster_forest | 0.193359 | 34.730896 | 43.972619 | -0.023902 |
| cluster_boosting | 0.185455 | 34.867161 | 43.064502 | -0.015998 |
| selected_pooled | 0.168521 | 34.351583 | 41.051321 | 0.000936 |
| selected_family | 0.169457 | 34.659577 | 41.165112 | 0.000000 |
| selected_cluster | 0.169453 | 34.691890 | 41.164641 | 0.000004 |
| partial_family | 0.168484 | 34.422163 | 41.046831 | 0.000972 |
| partial_cluster | 0.168461 | 34.414695 | 41.044004 | 0.000996 |
| historical_pooled | 0.169457 | 34.659577 | 41.165112 | 0.000000 |
| historical_family | 0.169457 | 34.659577 | 41.165112 | 0.000000 |
| historical_cluster | 0.169457 | 34.659577 | 41.165112 | 0.000000 |

![Validation and test improvements](results/prediction_performance.png)

## Paired uncertainty and multiplicity

Whole held-out events are bootstrapped, pairing identical observations across models, with 2,000
resamples in the default run. Percentile intervals hold models/settings/support fixed. P-values
use weighted block-sandwich standard errors and a two-sided normal approximation. Recurring-series
blocks (ticker prefixes) provide a coarser dependence sensitivity. Neither event IDs nor series
are authoritative independent-outcome groups. Training, clustering and selection uncertainty are
not included. Price support and composition are conditioned upon, and blocks can be concentrated.
Holm correction covers **all 25 aggregate contrasts** separately by split and block
unit, including the legacy contrasts. The complete comparisons.csv contains every contrast.
A nominal interval excluding zero is not enough to claim multiplicity-adjusted significance.

Test event-block historical and ML-versus-historical comparisons:

| candidate | baseline | improvement | ci_low | ci_high | p_value | holm_p |
| --- | --- | --- | --- | --- | --- | --- |
| historical_pooled | market_price | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| historical_family | historical_pooled | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| historical_cluster | historical_family | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| historical_cluster | historical_pooled | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| historical_family | market_price | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| historical_cluster | market_price | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| selected_pooled | historical_pooled | -0.002293 | -0.003477 | -0.001077 | 0.000208 | 0.005195 |
| selected_family | historical_family | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| selected_cluster | historical_cluster | 0.000017 | -0.000131 | 0.000157 | 0.825803 | 1.000000 |
| pooled_ridge | historical_pooled | -0.000094 | -0.000474 | 0.000297 | 0.646680 | 1.000000 |
| family_ridge | historical_family | -0.001618 | -0.003087 | -0.000217 | 0.037139 | 0.579114 |
| cluster_ridge | historical_cluster | -0.001620 | -0.003118 | -0.000176 | 0.036195 | 0.579114 |

Test series-block sensitivity for the same comparisons:

| candidate | baseline | improvement | ci_low | ci_high | p_value | holm_p | blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_pooled | market_price | 0.000000 | 0.000000 | 0.000000 | NA | NA | 99 |
| historical_family | historical_pooled | 0.000000 | 0.000000 | 0.000000 | NA | NA | 99 |
| historical_cluster | historical_family | 0.000000 | 0.000000 | 0.000000 | NA | NA | 99 |
| historical_cluster | historical_pooled | 0.000000 | 0.000000 | 0.000000 | NA | NA | 99 |
| historical_family | market_price | 0.000000 | 0.000000 | 0.000000 | NA | NA | 99 |
| historical_cluster | market_price | 0.000000 | 0.000000 | 0.000000 | NA | NA | 99 |
| selected_pooled | historical_pooled | -0.002293 | -0.003938 | -0.000618 | 0.006229 | 0.149503 | 99 |
| selected_family | historical_family | 0.000000 | 0.000000 | 0.000000 | NA | NA | 99 |
| selected_cluster | historical_cluster | 0.000017 | -0.000183 | 0.000192 | 0.852319 | 1.000000 | 99 |
| pooled_ridge | historical_pooled | -0.000094 | -0.000482 | 0.000401 | 0.693854 | 1.000000 | 99 |
| family_ridge | historical_family | -0.001618 | -0.003929 | 0.000031 | 0.080194 | 1.000000 | 99 |
| cluster_ridge | historical_cluster | -0.001620 | -0.003872 | 0.000148 | 0.085399 | 1.000000 | 99 |

## Becker-style direction-specific errors and equal-cost wedge

For each held-out NLP cluster, compare YES at cost c with NO at **the same cost c**, coming from
YES executions at 100-c. Compute execution-weighted observed and predicted calibration errors
separately by side at each exact cost. Then average equally across bilateral costs in 1-49 cents.
Require 5 contracts and 5 events per side/cost and at least 10 bilateral costs per cluster/split.
This exact-cost evaluation is shared by all methods even if training uses a wider price band.
It is an available-support comparison, not identical support across clusters or splits.

The wedge is YES error minus NO error. A negative wedge does not alone establish absolute YES
overpricing. Export both signed direction-specific means and exact-cost curves. The following
scores give equal weight to supported clusters: MAE/RMSE of the mean wedge, and MAE of each
cluster's mean YES and NO errors over those same costs. These are different estimands from
execution-weighted Brier. Sign agreement is descriptive and includes uncertain near-zero wedges;
zero predicted wedge has undefined sign agreement. No wedge-accuracy intervals or individual
significant wedge discoveries are claimed.

| model | clusters | wedge_mae_pp | sign_agreement | wedge_rmse_pp | yes_mae_pp | no_mae_pp |
| --- | --- | --- | --- | --- | --- | --- |
| cluster_boosting | 243 | 13.708405 | 0.600823 | 20.336795 | 7.244726 | 8.670605 |
| cluster_forest | 243 | 13.789522 | 0.617284 | 20.470905 | 7.241355 | 8.763055 |
| cluster_ridge | 243 | 13.252984 | 0.609053 | 19.840217 | 7.002042 | 8.375259 |
| family_boosting | 243 | 13.186735 | 0.572016 | 19.692143 | 6.707086 | 8.603245 |
| family_forest | 243 | 13.230437 | 0.567901 | 19.870961 | 6.802964 | 8.790808 |
| family_ridge | 243 | 12.608575 | 0.559671 | 19.004921 | 6.525948 | 8.250038 |
| historical_cluster | 243 | 12.487291 | NA | 17.533683 | 6.357502 | 8.175807 |
| historical_family | 243 | 12.487291 | NA | 17.533683 | 6.357502 | 8.175807 |
| historical_pooled | 243 | 12.487291 | NA | 17.533683 | 6.357502 | 8.175807 |
| market_price | 243 | 12.487291 | NA | 17.533683 | 6.357502 | 8.175807 |
| partial_cluster | 243 | 12.441726 | 0.510288 | 17.666688 | 6.341951 | 8.190832 |
| partial_family | 243 | 12.513899 | 0.456790 | 17.715698 | 6.349979 | 8.229472 |
| pooled_boosting | 243 | 12.764717 | 0.456790 | 17.899954 | 6.413598 | 8.410462 |
| pooled_forest | 243 | 12.369617 | 0.576132 | 16.788296 | 6.738971 | 8.222065 |
| pooled_ridge | 243 | 12.325228 | 0.580247 | 17.226458 | 6.358111 | 8.057643 |
| selected_cluster | 243 | 12.377303 | 0.609053 | 17.553061 | 6.341080 | 8.104603 |
| selected_family | 243 | 12.487291 | NA | 17.533683 | 6.357502 | 8.175807 |
| selected_pooled | 243 | 12.567658 | 0.456790 | 17.664218 | 6.352039 | 8.260426 |

## Complete exploratory within-cluster screen

All supported NLP clusters are screened with normal event-sandwich comparisons. At least 30
held-out events are required for inference; other cells get p=1 in multiplicity correction.
Holm covers all 10 contrasts times
all 286 clusters per split. This screen was introduced after earlier
aggregate exploration. Any candidate remains exploratory and needs independent replication.
The exported table includes event concentration; separate event IDs can share settlement drivers.
A subgroup improvement over a weak ML baseline is not necessarily improvement over market price.

| candidate | baseline | clusters | eligible | holm_positive | holm_negative |
| --- | --- | --- | --- | --- | --- |
| cluster_ridge | market_price | 286 | 206 | 0 | 0 |
| cluster_ridge | pooled_ridge | 286 | 206 | 0 | 0 |
| historical_cluster | historical_family | 286 | 0 | 0 | 0 |
| historical_cluster | historical_pooled | 286 | 0 | 0 | 0 |
| historical_cluster | market_price | 286 | 0 | 0 | 0 |
| historical_family | historical_pooled | 286 | 0 | 0 | 0 |
| selected_cluster | historical_cluster | 286 | 206 | 0 | 0 |
| selected_cluster | market_price | 286 | 206 | 0 | 0 |
| selected_cluster | selected_family | 286 | 206 | 0 | 0 |
| selected_cluster | selected_pooled | 286 | 206 | 0 | 0 |

First twelve corrected hits (full successes and failures are exported):

No rows meet these criteria.

## Interpretation limits and reproduction

This study tests historical calibration transfer and predictive scope under the supplied
retrospective event-hash splits. It does not establish chronological label availability, a fixed-
horizon forecast, future profitability, causal behavioral bias, or an exhaustive test of ML/NLP.
Recurring series and related outcomes cross splits. Wording is unversioned, settlement-publication
timestamps are unavailable, and this extract has been explored before. Further positive results
must be presented with those limitations; null results do not prove mispricing is unpredictable.
No fees, spread, fills or executable quote model is included. Probability points are not returns.
A later unexamined sample with verified label availability and purged outcome groups would be
needed for confirmatory future-forecasting claims. No chronological redesign is introduced here.

Run `.venv/bin/python -m extension.clustering.supervised.pipeline` from the repository root.
See [README.md](README.md), [METHODOLOGY.md](METHODOLOGY.md), and [CHANGES.md](CHANGES.md).
The manifest must say completed=true. It records metadata fingerprints (not full raw-file hashes),
source hashes, package versions, seed, eligibility and selected configurations. Generated results
are ignored under results/. Prior session outputs are retained under results/previous_supervised/.

Key outputs: historical_training_estimates.csv (training residuals, event support and price bands),
historical_validation_search.csv and historical_selected.json (full search/selection), metrics.csv,
comparisons.csv, heldout_predictions.parquet, contract_predictions.csv.gz, matched_cost_curves.csv.gz,
cluster_wedges.csv, wedge_metrics.csv, cluster_metrics.csv and cluster_comparisons.csv. Other tables
record eligibility, fit sizes and coefficients. The compressed Parquet is an output, not an input
to the historical dataset extraction.

Source: Becker, *The Microstructure of Wealth Transfer in Prediction Markets*, local repository
PDF, section 3 equation (1), sections 5.1 and 6.3. All shrinkage and supervised prediction choices
above are this team's extension.
