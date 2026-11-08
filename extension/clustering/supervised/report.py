"""Generate a research report from the completed historical-transfer comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def table(frame: pd.DataFrame, digits: int = 6) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "NA"
        return str(value)

    if frame.empty:
        return "No rows meet these criteria."
    return "\n".join(
        ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    )


def write_report(output: Path, target: Path) -> None:
    metrics = pd.read_csv(output / "metrics.csv")
    comparisons = pd.read_csv(output / "comparisons.csv")
    coverage = pd.read_csv(output / "holdout_coverage.csv")
    support = pd.read_csv(output / "training_support.csv")
    wedges = pd.read_csv(output / "wedge_metrics.csv")
    selection = json.loads((output / "historical_selected.json").read_text())
    ml_selection = json.loads((output / "selected_models.json").read_text())
    event = comparisons[comparisons.split.eq("test") & comparisons.dependence_unit.eq("event_id")]
    series = comparisons[comparisons.split.eq("test") & comparisons.dependence_unit.eq("series")]
    historical = event.candidate.str.startswith("historical_") | event.baseline.str.startswith("historical_")
    cols = ["candidate", "baseline", "improvement", "ci_low", "ci_high", "p_value", "holm_p"]

    def conclusion(candidate, baseline):
        row = event[event.candidate.eq(candidate) & event.baseline.eq(baseline)].iloc[0]
        coarse = series[series.candidate.eq(candidate) & series.baseline.eq(baseline)].iloc[0]
        if row.se == 0 and row.improvement == 0:
            return (
                f"**{candidate} versus {baseline}:** identical held-out probabilities after "
                "validation selection; Brier difference 0. No distinct correction was selected "
                "for this contrast, and a significance p-value is not applicable."
            )
        if row.improvement > 0 and row.holm_p < 0.05 and coarse.holm_p < 0.05 and min(row.ci_low, coarse.ci_low) > 0:
            interpretation = "positive evidence survives both reported dependence checks"
        elif row.improvement > 0 and row.holm_p < 0.05 and row.ci_low > 0:
            interpretation = "positive event-block evidence does not survive the full series-block checks"
        else:
            interpretation = "no statistically supported improvement under the reported correction"
        return (
            f"**{candidate} versus {baseline}:** Brier improvement {row.improvement:.8f}, "
            f"event-bootstrap 95% interval [{row.ci_low:.8f}, {row.ci_high:.8f}], "
            f"event Holm p={row.holm_p:.4f}; {interpretation}."
        )

    fallback_statement = (
        "Validation selected `inherit` at all three historical levels. Their held-out predictions "
        "therefore equal raw market prices exactly. This is a no-correction selection result within "
        "the evaluated grid, not proof that true calibration errors are zero. All pooled widths tie "
        "when inheriting price; width 1 is chosen by grid order and has no substantive meaning here."
        if all(row["strength"] is None for row in selection.values())
        else "Inspect selected smoothing and parent fallbacks below to distinguish transferred local "
        "signals from inherited predictions."
    )
    names = [
        "market_price",
        "historical_pooled",
        "historical_family",
        "historical_cluster",
        "selected_pooled",
        "selected_family",
        "selected_cluster",
    ]
    labels = [
        "Market price",
        "Historical pooled",
        "Historical structural",
        "Historical structural + NLP",
        "Selected ML pooled",
        "Selected ML structural",
        "Selected ML structural + NLP",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained", sharex=True)
    for ax, split in zip(axes, ["validation", "test"]):
        view = metrics[metrics.split.eq(split)].set_index("model").loc[names]
        ax.barh(labels, view.improvement_vs_price, color=["#64748b"] + ["#0d9488"] * 3 + ["#2563eb"] * 3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(split.capitalize())
        ax.set_xlabel("Brier improvement over market price (positive is better)")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "prediction_performance.png", dpi=180)
    plt.close(fig)
    relative_plot = os.path.relpath(output / "prediction_performance.png", target.parent)
    local = pd.read_csv(output / "cluster_comparisons.csv")
    local_test = local[local.split.eq("test")]
    local_summary = (
        local_test.groupby(["candidate", "baseline"])
        .apply(
            lambda g: pd.Series(
                {
                    "clusters": len(g),
                    "eligible": int(g.inference_eligible.sum()),
                    "holm_positive": int((g.holm_p.lt(0.05) & g.improvement.gt(0)).sum()),
                    "holm_negative": int((g.holm_p.lt(0.05) & g.improvement.lt(0)).sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    local_hits = local_test[local_test.holm_p.lt(0.05)].sort_values("holm_p")
    short_metrics = metrics[metrics.split.eq("test") & metrics.model.isin(names)]
    score_cols = ["model", "brier", "residual_mae_pp", "residual_rmse_pp", "improvement_vs_price"]
    report = f"""# Historical mispricing transfer and supervised prediction in Kalshi

## Main results

{fallback_statement}

{conclusion("historical_family", "historical_pooled")}

{conclusion("historical_cluster", "historical_family")}

{conclusion("historical_cluster", "market_price")}

Positive improvements favor the candidate; lower Brier scores are better. Improvements against a
weaker learned comparator do not by themselves show improvement over market prices. These are
retrospective results from an already explored extract, not a fresh confirmatory test.

{table(short_metrics[score_cols])}

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

{table(pd.DataFrame(selection.values()))}

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
**{int(support.eligible.sum())} of {len(support)}** traded clusters pass. Every method uses exactly
the same resulting contracts, prices, outcomes, weights and holdouts. This isolates grouping
resolution; structural-only performance on all rule-eligible markets is a separate question.
Historical sparse price cells receive parent fallback; no additional evaluation rows are dropped.

{table(coverage, 0)}

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

{table(pd.DataFrame(ml_selection.values()))}

Brier equals execution-weighted MSE of both settlement-probability predictions and residual
mispricing predictions. Residual RMSE in percentage points is 100*sqrt(Brier); residual MAE is
100*mean(abs(y-q)). Both refer to noisy individual realized deviations, not latent true mispricing.
MAE is descriptive: unlike Brier, minimizing binary-outcome MAE can favor extreme predictions.
Unit-weighted and equal-contract Brier (uniform across observed distinct prices within contract)
are sensitivities without retraining, not independent replications.

All test models:

{table(metrics[metrics.split.eq("test")].drop(columns="split"))}

Validation (used for selection, not independent inferential evidence):

{table(metrics[metrics.split.eq("validation")][score_cols])}

![Validation and test improvements]({relative_plot})

## Paired uncertainty and multiplicity

Whole held-out events are bootstrapped, pairing identical observations across models, with 2,000
resamples in the default run. Percentile intervals hold models/settings/support fixed. P-values
use weighted block-sandwich standard errors and a two-sided normal approximation. Recurring-series
blocks (ticker prefixes) provide a coarser dependence sensitivity. Neither event IDs nor series
are authoritative independent-outcome groups. Training, clustering and selection uncertainty are
not included. Price support and composition are conditioned upon, and blocks can be concentrated.
Holm correction covers **all {len(event)} aggregate contrasts** separately by split and block
unit, including the legacy contrasts. The complete comparisons.csv contains every contrast.
A nominal interval excluding zero is not enough to claim multiplicity-adjusted significance.

Test event-block historical and ML-versus-historical comparisons:

{table(event.loc[historical, cols])}

Test series-block sensitivity for the same comparisons:

{table(series[series.candidate.str.startswith("historical_") | series.baseline.str.startswith("historical_")][cols + ["blocks"]])}

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

{table(wedges[wedges.split.eq("test")].drop(columns="split"))}

## Complete exploratory within-cluster screen

All supported NLP clusters are screened with normal event-sandwich comparisons. At least 30
held-out events are required for inference; other cells get p=1 in multiplicity correction.
Holm covers all {local_test[["candidate", "baseline"]].drop_duplicates().shape[0]} contrasts times
all {local_test.cluster.nunique()} clusters per split. This screen was introduced after earlier
aggregate exploration. Any candidate remains exploratory and needs independent replication.
The exported table includes event concentration; separate event IDs can share settlement drivers.
A subgroup improvement over a weak ML baseline is not necessarily improvement over market price.

{table(local_summary, 0)}

First twelve corrected hits (full successes and failures are exported):

{table(local_hits[["cluster", "candidate", "baseline", "events", "improvement", "holm_p"]].head(12))}

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
"""
    target.write_text(report)
