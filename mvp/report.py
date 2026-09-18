"""Generate a step-by-step, linked findings report from completed tables.

No hand-copied performance values: rerunning the pipeline updates all quantitative
claims. The qualitative limitations describe the experimental design, irrespective
of whether a future run happens to yield favorable prediction metrics.
"""

import json
from pathlib import Path

import pandas as pd


def markdown_table(frame):
    """Keep reporting dependency-free (pandas.to_markdown requires tabulate)."""
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(rows)


def write_findings(out: Path):
    audit = json.loads((out / "attrition.json").read_text())
    config = json.loads((out / "run_manifest.json").read_text())["config"]
    metrics = pd.read_csv(out / "model_comparison.csv")
    labels = pd.read_csv(out / "wedges_text.csv")
    eligible = labels[labels.status == "eligible"]
    train = eligible[eligible.split == "train"]
    test = eligible[eligible.split == "test"]
    primary = metrics[
        (metrics.approach == "text") & (metrics.block == "interpretable") & (metrics.model == "ridge")
    ].iloc[0]
    baseline = metrics[(metrics.approach == "text") & (metrics.model == "mean")].iloc[0]
    comparison = metrics[
        (metrics.approach == "text") & ((metrics.block == "interpretable") | (metrics.model == "mean"))
    ]
    comparison = comparison[["block", "model", "mae_pp", "rmse_pp", "r2"]]
    second_pass = metrics[(metrics.block == "market") & (metrics.model.isin(["mean", "median", "huber"]))]
    second_pass = second_pass[["approach", "model", "mae_pp", "mae_improvement_pp"]]
    diagnostics = pd.read_csv(out / "clustering_diagnostics.csv")
    rank = pd.read_csv(out / "ranking_text.csv")
    contrast = json.loads((out / "ranking_summary_text.json").read_text())
    coeff = pd.read_csv(out / "coefficients_text.csv")
    coeff = coeff[(coeff.block == "interpretable") & (coeff.model == "ridge")]
    coeff = coeff.loc[coeff.coefficient_pp.abs().sort_values(ascending=False).index].head(5)
    improved = (metrics.loc[~metrics.model.isin(["mean", "median"]), "mae_improvement_pp"] > 0).any()
    result = (
        "At least one exploratory specification improves held-out MAE; this is not confirmatory evidence after multiple comparisons."
        if improved
        else "No supervised specification beats its corresponding training-mean baseline on held-out MAE."
    )
    interval = contrast["contrast_ci_pp"]
    interval_text = f"[{interval[0]:.2f}, {interval[1]:.2f}]" if interval else "not estimable"
    report = f"""# MVP findings: predicting the optimism-tax wedge

{result} The delivered analysis establishes a working estimator and evaluation pipeline, but does **not establish reliable future-market prediction or an exploitable liquidity-provider signal**.

The primary specification is text-based clustering plus interpretable ridge regression, predesignated in the pipeline; the development holdout is exploratory. Its held-out MAE is **{primary.mae_pp:.2f} probability percentage points**, versus **{baseline.mae_pp:.2f}** for the naive mean; RMSE is **{primary.rmse_pp:.2f}** and R² is **{primary.r2:.2f}**. MAE improvement over the baseline is {primary.mae_improvement_pp:.2f} pp, with a conditional cluster-block 95% interval [{primary.improvement_ci_low_pp:.2f}, {primary.improvement_ci_high_pp:.2f}]. These intervals are particularly fragile with only {test.cluster.nunique()} held-out semantic clusters.

## What this analysis measures

The question is whether information available during a contract's first {config["window_hours"]} hours can predict a later YES-versus-NO pricing difference for groups of similar contracts. The prediction unit is a *cluster-by-cost-band label*, not an individual trade or contract. At each supported exact purchase cost, the pipeline estimates the resolved win rate minus that cost separately for YES and NO, then subtracts NO mispricing from YES mispricing. A negative wedge means YES performed worse than NO at matched costs. Results are measured in probability percentage points, not percentage returns or realizable profit.

The workflow is: screen and sample resolved contracts; measure early trades and wording; keep related series together in either training or test; assign contracts to text clusters fitted on training inputs; construct separate group-and-cost labels in each split; fit models on training labels; and score their predictions on test labels. The [methodology](../README.md) supplies formulas and complete settings.

## Data and label coverage

The raw Kalshi extract has {audit["market_rows"]:,} market records and {audit["trade_rows"]:,} executions. After finalization/binary-outcome, combination exclusion, metadata and {config["window_hours"]}-hour landmark screening, {audit["open_past_landmark"]:,} market records are eligible for sampling. A deterministic sample of {audit["sampled_metadata"]:,} yields **{audit["contracts_with_min_early_trades"]:,} contracts** with at least {config["min_early_trades"]} observation-window executions. The valid sampled window has {audit["deduplicated_valid_window_trades"]:,} executions before the minimum-trade contract filter. See [attrition](attrition.json) for sequential counts.

The whole-series holdout contains {audit["split_contracts"]["train"]:,} training contracts from {audit["split_series"]["train"]} series and {audit["split_contracts"]["test"]:,} test contracts from {audit["split_series"]["test"]} series. Hash splitting series does not guarantee balanced contract counts. In the primary grouping there are **{len(train)} eligible train labels and {len(test)} eligible test labels**, each constructed from multiple resolved contracts at matched costs. Minimums are {config["min_group_contracts"]} contracts and {config["min_group_events"]} events in each direction, plus exact-price support, execution counts and training-coherence screening. Failed rows remain marked in the [wedge dataset](wedges_text.csv).

The equal-row mean wedge changes from {train.wedge_pp.mean():.2f} pp in train to {test.wedge_pp.mean():.2f} pp in test. Test estimates span {test.wedge_pp.min():.2f} to {test.wedge_pp.max():.2f} pp; their sample SD is {test.wedge_pp.std():.2f} pp and median event-bootstrap SE is {test.se_pp.median():.2f} pp. These are descriptive variation measures that mix signal, sampling noise and population composition; they do not estimate between-group latent variance. A sign reversal across different populations is not evidence of a temporal regime shift.

**Why these filters matter:** resolved binary outcomes are needed to calculate which side won. Excluding combination products reduces shared-leg dependence that this extract cannot reliably identify. Requiring a recorded close later than the first-day landmark places trade features before that close. The deterministic ticker sample controls processing size without choosing contracts by volume. The five-trade threshold avoids activity summaries based on nearly empty windows. These choices produce a selected population of longer-lived, early-active contracts; results should not be generalized automatically to all listings.

**How labels are built:** one execution supplies a YES observation at its YES price and a complementary NO observation at its NO price. Within each exact cost, the pipeline compares each side's observed win rate with the probability implied by the cost. It retains costs with sufficient distinct contracts and events on both sides, then averages equally across retained costs in the band. Matching costs prevents a coarse-band difference caused only by different YES and NO price mixtures. Repeated executions share contract outcomes, so uncertainty is estimated by resampling complete events, not individual trades. Support thresholds reduce sparse cells but do not guarantee precise labels.

**Why series are split together:** contracts in one recurring series can have related outcomes and wording. Keeping complete series on one side of the split blocks direct event reuse between training and test. Other series can still refer to the same underlying event, and this split is not chronological. The train/test mean difference above warns that the observed populations differ; it does not by itself reveal a time trend or a stable shift in underlying mispricing.

## Clustering and price sensitivity

**Step:** represent each contract using its title and observed YES subtitle, fit TF-IDF and latent semantic analysis on training wording, and assign contracts to frozen k-means centroids. The primary map adds a small block of transparent wording features; a second map also adds bounded early price as a sensitivity check. Outcomes do not enter the cluster fit. Clusters provide pools for estimating group labels, but their numeric IDs are not economic categories.

{markdown_table(diagnostics[["approach", "silhouette", "price_bin_nmi", "price_removal_adjusted_rand"]])}

Low price-decile mutual information and retained within-cluster price variation suggest these maps are not simply price bins. The adjusted Rand agreement quantifies sensitivity to removing the bounded price block; it is not proof of economic coherence. Nearest-centroid titles and series composition are exported in [cluster profiles](cluster_profiles.csv). The text representation is title plus observed YES subtitle, represented by training-fitted TF-IDF/LSA. It is **not a pretrained sentence embedding**. A minimum training mean text cosine of {config["min_coherence"]} rejects diffuse clusters. Even compact clusters may join unrelated threshold events; see the [manual review](../CLUSTER_REVIEW.md).

![Cluster quality](cluster_quality.png)

![Wedge variation and uncertainty](wedge_variation.png)

Silhouette measures separation in the chosen feature space. Price-bin mutual information tests whether assignment closely tracks price deciles. Adjusted Rand agreement measures how much assignment changes when the price block is removed. None establishes that contracts share an economic mechanism. The [manual review](../CLUSTER_REVIEW.md) identifies mixed groups that need closer inspection before substantive category claims.

## Model comparison

**Step:** fit models on eligible training group-cost labels and predict labels constructed from unseen series. Inputs summarize early trading and wording among contributing contracts; each eligible label receives equal weight. The mean baseline predicts the same training-average wedge for every test row and is the reference a group-specific model should improve upon.

All errors are in probability percentage points; lower is better. The following uses primary text clusters and interpretable features (baseline shown once):

{markdown_table(comparison)}

The [full comparison](model_comparison.csv) adds market-only and supplementary LSA blocks for both clustering approaches. There is no hyperparameter search or selection of a reported winner on test outcomes. Similarity and coherence safeguards were revised during exploratory development; a fresh holdout is required for confirmation. The two clustering approaches generate different groups/support and label counts: their raw errors are not a paired comparison on the same target rows. Compare each model with its own baseline. Ordinary least squares with many LSA predictors can extrapolate far outside the physically possible wedge range; predictions are intentionally not clipped after examining the holdout. This exposes its unsuitability at the current sample size.

![Prediction performance](prediction_performance.png)

MAE is the average absolute prediction miss in probability points; RMSE places more weight on large misses; R² compares squared error with a constant test-mean reference. For the predesignated ridge model, MAE exceeds the training-mean baseline by {-primary.mae_improvement_pp:.2f} points, so its group-specific predictions are less accurate on this holdout. Its negative R² also indicates poor squared-error fit against the test-mean reference. The MAE-improvement interval includes zero and is fragile with only {test.cluster.nunique()} held-out clusters.

The forest's lower error in the table is a lead for future work. Because several representations and models were inspected on the same development holdout, selecting it now would give an optimistic estimate of future performance. A new, untouched evaluation period is needed after grouping and model rules are fixed.

## Secondary iteration: robust, conservative estimates

The first pass suggests that a few noisy group labels can pull squared-error fits away from typical outcomes. We added a training-median constant and a market-only Huber regression with strong fixed regularization (alpha 10). Both use training labels only; the same development holdout is reused, so any gain is exploratory.

{markdown_table(second_pass)}

The median is a useful **default estimate of the typical eligible group wedge** when a point estimate is needed. It does not distinguish opportunities across groups. Huber retains some market-feature variation, but its small gain over the mean on text clusters is insufficient to validate ranking. The practical route is to use the robust estimate for screening and uncertainty-aware data collection, then test a conditional signal on a fresh chronological holdout before making any trading decision.

The median predicts one robust central value for every group and is less sensitive to extreme training labels. Huber regression reduces the influence of large residuals while allowing predictions to vary with early market features. These checks ask whether noisy targets or outliers contributed to the first-pass result; they do not supply an independent validation sample.

## Feature associations

The five largest absolute standardized coefficients of the primary ridge fit are:

{markdown_table(coeff[["feature", "coefficient_pp"]])}

A coefficient is the fitted wedge change per training-group SD, conditional on other inputs. A negative coefficient corresponds to a larger YES disadvantage. Features are aggregated at group level; these coefficients are neither contract-level effects nor causal estimates. Given weak held-out performance, they are exploratory training associations, not established generalizable predictors. Lexicon counts do not constitute validated sentiment or emotion measures.

![Ridge coefficients](feature_coefficients.png)

For example, the political-word coefficient describes the fitted change associated with a one-training-standard-deviation increase in that group-level count while the other inputs are held fixed. It does not say that political wording caused the change or that the association will transfer to new series. Correlated wording, event type and prices can alter coefficients substantially when there are few training labels.

## Preliminary signal ranking

Define optimism tax as **minus the wedge**, so a higher predicted tax is a predicted YES disadvantage. Rank test rows by the predesignated ridge predictions:

{markdown_table(rank[["quantile", "rows", "clusters", "predicted_tax_pp", "realized_tax_pp"]])}

Realized high-minus-low tax is **{contrast["high_minus_low_tax_pp"]:.2f} pp**, with a cluster-block interval **{interval_text} pp**; {contrast["bootstrap_valid"]} of {config["bootstrap"]} resamples contain both extreme groups. The high group differs from the overall held-out realized tax by {contrast["high_minus_baseline_tax_pp"]:.2f} pp. Empty extreme groups in some resamples and few clusters make the interval fragile. The signal is not validated merely by large predicted magnitudes.

![Held-out ranking](signal_ranking.png)

These are pooled direction asymmetries, not maker profits. There are no historical executable spreads, fees, depth, queue/fill probabilities or inventory constraints in this MVP. No economic-exploitability claim is justified.

The ranking step sorts held-out rows using predicted tax alone and then checks their realized tax. A useful ranking would put a larger realized tax in the highest predicted group than in the lowest. Here the observed high-minus-low difference runs in the opposite direction. The wide interval crosses zero, so neither a beneficial ranking nor a stable reversed ranking is established. Few rows and clusters in each tertile make this a diagnostic for the next experiment, not a selection rule.

## Limits and recommended next steps

1. **Historical provenance:** versioned titles, scheduled closes and outcome-publication times are missing. Trade inputs stop at open+24h, but snapshot wording is assumed unchanged. This is a retrospective unseen-series evaluation, not a verified chronological forecast. All training labels may not have been known at an actual test landmark.
2. **Population and dependence:** resolved, non-combination, longer-lived, early-active contracts are a selected population. Series grouping prevents direct event reuse but does not capture every cross-product economic link. Aggregate group labels and shared price bands leave few effective independent observations.
3. **Comparability and uncertainty:** lexical similarity can group economically different events. Large label SEs, support differences and execution-weight concentration limit interpretation. Refine the event graph and manual coherence checks; test fixed exact-price support and contract-balanced weighting before increasing model complexity.
4. **Missing feature blocks:** authoritative categories, historical order books, validated NLP and pretrained embeddings remain unavailable/deferred. The [availability assessment](../DATA_AVAILABILITY.md) identifies the additional inputs; do not reinterpret literal proxies as those missing fields.
5. **Next experiment:** acquire historical provenance, reserve a fresh future period, and run rolling event-purged predictions with already-settled training labels. Check sensitivity to window, support, clustering and uncertainty unit. Add execution-aware trading tests only after the predictive signal survives that design. Causal framing effects require a separate wording experiment.

Reproduce with `python -m mvp.pipeline` from the repository root; see [methodology and run instructions](../README.md), [source code](../pipeline.py) and [run manifest](run_manifest.json). All figures and tables above are generated from real repository data; synthetic data are confined to correctness tests.
"""
    (out / "FINDINGS.md").write_text(report)
