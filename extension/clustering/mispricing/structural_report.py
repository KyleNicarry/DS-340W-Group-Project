"""Report the rule-only grouping experiment without overwriting learned-group results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .report import table, wedges


def write_report(out: Path, previous: Path, target: Path) -> None:
    summary = pd.read_csv(out / "structural_summary.csv")
    tests = pd.read_csv(out / "heterogeneity_tests.csv")
    equality = pd.read_csv(out / "fixed_support_equality_tests.csv")
    coverage = pd.read_csv(out / "population_coverage.csv")
    pooled = pd.read_csv(out / "pooled_curves.csv")
    primary = summary.query("population=='all_structural' and weighting=='execution' and band=='1-49'")
    fixed = primary[primary.n_costs.eq(49)]
    main_test = tests.query("population=='all_structural' and weighting=='execution' and band=='1-49'").iloc[0]
    event = equality.query(
        "population=='all_structural' and weighting=='execution' and dependence_unit=='event_id'"
    ).iloc[0]
    market = equality.query(
        "population=='all_structural' and weighting=='execution' and dependence_unit=='ticker'"
    ).iloc[0]
    unit = equality.query("population=='all_structural' and weighting=='unit' and dependence_unit=='event_id'").iloc[0]
    old_tests = pd.read_csv(previous / "fixed_support_equality_tests.csv")
    comparison = []
    for weighting in ["execution", "unit"]:
        for dep in ["event_id", "ticker"]:
            old = old_tests[
                (old_tests.weighting == weighting)
                & (old_tests.dependence_unit == dep)
                & old_tests.question.eq("all_clusters")
            ].iloc[0]
            comparison.append(
                {
                    "grouping": "Learned subclusters, prior assigned population",
                    "weighting": weighting,
                    "dependence": dep,
                    "complete_support_groups": old.tested_clusters,
                    "equal_wedge_p": old.p_value,
                }
            )
            for population, label in [
                ("all_structural", "Structural families, all eligible"),
                ("same_ml_population", "Structural families, prior assigned population"),
            ]:
                row = equality[
                    (equality.weighting == weighting)
                    & (equality.dependence_unit == dep)
                    & equality.population.eq(population)
                ].iloc[0]
                comparison.append(
                    {
                        "grouping": label,
                        "weighting": weighting,
                        "dependence": dep,
                        "complete_support_groups": row.tested_groups,
                        "equal_wedge_p": row.p_value,
                    }
                )
    comparison = pd.DataFrame(comparison)
    pw = wedges(pooled, ["population", "weighting", "cost"])
    pooled_summary = (
        pw[pw.cost.between(1, 49)]
        .groupby(["population", "weighting"])
        .wedge_pp.mean()
        .rename("equal_cost_wedge_pp")
        .reset_index()
    )
    prior_pooled = wedges(pd.read_csv(previous / "pooled_curves.csv"), ["population", "weighting", "cost"])
    figures = target.parent / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for weighting, ax in zip(["execution", "unit"], axes):
        for population, label in [
            ("all_structural", "All rule-eligible contracts"),
            ("same_ml_population", "Prior ML-assigned contracts only"),
        ]:
            g = pw[(pw.weighting == weighting) & pw.population.eq(population)]
            ax.plot(g.cost, g.wedge_pp, label=label)
        old = prior_pooled[(prior_pooled.weighting == weighting) & prior_pooled.population.eq("all_resolved")]
        ax.plot(old.cost, old.wedge_pp, ls="--", color="gray", label="Ungrouped paper-style CSV baseline")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(
            title=f"{weighting.capitalize()} weighting", xlabel="Cost (cents)", ylabel="YES minus NO mispricing (pp)"
        )
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "structural_population_curves.png", dpi=180)
    plt.close(fig)
    show = (
        pd.concat([fixed.nsmallest(10, "wedge_pp"), fixed.nlargest(10, "wedge_pp")])
        .drop_duplicates("cluster")
        .sort_values("wedge_pp")
    )
    if show.empty:
        show = primary.sort_values("wedge_pp").head(20)
    fig, ax = plt.subplots(figsize=(11, max(5, len(show) * 0.30)))
    y = np.arange(len(show))
    ax.errorbar(
        show.wedge_pp,
        y,
        xerr=1.96 * show.se_event_pp,
        fmt="o",
        capsize=2,
        label="Structural wedge; pointwise event 95% CI",
    )
    ax.scatter(
        show.pooled_matched_pp, y, marker="x", color="black", label="Same-population, same-price pooled reference"
    )
    ax.set_yticks(y, show.cluster)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("YES minus NO mispricing (probability pp)")
    ax.set_title("Rule-only families: selected complete-support extremes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "structural_family_wedges.png", dpi=180)
    plt.close(fig)
    strongest = primary.sort_values("p_event_maxT").head(12)
    strongest = strongest[
        [
            "cluster",
            "contracts",
            "events",
            "n_costs",
            "wedge_pp",
            "pooled_matched_pp",
            "difference_pp",
            "se_event_pp",
            "q_event",
            "q_market",
            "p_event_maxT",
        ]
    ]
    robust = primary[primary.p_event_maxT.lt(0.05) & primary.q_market.lt(0.05) & primary.se_event_pp.gt(1e-8)]
    sentence = (
        "rejects equal structural-family wedges"
        if event.p_value < 0.05
        else "does not reject equal structural-family wedges at 5%"
    )
    if 0.01 <= event.p_value <= 0.10:
        sentence = "shows borderline evidence of unequal structural-family wedges"
    text = f"""# Rule-only structural groups: YES/NO mispricing results

## Main findings

This rerun uses **no TF-IDF, LSA, sentence embeddings, k-means, or hierarchical clustering**.
Contracts are grouped only by the existing rules: **domain → underlying → contract structure**.
Those rules were recomputed directly from the master market CSV's titles, YES subtitles and event-series codes.

The full structural population contains **{coverage.loc[coverage.population.eq("all_structural"), "contracts"].iloc[0]:,} resolved contracts**
across **{coverage.loc[coverage.population.eq("all_structural"), "structural_groups"].iloc[0]} rule-defined families**, many of them sparse series-fallback groups.
**{len(primary)} families** have adequate primary price support; **{len(fixed)}** support every cost from 1–49¢.
The complete-support families' observed execution-weighted wedges range from
**{fixed.wedge_pp.min():.2f} to {fixed.wedge_pp.max():.2f} pp**.

The direct common-support equality test **{sentence}**:
event-level p = **{event.p_value:.4f}**, market-level p = **{market.p_value:.4f}**.
The corresponding unit-weighted event-level p-value is **{unit.p_value:.4f}**.
These are exploratory normal/Gaussian-approximation tests conditional on rules, support and observed outcomes.
A p-value near 0.05 is suggestive rather than decisive, particularly across the multiple sensitivity checks.
Direct equality tests use 100,000 Gaussian draws to reduce Monte Carlo error near that threshold.

Against the matched pooled reference, **{int(main_test.event_fdr_05)}** families pass event FDR 5%,
**{int(main_test.market_fdr_05)}** pass market FDR 5%, and **{int(main_test.maxT_05)}** pass event max-T 5%.
**{len(robust)}** pass both event max-T and market FDR with nonzero estimated family variance.
Raw counts can include boundary-win-rate cases; zero variance is not proof of precision.

## What differs from the original paper?

The original paper reports a pooled YES disadvantage at matched purchase costs.
That general pattern remains in this CSV subset. Our extension asks whether the wedge differs
across economic contract families; the original pooled curve does not answer that question.
Exact magnitudes need not reproduce its full-history figures because this is a recent-year
CSV extract, with explicit validation and a different mix of markets.

There are two distinct extensions: **structural-family differences** and **additional differences
between learned subclusters within one structural family**. Evidence for the former would not
establish the latter. The previous primary learned-subcluster within-family test had p=0.8659;
its secondary unit-weighted version had p=0.0287. Neither is a forecast-performance comparison.

## Isolate grouping from selection

The primary population includes all rule-eligible contracts, including those that the ML pipeline
left unassigned for distance, sparse training support, or unseen training families. Family eligibility
does not depend on a learned centroid or training-set coverage. Unknown categories are retained under
explicit unknown/series-fallback families rather than invented labels.

A second run uses **exactly the previous ML-assigned contract population**, with all its learned
cluster IDs replaced by structural family IDs. This controls the contract/trade population when
comparing grouping resolutions. The learned labels determine only this clearly labeled sensitivity's
population membership; they do not determine any structural family or distance.

{table(coverage)}

Both populations use the same cached, audited execution sufficient statistics from the master CSVs.
The program verifies CSV file fingerprints before reuse and recomputes structural attributes afresh.
No raw Parquets or MVP code/data are used.

The ungrouped equal-cost mean wedge over 1–49¢ is:

{table(pooled_summary)}

![Structural population curves](figures/structural_population_curves.png)

## Compare the grouping resolutions

The table tests equality of equal-cost wedges among groups with complete 1–49¢ support.
Its null is equal group-level wedges, not zero mispricing and not equality of each group's exact-price curve.

{table(comparison)}

Even when contracts are matched, the complete-support cohorts and number of groups can differ:
aggregating subclusters changes support and statistical power. These are transparent sensitivity
comparisons, not a paired model-selection test proving that one grouping approach is superior.
Market/event dependence, unit weighting and repeated exploratory checks also affect interpretation.

## Which structural groups differ?

![Structural family wedges](figures/structural_family_wedges.png)

The figure selects extreme observed levels among complete-support families. Its intervals are pointwise,
not simultaneous. The following table instead orders primary comparisons by event-adjusted max-T p-value;
its support can vary by family, so the pooled reference always uses the same costs as that family.

{table(strongest)}

Negative wedge means YES underperforms equally priced NO; positive means the reverse.
The units are probability percentage points, not return percentages or net trading profit.
At one cent a one-point calibration error is a 100% gross return error; the same translation
does not hold at other costs.

## Method and inference

The paper's estimator is retained: realized payoff frequency minus cost/100 for each direction
at each exact cost, then YES minus NO. Every valid execution contributes one complementary position
in each direction, pooling maker and taker roles. Primary weighting is execution count; the
sensitivity uses traded units. Finalized binary YES/NO markets need at least $100 observed notional
(sum of traded units). This is an observed-window screen, not a full-history notional reconstruction.

Structural rules are in `../features.py`; they recognize selected domains/underlyings and spreads,
totals, intervals, above/below thresholds and winners, falling back to series where appropriate.
They are heuristic economic families, not verified settlement equivalence or authoritative taxonomy.
No numerical date/strike-distance metric or probabilistic encoder is used.

A cost contributes only with at least 20 contracts and 10 event IDs **in each direction**.
The primary equal-cost average spans supported 1–49¢ costs, requiring at least 10. Full-population
secondary bands require six supported costs each. The 51–99¢ wedge mirrors 1–49¢ exactly under
all-side pooling; 50¢ is a special calibration case and is omitted from the summary. Full curves
are still saved. Missing support is not zero mispricing.

Clustered sandwich errors aggregate all positions at event ID or market ticker, preserving
within-unit dependence. Cluster-minus-pool variance uses paired influences, including overlap.
Benjamini–Hochberg adjusts within a weighting/population/band family; the joint Gaussian max-T
check uses correlated comparisons and 10,000 draws, seed 340. The direct equality check uses 100,000 draws and centers
complete-support family wedges around their unweighted mean and uses their joint covariance.
See [the earlier methodology](FINDINGS.md) for equations and limitations.

All per-band matched-reference tests:

{table(tests)}

All direct complete-support equality checks:

{table(equality)}

The tests are not jointly corrected across every population, weighting and band; secondary positive
results require independent confirmation. Boundary win rates, concentrated weights, related events
with different IDs, and the selected resolved population can undermine asymptotic inference.
Similarity or structural grouping cannot eliminate sampling noise or identify a causal wording effect.

## Reproduce and inspect

```bash
# Requires the audited baseline statistics from the previous analysis.
.venv/bin/python -m extension.clustering.mispricing.structural
```

The new results are under `results/structural/`; the learned-cluster results remain intact.
`structural_assignments.csv` contains fresh rule-only assignments, `structural_summary.csv`
contains supports and uncertainty, and exact-price curves retain every contributing family.
`run_manifest.json` records inputs, source hashes, versions and completion. This report and
its figures are separate reviewable outputs alongside the previous report.
"""
    target.write_text(text)
