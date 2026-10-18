"""Generate a research report and figures entirely from saved CSV analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def table(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:,.3f}"
        return str(value).replace("|", "/").replace("\n", " ")

    return "\n".join(
        ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        + ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    )


def wedges(curves: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pivot = curves.pivot(index=keys, columns="direction", values="mispricing_pp").reset_index()
    pivot["wedge_pp"] = pivot.YES - pivot.NO
    return pivot


def write_report(out: Path, target: Path, recompute_equality: bool = True) -> None:
    from .heterogeneity import run_fixed_support

    print("Testing equality of wedges on complete price support...", flush=True)
    if recompute_equality or not (out / "fixed_support_equality_tests.csv").exists():
        run_fixed_support(out)
    equality = pd.read_csv(out / "fixed_support_equality_tests.csv")
    direct = equality.query("weighting=='execution' and dependence_unit=='event_id' and question=='all_clusters'").iloc[
        0
    ]
    within = equality.query(
        "weighting=='execution' and dependence_unit=='event_id' and question=='within_structural_family'"
    ).iloc[0]
    summary = pd.read_csv(out / "cluster_summary.csv")
    tests = pd.read_csv(out / "heterogeneity_tests.csv")
    coverage = pd.read_csv(out / "population_coverage.csv")
    pooled = pd.read_csv(out / "pooled_curves.csv")
    audit = __import__("json").loads((out / "data_audit.json").read_text())
    primary = summary.query("weighting=='execution' and reference=='pooled' and band=='1-49'").copy()
    parent = summary.query("weighting=='execution' and reference=='family' and band=='1-49'").copy()
    full = primary[primary.n_costs.eq(49)]
    units = summary.query("weighting=='unit' and reference=='pooled' and band=='1-49'")
    test = tests.query("weighting=='execution' and reference=='pooled' and band=='1-49'").iloc[0]
    parent_test = tests.query("weighting=='execution' and reference=='family' and band=='1-49'").iloc[0]
    merged = primary.merge(units, on="cluster", suffixes=("_execution", "_unit"))
    concordance = float((np.sign(merged.wedge_pp_execution) == np.sign(merged.wedge_pp_unit)).mean())
    assets = target.parent / "figures"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    pw = wedges(pooled, ["population", "weighting", "cost"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = {
        "all_resolved": "All resolved CSV markets",
        "non_combination": "Non-combination",
        "assigned_matched": "Assigned: matched population",
    }
    for weighting, ax in zip(["execution", "unit"], axs):
        for population, label in labels.items():
            g = pw[(pw.weighting == weighting) & (pw.population == population)]
            ax.plot(g.cost, g.wedge_pp, label=label)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(
            title=f"{weighting.capitalize()} weighting",
            xlabel="Purchase cost (cents)",
            ylabel="YES minus NO mispricing (pp)",
        )
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "pooled_wedges.png", dpi=180)
    plt.close(fig)
    # Show full-price-support groups to keep displayed levels directly comparable.
    show = (
        pd.concat([full.nsmallest(10, "wedge_pp"), full.nlargest(10, "wedge_pp")])
        .drop_duplicates("cluster")
        .sort_values("wedge_pp")
    )
    if show.empty:
        show = (
            pd.concat([primary.nsmallest(10, "wedge_pp"), primary.nlargest(10, "wedge_pp")])
            .drop_duplicates("cluster")
            .sort_values("wedge_pp")
        )
    fig, ax = plt.subplots(figsize=(11, max(5, len(show) * 0.28)))
    y = np.arange(len(show))
    ax.errorbar(
        show.wedge_pp,
        y,
        xerr=1.96 * show.se_event_pp,
        fmt="o",
        capsize=2,
        label="Cluster wedge; pointwise event 95% CI",
    )
    ax.scatter(show.pooled_matched_pp, y, marker="x", color="black", label="Pooled wedge on identical prices")
    ax.set_yticks(y, show.cluster)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("YES minus NO mispricing (probability percentage points)")
    ax.set_title(
        "Selected extremes with complete 1–49¢ support" if len(full) else "Selected extremes; price support varies"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(assets / "cluster_wedges.png", dpi=180)
    plt.close(fig)
    cc = pd.read_csv(out / "cluster_curves_execution.csv")
    known_clusters = pd.read_csv(Path(audit["inputs"][2]["path"]), usecols=["cluster"], keep_default_na=False)
    known_clusters = sorted(set(known_clusters.cluster) - {""})
    eligibility = pd.DataFrame({"cluster": known_clusters})
    eligibility["contributes_resolved_trades"] = eligibility.cluster.isin(cc.cluster)
    eligibility["primary_eligible"] = eligibility.cluster.isin(primary.cluster)
    eligibility.to_csv(out / "cluster_eligibility.csv", index=False)
    cw = wedges(cc, ["cluster", "cost"])
    cw = cw[cw.cost.between(1, 49)]
    # Mask individual costs lacking bilateral support, even in otherwise eligible groups.
    supported = cc[(cc.contracts >= 20) & (cc.events >= 10)].groupby(["cluster", "cost"]).direction.nunique()
    supported = supported[supported.eq(2)].reset_index()[["cluster", "cost"]]
    grid = cw.merge(supported, on=["cluster", "cost"]).pivot(index="cluster", columns="cost", values="wedge_pp")
    ordered = primary.sort_values("wedge_pp").cluster
    grid = grid.reindex(ordered).reindex(columns=range(1, 50))
    fig, ax = plt.subplots(figsize=(10, 7))
    limit = max(5, float(np.nanquantile(np.abs(grid.to_numpy()), 0.95)))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks([0, 9, 19, 29, 39, 48], [1, 10, 20, 30, 40, 49])
    ax.set_yticks([])
    ax.set(
        xlabel="Purchase cost (cents)",
        ylabel=f"{len(grid)} eligible clusters, sorted by average wedge",
        title="Cluster-specific exact-price wedges; blank cells lack support",
    )
    fig.colorbar(im, ax=ax, label="YES minus NO mispricing (pp); colors clipped at 95th absolute percentile")
    fig.tight_layout()
    fig.savefig(assets / "wedge_heatmap.png", dpi=180)
    plt.close(fig)
    # Add a human-readable representative title/subtitle for interpretation.
    assignment_path = Path(audit["inputs"][2]["path"])
    a = pd.read_csv(assignment_path, keep_default_na=False, usecols=["cluster", "text", "similarity"])
    a["similarity"] = pd.to_numeric(a.similarity, errors="coerce")
    examples = (
        a[a.cluster.ne("")]
        .sort_values("similarity", ascending=False)
        .drop_duplicates("cluster")
        .set_index("cluster")
        .text
    )
    summary["representative_text"] = summary.cluster.map(examples)
    boundary_maps = {}
    for weighting in ["execution", "unit"]:
        raw = pd.read_csv(out / f"cluster_curves_{weighting}.csv")
        boundary_maps[weighting] = {
            (r.cluster, r.cost, r.direction): r.win_rate in (0.0, 1.0) for r in raw.itertuples()
        }
    summary["boundary_price_cells"] = [
        sum(
            boundary_maps[r.weighting].get((r.cluster, int(c), d), False)
            for c in r.costs.split(",")
            for d in ["YES", "NO"]
        )
        for r in summary.itertuples()
    ]
    summary["degenerate_event_variance"] = summary.se_event_pp.lt(1e-8)
    summary.to_csv(out / "cluster_summary.csv", index=False)
    standout = summary.query("weighting=='execution' and reference=='pooled' and band=='1-49' and q_event<.05")
    standout = standout[
        [
            "cluster",
            "n_costs",
            "wedge_pp",
            "difference_pp",
            "q_event",
            "q_market",
            "p_event_maxT",
            "degenerate_event_variance",
        ]
    ]
    selected = show[
        [
            "cluster",
            "events",
            "n_costs",
            "wedge_pp",
            "se_event_pp",
            "pooled_matched_pp",
            "difference_pp",
            "q_event",
            "p_event_maxT",
        ]
    ].copy()
    selected["95% CI"] = selected.apply(
        lambda r: f"[{r.wedge_pp - 1.96 * r.se_event_pp:.2f}, {r.wedge_pp + 1.96 * r.se_event_pp:.2f}]", axis=1
    )
    selected = selected.drop(columns="se_event_pp")
    parent_show = parent[parent.n_costs.eq(49)].sort_values("p_event_maxT").head(12)
    if parent_show.empty:
        parent_show = parent.sort_values("p_event_maxT").head(12)
    psummary = (
        pw[pw.cost.between(1, 49)]
        .groupby(["population", "weighting"])
        .wedge_pp.agg(["mean", "min", "max"])
        .reset_index()
    )
    point = pooled[
        (pooled.population == "all_resolved") & (pooled.weighting == "execution") & pooled.cost.isin([1, 5, 10, 25, 50])
    ]
    point = point[["cost", "direction", "mispricing_pp", "return_pct", "contracts", "events"]]
    pair_significance = parent[(parent.p_event_maxT < 0.05) & (parent.q_market < 0.05)]
    fixed = full.wedge_pp
    paper_comparison = [
        {
            "sample/weight": "Original paper, full historical sample (reported)",
            "YES return %": -41.0,
            "NO return %": 23.0,
            "YES minus NO return pp": -64.0,
        }
    ]
    for weighting in ["execution", "unit"]:
        price_one = pooled[
            (pooled.population == "all_resolved") & (pooled.weighting == weighting) & pooled.cost.eq(1)
        ].set_index("direction")
        yes_return = price_one.loc["YES", "return_pct"]
        no_return = price_one.loc["NO", "return_pct"]
        paper_comparison.append(
            {
                "sample/weight": f"Recent-year CSV, {weighting} weighted",
                "YES return %": yes_return,
                "NO return %": no_return,
                "YES minus NO return pp": yes_return - no_return,
            }
        )
    paper_comparison = pd.DataFrame(paper_comparison)
    unit_within = equality.query(
        "weighting=='unit' and dependence_unit=='event_id' and question=='within_structural_family'"
    ).iloc[0]
    unit_within_market = equality.query(
        "weighting=='unit' and dependence_unit=='ticker' and question=='within_structural_family'"
    ).iloc[0]
    lowest = primary.nsmallest(1, "wedge_pp").iloc[0]
    touchdown_cases = []
    for name in ["sports/NFLANYTD/unknown/000", "sports/NFLFIRSTTD/unknown/000"]:
        matching = primary[primary.cluster.eq(name)]
        if len(matching):
            row = matching.iloc[0]
            touchdown_cases.append(
                f"- `{name}`: wedge {row.wedge_pp:.2f} pp over {row.n_costs} costs; "
                f"event max-T p={row.p_event_maxT:.4f}, market FDR q={row.q_market:.4f}."
            )
    touchdown_text = "\n".join(touchdown_cases)
    text = f"""# Does the YES/NO mispricing wedge vary across similar-contract clusters?

## Findings

**The pooled YES disadvantage remains, and a few contract types differ from that average.
However, the primary execution-weighted tests do not establish additional variation among
learned subclusters within the same structural family.** A secondary unit-weighted common-support
test does find within-family variation, so that lead depends on weighting and needs new-data validation.
The most dramatic observed cluster wedges should not be treated as precise estimates.

The recent-year CSV analysis contains **{coverage.loc[coverage.population.eq("assigned_matched"), "contracts"].iloc[0]:,} resolved, volume-screened assigned contracts**.
Of the learned clusters, **{len(primary)}** have enough bilateral exact-price support for the primary summary.
The frozen assignments contain **{len(known_clusters)} groups**, of which **{cc.cluster.nunique()}** contribute eligible resolved trades;
remaining groups or prices are omitted from supported summaries rather than assigned zero wedges.
Their observed standardized wedges range from **{primary.wedge_pp.min():.2f} to {primary.wedge_pp.max():.2f} probability percentage points**.
These extremes alone are descriptive: some groups have different price support and some estimates are imprecise.

A direct equal-wedge test on the **{int(direct.tested_clusters)} clusters with complete 1–49¢ support**
has event-level p = **{direct.p_value:.4f}**. Centering within structural families instead
compares **{int(within.tested_clusters)} clusters** and gives p = **{within.p_value:.4f}**.
These tests compare cluster wedges directly on the same prices, rather than relying on
cluster-versus-pooled contrasts alone.

Against the ungrouped baseline built from exactly the assigned population and matched prices,
**{int(test.event_fdr_05)} of {len(primary)} clusters differ at event-level FDR 5%**, and
**{int(test.maxT_05)}** survive the more stringent joint event max-T adjustment.
The global event max-T p-value is **{test.global_maxT_p:.4f}** (simulation resolution 0.0001).
Using market-level rather than event-level uncertainty, **{int(test.market_fdr_05)}** pass FDR 5%.

To separate learned subgroup variation from broad family differences, clusters were also compared
with their own structural parent on the same exact prices. **{int(parent_test.maxT_05)}** survive
joint event max-T adjustment in that comparison; **{len(pair_significance)}** also pass market-level FDR.
This is the more direct check of whether learned subgroups add information beyond the structural parents.
One apparently strong pooled result, `sports/NFLWINS/above/000`, has zero estimated
cluster-level event variance: all its retained direction/price cells have boundary win rates.
Its asymptotic p-value is not credible evidence of a precisely known zero wedge. The raw test counts
above include this flagged case; it should not be promoted as an established discovery.

All tests are exploratory, conditional on the fixed grouping and support rules; they do not establish
causal framing effects, persistent predictability or profitable trading.

## Source methodology and what was reproduced

The source is Jonathan Becker, *The Microstructure of Wealth Transfer in Prediction Markets*,
the original PDF in the repository root, rather than the MVP or the recreated article/figure scripts.
PDF page 6 (printed page 5), Equation (1), defines mispricing as realized win rate minus implied probability.
PDF page 11 (printed page 10), Section 6.3 and Equation (3), defines the same-cost YES/NO wedge.
Section 5.1 and Figure 5 report gross excess returns by direction and price.

For price c, group g and direction d:

```text
q[g,c,d]     = sum(weight × resolved payoff) / sum(weight)
delta[g,c,d] = q[g,c,d] - c/100
Delta[g,c]   = delta[g,c,YES] - delta[g,c,NO]
report pp    = 100 × Delta[g,c]
return %     = 100 × (100 × q[g,c,d] - c) / c
```

Each execution supplies one YES position at yes_price and one NO position at no_price;
their payoffs are y and 1-y. Both execution roles are pooled. This does not assume that one
trader bought both sides. Negative Delta means a YES disadvantage; positive means a NO disadvantage.

Primary weighting is one observation per execution per direction, consistent with the paper's
mean-per-trade formulation. The unit-weighted sensitivity uses trade `count`, matching the repository's
`ev_yes_vs_no.py` implementation. That script is not substituted for the PDF's complete methodology.
Neither weighting uses notional dollars as the outcome weight. All results are gross of fees.

A probability-point wedge is different from a return-percentage gap: at 1¢, a 0.64-point wedge
corresponds to a 64% gross-return gap, while at 50¢ it corresponds to 1.28%.
The original full-sample 1¢ returns (about -41% YES and +23% NO) are context, not replication targets
for a different time window and selected population.

## CSV sample and comparable pooled baselines

Only `data/kalshi_recent_year_csv/master/markets.csv`, `master/trades.csv`, and the frozen
`output/clustering/assignments.csv` were read. No Parquets and no MVP code/results are used.
The extraction audited **{audit["raw_trades"]:,} trade rows**, **{audit["distinct_trade_ids"]:,} distinct IDs**,
and **{audit["invalid_trades"]:,} invalid executions**. Missing or repeated IDs stop the pipeline;
invalid prices, complementary-price inconsistencies, nonpositive sizes and invalid taker sides are excluded.

Markets must be finalized, binary and resolved YES/NO. The paper's $100 minimum notional-volume
screen is implemented as at least 100 traded units in the observed CSV window: complementary
positions together cost $1 per unit, consistent with the repository's `volume_over_time.py` convention.
The PDF does not fully specify every extraction detail; this is a documented implementation choice.
The recent-year window can truncate older contracts' activity, so this is not a full historical replication.
Unlike the MVP, there is no first-24-hour restriction or five-early-trade requirement.

{table(coverage)}

Three ungrouped curves expose selection effects: all eligible markets (including combinations),
non-combination markets, and precisely the assigned population. The first is the closest available
paper-style baseline; only the third is the direct reference for cluster comparisons.
Unknown/void/open markets do not contribute outcomes. Combination products are excluded from
cluster inference, but retained in the broad baseline when otherwise eligible.

![Pooled wedges](figures/pooled_wedges.png)

Mean and range of pooled exact-price wedges across 1–49¢ (equal weight per cost):

{table(psummary)}

Selected exact-price results from the execution-weighted broad CSV baseline:

{table(point)}

For a direct numerical comparison with the paper, here is the 1¢ result in its gross-return units:

{table(paper_comparison)}

The first row is the paper's rounded, reported full-sample result, not recalculated from unavailable
full-history CSVs. The next two rows rerun the same-cost direction comparison on the recent-year CSVs.
The unit-weighted row follows the repository's figure weighting; the execution-weighted row follows
the mean-per-trade convention used for this report's primary analysis. Differences in sample period,
screening and weight convention prevent interpreting numerical closeness as an exact replication.

Full 1–99¢ curves, including return-percentage equivalents, are in [pooled_curves.csv](results/pooled_curves.csv).
Differences from the paper can reflect time, eligibility, weighting and composition; they cannot
be attributed to clustering itself.

## Price matching and supported cluster summaries

The paper defines an exact-price curve, not a single cluster-level scalar. The following
standardized summaries are extensions for comparing curves, explicitly separated from that replication:

- Retain an exact cost only if each direction has at least **20 distinct contracts and 10 event IDs**.
- Primary summary: equal-cost mean of Delta over supported costs from **1–49¢**, requiring at least 10 costs.
- Additional bands: 1–10, 11–20, 21–30, 31–40 and 41–49¢, each requiring at least six costs.
- Compare the cluster with the pooled or parent-family curve averaged over **identical retained costs**.
  Support is determined without using realized win rates or significance.
- Unsupported costs are missing, never zero mispricing. Raw exact-price curves are still exported.

For all-side pooling, Delta(c) = Delta(100-c) exactly. Thus 51–99¢ duplicate 1–49¢,
and 50¢ is a special twice-YES-calibration contrast. Full curves are exported, but neither
those mirrored observations nor 50¢ are counted as additional evidence in the summaries.
Equal-cost averaging avoids giving the YES and NO samples different price mixtures.

**{len(full)} clusters have complete support at all 49 primary prices**.
For this common-support subset, observed wedges range from **{fixed.min():.2f} to {fixed.max():.2f} pp**
with cross-cluster sample SD **{fixed.std():.2f} pp**. These are directly comparable observed
levels, but their dispersion still includes sampling noise and is not a latent variance estimate.

![Cluster wedges](figures/cluster_wedges.png)

Selected extreme groups are shown for interpretation, not selected as trading opportunities.
Pointwise confidence intervals for wedge levels differ from simultaneous inference for contrasts.

{table(selected)}

![Exact-price variation](figures/wedge_heatmap.png)

## Which differences deserve follow-up?

Execution-weighted clusters passing the exploratory event-FDR screen:

{table(standout)}

Two concrete contract-family comparisons are:

{touchdown_text}

These are distinct contract families, each represented here by cluster 000; they do not
by themselves demonstrate that NLP-discovered subdivisions improve on the structural grouping.
Narrower-support products describe different purchase-cost sets than the all-49-cost cohort.

The smallest observed wedge is `{lowest.cluster}` at {lowest.wedge_pp:.2f} pp, with
event SE {lowest.se_event_pp:.2f} pp and event FDR q={lowest.q_event:.3f}.
Large observed values must be assessed alongside uncertainty, not ranked by magnitude alone.

## Variation beyond the structural parents

Parent references pool assigned siblings in the same domain/underlying/structure family.
They include the focal cluster, so contrasts are correlated and must use paired uncertainty.
For a family with only one contributing cluster, its cluster-parent contrast is identically zero.
Representative comparisons (complete-support groups first, ordered by adjusted event p-value):

{table(parent_show[["cluster", "n_costs", "wedge_pp", "pooled_matched_pp", "difference_pp", "q_event", "q_market", "p_event_maxT"]])}

The reference column `pooled_matched_pp` means the matched **parent** in family-reference rows.
Cluster IDs are identifiers, not economic labels. The complete summary CSV includes representative
title/subtitle text to help interpret each group without guessing from its numeric suffix.

## Statistical uncertainty and multiple comparisons

The analysis uses an analytic ratio-estimator influence function, aggregating all repeated
positions by `event_ticker` for event-level sandwich standard errors. Market-level sandwiches
use ticker instead. The cluster and its reference share observations, so their influence functions
are subtracted before calculating the contrast's variance; independent-sample errors would be wrong.
Both directions and all exact costs remain correlated through their shared outcomes.

Two-sided normal-approximation p-values are adjusted using Benjamini–Hochberg FDR within each
weighting/reference/band comparison family. A joint Gaussian max-T simulation (10,000 draws,
seed 340), using the estimated event-level cross-cluster covariance, supplies a global check,
familywise-adjusted cluster p-values and simultaneous contrast intervals. This is asymptotic
inference, not a raw-trade bootstrap or a finite-sample randomization test. BH control is not
guaranteed under arbitrary dependence; the correlated max-T results provide a complementary check.
Secondary bands and weighting choices are reported separately, not collectively corrected as one
preregistered experiment. Do not choose a favorable band after reading its results.

{table(tests)}

Win rates exactly zero or one can make empirical sandwich variances collapse. The summary
exports `boundary_price_cells` and `degenerate_event_variance` to make this limitation visible.
No continuity adjustment or pseudo-outcomes are silently added. Normal-approximation tests
are particularly unreliable in these boundary cases.

Minimum event counts do not guarantee a large effective sample when executions or units are
concentrated in a few events. Event IDs do not capture every shared underlying outcome.
Groups, price support and the resolved population are treated as fixed: uncertainty excludes
learned-cluster instability, support selection and historical metadata revisions.

## Direct equality of cluster wedges on common support

The strongest check of heterogeneity restricts to clusters supported at all 49 costs,
then tests whether their equal-cost wedges are identical. The statistic is the largest
absolute standardized deviation from the unweighted cluster mean. A second version
centers within structural families and excludes families with only one complete-support
cluster. Gaussian simulation uses the full event- or market-cluster covariance of those
centered contrasts. These tests avoid differences in each cluster's price support and
direction-specific composition of a trade-weighted pooled reference.

{table(equality)}

For the primary execution-weighted cohort, the event-level all-cluster test gives p={direct.p_value:.4f}
and the within-family test gives p={within.p_value:.4f}. Unit weighting yields
within-family p={unit_within.p_value:.4f} with event errors and p={unit_within_market.p_value:.4f} with market errors. This is a potentially
useful but secondary, weighting-sensitive result, not consistent evidence across definitions.
These are global tests; they do not identify a validated individual subgroup or trading rule.
The cohort excludes clusters without complete price support, so a non-rejection here does
not negate differences in narrower-support products such as first-touchdown contracts.

The all-cluster and within-family questions, weights and dependence units are reported
as separate exploratory checks; they are not a single preregistered family of tests.

## Weighting sensitivity and split interpretation

Execution and unit weighting have the same wedge sign for **{100 * concordance:.1f}%** of the
primary eligible clusters. Their wedge levels and uncertainties can differ because large executions
receive more weight under the second convention. All unit results are supplied rather than
silently treating one convention as interchangeable with the other.

Train/validation/test contracts are pooled for this retrospective heterogeneity analysis: no
mispricing model is trained or evaluated. Cluster assignments remain frozen from training text.
`split_curves_execution.csv` and `split_curves_unit.csv` retain split-specific curves for inspection;
these event-hash splits do not constitute temporal replication or evidence of persistence.

## Interpretation and limits

Observed cross-cluster variation is assessed against sampling uncertainty and matched-price references,
not inferred solely from a colorful heatmap. Structural parents, strikes, dates, entities and event
composition can all contribute. Even a supported cluster-parent contrast does not establish that
wording caused the difference or that contracts in that cluster offer executable opportunities.

The paper's causal interpretation is not re-established by this exercise. Snapshot text may not be
historically versioned; executed prices are not current quotes; no fee, spread, fill, inventory or
capital-duration simulation is performed. Active/finalized selection and the recent-year window alter
the population. A new time period, dependence-aware event graph and prespecified grouping are the
next checks before treating any observed wedge difference as a predictive signal.

## Reproduce and inspect

```bash
.venv/bin/python -m src.analysis.kalshi.clustering.mispricing.pipeline
# Avoid rescanning raw CSVs if audited input file metadata are unchanged:
.venv/bin/python -m src.analysis.kalshi.clustering.mispricing.pipeline --reuse-stats
```

Read [cluster_summary.csv](results/cluster_summary.csv) for all groups, supports, event/market errors,
matched references, adjusted p-values and representative text. Exact-price curves and split curves
are in `results/`; the input audit and completed run manifest record provenance. Results and compressed
sufficient statistics are ignored generated artifacts; this report and its figures are reviewable outputs.
The original PDF remains the source for Equations (1)–(3); the MVP is not an input or methodological source.
"""
    target.write_text(text)
