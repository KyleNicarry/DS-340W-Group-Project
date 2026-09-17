# Predicting the optimism-tax wedge: reproducible MVP

This extension estimates **group-by-cost-band YES/NO mispricing differences**, never an observed contract-level wedge. Its purpose is to test whether similarities in early market activity and question wording help predict these constructed labels. Read [FINDINGS.md](results/FINDINGS.md) for the delivered run and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for field-level limitations.

## Run

From `DS-340W-Group-Project` with Python 3.9:

```bash
python3 -m venv .venv-mvp
.venv-mvp/bin/python -m pip install -r mvp/requirements.txt
.venv-mvp/bin/python -m mvp.pipeline
```

The existing project `.venv` can also run `python -m mvp.pipeline` after installing these dependencies. Inputs must already exist at `data/kalshi/{markets,trades}/*.parquet`; there is no hidden download or live API call. Allow several minutes and several GB of RAM/disk for the full scan. DuckDB is capped at 3 GB with disk spill, four SQL threads and two numerical-library threads. Larger data can take longer.

```bash
# Explicit, reproducible sensitivity run. Do not mix output directories.
.venv-mvp/bin/python -m mvp.pipeline --clusters 12 --bootstrap 500 --output mvp/results_k12
# Core correctness tests (pytest is a development dependency).
.venv/bin/python -m pytest tests/mvp -q
```

All defaults and package versions are written to `results/run_manifest.json`, along with file paths, sizes, modification times and the pipeline's SHA-256. File metadata identify the local extract but are not cryptographic hashes of the large raw data. Seeds fix ticker sampling, series splits, clustering, models and bootstrap sampling; different BLAS/platform versions may yield small floating-point differences. `requirements.txt` pins direct analysis dependencies; the manifest records the numerical environment. The default invocation recomputes everything and replaces the same named outputs. Do not read an incomplete run as a completed analysis: check `completed: true` in its manifest.

## Population, observation window and split

1. Use Kalshi only. The original paper's equation (1), page 6 of the supplied PDF, defines mispricing as mean outcome minus mean cost; equation (3), page 11, defines the pooled YES/NO wedge. Polymarket needs a separate verified asset/outcome/timestamp join and is out of scope.
2. Require unique market tickers, `finalized` status, binary type and a YES/NO outcome. Unknown/void outcomes are excluded. Fail on duplicate market IDs instead of multiplying trades in a metadata join.
3. Exclude MVE combination products by explicit `KXMVE`/`MVE` ticker prefix. Their shared legs create dependence that cannot be recovered from the available schema. Require nonempty title/event ID and opening/closing times.
4. Fix each contract's feature landmark at **open time + 24 hours**, requiring recorded close time strictly later. This is an ex-post eligibility check, not a scheduled-close predictor. It omits all shorter-lived markets and may materially change category composition.
5. Deterministically sample up to 60,000 eligible tickers, ordered by MD5 of ticker plus fixed seed. This bounds memory and is not a volume-ranked selection. Within the sampled contracts, keep valid executions from opening through the landmark. Deduplicate identical trade IDs, fail on conflicting duplicates, and reject nonpositive size, invalid side or noncomplementary/invalid prices. Require five early executions per contract. Report sequential counts in `attrition.json`.
6. Assign complete recurring series to a stable 75% train / 25% test hash split, using the prefix of `event_ticker` before its first hyphen and removing optional `KX`. This is coarser than event ID; full event IDs are asserted disjoint. Outcomes never influence the split. Related products with different series names can still be dependent.

**This is a retrospective unseen-series holdout, not a chronological trading backtest.** There is no settlement-publication time or historical title revision log. Latest snapshot wording is assumed unchanged at the landmark, and recorded opening/closing times are assumed to delimit actual trading availability. These assumptions are unverified. The first-day execution predictors are demonstrably bounded in time, but a strict as-of guarantee for metadata is impossible. Training labels may resolve after some test contracts. Thus this experiment cannot establish availability of a historical training set at a deployment date or the ability to forecast future markets. A group row also combines multiple contract landmarks; it is not a contemporaneous portfolio recommendation.

## Features and similarity

Early market features are equal-execution mean YES price, standard deviation of YES price levels, log(1+execution count) and log(1+units traded). Price-level standard deviation is a **dispersion measure**, not return volatility; it mixes path drift with variation and is affected by irregular trading. Unit volume is not dollar volume. Neither final market volume nor final quotes are used.

Interpretable title features include character/word counts, mean word length, a question-mark indicator, a threshold indicator, counts from documented positive/negative/negation/political word lists, capitalized tokens excluding the initial word, numeric tokens and selected punctuation. These are transparent surface proxies, not validated sentiment, emotion, partisan affiliation, entities, readability or parsing. Question titles are often fragments; standard prose readability formulas would be misleading.

**Offline text alternative:** title plus the observed YES subtitle (which often holds the strike or named alternative), represented using word/bigram TF-IDF (maximum 12,000 features, training minimum document frequency 3), followed by up to 32 truncated-SVD latent semantic analysis dimensions and unit normalization. Vocabulary and SVD are learned only on training series. LSA represents lexical relationships; it is not a pretrained sentence-transformer embedding. Neural semantic embeddings are deferred until pinned model weights, licenses and a reproducible cache are available. An LSA feature ablation is delivered; no claim about the incremental value of pretrained embeddings is supported.

Two frozen maps are compared with 16-cluster k-means (10 starts, seed 340):

- **text:** unit-length LSA vectors plus a unit-normalized, training-standardized/clipped surface-feature block scaled by 0.15;
- **text_price:** the above plus one training-standardized mean-price dimension, clipped to [-2,2] and scaled by 0.15.

Euclidean distance between unit text vectors corresponds to cosine distance; bounded added blocks make the mixed distance explicit. The price block's maximum pairwise squared contribution is 0.36, versus up to 4 for text. It cannot dominate by units alone, though price may correlate with topics. Standardizers, vocabulary, SVD and centroids see training inputs only; test title/subtitle pairs are assigned to frozen centroids.

`cluster_profiles.csv` contains sizes, series composition, nearest-centroid representative titles, within-cluster cosine coherence, and price P10/P90/SD. Authoritative category is unavailable and is marked so. `clustering_diagnostics.csv` reports sampled training silhouette, normalized mutual information with price deciles, and adjusted Rand agreement after removing price. IDs do not correspond between the two clustering approaches. Manual coherence review is required: numerical compactness cannot certify economic comparability. No outcome is used to select cluster count or representation. Clusters with training mean text cosine below 0.65 are marked `insufficient_coherence` and excluded from labels/models; remaining clusters still need substantive review. Price-inclusive clustering is a sensitivity analysis; text clustering is predesignated primary.

## Exact target, weights and sparse cells

Each execution at YES price p yields one YES observation with cost p and outcome y, and one NO observation with cost 100-p and outcome 1-y. Both maker- and taker-held positions are included, as in the source paper's direction cells. This does not assume that the taker bought both sides. `count` (units) is **not** the outcome weight: one recorded execution has weight one in each direction. Repeated executions share the same contract outcome and are never treated as independent Bernoulli trials for uncertainty.

For a split-specific cluster g and exact integer cost c:

```
q[g,c,Y] = sum(n[ticker,c,Y] * y[ticker]) / sum(n[ticker,c,Y])
q[g,c,N] = sum(n[ticker,c,N] * (1-y[ticker])) / sum(n[ticker,c,N])
delta[g,c,Y] = q[g,c,Y] - c/100
delta[g,c,N] = q[g,c,N] - c/100
Delta[g,c] = delta[g,c,Y] - delta[g,c,N]
```

Retain an exact cost only if **each direction** has at least five distinct contracts and three distinct events. In each cost band, average the direction-specific deltas **equally across the same retained costs**. This standardization avoids a YES/NO difference caused solely by different price distributions within a coarse band. Each direction's cost mixture is exactly matched. The band target is a uniform average over its reported supported costs, not a claim about the entire nominal band. Support can differ across groups and splits, so `costs`, `n_costs`, and the observable mean supported cost are retained. A full analysis should test common predefined support and alternative cost weights.

Bands are 1–10, 11–20, 21–30, 31–40 and 41–49 cents. The upper half is deliberately omitted: for both-side execution weighting, **Delta(c) = Delta(100-c)** exactly, so including both would duplicate labels and inflate sample size. The 50-cent contrast uses the same executions for both sides and is twice YES calibration; omit this special case.

A band label is eligible only with at least three retained exact costs, **20 distinct contracts, 10 distinct events and 50 executions in each direction**. These are transparent minimum safeguards, not a guarantee of precision. Sparse rows remain in the wedge CSV with `status=insufficient_support`, missing deltas and available sample counts; they are never zero labels. If fewer than 90% of bootstrap draws retain valid denominators, the row is marked `insufficient_bootstrap_support` and excluded from regression. Full cell counts and inclusion decisions are in `exact_cost_support_*.csv`.

Outputs use **probability percentage points**: 100 times the formulas above. A negative wedge means YES underperforms equal-cost NO. Define tax = -Delta for rankings so a higher predicted tax corresponds to a larger predicted YES disadvantage. These units are not return-on-capital percentages: a one-point probability error can imply a very different return at 1 versus 50 cents.

## Uncertainty and modeling

Resample complete event IDs with replacement 200 times within each label, retaining all contracts, price cells and both directions together. Recompute execution-weighted win rates and standardized wedge on fixed original exact-cost support. Percentile 95% CIs and bootstrap SD are saved. This avoids spurious precision from repeated trades. It does not capture learned-cluster uncertainty, selection of support, hidden event links, or all cross-series dependence. Increasing the number of draws improves Monte Carlo precision, not identification.

Separate group populations are constructed from train and test contracts **after the series split**. They can share a frozen semantic centroid, but they share no contract, event series, realized target contribution, or fitted preprocessing observation. This tests transfer within similarity regions to unseen series. Sharing the map is not random row splitting of an already estimated label. Cluster ID is excluded from model predictors; this is not a test of extrapolation to wholly unseen semantic clusters. Cross-band dependence remains, and error intervals resample test semantic clusters as blocks rather than treating bands as independent.

For each representation compare:

- training-mean baseline;
- standardized ordinary least squares;
- standardized ridge, alpha 10;
- random forest, 200 trees, maximum depth 3 and minimum leaf size 5.

Fit each supervised model with market-only, market-plus-interpretable, and market-plus-interpretable-plus-LSA inputs. Predictors are equal-contract means among contributing contracts; the supported cost mean is also included. This differs deliberately from execution-weighted outcomes. Cluster/sample reliability measures, outcomes, final fields, IDs, CIs and bootstrap SE are excluded from inputs. There is no inverse-realized-variance weighting and no smoothing toward test outcomes. Each eligible label has equal regression/evaluation weight, targeting the average eligible group-cost cell.

Report MAE, RMSE, R² and MAE improvement over the mean. The number of eligible labels is small; ordinary least squares can be severely unstable, especially with LSA. Model complexity and hyperparameters are fixed, not selected using test results. The predesignated primary specification is **text clusters + interpretable ridge**. All model comparisons are exploratory; inspecting them does not justify choosing a winner and reusing the same holdout as confirmatory evidence. Paired MAE-improvement CIs use test-cluster block resampling conditional on the fixed training fit and constructed labels, not the complete uncertainty of the data-generating process.

Training-standardized linear/ridge coefficients supply the feature analysis. Magnitudes describe associations conditional on other features, are sensitive to collinearity and cluster aggregation, and do not establish causality. There are no claim-level p-values or causal tests.

## Signal assessment

Rank held-out labels into predicted-tax tertiles using the predesignated ridge model and predicted values only. Tied predictions remain together. Compare realized tax in each tertile, high-minus-low, and high-minus-overall-baseline. A cluster-block bootstrap interval accompanies the high-minus-low contrast. The plot and tables are descriptive selection diagnostics; they are not a strategy P&L, a maker return estimate, or a claim of profitable market making.

The pool includes both execution roles and retrospective early executions, not executable quotes at the landmark. No historical spread, fee schedule, order-book depth, queue priority, fill probability, inventory exposure, capital duration or adverse-selection model is available. An optimism-tax ranking cannot be called exploitable without those inputs. Disjoint series reduce some dependence; different products can still describe the same underlying event. No separate category models are fit because an authoritative category field is absent.

The report `results/FINDINGS.md` is regenerated from output tables on every successful run.

## Output map

| File | Purpose |
|---|---|
| `attrition.json`, `*_schema.csv`, `run_manifest.json` | Data inventory, inclusion counts, configuration and provenance |
| `contract_features.parquet` | Contract inputs, outcome audit fields, split and cluster assignments; **not contract wedge labels** |
| `contract_price_counts.parquet` | Early-window execution sufficient statistics |
| `cluster_profiles.csv`, `clustering_diagnostics.csv` | Similarity diagnostics and titles for review |
| `wedges_text.csv`, `wedges_text_price.csv` | Group-cost labels, predictors, support, status and uncertainty |
| `exact_cost_support_*.csv` | Counts and retained/excluded direction-price cells |
| `model_comparison.csv`, `metrics_*.csv` | Baseline/regression comparisons and error-improvement intervals |
| `predictions_*.csv`, `coefficients_*.csv` | Held-out predictions and standardized coefficient analysis |
| `ranking_*.csv`, `ranking_summary_*.json` | Preliminary signal ranking and uncertainty |
| five PNG figures | Cluster quality, wedge variation, model performance, coefficients and ranking |

## Next stage

Acquire versioned question/metadata records, scheduled close histories and timestamps when outcomes became public. Rebuild a rolling chronological experiment in which all training outcomes were known before the next prediction cutoff, purging overlapping economic events across product series. Add a true event graph, category metadata and optional pinned sentence-transformer embeddings; validate clusters manually before outcomes are inspected. Test label sensitivity to uniform-contract versus execution weights, minimum-event counts, fixed exact-price support, opening-window length, cluster count and event/series bootstrap units. More contracts or a hierarchical model may improve precision, but shrinkage cannot cure missing historical provenance or poor economic grouping. Reserve a fresh test period before tuning. Only then add execution-aware maker simulations and cost/inventory constraints. A wording experiment holding the underlying event constant would be needed for causal claims.
