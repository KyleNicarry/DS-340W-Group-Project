# Structured contract clustering

See the [methodology walkthrough](METHODOLOGY.md)
for the full rationale, fitting and assignment procedures, saved-run counts,
manual review protocol and limitations.

This standalone workflow reads **only CSV market metadata** from
`data/kalshi_recent_year_csv/split/{train,validation,test}/markets.csv`.
It preserves those splits, fits on train, and assigns validation/test using frozen
centroids. It does not scan the Parquets or the 65-million-row trade CSV.

```bash
.venv/bin/python -m src.analysis.kalshi.clustering.pipeline
# Matched baseline: same economic families and rejection rules, different algorithm.
.venv/bin/python -m src.analysis.kalshi.clustering.pipeline --method kmeans --output output/clustering_kmeans
```

## Representation and grouping

1. Exclude combination products, nonbinary records and missing titles/event IDs.
   Outcomes, final prices, volume and activity do not enter selection or similarity.
2. Extract heuristic domain, underlying and proposition structure. Preserve raw
   numbers, original title/subtitle, event ID, and a number-masked template for review.
   Templates are exported, not currently embedded. Numbers are literal tokens,
   not verified bounds or normalized economic distances. Unknown fields stay unknown.
3. Constrain grouping to domain/underlying/structure families. Weather retains
   station-series identity; sports distinguishes leagues and proposition mechanisms.
   Unrecognized subjects fall back to series identity. These are conservative
   heuristics, not authoritative taxonomy or validated named-entity extraction.
4. Default offline representation: training-fitted word/bigram TF-IDF and up to
   64 normalized LSA dimensions. Deterministic training sample capped at 20,000.
5. Fit average-linkage cosine hierarchies separately in each family, with distance
   cutoff 0.35. At most 1,000 deterministic training contracts per family enter
   quadratic linkage. All training contracts then contribute to support checks.
6. Assign each contract to its closest frozen family centroid, requiring cosine
   similarity >= 0.75, at least 20 training contracts, and 10 distinct training
   events. Weak matches, unsupported clusters and unseen families stay unassigned.
   Assignment is a centroid approximation to the training hierarchy; original leaf
   membership can change. Unsupported nearest clusters are not replaced by a more
   distant supported cluster.

Counts/thresholds are initial settings, not validated optimal choices. Unlike the
MVP's fixed 16 global groups, this produces variable-resolution groups constrained
by economic identity. No claim of improved mispricing prediction follows yet.

## Sentence embeddings

Install the optional dependency in `requirements-embeddings.txt` and provide a
local Sentence Transformers checkpoint (including tokenizer/config/weights):

```bash
.venv/bin/python -m pip install -r src/analysis/kalshi/clustering/requirements-embeddings.txt
.venv/bin/python -m src.analysis.kalshi.clustering.pipeline \
  --backend sentence --model /absolute/path/to/checkpoint \
  --output output/clustering_sentence
```

This backend embeds original title + YES subtitle, normalizes embeddings, and
uses the same grouping rules. It never silently falls back to LSA or downloads
weights. Pin and record the checkpoint revision when acquiring weights. General
embeddings do not guarantee sensitivity to negation, thresholds or settlement
rules; explicit family boundaries remain necessary.

## Review and downstream analysis

Generated files live under ignored `output/clustering/`:

- `assignments.csv`: IDs, splits, structured attributes, group, cosine and rejection status.
- `cluster_profiles.csv`: split-specific contract/event/series support and coherence.
- `review_examples.csv`: central, peripheral and deterministic random members in all splits.
- `coverage.csv`: assigned/rejected counts by split.
- `run_manifest.json`: parameters, input file metadata, attrition and completion status.

Join assignments to the CSV trades using ticker for later mispricing estimation.
Keep matched-price targets and event-block uncertainty; a thousand contracts from
one event are not a thousand independent outcomes. No wedge estimator is changed
in this initial clustering workflow. Do not interpret cosine as confidence of
mispricing or threshold similarity as logical equivalence.

Inspect training/validation examples and compare support, rejection coverage and
stability across settings before freezing a design. Reserve test outcomes for the
final downstream evaluation. The supplied splits separate events, **not recurring
series or chronology**. Distinct series/events can still share underlying outcomes;
an authoritative dependence graph, historical rules, normalized thresholds,
chronological evaluation and hierarchical partial pooling remain future work.

## Inspect contract titles interactively

Generate an offline browser containing **all assignments**, including rejected
contracts (about 60 MB for the current run; allow a moment to load):

```bash
.venv/bin/python -m src.analysis.kalshi.clustering.viewer
```

Open `output/clustering/title-review.html` in your browser. No server, external
resources or model downloads are needed. To review the k-means run, pass
`--assignments output/clustering_kmeans/assignments.csv --output output/clustering_kmeans/title-review.html`.

Recommended review sequence:

1. Start with train and assigned contracts. Review weakest-mean-similarity groups
   first. Search group names to focus on a domain, or search titles/subtitles.
2. Read peripheral members, then central members and a reshuffled random sample.
   Keep YES subtitles visible: a title alone often omits the selected alternative.
3. Use distinct title/subtitle display to reduce repetition. Switch to all
   contracts when checking event repetition and exact assignments.
4. Record whether each group is coherent, mixed, needs splitting or might merge.
   Notes should name concrete misplaced titles and the economic distinction.
5. Save each group review explicitly before switching groups. Export reviews to
   JSON for durable storage; browser storage availability varies for local files.
6. Inspect unassigned contracts for useful families being discarded. Then examine
   validation titles for transfer. Keep test outcomes untouched while refining.

The viewer deliberately emphasizes readable titles over a 2D embedding plot.
Displayed support and similarity describe the current search/split filter.
Group reviews are keyed to cluster identity and the exact assignments dataset,
not to individual filter settings. Reviews do not change assignments. Applying
reviewed constraints requires an explicit subsequent pipeline revision.

## Cluster-level mispricing results

The [mispricing analysis](mispricing/README.md) reruns the original paper's
same-cost YES/NO comparison on the recent-year CSVs, with matched pooled baselines,
event/market uncertainty, and checks of variation within structural families.
See its [findings report](mispricing/FINDINGS.md) for the completed results.
