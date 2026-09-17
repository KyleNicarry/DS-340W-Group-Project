# Manual review of the default run

This review refers to the default 60,000-ticker sample, 16 clusters and seed 340. Cluster IDs change when the representation/configuration changes. The evidence is the five nearest-centroid **training** title/subtitle pairs and series composition in `results/cluster_profiles.csv`, not realized outcomes. These are qualitative descriptions of inspected examples, **not inferred authoritative category fields**. Reviewing nearest-centroid examples is only a first pass: peripheral and held-out contracts require review in the full analysis.

| Text cluster | Observed representative content | Assessment |
|---|---|---|
| 0 | Temperature intervals around 82–83 degrees, including Miami | Related weather wording, but grouped partly by numeric strike across cities; city/weather dependence remains. |
| 1 | Words mentioned in political speeches/appearances | Coherent mention-contract format; speakers and occasion vary, and cannot be interpreted as a homogeneous partisan group. |
| 2 | New York City temperature intervals | Recognizable topic; repeated dates/strikes can share weather outcomes. |
| 3 | Bitcoin price above a specified threshold | Useful topic/structure group; strike and horizon differ. |
| 4 | Xfinity winner, Nobel literature and extreme temperature thresholds | **Diffuse and economically mixed**. Mean text cosine 0.484; excluded by the 0.65 training-coherence floor. |
| 5 | Chicago temperature intervals | Recognizable topic; location/season dependence still matters. |
| 6 | Bitcoin price ranges | Recognizable topic; differs structurally from above-threshold contracts. |
| 7 | Spotify top-song alternatives | Related event type; music popularity and competing outcomes induce dependence. |
| 8 | Basketball total-points thresholds | Coherent wording/market structure, although team/game heterogeneity remains. |
| 9 | COVID case counts, subway ridership, other above-threshold questions | **Only structurally similar**. Numerical compactness overstates economic similarity. Retained as an explicitly exploratory mixed group; refine or exclude before a confirmatory study. |
| 10 | Treasury-yield intervals and presidential approval intervals | **Mixed economic topics**, despite similar interval syntax. Do not interpret this cluster as one economic category. |
| 11 | Named golf/F1/tennis winners | Coherent affirmative winner wording, not a homogeneous sport or event population. |
| 12 | Ethereum price intervals | Recognizable topic; timing, strikes and repeated underlyings differ. |
| 13 | College/professional game winners | Broad sports-winner format; team/sport distributions can shift in held-out series. |
| 14 | Austin temperature intervals | Related weather topic; concentration near high-temperature ranges shows a strike/season component. |
| 15 | College basketball point spreads | Coherent point-spread format; different teams and games remain heterogeneous. |

For the price-inclusive map, clusters 2 and 13 fail the same floor (mean text cosines 0.618 and 0.570). Representatives of cluster 2 mix coaching appointments with rain questions. Cluster 13 includes extreme-temperature wording but is diffuse in the full assigned population. Remaining groups include Bitcoin thresholds/ranges (12/1), Ethereum ranges (15), mentions (6), Spotify songs (5), point spreads (9), basketball totals (11) and named winners (14). Cluster 8 spans generic numeric event counts; cluster 3 mixes several sports-winner formats. These remain broad lexical groups.

The observed price P90–P10 spans are wide; normalized mutual information with price deciles is 0.075 (text) and 0.077 (text+price). These results argue against simple price-bin reproduction. Removing the price dimension gives adjusted Rand agreement 0.640: price has a material but not dominant effect. Different group definitions produce different label supports, so raw MAEs across approaches must not be treated as a paired ablation on identical labels.

Titles alone omitted some strikes/alternatives. The pipeline therefore adds the observed YES subtitle for similarity while retaining title-only wording features. This revision and the coherence floor were introduced during exploratory development after interim outputs had been examined; the holdout is **not a preregistered confirmatory test**. The primary model and hyperparameters were not chosen to maximize held-out performance. All final specifications and the negative result are reported. A fresh holdout is required after further refinements.

The next review should sample peripheral and held-out titles, build a true event/underlying graph and distinguish semantic phrasing from economic substitutability. A validated category or taxonomy could help, but it should be acquired or explicitly human-annotated rather than silently inferred as fact.
