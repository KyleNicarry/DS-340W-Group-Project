# Data availability assessment

The repository contains real local Parquet data, not just download scripts: Kalshi markets (~570 MB), Kalshi trades (~3.3 GB), plus Polymarket markets, trades, blocks and legacy trades. The MVP uses Kalshi because its direct ticker/outcome join and complementary prices match the supplied paper. See generated schema CSVs and attrition counts for the precise extract used. No additional data were synthesized for analysis.

| Proposed field or feature | Status and MVP treatment |
|---|---|
| Contract ID and event ID | `ticker` and `event_ticker` available. Prefix-derived recurring series used only for split/composition; it is not an authoritative economic-event graph. |
| YES/NO outcomes | `result` and finalization status available; only finalized binary YES/NO included. Void/unknown outcomes excluded. |
| Settlement-publication timestamp | **Absent.** Final metadata fetch time is not settlement time. No verified chronological backtest is claimed. |
| Question/title text | `title` and side subtitles available. MVP uses title for wording features and title plus YES subtitle for clustering, preserving strikes/named alternatives. No version history, description or full rule text in this schema; assume title unchanged, with explicit leakage caveat. |
| Market category | **Absent as an authoritative column.** Existing repository ticker mappings are heuristics and are not silently treated as true categories. Output reports series composition and unavailable category. |
| Execution price and direction | `yes_price`, `no_price`, `taker_side` available; validate complementary integer-cent prices and known sides. Both position directions enter target. |
| Trading timestamps | Trade `created_time` available. Features and target-position sampling restricted to open through open+24h. |
| Trading volume / number of trades | Early execution count and summed units available from trade history. Final `volume`, `volume_24h` and `open_interest` are snapshot fields and excluded. |
| Price volatility | Early price-level dispersion available and labeled as such. Not inferred return volatility or a time-regularized volatility estimator. |
| Historical bid–ask spread | **Unavailable.** Snapshot bid/ask fields exist but do not represent the landmark; excluded. |
| Historical liquidity depth | **Unavailable.** No historical order book or queue information in the supplied schema. |
| Time remaining / scheduled duration | Opening and recorded close times exist, but no versioned scheduled-close history or settlement-publication time. Close is an eligibility check only. Remaining time and duration are excluded as predictors. |
| Question length / threshold wording | Computed transparently from title. No inferred event type labels. |
| Sentiment / framing | Small positive/negative/negation word counts supplied as explicit lexical proxies, not validated sentiment scores. Context, irony and side semantics are not modeled. |
| Emotionality / arousal | **Not constructed.** Requires a validated lexicon/model and domain validation. |
| Political / partisan language | Explicit generic political-word count provided. Partisan affiliation and bias are **not inferred**. |
| Proper nouns / persons / teams / organizations | Validated entity analysis **deferred**. Capitalized token count is supplied with a literal name and must not be interpreted as an entity count. |
| Readability / syntactic complexity | Word length and punctuation proxies only; no validated readability, parse-tree or narrative-complexity model. |
| Pretrained sentence embeddings | No pinned checkpoint/cache in the project workflow. Offline train-fitted TF-IDF/LSA alternative is used and separately ablated. Claims about sentence embeddings are deferred. |
| Fees / fills / inventory / execution constraints | **Unavailable for a credible simulation.** No trading-profit claim. |

The resolved-only population has survivorship/selection bias. Requiring more than 24 hours of life and five early executions further selects longer-lived and actively traded contracts. Excluding combination products is intentional: their underlying leg graph is needed to handle overlapping outcomes. Full-sample paper magnitudes are not replication targets for this changed population, observation window and standardized band estimand.
