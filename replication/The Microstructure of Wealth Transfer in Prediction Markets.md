---
title: "The Microstructure of Wealth Transfer in Prediction Markets"
author: Jonathan Becker
date: 2026-01-18
source_pdf: "The Microstructure of Wealth Transfer in Prediction Markets.pdf"
dataset: "Kalshi transactions, June 2021 - November 2025"
---

# The Microstructure of Wealth Transfer in Prediction Markets

## Abstract

We analyze 72.1 million transactions executed on Kalshi, a CFTC-regulated prediction market, from June 2021 through November 2025. Aggregate prices are well calibrated, but calibration conceals a persistent internal transfer. YES contracts underperform NO contracts of equivalent cost basis, especially at longshot prices; we call the premium paid by affirmative-framed order flow the **optimism tax**. Liquidity takers earn mean excess returns of -1.12% per trade while makers earn +1.12%. The gap varies from 0.17 percentage points in Finance to more than 7 points in Media and World Events, and it widened after the 2024 election as volume attracted professional liquidity providers. The forecasting performance of prediction markets therefore coexists with a market structure that monetizes behaviorally biased order flow.

**Keywords:** prediction markets, market microstructure, longshot bias, liquidity, optimism tax

## 1. Prediction markets and Kalshi

Prediction markets trade binary contracts that settle at either $1 or $0. A price of 5 cents is conventionally interpreted as a 5% probability. Because each position has an offsetting counterparty, the market is zero-sum before fees: one participant's gain is another's loss.

Kalshi launched in 2021 as a CFTC-regulated U.S. prediction market. The platform's growth accelerated around the 2024 election cycle and its expansion into politics and sports. The data end at 2025-11-25 17:00 ET; Q4 2025 is therefore incomplete.

### Data and definitions

The underlying project records market resolution, execution price, taker side, number of contracts, and execution time. Analyses use resolved YES/NO markets and exclude voided, delisted, open, and very-low-volume markets.

For a trade at price \(p_i\) cents and binary outcome \(o_i\), define the mispricing of a subset \(S\) as:

$$
\delta_S = \frac{1}{|S|}\sum_{i \in S} o_i - \frac{1}{|S|}\sum_{i \in S}\frac{p_i}{100}.
$$

Gross excess return relative to the capital at risk is:

$$
r_i = \frac{100o_i-p_i}{p_i}.
$$

To compare affirmative and negative contracts fairly, price means **cost basis**: the capital paid for the position. A 5-cent YES and a 5-cent NO each risk five cents for the same one-dollar payoff.

![Figure 1. Market calibration: actual win rate versus contract price. The diagonal is perfect calibration.](output/win_rate_by_price.png)

*Figure 1. Recreated by `src/analysis/kalshi/win_rate_by_price.py`. The close fit to the diagonal shows that aggregate calibration can coexist with economically meaningful tail errors.*

## 2. The longshot bias

The longshot bias is the tendency to overpay for low-probability outcomes. On Kalshi, contracts at 5 cents win about 4.18% of the time, below the implied 5%; contracts at 95 cents win about 95.83% of the time. The error is concentrated in the tails: contracts below 20 cents generally underperform their stated odds and high-priced contracts outperform.

![Figure 2. Mispricing by contract price for takers, makers, and the combined market.](output/mispricing_by_price.png)

*Figure 2. Recreated by `src/analysis/kalshi/mispricing_by_price.py`. The combined series masks the offsetting experience of makers and takers.*

The aggregate curve is informative about price discovery, but it does not identify who bears the longshot loss. That question is central in a zero-sum market.

## 3. The maker-taker wealth transfer

A liquidity maker posts resting liquidity; a taker executes against it. Every Kalshi trade identifies the taker and makes the maker the counterparty in the opposite direction.

| Role | Average excess return | 95% confidence interval |
| --- | ---: | --- |
| Taker | -1.12% | [-1.13%, -1.11%] |
| Maker | +1.12% | [+1.11%, +1.13%] |

At one cent, takers win roughly 0.43% of the time against a 1% implied probability, while makers on the corresponding positions win roughly 1.57%. Mispricing narrows around the middle of the price range but the role-specific pattern remains: takers have negative excess return at most price levels and makers have positive excess return at those same levels.

![Figure 3. Maker and taker excess returns by price.](output/maker_vs_taker_returns.png)

*Figure 3. Recreated by `src/analysis/kalshi/maker_vs_taker_returns.py`. It decomposes the aggregate calibration error into the two sides of each execution.*

### Is this merely payment for spread provision?

Some maker profit is compensation for posting liquidity, so a positive maker return alone does not establish exploitation of biased flow. Two results indicate that spread capture is not the whole explanation. First, maker returns differ slightly by position direction: makers buying YES earn about +0.77 percentage points and makers buying NO about +1.25 points. The difference is small in standardized terms (Cohen's d around 0.02-0.03), however. Second, the maker-taker gap differs sharply across categories, which simple, uniform spread compensation does not explain well.

![Figure 4. Maker returns by position direction.](output/maker_returns_by_direction.png)

*Figure 4. Recreated by `src/analysis/kalshi/maker_returns_by_direction.py`.*

## 4. Category variation

The return gap is smallest where participants are likely to use probabilistic, financially disciplined reasoning and widest where questions invite narrative or emotionally engaged trading.

| Category | Taker return | Maker return | Gap | Trades |
| --- | ---: | ---: | ---: | ---: |
| Sports | -1.11% | +1.12% | 2.23 pp | 43.6M |
| Politics | -0.51% | +0.51% | 1.02 pp | 4.9M |
| Crypto | -1.34% | +1.34% | 2.69 pp | 6.7M |
| Finance | -0.08% | +0.08% | 0.17 pp | 4.4M |
| Weather | -1.29% | +1.29% | 2.57 pp | 4.4M |
| Entertainment | -2.40% | +2.40% | 4.79 pp | 1.5M |
| Media | -3.64% | +3.64% | 7.28 pp | 0.6M |
| World Events | -3.66% | +3.66% | 7.32 pp | 0.2M |

![Figure 5. Maker and taker returns by category.](output/maker_taker_returns_by_category.png)

*Figure 5. Recreated by `src/analysis/kalshi/maker_taker_returns_by_category.py`. The figure uses the repository's category grouping and volume-weighted returns.*

Finance is a useful near-efficiency benchmark. Its questions tend to draw participants used to probabilities and expected value. Sports, entertainment, media, and world events admit more narrative-based participation, and their larger gaps are consistent with an optimism premium rather than an equal, platform-wide cost of liquidity.

![Figure 6. Distribution of market types by notional volume.](output/market_types.png)

*Figure 6. Recreated by `src/analysis/kalshi/market_types.py`.*

## 5. Evolution over time

The maker-taker gap was not constant. In the early period, takers earned positive excess returns and makers lost money. The pattern reversed in 2024 Q2 and widened after the 2024 election. From launch through 2023, takers averaged about +2.0% and makers about -2.0%; after the election-driven growth period, the opposite pattern dominates.

![Figure 7. Quarterly maker and taker returns, with notional volume.](output/maker_taker_gap_over_time.png)

*Figure 7. Recreated by `src/analysis/kalshi/maker_taker_gap_over_time.py`. The script marks the 2024 election/legal-victory period and overlays notional volume.*

Pre-election the mean gap was about -2.9 percentage points (takers winning); post-election it was about +2.5 points (makers winning), a swing of 5.3 points. Volume rose from roughly $30 million in 2024 Q3 to $820 million in 2024 Q4. This is consistent with an influx of professional liquidity providers once market depth made the activity worthwhile.

The composition of taker flow did not move toward longshots. Longshot contracts (1-20 cents) remained about 4.8% of taker volume before the election and 4.6% afterward. Instead, volume shifted away from the 91-99 cent bucket and into middle prices. Thus growing taker losses are not explained by a simple increase in longshot demand.

![Figure 8. Taker longshot-volume share by quarter.](output/longshot_volume_share_over_time.png)

*Figure 8. Recreated by `src/analysis/kalshi/longshot_volume_share_over_time.py`.*

![Figure 9. Quarterly Kalshi notional volume.](output/volume_over_time.png)

*Figure 9. Recreated by `src/analysis/kalshi/volume_over_time.py`.*

## 6. The YES/NO asymmetry

The maker-taker decomposition identifies the side that captures the transfer, but not the source of the biased flow. The cost-basis comparison exposes a directional asymmetry. At a one-cent cost basis, a YES contract has a historical expected value of about -41%, while an equivalent-cost NO contract has an expected value near +23%: a difference of roughly 64 percentage points.

NO outperforms YES at 69 of 99 price levels. Its advantage is especially clear from 1-10 cents and from 91-99 cents. Dollar-weighted returns are approximately -1.02% for YES buyers and +0.83% for NO buyers, a 1.85-point gap.

![Figure 10. Expected value of YES and NO contracts at the same cost basis.](output/ev_yes_vs_no.png)

*Figure 10. Recreated by `src/analysis/kalshi/ev_yes_vs_no.py`. It calculates expected value as `100 * win_rate - price` for both directions.*

### Takers prefer affirmative bets

Order flow is directionally tilted. In the 1-10 cent range, where YES is the longshot, takers provide roughly 41-47% of YES volume while makers provide only 20-24%. At the other end, when NO is the longshot, makers purchase NO at a much higher share than takers. The evidence is consistent with an affirmative-framing preference: takers pay a premium to buy hope-laden YES longshots and makers take the counterposition.

![Figure 11. YES/NO volume by price, split by taker and maker.](output/yes_vs_no_by_price.png)

*Figure 11. Recreated by `src/analysis/kalshi/yes_vs_no_by_price.py`.*

This pattern is the **optimism tax**. It does not require makers to be better forecasters in a direction-specific sense. Maker returns are nearly symmetric across YES and NO positions; the structural advantage comes from accommodating a population that arrives at the book with biased demand.

## 7. Discussion and limitations

The results distinguish market-level accuracy from participant-level welfare. Prediction-market prices can closely track frequencies while a systematic transfer occurs inside the market. In low-volume early years, less sophisticated makers may have lost to informed takers. As volume rose, professional makers could harvest the spread and the biased component of taker flow more reliably.

The interpretation has limits. Unique trader identifiers are unavailable, so maker/taker is a proxy rather than a direct label for sophistication. Historical trade records also do not provide full contemporaneous order-book spreads, making a strict separation of spread compensation and behavioral extraction impossible. Finally, these results concern a regulated U.S. venue and may not transfer unchanged to offshore markets with different fees, leverage, or participant pools.

## 8. Conclusion

Kalshi's aggregate probabilities are broadly calibrated, but that accuracy conceals a persistent transfer from liquidity takers to liquidity makers. Takers underperform, especially when buying affirmative longshots; makers capture the offsetting return. The gap varies systematically with category and appeared most strongly after the platform became deep enough to attract professional liquidity provision.

Prediction markets therefore should not be interpreted solely as impersonal aggregators of belief. Their prices are also outcomes of market structure, participant selection, and the demand to buy optimistic narratives.

## Figure regeneration

This Markdown intentionally links to generated figure files rather than embedding copies from the PDF. The repository's analysis code is the figure source of record. After installing the project dependencies and downloading the dataset described in [README.md](README.md), run the following analyses from the repository root:

```sh
make setup
make run win_rate_by_price
make run mispricing_by_price
make run maker_vs_taker_returns
make run maker_returns_by_direction
make run maker_taker_returns_by_category
make run market_types
make run maker_taker_gap_over_time
make run longshot_volume_share_over_time
make run volume_over_time
make run ev_yes_vs_no
make run yes_vs_no_by_price
```

Each command writes a PNG into `output/` with the filename used above. The analyses also emit CSV, JSON, and PDF versions, allowing the numerical claims and visuals to be audited independently.

## References

- Becker, J. (2026). *The Microstructure of Wealth Transfer in Prediction Markets*.
- Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*.
- Griffith, R. M. (1949). Odds Adjustments by American Horse-Race Bettors. *American Journal of Psychology*.
- Reichenbach, F., & Walther, M. (2025). Exploring Decentralized Prediction Markets: Accuracy, Skill, and Bias on Polymarket.
- Thaler, R. H., & Ziemba, W. T. (1988). Anomalies: Parimutuel Betting Markets: Racetracks and Lotteries. *Journal of Economic Perspectives*.
- Whelan, K. (2025). Agreeing to Disagree: The Economics of Betting Exchanges.
