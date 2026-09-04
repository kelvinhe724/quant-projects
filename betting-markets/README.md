# Are bookmaker odds efficient?

37,725 football matches from six European leagues, 2010-11 to 2025-26, with up to
eleven bookmakers quoting each one. I measure the margin, test the classic
favourite-longshot bias, check whether closing prices forecast better than
opening prices, and try to find a betting rule that survives the vig.

Four findings, in descending order of how confident I am:

1. **The margin is real and it is most of the story.** 2.7% at Pinnacle, 4.4-7.6%
   at retail books, and it has compressed at the retail end over fifteen years
   while Pinnacle's has widened from 2.2% to 3.8%.
2. **Closing lines beat opening lines**, by 0.00154 of Brier score, t = -5.89.
   That is the textbook efficiency result and it holds in 13 of 14 seasons.
3. **The favourite-longshot bias mostly is not there once you remove the vig
   properly.** Backing 10-to-20 shots loses 17.3% and backing odds-on favourites
   breaks even (+0.07%),
   which looks like a huge bias, but a regression of profit on log odds goes from
   t = -3.43 at quoted prices to t = -1.36 at de-vigged prices. Most of the
   gradient is the margin being fatter on longshots, not the book being wrong.
4. **Nothing survives.** The one rule that made money made it by assuming I could
   always take the best price across every bookmaker. Take the identical
   selections at the average price instead and the return is **-6.1%**.

The de-vigging choice is not a detail here. It flips finding 3 on and off.

## Data

`football-data.co.uk` publishes a free CSV per league-season with results and the
1X2 odds from every bookmaker it tracks. No key, no auth, no rate limit. I pulled
96 league-seasons directly and cached them.

| | |
|---|---|
| leagues | E0 Premier League, E1 Championship, D1 Bundesliga, SP1 La Liga, I1 Serie A, F1 Ligue 1 |
| seasons | 2010-11 to 2025-26, sixteen of them |
| matches | 37,725 |
| quotes | 464,832 (match x bookmaker x phase) |
| dates | 2010-08-06 to 2026-05-24 |

Each file carries two prices per bookmaker. Columns like `B365H` are collected
mid-week; columns like `B365CH` are collected shortly before kickoff. I call
these open and close throughout, but **neither is a true market open or a true
close**. The mid-week price is already several days into trading, and the
"closing" price is a snapshot the site took, not the last tick. This
understates closing line value, since I am comparing two interior points of the
price path rather than the endpoints.

Coverage differs by book, which constrains every comparison below:

| book | opening quotes | closing quotes | first | last |
|---|---|---|---|---|
| Bet365 | 37,715 | 16,322 | 2010-11 | 2025-26 |
| Pinnacle | 31,797 | 31,821 | 2012-13 | 2025-26 |
| Bwin | 36,811 | 15,405 | 2010-11 | 2025-26 |
| William Hill | 34,883 | 13,495 | 2010-11 | 2024-25 |
| VC Bet | 33,094 | 11,715 | 2010-11 | 2023-24 |
| Interwetten | 31,939 | 10,556 | 2010-11 | 2023-24 |
| Ladbrokes | 20,752 | 1,729 | 2010-11 | 2025-26 |

Closing odds for most books only start in 2019-20. Pinnacle has them from
2012-13, which is why Pinnacle carries the headline results: it is both the
longest closing series and the sharpest book in the file.

Two more columns matter. `Max` is the best price any tracked book offered, and
`Avg` is their mean. **Neither is a price you can bet.** `Max` in particular is
the maximum over quotes collected at different moments from a changing set of
books, so it is a best-of-many-snapshots number, not a screen you could have had
open. Almost every apparent edge in this project traces back to that column, and
section 5 is mostly about proving it.

`check.py` runs entirely offline on simulated books.

## The vig

Sum the three implied probabilities `1/odds` and subtract one.

| book | opening | closing |
|---|---|---|
| Pinnacle | 2.69% | 2.49% |
| VC Bet | 4.38% | 5.55% |
| Bet365 | 5.23% | 5.66% |
| Bwin | 6.17% | 5.57% |
| Interwetten | 7.57% | 5.53% |
| William Hill | 6.15% | 6.20% |
| Ladbrokes | 6.69% | 6.49% |
| consensus (`Avg`) | 5.49% | 4.89% |
| best-of-books (`Max`) | 0.42% | -0.13% |

Pinnacle runs at less than half the margin of anyone else. That is their stated
business model: low margin, high volume, and they take sharp action instead of
refusing it. The rest cluster at 5-7%.

Over time (opening prices, `reports/overround_by_season.png`):

| season | Bet365 | Bwin | Interwetten | Pinnacle | William Hill |
|---|---|---|---|---|---|
| 2010-11 | 6.37 | 8.72 | 11.00 | | 6.66 |
| 2013-14 | 4.98 | 6.94 | 8.81 | 2.19 | 6.40 |
| 2016-17 | 4.24 | 5.77 | 7.46 | 2.21 | 5.04 |
| 2019-20 | 5.37 | 5.31 | 5.57 | 3.19 | 5.32 |
| 2022-23 | 5.37 | 5.87 | 5.66 | 2.68 | 6.44 |
| 2025-26 | 6.07 | 5.83 | | 3.82 | |

Convergence, not a race to the bottom. Interwetten came down from 11.0% to 5.4%
and Bwin from 8.7% to 5.8%, but Bet365 bottomed at 4.2% in 2015-16 and has drifted
back up to 6.1%. Pinnacle has gone from 2.25% in 2012-13, and a low of 2.19%,
to 3.82%. The spread
across books narrowed from about 7 points to about 2.

By league the margin is flat: 4.83% in the Premier League up to 5.58% in the
Championship, across all named books. Lower-division English football is the most
expensive market in the sample, which is what you would expect if margin tracks
how hard the book finds it to price.

## De-vigging, and why the method decides the answer

The three quoted probabilities sum to about 1.05. To ask whether the book is
*right*, I need three numbers that sum to 1, and there is no way to observe which
ones the book actually believes. Three standard answers:

- **Proportional.** Divide each by the total. Every outcome surrenders the same
  proportion of its probability. Assumes the margin is spread evenly.
- **Logarithmic (power).** Find `k` with `sum(q_i^k) = 1`. Because `q < 1`,
  raising to `k > 1` shrinks small probabilities proportionally harder, so the
  longshot gives up more.
- **Shin.** Model the margin as the book's protection against a fraction `z` of
  bettors who know the outcome. Solve for the `z` that makes the fair
  probabilities sum to 1. Also shades the longshot down, on an explicit economic
  story rather than a curve-fit.

On Pinnacle closing lines the estimated insider share is small and stable:
median `z` = 0.0117, mean 0.0125, 95th percentile 0.0173. The three methods then
disagree by this much:

| method | mean favourite prob | mean longshot prob | favourite vs proportional | longshot vs proportional |
|---|---|---|---|---|
| proportional | 0.5174 | 0.2170 | | |
| power | 0.5224 | 0.2144 | +0.50pp | -0.26pp |
| Shin | 0.5209 | 0.2149 | +0.35pp | -0.22pp |

A third of a percentage point sounds negligible. It is not, because the effect
being tested is the same size. Here is the calibration regression, a logistic fit
of the realised outcome on `logit(p)`. Slope 1 means calibrated; slope above 1
means longshots are quoted at probabilities they do not deserve:

| line | de-vig | slope | se | t vs 1 | p |
|---|---|---|---|---|---|
| Pinnacle close | proportional | 1.0516 | 0.0140 | 3.69 | **0.0002** |
| Pinnacle close | power | 1.0148 | 0.0136 | 1.09 | 0.28 |
| Pinnacle close | Shin | 1.0256 | 0.0137 | 1.87 | 0.062 |
| Bet365 close | proportional | 1.0524 | 0.0198 | 2.65 | **0.008** |
| Bet365 close | power | 0.9759 | 0.0186 | -1.30 | 0.20 |
| Bet365 close | Shin | 0.9994 | 0.0189 | -0.03 | 0.97 |
| consensus close | proportional | 1.0593 | 0.0200 | 2.96 | **0.003** |
| consensus close | power | 0.9945 | 0.0190 | -0.29 | 0.77 |
| consensus close | Shin | 1.0140 | 0.0193 | 0.73 | 0.47 |

Same odds, same results, opposite conclusions. Proportional de-vigging finds a
significant favourite-longshot bias in all three lines. Power finds nothing
anywhere. Shin lands in between and never clears 5%.

This is close to circular, and I want to be plain about it. Power and Shin both
*assume* the margin is loaded onto longshots. Proportional assumes it is not.
The favourite-longshot bias is exactly the claim that longshot prices are too
short. So choosing power or Shin means assuming a large part of the effect into
the margin and then reporting that the residual is small. Choosing proportional
means assuming none of it is margin and reporting that all of it is bias.
Neither is neutral, and I cannot break the tie from odds data alone.

Standard errors are clustered by match. The three outcomes of one match are
mechanically dependent, and on simulated data the unclustered standard error runs
about 25% too small, which is enough to turn the Shin result significant when it
is not. `check.py` asserts the clustered error is wider.

## Favourite-longshot bias

The de-vig-free version of the question: bet one unit on every outcome inside an
odds bucket and count the money. Pinnacle closing prices, 95,463 outcome-bets.

| odds bucket | bets | hit rate | de-vigged prob | ROI at quoted odds | ROI at Shin fair odds |
|---|---|---|---|---|---|
| 1.00-1.50 | 5,391 | 75.9% | 74.7% | +0.07% | +1.63% |
| 1.50-2.00 | 10,501 | 56.5% | 56.2% | -1.25% | +0.49% |
| 2.00-2.75 | 15,701 | 41.9% | 41.8% | -2.08% | -0.07% |
| 2.75-4.00 | 37,861 | 28.9% | 28.8% | -2.58% | +0.06% |
| 4.00-6.00 | 16,199 | 20.0% | 20.6% | -6.29% | -2.74% |
| 6.00-10.00 | 6,855 | 12.9% | 12.9% | -5.91% | -0.40% |
| 10.00-20.00 | 2,448 | 6.5% | 7.1% | **-17.30%** | -8.17% |
| 20.00+ | 507 | 3.6% | 3.2% | -13.68% | +3.87% |

The quoted-odds column is a textbook favourite-longshot picture: near breakeven on
heavy favourites, catastrophic on 10-to-1 shots. Formally, regressing per-bet
profit on log odds:

| prices | slope on log odds | se | t | p |
|---|---|---|---|---|
| at quoted odds | -0.0547 | 0.0159 | -3.43 | **0.0006** |
| at Shin fair odds | -0.0242 | 0.0178 | -1.36 | 0.17 |

So: the bias is unambiguous in money, and about 56% of it is the margin's shape.
The residual after removing the margin points the right way and is not
statistically distinguishable from zero at this sample size. The honest summary
is that a bettor absolutely does lose more on longshots, but mostly because he is
charged more there, not because the bookmaker misjudges longshots.

The one bucket that resists this is 10.00-20.00, still at -8.17% after de-vigging
on 2,448 bets. I do not want to lean on it. It is one bucket out of eight, chosen
after seeing the table, and the 20.00+ bucket immediately above it goes the other
way at +3.87%.

Brier score for Pinnacle closing, Shin-de-vigged: **0.5835**. A book that quoted
1/3-1/3-1/3 on every match would score 0.6667. `reports/calibration.png` shows
realised frequency against de-vigged probability with 95% Wilson intervals; every
bin is within 2.3 sigma and the largest single miss is +2.0pp in the 0.70-0.80
bin.

## Closing line value

The classic efficiency test. If the market aggregates information as it trades,
the last price should forecast better than an earlier one. Same matches, both
prices de-vigged with Shin, paired t-test across matches:

| book | matches | Brier open | Brier close | gain | t | p |
|---|---|---|---|---|---|---|
| Pinnacle | 31,783 | 0.58517 | 0.58363 | 0.00154 | -5.89 | 4e-9 |
| Bet365 | 16,318 | 0.58936 | 0.58787 | 0.00149 | -3.79 | 0.0002 |
| consensus | 16,319 | 0.58923 | 0.58780 | 0.00143 | -4.01 | 0.0001 |

Log loss agrees: Pinnacle 0.98215 to 0.97985, p < 1e-5. The result holds in 13 of
14 Pinnacle seasons, the exception being 2025-26 at -0.00094.

The effect is tiny. A 0.26% relative improvement in Brier is not a number you can
trade. But the sign is consistent, it replicates across three independent price
series, and it is what market efficiency predicts: information arriving between
Tuesday and kickoff gets into the price.

The other half of the same test: on 15,178 matches where both quote a closing
price, Pinnacle's Brier is 0.58685 and Bet365's is 0.58701. Pinnacle is better by
0.00016, p = 0.29. **The sharp book's closing line is not measurably more accurate
than the retail book's closing line.** Pinnacle's advantage is the 2.5% margin
against Bet365's 5.7%, not superior forecasting. For a bettor that is the whole
ballgame, and it is a cleaner statement of what "sharp" means than the one usually
offered.

## Does anything beat the vig?

Flat stakes, one unit per match, closing prices. "ROI no vig" prices the same bet
at the Shin fair odds, which isolates whether the selection had any edge before
the margin.

| rule | bets | mean odds | hit rate | ROI | t | ROI no vig |
|---|---|---|---|---|---|---|
| back the favourite (Pinnacle) | 31,821 | 1.99 | 52.4% | -1.49% | -2.68 | +0.35% |
| back the longshot (Pinnacle) | 31,821 | 5.56 | 21.3% | -5.05% | -4.39 | -1.01% |
| back the draw (Pinnacle) | 31,821 | 4.06 | 25.5% | -3.45% | -3.59 | -0.31% |
| back the home side (Pinnacle) | 31,821 | 2.79 | 44.2% | -2.95% | -3.99 | -0.62% |
| back the favourite (Bet365) | 16,322 | 1.93 | 52.4% | -3.76% | -4.97 | +0.34% |
| back the longshot (Bet365) | 16,322 | 5.02 | 21.5% | -8.64% | -5.73 | -0.14% |
| back the favourite (`Max`) | 16,306 | 2.04 | 52.3% | +1.18% | 1.48 | +0.90% |
| back the longshot (`Max`) | 16,306 | 5.62 | 21.5% | -0.69% | -0.41 | -0.63% |

Every rule at a real book loses, and loses roughly the margin. The `Max` rows go
positive, which is the whole illusion in one place: the identical strategy flips
from -3.76% to +1.18% purely by assuming the best available fill.

Shopping for the outlier price, closing lines:

| rule | bets | ROI | t |
|---|---|---|---|
| best price beats consensus by >0% | 25,473 | +0.39% | 0.31 |
| best price beats consensus by >2% | 13,658 | -0.39% | -0.20 |
| best price beats consensus by >5% | 5,995 | +3.47% | 0.99 |
| best price beats consensus by >10% | 1,719 | +5.31% | 0.62 |
| best price beats de-vigged Pinnacle | 22,608 | +1.27% | 1.01 |
| consensus price beats de-vigged Pinnacle | 488 | -1.77% | -0.19 |
| **same selections, filled at consensus price** | 22,608 | **-6.06%** | **-5.33** |

That last row is the control and it settles the question. Take the 22,608 bets
the value rule picked, change nothing except the fill from `Max` to `Avg`, and
+1.27% becomes -6.06% at t = -5.33. There is no information in the selection. The
entire apparent edge is the best-price assumption, and worse than that: selecting
on the maximum of a noisy set of quotes systematically picks the quotes that are
too long, so the selection is *adversely* chosen against any realistic fill.

Kelly staking, sizing off the de-vigged Pinnacle line and betting `Max`:

| rule | bets | turnover | ROI | t | log growth |
|---|---|---|---|---|---|
| full Kelly, best price | 21,258 | 341.3 | +2.90% | 2.37 | 1.317 |
| half Kelly, best price | 21,258 | 170.6 | +2.90% | 2.37 | **2.786** |
| quarter Kelly, best price | 21,258 | 85.3 | +2.90% | 2.37 | 1.930 |
| full Kelly, consensus price | 368 | 4.2 | +3.59% | 0.23 | -0.030 |
| half Kelly, consensus price | 368 | 2.1 | +3.59% | 0.23 | 0.027 |

Half Kelly outgrows full Kelly, which is diagnostic rather than encouraging: it
means the edges are overestimated, so the Kelly fraction computed from them
overbets. At the consensus price the rule fires 368 times instead of 21,258 and
goes nowhere. Same conclusion as the flat control.

So the answer is no. The best I found is +2.9% on turnover at t = 2.37, and it
requires a fill that does not exist, on a rule whose selections lose 6% at any
fill you can actually get. Against a 2.5-5.7% margin, an edge has to be large
before it is even visible, and nothing here is.

## Arbitrage across books

If the best price on each outcome sums below 1, split stakes in proportion and
you are paid whatever happens.

| source | matches | arb rate | mean return when it exists | max | median book sum |
|---|---|---|---|---|---|
| `Max` column, all tracked books | 16,306 | **52.7%** | 1.38% | 10.8% | 0.9990 |
| best of the seven named books, 4+ quoting | 14,333 | **5.92%** | 1.17% | 16.0% | 1.0202 |

A 52.7% arb rate is not credible, and the gap between the rows is the point.
`Max` is the maximum over every book the site tracks, collected at different
moments, so its median book sum of 0.9990 is an artifact of maximising over noise.
Restricting to seven books I can name, requiring at least four of them to have
quoted, gives 5.92%, and those closing snapshots are still not simultaneous.
Treat 5.92% as an upper bound on what was really there.

By season, on the named books:

| season | matches | arb rate | mean return |
|---|---|---|---|
| 2019-20 | 2,277 | 11.42% | 1.42% |
| 2020-21 | 2,378 | 5.80% | 0.80% |
| 2021-22 | 2,378 | 11.14% | 1.15% |
| 2022-23 | 2,378 | 4.37% | 1.15% |
| 2023-24 | 2,304 | 1.30% | 0.87% |
| 2024-25 | 1,452 | 0.34% | 1.92% |
| 2025-26 | 1,166 | 3.95% | 1.15% |

Cross-book arbitrage has largely closed: 11.4% of matches in 2019-20 down to 0.3%
in 2024-25. Some of that is genuine convergence, visible in the margin table too,
and some is that fewer books are in the later files (median 6 books quoting,
falling to 4 by 2024-25), which mechanically reduces the chance of finding a
disagreement. I cannot separate the two.

Even at 1.17% per arb, this is not a business. It needs funded accounts at every
book, simultaneous execution before the price moves, stakes large enough to matter
against limits that are small precisely on the mispriced side, and it ends with
the account closed.

## Limitations

**The open/close labels are not opens and closes.** The mid-week price is already
days into trading and the "closing" price is a snapshot, not the last tick. Both
compress the measured closing line value, so 0.00154 of Brier is a floor rather
than an estimate.

**The de-vigging choice is not identified.** Section 3 is the honest statement of
this: the FLB result is significant under one defensible method and absent under
two others, and the methods differ precisely in how much of the effect they
assume into the margin. Settling it needs something odds data cannot give me,
such as bet-level volume or an independent forecast to anchor the fair
probabilities.

**`Max` and `Avg` are not tradeable, and I use them anyway.** Every positive
number in this project comes from `Max`. They are best-of and mean-of quotes
collected at unknown, non-simultaneous times from a book set that changes across
seasons. I have controlled for this where it decides a conclusion (the fill
control, the named-book arb) but the arbitrage section in particular would need
timestamped simultaneous quotes to be a real measurement.

**No closing odds before 2019-20 for most books.** Pinnacle carries the closing
line result nearly alone from 2012-13, and the 2010-11 to 2011-12 seasons
contribute only to the margin analysis. The book set also changes underneath the
time series: Interwetten and VC Bet vanish after 2023-24, Ladbrokes has a
seven-season hole. Some of what looks like margin convergence is composition.

**Multiple testing.** I ran three de-vig methods on three price series, eight odds
buckets, four flat rules on three price sources, four value thresholds and three
Kelly fractions. Roughly seventy tests. At 5% that is three or four significant
results from noise alone, and the +2.9% Kelly number at t = 2.37 is exactly the
size and significance you would expect the best of seventy draws to reach. I have
applied no correction; the reason I am comfortable calling it a null result is the
fill control, not the p-value.

**Selection on the maximum is biased, not just noisy.** Choosing bets where one
book's price exceeds a consensus preferentially selects quotes that are stale,
mistaken, or about to be pulled. The -6.06% fill control measures this directly
and it is much larger than the apparent edge.

**No stake limits, no account closure, no commission, no timing.** A book that
offers 15.00 on a longshot offers it in fifty-pound units and stops offering it to
anyone who wins. Nothing here models that.

## Files

- `data.py` download and cache the league-season CSVs, reshape to one row per match, book and phase
- `odds.py` odds to probability, three de-vig methods, Shin forward map, calibration, Brier, log loss
- `strategy.py` betting rules, Kelly staking, arbitrage detection
- `check.py` 48 offline checks on simulated books
- `run.py` full pipeline, tables and charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline. `run.py` downloads 96 CSVs on the first run (about 100
seconds), caches them under `data/`, and reads the cache after that.
