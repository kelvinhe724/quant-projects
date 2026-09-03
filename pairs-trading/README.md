# Pairs trading on S&P 500 stocks

Screens within-sector stock pairs for cointegration, trades the spread when its
trailing z-score gets far from zero, and evaluates the result on a window that
was never used to pick the pairs or the parameters.

The headline is a negative one. In-sample Sharpe is 3.01. Out-of-sample Sharpe
is 0.08, and it goes negative if I assume more than about 17bps of round-trip
cost. That gap is the entire point of the project: the in-sample number is what
you get when you let the same data choose the pairs and score the strategy.

## Data

Daily adjusted closes from Yahoo Finance via `yfinance`, 2015-01-02 to
2026-08-28, cached to `reports/prices.csv` so only the first run needs internet.

The universe is 182 liquid S&P 500 names, roughly 12-20 per GICS sector, not the
full 500. That is a deliberate simplification and it matters for the numbers
below: 182 names give 1,480 within-sector pairs, where the full index would give
124,750 all-pairs tests. Eight of the 190 tickers I listed dropped out for lack
of continuous history (DOW spun off in 2019, HES was acquired) or a failed
download.

`check.py` runs offline on simulated data with a planted cointegrating relation.

## Periods

| window | dates | used for |
|---|---|---|
| formation | 2015-01-02 to 2019-12-31 | cointegration screen, hedge ratios, parameter grid |
| out-of-sample | 2020-01-02 to 2026-08-28 | the headline number, nothing else |

The two do not overlap. Every pair, every hedge ratio and every threshold is
fixed before 2020-01-02 and never revisited.

## Method

Engle-Granger, in the standard two steps. Regress log price A on log price B to
get the hedge ratio, then test the residual for a unit root. I use
`statsmodels.tsa.stattools.coint` rather than feeding OLS residuals to
`adfuller`: the residuals come from an estimated cointegrating vector, so the
standard ADF critical values are wrong and would overstate significance.

The spread is `log(A) - beta*log(B) - intercept` with beta frozen from the
formation window. The signal is a trailing z-score over a rolling window; entry
when |z| exceeds the entry threshold, exit when z crosses back through the exit
level, stop out at |z| > 4, and a hard 60-day maximum holding period. Each pair
is dollar-neutral, so the daily return on gross capital is
`position * (r_A - beta*r_B) / (1 + |beta|)`. Capital is split equally across
the selected pairs; idle capital earns nothing.

Costs are 10bps round trip on gross notional, charged whenever the position
changes. That is the middle of a realistic range for large-cap US equities and
it is swept from 0 to 30bps below.

### Multiple testing

This is the part that decides whether the strategy is real.

Testing 1,480 pairs at p < 0.05 means about 74 pairs should pass on noise alone.
That is what the screen actually returns:

| threshold | pairs passing | expected by chance |
|---|---|---|
| p < 0.10 | 201 | 148.0 |
| p < 0.05 | 99 | 74.0 |
| p < 0.01 | 29 | 14.8 |
| Bonferroni, p < 3.4e-05 | 0 | 0.1 |

So 99 pairs clear the 5% bar and 74 of them are what you would get from
1,480 coin flips. The excess is 25 pairs out of 1,480 tests. At the 1% level the
excess is 14. Under a Bonferroni correction nothing survives at all, which is a
result rather than a failure: with five years of daily data and this many tests,
there is no pair in this universe whose cointegration I can assert at a 5%
family-wise error rate.

Two things keep this from being worse. Restricting candidates to within-sector
cuts the test count from 16,471 to 1,480, which both raises the prior that a
surviving pair is economically real and shrinks the Bonferroni penalty by an
order of magnitude. And `select()` refuses to reuse a stock, so 35 selected
pairs are 70 distinct names rather than one crowded trade wearing 35 hats.

The false-positive rate is verified directly in `check.py`: 200 pairs of
independent random walks return p < 0.05 6.5% of the time, and Bonferroni leaves
one of them.

### Parameter choice

Grid over the p-value threshold (0.01, 0.05, Bonferroni), the z-score lookback
(30, 60, 90), the entry threshold (1.5, 2.0, 2.5) and the exit level (0, 0.25,
0.5), scored on formation-period Sharpe net of costs. 54 of the 81 combinations
ran; the Bonferroni ones select fewer than five pairs so they were skipped.

The winner is p <= 0.05, lookback 90, entry 1.5, exit 0.0, 35 pairs. The surface
is flat, which is mildly reassuring:

| entry \ exit | 0.00 | 0.25 | 0.50 |
|---|---|---|---|
| 1.5 | 3.01 | 2.93 | 2.93 |
| 2.0 | 2.94 | 2.88 | 2.78 |
| 2.5 | 2.58 | 2.52 | 2.51 |

## Results

| | in-sample (formation) | out-of-sample |
|---|---|---|
| annual return, gross | +6.60% | +0.72% |
| annual return, net | +6.11% | +0.25% |
| annual vol | 1.98% | 3.86% |
| Sharpe, gross | 3.24 | 0.21 |
| Sharpe, net | 3.01 | 0.08 |
| max drawdown | -1.26% | -7.19% |
| trades | 809 | 1,127 |
| hit rate | 77.4% | 62.9% |
| average hold | 30 days | 34 days |
| annual turnover | 9.1x | 9.5x |

Returns are unlevered on gross notional with one capital slot per pair, so the
absolute numbers are small by construction and Sharpe is the number to read.

Out-of-sample, by year:

| year | net return | Sharpe |
|---|---|---|
| 2020 | -0.16% | 0.01 |
| 2021 | -2.73% | -0.97 |
| 2022 | +0.40% | 0.13 |
| 2023 | +4.20% | 1.53 |
| 2024 | -0.70% | -0.25 |
| 2025 | +2.47% | 0.83 |
| 2026 (to Aug) | -1.68% | -0.65 |

Four of seven years are negative. Per pair, the median out-of-sample Sharpe is
0.03 and 19 of the 35 pairs are positive, which is a coin flip. The best is
LIN/SHW at 1.00, the worst C/MET at -0.53.

Transaction costs:

| round trip | in-sample Sharpe | out-of-sample Sharpe |
|---|---|---|
| 0bps | 3.24 | 0.21 |
| 5bps | 3.13 | 0.14 |
| 10bps | 3.01 | 0.08 |
| 20bps | 2.78 | -0.04 |
| 30bps | 2.55 | -0.16 |

At 9.5x annual turnover, 10bps of round-trip cost is about 47bps a year, which
eats two thirds of the 72bps gross out-of-sample return. Breakeven is somewhere
near 17bps. In-sample the strategy is comfortably profitable at any of these costs,
which is exactly the trap: cost realism alone would not have caught this.

`reports/sample_pair.png` shows what went wrong on one pair. BDX/ISRG has the
lowest formation p-value in the whole universe (6.4e-05) and its spread sits
flat around zero from 2015 to 2019. From 2020 the spread walks steadily down to
-0.8 and never comes back. The relationship was real in-sample and gone after.

## Look-ahead audit

Places I checked, and what the code does:

- Rolling statistics. The z-score mean and standard deviation are
  `rolling(lookback)`, not full-sample. Verified in `check.py` by adding a spike
  at t=900 and asserting every z-score before t=900 is unchanged.
- Execution lag. `pair_backtest` computes the target position from data through
  t, then takes `.shift(1)`. Nothing downstream reads the unshifted series.
- Hedge ratios. Estimated once on the formation window, frozen. The
  out-of-sample spread uses a beta from 2015-2019 only.
- Screening. Run on formation log prices only.
- Warm-up. Out-of-sample signals let the trailing window fill using formation
  prices rather than starting from NaN. Those are past observations at every
  point they are used, so it is not look-ahead, but it does mean the 2020 signals
  inherit a window that straddles the split.

One genuine leak remains and I have not fixed it: the universe filter drops any
ticker without near-complete history over 2015-2026, so the 2015 formation
screen is run on a universe defined using data through 2026. See below.

## Limitations

**Survivorship bias.** `yfinance` gives current index members, and my coverage
filter then removes anything that did not trade continuously through 2026. Both
push the same way. Names that were delisted, acquired or dropped from the index
are absent, and those are disproportionately the ones whose spreads blew out and
never reverted. The bias inflates the results, and it inflates the in-sample
number more than the out-of-sample one because the formation window is further
in the past. I cannot size it without point-in-time index membership, but the
standard estimate in the literature for US equity strategies is on the order of
1-4% a year on returns; against a 0.72% gross out-of-sample return, that alone
could account for the whole thing.

**One split, no rolling re-formation.** Pairs chosen in 2019 are still being
traded in 2026 on a hedge ratio estimated seven years earlier. Real pairs books
re-form on something like a 12-month formation and 6-month trading cycle. My
setup is the stricter test of whether cointegration persists, and it answers no,
but it is not how the strategy would actually be run, so the out-of-sample
number is a lower bound rather than a fair estimate of a live book.

**Parameters were still tuned, just on the right data.** The grid picked the
best of 54 configurations on formation Sharpe. The formation number is therefore
the maximum of 54 draws and is biased upward on top of the selection bias from
choosing pairs on the same window. The out-of-sample number is clean of both.

**Costs are a flat assumption.** 10bps round trip, no market impact, no borrow
cost on the short leg, no financing, and I assume every fill happens at the next
close. Shorting the smaller names in this universe is not free and a real
implementation would pay for it.

**No risk overlay.** Equal capital per pair, no volatility targeting, no sector
or beta neutralisation at the portfolio level, no cap on concurrent positions.
The March 2020 drawdown in the equity curve is a single day and would have been
smaller with any position limit.

## Files

- `data.py` universe with sector labels, cached price download, formation/OOS split
- `pairs.py` Engle-Granger test, screening with multiple-testing controls, z-score, entry/exit rules
- `backtest.py` per-pair P&L with costs, portfolio aggregation, trade extraction, metrics
- `check.py` offline checks on synthetic data
- `run.py` full pipeline, tables and charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline. `run.py` downloads once, then reads the cache.
