# Dispersion: implied vs realised correlation on SPY and its ten largest names

Index options price a correlation between the constituents. Realised correlation
is usually lower. A dispersion book sells the index straddle, buys the
constituents' straddles, delta hedges everything, and collects the difference.
It is short correlation, which is a polite way of saying it is short crashes.

This project does three things with free data:

1. Backs out ATM implied vols from live option chains with my own Black-Scholes
   solver and turns them into an implied correlation for a ten-name basket.
2. Builds an eleven-year daily history of the implied-minus-realised correlation
   gap. Single-name option history is not free, so the implied side is a proxy
   built from VIX and trailing realised vols. The proxy is stated in full below,
   and the results section says how much of the answer it decides.
3. Simulates the monthly delta-hedged book with costs and reports the
   distribution, the tail, and which of its numbers the data actually pins down.

The short version: today's chains say implied correlation is about 0.03 against
a realised 0.08 on the same basket (Cboe's own COR1M printed 9.5 the same day).
Historically, the sign of the trade's P&L depends on two constants I cannot
observe, and the calibration table shows the full-sample net Sharpe swinging
from +0.91 to -1.19 as those constants move across plausible values. What the
data does pin down is the shape: a -0.76 correlation between monthly P&L and the
correlation surprise, skew of -0.9, and the worst months on record being April
2020 and May 2025.

## Data

- Daily adjusted closes for AAPL, MSFT, NVDA, AMZN, META, GOOGL, BRK-B, JPM,
  XOM, UNH, SPY and VIX, 2015-01-02 to 2026-09-02, from yfinance.
- Option chains for SPY and all ten names, every listed expiry between 14 and
  75 days out, snapshot taken after the 2026-09-03 close (03:36 UTC on the 4th).
  Five expiries per name, seven for SPY. Bid and ask only; last trades are not
  used.
- Basket weights are market caps normalised within the basket (NVDA 21.5%,
  AAPL 18.6%, GOOGL 16.3%, MSFT 14.7%, AMZN 10.9%, META 6.1%, BRK-B 4.2%,
  JPM 3.7%, XOM 2.6%, UNH 1.4%), fetched once and cached.
- Cboe's COR1M and COR3M implied correlation indices, latest print only. Yahoo
  does not carry their history.

Everything downloaded is cached under `source-material/dispersion/` and gitignored.

## Method

**Black-Scholes and the solver.** `bs_price` is the standard formula. The
solver is bisection on [0.0001, 5] for 60 steps, vectorised over the chain.
Newton would be faster but can run off on far-from-the-money quotes; bisection
cannot. Quotes below intrinsic or above the spot return nan rather than a vol.

**ATM vol.** For each expiry, the strike nearest the forward. Call and put mids
at that strike are inverted separately and averaged, which cancels most of the
parity error from a wrong rate or dividend. Per-expiry vols are interpolated to
30 and 60 days linearly in total variance.

**Implied correlation.** The standard single-parameter identity,

    sigma_I^2 = sum_i w_i^2 sigma_i^2 + rho * sum_{i != j} w_i w_j sigma_i sigma_j

solved for rho with the index vol on the left and the basket's single-name vols
on the right. `check.py` plants a rho, builds the index vol from it, and the
formula returns it to machine precision.

**Realised correlation.** Two estimators. The one I compare against implied is
the same identity with realised variances substituted, because that is exactly
the quantity the option prices are encoding. I also report the plain average of
the off-diagonal correlation matrix, which is what most people mean by "realised
correlation" and which is the right number for the basket on its own.

**Why those two realised numbers differ so much.** The basket is ten names and
SPY is five hundred. Plugging SPY's vol into a ten-name identity understates the
correlation, because SPY is diversified across 490 names the identity does not
know about. Today the weighted realised correlation is -0.16 on 21 days while
the pairwise average is +0.08. The implied number has the same bias in it, since
it uses SPY's implied vol against ten single-name vols. So implied and the
weighted realised are comparable with each other, and neither is comparable
with the pairwise average. Cboe's COR1M has the same problem with 50 names
covering about 60% of the index; mine is worse with ten names covering about a
third. The subsection on limitations returns to this.

**The historical proxy.** There is no free history of single-name implied vol.
The daily implied correlation series is therefore built as:

- index implied vol = VIX minus a wedge. VIX is a variance-swap strip that
  includes the skew; an ATM straddle is not. The wedge is measured once from
  today's chain (VIX 14.32 against a 30-day ATM of 11.7%, so 2.6 vol points)
  and subtracted throughout.
- single-name implied vol = k times trailing 21-day realised vol, with k the
  cap-weighted implied-to-realised ratio from today's chains, 1.09, applied to
  every name and every day.

I also show the `k=1, no wedge` version, which uses raw VIX and treats
single-name implied as equal to realised. That attributes the entire VIX
premium to correlation and is an upper bound on the gap. Neither series is what
a desk would have seen in 2018; they are what the data allows.

**The book.** Vega-weighted dispersion: short one vega of the index straddle,
long w_i vega of each name's straddle. A delta-hedged straddle held to expiry
earns roughly vega times (realised minus implied), so the monthly P&L per unit
of index vega, in vol points, is

    sum_i w_i (sigma_i^realised - sigma_i^implied) - (sigma_I^realised - sigma_I^implied)

This is the realised-vol approximation the brief asks for. It ignores gamma
path dependence, which is a real omission the limitations discuss. I also run a
correlation-weighted book, where the single-name vega is scaled by the change
in index vol per unit parallel move in all single-name vols (close to the
square root of implied correlation), so the book is flat to a market-wide vol
shift and holds only correlation. That is closer to what desks run.

**Clock.** Decide at the month-end close, fill at the next close using that
day's VIX and trailing vols, realise over the 21 returns after the fill. The
signal never sees the returns it is graded on; `check.py` shocks a return the
day after the fill and the day after the window and confirms only the first
changes the P&L, and spikes VIX on the decision date to confirm it is not the
fill price.

**Costs.** Half the bid-ask spread crossed once at entry, 0.25 vol points on
the index straddle and 0.75 on each single-name straddle, plus a 0.10 vol point
hedging allowance per leg per month. That is 1.2 vol points a month for the
vega-weighted book. These are assumptions, not measurements.

**Formation and test.** 2015-2020 is formation, 2021 onward is test. The only
thing fitted on formation is the threshold for the conditional variant.

## Results

Everything below is generated by `run.py` and written to `reports/`. P&L is in
vol points per unit of index vega. If the book is sized so one vol point is 1%
of capital, read vol points as percent.

### Today, from the chains (snapshot after the 2026-09-03 close)

| | Weight | Implied 30d | Implied 60d | Realised 21d | Realised 63d | Implied / realised |
|---|---|---|---|---|---|---|
| AAPL | 18.6% | 24.4% | 24.3% | 18.3% | 31.8% | 1.34 |
| MSFT | 14.7% | 24.1% | 24.5% | 21.2% | 41.8% | 1.14 |
| NVDA | 21.5% | 33.4% | 33.9% | 44.4% | 42.3% | 0.75 |
| AMZN | 10.9% | 29.2% | 29.8% | 25.9% | 42.6% | 1.13 |
| META | 6.1% | 35.2% | 35.4% | 29.7% | 45.9% | 1.19 |
| GOOGL | 16.3% | 27.3% | 28.3% | 25.0% | 36.4% | 1.09 |
| BRK-B | 4.2% | 13.6% | 14.1% | 15.0% | 15.1% | 0.91 |
| JPM | 3.7% | 20.0% | 23.6% | 12.3% | 21.0% | 1.63 |
| XOM | 2.6% | 27.7% | 27.4% | 26.5% | 25.8% | 1.04 |
| UNH | 1.4% | 25.5% | 32.1% | 20.2% | 25.2% | 1.26 |
| SPY | | 11.7% | 12.8% | 7.1% | 13.6% | 1.66 |

The 63-day realised column is the summer earnings season; the 21-day column is
the quiet tape since. The premium ratio pooled across names, cap weighted, is
1.09. For SPY it is 1.66, which is the whole trade in one number: the index
carries a much larger premium over its own realised vol than the names do.

| | Value |
|---|---|
| Implied correlation, 30 day | **0.027** |
| Implied correlation, 60 day | 0.059 |
| Realised correlation, weighted identity, 21d / 63d / 252d | -0.158 / -0.031 / 0.005 |
| Realised correlation, average pairwise, 21d / 63d / 252d | 0.075 / 0.076 / 0.096 |
| Implied minus realised (weighted, 21d / 63d) | +0.185 / +0.058 |
| Cboe COR1M / COR3M, same day | 9.5 / 10.1 |

Implied correlation is low in absolute terms and Cboe's independent number
agrees: 0.095 on their 50-name basket against my 0.027 on ten names, both
saying this is a stock-picker's tape where the index barely moves while the
names do. Against the comparable realised number (the weighted identity) the
gap is positive on both windows. Against the pairwise average it is slightly
negative, -0.05, which is the subset bias, not a signal.

### The gap since 2015

Daily statistics of the proxy series, full sample:

| Series | Mean | Median | Std | Share > 0 |
|---|---|---|---|---|
| Implied proxy, k=1, raw VIX (upper bound) | 0.279 | 0.214 | 0.299 | 88% |
| Implied proxy, k=1.09, wedge 2.6 | 0.078 | 0.035 | 0.213 | 61% |
| Realised, weighted identity, 21d | 0.055 | 0.035 | 0.189 | 59% |
| Realised, weighted identity, 63d | 0.060 | 0.042 | 0.158 | 69% |
| Realised, average pairwise, 21d | 0.358 | 0.333 | 0.189 | 99% |
| Gap, k=1 vs realised 21d | **0.224** | 0.177 | 0.286 | 85% |
| Gap, k=1.09 wedge 2.6 vs realised 21d | **0.023** | 0.008 | 0.205 | 52% |
| Gap, k=1.09 wedge 2.6 vs realised 63d | 0.018 | -0.009 | 0.206 | 48% |

Full table with formation and test splits in `reports/gap_stats.csv`. The
formation and test windows tell the same story: the k=1 gap is 0.224 in both,
the calibrated gap is 0.008 and 0.037.

Read those two gap rows together. If you attribute the whole VIX premium to
correlation, implied correlation runs 0.22 above realised on average and is
above it 85% of the time, which is the textbook picture. If you first take out
a 2.6 point skew wedge from VIX and grant the single names a 9% premium of their
own, the gap collapses to 0.02 and is positive half the time. The truth is
somewhere in between and the data cannot say where, because both constants are
measured on one day in September 2026 and applied to eleven years.

The five most negative days are all the first week of April 2020, when 21-day
realised correlation was 0.70 and the proxy implied was near zero: the crash had
happened, trailing single-name vols were enormous, and VIX had already come off
its high. That is the pattern the trade is short.

### The simulated book

Always on, cap weights, k=1.09, wedge 2.6, base costs:

| | Formation 2015-2020 | Test 2021-2026 |
|---|---|---|
| Months | 70 | 68 |
| Mean, gross / net (vol pts) | -2.53 / -3.73 | -1.47 / -2.67 |
| Std | 9.94 | 8.70 |
| Sharpe, gross / net | -0.88 / **-1.30** | -0.58 / **-1.06** |
| Hit rate, net | 31% | 43% |
| Skew | -0.91 | -0.78 |
| 5th / 1st percentile month, net | -17.95 / -32.19 | -17.85 / -26.02 |
| Worst month, net | -45.84 (fill 2020-04-01) | -35.37 (fill 2025-05-01) |
| Max drawdown, net | -264 (2015-04-01 to 2020-06-01) | -219 (2021-12-01 to 2026-08-03) |

Full sample net Sharpe **-1.19**, max drawdown **-434 vol points from
2015-04-01 to 2026-08-03**, which is the whole sample: under these constants
the book never makes a new high. It loses money in eleven of twelve calendar
years; the exception is 2021, +37.9 net, when realised correlation collapsed to
0.03 against an implied 0.21.

That headline is a function of the calibration, and the right presentation is
the sensitivity table rather than any one row of it:

| Constants | Net mean | Single leg | Index leg | Net Sharpe | Max drawdown |
|---|---|---|---|---|---|
| k=1.00, no wedge | +2.26 | -0.10 | +3.56 | **+0.91** | -43.8 |
| k=1.00, wedge 2.6 | -0.33 | -0.10 | +0.97 | -0.13 | -103.0 |
| k=1.09, no wedge | -0.62 | -2.97 | +3.56 | -0.23 | -111.8 |
| k=1.09, wedge 2.6 | -3.20 | -2.97 | +0.97 | **-1.19** | -434.4 |
| k=1.00, no wedge, correlation-weighted | +3.48 | +0.76 | +3.56 | +1.93 | -28.5 |
| k=1.09, wedge 2.6, correlation-weighted | -0.26 | -0.53 | +0.97 | -0.15 | -86.5 |

Three things the table pins down and one it does not.

- **The index leg is measured, not assumed.** Short SPY vol at raw VIX earned
  3.56 vol points a month over 138 months against subsequent realised. After
  the 2.6 point wedge it is 0.97. Both are real numbers from real VIX and real
  SPY returns; the wedge is the only assumption in that column.
- **The single-name leg at k=1 is flat.** Long single-name straddles at
  trailing realised vol earned -0.10 a month, which says trailing 21-day vol is
  an unbiased forecast of the next 21 days on these names. Each 1% of premium
  above that costs the book about 0.3 vol points a month (30 per unit of k),
  so the break-even single-name premium ratio is **k = 0.99 net, 1.03 gross**.
  Today's chains say 1.09. If that ratio is typical, vega-weighted dispersion
  on this basket is a losing trade after costs.
- **The shape is robust to the constants.** Monthly gross P&L has a -0.76
  correlation with the realised-minus-implied correlation surprise, skew is
  negative in every variant except the one that scales the single-name leg to
  nothing, and the two worst months are the same two months in every row:
  the fill on 2020-04-01, when realised correlation over the month was 0.46
  against an implied 0.08, and the fill on 2025-05-01 after the tariff crash.
  The trade is short correlation and correlation spikes in crashes.
- **What it does not pin down is the sign.** Rows one and four are both
  defensible readings of the same price data and they differ by a Sharpe of
  two.

Correlation weighting helps in every row, because it stops the book being long
a parallel vol move it was never paid for. Under the calibrated constants it
takes the full-sample net Sharpe from -1.19 to -0.15 and the drawdown from -434
to -86.5 (2017-12-01 to 2020-06-01), and its worst month is the March 2020 fill
rather than April, -29.58. The single-name scale averaged 0.41 across the
sample and was clipped to zero on the days the subset bias pushed implied
correlation negative.

### The conditional variant, and why I do not trust it

Trading only when the decision-date gap (k=1.09, wedge 2.6, against 63-day
realised) exceeds the formation median turns the book on in 50% of months and
takes the net Sharpe from -1.30 to +0.48 in formation and from -1.06 to +0.50
in test, with the drawdown shrinking from hundreds of vol points to -19 and
-24. A filter that flips the sign of a strategy deserves suspicion, and the leg
breakdown shows where the flip comes from:

| Filter | Single leg | Index leg | Gross |
|---|---|---|---|
| Off | -8.02 | +1.40 | -6.62 |
| On | +2.07 | +0.54 | +2.61 |

The improvement is almost entirely in the single-name leg, not the index leg.
The filter is on when trailing single-name vols are low relative to VIX, which
is exactly when the proxy underprices single-name implied vol, because real
implied vol anticipates mean reversion and trailing realised does not. So the
conditional book is buying single-name vol at a proxy price that is too cheap
and being paid for the proxy's error. With real single-name implied vols I
would expect most of this to vanish. It is in the report because I ran it, not
because I believe it.

### Costs

Net Sharpe of the always-on vega-weighted book by cost scenario:

| Scenario (index / single / hedge, vol pts) | Cost per month | Formation | Test | Full |
|---|---|---|---|---|
| Frictionless (0 / 0 / 0) | 0.0 | -0.88 | -0.58 | -0.74 |
| Half (0.125 / 0.375 / 0.05) | 0.6 | -1.09 | -0.82 | -0.97 |
| Base (0.25 / 0.75 / 0.1) | 1.2 | -1.30 | -1.06 | -1.19 |
| Double | 2.4 | -1.72 | -1.54 | -1.64 |
| Triple | 3.6 | -2.14 | -2.02 | -2.08 |

Costs matter but do not decide anything here; the book is negative before them
under the calibrated constants and positive before and after them under k=1.
At 1.2 vol points a month against a gross standard deviation of 9-10, a
strategy needs a gross Sharpe of roughly 0.4 to survive base costs.

## Files

- `data.py` prices, weights, chain snapshot, Cboe cross-check, cache
- `dispersion.py` Black-Scholes and the solver, ATM extraction, implied and
  realised correlation, the book and the simulation clock
- `check.py` 53 offline checks: the solver inverts known prices, the identity
  recovers a planted correlation, equal implied and realised gives zero P&L, a
  correlation spike loses, the clock cannot see forward
- `run.py` the pipeline; prints to the terminal and `reports/run_log.txt`

Charts in `reports/`: `gap_history.png`, `equity_curve.png`,
`pnl_distribution.png`, `current_vols.png`. Tables: `atm_by_expiry.csv`,
`current_vols.csv`, `implied_corr_now.csv`, `gap_history.csv`, `gap_stats.csv`,
`book.csv`, `performance.csv`, `calibration_sensitivity.csv`,
`cost_sensitivity.csv`.

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline. `run.py` downloads once and caches everything under
`source-material/dispersion/`; delete `chains_*.csv` there to take a fresh
snapshot, `daily.csv` to refresh prices. No dependencies beyond the shared
`requirements.txt`.

## Limitations

In the order I would raise them.

**The historical implied side is a proxy with two constants measured on one
day.** This is the whole caveat. Single-name implied vol history is what
decides whether this trade made money, and I do not have it. The calibration
table is the real output; any single Sharpe from this project is a choice of
row. Fixing it needs OptionMetrics or a vendor vol surface history, and the
code is written so that swapping the proxy for real vols is a one-function
change in `simulate`.

**The proxy overshoots after crashes.** Trailing 21-day realised vol is highest
right after a spike, when real implied vol has already started to fall. So the
book's long single-name straddles are marked at an inflated entry vol in the
month after every crash, and that is where the two worst months come from.
The direction is right (single-name implieds were elevated in April 2020 and
May 2025) but the magnitude is overstated. The 63-day proxy variant is
smoother and worse, because it lags instead.

**Ten names is not the index.** The identity is written for the full basket
and I feed it SPY's vol against a third of SPY's weight. Both implied and
realised correlation are biased down by that, sometimes below zero, which is
why the correlation-weighted scale had to be clipped. Comparing implied with
the matching realised identity cancels most of the bias; comparing either with
the pairwise average does not. A 50-name basket would cut the problem
substantially and yfinance can supply the chains; I stopped at ten because
each name's chain is a separate download and the sample is already dominated
by the proxy problem.

**Survivorship and look-ahead in the basket.** The ten names are today's ten
largest and today's cap weights, applied back to 2015. NVDA at 21.5% of the
basket in 2015 is absurd. The equal-weight variant is the check on that and it
is worse under the calibrated constants (-1.38 formation, -1.07 test), so the
weights are not flattering the result, but a point-in-time basket would be the
right fix and I do not have constituent history.

**The P&L approximation ignores gamma.** Vega times (realised minus implied)
is the variance-swap limit of a delta-hedged straddle. A real straddle's P&L
depends on the path: gamma concentrates near the strike, so a stock that trends
away from it stops earning realised vol. On a 21-day hold this is a second-
order error in normal months and a first-order one in the crash months that
already dominate the tail.

**Chains are a single snapshot after the close.** Quotes taken after the close
are the closing market maker quotes, which is fine for ATM mids on these names,
but there is no intraday averaging and no second snapshot to test stability.
Realised vols run to the 2nd because Yahoo had not posted the 3rd's closes when
the chains were pulled.

**Costs are flat vol-point assumptions.** No spread model by name or by
regime, and single-name spreads widen exactly when the book is losing. Hedging
cost is a fixed allowance rather than a function of gamma and realised moves.

**One market, one basket, one sample.** Eleven years of one basket containing
two vol regimes, 2020 and 2025, that between them account for most of the
tail. Nothing here tests another index or another decade.
