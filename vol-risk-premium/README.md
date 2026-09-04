# Volatility risk premium on SPY

Implied volatility sits above the volatility that follows it. I measure that gap
on SPY from 2010 to 2026, check whether a GARCH forecast explains it, then sell
it with a delta-hedged short straddle and count what the trade costs in the
tail.

The gap is real: implied vol averages 18.4% against 14.7% subsequently realised,
a 3.78 point spread with a Newey-West t of 8.3. Harvesting it net of costs
returns 9.55% a year per dollar of straddle notional at a Sharpe of 1.74. The
same P&L series has skew -5.2, excess kurtosis 75, a worst day of -6.5% and a
maximum drawdown of -13.6%. Delete the worst 1% of days and the Sharpe goes to
3.75, so roughly half the apparent quality of this strategy is payment for the
days I deleted. And two further adjustments I could not rule out - selling at
the at-the-money vol rather than the VIX, and the survivorship of the sample
period itself - are each large enough to take the whole thing to zero.

## Data

Daily SPY and VIX closes from Yahoo Finance via `yfinance`, 2010-01-05 to
2026-09-01, 4,189 days, cached to `reports/daily.csv` so only the first run
needs internet. SPY is the tradable proxy for the index; the VIX is the implied
vol series.

I also pull a live SPY option chain and back an at-the-money implied vol out of
real mid quotes, because the whole study rests on the VIX being a fair stand-in
for what a straddle actually sells at. It is not, and that is the single largest
caveat in the project. See "The VIX is not the price" below.

`check.py` runs offline on simulated paths with planted volatility.

## Measuring the premium

For each date t, implied vol is `VIX_t / 100` and realised vol is the
annualised close-to-close volatility of the 21 trading days strictly after t,
`sqrt(252 * mean(r^2))` on log returns with a zero mean. The VIX is a 30
calendar day measure and 21 trading days is 30 calendar days, so the horizons
match.

| | |
|---|---|
| mean implied | 18.44% |
| mean realised, next 21 days | 14.65% |
| mean gap | +3.78% |
| median gap | +4.63% |
| gap standard deviation | 7.26% |
| 5th / 95th percentile | -5.93% / +11.83% |
| worst gap | -68.89% |
| days with a positive gap | 84.6% |
| median implied / realised | 1.38x |

The windows overlap on 20 of their 21 days, so the naive t-stat of 33.6 is
meaningless. With 31 Newey-West lags the mean gap has a standard error of 0.45
points and a t of 8.35. That is the number I stand behind.

The gap is autocorrelated 0.95 at one day and 0.77 at five days, which is just
the persistence of the overlapping windows, and it is gone by 21 days (0.03).
There is no exploitable predictability in the gap itself beyond the vol level.

Where the gap comes from matters more than its average:

| | correlation with the gap |
|---|---|
| forward 21-day SPY return | **+0.709** |
| VIX level | +0.195 |
| trailing 21-day realised vol | +0.143 |

The gap is enormous when the market goes up over the next month and negative
when it falls. That single number is the economic content of the whole study:
this is not a mispricing sitting there to be collected, it is a payment for
holding an exposure that loses exactly when equities lose.

By starting VIX quintile:

| starting VIX | mean gap | worst gap | share positive | days |
|---|---|---|---|---|
| q1 (calm) | +2.20% | -18.67% | 80.6% | 831 |
| q2 | +2.43% | -68.89% | 84.7% | 832 |
| q3 | +3.93% | -67.77% | 86.7% | 829 |
| q4 | +4.36% | -32.84% | 85.5% | 826 |
| q5 (panic) | +6.03% | -64.22% | 86.3% | 830 |

Selling vol is best paid when vol is already high, which is the reverse of the
naive intuition, and the worst single outcomes are in the middle quintiles
because those are the dates that were calm the day before a crash.

## Is it just a bad forecast?

If the VIX were simply a poor predictor, a decent statistical forecast should
beat it. I fit a GARCH(1,1) with t innovations on an expanding window, refit
every 21 days, and forecast the 21-day-ahead annualised vol analytically.

| forecast | mean level | mean error vs realised | RMSE | corr | MZ intercept | MZ slope | MZ R2 |
|---|---|---|---|---|---|---|---|
| VIX implied | 17.76% | +3.64% | 8.04% | 0.594 | -0.0016 | 0.804 | 0.353 |
| GARCH(1,1) | 16.62% | +2.51% | 8.15% | 0.564 | 0.0348 | 0.640 | 0.318 |

The VIX is the more accurate forecast on RMSE, correlation and R2, and its
Mincer-Zarnowitz intercept is essentially zero with a slope of 0.80. GARCH is
worse on every measure and is itself biased high by 2.51 points. In an
encompassing regression of realised vol on both, the VIX carries a coefficient
of 0.56 (t 6.2) and GARCH 0.23 (t 2.4), so the VIX contains information GARCH
does not.

That is the answer to the question. The premium is not a forecasting failure
that a better model would arbitrage away. The VIX is the best forecast available
here and it is still systematically too high, which is what a risk premium looks
like.

The 2.51 point bias in GARCH is worth flagging on its own: it fits the sample
average of squared returns, and the sample average is pulled up by the crashes,
so a model with no risk premium in it still forecasts high. Roughly a third of
what I would otherwise call a premium is that estimator effect rather than
compensation for risk.

## The strategy

Every 21 trading days, sell a one-month at-the-money straddle struck at that
day's close, hedge to zero delta, rehedge at every close, hold to expiry. 199
cycles, 4,180 days. Average premium sold is 4.22% of spot.

Stated assumptions, all of them:

- Options are priced and marked with Black-Scholes at that day's VIX, zero rate,
  zero dividend. Marking daily at the VIX means the daily marks telescope
  exactly into the hold-to-expiry P&L, so nothing is created by the marking.
- 1bp one-way slippage on every hedge share traded. SPY quotes a penny wide,
  0.13bp on a 760 handle, so 1bp one-way is about fifteen times the touch and
  leaves room for impact. Hedge turnover runs 39.3x notional a year, so this
  costs 0.39% a year.
- 1% of mid premium given up on the option, charged once on entry, which costs
  0.51% a year. Expiry settles at intrinsic with no exit spread, which flatters
  the result slightly.
- Hedges execute at the close, on the close price used to compute the signal.
- P&L is expressed per $1 of straddle notional and accumulated additively on
  constant notional. There is no margin model and no financing, so this is an
  excess return over cash. Sharpe, skew and kurtosis do not depend on that
  choice; the level of the return does.

| | gross of costs | net of costs |
|---|---|---|
| total P&L, notional | +1.733 | +1.584 |
| annual P&L, notional | +10.45% | +9.55% |
| annual vol | 5.49% | 5.50% |
| Sharpe | 1.90 | **1.74** |
| Sharpe excl. worst 1% of days | 3.98 | **3.75** |
| skew | -5.18 | -5.16 |
| excess kurtosis | 75.9 | 75.4 |
| worst day | -6.49% | -6.49% |
| best day | +3.05% | +3.04% |
| max drawdown | -13.38% | **-13.61%** |
| hit rate | 72.5% | 70.2% |

Per cycle, 82.9% of the 199 months are positive, median +0.92%, worst -5.29%,
best +4.15%. A cycle's expiry day is also the next cycle's entry day and is
booked to the new cycle, so each cycle's figure carries its predecessor's
expiry settlement; the total is unaffected.

## The tail

A Sharpe of 1.74 on a series with skew -5.2 and excess kurtosis 75 is not the
same object as a Sharpe of 1.74 on a normal one. A normal series with the same
standard deviation returns excess kurtosis 0.06 in this sample.

Ten worst days:

| date | P&L | SPY return | VIX | in sigmas |
|---|---|---|---|---|
| 2020-03-16 | -6.49% | -11.59% | 82.7 | -18.7 |
| 2020-03-12 | -6.08% | -10.06% | 75.5 | -17.6 |
| 2011-08-08 | -4.48% | -6.73% | 48.0 | -12.9 |
| 2020-06-11 | -3.90% | -5.94% | 40.8 | -11.3 |
| 2018-02-05 | -3.52% | -4.27% | 37.3 | -10.2 |
| 2025-04-03 | -2.68% | -5.05% | 30.0 | -7.7 |
| 2024-12-18 | -2.59% | -3.03% | 27.6 | -7.5 |
| 2021-01-27 | -2.36% | -2.47% | 37.2 | -6.8 |
| 2011-08-18 | -2.34% | -4.41% | 42.7 | -6.8 |
| 2011-11-09 | -2.32% | -3.76% | 36.2 | -6.7 |

An 18-sigma day is not a tail event in any distribution I fitted; it is evidence
that the distribution is wrong. Under a normal, a single 18-sigma day has
probability around 1e-75 and I have two above 17 sigma in sixteen years.

How much of the return is payment for those days:

| worst days removed | annual P&L | Sharpe | share of total P&L removed |
|---|---|---|---|
| 0 | 9.55% | 1.74 | 0.0% |
| 1 | 9.94% | 1.89 | 4.1% |
| 3 | 10.59% | 2.15 | 10.8% |
| 5 | 11.04% | 2.33 | 15.4% |
| 10 | 11.79% | 2.60 | 23.2% |
| 21 | 13.07% | 3.07 | 36.2% |

Ten days out of 4,180, a quarter of one percent of the sample, carry 23% of the
total P&L as losses. Deleting the worst 1% of days more than doubles the Sharpe.
The gap between 1.74 and 3.75 is the price of the tail, and it is the honest
description of what the strategy is: a Sharpe-3.75 business that pays out a
large negative lump every few years.

### Crisis episodes

| episode | days | P&L | worst day | SPY | VIX high | in months of average P&L |
|---|---|---|---|---|---|---|
| Aug 2011 US downgrade | 28 | -1.55% | -4.48% | -8.7% | 48.0 | -2.0 |
| Aug 2015 China devaluation | 32 | -1.92% | -1.47% | -8.5% | 40.7 | -2.4 |
| Feb 2018 Volmageddon | 23 | -1.70% | -3.52% | -5.2% | 37.3 | -2.1 |
| Dec 2018 selloff | 19 | -2.12% | -2.17% | -10.0% | 36.1 | -2.7 |
| Mar 2020 Covid | 51 | -5.78% | -6.49% | -13.6% | 82.7 | -7.3 |
| Aug 2024 yen carry unwind | 12 | -0.10% | -0.64% | +0.4% | 38.6 | -0.1 |
| Apr 2025 tariffs | 23 | -3.74% | -2.68% | -0.2% | 52.3 | -4.7 |

March 2020 cost 7.3 months of average P&L. The deepest drawdown ran -13.61%
from 2020-01-23 to 2020-03-16, 53 calendar days to lose it and until 2020-07-22
to get it back.

February 2018 looks mild at -1.70% and it is worth saying why rather than
claiming the strategy survived it. The delta-hedged straddle is a much smaller
short-vol position than the levered inverse-VIX ETNs that actually blew up that
week, and my cycle happened to be two days from expiry on February 5th, so there
was very little time value left to lose even as the VIX closed at 37.3. Started
on a different day of the month the same episode costs 3.41% (see the calendar
table below). Feb 2018 killed XIV, not this trade.

April 2025 is the more interesting episode: SPY finished the window flat at
-0.2% and the strategy still lost 3.74%, its second worst episode. That is the
pure gamma cost of a round trip - the market fell hard and came back, the daily
rehedges sold low and bought high the whole way, and no amount of ending up
where you started refunds that.

August 2024 shows -0.10% and I would not read anything into it. The VIX closed
at 38.57 on August 5th and back at 27.71 the next day. My straddle was six days
from expiry, lost 0.64% on the spike and made most of it back inside the same
cycle. This is timing luck, not resilience.

### Timing luck

The cycle calendar is arbitrary. Running the identical strategy started on each
of the 21 possible offsets:

| | min | median | max |
|---|---|---|---|
| annual P&L | 9.09% | 9.58% | 10.41% |
| Sharpe | 1.69 | 1.89 | 2.20 |
| max drawdown | -14.58% | -12.69% | -6.50% |
| Feb 2018 | -3.41% | -2.01% | -0.56% |
| Mar 2020 | -11.19% | -6.69% | +2.67% |

The average return is stable across calendars, so the premium itself is not a
timing artifact. The tail is not stable at all: March 2020 ranges from -11.19%
to +2.67% depending on nothing more than which day of the month the position
happened to roll. My headline max drawdown of -13.6% is the median-ish draw from
a distribution whose bad end is -14.6%, and a single-path drawdown number for a
short-vol strategy should always be read that way.

### Equity beta

Regressing daily P&L on SPY returns gives a beta of 0.198 (t 9.1) and an alpha
of 6.93% a year (t 6.8), R2 0.38. On the worst 5% of equity days the strategy
loses 0.79% a day on average against +0.08% on all other days.

Selling that beta away leaves 6.93% a year at a Sharpe of 1.60, skew -3.86,
worst day -4.20%. So about a quarter of the return is plain equity exposure in
disguise, and hedging it out reduces but does not remove the tail: the residual
is still badly left-skewed, because the losses are convex in the size of the
move and a linear beta hedge cannot catch that.

## Costs

| hedge bps (one way) | option spread | annual P&L | Sharpe | max drawdown |
|---|---|---|---|---|
| 0.0 | 0.0% | 10.45% | 1.90 | -13.38% |
| 0.5 | 0.5% | 10.00% | 1.82 | -13.50% |
| 1.0 | 1.0% | 9.55% | 1.74 | -13.61% |
| 2.0 | 2.0% | 8.65% | 1.57 | -13.84% |
| 5.0 | 5.0% | 5.95% | 1.06 | -14.54% |

Costs are not what kills this. Even at 5bp hedges and a 5% option spread, which
is far worse than SPY actually trades, the Sharpe is 1.06. That is a warning
sign rather than a comfort: when a strategy survives a 5x cost stress, the real
risk is somewhere the cost model does not look.

## The VIX is not the price

Here is where it does look. The simulator sells at the VIX. The VIX is a
variance-swap rate integrated across the whole strike ladder, so it embeds the
put skew and sits above the at-the-money volatility that an actual ATM straddle
trades at. A one-day snapshot of the live SPY chain, taken 2026-09-03 against a
last VIX close of 16.34:

| expiry | days | strike | straddle mid | ATM IV from mids | F/S - 1 |
|---|---|---|---|---|---|
| 2026-09-25 | 23 | 765 | 18.60 | 12.22% | +0.050% |
| 2026-09-30 | 28 | 765 | 20.56 | 12.23% | +0.059% |
| 2026-10-02 | 30 | 765 | 21.35 | 12.26% | +0.087% |
| 2026-10-09 | 37 | 765 | 24.84 | 12.84% | +0.178% |
| 2026-10-16 | 44 | 765 | 27.21 | 12.89% | +0.225% |

Two things come out of this. The forward implied by put-call parity is within
0.09% of spot at 30 days, so my zero-rate zero-dividend Black-Scholes is
accurate to under a tenth of a percent on strike placement, which I am
comfortable with. And the at-the-money vol is roughly four points under the
VIX, which I am not comfortable with at all.

| vol points sold below the VIX | annual P&L | Sharpe | max drawdown |
|---|---|---|---|
| 0% | 9.55% | 1.74 | -13.61% |
| 1% | 6.80% | 1.23 | -13.96% |
| 2% | 4.04% | 0.73 | -14.31% |
| 3% | 1.29% | 0.23 | -20.24% |
| 4% | -1.46% | -0.26 | -40.24% |

At the four points the snapshot suggests, the strategy loses money and the
drawdown triples. The honest statement is that I have one after-hours snapshot,
taken two days after the VIX close I compare it against, and the true average
basis over 2010-2026 is smaller than four points and varies with the VIX level.
But I cannot rule out two, and at two points the Sharpe is 0.73 rather than
1.74. Fixing this properly needs a historical ATM implied vol series, which is
the first thing I would buy.

## Files

- `data.py` SPY and VIX download with a CSV cache, live option chain fetch
- `vrp.py` Black-Scholes price/delta/vega, realised vol, rolling GARCH forecast, the straddle simulator, tail metrics
- `check.py` offline checks against known Black-Scholes values and planted volatility
- `run.py` the whole study, tables and charts to `reports/`

Charts in `reports/`: `premium.png`, `gap_distribution.png`, `equity_curve.png`,
`pnl_distribution.png`, `forecast_vs_realised.png`.

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline. `run.py` downloads once, then reads the cache. The
rolling GARCH takes about a minute on the first run and is cached to
`reports/garch_forecast.csv`. The option chain snapshot is cached too, as
`reports/chain_SPY_2026-09-03.csv`, and the table above is that file; delete it
to take a fresh snapshot, which will not match the table.

## Limitations

**The VIX overstates what I could sell.** Covered above and it is the largest
one. A four point basis takes the strategy negative. This is not fixable with
free data.

**Sixteen years, one market, and a good one for short vol.** 2010-2026 contains
one true crash and a long bull market. There is no 1987, no 2008. A period that
includes either would carry a loss much larger than -13.6%, and the sample's
worst day is a hard lower bound on how bad this gets rather than any kind of
estimate. Sizing this strategy off a sample maximum is how short-vol books die.

**No margin model.** Real short-option positions are margined, margin
requirements rise with volatility, and they rise fastest exactly when the
position is losing. The March 2020 drawdown of 13.6% of notional would have come
with a margin call that forced part of the position closed at the worst
possible moment. Nothing in this simulation can be forced to liquidate, which
means the -13.6% is optimistic in a way that has no bound.

**Discrete hedging with no bid-ask on the underlying at the close.** I rehedge
once a day at the close and pay 1bp. A real book rehedges on bands or
intraday, and a large SPY hedge at the close in March 2020 does not execute at
the close price.

**One strike, one tenor.** ATM straddles only. The variance risk premium is
strike-dependent - the out-of-the-money puts carry most of it - and a real
implementation would trade the strip, which changes both the return and the
shape of the tail.

**Realised vol is close-to-close.** That ignores overnight jumps' contribution
and the intraday path entirely. Parkinson or Garman-Klass estimators would be
more efficient and would raise the measured realised vol a little, which shrinks
the measured gap.

**The GARCH forecast is only quasi out-of-sample.** Parameters are refit on an
expanding window every 21 days and never see future data, but the model
specification, the horizon and the refit frequency were chosen by me knowing how
the sample turned out.
