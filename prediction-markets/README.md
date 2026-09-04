# Favorite-longshot bias on Kalshi

Tests whether prediction market prices are calibrated, whether they show the
classic favorite-longshot bias, and whether you can make money fading it once
you pay the spread.

The short version: the bias has the textbook sign, and most of it is not a
price. Pooled over every contract priced below 50 cents, the quoted mid
overstates the YES rate by 2.8 cents (t = 7.7). Split that by how wide the book
was when the mid was read and the number comes apart: in books quoted 5 cents
wide or tighter, which is 63% of the sample, the bias is 0.8 cents at t = 1.9.
The rest of it lives in books quoted 20 cents or wider, where the mid is the
midpoint of nothing. Trading the pooled curve at the mid gives +2.2 cents per
dollar staked out of sample (t = 2.70); restricted to tight books it gives -0.3
cents (t = -0.31); charged the spread that was actually quoted it gives
**-1.4 cents** (t = -1.45). There is a small bias among liquid contracts and it
is well inside the cost of taking it.

## The mistake that nearly became the result

The obvious way to build this dataset is to page through
`/markets?status=settled` and use `last_price_dollars`. Do that and you get a
spectacular result: every price bin from 0.01 to 0.98 looks overpriced, the
0.40-0.50 bin resolves YES only 7.3% of the time, and a fade strategy prints a
Sharpe in the forties.

It is entirely an artifact. `last_price_dollars` is the last trade at an
unspecified time, not a price at any fixed horizon. On a thin market that trade
can be days old. Someone buys a sports prop at 45 cents on Tuesday, the
situation turns against it, nobody ever trades it again, and it settles at zero
with a "last price" of 0.45 still attached. The markets that sit in the middle
of the price range are precisely the ones this happens to: their median volume
is 133 contracts against 2,881 for the sample as a whole.

So the mid-price bins are not mispriced contracts, they are stale quotes on dead
ones. Both curves are in `reports/calibration.png` and the gap between them is
the whole lesson. I kept the broken version in the code as
`stale_price_diagnostic()` because the difference is more instructive than the
answer.

The fix is to take the quoted book at a fixed horizon. For every sampled market
I pull hourly candlesticks and read the last valid bid and ask at or before 24
hours from close, then use the mid. A quoted book exists whether or not anyone
trades, so it does not go stale the same way, and it hands me a measured spread
instead of an assumed one.

## Data

Two pulls against `api.elections.kalshi.com/trade-api/v2`, no auth needed.

| | |
|---|---|
| settled binary markets listed | 468,432 |
| with a quoted bid/ask at T-24h | 14,560 |
| close dates | 2026-06-26 to 2026-09-03 |
| YES resolution rate | 33.5% |

Getting the listing means walking all 13,772 series rather than paging the
global `/markets` feed, which is swamped by auto-generated 15-minute crypto
parlays (`KXMVE*`) that would otherwise be 90% of the sample. I drop those, plus
anything that never traded, plus anything whose price is exactly 0 or 1.

The snapshot sample is 77% sports, mostly game props. That is what Kalshi lists,
not a choice I made, but it means this is closer to a study of sports betting
markets than of political ones. Politics and elections together are 134 markets.

Both files cache to `../source-material/kalshi/`. The snapshot pull is resumable
and appends as it goes, because Kalshi rate-limits candlestick calls to roughly
5 per second and the full pull takes about an hour.

## Calibration

Brier score 0.1527 against a base-rate baseline of 0.2229. The Murphy
decomposition is reliability 0.00086, resolution 0.07031, uncertainty 0.22292.
Reliability being that small is the headline: these prices are close to
calibrated. Whatever bias exists is a couple of cents, not a couple of dimes.

Bins are deciles with the tails split finer, since that is where the longshot
story lives. Significance uses an Agresti-Coull standard error rather than the
plain binomial one, which collapses to zero when a bin resolves all-yes or
all-no and would otherwise report an infinite z on 81 contracts.

| bin | n | mean price | realised | 95% CI | bias | z |
|---|---|---|---|---|---|---|
| 0.01-0.02 | 418 | 0.0149 | 0.0024 | [0.000, 0.013] | +0.0125 | +3.1 |
| 0.02-0.05 | 1233 | 0.0325 | 0.0187 | [0.012, 0.028] | +0.0139 | +3.5 |
| 0.05-0.10 | 1443 | 0.0714 | 0.0679 | [0.056, 0.082] | +0.0035 | +0.5 |
| 0.10-0.20 | 1854 | 0.1456 | 0.1187 | [0.105, 0.134] | +0.0269 | +3.6 |
| 0.20-0.30 | 1612 | 0.2469 | 0.2171 | [0.198, 0.238] | +0.0298 | +2.9 |
| 0.30-0.40 | 1525 | 0.3504 | 0.3141 | [0.291, 0.338] | +0.0363 | +3.1 |
| 0.40-0.50 | 2337 | 0.4535 | 0.4048 | [0.385, 0.425] | +0.0487 | +4.8 |
| 0.50-0.60 | 1561 | 0.5370 | 0.5151 | [0.490, 0.540] | +0.0220 | +1.7 |
| 0.60-0.70 | 741 | 0.6437 | 0.6464 | [0.611, 0.680] | -0.0027 | -0.2 |
| 0.70-0.80 | 652 | 0.7463 | 0.7669 | [0.733, 0.798] | -0.0205 | -1.2 |
| 0.80-0.90 | 519 | 0.8463 | 0.8805 | [0.850, 0.906] | -0.0342 | -2.4 |
| 0.90-0.95 | 258 | 0.9257 | 0.9419 | [0.906, 0.964] | -0.0161 | -1.1 |
| 0.95-0.98 | 204 | 0.9637 | 0.9853 | [0.958, 0.995] | -0.0216 | -2.0 |

Full table including the sparse end bins in `reports/calibration_quoted_mid.csv`.

The sign pattern is the classic one. Cheap contracts are overpriced, expensive
contracts are underpriced, and the crossover sits around 0.60. Aggregating:

- below 0.50: n = 10,542, mean bias **+0.0280**, t = **7.68**
- 0.50 and above: n = 4,018, mean bias -0.0022, t = -0.32

So the longshot half is where everything is happening. The favorite half is
individually suggestive in a couple of bins but collectively indistinguishable
from zero, which is weaker than the literature would predict. I would not read
much into the 0.80-0.90 bin on 519 contracts.

One caveat on the per-bin z-scores: I am testing 16 bins, so at 5% a couple of
them should light up on noise alone. The bins below 0.5 are not that, they are
consistently signed and the pooled t is 7.7. The individual favorite-side bins
plausibly are.

### Most of the bias is in books with no real mid

The median spread is 3 cents, but the distribution has a long tail: 19% of
markets were quoted more than 20 cents wide at T-24h and 11% more than 50
cents wide. A book of 0.05 bid, 0.95 ask has a mid of 0.50 and says nothing
about the contract's probability. Those markets sit in the middle price bins
and they resolve NO far more often than 0.50, which shows up as a large
positive "bias". Splitting the below-0.50 half by quoted spread:

| quoted spread | n | mean bias | t |
|---|---|---|---|
| under 0.05 | 6,530 | +0.0081 | +1.86 |
| 0.05 to 0.10 | 1,418 | +0.0287 | +3.05 |
| 0.10 to 0.20 | 602 | +0.0396 | +2.69 |
| 0.20 and wider | 1,992 | +0.0889 | +8.97 |

The bias grows monotonically with the spread and the t = 7.7 headline is
carried by the widest tier. In the tight tier, which is the only one where
the mid is a price someone would quote, the longshot bias is under a cent and
not significant at 5%. Below 5 cents the tight-book bias is +1.3 cents at
t = 5.1 on 1,705 contracts, so the very cheap end does show the textbook
effect even in liquid books. Everywhere else it is a wide-book artifact.

I kept the pooled sample as the headline calibration table because that is
what the market listed, but every claim about "the bias" below should be read
against this split.

## Does it trade

Bias is estimated on the first half of the sample by close date and the
strategy is tested on the second half. Nothing about the test window touches the
fitted curve. This matters more than usual here, because the temptation is to
compute "edge" as price minus realised frequency using the same contracts you
then trade, which guarantees a profit and measures nothing.

One more cut is needed to make that true. A test market is traded 24 hours
before it closes, so a market closing an hour after the fit window ends would
be traded on a curve that already knows the outcomes of fit markets that had
not closed when the bet was placed. Test markets closing within 24 hours of
the fit window's last close are dropped, which removes 252 of them.

For each test market the fitted curve gives an estimated true probability, and I
take the exact binary Kelly stake, f\* = p - (1-p)·c/(1-c), on whichever side is
positive. Buying YES costs the ask, buying NO costs one minus the bid. Stakes
are held in Kelly proportion and rescaled so a day's total exposure is
k·min(Σf, 1), then the bankroll compounds daily.

Sharpe ratios are annualised with 365, since these markets settle on calendar
days and the daily series has no weekends missing.

| scenario | markets traded | payoff per $ staked | t | Sharpe | bootstrap 95% CI |
|---|---|---|---|---|---|
| mid to mid, no cost | 7,028 (100%) | +0.0221 | +2.70 | +10.87 | [+4.26, +20.44] |
| assumed 0.02 spread | 6,890 (98%) | +0.0098 | +1.23 | +4.02 | [-2.98, +11.91] |
| actual quoted spread | 4,223 (60%) | **-0.0140** | -1.45 | -3.76 | [-13.14, +3.24] |
| mid to mid, tight books only (spread under 0.05) | 4,486 | **-0.0030** | -0.31 | | |

The spec suggested assuming 2 cents. That assumption is too kind. The median
quoted spread in this sample is 3 cents and **59.4% of markets are wider than 2
cents**, so the 2-cent row is roughly the best case and it is already
insignificant. Charged the spread that was actually quoted, the strategy loses
money.

The last row is the one that settles it. Take the same fitted curve and trade
only the test markets whose book was 5 cents wide or tighter, still at the
mid with no cost at all, and the gross edge is gone: -0.3 cents, t = -0.31.
The +2.2 cents in the first row is earned on markets whose mid was the middle
of an empty book, which is not a fill anyone gets.

The -1.4 cents at the quoted spread is not itself significant either
(t = -1.45). The honest statement is not "fading longshots loses money", it is
that the gross edge is a wide-book artifact, the liquid-book edge is under a
cent, and neither survives a 3-cent median cost.

### Where the spread makes it hopeless

Spread in cents is flat-ish across the price range, but as a fraction of the
price it explodes in the tail, which is exactly where the bias is largest.

| price | n | median spread | spread / price |
|---|---|---|---|
| 0.00-0.05 | 1,771 | 0.010 | 0.67 |
| 0.05-0.20 | 3,297 | 0.030 | 0.33 |
| 0.20-0.80 | 8,428 | 0.050 | 0.12 |
| 0.80-0.95 | 777 | 0.030 | 0.03 |
| 0.95-1.00 | 287 | 0.020 | 0.02 |

Below 5 cents you pay two thirds of the contract's value in spread. The bias
there is about 1.3 cents. No amount of edge survives that, and this is the part
of the curve the favorite-longshot literature cares about most. If you want to
trade this bias you have to do it as a maker, not a taker, and that is a
different project with a different set of problems.

### Kelly sizing

`reports/kelly_sensitivity.png` sweeps k from 0.05 to 1.0. The Sharpe is
perfectly flat in k, which is not a bug: daily returns scale linearly in k, so
the ratio is invariant by construction. Only the growth rate and the drawdown
respond to sizing, so the sweep plots annualised log growth instead.

At the quoted spread, log growth is negative at every k and gets worse linearly,
which is what over-betting a negative edge does. In the no-cost case growth is
still rising at k = 1.0 rather than turning over, because with thousands of
near-independent daily bets the exposure cap binds long before individual
stakes get dangerous. That flatters full Kelly. With a realistically correlated
book (the same game, the same weather station) the turnover point would come in
much earlier, and I have not modelled that correlation at all.

## Running it

```
../.venv/bin/python check.py    # offline tests, no network
../.venv/bin/python data.py     # both pulls, ~1.5h, resumable
../.venv/bin/python run.py      # tables to stdout, charts to reports/
```

`check.py` runs on simulated markets with a planted Prelec-style distortion and
asserts the machinery recovers it. The test that matters is the null one: on
perfectly calibrated simulated markets, across 40 replications, the rate of
"significant" bins must stay near the nominal 5% and the pooled bias must not
lean, and a strategy fitted on half that data must not make money on the other
half. If the pipeline could manufacture an edge from noise, that test is what
catches it.

## What I would not claim from this

**The window is ten weeks and the test half is 27 calendar days.** Every
annualised number above is built on 27 daily observations, which is why the
bootstrap intervals are as wide as they are. The Sharpe point estimates should
be read as signs, not magnitudes.

**It is a sports dataset.** 77% of the snapshot is sports props. The classic
favorite-longshot literature is horse racing, which is closer to this than to
politics, but I cannot say anything about political markets from 134 contracts.

**Selection on which markets have a quoted book.** I only keep markets with a
live two-sided quote 24 hours before close, which is 38% of the markets whose
attempts were logged. Markets that fail that test are thinner than the ones
that pass, and thin markets are plausibly where mispricing concentrates. The
bias I measure is therefore a bias among the relatively liquid, and the
spread-tier table shows that even within this set the wide books carry most
of it.

**One horizon only.** Everything is measured at T-24h. Bias almost certainly
varies with time to resolution, and a market a week out is a different animal
from one an hour out. Checking that would mean a snapshot per horizon.

**The spread is a snapshot, not a fill.** I charge the quoted spread at T-24h
and assume I can take the whole stake there. Depth at the touch on a market
trading 133 contracts is not going to fill a Kelly-sized order, so the real cost
is worse than the number I use, not better.
