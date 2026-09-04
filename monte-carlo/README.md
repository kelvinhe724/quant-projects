# Monte Carlo option pricing, with variance reduction that actually pays

A risk-neutral GBM pricing engine for European, arithmetic Asian and barrier
options. The European case exists to be validated against Black-Scholes; the
Asian and barrier cases are what the engine is for, since neither has a closed
form. The depth of the project is in the variance reduction: antithetic
variates, control variates, and scrambled Sobol, each measured against plain
Monte Carlo on variance, wall time, and paths needed for a target standard
error.

Dividends are zero throughout. Time is in trading days, 252 to the year. The
discretisation follows the spec: half-day steps when expiry is under 30 days,
whole days at or above.

## Data

The core engine needs no data. `data.py` pulls one thing: a live SPY option
chain from Yahoo, plus the 13-week T-bill rate from `^IRX`, so the model can be
pointed at a real quote. `check.py` runs offline against closed forms and
identities only.

## Files

- `mc.py` the engine: `price_option`, `price_qmc`, Black-Scholes and geometric-Asian closed forms, `greeks_fd` / `greeks_pathwise` / `greeks_lr`
- `data.py` live spot, rate, near-the-money call quote, realised volatility
- `check.py` 26 offline checks against closed forms and pricing identities
- `run.py` full pipeline, charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`run.py` takes about 30 seconds.

## Results

Test option: S0 = 100, K = 100, 63 trading days, r = 4%, sigma = 25%.
Black-Scholes call = 5.4721.

### Convergence to the closed form

| paths | MC price | SE | error | error / SE |
|---|---|---|---|---|
| 4,096 | 5.2712 | 0.1237 | -0.2009 | -1.62 |
| 16,384 | 5.4928 | 0.0634 | +0.0207 | +0.33 |
| 65,536 | 5.5245 | 0.0321 | +0.0524 | +1.64 |
| 262,144 | 5.4989 | 0.0160 | +0.0268 | +1.67 |
| 1,048,576 | 5.4650 | 0.0080 | -0.0071 | -0.89 |

The error stays inside two standard errors at every path count and does not
trend, which is what an unbiased estimator looks like. The standard error falls
by a factor of 15.5 across a 256-fold increase in paths, against the 16 that
1/sqrt(n) predicts.

Put-call parity holds in the simulated prices to within the combined standard
error, and in-out barrier parity (knock-out + knock-in = vanilla) holds to
machine precision, because both legs are priced off the same paths.

### Variance reduction

European call, at 1,048,576 paths. Plain SE is 0.00800 in 0.54s.

| method | price | SE | variance cut | time | effective speedup | fitted rate | paths for SE 0.005 |
|---|---|---|---|---|---|---|---|
| plain | 5.4650 | 0.00800 | 1.0x | 0.54s | 1.0x | 0.49 | 2,712,533 |
| antithetic | 5.4763 | 0.00595 | 1.8x | 0.45s | 2.2x | 0.50 | 1,490,402 |
| control (terminal price) | 5.4749 | 0.00364 | 4.8x | 0.53s | 4.9x | 0.50 | 550,700 |
| Sobol (scrambled, 16 replicates) | 5.4707 | 0.00183 | 19.0x | 1.15s | 8.9x | 0.57 | 177,585 |

Arithmetic Asian call, same path count. Plain SE is 0.00453 in 0.53s.

| method | price | SE | variance cut | time | effective speedup | fitted rate | paths for SE 0.005 |
|---|---|---|---|---|---|---|---|
| plain | 3.1566 | 0.00453 | 1.0x | 0.53s | 1.0x | 0.50 | 862,098 |
| antithetic | 3.1482 | 0.00332 | 1.9x | 0.44s | 2.3x | 0.50 | 462,679 |
| control (geometric Asian) | 3.1520 | 0.00007 | 3,666x | 0.52s | 3,693x | 0.51 | 258 |
| Sobol (scrambled, 16 replicates) | 3.1532 | 0.00078 | 33.4x | 1.16s | 15.2x | 0.60 | 48,839 |

Effective speedup is the variance cut divided by the relative runtime, which is
the number that matters when you are buying accuracy with a compute budget.
Prices and standard errors are seeded and reproduce exactly; the wall times and
the speedups derived from them move by a few percent between runs.
Sobol costs about 2.2x the wall time of plain sampling, so its 19x variance cut
is worth 8.9x in practice.

The `fitted rate` column is the slope of log(SE) against log(paths). Plain,
antithetic and control variates all sit at 0.50, the Monte Carlo law. Sobol
comes in at 0.57 and 0.60, better than 0.50 but far short of the near-1.0 that
QMC achieves in low dimensions, because the effective dimension here is 63.
The paths-needed column extrapolates with each method's own fitted rate rather
than assuming 0.5 for everything, since assuming 0.5 would understate Sobol.

Three things stand out. Antithetic is nearly free but caps out under 2x, because
the max(., 0) kink destroys most of the negative correlation it induces.
Scrambled Sobol is a solid, general 9x to 15x with no problem-specific work. And
the geometric-Asian control is worth more than everything else combined by three
orders of magnitude, because a control variate's payoff is 1/(1 - rho^2) and the
geometric and arithmetic averages of the same path are correlated at about
0.9999. The 258-path figure is a formal extrapolation and should be read as "a
few thousand", since the beta estimate and the CLT are both unreliable at that
sample size.

### Prices with no closed form

At 1,000,000 antithetic paths:

| option | price | SE |
|---|---|---|
| European call (closed form 5.4721) | 5.4721 | - |
| arithmetic Asian call | 3.1525 | 0.0034 |
| arithmetic Asian, control variate | 3.1522 | 0.0001 |
| geometric Asian call (closed form 3.0804) | 3.0808 | 0.0034 |

Up-and-out calls, barrier checked at every simulated step, against a vanilla of
5.4785 on the same paths:

| barrier | knock-out | knock-in | sum |
|---|---|---|---|
| 105 | 0.0634 | 5.4151 | 5.4785 |
| 110 | 0.5310 | 4.9475 | 5.4785 |
| 115 | 1.5187 | 3.9598 | 5.4785 |
| 120 | 2.7072 | 2.7713 | 5.4785 |
| 130 | 4.4905 | 0.9880 | 5.4785 |
| 150 | 5.4261 | 0.0524 | 5.4785 |

The barrier is a brutal price lever. Moving it from 150 to 105, a 30% move in
the barrier, takes 99% of the option's value away.

Monitoring frequency matters nearly as much. On a 20-day up-and-out call at
B=110, half-day steps price it at 1.3189 and whole-day steps at 1.3814, a 4.7%
difference from the observation schedule alone, against a standard error of
0.0019 on both. Discrete monitoring is always worth more than continuous, since
a path can cross the barrier and come back between observations. This is why the
spec's step rule exists and why it is not cosmetic.

### Greeks

Closed-form delta is 0.5567 and vega 0.1975 per volatility point. At 200,000
paths:

| estimator | delta | SE | vega | SE |
|---|---|---|---|---|
| pathwise | 0.5562 | 0.00124 | 0.1974 | 0.00076 |
| likelihood ratio | 0.5577 | 0.00290 | 0.1994 | 0.00278 |
| finite difference, common random numbers | 0.5561 | - | 0.1974 | - |
| finite difference, independent seeds | 0.5513 | - | - | - |

Across 20 independent seeds at 50,000 paths, delta comes out at sd 0.00242
(pathwise), 0.00232 (finite difference with common random numbers) and 0.00489
(likelihood ratio), with biases of +0.00095, +0.00070 and +0.00026 respectively.

Two results worth stating plainly. First, finite difference on common random
numbers is as good as pathwise here, and the single line that makes it work is
reusing the seed across the bumped runs. With independent draws the same
estimator gives 0.5513, and its noise is roughly sqrt(2)/(2h) times the price
standard error: across 20 seeds at 200,000 paths the independent-seed delta has
a standard deviation of 0.0114 against 0.0012 with common random numbers, about
10x worse in standard deviation and 90x in variance. Second, likelihood ratio is twice as noisy as pathwise
on delta, which is the price of its generality: it differentiates the density
rather than the payoff, so it survives discontinuous payoffs where pathwise
fails outright, including my own knock-out.

### Live market check, 2 September 2026

SPY at 765.16. The 30 September call struck at 765, 28 calendar days and 19
trading days out (Labor Day falls inside the window; the day count skips US
federal holidays), quoted 10.05 / 11.11 with 1,552 contracts traded and 2,802
open. Rate 3.77% from `^IRX`. This section is a snapshot: a rerun pulls a new
quote and prints different numbers.

| quantity | value |
|---|---|
| implied vol at bid / mid / ask | 10.56% / 11.19% / 11.83% |
| Yahoo's reported implied vol | 13.04% |
| realised vol, last 252 days | 12.78% |
| MC at the mid implied vol | 10.5882 +/- 0.0070, against a 10.58 mid |
| MC at the realised vol | 11.9106 +/- 0.0080, a +1.33 gap |

The first pricing is circular and I would not present it as anything else: the
vol was backed out of the market price with Black-Scholes and then fed to a
simulator that converges to Black-Scholes. It is a plumbing check that the
root-finder and the engine agree, and it passes to within a standard error.

The second is the interesting one. Pricing with realised volatility instead of
implied gives 11.91 against a 10.58 mid, because implied vol at 11.19% sits
1.6 points below the 12.78% the index actually delivered over the past year.
That gap is the variance risk premium, and it is the reason nobody prices
vanillas with a model in the first place.

Yahoo's own implied vol field disagrees with all three of my quote-based
numbers, and it is not reconcilable: 13.04% implies a price of 12.12, above the
11.11 ask. A 1.1% dividend yield would move my mid implied vol to 11.61% and
using calendar days instead of trading days would move it to 11.07%, so neither
convention explains the gap. I trust the number I backed out of the quote.

Using the same 11.19% vol on things the market does not quote: an up-and-out
call at a barrier of 788 prices at 2.0727 +/- 0.0030, and the arithmetic Asian
at 6.1728 +/- 0.000025.

Charts in `reports/`: `convergence.png`, `standard_error.png`,
`paths_and_payoffs.png`, `greeks.png`.

## Limitations

Constant volatility is the big one. Everything here assumes a single sigma, so
the engine cannot reproduce a volatility smile, and pricing a strike-dependent
exotic off one at-the-money implied vol is wrong in a way that no amount of
paths will fix. Local or stochastic volatility is the real fix and I have not
built it.

American options are in the underlying spec and are not here. Least-squares
Monte Carlo is the right method and it would round out the payoff coverage;
without it, the engine cannot price early exercise.

The control-variate beta is estimated on the same sample it is applied to, which
introduces a small downward bias in the variance estimate. A pilot run for beta
would remove it. At a million paths the effect is negligible; at a thousand it
would not be.

Barriers are monitored discretely at the simulation steps and I have not
implemented the Broadie-Glasserman-Kou continuity correction, so these prices
are for discretely monitored contracts only. The 4.7% swing between daily and
half-daily monitoring is the size of what that correction addresses.

The Sobol standard errors come from 16 scrambles, which is few enough that the
error bars on the error bars are wide. The fitted convergence rates are read off
five grid points and should be treated as indicative.

On the market side, spot is Yahoo's last traded price while the option quote is
whatever Yahoo last saw, so the two are not synchronised (an earlier version
read spot from the daily-close history, which after hours can lag the chain by
a full session and pick the wrong strike as at-the-money), and a wide bid-ask
(10.05 / 11.11, over 10% of mid) means the implied vol is a 1.2-point band
rather than a number. The rate is a 13-week bill against a 20-day option, and
SPY's dividend is set to zero, both of which push the implied vol down by
roughly 0.4 points.
