# Portfolio construction: does optimisation beat 1/N out of sample?

DeMiguel, Garlappi and Uppal (2009) asked a simple question: given how noisy
return estimates are, does a mean-variance optimiser actually beat splitting the
money equally? Their answer was no, not with the amount of data anyone has. This
project reruns that test on fourteen ETFs from 2005 to 2026 with a rolling
out-of-sample design, and adds the fixes practitioners usually reach for:
long-only constraints, minimum variance, risk parity, and shrinkage estimators.

The short version: unconstrained sample mean-variance is a disaster, shrinkage
closes most of the gap, and nothing beats 1/N by a margin that survives a
significance test. The most useful output is the break-even calculation, which
says how many months of history the optimiser would need before it should be
expected to win.

## Data

Fourteen ETFs, monthly total returns from adjusted closes via yfinance: the nine
original SPDR sectors (XLK XLF XLE XLV XLI XLP XLU XLY XLB), long and
intermediate Treasuries (TLT IEF), gold (GLD), developed and emerging ex-US
equity (EFA EEM). The risk-free rate is the 3-month T-bill from FRED (TB3MS),
converted to a monthly rate. All returns below are in excess of that rate.

The panel runs 2004-12 to 2026-08, 261 months, limited on the left by GLD's
November 2004 launch. Both downloads are cached under
`source-material/portfolio-construction/` so reruns are offline.

## Design

At the end of every month t, each rule estimates whatever it needs from the M
months ending at t, forms a weight vector, and holds it through month t+1. The
weights are formed from returns up to and including t and never touch t+1. The
turnover charged at each rebalance is the gap between the new target and where
the previous target has drifted to after month t's returns, so a rule that never
changes its target still pays for drift. Costs are 10bps one way on traded
notional. Two windows are run, 60 and 120 months. A 60-month window means the
first out-of-sample month is 2009-12; 120 means 2014-12.

Rules, all fully invested:

| Rule | Estimate | Weights |
|---|---|---|
| 1/N | nothing | 1/14 each, rebalanced monthly |
| tangency | sample mean, sample cov | inv(S) m, scaled to sum one, no constraints |
| tangency long-only | same | max Sharpe with weights in [0, 1] |
| min variance | sample cov | inv(S) 1, scaled, no constraints |
| inverse vol | sample vol | 1/sigma_i, scaled |
| ERC | sample cov | every asset contributes 1/14 of variance |
| shrunk tangency | Ledoit-Wolf cov, Jorion means | tangency on the shrunk moments |
| shrunk min variance | Ledoit-Wolf cov | min variance on the shrunk cov |

Ledoit-Wolf shrinks the sample covariance toward a scaled identity with an
intensity chosen from the data (sklearn's implementation). The mean shrinkage is
Jorion's Bayes-Stein rule: pull each sample mean toward the grand mean with
intensity (N+2) / ((N+2) + T d), where d is the Mahalanobis spread of the sample
means. It is a James-Stein estimator with a data-driven intensity.

`check.py` runs offline on simulated returns with known moments. It confirms
the tangency function matches the closed form and that no random perturbation of
its output has a higher Sharpe; that tangency estimated on 400k simulated months
recovers the true weights; that 1/N sums to one; that ERC's risk contributions
are equal to 1e-6; that Ledoit-Wolf lands strictly between the sample covariance
and its target and is closer to the truth on 36 months; that turnover matches a
two-asset case worked by hand; that altering returns after t leaves every weight
set at or before t byte-identical; and that on planted moments the plug-in
tangency portfolio loses to 1/N with 24 months of data and wins with 2400. It
exits nonzero if anything fails.

## Results

All numbers from `run.py`. Sharpe ratios are annualised from monthly excess
returns. The p-value is the Jobson-Korkie test with Memmel's correction for the
difference in net Sharpe against 1/N.

The full-sample tangency portfolio, computed with perfect hindsight on all 261
months, has a Sharpe of 1.29 against 0.68 for 1/N. That 0.61 gap is what an
optimiser is trying to capture. Out of sample, this is what it actually gets.

### 60-month window, 2009-12 to 2026-08 (201 months)

| | 1/N | tangency | tangency long-only | min variance | inverse vol | ERC | shrunk tangency | shrunk min variance |
|---|---|---|---|---|---|---|---|---|
| Excess return, gross | +9.21% | +65.74% | +8.72% | +1.91% | +8.21% | +6.99% | +10.94% | +5.17% |
| Excess return, net | +9.17% | +5.36% | +8.50% | +1.69% | +8.17% | +6.94% | +10.46% | +5.02% |
| Vol | 11.14% | 665.90% | 9.68% | 4.46% | 9.47% | 7.92% | 10.98% | 6.26% |
| Sharpe, gross | 0.83 | 0.10 | 0.90 | 0.43 | 0.87 | 0.88 | 0.99 | 0.83 |
| Sharpe, net | 0.82 | 0.01 | 0.88 | 0.38 | 0.86 | 0.88 | 0.95 | 0.80 |
| Max drawdown, net | -18.0% | ruin | -26.8% | -17.1% | -18.9% | -19.0% | -23.5% | -22.5% |
| Monthly turnover | 0.031 | 50.3 | 0.188 | 0.190 | 0.033 | 0.038 | 0.396 | 0.128 |
| Avg gross exposure | 1.00 | 23.2 | 1.00 | 2.53 | 1.00 | 1.00 | 2.16 | 1.46 |
| Largest weight | 0.07 | 298.7 | 0.74 | 1.67 | 0.20 | 0.40 | 1.13 | 0.76 |
| p-value vs 1/N | | 0.02 | 0.79 | 0.10 | 0.35 | 0.62 | 0.63 | 0.91 |

### 120-month window, 2014-12 to 2026-08 (141 months)

| | 1/N | tangency | tangency long-only | min variance | inverse vol | ERC | shrunk tangency | shrunk min variance |
|---|---|---|---|---|---|---|---|---|
| Excess return, gross | +8.09% | +85.85% | +8.79% | +0.70% | +6.81% | +5.48% | +9.06% | +2.44% |
| Excess return, net | +8.05% | +65.52% | +8.64% | +0.56% | +6.77% | +5.44% | +8.78% | +2.33% |
| Vol | 11.54% | 112.27% | 10.81% | 4.12% | 10.14% | 8.59% | 10.15% | 5.97% |
| Sharpe, gross | 0.70 | 0.72 | 0.81 | 0.17 | 0.67 | 0.64 | 0.89 | 0.41 |
| Sharpe, net | 0.70 | 0.58 | 0.80 | 0.14 | 0.67 | 0.63 | 0.86 | 0.39 |
| Max drawdown, net | -18.0% | -90.3% | -23.4% | -15.0% | -18.9% | -18.8% | -23.9% | -21.0% |
| Monthly turnover | 0.034 | 16.9 | 0.127 | 0.118 | 0.033 | 0.035 | 0.236 | 0.086 |
| Avg gross exposure | 1.00 | 19.8 | 1.00 | 2.39 | 1.00 | 1.00 | 2.19 | 1.45 |
| Largest weight | 0.07 | 55.0 | 0.71 | 1.59 | 0.18 | 0.31 | 0.75 | 0.88 |
| p-value vs 1/N | | 0.78 | 0.65 | 0.07 | 0.49 | 0.57 | 0.54 | 0.14 |

"Ruin" means the book had at least one month with a return below -100%, after
which a drawdown figure has no meaning. The unconstrained tangency portfolio runs
at 20x to 23x gross, turns over 17 to 50 times its capital every month, and its
"gross" +66% a year at 666% vol in the 60-month run is an artefact of averaging
a series that includes single months of +2270% and -1000%. Part of the mechanism
is the normalisation itself: inv(S) m is scaled to sum to one, and in 43 of the
201 sixty-month windows that sum is negative, which flips the whole book, while
a sum near zero is what produces a 299x single weight. This is the DeMiguel
result in its rawest form and it is not a coding error: the same function
recovers the analytic tangency weights to 1e-12 in `check.py`. It is what
inv(S) does to sampling noise when the assets are as correlated as nine equity
sectors. It is also chaotic: adding one basis point of random noise to the
monthly returns moves the 60-month net Sharpe from 0.01 to about -0.3 and the
annual excess return from +5% to somewhere between -60% and -240%, while every
other column in these tables is unchanged to the second decimal. Nothing in the
tangency column should be quoted as a number; the column says "ruin" and that
is all it says.

What I take from the two tables:

- **Nothing beats 1/N significantly.** The only p-values at or below 0.10
  belong to rules that lose to it (unconstrained tangency at 60 months, min
  variance in both windows). Shrunk tangency
  has the best net Sharpe in both windows, 0.95 and 0.86, but the p-values are
  0.63 and 0.54. With 141 to 201 months and Sharpe ratios near 0.8, the standard
  error of a Sharpe difference is about 0.3, so this test could not have detected
  the gap anyway.
- **Constraints do most of the work.** Long-only tangency goes from ruin to a
  net Sharpe of 0.88, roughly 1/N, with half the turnover of the shrunk
  version. Jagannathan and Ma's point that constraints are shrinkage in disguise
  shows up directly.
- **Shrinkage works, but it is still levered.** Shrunk tangency has a gross
  exposure of 2.2 and a largest single weight above 1.0 in the 60-month run. A
  Sharpe of 0.95 with a 2x levered long-short book is not the same product as
  1/N at 0.82 with no leverage, and the cost table below shows the gap closes at
  50bps.
- **Minimum variance is the odd failure.** It has the lowest vol, as it should,
  but an excess return of 1.7% a year, because it is 2.5x gross with heavy
  positions in IEF against TLT and hedges away almost all the equity premium. It
  minimises variance, not the thing anyone wants minimised.
- **Risk parity is 1/N with extra steps.** Inverse vol and ERC both land within
  0.05 Sharpe of 1/N in both windows, at similar turnover. They tilt toward bonds,
  which lowered vol and returns in equal proportion over this sample.

### Costs

Net Sharpe, 60-month window, by one-way cost:

| | 0bps | 10bps | 25bps | 50bps |
|---|---|---|---|---|
| 1/N | 0.83 | 0.82 | 0.82 | 0.81 |
| tangency | 0.10 | 0.01 | -0.11 | -0.23 |
| tangency long-only | 0.90 | 0.88 | 0.84 | 0.78 |
| min variance | 0.43 | 0.38 | 0.30 | 0.17 |
| inverse vol | 0.87 | 0.86 | 0.86 | 0.85 |
| ERC | 0.88 | 0.88 | 0.87 | 0.85 |
| shrunk tangency | 0.99 | 0.95 | 0.89 | 0.79 |
| shrunk min variance | 0.83 | 0.80 | 0.76 | 0.70 |

At 50bps, which is not unreasonable once you count spread and impact on the less
liquid ETFs, every optimiser is at or below 1/N. 1/N's monthly turnover of 3%
is pure drift; it barely notices costs.

### The break-even window

DeMiguel et al.'s central calculation asks how long the estimation window has
to be before the sample tangency portfolio's expected out-of-sample performance
exceeds 1/N. I answer it two ways.

**Simulation.** Fix the true means and covariance at their full-sample values,
draw M months of normal returns, estimate the tangency weights from the draw,
and score those weights on the true moments. Repeat 2000 times per M. This is
the counterpart of the paper's Section 2 calculation, using Sharpe instead of
certainty-equivalent utility.

| Window (months) | 24 | 36 | 48 | 60 | 72 | 84 | 90 | 96 | 120 | 180 | 240 | 360 | 720 | 2400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Expected Sharpe of sample tangency | 0.22 | 0.33 | 0.43 | 0.53 | 0.60 | 0.65 | 0.69 | 0.73 | 0.81 | 0.95 | 1.03 | 1.12 | 1.20 | 1.26 |
| Share of draws beating 1/N | 13% | 27% | 41% | 53% | 63% | 69% | 75% | 78% | 87% | 96% | 99% | 100% | 100% | 100% |

1/N's true Sharpe on the same moments is 0.68. **The break-even is 90 months:
seven and a half years of monthly data before the optimiser is expected to do
better than not optimising.** With the 60-month window most practitioners use,
it wins only 53% of the time and its expected Sharpe is 0.53, below 1/N. Even at
240 months, twenty years, it captures only 1.03 of the 1.29 available. Half the
gap between 1/N and the true optimum is still missing at ten years.

This break-even is optimistic. The simulation assumes iid normal returns with
moments that never move, and it hands the simulated estimator the true full-sample
tangency Sharpe of 1.29 as the prize. Fat tails and a shifting covariance both
make real estimates worse.

**Real data.** Net out-of-sample Sharpe by estimation window. Each row is
evaluated on its own out-of-sample period, which shrinks as the window grows.

| Window | OOS months | 1/N | tangency | tangency long-only | min variance | shrunk tangency |
|---|---|---|---|---|---|---|
| 24 | 237 | 0.64 | -0.25 | 0.62 | 0.16 | -0.35 |
| 36 | 225 | 0.62 | -0.25 | 0.66 | 0.21 | 0.24 |
| 48 | 213 | 0.86 | -0.40 | 0.86 | 0.32 | -0.12 |
| 60 | 201 | 0.82 | 0.01 | 0.88 | 0.38 | 0.95 |
| 84 | 177 | 0.83 | -0.31 | 0.83 | 0.35 | 0.95 |
| 120 | 141 | 0.70 | 0.58 | 0.80 | 0.14 | 0.86 |
| 150 | 111 | 0.69 | 0.85 | 0.67 | 0.02 | 0.80 |
| 180 | 81 | 0.67 | 0.90 | 0.71 | -0.01 | 0.76 |
| 240 | 21 | 1.09 | 1.08 | 1.34 | 0.36 | 1.13 |

On real data the sample tangency portfolio loses to 1/N at every window up to
120 months, consistent with the simulated break-even. It pulls ahead at 150 and
180 months, but those rows cover 111 and 81 out-of-sample months, and the 150
row's Sharpe of 0.85 comes with 5.5x gross exposure, a largest weight of 4.5 and
140% monthly turnover before the numbers are netted at 10bps. The 240 row is 21
months and is there for completeness only. Shrunk tangency needs about 60
months before its estimates are stable enough to help; below 60 the covariance
shrinkage cannot save a mean estimate built on two to four years, and the
column's sign flips from window to window.

## Limitations

**Survivorship in asset selection.** All fourteen ETFs exist today and had a
clean 21-year history, which is why I picked them. Asset classes are less prone
to this than single stocks, but the choice was still made in 2026 knowing what
survived. The equal-weight benchmark inherits the same selection, so the
comparison between rules is fair; the absolute Sharpe levels are not. For
scale, the `pit-universe` project measured the single-stock version of this on
the S&P 500: an equal-weight basket of today's members applied backwards beats
the point-in-time basket by +3.97%/yr in-sample and +4.23%/yr in the final
test, about 3.2 points of it from holding future members before they joined
and 0.7 from deleted losers, and a lower bound because 257 of the 390 names
ever removed have no Yahoo prices. Fourteen sector and asset-class ETFs carry
far less of that than a stock list, and I have not measured how much.
`data.monthly_panel(universe="pit")` loads that stock panel here.

**A discount-basis, lagged risk-free rate.** TB3MS is the monthly average of
the 3-month bill's discount yield, dated by FRED to the first of its month. I
convert it to a monthly compound rate and subtract the previous month's value
from each month's return, so the rate is always one that was known before the
month began. The difference against a bond-equivalent yield is a couple of
basis points a year and does not affect any ranking.

**The p-values are low-powered.** A Jobson-Korkie test on 141 to 201 months
cannot separate Sharpe ratios 0.1 apart. "Not significant" here means "I cannot
tell", not "equal".

**Fourteen assets is small.** DeMiguel et al. show the break-even window grows
roughly linearly in N. Ten sector ETFs behave like a 14-asset problem, but a
25-industry or 100-stock problem would push the 90 months to several hundred.
This project measures the friendly case for optimisation.

**Costs are flat.** 10bps one way on every ETF ignores that EEM, XLB and XLU are
wider than XLK, and ignores impact. The 50bps column is the honest reading for
the levered rules.

**One rebalance frequency, one cost model, no transaction-cost-aware optimiser.**
Turnover penalties inside the objective would help shrunk tangency most and are
the obvious next step.

## Running it

```
python3 check.py     offline tests on simulated moments, exits nonzero on failure
python3 data.py      downloads or reads the cache, prints the panel summary
python3 run.py       full pipeline, writes tables and charts to reports/
```

Dependencies are in the shared `requirements.txt` one level up.

`data.monthly_panel(universe="pit")` returns month-end returns, the same
risk-free rate and excess returns for the point-in-time S&P 500 panel in
`../pit-universe`, one column per name that was ever a member and NaN in the
months it was not one. The rules in `run.py` need a complete panel, so a caller
picks a window and drops names before optimising. The default is `"etf"`, the
fourteen names above, so `run.py` reproduces the numbers unchanged.

## Files

- `data.py` downloads and caches prices and the T-bill rate, builds monthly excess returns
- `portfolio.py` the eight rules and the two shrinkage estimators
- `backtest.py` rolling estimation, drift-aware turnover, costs, metrics, the Sharpe test
- `check.py` planted-truth tests
- `run.py` real-data pipeline
- `reports/` output tables (`oos_60.csv`, `oos_120.csv`, `cost_sensitivity.csv`,
  `break_even_empirical.csv`, `break_even_simulated.csv`, one weight path per
  rule and window) and charts

## Reference

DeMiguel, V., Garlappi, L., Uppal, R. (2009). Optimal versus naive
diversification: how inefficient is the 1/N portfolio strategy? Review of
Financial Studies 22(5).
