# Dynamic Nelson-Siegel on the US Treasury curve

Fits a Nelson-Siegel curve (level, slope, curvature, decay) to the US Treasury
constant-maturity yield curve every trading day from 2015 to 2025. Rather than
fit each day independently, dates are processed chronologically: each day
warm-starts from the previous solution and pays a ridge or smooth-L1 penalty on
scaled parameter *changes*. The aim is a factor path that stays quiet when the
market is quiet but still follows real moves. The 2020 collapse to the zero
floor and the 2022-23 inversion are both in the sample, so there is plenty to
follow.

## Data

Eleven FRED series, DGS1MO through DGS30, covering maturities from one month to
30 years. Pulled with `pandas-datareader` and cached to
`reports/treasury_curve.csv`, so every rerun after the first is offline. Series
are forward-filled individually and days with fewer than six available
maturities are dropped. The real pull came back with 31,559 observations over
2,869 trading days, 2015-01-02 to 2025-12-31, with no day missing a maturity.

A Brent futures path sits behind `run.py --brent`. It uses a Bloomberg CSV at
`../source-material/brent_settles.csv` if one is present and a clearly labelled
synthetic panel otherwise. That synthetic generator also backs `check.py`.

## Files

- `data.py` FRED loader and curve slices; Brent contract map, expiry calendar, panel loader
- `ns.py` NS basis, static daily fits, sequential fits with change penalties
- `check.py` offline checks: planted curve recovery, smoothness against a random-start benchmark, responsiveness to a level shift
- `run.py` benchmark, chronological tuning, untouched final eval, charts

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
../.venv/bin/python3 run.py --brent
```

`run.py` takes about a minute.

## Results

Penalty strength was tuned on the first 75% of the sample and scored on the
untouched remainder.

| model | full-sample RMSE | eval RMSE | mean daily factor change | lambda sd |
|---|---|---|---|---|
| static, random starts | 5.25 bp | 6.50 bp | 7.74 bp | 1.89 |
| ridge change penalty | 5.25 bp | 6.53 bp | 4.98 bp | 1.50 |
| L1 change penalty | 5.22 bp | 6.48 bp | 5.28 bp | 1.59 |

Same fit quality, roughly 35% less day-to-day factor churn. That is the whole
result: the penalty buys stability at no cost in cross-sectional fit.

The factors mean what they are supposed to. corr(beta0, actual 10y yield) is
+0.9463 for ridge, +0.9511 static, +0.9461 for L1. corr(-beta1, 10y minus 3m
spread) is +0.98 across all three. The level factor is the long yield and the
slope factor is the term spread, which is the check worth running before
trusting any of the rest.

Charts in `reports/`: `fitted_curves.png` (a normal 2019 curve at 5.1 bp, the
2020 zero floor at 2.4 bp, the 2023 inversion at 14.5 bp), `parameter_paths.png`
(level, slope, curvature and lambda, with the actual 10y overlaid on beta0),
`parameter_changes.png` (daily movement on a log scale plus a per-factor
comparison), `rmse.png`.

## Limitations

Four-factor Nelson-Siegel cannot bend enough for the sharply humped 2023
inversion. It fits at 14.5 bp on 2023-07-03 against 2.4 bp on the flat 2020
curve, and that gap is structural, not a tuning problem. Svensson's second hump
term is the standard fix and I did not implement it.

On real data the tuner picks the weakest strength in the grid (1e-5 for both
penalties), because real curves move enough every day that the fit term
dominates the grid score. I left that honest rather than reweighting the score
until the penalty looked more important. The stability gain shows up anyway.
