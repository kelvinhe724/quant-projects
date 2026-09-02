# GARCH(1,1) on four equity indices

Fits GARCH(1,1) to daily log returns of the S&P 500, EURO STOXX 50, Nikkei 225
and FTSE 100, then compares volatility persistence (alpha + beta) across the
four markets. Both normal and student-t innovations are fit and scored on
AIC/BIC, with a 5% VaR hit-rate backtest on the student-t fits.

Sample is 2015-2026. Returns are scaled x100 so the optimizer works on numbers
near 1.

## Data

Index prices from Yahoo Finance via `yfinance`, adjusted closes. The four
indices trade on different calendars, so each series keeps its own days rather
than being forced onto a common one, which is why the return counts differ.
`check.py` runs offline on a simulated GARCH series with known parameters.

## Files

- `data.py` index downloads, log returns
- `garch.py` `fit_garch`, `fit_all`, `var_backtest`
- `check.py` offline checks against planted parameters
- `run.py` full pipeline, charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

## Results

Normal-innovation fits:

| index | omega | alpha | beta | persistence | days |
|---|---|---|---|---|---|
| FTSE 100 | 0.0590 | 0.176 | 0.755 | 0.931 | 2,718 |
| S&P 500 | 0.0389 | 0.177 | 0.793 | 0.970 | 2,671 |
| Nikkei 225 | 0.0989 | 0.154 | 0.788 | 0.942 | 2,553 |
| EURO STOXX 50 | 0.0693 | 0.166 | 0.787 | 0.953 | 2,693 |

The S&P 500 is the most persistent at 0.970. A volatility shock there decays
with a half-life of about 23 days, against 10 days for the FTSE at 0.931. The
Nikkei is the most volatile in level terms (daily sd 1.30 vs the FTSE's 0.97)
but not the most persistent, which is a useful reminder that the two are
different questions.

Student-t beats normal on both AIC and BIC for all four indices, by a wide
margin: S&P AIC 6712.9 vs 6902.4, FTSE 6414.6 vs 6612.6, Nikkei 7807.4 vs
7973.6, EURO STOXX 7587.4 vs 7780.1. Daily equity returns have fat tails even
after conditioning on GARCH volatility.

The 5% VaR backtest comes in a little conservative everywhere: FTSE 4.67%, S&P
4.16%, Nikkei 4.50%, EURO STOXX 4.20% against an expected 5.0%. The model
over-reserves slightly, which is the direction you would rather err in, but it
means the t-distribution's tails are a shade too heavy for this sample.

Charts in `reports/`: `returns.png`, `conditional_vol.png`, `persistence.png`.

## Limitations

The VaR backtest is in-sample, which flatters it. A rolling out-of-sample
refit with a Kupiec or Christoffersen test is the honest version, and I did not
build it. GARCH(1,1) is symmetric, so it misses the leverage effect entirely.
GJR-GARCH or EGARCH would pick up that down moves raise tomorrow's variance more
than up moves of the same size. Nothing here is a forecast; every number is a
description of the fitted sample.
