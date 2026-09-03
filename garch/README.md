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
| FTSE 100 | 0.0599 | 0.171 | 0.759 | 0.930 | 2,777 |
| S&P 500 | 0.0395 | 0.175 | 0.794 | 0.968 | 2,765 |
| Nikkei 225 | 0.1198 | 0.156 | 0.773 | 0.929 | 2,687 |
| EURO STOXX 50 | 0.0625 | 0.154 | 0.805 | 0.959 | 2,762 |

The S&P 500 is the most persistent at 0.968. A volatility shock there decays
with a half-life of about 22 days, against 10 days for the FTSE at 0.930. The
Nikkei is the most volatile in level terms (daily sd 1.32 vs the FTSE's 0.97)
but the least persistent, which is a useful reminder that the two are
different questions.

Student-t beats normal on both AIC and BIC for all four indices, by a wide
margin: S&P AIC 6952.3 vs 7152.2, FTSE 6595.8 vs 6793.2, Nikkei 8338.2 vs
8500.3, EURO STOXX 7834.4 vs 8045.5. Daily equity returns have fat tails even
after conditioning on GARCH volatility.

The 5% VaR backtest uses the parametric quantile of the fitted student-t, mean
included, and comes in hot everywhere: FTSE 6.12%, S&P 6.76%, Nikkei 6.40%,
EURO STOXX 6.84% against an expected 5.0%. The model under-reserves: the fitted
t tails are a shade too thin for this sample, so the 5% line gets crossed more
often than it should. An earlier version of this test scaled by the empirical
quantile of the model's own residuals, which pins the hit rate to 5% by
construction and tests nothing.

Charts in `reports/`: `returns.png`, `conditional_vol.png`, `persistence.png`.

## Limitations

The VaR backtest is in-sample, which flatters it. A rolling out-of-sample
refit with a Kupiec or Christoffersen test is the honest version, and I did not
build it. GARCH(1,1) is symmetric, so it misses the leverage effect entirely.
GJR-GARCH or EGARCH would pick up that down moves raise tomorrow's variance more
than up moves of the same size. Nothing here is a forecast; every number is a
description of the fitted sample.
