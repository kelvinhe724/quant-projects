# CAPM on 10 US stocks

Regresses each stock's daily excess return on the S&P 500's to get alpha, beta,
R-squared and the alpha p-value. The risk-free leg is the 3-month T-bill (DTB3
from FRED) converted to a daily rate. A 60-day rolling beta shows how stable
each exposure is over the sample.

Ten tickers, one per GICS sector: AAPL, JPM, XOM, JNJ, PG, CAT, AMZN, NEE, DIS,
LIN. Daily returns, 2023-01 to 2026-01.

## Data

Prices from Yahoo Finance via `yfinance`, adjusted for dividends and splits.
Risk-free rate from FRED via `pandas-datareader`. Both need internet; `check.py`
runs offline on simulated data with a planted beta.

## Files

- `data.py` price and T-bill downloads, excess returns
- `capm.py` `fit_capm`, `fit_all`, `rolling_beta`
- `check.py` offline checks against a known beta
- `run.py` full pipeline, table and charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

## Results

| ticker | beta | annualised alpha | R² | alpha p-value |
|---|---|---|---|---|
| JNJ | 0.10 | +3.3% | 0.007 | 0.74 |
| PG | 0.17 | -5.4% | 0.025 | 0.56 |
| NEE | 0.43 | -6.4% | 0.057 | 0.68 |
| XOM | 0.45 | -1.7% | 0.089 | 0.89 |
| LIN | 0.61 | -1.8% | 0.265 | 0.84 |
| JPM | 0.92 | +14.9% | 0.360 | 0.16 |
| DIS | 0.95 | -7.2% | 0.276 | 0.60 |
| CAT | 1.17 | +11.9% | 0.371 | 0.37 |
| AAPL | 1.17 | +6.4% | 0.475 | 0.55 |
| AMZN | 1.46 | +10.3% | 0.474 | 0.44 |

The sector story holds up. The defensive names sit well below 1 (JNJ 0.10, PG
0.17, NEE 0.43), the cyclicals and tech above it (CAT 1.17, AAPL 1.17, AMZN
1.46). Not one alpha is significant at 5%. JPM's +14.9% is the closest at p = 0.16,
and that is the point: three years of daily data is not enough to
separate skill from noise.

R² ranges from 0.007 (JNJ) to 0.475 (AAPL). Low-beta staples are almost entirely
idiosyncratic; the market explains under 1% of JNJ's daily variance. The rolling
beta chart shows AMZN and CAT swinging by roughly half a point across 60-day
windows, so the full-sample number is an average, not a constant.

Charts in `reports/`: `scatter_grid.png`, `residuals.png`, `rolling_beta.png`,
`beta_bars.png`. Table in `capm_table.csv`.

## Limitations

The betas here are OLS on daily data, which understates beta for less liquid
names because of non-synchronous trading. Dimson or Scholes-Williams lags would
correct that, and I did not do it. There is no correction for heteroskedasticity
in the standard errors either, so the alpha p-values are optimistic; Newey-West
would be the fix. Three years is a short sample. The rolling beta is a plain
cov/var ratio rather than a Kalman filter, which is the more defensible way to
track a time-varying exposure.
