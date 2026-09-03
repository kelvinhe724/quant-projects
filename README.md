# Quant projects

Five projects from the OSG Global Quant Curriculum, each implemented from the
spec rather than from a reference solution. They run on real market data:
Yahoo Finance for equity prices, FRED for Treasury yields and the T-bill rate,
and the UCI archive for the credit dataset. Every project has an offline
`check.py` that fits the same models to simulated data with planted parameters,
so the code can be verified without a network connection.

| project | what it does | headline result |
|---|---|---|
| `capm/` | Alpha, beta, R² and rolling beta for 10 US stocks against the S&P 500 | Betas 0.10 (JNJ) to 1.46 (AMZN); no alpha significant at 5% |
| `garch/` | GARCH(1,1) volatility on four equity indices, normal vs student-t, VaR backtest | S&P persistence 0.968, FTSE 0.930; student-t wins on AIC/BIC everywhere |
| `nelson-siegel/` | Dynamic Nelson-Siegel on the US Treasury curve, ridge and L1 penalties on daily factor changes | Ridge cuts factor churn 43% full-sample, 19% on the untouched eval period, at ~0.2 bp of eval RMSE |
| `gradient-boosting/` | LightGBM against a logistic baseline on credit-card default | AUC 0.784 vs 0.747 baseline; 5-fold CV 0.7851 ± 0.0084 |
| `pairs-trading/` | Cointegration pair screening on 182 S&P 500 names, strict formation/out-of-sample split | Sharpe 3.01 in-sample, 0.08 out of sample; 99 pairs pass p<0.05 against 74 expected by chance |

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then from any project directory:

```
../.venv/bin/python3 check.py    # offline
../.venv/bin/python3 run.py      # real data, writes to reports/
```

Nelson-Siegel caches its FRED pull and gradient boosting caches its download, so
only the first `run.py` in those two needs internet.

Each project has its own README with the data source, the full result tables and
what I would do differently.
