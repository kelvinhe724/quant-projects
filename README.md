# Quant projects

Eight projects from the OSG Global Quant Curriculum, each implemented from the
spec rather than from a reference solution. They run on real market data:
Yahoo Finance for equity prices, the VIX and option chains, FRED for Treasury
yields and the T-bill rate, and the UCI archive for the credit dataset. Every
project has an offline `check.py` that fits the same models to simulated data
with planted parameters, so the code can be verified without a network
connection.

### Estimation

| project | what it does | headline result |
|---|---|---|
| `capm/` | Alpha, beta, R² and rolling beta for 10 US stocks against the S&P 500 | Betas 0.10 (JNJ) to 1.46 (AMZN); no alpha significant at 5% |
| `garch/` | GARCH(1,1) volatility on four equity indices, normal vs student-t, VaR backtest | S&P persistence 0.968, FTSE 0.930; student-t wins on AIC/BIC everywhere |
| `nelson-siegel/` | Dynamic Nelson-Siegel on the US Treasury curve, ridge and L1 penalties on daily factor changes | Ridge cuts factor churn 43% full-sample, 19% on the untouched eval period, at ~0.2 bp of eval RMSE |
| `gradient-boosting/` | LightGBM against a logistic baseline on credit-card default | AUC 0.784 vs 0.747 baseline; 5-fold CV 0.7851 ± 0.0084 |

### Factors & Strategies

| project | what it does | headline result |
|---|---|---|
| `pairs-trading/` | Cointegration pair screening on 182 S&P 500 names, strict formation/out-of-sample split | Sharpe 3.01 in-sample, 0.08 out of sample; 99 pairs pass p<0.05 against 74 expected by chance |
| `momentum/` | 12-1 cross-sectional momentum vs 50/200 crossover on 190 US large caps, 2005-2026, one-day lag and costs, untouched 2021+ test | Long-short 12-1 net Sharpe -0.03 in-sample, 0.12 out of sample; long-only crossover 0.76 / 1.02; a split-adjusted $5 price floor was a look-ahead and was removed |

### Derivatives & Volatility

| project | what it does | headline result |
|---|---|---|
| `vol-risk-premium/` | Implied vs subsequent realised vol on SPY 2010-2026, GARCH forecast, delta-hedged short straddle with cost and tail accounting | Gap +3.78 vol points, Newey-West t 8.35; strategy 9.55%/yr net, Sharpe 1.74, skew -5.2, max drawdown -13.6%; dropping the worst 1% of days lifts Sharpe to 3.75 |

### Market Microstructure

| project | what it does | headline result |
|---|---|---|
| `market-making/` | Simulated maker with informed order flow: naive, inventory-skewed and Avellaneda-Stoikov quotes over 1,000 sessions, P&L split into spread, adverse selection and inventory | Per-session Sharpe 3.65 naive vs 7.15 Avellaneda-Stoikov; naive gives back 4.81 a session to informed flow; at 100% informed all three makers lose |

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

Nelson-Siegel caches its FRED pull, gradient boosting caches its download,
momentum caches its price panel, and vol-risk-premium caches its daily series
and option chain, so only the first `run.py` in those needs internet.

Each project has its own README with the data source, the full result tables and
what I would do differently.
