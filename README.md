# Quant projects

Eleven projects from the OSG Global Quant Curriculum, each implemented from the
spec rather than from a reference solution. They run on real market data:
Yahoo Finance for equity prices, the VIX and option chains, FRED for Treasury
yields and the T-bill rate, the UCI archive for the credit dataset, Binance's
public archives for perpetual funding, football-data.co.uk for bookmaker odds
and the Kalshi API for prediction-market contracts. Every
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
| `funding-carry/` | Spot-long, perp-short funding carry on Binance BTCUSDT and ETHUSDT, 2020-2026, 7,211 eight-hour periods, fees, margin buffer and liquidation modelled | BTC funding 11.86%/yr simple, 92% of it the exchange's 0.01% floor; net Sharpe over T-bills 5.09 BTC, 5.86 ETH; excess over cash +12.6% in 2020, +23.7% in 2021, then +0.7%, +0.3%, +3.6%, -0.7%, -3.1% |

### Derivatives & Volatility

| project | what it does | headline result |
|---|---|---|
| `vol-risk-premium/` | Implied vs subsequent realised vol on SPY 2010-2026, GARCH forecast, delta-hedged short straddle with cost and tail accounting | Gap +3.78 vol points, Newey-West t 8.35; strategy 9.55%/yr net, Sharpe 1.74, skew -5.2, max drawdown -13.6%; dropping the worst 1% of days lifts Sharpe to 3.75 |

### Market Microstructure

| project | what it does | headline result |
|---|---|---|
| `market-making/` | Simulated maker with informed order flow: naive, inventory-skewed and Avellaneda-Stoikov quotes over 1,000 sessions, P&L split into spread, adverse selection and inventory | Per-session Sharpe 3.65 naive vs 7.15 Avellaneda-Stoikov; naive gives back 4.81 a session to informed flow; at 100% informed all three makers lose |

### Prediction & Betting Markets

| project | what it does | headline result |
|---|---|---|
| `betting-markets/` | Bookmaker efficiency on 37,725 football matches, six leagues, 2010-2026, up to eleven books: margin, favourite-longshot bias, open vs close, betting rules | Margin 2.7% Pinnacle, 4.4-7.6% retail; closing beats opening by 0.00154 Brier, t = -5.89; longshot gradient t = -3.43 at quoted odds, -1.36 de-vigged; the only positive rule is +1.27% taking the best price across books and -6.06% at consensus price |
| `prediction-markets/` | Calibration and favourite-longshot bias on settled Kalshi binaries, out-of-sample fade of the fitted curve with quoted spreads | Below-50c mid overstates YES by 2.8c pooled (t = 7.7), 0.8c (t = 1.9) in books 5c wide or tighter; fade at mid +2.2c/$ (t = 2.70), tight books only -0.3c, at quoted spread -1.4c (t = -1.45) |

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
momentum caches its price panel, vol-risk-premium caches its daily series and
option chain, and funding-carry, betting-markets and prediction-markets cache
their downloads, so only the first `run.py` in those needs internet.

Each project has its own README with the data source, the full result tables and
what I would do differently.
