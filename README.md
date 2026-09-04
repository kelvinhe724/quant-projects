# Quant projects

Twenty-nine projects from the OSG Global Quant Curriculum, each implemented from the
spec rather than from a reference solution. They run on real market data:
Yahoo Finance for equity prices, the VIX and option chains, FRED for Treasury
yields and the T-bill rate, the UCI archive for the credit dataset, Binance's
public archives for perpetual funding, four spot venues polled live for cross-exchange gaps, football-data.co.uk for bookmaker odds
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
| `fama-french/` | CAPM, FF3, FF5 and FF5+MOM time-series regressions on the 25 size/value portfolios and ten stocks, 1963-2026, GRS joint alpha test, inverse-vol blend of the long-short factors, pre/post-2010 split | FF3 lifts mean R² from 0.73 to 0.91 and halves mean abs alpha; every model rejected by GRS (FF3 3.66, p 7e-9); blend Sharpe 0.80 vs market 0.45 full sample, 0.22 vs 0.88 post-2010; SMB, HML, RMW, CMA all lose significance after 2010 |
| `kalman-pairs/` | Kalman-filter hedge ratios on the 35 pairs and rules from `pairs-trading/`, state noise tuned on the 2015-2019 formation window, untouched 2020+ test | Adaptive hedge scores net out-of-sample Sharpe -0.16 against 0.08 for the frozen OLS beta; nothing in the state-noise grid beats it, the top of the grid finds gross Sharpe 0.51 and spends all of it on turnover |
| `pit-universe/` | Point-in-time S&P 500 membership from Wikipedia change rows and 39 dated list snapshots, 893 names 2003-2026, then `momentum/` and `pairs-trading/` rerun on today's list vs the point-in-time panel with the same code, dates and costs | Equal-weight level bias +4.0%/yr, 3.2 of it from holding future members before they joined; 12-1 momentum gross return overstated +2.2%/yr in-sample and +6.3%/yr on the 2021+ test, all from the long leg; pairs screen bias -0.7%/yr, inside noise; 257 of 390 deleted names have no Yahoo prices so every number is a lower bound |

### Derivatives & Volatility

| project | what it does | headline result |
|---|---|---|
| `vol-risk-premium/` | Implied vs subsequent realised vol on SPY 2010-2026, GARCH forecast, delta-hedged short straddle with cost and tail accounting | Gap +3.78 vol points, Newey-West t 8.35; strategy 9.55%/yr net, Sharpe 1.74, skew -5.2, max drawdown -13.6%; dropping the worst 1% of days lifts Sharpe to 3.75 |
| `monte-carlo/` | Risk-neutral GBM engine for European, arithmetic Asian and barrier options, antithetic / control-variate / scrambled-Sobol variance reduction, pathwise, likelihood-ratio and finite-difference Greeks, live SPY quote check | European MC matches Black-Scholes 5.4721 inside 2 SE at every path count; Sobol cuts variance 19x (8.9x after wall time), geometric-Asian control variate 3,666x on the Asian; barrier from 150 to 105 removes 99% of value; common-random-number finite difference matches pathwise delta, independent seeds are 10x noisier |
| `svi/` | Raw SVI fits to a live SPY option chain (7 expiries, 407 clean quotes after filtering), implied vols from an own Black-76 solver, Durrleman butterfly and calendar no-arbitrage checks enforced in the fit | Mean slice RMSE 0.25 vol points (worst 0.35); unconstrained fits violate butterfly on 7/7 slices and calendar on 5/6, constrained fits 0/7 and 0/6; the market mids themselves show negative butterflies on 28% of strikes; own IVs vs Yahoo's column RMSE 1.41 vp |

### Anomalies & Regimes

| project | what it does | headline result |
|---|---|---|
| `overnight-anomaly/` | Close-to-open vs open-to-close return decomposition on SPY, QQQ, IWM, nine sector SPDRs and twelve large caps, 2000-2026, Newey-West t-stats, stale-open filter, pre/post-2016 split, two-trades-a-day cost model | SPY +7.1%/yr overnight vs +1.1% intraday (NW t 3.77 vs 0.86); dividends are 24% of the SPY overnight leg; breakeven one-way cost 1.49bps, so the overnight-only trade is worth nothing after costs on every asset but one |
| `dispersion/` | Implied vs realised correlation on SPY and its ten largest names: own Black-Scholes solver on live chains, eleven-year gap history from a VIX-and-trailing-vol proxy, monthly delta-hedged dispersion book with costs, 2015-2020 formation and 2021+ test | Live chains put implied correlation at 0.03 vs realised 0.08; full-sample net Sharpe swings from +0.91 to -1.19 across plausible values of two unobservable proxy constants; what the data does pin down is a -0.76 correlation between monthly P&L and the correlation surprise, skew -0.9, and worst months April 2020 and May 2025 |
| `regime-hmm/` | 2-state Gaussian HMM on daily SPY, 2000-2026, causal forward filter vs smoothed probabilities, expanding-window refit each January from 2010, 5bps costs, benchmarked against 15% vol targeting | Out of sample the HMM lifts net Sharpe 0.86 to 0.90 and cuts max drawdown -33.7% to -13.8%, but 15% vol targeting gets 0.97 and the HMM gives up a third of the return; the in-sample smoothed version shows Sharpe 1.88, which is the trap |
| `yield-curve-recession/` | Estrella-Mishkin probit of NBER recession within 12 months on the 10y-3m spread, 1962-2026, expanding-window out-of-sample scoring, HAC t-stats, Sahm-rule and base-rate benchmarks, month-by-month read of the 2022-24 inversion | In-sample coefficient -0.420 (t -3.43), AUC 0.747; out of sample AUC 0.650, 0.754 with the Sahm gap; 0.455 in 2008-2025; the 2022-24 curve kept the model above 50% for 24 straight months, peaking at 81% in May 2023, with no recession through August 2026 |

### Market Microstructure

| project | what it does | headline result |
|---|---|---|
| `market-making/` | Simulated maker with informed order flow: naive, inventory-skewed and Avellaneda-Stoikov quotes over 1,000 sessions, P&L split into spread, adverse selection and inventory | Per-session Sharpe 3.65 naive vs 7.15 Avellaneda-Stoikov; naive gives back 4.81 a session to informed flow; at 100% informed all three makers lose |
| `crypto-arbitrage/` | Four spot venues polled every three seconds for 55 minutes (1,099 ticks, 8,792 quotes, zero failed requests) plus hourly and daily candles, best executable BTC and ETH gap across twelve ordered venue pairs, fee grid, persistence and simulated fills | Median best gap 1.66 bps BTC, 0.97 bps ETH, never above 5 bps on mids; 0% of samples clear any published taker fee; the zero-fee upper bound is +$107 BTC and +$131 ETH on 1.1 and 0.5 bps a trade, with 91% of BTC and 78% of ETH signals blocked by inventory |

### Prediction & Betting Markets

| project | what it does | headline result |
|---|---|---|
| `betting-markets/` | Bookmaker efficiency on 37,725 football matches, six leagues, 2010-2026, up to eleven books: margin, favourite-longshot bias, open vs close, betting rules | Margin 2.7% Pinnacle, 4.4-7.6% retail; closing beats opening by 0.00154 Brier, t = -5.89; longshot gradient t = -3.43 at quoted odds, -1.36 de-vigged; the only positive rule is +1.27% taking the best price across books and -6.06% at consensus price |
| `prediction-markets/` | Calibration and favourite-longshot bias on settled Kalshi binaries, out-of-sample fade of the fitted curve with quoted spreads | Below-50c mid overstates YES by 2.8c pooled (t = 7.7), 0.8c (t = 1.9) in books 5c wide or tighter; fade at mid +2.2c/$ (t = 2.70), tight books only -0.3c, at quoted spread -1.4c (t = -1.45) |
| `kalshi-desk/` | Live prediction-market desk on top of the longshot-bias research: RSA-PSS signed Kalshi client, public-data scanner scoring every open market by edge after the quoted spread with Kelly sizing, three books (shadow with simulated fills at the recorded touch, demo exchange, production) sharing one ledger and the backtest's statistics, local dashboard | Scanner ranks ~1,000 candidates from 5,000 listed markets; shadow book forward-tests the backtest's finding (+2.2c at mid, -1.4c at the spread) on live markets with no account; production orders are off by default behind a flag, a typed confirmation and hard dollar caps |
| `diamonds/` | 18-term multiple regression on diamond prices, OLS vs ridge/lasso/elastic net, BIC-selected showcase model | No regulariser beats plain OLS out of sample; the apparent quality-grade price inversion is carat confounding |
| `commodities-hedging/` | Cross-hedge screen over 78 commodity futures pairs, integer contract hedge ratios, out-of-sample variance reduction | WTI/Brent hedge cuts 83% of out-of-sample variance; levels regression shows R² 0.45 vs 0.09 on returns, the spurious-regression trap |
| `fx-carry/` | G10 carry: borrow the three lowest yielders, lend the three highest, monthly roll, UIP test | Net Sharpe 0.35 with a −31.7% drawdown in 2007–09; UIP slope 0.05, rejected |
| `portfolio-construction/` | Mean-variance, min-variance, shrinkage and constrained optimisers against 1/N, out of sample | Nothing beats 1/N at p<0.10; unconstrained tangency runs 18–32× gross leverage |
| `trend-following/` | Time-series momentum across asset classes, 12-1 signal, class-balanced volatility targeting | 2008 +28.5% vs SPY −36.8% and 2022 +30.5% vs −18.2%, but 2020 −2.3% on whipsaw; post-2010 Sharpe ≈ 0.3 |
| `orderbook-imbalance/` | Limit-order-book imbalance vs next-seconds mid move on BTC-USDT, by depth and horizon | Depth-1 imbalance predicts the 1s mid move with t = 12.7 and a 73% hit rate, on a 50-minute sample |

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
option chain, and funding-carry, betting-markets, prediction-markets, kalman-pairs and
yield-curve-recession cache their downloads, so only the first `run.py` in those needs internet.

Each project has its own README with the data source, the full result tables and
what I would do differently.

## Portfolio & Execution

| project | what it does | headline result |
|---|---|---|
| `framework/` | The book: a daily target-weight simulator (as-of data guard, next-open fills, commission + spread + impact costs, vol targeting, drawdown and kill overlay) wrapped by skfolio, purgedcv, quantstats and alpaca-py; three ETF-only sleeves ported from the research (trend, FX carry, long/flat crypto trend), an EWMAC rule and a vol-targeted 1/N core, four candidate books, 1/N vs ERC allocation, an engine-level position buffer, purged walk-forward, an untouched final window and deflated Sharpe with the real trial count, a signal-vs-beta attribution per sleeve, a shadow paper daemon and an Alpaca paper bridge | Live book since 2026-09-04 is beta+alpha (1/N of the 14 ETFs at 10% vol plus EWMAC, 1/2 each): net Sharpe 0.81 over 23 years, 5.7%/yr at 7.2% vol, max drawdown -12.1%, DSR 0.63 over 56 trials, 0.79 on the untouched last fifth of the panel against 0.40 for the v1 three-sleeve book; the win is mostly beta (the ETF universe held 1/N scores 0.98 on that window) and EWMAC alone fails its promotion test there (0.41 against TrendETF's 0.87, alpha t 0.75) after 0.68 with alpha t 3.1 on the full panel; moving the 10% buffer into the engine cut EWMAC's trade count 63% but its turnover only 4% (19.3x to 18.5x) and its net Sharpe 0.71 to 0.68, because the turnover is in the large daily moves of its own targets; ERC does not beat 1/N out of sample for any book; the trend sleeve ports exactly (targets match the research project to 0 at all 266 month ends) |
