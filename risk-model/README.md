# Risk model

A factor risk model for the desk: four covariance estimators raced on the out-of-sample vol of the minimum-variance portfolio they imply, over the point-in-time S&P 500 and the book's 14 ETFs, then the live premia book's exposures, risk contributions, 1-day VaR and stress replays, behind a three-function API that `framework/book` can import.

The one-line honest summary: on the full point-in-time panel the PCA model with the factor count set by the Marchenko-Pastur edge wins (11.4% realised vol on the untouched 2022-2026 window against 13.0% for Ledoit-Wolf, 14.8% for the Fama-French model and 33.6% for the sample covariance, which runs 31x gross), but on the 14 ETFs the book actually holds the plain sample covariance wins and PCA is second-worst (4.5% vs 6.1%), so the estimator the frozen rule picked is the wrong one for the book it was built for. The live book runs at 10.6% vol with 0.43 market beta, 71% of its variance on Mkt-RF, a 1-day 99% VaR of 1.5% parametric, 1.6% historical and 1.9% filtered historical, and today's weights held through Lehman-to-March-2009 lose 36.8%.

## Data and plumbing

- Prices: `lake.load("equities_daily", ..., universe="sp500")` for the point-in-time panel, 2003-06-02 to 2026-08-31 (the membership file's last date), 5,850 sessions, 636 names ever present, close hash `d1329973a978efd9`. The lake held SPY but none of the other 13 book ETFs, so `risk.ensure_etfs()` writes them through `lake.write` from yfinance with the collector's own schema (unadjusted OHLCV plus Yahoo's adjusted close), 70,352 rows on the first run, and refreshes any part that falls behind SPY's. Returns are on the adjusted close.
- Returns go through `research.features.FeatureStore` as one lag-0 feature with the peek audit on, so a return dated t uses the close of t and nothing later.
- Factors: Ken French's daily five factors plus momentum and RF, the same library `fama-french/data.py` reads monthly, cached a week under `source-material/risk-model/`. They end 2026-07-31; every Fama-French number below is on betas fit through that date.
- GARCH: `garch/garch.py::fit_garch`, student-t, on 1,000-day windows.
- Positions: the last shadow row of `framework/book/reports/ledger.csv` through `daemon.read_ledger`, the `beta+alpha` book on 2026-09-04. The lake's SPY part ended 2026-09-03 at run time, so the risk date is 09-03.
- Selection, lock, registry: `research.alpha.Untouched` on the S&P calendar (last fifth, 2021-12-31 to 2026-08-31, 1,170 sessions, the same window `research/` and `framework/book/validate.py` use), `research.registry.Registry` with twelve trials logged (four estimators on three universes).

`check.py` plants a three-factor structure on 60 names and checks the count, the betas, the estimator errors against the truth, the exposure arithmetic, VaR coverage on simulated GARCH-t returns and the stress arithmetic (32 checks, exits 1 on failure, five seconds). `run.py` is the study, 17 seconds after the first run's downloads.

## The estimators

All four take a 504-session window of daily returns and return an N x N covariance.

| name | what |
|---|---|
| `sample` | `np.cov`, unbiased |
| `lw` | sklearn `LedoitWolf`, shrinkage toward the scaled identity with the analytic intensity |
| `pca` | eigendecomposition of the correlation matrix, the top k eigenvectors kept and a diagonal residual, rescaled by the vols; k is the number of eigenvalues above the Marchenko-Pastur edge (1 + sqrt(N/T))^2 plus the Tracy-Widom 99% cushion for the largest eigenvalue (Johnstone 2001), at unit noise variance |
| `ff` | OLS of each name's excess return on Mkt-RF, SMB, HML, RMW, CMA, Mom over the window, B Cov(F) B' plus the diagonal residual variance |

The factor count is conservative on purpose. Re-estimating the noise variance from what the signal leaves (Laloux et al.) counted 44 factors on the planted panel because heterogeneous loadings widen the bulk; at unit variance it counts three. On pure noise it counts zero at N=100/T=1000 and at N=300/T=400 with t(5) tails. On the real S&P panel the count runs from 5 (2011-2013) to 14 (2025), 9 on average; on the last window (N=464) it is 13 with the top eigenvalue at 108 and the edge at 3.92. On the 14 ETFs it is 2 or 3.

## Minimum-variance race

At each month end from 2005-06-30 (254 rebalances) each estimator is fit on the trailing 504 sessions of every name with a complete window, the unconstrained global minimum-variance weights are held to the next month end, no costs. Three universes: the 14 ETFs; the 100 most liquid point-in-time members by trailing 252-day median dollar volume; every point-in-time member with a complete window (299 in 2005, 480 by 2025). Frozen rule before any number: the desk estimator is the one with the lowest out-of-sample vol on the full panel before the untouched window, and the book's parametric VaR uses it.

Annualised realised vol of the minimum-variance portfolio, pre-window 2005-07 to 2021-12 (4,154 days) and on the untouched window (1,170 days), with the median gross leverage of the weights:

| universe | estimator | vol pre | vol window | leverage median | leverage max |
|---|---|---|---|---|---|
| etf (N=14) | sample | **3.65%** | **4.55%** | 2.1 | 2.9 |
| | lw | 4.08% | 4.81% | 1.7 | 2.5 |
| | pca | 5.17% | 6.08% | 1.5 | 3.2 |
| | ff | 5.08% | 6.67% | 1.1 | 1.3 |
| sp100 | sample | 14.27% | 13.29% | 3.7 | 6.6 |
| | lw | **13.85%** | **12.75%** | 3.2 | 4.5 |
| | pca | 14.78% | 13.60% | 2.9 | 4.4 |
| | ff | 14.32% | 14.68% | 2.6 | 3.9 |
| sp500 (N to 480) | sample | 18.27% | 33.63% | 14.5 | 44.0 |
| | lw | 12.78% | 13.02% | 7.2 | 11.6 |
| | pca | **11.99%** | **11.44%** | 4.4 | 5.7 |
| | ff | 14.19% | 14.77% | 3.4 | 4.5 |

`pca` wins the full panel before the window and holds on it. Every structured estimator beats the sample there, and the sample gets worse as coverage improves: its median leverage goes from 7x in 2005 to 31x in 2022-2026 as N approaches T, which is the N/T = 0.9 regime where the sample inverse is mostly noise. On the 100 most liquid names (N/T = 0.2) the four are within a point of each other and Ledoit-Wolf edges it. On the 14 ETFs the ranking inverts: sample, then Ledoit-Wolf, then the two factor models, before and on the window. With N/T = 0.03 there is no estimation problem to fix, and both factor models throw away the pair structure the minimum-variance weights live on (IEF against TLT, GLD against SLV, FXE against FXB) by forcing a diagonal residual; the Fama-French model additionally explains the commodity, currency and Treasury ETFs badly (R² below 0.13 for all ten, 0.03 for IEF and FXE) so its covariance is nearly diagonal for them. The frozen rule was written for the S&P panel and picked `pca`; for the book itself it should have picked `sample`, and I report both below.

`reports/minvar_rolling_vol.png` has the 126-day rolling vol of each portfolio on each universe with the window shaded; `reports/mp_spectrum.png` the last window's eigenvalue histogram against the edge and the factor count over time.

The registry's DSR of the winning stream on the window is 0.93 over 12 trials (Sharpe 0.91 on the window; the minimum-variance portfolio of 480 names returned 10%/yr pre-window at 12% vol). That is logged because the harness logs it; vol is the metric here and the DSR is not the evidence.

## The live book

`beta+alpha` shadow book, ledger 2026-09-04, weights as fractions of equity: gross 1.19, net +0.96. Long everything but the two Treasury ETFs.

| | SPY | QQQ | EFA | EEM | IEF | TLT | USO | GLD | SLV | DBA | DBC | FXE | FXY | FXB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weight | 0.144 | 0.071 | 0.134 | 0.064 | -0.057 | -0.059 | 0.042 | 0.041 | 0.035 | 0.101 | 0.079 | 0.123 | 0.020 | 0.217 |
| vol | 16.6% | 21.9% | 16.4% | 21.9% | 5.5% | 11.6% | 41.3% | 24.0% | 47.9% | 12.6% | 18.5% | 7.2% | 9.9% | 7.0% |
| share of variance | 16.5% | 10.9% | 16.6% | 10.6% | 0.0% | 0.0% | 4.0% | 6.0% | 11.7% | 6.1% | 6.6% | 3.2% | 0.2% | 7.6% |

Book vol 10.55% on the `pca` covariance (10.16% Ledoit-Wolf, 10.2% sample, 8.95% Fama-French). By class: equities hold 41% of equity and 55% of variance, commodities 30% and 34%, currencies 36% and 11%, the bond shorts -12% and nothing (their marginal contribution is zero within rounding: short duration against long equities and commodities nets out). SLV is 3.5% of the book and 12% of its risk.

Fama-French exposures of the book, betas through 2026-07-31, factor covariance over the same window:

| | Mkt-RF | SMB | HML | RMW | CMA | Mom | residual |
|---|---|---|---|---|---|---|---|
| exposure | 0.43 | -0.09 | 0.01 | -0.13 | 0.05 | 0.02 | |
| share of variance | 71.2% | -0.9% | -0.5% | 8.6% | -0.5% | 1.3% | 20.9% |

Statistical factors on the 14 ETFs, k=3: PC1 carries 66.5% of the book's variance, PC2 24.9%, PC3 1.9%, residual 6.7%. The two decompositions agree that this is a 0.4-beta book with one dominant factor; the Fama-French model leaves three times the residual because it has no factor for commodities or currencies, which is where 45% of the book's variance sits.

`reports/book_risk.png` has both panels; `reports/book_assets.csv` and `book_betas.csv` the tables.

### VaR

1-day 99%, as a fraction of equity, 504-day window, GARCH on 1,000 days:

| method | VaR | ES |
|---|---|---|
| parametric normal, `pca` covariance | 1.55% | 1.77% |
| parametric normal, other covariances | 1.50% sample, 1.49% lw, 1.31% ff | |
| historical | 1.65% | 2.43% |
| filtered historical (per-ETF GARCH-t, joint residual dates) | 1.87% | 2.40% |

Backtest of today's weights held fixed, 2011-02-02 to 2026-08-31, 3,917 days, each VaR from data through the prior close (`reports/var_backtest.csv`, `.png`). The backtested FHS fits one GARCH-t to the fixed-weight portfolio series, refit every 21 days, not the per-ETF GARCH of the headline number, so it tests the filtering idea rather than that exact estimator:

| method | hit rate pre-window (2,747 days) | Kupiec p | hit rate window (1,170 days) | Kupiec p |
|---|---|---|---|---|
| parametric normal | 2.11% | 0.000 | 1.62% (19) | 0.049 |
| historical | 1.38% | 0.056 | 1.11% (13) | 0.71 |
| filtered historical | 1.20% | 0.30 | 1.03% (12) | 0.93 |

The normal parametric VaR is rejected on the pre-window sample and marginal on the window; it is the number a 2.33-sigma rule gives and the book's daily returns have fatter tails than that. FHS is the only method whose coverage is inside sampling error both before and on the window; historical is close behind and wider (it holds the 2020 quantile for two years). The FHS line in the chart is also the one that moves: it went from 1.1% to 11.2% across March 2020 and from 1.5% to 4.0% in April 2025, while the historical and parametric lines moved by well under a point in both.

### Stress

Today's weights held fixed through each window, ETF total returns, rebalanced back to today's weights each day (a buy-and-hold that lets them drift gives -37.1%, -23.4% and -5.8%), no vol targeting:

| window | return | worst day | max drawdown | biggest contributors |
|---|---|---|---|---|
| 2008-09-01 to 2009-03-09 | **-36.8%** | -6.3% (2008-09-29) | -35.5% | SPY -7.8, EFA -7.6, FXB -5.6, DBC -4.9, USO -4.3 |
| 2020-02-19 to 2020-03-23 | **-23.6%** | -7.2% (2020-03-16) | -23.8% | SPY -5.4, EFA -4.9, USO -3.0, FXB -2.7, DBC -2.2 |
| 2022-01-03 to 2022-12-30 | **-5.5%** | -2.8% (2022-09-23) | -17.5% | SPY -2.5, QQQ -2.4, FXB -2.2; TLT short +2.1, DBC +1.7, USO +1.4 |

These are what today's positions are exposed to, not what the book would have done: the engine's 10% vol target and drawdown cut would have shrunk it through both crashes (the book's worst drawdown over 23 years in `framework/README.md` is -12.1%). The 2022 replay is the interesting one: the bond shorts and the commodity longs paid, and the book ended the year down 5% with a 17% drawdown in the middle, because sterling and the equity ETFs went together.

## API

```python
import sys; sys.path.insert(0, "../risk-model")
import risk
risk.exposures(positions, "2026-09-03")           # dict of tables: assets, by_class, ff, stat, betas
risk.var(positions, "2026-09-03")                 # {"parametric": {...}, "historical": {...}, "fhs": {...}}
risk.var(positions, date, method="fhs", estimator="sample")
risk.stress(positions)                            # the three windows
```

`positions` is `{ticker: weight}` as the ledger stores it; `date` is the as-of close; the panel is read from the lake on first use (1.1 seconds). A name without a complete 504-day history through the date raises. `estimator` is any of the four; the default is `lw` because on the book's own universe it is the best of the three that stay well-conditioned when a name is added, and the sample is one call away.

## Limitations

- **The rule picked the wrong estimator for the book.** It was frozen on the S&P panel because that is where an estimator can fail, and `pca` won there; on the 14 ETFs it is second-worst. The book's headline VaR is reported on `pca` because the rule said so, and on the sample (1.50%) because that is what the ETF race says to use. The gap is 5 bps of equity a day; it matters for the S&P book that does not exist yet, not for this one.
- **Fixed weights, no overlay.** Every backtest and stress number holds today's weights constant. The live book is vol-targeted and buffered, so its realised risk through a crash would be lower; the VaR of the book as it would actually trade is a different backtest that would need the engine.
- **Factor count is biased low.** Unit noise variance under-counts weak factors once strong ones absorb variance. The right fix is a fitted noise variance that survives heterogeneous loadings (de Prado's KDE fit or a two-stage residual PCA); the Laloux iteration I tried over-counts and was dropped.
- **Fama-French betas are a month stale** (factors end 2026-07-31), and the model is the wrong shape for a book that is 45% commodities and currencies by variance; a fundamental model for this book would need commodity and currency factors, which French does not publish.
- **The ETF parts in the lake are this project's.** The lake's collector only pulls its own universe, so the 13 ETFs will go stale until `ensure_etfs` runs again (it re-pulls any part behind SPY's date). Yahoo's adjusted close is a snapshot, the lake's known limitation; a dividend between runs leaves a small return error on the older rows.
- **Twelve trials, one window read.** The untouched window was opened once for the race vols and the VaR coverage together; both were read from that one call. The registry logs the twelve minimum-variance streams; the four parametric VaRs on the book are not trials, they are the same estimators reported four ways.
- **No costs in the race**, so a 31x-leverage sample portfolio's 34% realised vol understates how bad it would be to run; it also flatters `pca` and `lw` a little against `ff`, whose weights are the least levered.

## Files

- `reports/summary.json`, every number above at full precision
- `reports/untouched.json`, the lock, spent 2026-09-05
- `reports/registry/runs.jsonl`, the twelve streams
- `reports/minvar_returns.csv` (gitignored, 1.4 MB), `minvar_leverage.csv`, `var_backtest.csv`, `book_assets.csv`, `book_betas.csv`
- `reports/minvar_rolling_vol.png`, `mp_spectrum.png`, `book_risk.png`, `var_backtest.png`
