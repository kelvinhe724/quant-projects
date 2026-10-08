# Validation

Run 2026-10-08 03:15, panel hash `aa436bbc6c2e4712`, 2003-06-03 to 2026-08-31 (23.2 years), shadow cost model, kill switch off for backtests (see below).

## Headline (book, live allocator, net)

| Sharpe | ann. return | ann. vol | max drawdown | PSR | DSR | trials | MinBTL | MinTRL |
|---|---|---|---|---|---|---|---|---|
| 0.762 | 5.38% | 7.22% | -12.1% | 1.000 | 0.549 | 63 | 9.6 y (have 23.2) | 4.8 y |

Live book `beta+alpha`. DSR uses n_trials = 63: every configuration this file runs on the real panel (sleeves, the trend grid, allocators, cost and delay variants, kill on, the EWMAC variants, the candidate books, the candidate allocator) is appended to `trials.csv` (308 rows so far) and identical return streams count once; the variance of their daily Sharpes is 3.85e-04. Trials are tagged with the engine settings ({'buffer': 0.1}), so the runs before the buffer moved into the engine still count: they were looked at. The per-asset vol target is undone by the sleeve-level 10% target, so the trend grid is really four lookbacks. The 20 placebo draws are a null, not candidates; counting them too gives DSR 0.490. PSR is the probability the true Sharpe is above zero; DSR the same after deflating for selection. MinTRL is the track length needed to reject zero at 95%.

## Allocator: ERC vs 1/N, out of sample

Quarterly refits, three-year window, Ledoit-Wolf covariance, weights applied to the next quarter. Live allocator: **equal** with weights {'ETFBeta': 0.5, 'EWMAC': 0.5} (ERC would have been {'ETFBeta': 0.4284, 'EWMAC': 0.5716}).

|               | erc        | equal      |
|:--------------|:-----------|:-----------|
| sharpe        | 0.7933     | 0.8106     |
| annual_return | 0.0571     | 0.0594     |
| annual_vol    | 0.0734     | 0.0746     |
| max_drawdown  | -0.1103    | -0.1171    |
| start         | 2004-07-01 | 2004-07-01 |
| end           | 2026-08-31 | 2026-08-31 |

Sharpe by year:

|      |   erc |   equal |
|-----:|------:|--------:|
| 2004 |  3.04 |    3.04 |
| 2005 |  1.31 |    1.31 |
| 2006 |  1.28 |    1.28 |
| 2007 |  1.89 |    1.88 |
| 2008 |  0.16 |    0.2  |
| 2009 |  0.59 |    0.55 |
| 2010 |  1.24 |    1.22 |
| 2011 |  0.73 |    0.79 |
| 2012 | -0.16 |   -0.21 |
| 2013 |  0.31 |    0.38 |
| 2014 | -0.17 |   -0.17 |
| 2015 | -1.31 |   -1.28 |
| 2016 |  0.35 |    0.26 |
| 2017 |  2.08 |    1.97 |
| 2018 | -0.52 |   -0.43 |
| 2019 |  2.36 |    2.22 |
| 2020 |  0.87 |    1.15 |
| 2021 |  0.49 |    0.46 |
| 2022 | -0.12 |    0.08 |
| 2023 | -0.51 |   -0.62 |
| 2024 |  0.88 |    0.87 |
| 2025 |  1.73 |    1.77 |
| 2026 |  0.96 |    1.06 |

## Walk-forward over the trend grid

Grid: lookback (3, 6, 9, 12) months x per-asset vol target (0.2, 0.4). purgedcv WalkForwardSplit, 5 folds of 3 years, the variant with the best training Sharpe is held in the test window. Selected-in-sample OOS Sharpe **0.249** vs frozen v1 **0.249** over the same test windows.

|   fold | test_start   | test_end   | picked                          |   picked_train_sharpe |   picked_test_sharpe |   v1_test_sharpe |
|-------:|:-------------|:-----------|:--------------------------------|----------------------:|---------------------:|-----------------:|
|      1 | 2011-08-18   | 2014-08-20 | TrendETF lookback=12 target=0.2 |                 0.803 |                0.015 |            0.015 |
|      2 | 2014-08-21   | 2017-08-21 | TrendETF lookback=12 target=0.2 |                 0.604 |                0.059 |            0.059 |
|      3 | 2017-08-22   | 2020-08-21 | TrendETF lookback=12 target=0.2 |                 0.499 |                0.171 |            0.171 |
|      4 | 2020-08-24   | 2023-08-24 | TrendETF lookback=12 target=0.2 |                 0.453 |               -0.269 |           -0.269 |
|      5 | 2023-08-25   | 2026-08-31 | TrendETF lookback=12 target=0.2 |                 0.383 |                1.071 |            1.071 |

## Falsification

**Placebo.** Trend sleeve with a fixed random sign per ETF, 20 draws: actual Sharpe 0.471, placebo mean -0.037, p95 0.299, beats 20 of 20.

**Vol scaling without the signal.** Same sizing, every sign forced long:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| trend_signal | 0.471 | 0.037 | 0.085 | -0.200 |
| long_only_vol_scaled | 0.618 | 0.047 | 0.079 | -0.229 |

**Cost stress.** Book at multiples of the cost model:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| 1x | 0.762 | 0.054 | 0.072 | -0.121 |
| 2x | 0.688 | 0.047 | 0.071 | -0.133 |
| 4x | 0.558 | 0.035 | 0.066 | -0.200 |

**Signal delay.** Every sleeve's targets held back N extra bars:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| +1 | 0.752 | 0.053 | 0.072 | -0.121 |
| +2 | 0.757 | 0.054 | 0.073 | -0.123 |
| +5 | 0.741 | 0.053 | 0.073 | -0.132 |

## v2 candidate: EWMAC

EWMAC at 16/64, 32/128, 64/256 and the three combined with a forecast diversification multiplier, forecast scalars estimated on trailing data only (expanding, NaN for the first 500 days), cap 20, forecast / 10 x TrendETF's 40% per-asset target (35-day vol where TrendETF uses 60-day) and class balance, sent daily, same universe, cost model and overlay. The engine's 10% buffer sits on the final weights after the overlay, per instrument: an instrument trades only when its held weight is outside target x (1 +/- 0.1). Window 2005-06-09 to 2026-08-31, where both rules are live. Gross is the same path with each day's trade costs and borrow added back. Turnover is traded notional over equity per year. Each variant is a logged trial.

| variant | sharpe_gross | sharpe | annual_return | annual_vol | max_drawdown | turnover | trades |
|---|---|---|---|---|---|---|---|
| TrendETF | 0.514 | 0.447 | 0.034 | 0.084 | -0.200 | 5.757 | 2165 |
| EWMAC | 0.784 | 0.654 | 0.059 | 0.095 | -0.162 | 15.466 | 22252 |
| EWMAC 16/64 | 0.590 | 0.400 | 0.032 | 0.089 | -0.168 | 21.826 | 28589 |
| EWMAC 32/128 | 0.801 | 0.679 | 0.065 | 0.100 | -0.169 | 15.075 | 21293 |
| EWMAC 64/256 | 0.700 | 0.608 | 0.055 | 0.096 | -0.186 | 10.715 | 16657 |

## EWMAC: walk-forward and the untouched window

Walk-forward over the four EWMAC variants on the sessions before the untouched window, 4 folds of 3 years, best training Sharpe held in the test window. Selected-in-sample OOS Sharpe **0.603** vs the frozen combined rule **0.611** over the same test windows.

|   fold | test_start   | test_end   | picked       |   picked_train_sharpe |   picked_test_sharpe |   v1_test_sharpe |
|-------:|:-------------|:-----------|:-------------|----------------------:|---------------------:|-----------------:|
|      1 | 2009-12-28   | 2012-12-27 | EWMAC        |                 1.025 |                0.499 |            0.499 |
|      2 | 2012-12-28   | 2015-12-29 | EWMAC        |                 0.818 |                0.578 |            0.578 |
|      3 | 2015-12-30   | 2018-12-31 | EWMAC        |                 0.757 |                0.464 |            0.464 |
|      4 | 2019-01-02   | 2021-12-30 | EWMAC 32/128 |                 0.7   |                0.904 |            0.934 |

**Untouched window** 2021-12-31 to 2026-08-31, the last 20% of the panel's sessions. Nothing was fit or picked on it; its return was inside the one full-panel look above, so it confirms rather than discovers. Promotion rule, fixed before this run: net Sharpe above zero on the untouched window, above TrendETF's on the same window, and a positive alpha t against the ETF universe there.

| rule | Sharpe gross | Sharpe net | ann. return | ann. vol | max drawdown | turnover | trades |
|---|---|---|---|---|---|---|---|
| EWMAC | 0.446 | 0.321 | 2.29% | 8.06% | -15.6% | 12.5x | 5223 |
| TrendETF | 0.910 | 0.834 | 5.58% | 6.79% | -7.1% | 5.5x | 566 |

EWMAC against its universe on the window: 1/N: alpha -0.10% (t -0.03), beta 0.27, benchmark Sharpe 0.98, residual Sharpe -0.01; 1/N vol-targeted: alpha -0.63% (t -0.19), beta 0.31, benchmark Sharpe 1.01, residual Sharpe -0.09. **Clears the rule: no.**

## Candidate books

Every book in `strategies.BOOKS` at 1/N of its sleeves, kill off, from 2005-06-09 (the first day all four are live) to the end; the same window, engine and cost model for all four. DSR uses the trial count above. OOS is the untouched window; the live book is the OOS winner only if it beats `v1` there. **OOS winner: `beta+alpha`** (live book `beta+alpha`).

| variant | sharpe_gross | sharpe | dsr | annual_return | annual_vol | max_drawdown | turnover | oos_sharpe | oos_max_drawdown |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.374 | 0.294 | 0.022 | 0.012 | 0.044 | -0.137 | 3.269 | 0.591 | -0.073 |
| v1+ewmac | 0.651 | 0.529 | 0.174 | 0.028 | 0.056 | -0.115 | 7.769 | 0.471 | -0.089 |
| ewmac-for-trend | 0.640 | 0.501 | 0.142 | 0.026 | 0.053 | -0.116 | 8.747 | 0.239 | -0.116 |
| beta+alpha | 0.801 | 0.708 | 0.451 | 0.051 | 0.074 | -0.121 | 8.602 | 0.686 | -0.108 |

ERC vs 1/N gate on `beta+alpha`'s sleeves, quarterly refits, out of sample:

|               | erc        | equal      |
|:--------------|:-----------|:-----------|
| sharpe        | 0.7919     | 0.8056     |
| annual_return | 0.0572     | 0.0591     |
| annual_vol    | 0.0737     | 0.0748     |
| max_drawdown  | -0.1102    | -0.1172    |
| start         | 2004-07-01 | 2004-07-01 |
| end           | 2026-08-31 | 2026-08-31 |

## Candidate allocator: cost-aware mean-variance

`optimizer.py` on each book's sleeve streams: maximise mu'w - w'Sw / 2 - cost'|dw| over long-only sleeve weights, parameters {'risk_aversion': 1.0, 'target_vol': 0.1, 'max_gross': 1.5, 'max_net': 1.5, 'max_sleeve': 0.75, 'turnover': 0.5, 'kelly': 0.5, 'financing_spread_bps': 50.0} fixed before the run. mu is the trailing three-year mean, S Ledoit-Wolf (no `risk-model/` at run time), cost each sleeve's realised cost per traded notional times its gross. Quarterly refits, weights held for the next quarter, the move charged on its first day, leverage above 1 financed at data-lake fred/DTB3 plus the spread. Both columns are daily-rebalanced to their weights; the 1/N column here is that convention, the books table above is the engine's static 1/N, and the rule compares against the engine's. Every row is a logged trial and a research-registry run (4 there). Rule, written before the run: the candidate allocator goes live only if its best book's Sharpe on the untouched window beats the best 1/N book's there, net of reallocation costs and financing; otherwise 1/N stays live and the candidate is a documented trial.

| book | walk Sharpe mvo | walk Sharpe 1/N | DSR | DSR (registry) | untouched Sharpe mvo | untouched Sharpe 1/N (walk) | untouched Sharpe 1/N (engine) | mean leverage | turnover / refit | refits | budget bound | vol bound | Kelly scaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.225 | 0.238 | 0.024 | 0.242 | 0.319 | 0.526 | 0.591 | 0.67 | 0.17 | 86 | 21 | 4 | 15 |
| v1+ewmac | 0.458 | 0.483 | 0.109 | 0.574 | 0.213 | 0.498 | 0.471 | 1.05 | 0.25 | 86 | 19 | 44 | 5 |
| ewmac-for-trend | 0.389 | 0.425 | 0.068 | 0.455 | -0.303 | 0.232 | 0.239 | 0.81 | 0.15 | 82 | 16 | 4 | 5 |
| beta+alpha | 0.680 | 0.805 | 0.398 | 0.885 | 0.547 | 0.728 | 0.686 | 1.09 | 0.12 | 90 | 7 | 38 | 2 |

Best candidate book on the untouched window: `beta+alpha` at 0.547 against the best 1/N book `beta+alpha` at 0.686. **Promoted: no.** `allocate.LIVE_ALLOCATOR = "rule"`. The live book's candidate weight path is in `mvo_weights.csv`.

## Signal vs beta

Each sleeve's live net returns regressed on its own universe: the daily-rebalanced 1/N return of its instruments, and that return scaled to the sleeves' 10% vol target (60-day EWMA vol, one-day lag, 3x cap). Alpha is annualised with a Newey-West t; residual Sharpe is the Sharpe of the sleeve minus beta x benchmark. A sleeve whose residual Sharpe is below the benchmark's own Sharpe is not adding return beyond owning the universe.

| sleeve | benchmark | benchmark Sharpe | sleeve Sharpe | alpha | t | beta | residual Sharpe |
|---|---|---|---|---|---|---|---|
| TrendETF | 1/N | 0.626 | 0.471 | +2.17% | 1.30 | 0.268 | 0.273 |
| TrendETF | 1/N vol-targeted | 0.611 | 0.471 | +1.84% | 1.14 | 0.346 | 0.238 |
| FXCarryETF | 1/N | 0.076 | -0.289 | -1.47% | -1.41 | 0.147 | -0.313 |
| FXCarryETF | 1/N vol-targeted | 0.017 | -0.289 | -1.41% | -1.34 | 0.097 | -0.298 |
| CryptoTrend | 1/N | 0.440 | 0.073 | -2.13% | -0.96 | 0.106 | -0.438 |
| CryptoTrend | 1/N vol-targeted | 0.517 | 0.073 | -2.71% | -1.24 | 0.631 | -0.585 |
| ETFBeta | 1/N | 0.670 | 0.669 | +0.73% | 0.82 | 0.674 | 0.175 |
| ETFBeta | 1/N vol-targeted | 0.651 | 0.662 | +0.41% | 0.66 | 0.774 | 0.146 |
| EWMAC | 1/N | 0.598 | 0.654 | +4.99% | 2.58 | 0.188 | 0.538 |
| EWMAC | 1/N vol-targeted | 0.575 | 0.654 | +4.60% | 2.42 | 0.275 | 0.507 |

## Kill switch

Backtests above run with `kill_dd=None`. With the live 25% kill on, the sleeves would have been flattened permanently on: TrendETF never, FXCarryETF 2013-12-04, CryptoTrend never, ETFBeta 2016-01-07, EWMAC never. Live, a kill stops the sleeve until Kelvin restarts it, which no backtest can model.

## purgedcv vs archived formula (risk #8)

Best of 100 random series, 750 days: DSR purgedcv 0.6855 vs archive 0.6848; PSR 0.9970 vs 0.9970. purgedcv is pinned at 0.1.5 in requirements.txt.

Tearsheet: `reports/tearsheet.html` (quantstats, SPY benchmark). 3141s.
