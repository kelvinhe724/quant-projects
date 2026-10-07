# Validation

Run 2026-10-07 04:21, panel hash `aa436bbc6c2e4712`, 2003-06-03 to 2026-08-31 (23.2 years), shadow cost model, kill switch off for backtests (see below).

## Headline (book, live allocator, net)

| Sharpe | ann. return | ann. vol | max drawdown | PSR | DSR | trials | MinBTL | MinTRL |
|---|---|---|---|---|---|---|---|---|
| 0.836 | 5.94% | 7.21% | -12.1% | 1.000 | 0.587 | 63 | 8.0 y (have 23.2) | 4.0 y |

Live book `beta+alpha`. DSR uses n_trials = 63: every configuration this file runs on the real panel (sleeves, the trend grid, allocators, cost and delay variants, kill on, the EWMAC variants, the candidate books, the candidate allocator) is appended to `trials.csv` (268 rows so far) and identical return streams count once; the variance of their daily Sharpes is 4.44e-04. Trials are tagged with the engine settings ({'buffer': 0.1}), so the runs before the buffer moved into the engine still count: they were looked at. The per-asset vol target is undone by the sleeve-level 10% target, so the trend grid is really four lookbacks. The 20 placebo draws are a null, not candidates; counting them too gives DSR 0.524. PSR is the probability the true Sharpe is above zero; DSR the same after deflating for selection. MinTRL is the track length needed to reject zero at 95%.

## Allocator: ERC vs 1/N, out of sample

Quarterly refits, three-year window, Ledoit-Wolf covariance, weights applied to the next quarter. Live allocator: **equal** with weights {'ETFBeta': 0.5, 'EWMAC': 0.5} (ERC would have been {'ETFBeta': 0.3514, 'EWMAC': 0.6486}).

|               | erc        | equal      |
|:--------------|:-----------|:-----------|
| sharpe        | 0.8787     | 0.8985     |
| annual_return | 0.0616     | 0.0653     |
| annual_vol    | 0.0709     | 0.0734     |
| max_drawdown  | -0.1075    | -0.111     |
| start         | 2004-07-01 | 2004-07-01 |
| end           | 2026-08-31 | 2026-08-31 |

Sharpe by year:

|      |   erc |   equal |
|-----:|------:|--------:|
| 2004 |  3.04 |    3.04 |
| 2005 |  1.31 |    1.31 |
| 2006 |  1.35 |    1.35 |
| 2007 |  1.86 |    1.86 |
| 2008 |  0.39 |    0.43 |
| 2009 |  0.15 |    0.08 |
| 2010 |  1.56 |    1.55 |
| 2011 |  0.72 |    0.79 |
| 2012 | -0.18 |   -0.26 |
| 2013 |  0.55 |    0.67 |
| 2014 |  0.8  |    0.93 |
| 2015 | -0.78 |   -0.53 |
| 2016 |  0.55 |    0.45 |
| 2017 |  1.92 |    1.64 |
| 2018 | -0.45 |   -0.29 |
| 2019 |  2.23 |    1.92 |
| 2020 |  0.76 |    1.08 |
| 2021 |  0.58 |    0.55 |
| 2022 |  0.62 |    0.98 |
| 2023 | -0.65 |   -0.89 |
| 2024 |  0.89 |    0.88 |
| 2025 |  1.75 |    1.77 |
| 2026 |  0.92 |    1.14 |

## Walk-forward over the trend grid

Grid: lookback (3, 6, 9, 12) months x per-asset vol target (0.2, 0.4). purgedcv WalkForwardSplit, 5 folds of 3 years, the variant with the best training Sharpe is held in the test window. Selected-in-sample OOS Sharpe **0.359** vs frozen v1 **0.090** over the same test windows.

|   fold | test_start   | test_end   | picked                         |   picked_train_sharpe |   picked_test_sharpe |   v1_test_sharpe |
|-------:|:-------------|:-----------|:-------------------------------|----------------------:|---------------------:|-----------------:|
|      1 | 2011-08-18   | 2014-08-20 | TrendETF lookback=9 target=0.2 |                 0.703 |               -0.249 |           -0.289 |
|      2 | 2014-08-21   | 2017-08-21 | TrendETF lookback=9 target=0.2 |                 0.523 |                0.125 |            0.085 |
|      3 | 2017-08-22   | 2020-08-21 | TrendETF lookback=9 target=0.2 |                 0.425 |                0.205 |            0.073 |
|      4 | 2020-08-24   | 2023-08-24 | TrendETF lookback=9 target=0.2 |                 0.389 |                0.511 |            0.108 |
|      5 | 2023-08-25   | 2026-08-31 | TrendETF lookback=9 target=0.2 |                 0.406 |                0.976 |            0.402 |

## Falsification

**Placebo.** Trend sleeve with a fixed random sign per ETF, 20 draws: actual Sharpe 0.284, placebo mean -0.004, p95 0.418, beats 16 of 20.

**Vol scaling without the signal.** Same sizing, every sign forced long:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| trend_signal | 0.284 | 0.020 | 0.079 | -0.217 |
| long_only_vol_scaled | 0.618 | 0.047 | 0.079 | -0.229 |

**Cost stress.** Book at multiples of the cost model:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| 1x | 0.836 | 0.059 | 0.072 | -0.121 |
| 2x | 0.721 | 0.050 | 0.071 | -0.126 |
| 4x | 0.548 | 0.034 | 0.065 | -0.194 |

**Signal delay.** Every sleeve's targets held back N extra bars:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| +1 | 0.818 | 0.058 | 0.072 | -0.121 |
| +2 | 0.807 | 0.058 | 0.073 | -0.124 |
| +5 | 0.753 | 0.053 | 0.073 | -0.135 |

## v2 candidate: EWMAC

EWMAC at 16/64, 32/128, 64/256 and the three combined with a forecast diversification multiplier, forecast scalars estimated on trailing data only (expanding, NaN for the first 500 days), cap 20, forecast / 10 x TrendETF's 40% per-asset target (35-day vol where TrendETF uses 60-day) and class balance, sent daily, same universe, cost model and overlay. The engine's 10% buffer sits on the final weights after the overlay, per instrument: an instrument trades only when its held weight is outside target x (1 +/- 0.1). Window 2005-06-09 to 2026-08-31, where both rules are live. Gross is the same path with each day's trade costs and borrow added back. Turnover is traded notional over equity per year. Each variant is a logged trial.

| variant | sharpe_gross | sharpe | annual_return | annual_vol | max_drawdown | turnover | trades |
|---|---|---|---|---|---|---|---|
| TrendETF | 0.317 | 0.241 | 0.016 | 0.078 | -0.217 | 5.436 | 2326 |
| EWMAC | 0.871 | 0.717 | 0.070 | 0.102 | -0.193 | 18.256 | 25942 |
| EWMAC 16/64 | 0.755 | 0.537 | 0.049 | 0.097 | -0.174 | 25.866 | 33012 |
| EWMAC 32/128 | 0.855 | 0.707 | 0.071 | 0.105 | -0.170 | 17.873 | 24787 |
| EWMAC 64/256 | 0.767 | 0.652 | 0.064 | 0.103 | -0.196 | 13.141 | 19581 |

## EWMAC: walk-forward and the untouched window

Walk-forward over the four EWMAC variants on the sessions before the untouched window, 4 folds of 3 years, best training Sharpe held in the test window. Selected-in-sample OOS Sharpe **0.731** vs the frozen combined rule **0.720** over the same test windows.

|   fold | test_start   | test_end   | picked       |   picked_train_sharpe |   picked_test_sharpe |   v1_test_sharpe |
|-------:|:-------------|:-----------|:-------------|----------------------:|---------------------:|-----------------:|
|      1 | 2009-12-28   | 2012-12-27 | EWMAC        |                 0.95  |                0.576 |            0.576 |
|      2 | 2012-12-28   | 2015-12-29 | EWMAC        |                 0.801 |                1.079 |            1.079 |
|      3 | 2015-12-30   | 2018-12-31 | EWMAC        |                 0.88  |                0.421 |            0.421 |
|      4 | 2019-01-02   | 2021-12-30 | EWMAC 32/128 |                 0.78  |                0.877 |            0.828 |

**Untouched window** 2021-12-31 to 2026-08-31, the last 20% of the panel's sessions. Nothing was fit or picked on it; its return was inside the one full-panel look above, so it confirms rather than discovers. Promotion rule, fixed before this run: net Sharpe above zero on the untouched window, above TrendETF's on the same window, and a positive alpha t against the ETF universe there.

| rule | Sharpe gross | Sharpe net | ann. return | ann. vol | max drawdown | turnover | trades |
|---|---|---|---|---|---|---|---|
| EWMAC | 0.564 | 0.416 | 2.92% | 7.62% | -19.3% | 12.6x | 6063 |
| TrendETF | 0.593 | 0.502 | 3.24% | 6.81% | -14.2% | 5.9x | 612 |

EWMAC against its universe on the window: 1/N: alpha +2.80% (t 0.75), beta 0.04, benchmark Sharpe 0.98, residual Sharpe 0.37; 1/N vol-targeted: alpha +2.22% (t 0.60), beta 0.09, benchmark Sharpe 1.01, residual Sharpe 0.29. **Clears the rule: no.**

## Candidate books

Every book in `strategies.BOOKS` at 1/N of its sleeves, kill off, from 2005-06-09 (the first day all four are live) to the end; the same window, engine and cost model for all four. DSR uses the trial count above. OOS is the untouched window; the live book is the OOS winner only if it beats `v1` there. **OOS winner: `beta+alpha`** (live book `beta+alpha`).

| variant | sharpe_gross | sharpe | dsr | annual_return | annual_vol | max_drawdown | turnover | oos_sharpe | oos_max_drawdown |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.185 | 0.087 | 0.001 | 0.003 | 0.040 | -0.170 | 2.988 | 0.321 | -0.090 |
| v1+ewmac | 0.689 | 0.533 | 0.120 | 0.029 | 0.057 | -0.133 | 9.424 | 0.441 | -0.133 |
| ewmac-for-trend | 0.759 | 0.591 | 0.183 | 0.034 | 0.059 | -0.145 | 10.716 | 0.395 | -0.145 |
| beta+alpha | 0.910 | 0.789 | 0.497 | 0.057 | 0.073 | -0.121 | 10.404 | 0.798 | -0.119 |

ERC vs 1/N gate on `beta+alpha`'s sleeves, quarterly refits, out of sample:

|               | erc        | equal      |
|:--------------|:-----------|:-----------|
| sharpe        | 0.8788     | 0.8986     |
| annual_return | 0.0616     | 0.0653     |
| annual_vol    | 0.0709     | 0.0734     |
| max_drawdown  | -0.1083    | -0.111     |
| start         | 2004-07-01 | 2004-07-01 |
| end           | 2026-08-31 | 2026-08-31 |

## Candidate allocator: cost-aware mean-variance

`optimizer.py` on each book's sleeve streams: maximise mu'w - w'Sw / 2 - cost'|dw| over long-only sleeve weights, parameters {'risk_aversion': 1.0, 'target_vol': 0.1, 'max_gross': 1.5, 'max_net': 1.5, 'max_sleeve': 0.75, 'turnover': 0.5, 'kelly': 0.5, 'financing_spread_bps': 50.0} fixed before the run. mu is the trailing three-year mean, S Ledoit-Wolf (no `risk-model/` at run time), cost each sleeve's realised cost per traded notional times its gross. Quarterly refits, weights held for the next quarter, the move charged on its first day, leverage above 1 financed at data-lake fred/DTB3 plus the spread. Both columns are daily-rebalanced to their weights; the 1/N column here is that convention, the books table above is the engine's static 1/N, and the rule compares against the engine's. Every row is a logged trial and a research-registry run (4 there). Rule, written before the run: the candidate allocator goes live only if its best book's Sharpe on the untouched window beats the best 1/N book's there, net of reallocation costs and financing; otherwise 1/N stays live and the candidate is a documented trial.

| book | walk Sharpe mvo | walk Sharpe 1/N | DSR | DSR (registry) | untouched Sharpe mvo | untouched Sharpe 1/N (walk) | untouched Sharpe 1/N (engine) | mean leverage | turnover / refit | refits | budget bound | vol bound | Kelly scaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | -0.168 | 0.133 | 0.000 | 0.000 | -0.445 | 0.442 | 0.321 | 0.56 | 0.29 | 86 | 26 | 1 | 19 |
| v1+ewmac | 0.436 | 0.498 | 0.055 | 0.010 | -0.269 | 0.518 | 0.441 | 1.05 | 0.24 | 86 | 23 | 28 | 1 |
| ewmac-for-trend | 0.473 | 0.556 | 0.082 | 0.017 | -0.261 | 0.438 | 0.395 | 0.92 | 0.18 | 82 | 20 | 1 | 1 |
| beta+alpha | 0.763 | 0.898 | 0.449 | 0.190 | 0.686 | 0.891 | 0.798 | 1.13 | 0.10 | 90 | 7 | 37 | 0 |

Best candidate book on the untouched window: `beta+alpha` at 0.686 against the best 1/N book `beta+alpha` at 0.798. **Promoted: no.** `allocate.LIVE_ALLOCATOR = "rule"`. The live book's candidate weight path is in `mvo_weights.csv`.

## Signal vs beta

Each sleeve's live net returns regressed on its own universe: the daily-rebalanced 1/N return of its instruments, and that return scaled to the sleeves' 10% vol target (60-day EWMA vol, one-day lag, 3x cap). Alpha is annualised with a Newey-West t; residual Sharpe is the Sharpe of the sleeve minus beta x benchmark. A sleeve whose residual Sharpe is below the benchmark's own Sharpe is not adding return beyond owning the universe.

| sleeve | benchmark | benchmark Sharpe | sleeve Sharpe | alpha | t | beta | residual Sharpe |
|---|---|---|---|---|---|---|---|
| TrendETF | 1/N | 0.626 | 0.284 | +1.15% | 0.69 | 0.163 | 0.149 |
| TrendETF | 1/N vol-targeted | 0.611 | 0.284 | +0.80% | 0.50 | 0.234 | 0.106 |
| FXCarryETF | 1/N | 0.076 | -0.164 | -0.96% | -0.73 | -0.083 | -0.158 |
| FXCarryETF | 1/N vol-targeted | 0.017 | -0.164 | -1.00% | -0.75 | -0.081 | -0.164 |
| CryptoTrend | 1/N | 0.440 | 0.073 | -2.13% | -0.96 | 0.106 | -0.438 |
| CryptoTrend | 1/N vol-targeted | 0.517 | 0.073 | -2.71% | -1.24 | 0.631 | -0.585 |
| ETFBeta | 1/N | 0.670 | 0.669 | +0.73% | 0.82 | 0.674 | 0.175 |
| ETFBeta | 1/N vol-targeted | 0.651 | 0.662 | +0.41% | 0.66 | 0.774 | 0.146 |
| EWMAC | 1/N | 0.598 | 0.717 | +7.02% | 3.25 | 0.049 | 0.687 |
| EWMAC | 1/N vol-targeted | 0.575 | 0.717 | +6.59% | 3.06 | 0.126 | 0.650 |

## Kill switch

Backtests above run with `kill_dd=None`. With the live 25% kill on, the sleeves would have been flattened permanently on: TrendETF never, FXCarryETF 2010-08-31, CryptoTrend never, ETFBeta 2016-01-07, EWMAC never. Live, a kill stops the sleeve until Kelvin restarts it, which no backtest can model.

## purgedcv vs archived formula (risk #8)

Best of 100 random series, 750 days: DSR purgedcv 0.6855 vs archive 0.6848; PSR 0.9970 vs 0.9970. purgedcv is pinned at 0.1.5 in requirements.txt.

Tearsheet: `reports/tearsheet.html` (quantstats, SPY benchmark). 3441s.
