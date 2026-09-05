# Validation

Run 2026-09-05 02:59, panel hash `aa436bbc6c2e4712`, 2003-06-03 to 2026-08-31 (23.2 years), shadow cost model, kill switch off for backtests (see below).

## Headline (book, live allocator, net)

| Sharpe | ann. return | ann. vol | max drawdown | PSR | DSR | trials | MinBTL | MinTRL |
|---|---|---|---|---|---|---|---|---|
| 0.810 | 5.71% | 7.17% | -12.1% | 1.000 | 0.632 | 60 | 8.4 y (have 23.2) | 4.3 y |

Live book `beta+alpha`. DSR uses n_trials = 60: every configuration this file runs on the real panel (sleeves, the trend grid, allocators, cost and delay variants, kill on, the EWMAC variants, the candidate books, the candidate allocator) is appended to `trials.csv` (188 rows so far) and identical return streams count once; the variance of their daily Sharpes is 3.94e-04. Trials are tagged with the engine settings ({'buffer': 0.1}), so the runs before the buffer moved into the engine still count: they were looked at. The per-asset vol target is undone by the sleeve-level 10% target, so the trend grid is really four lookbacks. The 20 placebo draws are a null, not candidates; counting them too gives DSR 0.571. PSR is the probability the true Sharpe is above zero; DSR the same after deflating for selection. MinTRL is the track length needed to reject zero at 95%.

## Allocator: ERC vs 1/N, out of sample

Quarterly refits, three-year window, Ledoit-Wolf covariance, weights applied to the next quarter. Live allocator: **equal** with weights {'ETFBeta': 0.5, 'EWMAC': 0.5} (ERC would have been {'ETFBeta': 0.3516, 'EWMAC': 0.6484}).

|               | erc        | equal      |
|:--------------|:-----------|:-----------|
| sharpe        | 0.8553     | 0.8705     |
| annual_return | 0.0598     | 0.063      |
| annual_vol    | 0.0709     | 0.0733     |
| max_drawdown  | -0.1072    | -0.1122    |
| start         | 2004-07-01 | 2004-07-01 |
| end           | 2026-08-31 | 2026-08-31 |

Sharpe by year:

|      |   erc |   equal |
|-----:|------:|--------:|
| 2004 |  3.03 |    3.03 |
| 2005 |  1.33 |    1.33 |
| 2006 |  1.36 |    1.36 |
| 2007 |  1.88 |    1.88 |
| 2008 |  0.39 |    0.43 |
| 2009 |  0.13 |    0.07 |
| 2010 |  1.6  |    1.57 |
| 2011 |  0.73 |    0.8  |
| 2012 | -0.18 |   -0.27 |
| 2013 |  0.53 |    0.67 |
| 2014 |  0.79 |    0.92 |
| 2015 | -0.75 |   -0.5  |
| 2016 |  0.52 |    0.43 |
| 2017 |  1.94 |    1.65 |
| 2018 | -0.5  |   -0.36 |
| 2019 |  1.8  |    1.34 |
| 2020 |  0.52 |    0.8  |
| 2021 |  0.58 |    0.55 |
| 2022 |  0.52 |    0.89 |
| 2023 | -0.69 |   -0.89 |
| 2024 |  0.86 |    0.86 |
| 2025 |  1.75 |    1.77 |
| 2026 |  0.9  |    1.12 |

## Walk-forward over the trend grid

Grid: lookback (3, 6, 9, 12) months x per-asset vol target (0.2, 0.4). purgedcv WalkForwardSplit, 5 folds of 3 years, the variant with the best training Sharpe is held in the test window. Selected-in-sample OOS Sharpe **0.453** vs frozen v1 **0.378** over the same test windows.

|   fold | test_start   | test_end   | picked                          |   picked_train_sharpe |   picked_test_sharpe |   v1_test_sharpe |
|-------:|:-------------|:-----------|:--------------------------------|----------------------:|---------------------:|-----------------:|
|      1 | 2011-08-18   | 2014-08-20 | TrendETF lookback=9 target=0.2  |                 0.631 |                0.094 |            0.051 |
|      2 | 2014-08-21   | 2017-08-21 | TrendETF lookback=9 target=0.2  |                 0.523 |               -0.04  |            0.458 |
|      3 | 2017-08-22   | 2020-08-21 | TrendETF lookback=12 target=0.2 |                 0.462 |                0.219 |            0.219 |
|      4 | 2020-08-24   | 2023-08-24 | TrendETF lookback=9 target=0.2  |                 0.484 |                0.539 |            0.053 |
|      5 | 2023-08-25   | 2026-08-31 | TrendETF lookback=9 target=0.2  |                 0.492 |                1.141 |            0.98  |

## Falsification

**Placebo.** Trend sleeve with a fixed random sign per ETF, 20 draws: actual Sharpe 0.459, placebo mean -0.009, p95 0.382, beats 20 of 20.

**Vol scaling without the signal.** Same sizing, every sign forced long:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| trend_signal | 0.459 | 0.036 | 0.084 | -0.203 |
| long_only_vol_scaled | 0.618 | 0.047 | 0.079 | -0.234 |

**Cost stress.** Book at multiples of the cost model:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| 1x | 0.810 | 0.057 | 0.072 | -0.121 |
| 2x | 0.726 | 0.050 | 0.071 | -0.125 |
| 4x | 0.550 | 0.034 | 0.065 | -0.194 |

**Signal delay.** Every sleeve's targets held back N extra bars:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| +1 | 0.815 | 0.058 | 0.072 | -0.123 |
| +2 | 0.827 | 0.059 | 0.073 | -0.126 |
| +5 | 0.761 | 0.054 | 0.073 | -0.136 |

## v2 candidate: EWMAC

EWMAC at 16/64, 32/128, 64/256 and the three combined with a forecast diversification multiplier, forecast scalars estimated on trailing data only (expanding, NaN for the first 500 days), cap 20, forecast / 10 x TrendETF's 40% per-asset target (35-day vol where TrendETF uses 60-day) and class balance, sent daily, same universe, cost model and overlay. The engine's 10% buffer sits on the final weights after the overlay, per instrument: an instrument trades only when its held weight is outside target x (1 +/- 0.1). Window 2005-06-09 to 2026-08-31, where both rules are live. Gross is the same path with each day's trade costs and borrow added back. Turnover is traded notional over equity per year. Each variant is a logged trial.

| variant | sharpe_gross | sharpe | annual_return | annual_vol | max_drawdown | turnover | trades |
|---|---|---|---|---|---|---|---|
| TrendETF | 0.520 | 0.432 | 0.033 | 0.083 | -0.203 | 6.506 | 2506 |
| EWMAC | 0.837 | 0.681 | 0.066 | 0.102 | -0.194 | 18.537 | 27050 |
| EWMAC 16/64 | 0.746 | 0.527 | 0.048 | 0.098 | -0.177 | 26.098 | 34220 |
| EWMAC 32/128 | 0.861 | 0.712 | 0.072 | 0.105 | -0.165 | 18.028 | 25871 |
| EWMAC 64/256 | 0.774 | 0.658 | 0.065 | 0.103 | -0.196 | 13.254 | 20373 |

## EWMAC: walk-forward and the untouched window

Walk-forward over the four EWMAC variants on the sessions before the untouched window, 4 folds of 3 years, best training Sharpe held in the test window. Selected-in-sample OOS Sharpe **0.722** vs the frozen combined rule **0.659** over the same test windows.

|   fold | test_start   | test_end   | picked       |   picked_train_sharpe |   picked_test_sharpe |   v1_test_sharpe |
|-------:|:-------------|:-----------|:-------------|----------------------:|---------------------:|-----------------:|
|      1 | 2009-12-28   | 2012-12-27 | EWMAC        |                 0.954 |                0.569 |            0.569 |
|      2 | 2012-12-28   | 2015-12-29 | EWMAC        |                 0.801 |                1.084 |            1.084 |
|      3 | 2015-12-30   | 2018-12-31 | EWMAC        |                 0.881 |                0.388 |            0.388 |
|      4 | 2019-01-02   | 2021-12-30 | EWMAC 32/128 |                 0.778 |                0.877 |            0.597 |

**Untouched window** 2021-12-31 to 2026-08-31, the last 20% of the panel's sessions. Nothing was fit or picked on it; its return was inside the one full-panel look above, so it confirms rather than discovers. Promotion rule, fixed before this run: net Sharpe above zero on the untouched window, above TrendETF's on the same window, and a positive alpha t against the ETF universe there.

| rule | Sharpe gross | Sharpe net | ann. return | ann. vol | max drawdown | turnover | trades |
|---|---|---|---|---|---|---|---|
| EWMAC | 0.562 | 0.413 | 2.90% | 7.62% | -19.4% | 12.7x | 6196 |
| TrendETF | 0.978 | 0.874 | 5.93% | 6.86% | -8.0% | 6.8x | 614 |

EWMAC against its universe on the window: 1/N: alpha +2.79% (t 0.75), beta 0.04, benchmark Sharpe 0.98, residual Sharpe 0.37; 1/N vol-targeted: alpha +2.21% (t 0.60), beta 0.09, benchmark Sharpe 1.01, residual Sharpe 0.29. **Clears the rule: no.**

## Candidate books

Every book in `strategies.BOOKS` at 1/N of its sleeves, kill off, from 2005-06-09 (the first day all four are live) to the end; the same window, engine and cost model for all four. DSR uses the trial count above. OOS is the untouched window; the live book is the OOS winner only if it beats `v1` there. **OOS winner: `beta+alpha`** (live book `beta+alpha`).

| variant | sharpe_gross | sharpe | dsr | annual_return | annual_vol | max_drawdown | turnover | oos_sharpe | oos_max_drawdown |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.333 | 0.223 | 0.009 | 0.009 | 0.043 | -0.127 | 3.755 | 0.400 | -0.092 |
| v1+ewmac | 0.695 | 0.536 | 0.178 | 0.030 | 0.058 | -0.135 | 9.683 | 0.457 | -0.135 |
| ewmac-for-trend | 0.722 | 0.551 | 0.197 | 0.031 | 0.058 | -0.144 | 10.832 | 0.396 | -0.144 |
| beta+alpha | 0.882 | 0.760 | 0.538 | 0.054 | 0.073 | -0.121 | 10.479 | 0.789 | -0.117 |

ERC vs 1/N gate on `beta+alpha`'s sleeves, quarterly refits, out of sample:

|               | erc        | equal      |
|:--------------|:-----------|:-----------|
| sharpe        | 0.8556     | 0.8709     |
| annual_return | 0.0598     | 0.0631     |
| annual_vol    | 0.0709     | 0.0733     |
| max_drawdown  | -0.1072    | -0.1122    |
| start         | 2004-07-01 | 2004-07-01 |
| end           | 2026-08-31 | 2026-08-31 |

## Candidate allocator: cost-aware mean-variance

`optimizer.py` on each book's sleeve streams: maximise mu'w - w'Sw / 2 - cost'|dw| over long-only sleeve weights, parameters {'risk_aversion': 1.0, 'target_vol': 0.1, 'max_gross': 1.5, 'max_net': 1.5, 'max_sleeve': 0.75, 'turnover': 0.5, 'kelly': 0.5, 'financing_spread_bps': 50.0} fixed before the run. mu is the trailing three-year mean, S Ledoit-Wolf (no `risk-model/` at run time), cost each sleeve's realised cost per traded notional times its gross. Quarterly refits, weights held for the next quarter, the move charged on its first day, leverage above 1 financed at data-lake fred/DTB3 plus the spread. Both columns are daily-rebalanced to their weights; the 1/N column here is that convention, the books table above is the engine's static 1/N, and the rule compares against the engine's. Every row is a logged trial and a research-registry run (4 there). Rule, written before the run: the candidate allocator goes live only if its best book's Sharpe on the untouched window beats the best 1/N book's there, net of reallocation costs and financing; otherwise 1/N stays live and the candidate is a documented trial.

| book | walk Sharpe mvo | walk Sharpe 1/N | DSR | DSR (registry) | untouched Sharpe mvo | untouched Sharpe 1/N (walk) | untouched Sharpe 1/N (engine) | mean leverage | turnover / refit | refits | budget bound | vol bound | Kelly scaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.054 | 0.239 | 0.002 | 0.008 | -0.436 | 0.484 | 0.400 | 0.67 | 0.24 | 86 | 24 | 1 | 13 |
| v1+ewmac | 0.446 | 0.522 | 0.093 | 0.190 | -0.265 | 0.533 | 0.457 | 1.08 | 0.24 | 86 | 21 | 42 | 1 |
| ewmac-for-trend | 0.444 | 0.523 | 0.098 | 0.194 | -0.260 | 0.452 | 0.396 | 0.91 | 0.17 | 82 | 19 | 1 | 1 |
| beta+alpha | 0.726 | 0.870 | 0.475 | 0.652 | 0.660 | 0.863 | 0.789 | 1.13 | 0.10 | 90 | 7 | 36 | 0 |

Best candidate book on the untouched window: `beta+alpha` at 0.660 against the best 1/N book `beta+alpha` at 0.789. **Promoted: no.** `allocate.LIVE_ALLOCATOR = "rule"`. The live book's candidate weight path is in `mvo_weights.csv`.

## Signal vs beta

Each sleeve's live net returns regressed on its own universe: the daily-rebalanced 1/N return of its instruments, and that return scaled to the sleeves' 10% vol target (60-day EWMA vol, one-day lag, 3x cap). Alpha is annualised with a Newey-West t; residual Sharpe is the Sharpe of the sleeve minus beta x benchmark. A sleeve whose residual Sharpe is below the benchmark's own Sharpe is not adding return beyond owning the universe.

| sleeve | benchmark | benchmark Sharpe | sleeve Sharpe | alpha | t | beta | residual Sharpe |
|---|---|---|---|---|---|---|---|
| TrendETF | 1/N | 0.626 | 0.459 | +2.81% | 1.63 | 0.153 | 0.342 |
| TrendETF | 1/N vol-targeted | 0.611 | 0.459 | +2.44% | 1.45 | 0.225 | 0.303 |
| FXCarryETF | 1/N | 0.076 | -0.160 | -0.94% | -0.71 | -0.086 | -0.153 |
| FXCarryETF | 1/N vol-targeted | 0.017 | -0.160 | -0.97% | -0.73 | -0.084 | -0.159 |
| CryptoTrend | 1/N | 0.440 | 0.073 | -2.13% | -0.96 | 0.106 | -0.438 |
| CryptoTrend | 1/N vol-targeted | 0.517 | 0.073 | -2.71% | -1.24 | 0.631 | -0.585 |
| ETFBeta | 1/N | 0.670 | 0.662 | +0.67% | 0.76 | 0.673 | 0.163 |
| ETFBeta | 1/N vol-targeted | 0.651 | 0.656 | +0.35% | 0.57 | 0.774 | 0.126 |
| EWMAC | 1/N | 0.598 | 0.681 | +6.61% | 3.07 | 0.052 | 0.649 |
| EWMAC | 1/N vol-targeted | 0.575 | 0.681 | +6.17% | 2.89 | 0.132 | 0.611 |

## Kill switch

Backtests above run with `kill_dd=None`. With the live 25% kill on, the sleeves would have been flattened permanently on: TrendETF never, FXCarryETF 2010-08-31, CryptoTrend never, ETFBeta 2016-01-08, EWMAC never. Live, a kill stops the sleeve until Kelvin restarts it, which no backtest can model.

## purgedcv vs archived formula (risk #8)

Best of 100 random series, 750 days: DSR purgedcv 0.6855 vs archive 0.6848; PSR 0.9970 vs 0.9970. purgedcv is pinned at 0.1.5 in requirements.txt.

Tearsheet: `reports/tearsheet.html` (quantstats, SPY benchmark). 3087s.
