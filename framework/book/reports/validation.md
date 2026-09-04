# Validation

Run 2026-09-04 02:54, panel hash `aa436bbc6c2e4712`, 2004-07-01 to 2026-08-31 (22.1 years), shadow cost model, kill switch off for backtests (see below).

## Headline (book, live allocator, net)

| Sharpe | ann. return | ann. vol | max drawdown | PSR | DSR | trials | MinBTL | MinTRL |
|---|---|---|---|---|---|---|---|---|
| 0.262 | 1.04% | 4.30% | -13.5% | 0.890 | 0.058 | 22 | 55.0 y (have 22.1) | 39.8 y |

DSR uses n_trials = 22: every configuration this file runs on the real panel (sleeves, the trend grid, allocators, cost and delay variants, kill on) is appended to `trials.csv` (80 rows so far) and identical return streams count once; the variance of their daily Sharpes is 3.75e-04. The per-asset vol target is undone by the sleeve-level 10% target, so the trend grid is really four lookbacks. The 20 placebo draws are a null, not candidates; counting them too gives DSR 0.026. PSR is the probability the true Sharpe is above zero; DSR the same after deflating for selection. MinTRL is the track length needed to reject zero at 95%.

## Allocator: ERC vs 1/N, out of sample

Quarterly refits, three-year window, Ledoit-Wolf covariance, weights applied to the next quarter. Live allocator: **equal** with weights {'TrendETF': 0.3333, 'FXCarryETF': 0.3333, 'CryptoTrend': 0.3333} (ERC would have been {'TrendETF': 0.3184, 'FXCarryETF': 0.4203, 'CryptoTrend': 0.2613}).

|               | erc        | equal      |
|:--------------|:-----------|:-----------|
| sharpe        | 0.2275     | 0.2327     |
| annual_return | 0.0113     | 0.0118     |
| annual_vol    | 0.0565     | 0.0574     |
| max_drawdown  | -0.1495    | -0.148     |
| start         | 2005-07-01 | 2005-07-01 |
| end           | 2026-08-31 | 2026-08-31 |

Sharpe by year:

|      |   erc |   equal |
|-----:|------:|--------:|
| 2005 |  1.15 |    1.15 |
| 2006 | -0.01 |   -0.01 |
| 2007 |  1.82 |    1.81 |
| 2008 | -0.59 |   -0.63 |
| 2009 | -0.33 |   -0.32 |
| 2010 | -0.71 |   -0.68 |
| 2011 | -0.1  |   -0.13 |
| 2012 |  1.22 |    1.31 |
| 2013 |  0.26 |    0.33 |
| 2014 |  0.06 |    0.22 |
| 2015 |  0.44 |    0.74 |
| 2016 | -1.18 |   -1.28 |
| 2017 | -0.13 |    0.08 |
| 2018 | -0.67 |   -0.75 |
| 2019 |  0.95 |    1.01 |
| 2020 | -1.05 |   -1.03 |
| 2021 |  0.52 |    0.45 |
| 2022 | -0.22 |   -0.18 |
| 2023 | -0.08 |   -0.11 |
| 2024 |  1.31 |    1.31 |
| 2025 |  0.29 |    0.16 |
| 2026 |  1.55 |    1.47 |

## Walk-forward over the trend grid

Grid: lookback (3, 6, 9, 12) months x per-asset vol target (0.2, 0.4). purgedcv WalkForwardSplit, 5 folds of 3 years, the variant with the best training Sharpe is held in the test window. Selected-in-sample OOS Sharpe **0.441** vs frozen v1 **0.310** over the same test windows.

|   fold | test_start   | test_end   | picked                          |   picked_train_sharpe |   picked_test_sharpe |   v1_test_sharpe |
|-------:|:-------------|:-----------|:--------------------------------|----------------------:|---------------------:|-----------------:|
|      1 | 2011-08-18   | 2014-08-20 | TrendETF lookback=9 target=0.2  |                 0.619 |                0.085 |            0.062 |
|      2 | 2014-08-21   | 2017-08-21 | TrendETF lookback=9 target=0.2  |                 0.511 |               -0.087 |            0.476 |
|      3 | 2017-08-22   | 2020-08-21 | TrendETF lookback=12 target=0.2 |                 0.461 |                0.225 |            0.225 |
|      4 | 2020-08-24   | 2023-08-24 | TrendETF lookback=9 target=0.2  |                 0.475 |                0.534 |           -0.222 |
|      5 | 2023-08-25   | 2026-08-31 | TrendETF lookback=9 target=0.2  |                 0.483 |                1.068 |            0.87  |

## Falsification

**Placebo.** Trend sleeve with a fixed random sign per ETF, 20 draws: actual Sharpe 0.412, placebo mean -0.016, p95 0.413, beats 19 of 20.

**Vol scaling without the signal.** Same sizing, every sign forced long:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| trend_signal | 0.412 | 0.031 | 0.083 | -0.198 |
| long_only_vol_scaled | 0.590 | 0.046 | 0.082 | -0.234 |

**Cost stress.** Book at multiples of the cost model:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| 1x | 0.262 | 0.010 | 0.043 | -0.135 |
| 2x | 0.130 | 0.004 | 0.040 | -0.154 |
| 4x | -0.090 | -0.004 | 0.037 | -0.238 |

**Signal delay.** Every sleeve's targets held back N extra bars:

| variant | sharpe | annual_return | annual_vol | max_drawdown |
|---|---|---|---|---|
| +1 | 0.133 | 0.004 | 0.039 | -0.146 |
| +2 | 0.214 | 0.008 | 0.043 | -0.122 |
| +5 | 0.027 | 0.000 | 0.039 | -0.180 |

## v2 candidate: EWMAC

EWMAC at 16/64, 32/128, 64/256 and the three combined with a forecast diversification multiplier, forecast scalars estimated on trailing data only (expanding, NaN for the first 500 days), cap 20, forecast / 10 x TrendETF's 40% per-asset target (35-day vol where TrendETF uses 60-day) and class balance, daily, same universe, cost model and overlay. The 10% buffer sits on the raw targets; on any day one instrument leaves its band the whole sleeve is re-sent and the overlay re-scales it, so it cuts trades less than the same buffer on final positions would. Window 2005-06-09 to 2026-08-31, where both rules are live. Gross is the same run with every cost coefficient at zero, run for the combined rule only. Turnover is traded notional over equity per year. The combined rule gross and net and each speed net are logged trials.

| variant | sharpe_gross | sharpe | annual_return | annual_vol | max_drawdown | turnover | trades |
|---|---|---|---|---|---|---|---|
| TrendETF | 0.402 | 0.382 | 0.028 | 0.082 | -0.198 | 6.783 | 3900 |
| EWMAC | 0.842 | 0.711 | 0.070 | 0.103 | -0.201 | 19.288 | 72860 |
| EWMAC 16/64 | nan | 0.538 | 0.049 | 0.098 | -0.177 | 26.728 | 72867 |
| EWMAC 32/128 | nan | 0.703 | 0.072 | 0.106 | -0.167 | 18.844 | 72867 |
| EWMAC 64/256 | nan | 0.568 | 0.054 | 0.102 | -0.195 | 13.943 | 72810 |

## Signal vs beta

Each sleeve's live net returns regressed on its own universe: the daily-rebalanced 1/N return of its instruments, and that return scaled to the sleeves' 10% vol target (60-day EWMA vol, one-day lag, 3x cap). Alpha is annualised with a Newey-West t; residual Sharpe is the Sharpe of the sleeve minus beta x benchmark. A sleeve whose residual Sharpe is below the benchmark's own Sharpe is not adding return beyond owning the universe.

| sleeve | benchmark | benchmark Sharpe | sleeve Sharpe | alpha | t | beta | residual Sharpe |
|---|---|---|---|---|---|---|---|
| TrendETF | 1/N | 0.626 | 0.412 | +2.48% | 1.43 | 0.141 | 0.302 |
| TrendETF | 1/N vol-targeted | 0.611 | 0.412 | +2.12% | 1.25 | 0.212 | 0.263 |
| FXCarryETF | 1/N | 0.076 | -0.171 | -1.01% | -0.76 | -0.086 | -0.164 |
| FXCarryETF | 1/N vol-targeted | 0.017 | -0.171 | -1.04% | -0.78 | -0.084 | -0.170 |
| CryptoTrend | 1/N | 0.440 | 0.051 | -2.32% | -1.04 | 0.106 | -0.473 |
| CryptoTrend | 1/N vol-targeted | 0.517 | 0.051 | -2.90% | -1.32 | 0.635 | -0.622 |
| EWMAC | 1/N | 0.598 | 0.711 | +6.98% | 3.20 | 0.054 | 0.678 |
| EWMAC | 1/N vol-targeted | 0.575 | 0.711 | +6.55% | 3.02 | 0.134 | 0.641 |

## Kill switch

Backtests above run with `kill_dd=None`. With the live 25% kill on, the sleeves would have been flattened permanently on: TrendETF never, FXCarryETF 2011-08-08, CryptoTrend never. Live, a kill stops the sleeve until Kelvin restarts it, which no backtest can model.

## purgedcv vs archived formula (risk #8)

Best of 100 random series, 750 days: DSR purgedcv 0.6855 vs archive 0.6848; PSR 0.9970 vs 0.9970. purgedcv is pinned at 0.1.5 in requirements.txt.

Tearsheet: `reports/tearsheet.html` (quantstats, SPY benchmark). 1015s.
