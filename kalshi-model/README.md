# ML calibration on Kalshi

A supervised calibration model for Kalshi binaries: the quoted book, its
recent path, volume, open interest, time to close and category go in, a
probability of YES comes out, and it is traded only where the model's edge
at the mid exceeds the whole quoted spread. It sits on top of
`../prediction-markets/` (the settled-market snapshot and the bin-based
longshot fade) and runs through `../research/` (registry, model registry,
one untouched window), so every headline number here was read once on
markets nothing was fit or picked on.

The short version: the market wins. On the untouched last fifth of the
sample (2,318 markets closing 2026-08-25 to 2026-09-03) the mid scores
log-loss 0.5097 and Brier 0.1708; the best ML variant scores 0.5136 and
0.1718, the published bin curve 0.5139 and 0.1719. Every trade rule loses
money at the quoted touch plus the taker fee: the bin curve -6.5 cents a
dollar staked over 871 bets (t = -2.95), the ML model -5.6 cents over 684
(t = -2.33). Both are negative at the mid with no cost too, though not
significantly there (t = -1.59 and -0.46). The walk-forward on the
markets before the window picked the bin curve over any estimator in two of
four folds, so the ML added nothing to the published fade, and the fade
itself does not survive its costs, which is what the research project said.

## Data

Two sources, both read through their owners' loaders, nothing re-fetched.

- **Labels and the quote**: `prediction-markets/data.py::load_snapshot()`,
  one two-sided book per settled binary market read from hourly candles
  24 hours before close, 14,560 markets closing 2026-06-26 to 2026-09-03,
  77% sports. That project's README explains why the quote at a fixed
  horizon is used and not `last_price`.
- **The path**: `candles.py` pulls the 96 hourly candles that end at that
  quote for each snapshot market (`source-material/kalshi-model/candles.csv`,
  resumable, about 50 minutes at Kalshi's rate limit). 11,123 of the
  14,560 answered with at least one bar; the rest are the training set's
  missing quarter and I have not checked whether they differ.
- **The data lake**: `lake.load("kalshi_markets")` holds 30,000 listing rows
  on 7,121 tickers over 6 hourly snapshots, all on 2026-09-05 and all
  active. Zero settled rows, so it contributes no labels yet; it is the
  scoring universe the desk hook below reads, not training data. When the
  hourly collector has run for a few weeks the lake will hold a book at
  every horizon for every market and this project should retrain on that.

Nothing dated after the quote enters a feature: the API filters on
`end_period_ts`, `path_features()` drops later bars again, and `check.py`
asserts a planted bar one second after the quote changes nothing. The
listing's `volume` and `open_interest` are settlement-time values and are
kept out of the feature set; `hgb+none+settle_volume` is run alongside the
grid on purpose, as a leak detector, and is never a candidate.

## Features

26 columns per market, all as of the quote: `logit_mid`, `spread`,
`log_spread_rel` (spread over the cheaper side), `quote_age_h`,
`hours_to_close`, `log_duration`, `log_vol_96`, `log_vol_24`, `log_oi`,
`n_quoted`, `n_traded`, `mid_chg_24`, `mid_chg_96`, `mid_vol`, `mid_range`,
`last_gap` (last trade minus mid), `no_trade`, and nine category one-hots.

## Grid and protocol

Frozen before any result (top of `run.py`): three estimators (logistic
regression, histogram gradient boosting, a two-layer MLP in torch on MPS)
times three post-calibrations (none, isotonic, Platt, both through
`CalibratedClassifierCV` with five folds) plus the bin curve from
`prediction-markets/calibration.py`, ten candidates; baseline
`logit+platt`; four expanding walk-forward folds of 1,467 markets each on
the markets before the untouched window (`purgedcv.WalkForwardSplit` on
quote and close times, plus a strict purge because purgedcv keeps a
training market that closes at the same second as the first test quote);
inside each training window the variant with the lowest log-loss on the
last fifth, purged by the 24-hour label, is held on the test fold; the
chosen variant is the one picked most often, ties to the baseline; the
untouched window is the last fifth of markets by close time, its start
rounded back to midnight UTC by the lock, so 2,318 markets (20.8%) against
the lock's exact count of 2,225 (`research.alpha.Untouched`, opened once,
result stored in the lock); it is
read for the chosen variant, the baseline, the bin curve and the mid, four
cost scenarios each, and every one of those reads is a registry row.

The trade rule is stricter than the research project's Kelly rule: a market
is traded only when |P(YES) - mid| exceeds the whole quoted spread, so at
least half a spread of edge is left after crossing to the touch. Costs are
the touch (YES at the ask, NO at one minus the bid) plus Kalshi's taker fee,
0.07 x P x (1 - P) a contract, then exact binary Kelly at quarter size with
the research project's daily compounding (`kelly.py`).

## Walk-forward, before the window

| fold | train | test | picked | picked test log-loss | baseline | market |
|---|---|---|---|---|---|---|
| 1 | 06-26 to 07-22 (2,699) | 07-23 to 08-01 (1,467) | logit+platt | 0.4430 | 0.4430 | 0.4495 |
| 2 | 06-26 to 07-31 (4,237) | 08-01 to 08-09 | logit+isotonic | 0.4490 | 0.4420 | 0.4427 |
| 3 | 06-26 to 08-08 (5,677) | 08-09 to 08-17 | bias | 0.4439 | 0.4465 | 0.4476 |
| 4 | 06-26 to 08-16 (7,205) | 08-17 to 08-24 | bias | 0.4463 | 0.4434 | 0.4495 |

Every variant on the stitched 5,868 test markets, headline cost scenario:

| variant | log-loss | Brier | bets | payoff / $ | t |
|---|---|---|---|---|---|
| market mid | 0.4473 | 0.1475 | | | |
| bias curve | **0.4431** | 0.1464 | 2,368 | -0.007 | -0.65 |
| logit+platt (baseline) | 0.4437 | 0.1463 | 1,913 | -0.044 | -2.89 |
| logit+none | 0.4439 | 0.1463 | 2,131 | -0.031 | -2.23 |
| mlp+isotonic | 0.4449 | 0.1468 | 2,215 | -0.012 | -0.53 |
| mlp+none | 0.4462 | 0.1468 | 1,831 | -0.115 | -4.18 |
| logit+isotonic | 0.4469 | 0.1468 | 2,447 | -0.008 | -0.66 |
| hgb+platt | 0.4532 | 0.1498 | 2,390 | -0.156 | -6.05 |
| mlp+platt | 0.4538 | 0.1482 | 2,960 | -0.224 | -9.06 |
| hgb+isotonic | 0.4543 | 0.1500 | 2,617 | -0.109 | -5.16 |
| hgb+none | 0.4582 | 0.1517 | 2,897 | -0.042 | -2.42 |
| selected (walk-forward) | 0.4455 | 0.1466 | 2,399 | -0.013 | -0.99 |
| leak: hgb + settlement volume | 0.4435 | 0.1471 | 3,056 | +0.025 | +1.10 |

The linear models and the bin curve beat the mid by 0.003 to 0.004 of
log-loss on these folds; the tree and the network, with or without
post-calibration, are worse than the mid and lose the most money. Nothing
honest makes money after the spread. The one positive row is the planted
leak, and even that is only t = 1.1: settlement volume carries less about
the outcome than I expected, which is a reason to keep the detector, not to
trust it.

The chart `reports/walk_forward.png` is each variant's log-loss minus the
market's on these folds.

## Untouched window

2,318 markets closing 2026-08-25 to 2026-09-03 (ten calendar days, 82%
sports), models trained on the 8,627 markets that closed at least 24 hours
before the window's first quote, opened once at 2026-09-05 01:08
(`reports/untouched.json`).

| variant | log-loss | Brier | reliability | scenario | bets | payoff / $ | t |
|---|---|---|---|---|---|---|---|
| market mid | **0.5097** | **0.1708** | 0.0004 | | | | |
| logit+platt | 0.5136 | 0.1718 | 0.0019 | mid, no cost | 774 | -0.011 | -0.46 |
| | | | | quoted spread | 774 | -0.026 | -1.05 |
| | | | | **quoted spread + fee** | 684 | **-0.056** | **-2.33** |
| | | | | Kelly rule, spread + fee | 881 | -0.047 | -2.38 |
| bias curve (chosen) | 0.5139 | 0.1719 | 0.0016 | mid, no cost | 897 | -0.036 | -1.59 |
| | | | | quoted spread | 897 | -0.049 | -2.17 |
| | | | | **quoted spread + fee** | 871 | **-0.065** | **-2.95** |
| | | | | Kelly rule, spread + fee | 1,116 | -0.057 | -3.12 |

Registry rows for the two headline reads: `fbdfc429ea029290` (bias curve,
quoted spread + fee) and `a5d888e28572b54d` (logit+platt). Their DSR is
0.001 and 0.010 over the 20 trials logged in `reports/registry/runs.jsonl`
(ten grid variants, the leak, the selected stream, and the eight window
reads that produced distinct return streams), against a variance of the
trials' daily Sharpes of 0.136. Those DSRs deflate a daily Sharpe computed
on ten daily observations and should be read as "negative", not as a
probability. Per-bet t-statistics on 684 and 871 bets are the numbers that
mean something, and both are below -2.

What the window did that the folds did not: the YES rate rose from 32.4% to
37.8%, so cheap contracts paid more often than in July, and every rule here
is a longshot fade (810 of the bin curve's 871 bets and 596 of the model's
684 are NO). A fade in a week where the longshots come in loses even at the
mid. The reliability term of the mid (0.0004) says the market was better
calibrated on this window than on the training data (0.0018 on the folds),
which is the same statement.

By quoted spread on the window (log-loss, market / logit+platt / bias):

| spread | n | market | logit+platt | bias |
|---|---|---|---|---|
| 5c or tighter | 1,403 | **0.4866** | 0.4942 | 0.4961 |
| 5c to 10c | 264 | 0.4984 | **0.4983** | 0.5078 |
| 10c to 20c | 171 | 0.4598 | 0.4722 | **0.4526** |
| wider than 20c | 480 | 0.6010 | 0.5935 | **0.5914** |

The models beat the mid only in books wider than 10 cents, where the mid
is the midpoint of nothing and where the edge-over-spread rule never fires
(868 of the 871 bin-curve bets were in books 5 cents wide or tighter;
median spread traded, 1 cent). In tight books, the only ones traded, the
mid is better than both models by 0.008 to 0.010. That is the
`prediction-markets` finding again from the other side: what looks like
mispricing is mostly wide books, and the liquid books are priced better
than a model fit on ten weeks can price them.

By category: the model is better than the mid on weather (178 markets,
0.3872 vs 0.3914) and commodities (63) and worse on sports (1,906, 0.5388
vs 0.5349), which is the sample.

`reports/calibration.png` has the reliability curves of the mid, the bin
curve and the model on the window; `reports/equity.png` the quarter-Kelly
bankroll under each cost scenario.

## Desk hook

`kalshi-desk/scanner.py` scores markets with `load_curve()`. Set
`SCANNER_MODEL=kalshi-model` and it loads the chosen model from
`reports/models/` (`model.curve()`: pulls the market's candles, builds the
same feature row, returns P(YES)); unset, it is the bin curve exactly as
before. Given the table above the default should stay the bin curve, and
neither should be traded at the touch. The model registry holds the chosen
variant (`0ee881a512f0`, the bin curve refit on the 8,627 training markets)
and the baseline (`a45df96b92e5`, logit+platt), each with the training
frame's hash and the window log-loss as its OOS score.

## Running it

```
../.venv/bin/python3 check.py      # offline, planted markets, 57 checks, about 3 minutes
../.venv/bin/python3 candles.py    # the candle pull, resumable, network
../.venv/bin/python3 run.py        # about 2.5 minutes; a rerun reads the window back from the lock
```

`check.py` simulates markets with a Prelec distortion between the quoted
mid and the true probability and asserts that every estimator in the grid
recovers most of it, that on calibrated markets no variant produces a
t-statistic above 3 after the spread, that a bar dated after the quote
changes no feature, that every fold's training markets closed before the
first test quote (including one planted at the same second), that the fee
and payoff arithmetic match hand values, and then runs the whole `run.py`
pipeline, untouched window and all, on the planted table into a temporary
directory, twice, and checks that the second run reads the window from the
lock and logs no new trial. That last check is what let the real window be
opened once with code that had already been through its own crash paths.

## What I would not claim from this

- **Ten weeks of markets, a ten-day window.** The untouched window is
  2,318 markets on ten calendar days; the DSRs are computed on ten daily
  returns and are meaningless as probabilities. The per-bet t-statistics
  are the evidence, and they say the fade lost.
- **It is a sports dataset** at 77% (82% in the window), read 24 hours
  before close. The result is about game props in late August.
- **The training set is the markets that answered a candle request**,
  11,123 of 14,560. I have not compared the ones that did not.
- **The spread is a snapshot and the fee is the schedule.** No size at the
  touch, no partial fills, a Kelly stake assumed filled at the quote; a real
  fill on a market trading a few hundred contracts is worse. The 7% taker
  fee is Kalshi's published formula, not a fill record.
- **The lake has no labels yet.** One day of listings, zero settlements.
  The retrain on lake data, with a book at every horizon instead of one at
  T-24h, is the next version of this project and the one that could answer
  whether the T-24h result holds at T-1h or T-7d.
- **The walk-forward chose the bin curve**, so the model registry's
  "chosen" entry is a refit of the published fade and the ML variants are
  reported as the baseline. That is the protocol working, not a bug.
- **The window has been read.** Any change to the grid, the features or
  the rule judged against `reports/untouched.json` is in-sample from here.
