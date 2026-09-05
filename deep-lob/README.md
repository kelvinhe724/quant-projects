# DeepLOB-style classifier on one-second BTC bars

Does a DeepLOB-shaped network (conv stack, inception block, LSTM, one softmax
head per horizon) read the next 5, 10 or 30 seconds of BTC better than a
logistic on imbalance or a plain LightGBM, and is any of it worth a taker fee?
On seven days of Binance spot BTCUSDT, the network is the only one of the three
that predicts anything but "flat", its gross edge is a tenth of a basis point a
trade, and at a 1 bp fee every variant loses. The untouched last two days say
the same thing as the walk-forward.

Built through `../research/` (registry, deflated Sharpe, model registry, the
untouched-window lock, the harness's fill-and-cost backtest) and
`../data-lake/` (the L2 stream, when it holds enough), with the imbalance
baseline imported from `../orderbook-imbalance/`.

## Data, and what it is not

The spec was the lake's `crypto_l2` stream: twenty levels a side, once a
second. When this was built the stream held **0.8 hours** of BTCUSDT (it
started 2026-09-05 01:10 UTC), below the six-hour floor, so `data.load()`
backfilled and said so. The backfill is not an order book:

- Binance's `bookTicker` archive on data.binance.vision is the only book
  history there. It stopped on 2024-03 and runs 200 to 250 MB a day; nothing
  exists for 2026. The spot `bookTicker` prefix is empty at every date.
- What does exist for last week is spot `aggTrades` (12 to 19 MB a day). Seven
  days, 2026-08-28 to 2026-09-03, 604,800 seconds, 0.6 to 1.3 million trades a day.

`data.seconds_from_trades` turns the tape into one row per second: the last
print as `mid`, the log return in bps, taker buy and sell volume, trade count,
VWAP offset, and a touch reconstructed from the prints (a taker buy prints at
the ask, a taker sell at the bid, so the last of each inside the second is the
best quote as of that print; offsets in bps from `mid`), plus the flow
imbalance through `orderbook-imbalance.imbalance`. Eight features a second.
7.6% of seconds have no trade and carry the previous price with zero flow;
the reconstructed touch is crossed in 2.0% of seconds, where one side's last
print is stale.

The lake path is written and checked (`seconds_from_lake`: `mid` from the
touch, then each level's price offset in bps and size, 80 columns) and
`load()` takes it once the stream has six hours. The network's first
convolution spans the whole feature row, so the same code runs on 8 or 80
columns; the paper's level-walking convolutions are not there because the
archive has no levels to walk. Retraining on the lake is a new project run,
not a continuation: different venue (Bybit, OKX fallback, against Binance),
different schema.

## Design

Frozen before any result: horizons 5, 10 and 30 s; dead band 0.5 x sqrt(h)
bps (1.1, 1.6, 2.7); three classes; a 100-second window; the grid model x
horizon x tau with tau in {0.1, 0.3} on P(up) - P(down); the baseline variant
DeepLOB at 10 s and tau 0.1; selection by validation-slice daily Sharpe; the
untouched window the last two of the seven days.

**Timing.** A bar dated t holds what printed in [t, t+1). A prediction dated t
reads bars <= t. The position it implies fills at the price of bar t+1 and
earns from there. That lag is one line, `pos.shift(1)` in `lob.pnl`, ahead of
the harness's `backtest`, which charges the fee on traded weight. `check.py`
plants a jump between bar t and t+1 and asserts the position decided at t
earns nothing but the fee.

**Cost.** 1 bp a side on traded weight. The spread is not charged: on this
tape the reconstructed half-spread is a tick at the median and 0.03 bps on
average, thirty times smaller than the fee. Binance's own retail spot taker
fee is 10 bps; 1 bp is a wholesale assumption and the tables carry gross next
to net so any fee can be read off.

**Models.** `logit-imbalance` is one logistic per horizon on one feature, the
five-second flow imbalance (the book imbalance over five levels on the lake
path), which is the regressor `orderbook-imbalance/` found predictive.
`lightgbm` is 300 trees, learning rate 0.05, on the eight features at lags 0,
1, 2, 5, 10 and 30 s, early-stopped on the validation slice. `deeplob` is the
block structure of Zhang, Zohren and Roberts (2019) on a 100 x 8 window,
z-scored with the training rows' moments, Adam 1e-3, batch 512, up to 8
epochs, the epoch with the best validation loss kept. Trained on Apple MPS;
the final fit on 388,773 rows took 117 s.

**Walk-forward.** purgedcv `WalkForwardSplit`, three folds of one day (08-30,
08-31, 09-01) on the five pre-window days, expanding training window, the 31
seconds before each test day purged so no training label touches a test
price. The last 10% of each training window is the validation slice: every
model early-stops on it and every grid variant is scored on it. The fold's
pick runs on its test day; the 18 variants' stitched test P&Ls and the
selected stream are logged as trials. The most-picked variant is refit on the
five days and the window opened once for all 18. The harness's Alpha
interface (a cross-section at one date) does not fit a windowed classifier,
so the models are plain fit/proba objects and the rest of the harness is used
as is.

## Results

`run.py`, 2026-09-04, data hash `887ede0f45ece32a`, 388 s. Pre-window class
shares: 5 s 82% flat / 9% up / 9% down; 10 s 78 / 11 / 11; 30 s 73 / 13 / 14.

### Walk-forward

| fold | test day | picked | val daily Sharpe | test daily Sharpe | test net bps |
|---|---|---|---|---|---|
| 1 | 2026-08-30 | deeplob h=30 tau=0.1 | -15.1 | -33.7 | -3,030 |
| 2 | 2026-08-31 | lightgbm h=30 tau=0.3 | -4.0 | -3.5 | -24 |
| 3 | 2026-09-01 | deeplob h=10 tau=0.3 | -4.2 | -5.0 | -27 |

Three folds, three picks, every validation score negative: the selection is
choosing the variant that loses least, and what it picks is mostly the one
that barely trades. Stitched selected stream over the three days: -3,081 bps
net, +489 gross, 3,547 trades, in the market 2% of the time. Chosen for the
window by the most-picked rule with the tie broken by first pick:
`deeplob h=30 tau=0.1`.

Every variant's stitched test P&L, 72 hours (`reports/walk_forward_variants.csv`):

| variant | net bps | gross bps | trades | gross bps / trade | in market |
|---|---|---|---|---|---|
| deeplob h=5 tau=0.1 | -33,808 | +5,756 | 37,925 | +0.15 | 23% |
| deeplob h=10 tau=0.1 | -32,451 | +5,651 | 36,749 | +0.15 | 22% |
| deeplob h=30 tau=0.1 | -20,561 | +3,305 | 23,615 | +0.14 | 14% |
| lightgbm h=5 tau=0.1 | -30,565 | +3,619 | 33,499 | +0.11 | 12% |
| lightgbm h=10 tau=0.1 | -32,010 | +3,618 | 34,981 | +0.10 | 12% |
| lightgbm h=30 tau=0.1 | -24,433 | +2,143 | 26,433 | +0.08 | 8% |
| every tau=0.3 variant | -2 to -776 | 0 to +88 | 2 to 858 | | <1% |
| logit-imbalance, all six | 0 | 0 | 0 | | 0% |

The logistic never trades: with the flat class at 78 to 82% its P(up) - P(down)
never clears 0.1. The network and the trees do find something gross: +0.10 to
+0.15 bps a trade, which is the same size as the +0.11 to +0.12 bps a signal
the `orderbook-imbalance/` project measured on OKX with a real book. Then the
fee takes 2 bps a round trip.

Classification, mean over the three test days (`reports/classification.csv`).
Accuracy is within 0.002 of the majority share for every model at every
horizon; macro F1 is where they differ. An "always flat" classifier scores
2p / (1 + p) / 3 at majority share p:

| horizon | always flat | logit-imbalance | lightgbm | deeplob |
|---|---|---|---|---|
| 5 s | 0.298 | 0.298 | 0.309 | 0.341 |
| 10 s | 0.289 | 0.289 | 0.302 | 0.334 |
| 30 s | 0.276 | 0.275 | 0.289 | 0.321 |

### Untouched window

2026-09-02 00:00 to 2026-09-03 23:59 UTC, 172,800 seconds, opened
2026-09-04 21:08 and stored in `reports/untouched.json`. The models were refit
on every pre-window second whose label ends before the window.

Headline, the chosen variant `deeplob h=30 tau=0.1`: **-17,180 bps net over 48
hours** (-8,590 a day), +1,638 gross, 18,781 trades at -0.915 net and +0.087
gross a trade, in the market 8% of the time, daily-unit Sharpe -85. **PSR
0.000, DSR 0.000 over 14 trials** (the trials logged before the window was
read; identical all-zero streams count once). The registry entry is
`deeplob` with config `{horizon: 30, tau: 0.1, window: untouched}` in
`reports/registry/runs.jsonl`; the model is `9e99b8a4aa46` in
`reports/models/`.

All variants on the window (`reports/untouched_variants.csv`):

| variant | net bps | gross bps | trades | gross bps / trade | breakeven fee, bps a side | in market |
|---|---|---|---|---|---|---|
| deeplob h=5 tau=0.1 | -34,210 | +5,068 | 37,907 | +0.13 | 0.13 | 28% |
| deeplob h=10 tau=0.1 | -32,430 | +4,872 | 36,236 | +0.13 | 0.13 | 28% |
| deeplob h=30 tau=0.1 (chosen) | -17,180 | +1,638 | 18,781 | +0.09 | 0.09 | 8% |
| deeplob, tau=0.3, all three | 0 | 0 | 0 | | | 0% |
| lightgbm, all six | 0 | 0 | 0 | | | 0% |
| logit-imbalance, all six | 0 | 0 | 0 | | | 0% |

Breakeven fee is gross over traded weight, gross / (gross - net). The final
LightGBM fit early-stopped to a model whose edge never reaches 0.1 on the
window, so it and the logistic sit out; the network's gross edge on the two
new days is +0.13 bps a trade at 5 and 10 s, inside the walk-forward's +0.15,
and it would need a fee under 0.13 bps a side to keep it. Macro F1 on the
window: deeplob 0.323 / 0.317 / 0.315 at 5 / 10 / 30 s against 0.287 / 0.275 /
0.261 for both baselines, which is exactly the always-flat score at those
majority shares (`reports/f1.png`).

`reports/equity.png` is the stitched test days and the window for the three
models at the baseline variant and the selected stream; every line that
trades goes down at about the fee times the trade count.

## What this shows

- The network is the only model whose argmax ever leaves the flat class, and
  its gross edge is real and small: +0.13 to +0.15 bps a trade, on both the
  walk-forward days and the two days nothing was fit on.
- That edge is an order of magnitude under a 1 bp fee and two orders under
  the retail one. At these horizons the tape is a market maker's signal, not
  a taker's, which is the `orderbook-imbalance/` conclusion again from a
  different model on a different venue.
- The walk-forward selection had nothing to select between: every variant
  is negative on every validation slice, and the pick is whichever loses
  least. The DSR of the headline is 0 because the headline is a large,
  precise loss.
- Training loss fell every epoch (1.78 to 1.64) while validation loss did not
  (2.47 to 2.46, best at epoch 4): with 8 features a second and no book, the
  network has little left to learn after the first pass.

## Limitations

- **No book.** Trade prints reconstruct the touch and nothing below it. The
  DeepLOB result in the paper comes from ten levels of queue sizes, and the
  `orderbook-imbalance/` project's depth-1 t-stat of 12.7 came from real
  quotes. When `crypto_l2` holds six hours, `load()` switches on its own and
  every number here should be redone.
- **Seven days, one week.** Two of them a weekend: the 5-second flat share is
  94% on Saturday 08-29 and 73 to 78% on the weekdays. No regime change, no
  news day.
- **The band was set blind** and landed wide: 73 to 82% of seconds are flat.
  A narrower band would give the network more to predict and the tables more
  trades; changing it now is a trial.
- **The window is whole days**, 2/7 rather than the harness's last fifth,
  because `Untouched.lock` records its start as a date. The lock, the data
  and the reports agree at that grain; a seconds-grain lock would be a
  change to `research/`.
- **The trading rule re-decides every second.** A hold-for-horizon rule or a
  position buffer would cut the trade count; either is a new variant.
- **Sharpe is per second scaled by sqrt(86,400)**: a daily-unit number on
  serially correlated one-second returns, reported for ranking inside this
  project, not for comparison with the book's daily Sharpes. The registry
  drops zero-return seconds before its Sharpe, the book's convention for days
  before a sleeve is live.
- **lightgbm and torch share an OpenMP runtime badly on macOS**: torch first
  segfaults LightGBM, LightGBM first deadlocks torch's CPU pool. `lob.py`
  imports lightgbm first and pins torch to one CPU thread; MPS is unaffected.

## Files

- `data.py` archive download and one-second bars from trades; the lake path; `load()`
- `lob.py` labels, the three models, the purged folds, `pnl` with the one-second fill lag, summaries
- `check.py` 33 offline checks on planted data: bars from trades and from L2 rows, labels and the band, the fill lag and the fee, the purge, three models recovering a planted signal, shuffled labels killing it, the window mask
- `run.py` the walk-forward, the window, the registries, charts to `reports/`
- `reports/summary.json`, `walk_forward.csv`, `walk_forward_variants.csv`, `classification.csv`, `untouched_variants.csv`, `returns_1min.csv`, `untouched.json` (the lock, spent), `registry/runs.jsonl`, `models/`, `equity.png`, `f1.png`, `run_output.txt`

## How to run

```
../.venv/bin/python3 check.py     # offline, about a minute on the CPU
../.venv/bin/python3 run.py       # about 7 minutes on Apple MPS; downloads 100 MB of aggTrades on the first run
```

A rerun reproduces every walk-forward number (seeds are fixed) and reads the
window's result back from the lock.
