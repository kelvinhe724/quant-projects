# Order-book imbalance and the next few seconds of BTC-USDT

Does the shape of the limit order book say where the mid-price goes next? On
50 minutes of OKX BTC-USDT snapshots the answer is yes, at one second, with a
t-statistic of 12, and it is worth about a tenth of a basis point per signal.
The taker fee is ten basis points. So the signal is real and the trade is not.

- **Signal.** Imbalance = (bid size - ask size) / (bid size + ask size) over the
  top 1, 5 or 20 levels, in [-1, 1].
- **Target.** Mid-price move from snapshot t to the first snapshot at least 1,
  2, 5, 10 or 30 seconds later, in bps.
- **Regression.** OLS of the forward move on the imbalance, Newey-West errors
  with as many lags as the horizon has snapshots, since the targets overlap.
- **Strategy.** Read the signal at t, fill at t+1 crossing the spread, unwind
  after the horizon, never hold two positions at once. Three P&L lines: mid to
  mid (gross), less one half-spread (aggressive in, passive out), less the full
  spread and OKX's 10 bps taker fee each way.

## Data

`data.py` polls OKX's public REST book endpoint once a second for 20 levels a
side and the recent-trades endpoint every other second, and writes both to
`../source-material/orderbook/` as they arrive. Binance's depth endpoint
returns HTTP 451 from here, which is why OKX.

One session: 2983 snapshots, 2026-09-03 19:40 to 20:30 UTC, median interval
1.00 s and a worst gap of 1.2 s. Mid went 81,583 to 81,542 inside a
81,387 to 81,796 range. The spread was one tick (0.1 USDT, 0.012 bps) in
2981 of 2983 snapshots, so the half-spread is 0.006 bps and the interesting
cost is the fee, not the spread. The mid changed between consecutive snapshots
33% of the time.

The trade tape has holes. The trades endpoint was polled with `limit=100` and
BTC-USDT prints more than 100 trades in a busy two seconds, so 4,078 of about
21,160 trade ids (19%) in the window are missing. That only touches the two
robustness columns that use trades (last-trade price as target, signed flow as
a competing regressor); the snapshots themselves are complete. The collector
now asks for 500, the endpoint's cap.

Local timestamps are the request send time; the exchange's book timestamp
runs 140 to 600 ms later. Horizons and the t+1 fill both use local time, so
a "1 s" horizon is one poll, not exactly 1000 ms of exchange time.

## Design

The window is split in half by time. The first half is formation: the
strategy grid (3 depths x 3 thresholds x 3 horizons, 27 cells) is run there
and the cell with the best total net-of-half-spread P&L, among cells with at
least 20 trades, is picked. The second half is evaluation and is reported
once for that cell. Alongside it runs a configuration written down before any
result was looked at: depth 5, |imbalance| > 0.5, 5 s hold. The regressions
are also shown on each half separately.

**Look-ahead.** Nothing in `imbalance.py` shifts. The forward move at t is
built from the first snapshot at least the horizon after t, and the strategy's
one-snapshot lag lives in one line of `backtest`. `check.py` plants a known
slope in a synthetic book, recovers it, and checks that a signal read one
snapshot late scores higher than the honest one, so the lag is doing something.
Forward moves for the split regressions are built inside each half, so a
formation row near the boundary never reads an evaluation price.

## Results

All numbers from `run.py`; tables in `reports/`.

### Does the book predict the mid?

Forward mid move in bps regressed on imbalance, full window (n = 2982 at 1 s).

| Depth | Horizon | Slope, bps | t | R² | Hit rate |
|---|---|---|---|---|---|
| 1 | 1 s | 0.33 | 12.7 | 0.063 | 73% |
| 1 | 5 s | 0.60 | 6.6 | 0.037 | 62% |
| 1 | 30 s | 1.07 | 2.9 | 0.018 | 53% |
| 5 | 1 s | 0.35 | 11.8 | 0.056 | 71% |
| 5 | 5 s | 0.61 | 5.6 | 0.030 | 61% |
| 5 | 30 s | 1.14 | 2.6 | 0.016 | 51% |
| 20 | 1 s | 0.38 | 8.0 | 0.031 | 66% |
| 20 | 5 s | 0.68 | 3.7 | 0.017 | 57% |
| 20 | 30 s | 1.67 | 2.1 | 0.016 | 51% |

Hit rate is the share of nonzero moves whose sign matches the imbalance. The
slope grows with the horizon while the t-stat and R² shrink: the information
in the book is about the next second or two, and after that it is drowned by
everything else that moves the price. Depth 1 is the sharpest signal; depth 20
carries the same sign with a third of the R². The binned plot
(`reports/binned.png`) is monotone at 1 s: the bottom decile of depth-5
imbalance is followed by -0.76 bps over 5 s and the top decile by +0.51 bps.

Both halves agree. Depth 5 at 1 s: slope 0.39 (t 8.1) on the formation half,
0.31 (t 9.8) on the evaluation half. At 30 s the formation slope is 1.41
(t 2.7) and the evaluation slope 0.99 (t 1.5), which is the horizon at which
the effect stops being reliably measurable on 25 minutes of data.

Two robustness checks, depth 5. Against the last traded price instead of the
mid the 1 s slope is 0.42 (t 13.0), so this is not a quirk of how the mid is
defined. With 5 s of signed trade flow as a second regressor the imbalance
slope is unchanged at 0.35 (t 11.8) and the flow coefficient is zero (t -0.2),
so the book is not proxying for recent trades. Both use the holed trade tape.

### Can you trade it?

Per-trade bps. `net_half` is gross less one half-spread; `net_full` is gross
less both half-spreads and two 10 bps taker fees.

| | Window | Trades | Gross | Net half | Net full | Hit rate | t |
|---|---|---|---|---|---|---|---|
| Fixed in advance (d5, 0.5, 5 s) | formation | 196 | -0.02 | -0.03 | -20.03 | 54% | -0.1 |
| Fixed in advance (d5, 0.5, 5 s) | evaluation | 197 | +0.17 | +0.16 | -19.84 | 60% | 1.5 |
| Chosen on formation (d1, 0.5, 1 s) | formation | 488 | +0.12 | +0.12 | -19.89 | 64% | 2.8 |
| Chosen on formation (d1, 0.5, 1 s) | evaluation | 461 | +0.11 | +0.11 | -19.90 | 73% | 4.3 |

Hit rate is among trades where the mid moved; for the chosen cell the mid did
not move at all on 75% of evaluation trades.

The chosen cell holds up out of sample: +0.12 bps per trade on formation and
+0.11 on evaluation with a t of 4.3, and its gross move is 19 times the
half-spread. The pre-registered cell is noisier, negative on formation and
+0.17 on evaluation, which on 200 trades is not distinguishable from either.
Neither survives fees: the expected move is 0.1 to 0.2 bps and the round trip
costs 20. The only way to earn this signal is to already be quoting, where it
tells a market maker which side of the book to lean on rather than whether to
cross the spread.

## Files

- `data.py` collector and loader; `python3 data.py 45` polls for 45 minutes
- `imbalance.py` imbalance, forward moves, HAC regression, binning, the backtest
- `check.py` 38 offline checks on a synthetic book with a planted slope: book
  arithmetic, slope recovery, shuffled-signal null, the t+1 lag, spread and fee
  accounting
- `run.py` full pipeline, tables and charts to `reports/`
- `walk.py` five-snapshot walkthrough you can check by hand

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 data.py 45      # needs internet, ~45 minutes
../.venv/bin/python3 run.py
```

## Limitations

**Fifty minutes, one afternoon.** The slopes are precisely estimated inside the
window and say nothing about other regimes, other hours, or a day when the
spread is not one tick.

**One-second polling is not the book.** The book updates thousands of times a
second; a REST poll sees one state per second and the fill at t+1 assumes the
touch is still there a second later. A websocket feed and a queue-position
model would be the next step, and my expectation is that the
per-signal edge is smaller at the speed anyone actually trades this.

**The trade tape is 19% short**, as above. It affects only the two robustness
columns.

**Newey-West on overlapping targets, naive standard errors on the bins.** The
regression t-stats account for the overlap; the error bars on the binned plot
do not and are too narrow at the longer horizons.

**Fees are the retail tier.** 10 bps taker is OKX's entry level; VIP tiers and
maker rebates change the arithmetic, though not by the two orders of magnitude
needed here.
