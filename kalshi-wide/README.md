# KALSHI-WIDE: prior layer

The replay backtest that `PREREG.md` pre-registered, run once on 2026-10-07.
Verdict: **INSUFFICIENT FILLS** (2 fills, 300 required). The reason is the
data, not the rule: the lake's book snapshots are hourly, so a 60-minute
resting order is almost never observed, and the books it does record are the
200 most traded markets, which are not the wide ones.

## Reviewer notes

- **Built in a linked worktree**, not the main checkout (the harness blocked
  writes to the shared tree): `.claude/worktrees/kalshi-wide/kalshi-wide/`,
  branch `kalshi-wide-replay`, uncommitted. The worktree's `data-lake/store`
  is a symlink to the main store (read only; the path is gitignored). Nothing
  outside `kalshi-wide/` was changed.
- **Candles were not pulled.** The registry's "chosen" model `0ee881a512f0`
  is the `BiasCurve`, which prices from the mid alone; `check.py` asserts
  that random path features change none of its predictions and that the id
  used equals the registry's chosen id. `replay.py` refuses to run if the
  chosen model is ever a feature model, rather than carrying an untested
  candle path.
- **Window.** PREREG says replay from 2026-08-25; the lake's first Kalshi
  snapshot is 2026-09-05 01:07 UTC, so that is the start.
- **Cadence.** PREREG assumed 15-minute book snapshots. The lake's median gap
  between snapshots is 60.2 minutes; 2 of 247 gaps are inside a 60-minute
  window, both in the collector's first hour on 2026-09-05. The replay
  implements the 60-minute cancel exactly. `replay.py --window N` reruns with
  a longer window, logs it as a second trial tagged `second_look`, and was
  **not run**; whether that is a legitimate second look (collector drift of
  ten seconds past the hour) or a rule change is the reviewer's call under
  `REFUSALS.md §4`. The coverage diagnostic below says what it could find.
- **Assumptions the pre-reg left open** (each is a constant or one line in
  `replay.py`):
  1. Resting price rounded **down** to the cent (pays less, fills less).
  2. An order whose price is at or through the far touch at placement would
     take, not make; it is dropped. 0 of 343,834 candidates were dropped.
  3. "Cancel on a 3c move" is read as the PREREG wrote it: |mid now - mid at
     placement| >= 0.03 in either direction, on the listing mid.
  4. In one snapshot, a crossing touch counts as a fill before the cancel test.
  5. A fill costs the resting price (a maker gets its own price), stake =
     price + maker fee, payoff per $ = P&L / stake, t = mean / se of per-bet
     payoff (`prediction-markets/kelly.py::per_bet_edge`), daily Sharpe =
     sqrt(365) x mean / sd of daily P&L per $ staked, by fill date.
  6. Maker fee from the July 2026 schedule: round up(M x 0.0175 x P x (1-P))
     to the centicent on the 159 series whose `/series` `fee_type` names
     maker fees, zero elsewhere. 46,156 candidate rows sit on maker-fee series
     (Financials); neither fill did.
  7. Category and fee type come from `GET /series` (cached
     `reports/series.csv`); the lake has no category column. 3,176 rows on
     categories the model never saw were dropped.
  8. Candidates are markets closed by run time (132,463 still-open rows
     dropped), so every fill can settle. Results come from
     `GET /markets/{ticker}` for filled tickers only (`reports/results.csv`).
  9. The lake lists the first 5,000 open markets per snapshot (five API
     pages), so the candidate universe is that listing, not every open market.
  10. "24h to close" is the listing's `close_time`. Both fills settled within
      two hours of placement: sports markets close early when the game ends,
      so the 24h floor is nominal for 94% of candidates.

## What was run

`replay.py`, one configuration, logged once to `reports/registry/runs.jsonl`
(id `388960b21abf10e7`, 1 trial).

| | |
|---|---|
| rule | rest at P - 0.02 on the side the BiasCurve favours (YES at P - 0.02 if P > mid, NO at (1 - P) - 0.02 if P < mid), 1 contract, cancel at 60 min or on a 3c move of the mid |
| universe | `kalshi_markets` rows: binary, active, two-sided, spread >= 0.10, >= 24h to `close_time`, trained category, closed by 2026-10-07 |
| model | `kalshi-model` registry "chosen" = `0ee881a512f0` BiasCurve, no refit |
| fill | a later `kalshi_books` snapshot inside the window whose touch is at or through the resting price with >= 1 contract behind it (`kalshi-desk/shadow.py::fill`); recorded touches only |
| cost | maker fee per the July 2026 schedule (0 on these series); settle at the API `result` |
| data | 248 snapshots 2026-09-05 01:07 to 2026-10-07 09:58 UTC, 1,240,000 listing rows, 156 two-sided books a snapshot (median) |

Funnel: 1,240,000 rows -> 969,659 binary/active/two-sided -> 513,087 spread
>= 10c -> 479,473 >= 24h to close -> 476,297 trained category -> 343,834
closed by now = candidates. None dropped for P on the mid, price under 1c, or
a marketable price.

## The numbers

| | |
|---|---|
| candidates | 343,834 (79,467 markets, 1,073 series, 229 snapshots on 18 days) |
| candidates with any snapshot inside their 60-min window | 3,214 (0.9%) |
| observable (a recorded two-sided book inside the window) | 15 |
| fills | **2** (both NO, both Sports, both >20c, both at 01:07 on 2026-09-05, filled at the 01:21 snapshot) |
| fill rate | 0.0006% of candidates; 13.3% of the 15 observable |
| ends | 343,511 expired, 321 cancelled on a 3c move, 2 filled |
| net P&L | +$0.48 on $1.52 staked = +0.316 per $; mean payoff +0.318, t = 6.11 on n = 2 |
| daily Sharpe | not defined (one day) |
| mean Kelly fraction of the fills | 0.105 (reported, not applied) |

By spread bucket: 10-20c 49,755 candidates, 9 observable, 0 fills; >20c
294,079 candidates, 6 observable, 2 fills. By category: Sports 324,125
candidates and both fills; Financials 10,479, Climate and Weather 2,780,
Mentions 2,269, Commodities 1,657, Entertainment 1,578, Economics 705,
Politics 241, none observable.

A t of 6.11 on two bets is two bets. The fills were resting NO orders at 0.73
and 0.79 on MLB props whose YES bid had jumped to 0.35 and 0.25 fourteen
minutes later (through YES-leg prices of 0.27 and 0.21), and both settled NO.

**PREREG §5: 2 fills < 300. INSUFFICIENT FILLS.** Not cleared, not failed:
untested.

## The fill-rate caveat

Three facts about the lake decide this result before the rule does.

1. **Hourly, not 15-minute.** 60.2 minutes median between snapshots. A
   60-minute order sees its next snapshot about ten seconds after it would
   have been cancelled. 340,620 of 343,834 candidates (99.1%) had no snapshot
   of any kind inside their window.
2. **The books are the liquid 200.** The collector records depth on the 200
   markets with the most 24h volume. Candidates are the wide end: median
   quoted spread 0.70, a quarter wider than 0.87. Only 6,498 candidates
   (1.9%) ever get a later recorded book for their market, at a median lag of
   120 minutes, usually when the game is close and the book has become one of
   the liquid 200.
3. **The rule is a longshot fade in near-empty books.** 94% of candidates are
   NO orders (the BiasCurve sits below the mid for mids under 0.6), median
   |P - mid| 4.7c.

Coverage diagnostic (no P&L, no registry row): at the first recorded book
after placement, at any lag, the touch is at or through the resting price for
2,178 of those 6,498 candidates; 497 of them within 75 minutes and 542 within
120 minutes. Those are upper bounds before the 3c cancel, and most come from
books that were not wide any more by then.

## Rerun

```
cd kalshi-wide
../.venv/bin/python3 check.py      # 29 planted checks, offline, about 10 s
../.venv/bin/python3 replay.py     # about 10 s; offline once reports/series.csv and reports/results.csv exist
```

Outputs: `reports/replay.csv` (one row per candidate), `reports/summary.json`,
`reports/registry/runs.jsonl`. A rerun replaces the registry row (same key).

## Honest reading

The question the pre-reg asked, whether a resting order inside a wide book
gets filled often enough to earn the model's gap, cannot be answered from this
lake. The fill assumption ("only touches recorded in the book snapshots") was
chosen to be conservative; with hourly snapshots of the liquid 200 it is not
conservative, it is blind. The two fills are an accident of the collector's
first hour.

What would answer it: a collector that snapshots the books of the wide
candidates themselves (the markets this rule would quote, not the top 200 by
volume) at 15 minutes or better, run for the 60-day retire window the pre-reg
already set. That is a data-lake change, not a rule change, and it should
happen before the shadow track starts, because the gate uses the same fill
rule and would be just as blind. The author's prior ("fills are the whole
story; expect under 30%") stands untested; the numbers here say nothing about
it either way.
