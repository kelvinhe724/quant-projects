# XS-TREND, prior layer

The backtest that decides whether `PREREG.md` earns a forward track. Built 2026-10-07 against the signed pre-registration; nothing in it was chosen after a number was seen. The untouched window is locked and **has not been opened**. Every number below is printed by `prior.py` from the code in this directory.

## What was run

50/200 moving-average crossover, long only, on the point-in-time S&P 500. At each month end t: members as of t (pit-universe's panel), eligible by momentum's rules (252 prints, trailing 63-day median dollar volume at least $10m, window ending t-1), the top 100 of those by that dollar volume; long 1/N over the names whose 50-day average is above their 200-day average at the close of t-1, cash if fewer than 8 are (momentum's `long_only_weights`, imported); fills at t+1's open. Costs: 10 bps one-way plus the book's `CostModel` (5 bps commission, 2 bps half spread, sqrt impact at $100k capital), zero borrow. Scored path: the unlevered weights scaled to a 15% vol target on the sleeve's trailing 63-day realised vol, gross capped at 1, no leverage. Signal and eligibility come through `research.features.FeatureStore` at lag 1 and the peek audit passes; the data is the lake's `equities_daily` (Yahoo adjusted close for signal and returns, raw close x volume for liquidity), 2003-06-02 to 2026-08-31, 692 names, close hash `5c1c077e83e5a9d1`.

Two other configurations ran on the same engine, both logged: the published 190-name list (185 in the lake, all eligible names, no top-100 cut) as the reference, and an equal-weight monthly-rebalanced basket of the same PIT top-100 universe at the same vol target as the buy-and-hold comparator the window must beat.

## In-sample, 2005-01-03 to 2020-12-31

| | PIT N=100 | reference 190 | EW basket (B&H) |
|---|---|---|---|
| net Sharpe, scored (15% target) | **0.61** | 0.77 | 0.63 |
| net Sharpe, unlevered | 0.58 | 0.76 | 0.60 |
| annual return, scored / unlevered | 8.6% / 9.6% | 11.1% / 12.8% | 9.0% / 11.3% |
| annual vol, scored / unlevered | 15.5% / 18.8% | 15.1% / 18.1% | 15.6% / 22.0% |
| max drawdown, scored / unlevered | -31% / -48% | -36% / -43% | -33% / -57% |
| one-way turnover per year, scored | 3.5x | 2.8x | 1.4x |
| average names held | 68 | 123 | 100 |
| average gross, scored | 0.87 | 0.89 | 0.86 |
| cost drag per year, scored | 0.59% | 0.47% | 0.24% |
| beta to SPY, scored | 0.66 | | |

**PIT minus reference: Sharpe -0.16, annual return -2.5%/yr scored (-3.2%/yr unlevered).** PIT minus the EW basket: Sharpe **-0.02**. The reference's unlevered 0.76 reproduces the published in-sample 0.76 on a different data source, with monthly instead of daily rebalancing and heavier costs, which is the cross-check that this engine is the same strategy.

Per year, scored net Sharpe (PIT / reference / basket): 2005 0.96/1.34/1.04 · 2006 0.83/1.44/1.11 · 2007 1.06/1.02/0.73 · 2008 -1.29/-1.04/-0.86 · 2009 1.49/1.17/1.63 · 2010 0.62/1.07/0.73 · 2011 0.13/0.33/0.02 · 2012 0.66/0.87/0.80 · 2013 2.80/2.84/2.73 · 2014 0.91/1.44/1.00 · 2015 0.06/0.21/-0.07 · 2016 0.63/1.32/0.85 · 2017 2.57/3.16/2.85 · 2018 -0.04/-0.22/-0.14 · 2019 1.67/2.32/1.86 · 2020 0.12/0.04/0.02. Full rows in `reports/prior_insample.csv`.

**Walk-forward** (`research.alpha.base.walk_forward`, 4 folds x 504 sessions, expanding train from 2005, 21-session purge; one variant, so the folds are a stability check, not a selection): test Sharpe 1.94 (2013-14), 0.38 (2015-16), 0.61 (2017-18), 1.00 (2019-20) on the base engine (close fills, 17 bps flat, unlevered); the scored path on the same windows 1.94, 0.31, 0.72, 0.54. Stitched OOS 0.91. All positive, range 0.3 to 1.9: that is the period, there is no parameter to be unstable.

Registry: `reports/registry/runs.jsonl`, 8 trials (walk-forward grid and OOS stream, three configurations x two paths). In-sample scored PIT PSR 0.99, DSR 0.96 over those 8, which is an in-sample number and means little. The registry's Sharpes (0.617 for the scored PIT path) are on nonzero days, the research convention; the table's are on all days.

## How to open the window

Locked before any run in `reports/untouched.json`: **2021-01-04 to 2026-08-31, 1,421 sessions, data hash `5c1c077e83e5a9d1`, `opened_at: null`.** `check.py` fails if it is opened. The default run of `prior.py` reads nothing dated 2021 or later except the hash. To open it, once:

```
cd quant-projects/xs-trend && ../.venv/bin/python3 prior.py --open
```

That runs the scored PIT sleeve and the scored EW basket over the window, logs both, stores the result in the lock file and prints `clears`: true only if the sleeve's net Sharpe is above zero **and** above the basket's. A second `--open` raises `UntouchedWindowUsed` and reprints the stored result. Clears goes to the gate layer (6-month paper sleeve); fails goes to `REFUSALS.md §1`. The retire line of `PREREG.md §Kills`, "loses more than half the published Sharpe", is 0.38 against the published in-sample 0.76; in-sample the PIT run is at 0.61.

## The honest reading

The trend rule is beta. On the point-in-time universe the sleeve's in-sample Sharpe (0.61) is the same as holding the whole top-100 basket at the same vol (0.63), its beta to SPY is 0.66, and the only thing the crossover buys is a shallower unlevered drawdown (-48% against -57%), which the vol target then mostly delivers on its own (-31% against -33%). Had the window's test been applied in-sample, the second condition would have failed by 0.02. The author's prior in `PREREG.md`, "beta ~0.6 to SPY, so the attribution test is the one it fails", is what the in-sample data says too.

The survivorship the pre-registration set out to measure is there and has the sign and size pit-universe predicted: the published list beats the point-in-time universe by 2.5 to 3.2%/yr and 0.16 of Sharpe, even though both are long-only large caps on the same engine. Part of that gap is breadth (123 names against 68), not only membership. And the point-in-time number is still too kind: 214 names that were in the index have no lake price, and in 2007-2009 they include the large, liquid financials that would have been in the top 100 by dollar volume and were in an uptrend until they were not.

So the thing the window has to show is not whether 50/200 long-only makes money (it does, because large caps went up), but whether it did anything an equal-weight basket of the same names did not. In-sample it did not.

## Reviewer notes

- **Working tree.** This session could not write to the main checkout, so the files are in the linked worktree `.claude/worktrees/xs-trend-prior` on branch `xs-trend-prior`, uncommitted: `xs-trend/{prior.py, check.py, README.md, reports/}`. Nothing outside `xs-trend/` was changed; `data-lake/store` in that worktree is a symlink to the real store so the lake loads there, and is not part of the deliverable. `prior.py` imports from `research/`, `momentum/` (by path, as pit-universe does), `framework/engine/`, `framework/book/strategies.py` (`book_config().costs`, `CAPITAL`) and `data-lake/lake.py`; nothing was copied.
- **Window end is 2026-08-31, not "latest".** The membership panel ends there; the lake runs to 2026-10-06. A point-in-time universe past the panel's end would be a guess, so the panel was cut there and the lock covers that. Extending needs `pit-universe/run.py` rerun first, and that changes the hash, so decide before opening.
- **The lock hash is over Yahoo's adjusted close as fetched 2026-09-05.** The nightly collector only rewrites rows from the last week, so the hash is stable; `collect.py equities --full` would restate every adjusted close and the lock would (correctly) refuse to match.
- **Coverage.** Median 373 PIT members with a lake price in-sample, of about 500; the top 100 by dollar volume are better covered than that but the 2007-2009 deletions are the gap that matters. 21 stock-days were held past a name's last print (scored at zero) and 2 fills had no open.
- **What the pre-registration under-specified, and what was assumed.** (1) "Weight 1/N": read as 1/(names in an uptrend), the published rule, imported and checked for parity, not 1/100 per name with the rest in cash; the alternative is a different, breadth-scaled strategy and was not run. (2) The published 0.76 / 1.02 was rebalanced daily; this runs monthly as the pre-registration says, and the monthly reference reproduces 0.76 anyway. (3) The reference holds every eligible name of the 190 list, so the PIT gap bundles membership and breadth; the membership-only number is pit-universe's +4.0%/yr on equal weight. (4) "Buy-and-hold of the same PIT universe" is a monthly-rebalanced equal-weight basket of the top 100, because the universe changes monthly. (5) The 20% drawdown kill has no written re-entry rule, so the scored path does not apply it, following `book_config(kill=False)`'s convention for long backtests; the line was crossed 10 times in-sample (2008-09-29, 2008-10-02, 2009-07-29, and seven times in March to June 2020) and `--open` reports the window's crossings. Applying any re-entry rule now would be a choice made after the number. (6) Capital for the impact term is the book's $100k; impact is under 1 bp a trade, the walk-forward uses the 17 bps the scalar engine can take and the scored path adds impact. (7) Cash earns nothing, in both the sleeve and the basket. (8) The vol window ends the session before the rebalance; with no estimate there is no position.
- **Checks.** `check.py`, 22 planted-panel checks, exit 1 on any failure: audit live and passing, signal flips one session after a planted crossover and equals momentum's `ma_signal` shifted by one everywhere, PIT filter excludes a name the session before it joins, top-N and the dollar-volume floor, long only, weights sum to at most 1, parity with `long_only_weights`, exact cost on a planted trade under both the flat and the book model, open-to-close on the fill day, vol target halves a 30%-vol sleeve and never levers a 10% one, the lock exists unopened.

```
../.venv/bin/python3 prior.py     # about 10 seconds; deterministic, reruns keep the lock and the 8 trials
../.venv/bin/python3 check.py
```
