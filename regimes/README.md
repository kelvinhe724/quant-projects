# Regime overlay on the premia book

Does knowing the market's regime help the live book? A Gaussian HMM on SPY gives a filtered probability of being in the high-vol state each day, a macro nowcast says whether the economy is sagging, and the two become one number between 0 and 1 that scales the book's gross exposure. The book is `framework/`'s live `beta+alpha` (vol-targeted 1/N of 14 ETFs plus EWMAC, 1/2 each); the overlay multiplies each sleeve's 10% vol target by the scale and everything else about the engine, costs and risk overlay is unchanged.

The one-line honest summary: the overlay does not help. On the untouched last fifth of the panel the bare book scores net Sharpe 0.79 (DSR 0.87 over 65 trials) and the book with the chosen overlay scores 0.53 (DSR 0.71), with the drawdown shallower (-8.8% against -11.7%) and 22% more trades. The walk-forward picked the 3-state HMM without the macro leg in three of four folds and its stitched out-of-sample Sharpe was 0.50 against 0.59 for leaving the book alone. The overlay is not adopted. The regimes themselves are real and persistent; the problem is that an equity-vol regime is the wrong thing to scale a book whose alpha sleeve is long volatility.

## Data

- SPY adjusted close from the data lake (`lake.load("equities_daily", ..., universe=["SPY"], as_of=...)`), 2003-06-02 to 2026-08-31, 5,829 observations after the 21-day vol warm-up. The `as_of` is the book panel's last session so nothing later than the book can see reaches the model.
- FRED from the lake: `T10Y3M` (10y minus 3m, daily), `PAYEMS` (nonfarm payrolls, monthly), `VIXCLS`.
- The book panel through `framework.book.universe.load_bars()`, 5,850 sessions 2003-06-02 to 2026-08-31, close hash `aa436bbc6c2e4712`.

The spec asked for a credit-spread leg. The lake holds `BAMLH0A0HYM2` (HY OAS) only from 2023-09-05, which is what FRED's CSV endpoint returns for that series, and FRED timed out twice on 2026-09-04 when I tried to pull more, so there is no credit series long enough to use. I put the VIX in its place as a risk-appetite proxy. That is a substitution, not the thing asked for, and it overlaps with the realised vol the HMM already sees.

## Method

**HMM.** `hmmlearn.GaussianHMM`, full covariance, two observations a day: the log return in percent and the log of the trailing 21-day realised vol. K in {2, 3, 4}. After each fit the states are sorted by return variance so state 0 always means calm. Refit every January on every session before it, first fit at the start of 2006 on 2.5 years, so 21 refits. `regimes.filtered` is the forward recursion only, P(state | data through t); `predict_proba` is smoothed and reads the future. The probability for day t comes from the model fit through the previous December and the observations through t. Before 2006 the scale is 1.

**Nowcast.** Curve slope, the monthly payroll change moved 40 calendar days forward so a print dated the first of the month is not visible until after its release, and minus the VIX, each z-scored against a trailing ten-year window (minimum 500 sessions), averaged, then EWMA-smoothed with a 10-session half-life. There is no consensus series in the lake, so "surprise" is the change against its own trailing distribution, and FRED keeps no vintages so the payroll numbers are today's revised values, not what was printed.

**Scale.** `1 - P(highest-vol state)`, times `min(1, exp(nowcast))` when the macro leg is on, floored at 0.05 because `RiskManager` reads a zero vol target as "no vol target" and would send the raw weights. Six variants: K2, K3, K4, each with and without macro.

**Overlay.** `regimes.RegimeOverlay` wraps a sleeve and, inside `on_bar`, sets the shared `RiskConfig.target_vol` to 10% times the day's scale before the sleeve answers. The engine builds each sleeve's `RiskManager` around that same config object and reads `target_vol` on every apply and drift, so this is the one place the scale enters, and the buffer, drawdown cut, gross cap and cost model all still apply on top. `check.py` shows the overlay at scale 1 reproduces the bare sleeve's returns exactly and that a planted scale of 0.5 halves the engine's gross from the day after it changes.

**Selection.** Six scales times the bare book's daily net returns (scale at t applied to the return of t+1) are the grid; the bare book is the incumbent. `framework.book.validate.walk_forward`, four folds of 756 sessions on the sessions before the untouched window, picks the best training-window Sharpe per fold. The most-picked overlay variant is run through the engine on both sleeves and compared with the bare book on the untouched window, opened once through `research.alpha.Untouched` and stored in `reports/untouched.json`. The rule, fixed before the run: adopt only if the walk-forward picks the overlay more often than the bare book and it beats the bare book on the untouched window. The grid runs on scaled returns rather than six engine runs; only the chosen variant pays the engine's costs. That is a shortcut and the engine number below is the one that counts.

Every look is logged to `reports/registry/runs.jsonl` (11 trials here) and the DSR adds the book's own 56 trials from `framework/book/reports/trials.csv`, since the overlay sits on a book that was itself selected.

## Regimes

K=3, the walk-forward's choice. Last refit (January 2026, trained on 2003-07 to 2025-12):

| state | mean/yr | vol/yr | p(stay) | implied duration | empirical duration, pre-window | share of days |
|---|---|---|---|---|---|---|
| 0 calm | +16.6% | 9.1% | 0.979 | 48 sessions | 12.6 | 47% |
| 1 middle | +12.7% | 14.7% | 0.962 | 26 | 6.9 | 30% |
| 2 high-vol | -5.8% | 33.4% | 0.980 | 50 | 5.9 | 22% |

Transition matrix, last refit (rows from, columns to):

| | calm | middle | high-vol |
|---|---|---|---|
| calm | 0.979 | 0.020 | 0.001 |
| middle | 0.027 | 0.962 | 0.011 |
| high-vol | 0.000 | 0.020 | 0.980 |

The chain moves calm to middle to high and back; a jump from calm straight to high-vol has probability 0.001 in this fit. Averaged over the 21 refits the diagonal is 0.92 / 0.84 / 0.87, because the early fits on two to five years of data are far less persistent, which is why the scale chatters in 2006-2008 in `reports/regimes.png`. The empirical durations of the filtered argmax path (13, 7, 6 sessions) are much shorter than the model-implied ones (48, 26, 50) for the same reason: the path is a sequence of different models, and the filter flips on days the model is unsure. Share of days with P(high-vol) above 0.5: 39% in 2008, 71% in 2009, 64% in 2020, 79% in 2022, zero in 2013, 2014, 2017, 2021 and 2023.

## Walk-forward

Pre-window sessions, proxy returns, best training Sharpe held on the test fold:

| fold | test | picked | train Sharpe | test Sharpe | bare book, same fold |
|---|---|---|---|---|---|
| 1 | 2009-12 to 2012-12 | K2 | 1.49 | -0.02 | 0.67 |
| 2 | 2012-12 to 2015-12 | K3 | 1.06 | 0.56 | 0.45 |
| 3 | 2015-12 to 2018-12 | K3 | 0.95 | 0.62 | 0.48 |
| 4 | 2019-01 to 2021-12 | K3 | 0.89 | 0.85 | 0.74 |

Stitched selected-in-sample Sharpe 0.50 against 0.59 for the bare book over the same test windows; fold 1's train Sharpe of 1.49 is the 2008 look, and it bought a test Sharpe of -0.02. Full pre-window proxy Sharpes: none 0.82, K2 0.75, K2+macro 0.63, K3 0.88, K3+macro 0.77, K4 0.79, K4+macro 0.77. The macro leg loses in every pair. Average scale over the pre-window: K3 0.81, K2 0.70, K3+macro 0.72.

## Results

Engine, book cost model and overlay, kill off, both sleeves:

| | Sharpe net | ann. return | ann. vol | max drawdown | DSR | trials |
|---|---|---|---|---|---|---|
| pre-window 2003-06 to 2021-12, bare book | 0.815 | +5.8% | 7.3% | -12.1% | 0.99 | 67 |
| pre-window, book + K3 overlay | 0.700 | +4.4% | 6.4% | -12.6% | 0.96 | 67 |
| **untouched 2021-12-31 to 2026-08-31, bare book** | **0.789** | +5.2% | 6.7% | -11.7% | **0.865** | 65 |
| **untouched, book + K3 overlay** | **0.526** | +3.7% | 7.4% | -8.8% | **0.710** | 65 |
| full panel, bare book | 0.810 | +5.7% | 7.2% | -12.1% | | |
| full panel, book + K3 overlay | 0.659 | +4.2% | 6.6% | -12.6% | | |

The trial count is the 9 looks logged here before the window was opened plus the book's 56; with only this project's 9 the window DSRs are 0.90 and 0.78. The window was opened 2026-09-04 20:45 and the pre-window rows show the count after the two window reads were logged. Full-panel trades 30,820 against 42,548 and turnover 9.7x against 23.2x a year: a scale that moves every day moves the vol target every day, and the 10% buffer does not absorb a target that halves.

**Does the overlay change the DSR?** Yes, downward: 0.865 to 0.710 on the window, at the same trial count. The deflation is the same for both, so the gap is the Sharpe.

By year, the two years that decide it: 2007 the bare book made +20.3% and the overlay -0.3% (average scale 0.68, the HMM read the summer's vol as a bad regime while the trend sleeve was earning), and 2022 the bare book made +9.1% and the overlay +2.5% at an average scale of 0.25, with the monthly average scale below 0.10 in eight of twelve months. 2022 was the year EWMAC's short bonds paid; the overlay saw SPY's vol and stood aside. The overlay won 2013, 2015, 2019 and 2025 by less than it lost those two.

**The drawdown cut.** The overlay's untouched-window vol is higher than the bare book's (7.4% against 6.7%) even though the scale never exceeds 1. The engine halves a sleeve past a 15% drawdown until it is back inside 7.5%, and the paths diverge: on the window the bare book's EWMAC sleeve ran at half size on 74.5% of days (its 2022-2026 drawdown reached -19.4%) while the overlay's EWMAC, having sat out 2022, never triggered the cut; the overlay's ETFBeta, scaled to 5% for most of 2022, could not climb out of its earlier drawdown and ran at half size 66.6% of days against the bare book's 32.6%. Average gross on the window 0.84 against 1.02. A good part of what the overlay did on this window is not regime timing but which sleeves the drawdown cut happened to be holding down.

## What I would do differently

- Scale the sleeve that is exposed to the regime, not the book. ETFBeta is long everything and the equity-vol state is the right signal for it; EWMAC is a trend rule that earns in exactly the regimes the HMM flags. A per-sleeve scale, or the HMM run on the book's own returns, is the obvious next trial and it is a new trial.
- Run the grid through the engine. The proxy ignores the overlay's own turnover and the drawdown-cut interaction, which turned out to matter more than the signal.
- Get a real credit series and a consensus payroll series with vintages before calling anything a nowcast.

## Limitations

- The untouched window is the book's window (last 20% of sessions, the `validate.HOLDOUT` convention), which `framework/` has read once already for the book decision. This project reads the same dates again for a different decision, through its own lock, so the window is not untouched in the strict sense for anything that touches the book. The overlay's own parameters were never fit on it.
- The HMM is refit each January but the probability inside a year is filtered with that January's model; a regime shift in March waits until the next January to change the parameters, which is the point of the persistence and also why 2006-2008 chatters.
- Payroll revisions: the lake stores the current vintage. The 2020 print is -20.5 million in today's data and its z-score of -6.4 dominates the nowcast's distribution for years afterwards.
- SPY's adjusted close in the lake is Yahoo's adjustment as of the last fetch (see the lake README); returns are unaffected except across a dividend date fetched at different times.
- MPS/GPU are not used; there is nothing here to put on one. The whole run is 7 minutes, of which the two engine runs are 6.5.

## Files

- `regimes.py`: observations, fit and state sorting, forward filter, walk-forward refits, nowcast, scale, `RegimeOverlay`, the drawdown-cut replay.
- `check.py`: 23 offline checks on planted regimes and a synthetic engine run; exits 1 on any failure.
- `run.py`: everything above; `reports/summary.json`, `untouched.json` (the lock, spent), `registry/runs.jsonl`, `models/` (the January 2026 K=3 model, `02b84c3629cb`), `states.csv`, `transitions.csv`, `by_year.csv`, `probabilities.csv` (daily P(high-vol) for each K, the nowcast and every scale), `returns.csv` (daily returns, gross and sleeve equity for both runs), `regimes.png`, `overlay.png`.

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py      # about 7 minutes; a rerun reports the window's stored result
```
