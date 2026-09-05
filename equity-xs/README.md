# Cross-sectional equity ML on the point-in-time S&P 500

LightGBM and ridge, trained on the research platform's feature store, scoring every S&P 500 member at each month end on its next-month return relative to the rest of the index. Long the top decile, short the bottom decile, equal weight, against plain 12-1 momentum built from the same features. Everything runs through `../research/` (feature store with the peek audit, walk-forward, experiment registry, model registry, one untouched window) on a pinned copy of the `../data-lake/` panel; nothing here rolls its own loader or backtester.

The one-line honest summary: LightGBM scores a walk-forward OOS Sharpe of 0.29 net over 2011-2021 with a rank IC of 0.029 (t 2.3), then -0.56 net on the 2022-2026 window with IC -0.008 (t -0.4), while 12-1 momentum does the reverse (-0.15 then +0.23). A post-hoc ablation says the walk-forward number came from the three macro series, which are constant within a date and so let the trees memorise regimes. The window has been read three times, and the third read is the one after an audit changed the training rows; the section on the window says exactly what changed and what it did to the number.

## Data and features

Panel from `lake.load("equities_daily")`, pinned to `reports/snapshot/` on the first run and read from there after: 680 tickers that were ever in the index, 2003-06-02 to 2026-08-31, 5,850 sessions, close hash `1d802248c57a41de`. Yahoo's adjusted close is the close; open, high and low are scaled by the same factor. Membership is a lag-0 feature built from `pit-universe/`'s spells, and it is NaN before 2007-11-29, the first list snapshot that carries tickers: before that date the spells are a back-cast from late-2007 membership, so the feature says "unknown" rather than "in" or "out", and no row there is trained on or scored. Prices from 2003 are still loaded so the 12-month features have history when membership starts. Coverage is the number to remember: a median of 433 eligible names per month end from 2007-11 (min 307, max 503) against roughly 500 in the index, because Yahoo does not serve most delisted names. The names that are missing are the ones a short leg wants.

Fourteen features, all through `research.features.FeatureStore.build(audit=True)`:

| set | features | lag |
|---|---|---|
| `PRICE` | `ret_1`, `ret_21`, `mom_3_1`, `mom_6_1`, `mom_12_1`, `vol_21`, `vol_63`, `dollar_vol_21`, `amihud_21`, `hl_range_21` | 1 |
| macro | `VIXCLS`, `T10Y3M`, `DGS10` from the lake's FRED store | 1 |
| membership | `member`, 1.0 inside a spell, NaN before 2007-11-29 | 0 |

Label: the 21-session forward return minus its cross-sectional mean that day. Training rows are eligible members on the calendar's month ends only; a third of the month-end pairs sit 19 or 20 sessions apart, so consecutive labels can share one or two sessions, never a whole month. `XSModel` is handed the calendar's rebalance dates at construction and refuses to fit without them, so a training slice that ends mid-month (folds 3 and 5 do) cannot add its last session as an extra partial cross-section; an earlier version took month ends from the rows it was given and did exactly that. Stock features enter as within-date percentile ranks centred on zero (NaN as 0); the macro series enter LightGBM raw and ridge not at all, since a constant within a date is an intercept to a linear model.

## Models, frozen before any result

- `ridge`: `Ridge(alpha=10)` on the ten ranked stock features.
- `lgbm`: `LGBMRegressor` with 300 trees, learning rate 0.03, 15 leaves, `min_child_samples=200`, 80% row and column subsampling, on the ranks plus the three macro series. No hyperparameter search; these were written down once.
- `ensemble`: the mean of the two models' prediction ranks.
- `mom`: `mom_12_1` unfitted, the comparison.

Each is one `Alpha`; the signal is +1/k on the top decile of eligible scores and -1/k on the bottom, 200% gross, at each month end, held to the next. Grid of four, five walk-forward folds of 504 sessions with the 21-session label purged (the research harness, expanding train from the start of the calendar), untouched window the last fifth of sessions as in `framework/book/validate.py`. The chosen variant is the one the walk-forward picks most often, ties to the ensemble.

The purge is 21 sessions, counted on the calendar. The research harness used to hand purgedcv evaluation times of `date + BDay(21)`, and business days are not sessions, so a holiday inside the horizon left the last training label reaching one to three sessions into the test window. `research/alpha/base.py` now uses the session 21 places later on the calendar itself; `check.py` runs the walk-forward on a calendar with holidays and counts the gap in sessions, and every fold here has exactly 21. Exactly 21 means the last training label closes on the test window's first session, which only matters when the training slice ends on a month end: fold 3's does (2015-11-30), so its entry position comes from a cross-section the model was trained on and shares that one session's return with the label, one session of 504.

Three cost views on the same position path. `research` is the platform's 10 bps flat. `gross` is zero cost. `net` is the book's cost model from `framework/engine`: 5 bps commission, 2 bps half spread, sqrt impact with k 0.1 on the trade's share of trailing 21-day dollar volume at the book's $100k, and 50 bps a year borrow on the short leg; `net $10m` is the same at $10m.

## Walk-forward, 2007-11 to 2021-12

| fold | training rows | test | picked | train Sharpe | test Sharpe |
|---|---|---|---|---|---|
| 1 | 2007-11 to 2011-10 | 2011-12 to 2013-12 | lgbm | 4.76 | -0.25 |
| 2 | 2007-11 to 2013-10 | 2013-12 to 2015-12 | lgbm | 4.47 | 1.10 |
| 3 | 2007-11 to 2015-11 | 2015-12 to 2017-12 | lgbm | 4.16 | 0.49 |
| 4 | 2007-11 to 2017-10 | 2017-12 to 2019-12 | lgbm | 3.96 | 1.08 |
| 5 | 2007-11 to 2019-11 | 2020-01 to 2021-12 | lgbm | 3.75 | -0.05 |

The harness's training windows start at the calendar's first session in 2003-06; the rows in them start at the first month end with known membership, 2007-11-30, so fold 1 fits on 48 cross-sections and fold 5 on 145. The harness picks on training Sharpe, and a tree fit on its own training rows scores 3.7 to 4.8 there, so it picks LightGBM every time. Selected-in-sample OOS Sharpe **0.27** (research cost).

Every variant on every fold's test window, each fit on that fold's training rows only, 120 months stitched:

| variant | research | gross | net | net $10m | rank IC | IC t | hit | decile spread %/mo | spread t |
|---|---|---|---|---|---|---|---|---|---|
| mom | -0.14 | -0.08 | -0.15 | -0.18 | -0.004 | -0.23 | 50% | -0.19 | -0.34 |
| ridge | -0.20 | -0.08 | -0.20 | -0.25 | -0.012 | -0.69 | 47% | -0.28 | -0.56 |
| lgbm | 0.27 | 0.44 | **0.29** | 0.22 | 0.029 | 2.31 | 57% | 0.59 | 1.42 |
| ensemble | -0.02 | 0.14 | 0.00 | -0.05 | 0.006 | 0.43 | 47% | 0.14 | 0.32 |
| lgbm, no macro (post hoc) | -0.09 | 0.15 | -0.07 | -0.15 | -0.006 | -0.63 | 48% | 0.01 | 0.04 |

Feature importance across folds, LightGBM gain share (`reports/importance.png`, `reports/importance_lgbm.csv`): VIX 19%, 10-year yield 11%, `mom_12_1` 11%, 10y-3m slope 10%, `vol_63` 8%, everything else under 8%. Mean pairwise Spearman correlation of the five importance vectors is **0.97**; ridge coefficients rank-correlate 0.88 across folds with 86% sign agreement. That stability is partly by construction: the folds expand, so fold 5's training set contains all of fold 1's.

Three of the top four features are macro series. They are the same number for every stock on a given date, so a split on VIX cannot separate winners from losers within a cross-section; what it can do, below a split on a stock feature, is separate dates from each other, and with 169 training month ends and `min_child_samples=200` (under half of one month's cross-section) a leaf can be one month's worth of high-momentum names. That is how the training Sharpe reaches 4. The ablation row above was run after these results were seen, on the same folds, never on the window: without the macro series the walk-forward OOS falls from 0.27 to -0.09 and the IC from 0.029 to -0.006. Whatever LightGBM found out of sample in 2011-2021 lived in the macro conditioning, not in the stock features.

## The window, 2021-12-31 to 2026-08-31

1,170 sessions, 57 month ends (56 carry a full 21-session forward return, so IC and spread have n = 56). Current lock `reports/untouched.json`, opened 2026-09-05 01:57 with `{"model": "lgbm"}` fit on the 169 month ends from 2007-11-30 to 2021-11-30.

| | research | gross | net | net $10m | ann. return net | vol | max DD net | rank IC | IC t | hit | spread %/mo | spread t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | -0.59 | -0.37 | **-0.56** | -0.58 | -9.3% | 15.2% | -43.8% | -0.008 | -0.44 | 39% | -0.55 | -0.92 |
| 12-1 momentum | 0.23 | 0.28 | 0.23 | 0.22 | +2.5% | 27.3% | -34.1% | 0.011 | 0.38 | 59% | 0.40 | 0.41 |

PSR 0.11, **DSR 0.00 over 8 trials** for the LightGBM net path. The 8 is the registry's count of distinct keys (four grid variants, the stitched walk-forward stream, the ablation, the two window reads); the earlier reads described below were the same keys on different training rows and are not in that count, so the number of looks is higher than 8. A negative Sharpe deflates to zero whatever the count. Momentum's 0.23 over 4.7 years at 27% vol is inside one standard error of zero (about 1/sqrt(4.7) = 0.46).

**This window has been read three times, and the third read is not untouched in the strict sense.** The record:

1. 2026-09-04 20:32, `reports/untouched-first-open.json`. The result dictionary reused the key `chosen` for the variant's name and its metrics, so the stored result lost the name. Numbers: research -0.178, gross 0.021, net -0.151, IC -0.020.
2. 2026-09-04 20:33, `reports/untouched-second-open.json`, `reports/untouched_returns-second-open.csv`. The key renamed, nothing else changed, identical numbers to every digit. This was the published result: LightGBM trained on 2003-06 to 2021-11 with back-cast membership before 2007-12, on the lake as it stood at 20:41 (close hash `6117d9cc7b6d5470`).
3. 2026-09-05 01:57, `reports/untouched.json`. An audit found three defects: the lock was keyed to a live lake that a cron rewrote that evening, so `run.py` refused to run and the "reruns reproduce every number" claim was false; the first four and a half years of training rows carried back-cast membership while the README called the panel point-in-time; and the purge and the month-end selection were each short by a session or two in some folds. Fixing them (pinned snapshot, membership NaN before 2007-11-29, purge in sessions, rebalance dates from the calendar) changed the training set of the final model, so the frozen spec had to be refit and the window read again. The same model specification, trained on 2007-11 to 2021-11 instead of 2003-06 to 2021-11, went from -0.15 net to -0.56 net, and its IC from -0.020 to -0.008.

What that third read is and is not. The model parameters were not touched, the change was made to fix stated defects rather than to move the number, and the number moved against the model. But the training-row rule was changed after the window's first result was known, which is the thing the lock exists to make impossible, and a reader should treat 2022-2026 as a window this project has used rather than one it holds in reserve. The fair conclusion from the three reads is stronger than either number alone: the same frozen spec swings by 0.4 of Sharpe on the window when 4.5 years of doubtful training rows are removed, which is what a model with no signal does.

In-sample on the pre-window rows, for scale: LightGBM 3.25, ensemble 2.07, ridge 0.41, momentum -0.51 with a -96% drawdown. Point-in-time 12-1 momentum losing on this index is the same finding as `../momentum/` (-8.4%/yr gross PIT); the 2022-2026 gain is the momentum crash reversing rather than a signal returning.

## What this shows

- The platform's guarantees held on a 680-name panel: the peek audit passed with the membership feature attached, the walk-forward purged the full 21 sessions, the four full-window fits, the stitched OOS stream, the ablation and the window reads were logged (the 20 per-fold fits are not in the registry), and the window opened once per lock file.
- LightGBM with macro conditioning beat momentum and ridge on ten years of walk-forward folds, and the ablation attributes that to the macro series. On the window it lost 9.3% a year net.
- Rank IC reached 0.029 on the walk-forward (t 2.3, the one out-of-sample statistic here that clears 2) and -0.008 on the window. Decile spreads are 0.6%/mo at best with t under 1.5.
- Feature importances are stable across folds and that stability is not evidence of anything: expanding folds share most of their rows.

## Limitations

- **Macro features as date identifiers.** Three series constant within a date and 169 training months let the trees fit regimes. A fair version needs interactions built by hand (momentum times VIX rank) or a training row per date, not per stock-date, for anything macro.
- **Membership is known from 2007-11-29 and still imperfect after it.** `pit-universe/` logs 154 corrections in 2008-2010 where its change table is thin, each a snapshot-timing error of up to six months. Those years are in every training set. Every out-of-sample window here starts in 2011-12 or later, where the table is dense.
- **Coverage.** 433 of about 500 members have a price at the median month end. The missing names are delisted, disproportionately the losers the short leg would hold, and a name whose prices stop mid-hold earns zero rather than its delisting return. Both flatter the short leg; the momentum project sized the same hole.
- **One model spec, 56 months, three reads.** No hyperparameter search means no search bias and also no evidence the spec is a good one. The window section above says what the third read cost in evidential terms.
- **Stale adjustments.** `adj_close` is Yahoo's as of the day it was fetched; the lake's incremental runs only refresh the last week, so after a dividend the older rows keep the old factor. The snapshot freezes one fetch, so the panel is internally consistent and stays that way; it also stays behind any later corporate action.
- **The snapshot is local.** `reports/snapshot/` is 94 MB and gitignored like the lake's store. `reports/untouched.json` is committed and keyed to that snapshot's hash, so a clone with its own lake raises `LockError` on the first run rather than silently re-locking; to run it, remove the lock, and the window is then that clone's own first open on whatever its lake holds. The numbers in this README are reproducible from the snapshot on this machine, not from the repository alone, which is the same position every other project on the lake is in.
- **Two sessions of staleness.** Features are lag 1 and the research clock fills the day after the signal, so a month-end trade uses the close two sessions before it earns. The book's engine uses lag 0 and the next open; that path was not run here.
- **One session at the window's edge.** The first window return (2021-12-30 to 2021-12-31) is earned by the position set on 2021-11-30, a cross-section the final model trained on. No window return enters any label; one session of 1,170 is scored by a model that saw its features.
- **Impact at $100k is nothing.** On S&P names the participation is under 0.1% and the cost is the 7 bps of commission and spread. The $10m column is the same path at a hundred times the size and costs 0.07 of Sharpe; a real book of that size would also trade to the band, which is not modelled. Participation divides a real-dollar trade by adjusted-close dollar volume, which overstates it in early years, in the conservative direction.
- **Ridge coefficients are read off ranks with NaN as the median**, so a name with 21 days of history but no 252-day momentum sits at the middle of every missing feature rather than being dropped.

## Files

```
xs.py       load_raw (lake -> Raw, pinned to a snapshot directory), membership feature with known_from,
            XSModel (Alpha, takes the calendar's rebalance dates), ranks, scores, rank_ic, decile_spread,
            framework_net (book cost model on a position path), importance_stability
run.py      the whole thing, about 45 seconds; reruns reproduce every number from reports/snapshot/
check.py    33 offline checks on two planted panels, exits 1 on failure
reports/    summary.json, untouched.json (spent), untouched-first-open.json, untouched-second-open.json,
            snapshot/ (gitignored: equities_daily.parquet, fred.parquet, membership_intervals.csv),
            registry/runs.jsonl, models/index.jsonl + the pickled models, walk_forward_returns.csv,
            walk_forward_ic.csv, untouched_returns.csv, untouched_returns-second-open.csv,
            importance_lgbm.csv, importance_ridge.csv, equity.png, importance.png
```

Registry keys are hashes of name and config, so the ids are the same across the reads and `runs.jsonl` keeps every line with the latest per key winning: window read `0973047822ae2d85` (`XS[lgbm]`, config `{"model": "lgbm", "window": "untouched"}`), momentum read `af24a63ad382d6cf`, pre-window fit `2f21ad5381a7e5d0`, stitched walk-forward stream `5a006f9beaede414`, ablation `2d7b1e9eb252d191`. The current model is `afa96521665b` in `reports/models/index.jsonl`, whose meta carries the frozen LightGBM parameters, the feature lags, `member_from`, the first and last training cross-section and their count (169). The registry's own `n_train_rows` is the session count of the close frame it was handed, not a row count. The earlier id `514a6785755a` was keyed on the pre-window close and a meta that did not include the training-row rule, so the first post-fix run overwrote its pickle with the new model before the meta was extended; the old pickle is gone and the old numbers survive only in the lock files.

`check.py` plants a U-shaped effect (both momentum tails go up, the middle goes down) that LightGBM recovers at IC 0.25 while ridge and momentum score under 0.06, then a persistent drift all three find; it checks the membership feature including the NaN before `known_from`, within-date ranks, that a slice ending mid-month trains on calendar month ends only and that a fit without the calendar's dates refuses, that the walk-forward purge is 21 sessions on a calendar with holidays, the decile book's neutrality and counts, the cost model against a hand calculation, the snapshot round trip without the lake, and the diagnostics on known inputs. Three mutations (membership test removed, label not demeaned, borrow not charged) each fail at least one check.
