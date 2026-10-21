# Research platform

The desk's research layer: a point-in-time feature store, an experiment registry that feeds the deflated Sharpe, an Alpha interface that plugs into the premia book's engine, a model registry, and one untouched final window that can be opened once. Everything runs offline against `check.py`; `run.py` pushes a trivial momentum alpha through the whole thing on the book's 14 ETFs so the moving parts can be seen working together.

The one-line honest summary: cross-sectional 12-1 momentum across 14 ETFs scores research-path Sharpe 0.23 over 2004-2021 (DSR 0.49 over 7 trials; 0.59 over 5 before 2026-10-21, when the full-panel research and engine paths began to be logged as trials) and 0.44 on the untouched 2022-2026 window, but the walk-forward's selected-in-sample Sharpe is 0.04, so the pipeline is proven and the alpha is not.

## Layout

```
features/   store.py   Raw, Feature(name, fn, lag), FeatureStore.build(audit=True), cross_section, long_panel
            sets.py    PRICE (10 price/vol/momentum/liquidity features), macro(lags), edgar_counts(words)
registry/   experiments.py   Registry.record / stream / runs / trials / dsr, code_hash, use(path)
alpha/      base.py    Alpha, positions, backtest, walk_forward, Sleeve (framework adapter), Untouched (the lock)
models/     registry.py      ModelRegistry.save / load / entries, data_hash
check.py    71 offline checks on a planted panel, exits 1 on failure
prereg_check.py   a PREREG.md against template v2: tradability, window provenance, power
run.py      the worked example; writes reports/
```

## Feature store

A `Feature` is a name, a function from `Raw` to a dates x instruments frame (or a dates series, broadcast to every instrument) and a lag in sessions. There is no default lag; a feature without one does not construct. The store computes the function on the whole panel and shifts it by the lag, so the row dated t holds a value computed from data through t - lag, and that shift is the only lag in the system.

The shift cannot stop a function from reading rows after its own date, so `build(audit=True)` runs the peek audit: perturb every raw observation on one session, recompute, and require every feature row dated before that session plus its lag to be byte-identical. A feature that fails raises `PeekError` and the panel is not built. `check.py` shows the audit catching a `shift(-1)` and a centred rolling window, and passing the whole starter set.

`Raw` holds the wide OHLCV frames, any daily series (FRED, forward-filled onto the calendar) and a long frame of filings. `Raw.from_bars` builds one from a framework `Bars` view, which is how the sleeve adapter sees data. `Raw.from_lake(load, ...)` is written against the data-lake plan's one-line contract (`load(dataset, start, end)` returning a frame; datasets `equities/daily`, `fred/<id>`, `edgar/filings`); the lake did not exist when this was built, so that function is exercised only by the stub in `check.py`, which proves it builds the same panel as the direct path.

Starter sets, all declared at lag 1 so a value dated t is the close of t - 1:

| set | features |
|---|---|
| `PRICE` | `ret_1`, `ret_21`, `mom_3_1`, `mom_6_1`, `mom_12_1` (skip-month momentum), `vol_21`, `vol_63` (annualised log-return std), `dollar_vol_21`, `amihud_21`, `hl_range_21` |
| `macro({name: lag})` | one feature per attached series at the lag you declare; a FRED monthly print dated the 1st and released weeks later needs that release delay, not 1 |
| `edgar_counts(words)` | per-filing counts of each word or phrase in the filing text, carried forward until the next filing, plus `edgar_len` and `edgar_age` (sessions since the filing, as of t - lag) |

The lag-1 convention costs one session of staleness at a month-end rebalance. The book's own sleeves use the close of t and fill at t + 1's open; that is also honest, and an Alpha that wants it declares lag 0.

## Experiment registry

`Registry.record(name, config, universe, window, returns, code=...)` appends one JSON line and writes the return stream to `returns/<id>.parquet` (`stream(id)` reads it back, so the DSR of any row logged since `2d0b9a7` can be recomputed from disk; rows logged before that commit have no stream file, only their logged Sharpe and length). The key is the hash of name, config, sorted universe, window, code and a hash of the return stream itself: `code_hash(...)` of the rule (an Alpha's class source through its MRO, a function's source, a module's file bytes, falling back to the module file when inspect finds no source), or, when the caller passes none, the hash of the file that called `record`. The stream hash is what makes a rule edited where neither hash looks (a helper or a module constant in another file) a new row rather than a replaced one. `runs()` keeps the latest line per key, and `trials()` counts distinct keys with identical return streams (same daily Sharpe and length) counted once, the rule `framework/book/validate.py::trials()` uses, leaving out the walk-forward fold rows (stages `walk_forward_train` and `walk_forward_test`). Those rows are logged and stay on disk as the audit trail, but they are the folds of a rule already counted once on its whole window, and counting them made the DSR follow the fold settings rather than the number of rules tried (on a planted grid, 2, 3, 6 and 10 folds gave 11, 15, 27 and 43 trials and DSRs of 0.999, 0.997, 0.428 and 0.647 for the same three rules). A stream too close to constant to have a Sharpe (std below 1e-12 of its mean, or below 1e-15) is logged with a NaN Sharpe: it counts as a trial but stays out of the variance, and `dsr()` reports how many it left out (`n_nan_sharpe`); before, one constant 0.001-a-day stream logged a Sharpe of 2e15 and set every DSR in its registry to 0. `dsr(returns, n_trials=...)` takes the count a lock froze, so a read that logs itself before calling `dsr` is deflated by the number the lock holds. `dsr` calls `validate.deflated_sharpe_ratio` and `validate.probabilistic_sharpe_ratio` (purgedcv 0.1.5, the functions the book's validation uses) with that count and the variance of the counted daily Sharpes. Nothing here reimplements the formula.

The registry is per project: `research/reports/registry/` counts the looks taken here, `framework/book/reports/trials.csv` counts the book's. A W1 model that wants to be deflated against both passes the other count as `extra_trials`.

## Alpha and the adapter

```python
class Alpha:
    def fit(self, X, y): ...          # optional; X is (date, instrument) x features, y the forward return
    def signal(self, features): ...   # instruments x features at one date -> Series of weights
```

The research path is vectorised: `positions()` calls `signal` on the cross-section at each rebalance date, `backtest()` holds those weights until the next date with a one-session lag and 10 bps on traded weight. `Sleeve(alpha, store)` is a framework `Strategy`: at each rebalance it rebuilds the feature panel from the engine's as-of `Bars` view, takes the last row, and hands `signal`'s weights to the engine, which fills them at the next open under the book's cost model and risk overlay. The two paths compute features from different objects (the full panel vs the as-of view), so their targets agree only if every feature is causal; `check.py` asserts they agree to 1e-12 at every month end, and that a peeking feature breaks the agreement.

`walk_forward(make_alpha, grid, panel, close, n_splits, test_size)` uses purgedcv's `WalkForwardSplit` with the label horizon purged from the end of each training window, fits every grid variant on the training rows, keeps the best training Sharpe, holds it on the test window, and stitches the test returns.

The harness logs every evaluation itself. `backtest`, `window_returns` and `walk_forward` write one registry row per evaluation (for `walk_forward`: every variant on every fold's training rows, the pick on every test window, every variant on the whole window), keyed on the Alpha's code, the cost and a hash of the positions held, to the registry passed as `registry=` or the default `research.registry.use(path)` attached. With neither they raise `UnloggedEvaluation`. `log=False` with no registry attached runs unlogged, and it never stops an attached registry from logging. It is not the only way to run unlogged, and `grep` for the literal does not find them all: any expression that evaluates to `False` (`log=(1 > 2)`, `**{"log": bool(0)}`), a `functools.partial` that binds it once, and the private `base._returns`, which has no guard at all. The backstop is the lock, not the keyword: an untouched window names its registry and a sha256 of its `runs.jsonl` when it is locked, and records the hash again when it is opened, so a search run off the books shows up as trials the named registry does not hold, and a rewritten registry is refused.

## The untouched window

`Untouched(path).lock(calendar, frac, data_hash, registry=...)` writes the window's start, end, session count and the data hash, the real path of the registry the window belongs to (and its path from the lock's folder, which survives moving the repo) and a sha256 and byte length of that registry's `runs.jsonl` at the lock, plus a hash of the whole file (all of those, the opened stamp, the stored result and what `open` adds). A new lock needs a registry, passed in or attached with `use()`. Locking again with the same window is a no-op; any other window, or the same dates on different data, raises `LockError`, and so does a lock file whose contents no longer match its hash. `open(fn, registry=...)` runs `fn(start, end)` once, stores the result in the lock file and refuses every later call with `UntouchedWindowUsed`. It needs a registry (there is no `log=False` here) and refuses one that is not the registry the lock names, one with no counted trial (it would freeze `n_trials` 0), one whose `runs.jsonl` no longer starts with the bytes it held at the lock, and one that already holds a row whose window overlaps the untouched window (rows `fn` itself writes come after that check). It writes the registry's trial count and a sha256 of its `runs.jsonl` at the moment of opening, before `fn` runs, into the lock and its hash, so the DSR denominator of the one read is frozen with the result. Locks written before these fields existed still verify, and an unopened one takes the registry it is opened with (it still has to be non-empty and clear of the window). A rerun of `run.py` reports the stored result. To open the window again you have to delete the file by hand, and the commit history shows that you did.

`open` marks the window spent before it runs `fn`, so a crash inside `fn` also spends it. That is on purpose: a window you can retry until the code works is not untouched.

## Model registry

`ModelRegistry.save(name, model, train_data, train_window, oos_score, meta)` pickles the model and appends an index line with the metadata, the OOS score and a hash of the training frame's values, index and columns. The id is the hash of name, meta and data hash, so saving the same model on the same data twice is one entry and one changed training cell is a new one.

## Worked example (`run.py`)

`XSMom` ranks the 14 book ETFs (`framework.book.universe.ETFS`) on one momentum feature at each month end, goes long the top third and short the bottom third at equal weight. Panel 2003-06-02 to 2026-08-31, 5,850 sessions, read through the framework's frozen loader; close hash `03871989ff866efc`. Frozen before any result: baseline `mom_12_1`, grid `{mom_3_1, mom_6_1, mom_12_1}`, untouched window the last fifth of sessions as in the book's validation, chosen variant the one walk-forward picks most often with ties to the baseline.

**Walk-forward**, 4 folds of 504 sessions on the pre-window sessions, expanding train, 21-session purge:

| fold | train | test | picked | train Sharpe | test Sharpe |
|---|---|---|---|---|---|
| 1 | 2003-06 to 2013-11 | 2013-12 to 2015-12 | mom_3_1 | 0.18 | 0.10 |
| 2 | 2003-06 to 2015-12 | 2015-12 to 2017-12 | mom_12_1 | 0.30 | -0.32 |
| 3 | 2003-06 to 2017-11 | 2017-12 to 2019-12 | mom_12_1 | 0.23 | -0.29 |
| 4 | 2003-06 to 2019-12 | 2020-01 to 2021-12 | mom_12_1 | 0.19 | 0.49 |

Selected-in-sample OOS Sharpe **0.037**. Chosen: `mom_12_1`.

**Research path, pre-window** (2004-07-01 to 2021-12-30, 10 bps, no overlay): Sharpe 0.226, 2.6%/yr at 21.8% vol, max drawdown -57.7%. The other two variants score 0.18 and 0.19. PSR 0.83, **DSR 0.49 over 7 trials** (three grid variants, the walk-forward OOS stream, the untouched read, the full-panel research path and the engine path). Until 2026-10-21 it was DSR 0.59 over 5: the full-panel research path was not logged and the engine path was never a trial, though its untouched-window result is reported with a DSR. The walk-forward fold rows are logged but not counted (see the registry above).

**Untouched window** 2021-12-31 to 2026-08-31, 1,170 sessions, opened 2026-09-04 20:09, stored in `reports/untouched.json`:

| path | Sharpe | ann. return | ann. vol | max drawdown | DSR |
|---|---|---|---|---|---|
| research (10 bps, no overlay) | 0.440 | 6.9% | 19.4% | -35.8% | 0.77 |
| engine (book cost model, 10% vol target, drawdown overlay, 10% buffer) | 0.360 | 2.9% | 9.2% | -17.3% | 0.72 |

The DSR on the window uses the 4 trials logged before the window was read.

**Full panel**: research Sharpe 0.267, engine Sharpe 0.538 at 9.1% vol and -20.9% max drawdown, engine turnover 5.7x a year, 1,515 trades, average gross 0.83. (Rerun 2026-10-21; the 2026-09-04 run had engine Sharpe 0.386 at 9.5% vol and -22.2%, 6.9x, 1,876 trades, gross 0.97. The engine moved under it in later framework commits, among them the one order model and flattening the four ETF shorts this account cannot place; nothing in this project changed it, and the stored untouched result below is the 2026-09-04 read. Fold 3's train Sharpe moved from 0.24 to 0.23 the same way, with the purge counted on the calendar.) The engine beats the research path on the full panel because its vol target shrinks the book in 2008 and 2020 and its drawdown overlay halves it past 15%; that is the overlay, not the signal.

**Adapter parity**: the sleeve's targets equal the research path's at all 279 month ends, to 1e-12.

**Registry entry** for the chosen variant (`reports/registry/runs.jsonl`, latest line per key):

```
{"id": "...", "name": "XSMom[mom_12_1]", "config": {"feature": "mom_12_1"},
 "universe": ["DBA", ..., "USO"], "window": ["2003-06-02", "2021-12-30"],
 "metrics": {"sharpe": 0.226, "annual_return": 0.0259, "annual_vol": 0.218, "max_drawdown": -0.577, ...},
 "sharpe_daily": 0.01426, "n_obs": 4407, "tags": {"stage": "walk_forward_grid"}}
```

**Model entry** `19d062dc4a70` in `reports/models/index.jsonl`: the alpha's parameters and the store's feature lags, trained window 2003-06-02 to 2021-12-30, training close hash, OOS score 0.440 (the untouched research Sharpe).

`reports/equity.png` overlays the three research variants, the chosen one on the full panel and the engine run, with the untouched window shaded. `run.py` takes about 6 seconds and is deterministic; a second run logs the same keys (trial count stays 7) and reads the window's result back from the lock. Nothing past the window's start is evaluated before the lock is opened: the pre-window DSR comes from a backtest on the pre-window sessions, and the full-panel research and engine runs happen only after `open` succeeds or the lock is read as already spent, since `open` refuses a registry that holds a row over the window.

## What this does and does not show

- The pipeline works end to end and its guarantees are checked: lag enforcement, the peek audit, the lock, registry idempotence, adapter parity.
- The alpha is a placeholder. Selected-in-sample OOS of 0.04 says the lookback choice carries no information; 0.44 on the window is inside one standard error of zero over 4.6 years at this vol (about 1/sqrt(4.6) = 0.47).
- The engine number on the untouched window was read in the same `open` call as the research number, so the window was read once, but it was read for two cost models.

## Limitations

- **No data lake yet.** `Raw.from_lake` is written to a one-line contract and tested against a stub. When the lake lands its loader will need to match, or this function changes.
- **The audit perturbs one session.** It proves a feature does not read that session from earlier rows. A feature that peeks only at rows the audit did not perturb passes; `audit(raw, at=...)` can be run at more dates.
- **Rebuilding features per rebalance is O(T) per call**, so a daily-rebalance `Sleeve` over 23 years rebuilds the panel 5,850 times. Fine for the 14-ETF store (about 30 ms a build), not for thousands of names.
- **The code hash covers what is passed, not what it calls.** An Alpha is keyed on its class source, so a feature function or a module constant edited elsewhere is not in `code`; on the harness path the positions hash catches it, but a direct `record()` should pass `code_hash(...)` over every module its rule lives in (as `vol-risk-premium/prior.py`, `xs-trend/prior.py` and `kalshi-wide/replay.py` do). A cosmetic edit makes new keys, and their identical streams count once.
- **The worked example was rerun on 2026-10-21 under harness logging.** The harness now logs each walk-forward fold's training and test evaluations, the full-panel research path and the engine path. Counting the fold rows would have given 22 trials and a pre-window DSR of 0.29 (the note this replaces, checked against a copy of `reports/`); with fold rows logged but not counted it is 7 trials and 0.49, against 5 and 0.59 before. The window stayed spent and its stored result is unchanged.
- **A lock's hash has no secret.** `Untouched.write()` reseals any hand edit, so a lock file can be rewritten and rehashed by anyone with the repo. The tamper evidence is the git history of the tracked lock files, not the hash.
- **An attached registry logs everything, `log=False` included.** A project that attaches its registry with `use()` (every project `run.py` that goes through the harness does, at the top of `main()`) also logs its gross-of-cost diagnostics, which are marked `log=False` but still count as trials. An intraday project that runs `walk_forward` on bars also logs per-bar `walk_forward_grid` rows beside the daily rows it records by hand, and their Sharpes are in different units.
- **Lag is in sessions, not release calendars.** Macro publication lags vary by series and month; `macro({...: lag})` takes one number per series.
- **Research and engine numbers are not comparable head to head**: 10 bps flat with no overlay against the book's commission, spread, impact, borrow, vol target and buffer.

## Files

- `reports/summary.json`, everything printed above with full precision
- `reports/untouched.json`, the lock, spent
- `reports/registry/runs.jsonl`, `reports/models/index.jsonl` and the pickle
- `reports/returns.csv`, daily research and engine returns
- `reports/equity.png`
