# Realised-vol models

Four forecasters of realised volatility raced on SPY and twenty large names over 2003-2026, then the one question I actually cared about: does the best of them make the short-straddle sleeve in `vol-risk-premium/` better if it only sells when implied minus forecast is wide enough?

The one-line honest summary: GJR-GARCH wins the horse race (QLIKE 0.223 on the untouched window at 21 days, DM t -9.3 against trailing vol pooled across the names), and timing the straddle with it makes the sleeve worse, not better: on the untouched 2022-2026 window the timed rule scores Sharpe 0.37 on capital against 1.41 for selling every month, because it sits out 85% of the time and the premium was there anyway. DSR 0.52 over 16 trials for the timed rule, 0.83 for the untimed sleeve.

## Data and plumbing

Everything comes through the repo's shared layers rather than this project's own code:

- Prices: `lake.load("equities_daily", ...)` for SPY plus AAPL, MSFT, AMZN, NVDA, JPM, BAC, GS, XOM, CVX, JNJ, PFE, MRK, UNH, PG, KO, PEP, WMT, HD, MCD, DIS, 2003-06-02 to 2026-09-03, 5,853 sessions, every name with a full history. Returns are on Yahoo's adjusted close, the Parkinson range on the raw high and low. Close hash `5006a92054c4fdae`.
- Macro: `lake.load("fred", ...)` for VIXCLS, T10Y2Y and DTB3. The lake's HY OAS series only starts in 2023, so it is not used.
- Features: `research.features.FeatureStore` with the peek audit on. Ten features, all at lag 0 except the two rates at lag 1: `rv_1`, `rv_5`, `rv_21`, `rv_63` (annualised close-to-close, zero mean), `park_21`, `ret_1`, `ret_21`, `vix`, `t10y2y`, `dtb3`. Lag 0 means the value dated t uses the close of t, and the forecast dated t is for the sessions strictly after t.
- Selection, lock, registry, DSR: `research.alpha.walk_forward`, `research.alpha.Untouched`, `research.registry.Registry`, `research.models.ModelRegistry`. Nothing here reimplements any of them.
- The straddle: `vol-risk-premium/vrp.py::simulate`, one call per cycle.

`check.py` runs offline on six simulated GJR-GARCH paths with planted parameters (31 checks, exits 1 on failure); `run.py` is the study. First run fits the forecasters in about six minutes and caches them to `reports/forecasts.parquet`; a rerun takes five seconds and reads the untouched result back from the lock.

## The forecasters

All five forecast the annualised vol of the next h sessions, h = 5 and 21, dated the session the forecast is made on, refit on an expanding window every 63 sessions after a 756-session warm-up, parameters held between refits. Training rows whose label would cross the refit date are purged.

| model | what it is |
|---|---|
| naive | trailing realised vol over the same h sessions |
| HAR-RV | per name, OLS of log RV(t+1..t+h) on log rv_1, rv_5, rv_21 at t |
| GARCH(1,1) | per name, student-t, `arch`; between refits the parameters are frozen and arch's analytic multi-step forecast is run from every session |
| GJR-GARCH(1,1,1) | the same with the asymmetric term |
| LightGBM | one pooled model across the 21 names on log RV(t+1..t+h) from the ten features in logs; 300 trees, learning rate 0.03, 15 leaves, 200 minimum rows, 0.8 row and column sampling, chosen once and never tuned |

HAR and LightGBM take logs of rv_1, so a session with a zero return (0.7% of rows) gives no forecast from them the next day; the counts in the tables show it.

## Horse race, pre-window (2006-06-01 to 2022-01-03, 21 names pooled)

Scores are pooled over every name and session with a forecast. QLIKE is `RV²/F² - log(RV²/F²) - 1`, the loss the vol literature uses because it does not reward a forecast for being low when realised is high. Bias is forecast minus realised in vol points.

5-day horizon:

| model | RMSE | MAE | bias | QLIKE | corr | n |
|---|---|---|---|---|---|---|
| naive | 0.1645 | 0.1011 | -0.0000 | 1.236 | 0.656 | 82,446 |
| HAR | 0.1509 | 0.0852 | -0.0264 | 0.687 | 0.672 | 81,906 |
| GARCH | 0.1453 | 0.0929 | +0.0239 | 0.506 | 0.697 | 82,446 |
| GJR | **0.1430** | 0.0916 | +0.0233 | **0.496** | **0.707** | 82,446 |
| LightGBM | 0.1532 | **0.0852** | -0.0218 | 0.650 | 0.649 | 81,906 |

21-day horizon:

| model | RMSE | MAE | bias | QLIKE | corr | n |
|---|---|---|---|---|---|---|
| naive | 0.1381 | 0.0821 | -0.0001 | 0.502 | 0.705 | 82,446 |
| HAR | 0.1302 | **0.0732** | -0.0176 | 0.447 | 0.698 | 81,906 |
| GARCH | 0.1266 | 0.0765 | +0.0113 | 0.356 | 0.721 | 82,446 |
| GJR | **0.1255** | 0.0761 | +0.0107 | **0.352** | **0.725** | 82,446 |
| LightGBM | 0.1498 | 0.0784 | -0.0105 | 0.445 | 0.622 | 81,906 |

Diebold-Mariano on the QLIKE differential, pooled as the cross-sectional mean loss per session, Newey-West with h - 1 lags. Row beats column when negative.

21 days:

| | naive | HAR | GARCH | GJR | LightGBM |
|---|---|---|---|---|---|
| naive | | 1.49 | 6.15 | 6.29 | 2.61 |
| HAR | -1.49 | | 3.30 | 3.07 | 0.07 |
| GARCH | -6.15 | -3.30 | | 0.64 | -2.92 |
| GJR | -6.29 | -3.07 | -0.64 | | -2.86 |
| LightGBM | -2.61 | -0.07 | 2.92 | 2.86 | |

At 5 days every model beats naive at t below -11, GJR beats GARCH at -2.9 and both beat HAR and LightGBM at -8. Per name at 21 days, GARCH and GJR beat naive at 5% on 19 of 21, HAR on 7, LightGBM on 8 (`reports/dm_vs_naive_by_name_pre_h21.csv`). On squared error rather than QLIKE the picture is flatter: at 21 days GJR beats naive at t -3.3 and among the four models no pair differs at 5%.

So: the GARCH family wins because it gets the level right when vol is high, which is what QLIKE pays for; HAR and LightGBM have the lower MAE and the negative bias, they sit under realised in the spikes. LightGBM with ten features and no tuning is not better than a two-parameter recursion from 1986, and it is worse than HAR at 21 days on RMSE by a wide margin (0.150 vs 0.130), which I read as the pooled tree model extrapolating badly at the top of the vol range.

SPY alone at 21 days, with the VIX as a fifth forecast:

| model | RMSE | MAE | bias | QLIKE | corr |
|---|---|---|---|---|---|
| naive | 0.0982 | 0.0584 | -0.0000 | 0.633 | 0.659 |
| HAR | 0.0949 | 0.0533 | -0.0168 | 0.574 | 0.624 |
| GARCH | 0.0905 | 0.0554 | +0.0103 | 0.421 | 0.681 |
| GJR | **0.0865** | **0.0525** | +0.0023 | 0.436 | 0.699 |
| LightGBM | 0.1336 | 0.0631 | +0.0094 | 0.555 | 0.518 |
| VIX | 0.0919 | 0.0656 | +0.0369 | **0.404** | **0.711** |

DM, VIX against GJR: t 0.91 on QLIKE, nothing. The VIX is 3.7 points too high on average and still the best QLIKE forecast, the same finding `vol-risk-premium/` made with a weaker GARCH. GJR is the first model here that is not beaten by the VIX at 5%, which is the thing that makes the trading test worth running.

## The trading test

The sleeve is `vrp.simulate` run one cycle at a time from each month end to the next (279 cycles, 1 bp hedge slippage, 1% of premium on entry, sold and marked at the VIX, as in that project). Its daily P&L per $1 of notional becomes one synthetic instrument, `STRADDLE`, whose "close" is the compounded P&L index, and a research `Alpha` called `Timer` decides at each month end whether to hold it: sell when `VIX/100 - forecast_21d` exceeds a threshold, where the threshold is the q-quantile of that gap over the training rows. The harness's 10 bps on traded weight is charged when the rule switches. Frozen before any result: baseline always-on, grid of four forecasters by q in {0.25, 0.5, 0.75}, untouched window the last fifth of sessions, chosen variant the one the walk-forward picks most often with ties to the baseline.

Walk-forward, 4 folds of 504 sessions on the pre-window sessions, expanding train, 21-session purge:

| fold | train | test | picked | train Sharpe | test Sharpe |
|---|---|---|---|---|---|
| 1 | 2003-06 to 2013-12 | 2014-01 to 2015-12 | GJR q=0.75 | 3.05 | never traded |
| 2 | 2003-06 to 2015-12 | 2016-01 to 2018-01 | GJR q=0.75 | 2.90 | never traded |
| 3 | 2003-06 to 2017-12 | 2018-01 to 2020-01 | GJR q=0.75 | 2.73 | never traded |
| 4 | 2003-06 to 2019-12 | 2020-01 to 2022-01 | GJR q=0.75 | 2.90 | 2.90 |

Every fold picked the same rule and in three of the four it did not put on a single trade: the threshold learnt from 2008-2012 (5.8 vol points of gap) was never reached in 2014-2019. Selected-in-sample OOS Sharpe 2.90 on active days, 1.14 on all days.

Every variant on the pre-window sessions, fit once on all of them:

| variant | Sharpe, active days | Sharpe, all days | return/yr, all days | max drawdown | time in market |
|---|---|---|---|---|---|
| always on | 1.90 | **1.89** | **9.7%** | -14.1% | 100% |
| HAR q=0.25 / 0.5 / 0.75 | 1.66 / 1.56 / 1.72 | 1.34 / 1.08 / 0.85 | 6.3 / 4.7 / 3.1% | -14.1 / -14.9 / -12.0% | 65 / 48 / 25% |
| GARCH q=0.25 / 0.5 / 0.75 | 1.75 / 1.73 / 1.64 | 1.42 / 1.23 / 0.80 | 6.1 / 4.8 / 2.6% | -14.0 / -14.4 / -14.4% | 66 / 51 / 24% |
| GJR q=0.25 / 0.5 / 0.75 | 2.10 / 2.29 / **2.84** | 1.70 / 1.56 / 1.31 | 6.9 / 5.6 / 3.5% | **-5.5 / -5.5 / -4.3%** | 66 / 46 / 22% |
| LightGBM q=0.25 / 0.5 / 0.75 | 1.65 / 1.57 / 1.61 | 1.33 / 1.08 / 0.72 | 6.1 / 4.5 / 2.3% | -14.1 / -14.1 / -13.9% | 65 / 47 / 20% |

Read the two Sharpe columns together. The harness scores Sharpe on active days, the convention the book uses so a sleeve's dead time before launch does not count, and on that measure GJR q=0.75 is the best thing in the table at 2.84. On capital, which is what a book owner has, nothing in the grid beats selling every month, and the higher the threshold the worse it gets. The walk-forward chose by the first column, so it chose the rule that trades least. What the GJR rules do buy is the drawdown: -5.5% against -14.1%, because the threshold kept them out of the second half of 2008 and out of the Covid entry month; they were in for the recovery.

**Untouched window** 2022-01-04 to 2026-09-03, 1,171 sessions, opened 2026-09-04 20:43, stored in `reports/untouched.json`. Three reads, all logged: the walk-forward's choice, the always-on baseline, and the pre-window horse-race winner's rule, which is the same object as the choice.

| read | Sharpe, active days | Sharpe, all days | return/yr, all days | max drawdown | time in market | DSR | trials |
|---|---|---|---|---|---|---|---|
| chosen, GJR q=0.75 (threshold 5.76 points) | 0.95 | **0.37** | 0.8% | -2.9% | 15% | 0.52 | 16 |
| always on | 1.41 | **1.41** | 7.4% | -7.0% | 100% | 0.83 | 16 |

Timing loses on every measure but drawdown. (DSR in that table is on the active-day Sharpe, the way the registry computes it: 0.95 over 179 sessions for the timed rule, not the 0.37.) The rule was in the market for 179 of 1,171 sessions and the sessions it picked were not better than the ones it skipped, even per session (0.95 active-day Sharpe against 1.41 for all of them). The premium in 2022-2026 was paid in ordinary months, and a rule that waits for a 5.8-point gap waits through them.

Horse race on the same window, 21 days pooled:

| model | RMSE | MAE | bias | QLIKE | corr |
|---|---|---|---|---|---|
| naive | 0.1077 | 0.0737 | -0.0006 | 0.328 | 0.582 |
| HAR | **0.0916** | **0.0616** | -0.0191 | 0.267 | **0.653** |
| GARCH | 0.0956 | 0.0661 | +0.0103 | 0.225 | 0.637 |
| GJR | 0.0963 | 0.0658 | +0.0108 | **0.223** | 0.645 |
| LightGBM | 0.0965 | 0.0640 | -0.0078 | 0.239 | 0.620 |

Pooled DM at 21 days on the window: GJR beats naive at -9.3, HAR at -3.2, LightGBM at -1.6; GJR vs GARCH -0.4. Per name GARCH and GJR beat naive on 20 of 21, HAR on 14, LightGBM on 13. At 5 days every model beats naive at t below -13 and the GARCH pair beats HAR and LightGBM at -5 to -9. The ranking on QLIKE held out of sample; the ranking on RMSE flipped to HAR, which is the low-vol-window story (2022-2026 had no 2008 or 2020), and LightGBM's RMSE went from worst to middle for the same reason. SPY alone on the window: GJR QLIKE 0.218 against the VIX's 0.249, DM t -1.2, not significant.

Pre-window DSR for the chosen rule is 0.998, which is the active-day Sharpe of 2.84 over 1,022 sessions run through the same formula. It is an accurate number for a quantity nobody should size on.

`reports/models/index.jsonl`: `05b9ddc36d45`, the 21-day LightGBM refit once on the 95,872 pre-window rows, OOS score its RMSE on the window (0.0965).

## What this shows

- On QLIKE the GARCH family is the forecaster to use, GJR by a hair over GARCH, and the gap to HAR and LightGBM is significant in both samples at both horizons. On squared error the models are close and HAR is competitive. The asymmetric term is worth about nothing at 21 days once GARCH already has the level right.
- The VIX is not beaten by any of them on SPY. That is the same result as `vol-risk-premium/` and it is why the gap rule has little to work with: implied minus GJR is mostly the risk premium plus noise, not a forecast error to be timed.
- The timing test failed. It failed in sample on capital, and the harness's active-day selection hid that by picking the sparsest rule, and it failed out of sample by a wide margin. The sleeve's problem was never which months to sell; it is the tail inside the months it sells, and a level-of-vol threshold does not address that.

## Limitations

- **The active-day Sharpe.** `research.alpha.walk_forward` selects on `sharpe(r[r != 0])`, the book's rule for sleeves with a launch date. For an on/off rule it rewards sitting out, and the folds show it picking a rule that then traded zero times in six years of test windows. I report both columns and chose by the harness because the rule was frozen, but a timing study through this harness should select on all days. That is a harness change, not a change here.
- **The forecasts do not go through the peek audit.** The ten input features do (lag 0, audited), and the trading panel's pass-through features do trivially, but a rolling model is not a `Feature` and the audit cannot recompute it. `check.py` does the equivalent by hand: perturb every session after p and require every forecast dated before p unchanged, for all five models. It passes; it is a weaker test than the audit because it runs on one p.
- **Two lock files.** The first open stored a result whose dict keys were integers, which JSON turned into strings, so the lock's hash of its own contents failed on the reread. `reports/untouched_first_open.json` is that file, kept as written; the lock was recreated and opened again with the result cleaned before storage, and the second read reproduced the first's numbers exactly (same cached forecasts, same code path, identical to four decimals in every table above). The window was read twice by the code and once by me. I record it because the lock exists to record exactly this.
- **Sold at the VIX.** The sleeve inherits `vol-risk-premium/`'s largest caveat: an ATM straddle trades two to four points under the VIX, and that project showed a four-point basis takes the untimed sleeve negative. A rule that only sells on wide gaps might survive that basis better than always-on does; I did not test it, because the gap threshold would then need the same basis subtracted and there is no historical ATM series in the lake to do it honestly.
- **Month-end cycles, not 21-session cycles.** The harness rebalances at month ends, so the straddle tenor is 19 to 23 sessions and the always-on baseline here is not the same series as `vol-risk-premium/` (Sharpe 1.89 pre-window on 2003-2021 against 1.74 there on 2010-2026, a different calendar and a different period).
- **One threshold family.** Quantiles of the gap. A rule in the VIX level, in the forecast's own change, or in the term structure was not tried; three quantiles by four models was the whole grid, 16 trials.
- **Adjusted closes are as of the lake's last refresh.** The close hash pins the data the lock was taken on; a later dividend refetch changes the hash and the lock will refuse, which is correct and means `run.py` will not rerun without moving the lock aside.
- **LightGBM was not tuned**, by design, and pooled across names with no name identifier. A tuned per-name model would do better on RMSE; whether it would beat GJR on QLIKE I doubt, given how much of QLIKE is the level in spikes.

## Files

- `vol.py` features, the five forecasters, scoring, DM, the straddle cycles, `Timer`
- `check.py` 31 offline checks
- `run.py` the study, about six minutes cold and five seconds warm
- `reports/summary.json` everything above at full precision; `untouched.json` the spent lock; `untouched_first_open.json` the first one
- `reports/registry/runs.jsonl` 16 trials, `reports/models/` the LightGBM
- `reports/scores_pre_h{5,21}.csv`, `dm_pooled_qlike_pre_h{5,21}.csv`, `dm_vs_naive_by_name_pre_h{5,21}.csv`, `walk_forward.csv`, `returns.csv`
- `reports/spy_forecasts.png`, `horse_race.png`, `timing.png`

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```
