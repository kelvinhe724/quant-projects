# Event-driven NLP: FinBERT on earnings 8-Ks against the price reaction

Does the tone of an earnings press release predict the drift after the announcement, and does it say anything the first day's price move has not already said? 9,552 earnings 8-Ks (item 2.02) by point-in-time S&P 500 members, 2022-01 to 2026-08, scored by FinBERT, run through the research platform's walk-forward and its untouched final window.

The one-line honest summary: FinBERT reads the release (Spearman 0.094 with the reaction-day return, t 4.6), but by the first open a trader can use, nothing is left. Sentiment, the price reaction and sentiment residual to the reaction all have IC within 0.01 of zero at every horizon from 1 to 20 sessions; the long-short books lose money net of 10 bps at every horizon in the grid; on the untouched last year the pre-registered baseline scores net Sharpe -1.84, sentiment -0.77, residual sentiment -1.09, DSR 0.004 / 0.028 / 0.014 over 13 trials. Sentiment adds nothing beyond the price reaction after the reaction, because there is nothing after the reaction to add to.

## Data

- **Filings.** The lake's `edgar_8k` set only reached back three days, so `backfill.py` fills it from the SEC submissions API: for every name in the point-in-time panel with a CIK in SEC's current ticker map (550 of 586), every 8-K or 8-K/A with item 2.02 filed 2022-01-01 to 2026-08-31, keeping the first EX-99 exhibit (the press release; the 8-K body when there is none, 121 filings) and the acceptance timestamp, written through `lake.write` in the collector's columns plus `accepted`. 10,661 filings, 545 names, median 32,660 characters. About 40 minutes at the SEC's 10 requests a second, resumable.
- **Prices.** `lake.load("equities_daily", universe="sp500")`, so a name is only priced on days it was a member; Yahoo's adjusted close and the open scaled by the same factor. 554 names with prices, 1,212 sessions from 2021-11-01 (warm-up for the first reaction returns). 9,580 of the filings fall on a session where the ticker is in the priced panel; 9,552 are distinct (reaction session, ticker) events.
- **Model.** `ProsusAI/finbert`, cached by HuggingFace, on Apple MPS. Score = P(positive) - P(negative), mean over the first four windows of 510 tokens (the release's headline and first pages), 11.7 documents a second, scores cached in `reports/sentiment.parquet`. Mean score +0.22, median +0.23, three quarters above zero: press releases are written to sound good.

## Timing

Every filing carries its EDGAR acceptance time. 52% are accepted before 09:00 ET (the 06:00 to 08:00 cluster) and 39% in the 16:00 hour. The reaction session R is the session whose close-to-close return carries the announcement: the same day when accepted before 09:00 ET, the next session otherwise. Two signals are defined at the close of R and dated R at lag 0:

- `react`: close(R) / close(R - 1) - 1, the announcement-window return.
- `sent`: the FinBERT score.
- `resid`: sentiment residual to the reaction, from an OLS of `sent` on `react` fitted on the trailing window's events.

Each is turned into a percentile against every event whose reaction session fell in the trailing 126 sessions, the event itself included, so a rank uses nothing after its own close. The harness prices the strategy on p(t) = open(t + 1), so a weight set on row R fills at the open after the reaction session and a horizon of h sessions is open-to-open; `check.py` verifies that mapping by hand. The reaction is knowable at the entry open, not before it: a strategy that wants the reaction-day return itself cannot have it, and the entry gives up the day-one move by construction, as Bernard and Thomas did. The feature store's peek audit passes at lag 0 and a reaction computed from the next close raises `PeekError`.

## Event study, pre-window

7,426 events whose 20-session horizon ends before the untouched window. Excess return over SPY from the entry open, Spearman IC pooled across events, t from the 43 monthly ICs.

| horizon | react IC | t | sent IC | t | resid IC | t |
|---|---|---|---|---|---|---|
| 1 | 0.006 | -0.15 | -0.024 | -0.74 | -0.023 | -0.66 |
| 2 | 0.008 | 0.57 | -0.004 | 0.16 | -0.004 | 0.20 |
| 5 | 0.005 | -0.34 | 0.003 | 0.09 | 0.003 | 0.15 |
| 10 | 0.011 | -0.28 | 0.004 | 0.01 | 0.003 | 0.06 |
| 20 | 0.011 | -0.05 | 0.009 | -0.41 | 0.008 | -0.40 |

Nothing at any horizon. The reaction day itself is different: sentiment's IC with the reaction-day excess return is **0.094 (t 4.6)**, and, over the full sample of 9,552 events (window included), the bottom sentiment decile falls 1.24% on the day against +0.53% for the top (`reports/sentiment_deciles.png`; raw return, not excess). The model reads the direction of the release. That return is inside the gap at the open of R, or, for post-close filers, in the next morning's gap, and the first open after acceptance is on the far side of it. Entering at open(R) instead of open(R + 1), which only a pre-market filer allows, gives sentiment an IC of 0.034 at one session (t 2.05) and 0.02 after that: the tail of the reaction, not a drift.

The ten-session drift by sentiment decile runs from -0.33% to +0.23% with no order across deciles.

## Walk-forward and the untouched window

Grid, frozen before any result: three signals times three horizons (5, 10, 20 sessions), long the top fifth of the percentile and short the bottom fifth at equal weight, daily, 10 bps on traded weight (the research path's cost). Baseline `react` at 10 sessions; `sent` and `resid` at 10 sessions are the two pre-registered comparisons. Untouched window the last fifth of sessions, 2025-09-12 to 2026-08-28, 242 sessions, 2,079 events, locked on the close hash `d04e3b0d4acfa365` and opened once on 2026-09-04 21:51.

Every grid variant loses in the pre-window sessions (2021-11 to 2025-09): Sharpe -0.39 for the baseline, -1.48 for sentiment, -1.43 for the residual, -0.42 to -2.62 for the rest. The walk-forward (3 folds of 252 sessions, expanding train, 21-session purge) picks the baseline in every fold and its stitched selected-in-sample OOS Sharpe is **-0.37**, so the chosen variant is the baseline, which also means the final set is the three pre-registered variants.

| untouched window | Sharpe net | Sharpe gross | ann. return | ann. vol | max drawdown | DSR |
|---|---|---|---|---|---|---|
| react, 10 (baseline) | **-1.84** | -0.75 | -25.8% | 15.6% | -27.4% | 0.004 |
| sent, 10 | **-0.77** | 0.28 | -11.3% | 14.2% | -13.7% | 0.028 |
| resid, 10 | **-1.09** | -0.04 | -14.9% | 13.9% | -14.8% | 0.014 |

DSR over 13 trials (nine grid variants, the walk-forward stream, the three window reads), variance of the logged daily Sharpes 2.3e-3. IC on the window's 2,079 events: sentiment 0.039 at 10 sessions (t 1.15), residual 0.037 (t 1.05), reaction -0.028 (t -0.09). Sentiment's gross Sharpe on the window is positive and its net is not; a ten-session book turns over about 40% of its gross a day, and 10 bps on that is close to 10% a year, which is the whole gap between the gross and net columns.

Registry ids, `reports/registry/runs.jsonl`: baseline on the window `6262641bf686e1e4`, sentiment `f873be33b6274305`, residual `10c718a5ab503129`; walk-forward stream `297c779c0b42b915`. Model entry `148cbda234de` in `reports/models/index.jsonl` (the chosen alpha's parameters, the feature lags, the FinBERT name and chunking, trained window 2021-11-01 to 2025-09-11, OOS score the window's Sharpe).

Full panel, for the picture in `reports/equity.png`: baseline -0.68, sentiment -1.32, residual -1.36, by year in `reports/summary.json`.

## What this does and does not show

- FinBERT's score is informative about the release, contemporaneously. The 0.094 IC with the reaction day has a t of 4.6 over 43 months and a monotone decile pattern.
- There is no post-announcement drift in this sample to capture with either signal. Reaction-sorted drift, the textbook PEAD, has an IC of 0.011 at ten and twenty sessions with a t under 0.3 in the pre-window years; on the untouched year it is negative. That is consistent with the literature on large caps since the 2000s, and this is the largest-cap universe there is.
- Sentiment residual to the reaction is the direct test of "adds anything beyond", and its IC is 0.003 at ten sessions (t 0.06). Sentiment does not add to a signal that itself has nothing.
- Every strategy number here is net of 10 bps on traded weight and the research path's one-session convention; there is no vol targeting, no overlay, no borrow cost. Through the book's engine the numbers would be lower.

## Limitations

- **One regime, one untouched year.** 4.7 years of filings, 242 sessions on the untouched window; the standard error on an annualised Sharpe over one year is about 1.0, so the window says "not large and positive", nothing finer.
- **The score is tone, not surprise.** FinBERT on the first 2,040 tokens of a press release measures how the company chose to write, with no consensus estimate, no SUE and no sector adjustment; the percentile mixes a utility's release with a chip maker's. The one baseline is the price reaction, not an analyst-surprise drift.
- **Timing by EDGAR acceptance, not the wire.** The 8-K is furnished after the press release goes out, sometimes hours after. A company that releases at 06:30 and files at 09:30 gets R = next session while the market reacted today, so its `react` is the wrong day; 220 filings were accepted in the 09:00 hour and about 200 more between 10:00 and 15:59. The reaction-day IC and the decile picture are diluted by that, not inflated.
- **First EX-99 exhibit only.** A release split across EX-99.1 and EX-99.2 (tables in the second) is scored on the first; 121 filings had no EX-99 and are scored on the 8-K body, which is boilerplate.
- **Universe holes.** 36 of 586 panel names have no CIK in SEC's current ticker map (delisted, acquired or renamed since; ABMD, ATVI, TWTR and the like), and the map is today's, so a reused ticker can carry a filing by a company that was never in the panel under a ticker that was; the lake's own README says the same. 1,081 filings fall on a day the ticker is not in the priced point-in-time panel and are dropped.
- **No amendment handling beyond form type.** 8-K/A rows are excluded; a restated release is not.
- **Ranks warm up.** A percentile needs 50 events in its trailing window, so the first two weeks of 2022 have none; the pre-window Sharpes start 2022-01-26.

## Run

```
../.venv/bin/python3 check.py                # 35 offline checks on a planted panel, 3 s, exits 1 on failure
../.venv/bin/python3 backfill.py --minutes 9 # fills the lake's edgar_8k from 2022; rerun until it prints "complete"
../.venv/bin/python3 run.py --minutes 9      # scores (cached, resumable), event study, walk-forward, window, reports/
```

`run.py` takes about 40 seconds once the scores are cached; the first scoring pass is about 14 minutes on MPS. A rerun reports the stored window result and leaves the trial count at 13. `transformers` was added to `requirements.txt`.

Files: `reports/summary.json` (everything above at full precision), `reports/untouched.json` (the lock, spent), `reports/ic_by_horizon.csv` and `.png`, `reports/events.csv` (every event with its score, reaction and ranks), `reports/returns.csv`, `reports/equity.png`, `reports/sentiment_deciles.png`, `reports/sentiment.parquet` (score per accession).
