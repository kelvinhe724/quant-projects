# Options chain collector: a home-grown implied vol history

The volatility-risk-premium and dispersion projects both ended on the same
limitation: the backtest needs the implied vol that was quoted on each past
day, and free sources only have today. OptionMetrics sells that history. This
folder is the other route: snapshot the full chain every trading day after the
close, derive a clean surface from it with my own solver, and let it accumulate.

Thirteen underlyings: SPY, QQQ, IWM and the ten largest SPY weights (AAPL,
MSFT, NVDA, AMZN, META, GOOGL, BRK-B, JPM, XOM, UNH), which is the dispersion
project's basket. Every listed expiry out to 400 days.

## What accumulates

Under `source-material/options-collector/` (gitignored, as all downloaded data
in this repo is):

```
<date>/<TICKER>.csv     raw Yahoo quotes: bid, ask, last, volume, OI, Yahoo's IV,
                        last trade time, spot, rate, snapshot time, T
<date>/underlying.csv   last price per ticker and the date it belongs to (a close
                        only when the session is `close`)
<date>/rates.csv        FRED DTB3 on or before the date, with its observation date
<date>/weights.csv      market cap per single name, for the basket weights
<date>/manifest.json    snapshot time, session (close or intraday), what was
                        collected, what failed
summary/<date>.csv           one row per underlying per expiry
summary/<date>_horizons.csv  one row per underlying at 30, 60, 90 days,
                             plus the basket implied correlation on the SPY row
```

Storage on 2026-09-04: 5.6 MB raw per day (36,406 quotes), 41 KB of summary.
A year is roughly 1.4 GB raw, which is the price of never having to recollect
when a cleaning rule changes.

Nothing is ever overwritten. A rerun on a date that already has a ticker file
skips it, files are opened in mode `x` so even a race cannot clobber one, and
a download that fails is recorded in the manifest as missing rather than
filled in from anywhere else. The manifest is the only file that is rewritten,
and it is derived from what is on disk.

## Summary schema

`summary/<date>.csv`, one row per underlying per expiry:

| column | meaning |
|---|---|
| date, ticker, snapshot, session | session is `close` for the scheduled run, `intraday` otherwise |
| expiry, T | T in years to 4pm New York on the expiry date |
| spot, F, r | last price, parity forward for the expiry, bill rate as a decimal |
| n_raw, n_clean | quotes for that expiry before and after cleaning |
| k_min, k_max | log-moneyness range the clean quotes cover |
| atm_iv | mid implied vol interpolated to k = 0 |
| iv_25d_put, iv_25d_call | mid vol at the strikes with forward delta -0.25 and +0.25 |
| skew_25d | put minus call |
| stale | true if more than half the raw chain had neither bid nor ask |

`summary/<date>_horizons.csv`, one row per underlying: `iv30`, `iv60`, `iv90`
interpolated linearly in total variance across the expiries above, and on the
SPY row only, `rho30`, `rho60` (implied correlation of the single-name basket
against SPY), `n_names` (how many of the ten had a vol that day) and
`weights_from` (which day's market caps set the weights).

Both column lists are module constants in `surface.py` and `check.py` asserts
the files match them exactly, so a consumer can rely on the order.

## How the surface is built

Everything in the summary is re-derived from bid and ask. Yahoo's own
`impliedVolatility` column is stored but never used; it carries no rate,
forward or exercise assumption and is 0.00001 wherever the vendor's solver
gave up.

The cleaning and inversion are the SVI project's, imported by file path from
`../svi/` so there is one copy of that code:

1. drop zero bids, crossed markets, quotes with no trade in the last day,
   zero volume, spreads wider than 25% of mid unless under 10 cents
2. forward per expiry from put-call parity near the money, `F = K + (C - P) / D`,
   `D = exp(-rT)` with the FRED 3-month bill; no dividend yield is assumed
   because the parity forward already contains it
3. keep `|k| <= 0.5` and the out-of-the-money side only
4. invert Black-76 by brentq for every surviving mid, NaN outside no-arbitrage bounds
5. drop expiries with fewer than six clean quotes

ATM vol is the smile interpolated at k = 0. The 25-delta wings are read by
interpolating vol against forward delta separately on the put and call side,
without extrapolation, so an expiry whose clean quotes do not reach 25 delta
gets NaN there rather than a guess. Fixed horizons use the dispersion
project's `vol_at_horizon` (linear in total variance) and the implied
correlation is its `implied_correlation`, both imported.

## Today's numbers

The first real run, 2026-09-04 at 13:49 UTC, which is 9:49 New York, twenty
minutes into the session. That makes it an intraday snapshot and the manifest
says so. It is here because it exercises the whole pipeline on real quotes,
not because it is a good day of data.

| ticker | raw | no bid or ask | clean | expiries with ATM vol |
|---|---|---|---|---|
| SPY | 7,814 | 20% | 2,117 | 23 |
| QQQ | 6,905 | 8% | 2,122 | 23 |
| IWM | 2,935 | 3% | 726 | 21 |
| AAPL | 1,947 | 7% | 402 | 16 |
| MSFT | 2,263 | 14% | 415 | 15 |
| NVDA | 2,604 | 4% | 613 | 15 |
| AMZN | 1,608 | 10% | 473 | 17 |
| META | 3,542 | 5% | 680 | 15 |
| GOOGL | 2,510 | 19% | 317 | 14 |
| XOM | 735 | 3% | 24 | 3 |
| UNH | 1,227 | 33% | 65 | 5 |
| BRK-B | 1,160 | 44% | 0 | 0 |
| JPM | 1,156 | 57% | 0 | 0 |

JPM is flagged stale and BRK-B nearly is. That is Yahoo not having populated
the book yet, and the trade-recency and volume filters then remove most of
what is left on the thinner names. After the close the same filters keep far
more; the SVI project's evening SPY snapshot kept 407 of 2,456 with a stricter
twelve-quote slice floor.

SPY ATM implied vol by expiry, 25-delta skew alongside:

| expiry | days | ATM | skew |
|---|---|---|---|
| 2026-09-08 | 4 | 7.0% | +1.0% |
| 2026-09-11 | 7 | 9.7% | +1.6% |
| 2026-09-18 | 14 | 11.3% | +2.5% |
| 2026-10-16 | 42 | 12.2% | +3.3% |
| 2026-11-20 | 77 | 13.3% | +4.0% |
| 2026-12-18 | 105 | 13.9% | +4.4% |
| 2027-03-19 | 196 | 15.1% | +5.1% |
| 2027-06-17 | 286 | 16.0% | +5.6% |
| 2027-09-17 | 378 | 16.9% | +5.8% |

Fixed horizons: SPY 11.6% / 12.8% / 13.3% at 30 / 60 / 90 days against a VIX
of 14.10 printing at the time of the snapshot (14.32 at the 2026-09-03 close),
so the ATM-to-VIX gap the VRP project had to guess at is 2.5 to 2.7 points
today. QQQ 17.0 / 18.7 / 19.3, IWM 16.3 / 17.6 /
18.3, NVDA 33.8 / 36.8 / 37.7, META 34.5 / 38.8 / 40.1.

Implied correlation of the basket against SPY: **-0.015 at 30 days, -0.013 at
60**, over the eight names that had a vol. That is not a bug and it is not a
tradeable level. The ten names are about a third of SPY and its highest-vol
third, so a 30-day index vol of 11.6% against single names at 24 to 35% forces
the algebra to near zero. The dispersion project measured 0.027 on its own
snapshot with the same formula, and realised correlation for this basket was
negative over the prior month. The series is useful for its changes.

## How long until it is useful

Rough, and I would rather say it than not:

- **60 trading days** for term-structure work: whether the 30-to-90-day slope
  mean-reverts and how fast. Slopes move on a scale of weeks, so that is ten to
  fifteen distinct moves, enough for a sign and a scatter.
- **A year** for a VRP redo: the strategy sells 21-day straddles, so a year is
  twelve non-overlapping cycles, the point where a Sharpe has a standard error
  under one. The current VRP result is Sharpe 1.74 marked at the VIX, 0.73 if
  the ATM basis is two points, negative at four. A year of measured basis
  settles that.
- **A year** before the dispersion backtest can run on measured single-name
  vols instead of a scaled realised proxy.

## Swap-in paths

**Volatility risk premium.** `vrp.simulate(df, iv_col="vix", iv_offset=...)`
reads one column of the daily frame as the entry vol, in percent. Build the
series with

```python
h = surface.load_horizons()
spy = h[h.ticker == "SPY"].set_index("date")["iv30"] * 100
```

filtered to `session == "close"` rows, join it onto the VRP daily frame, and
pass `iv_col="atm30"` with `iv_offset=0`. Until the overlap is long enough to
run on directly, the first use is calibrating `iv_offset` against the VIX on
the collected days and applying that to the 2010 to 2026 history as a labelled
experiment.

**Dispersion.** The implied side of `simulate` there is the proxy the README
describes as a one-function change. The replacement inputs are `iv30` for SPY
and each of the ten names from the same horizons file, and `rho30` from the
SPY row, keyed by date. Rows with `n_names < 10` should be dropped or
reweighted, and the `weights_from` column says whose market caps were used.

## Files

- `data.py` universe, storage layout, chain fetch, FRED rate with a Yahoo ^IRX fallback
- `collect.py` the daily snapshot; `collect.py run --daily` or `--date YYYY-MM-DD`
- `surface.py` cleaning, own vols, ATM, 25-delta skew, horizons, implied correlation, summary files
- `check.py` 40 offline checks on a planted smile: idempotency, solver round-trip, schema, stale detection
- `run.py` summarises every collected day, prints the latest, writes `reports/`
- `com.kelvinhe.options-collector.plist` launchd job, written but not loaded; it
  carries this machine's absolute paths so it stays local and gitignored

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 collect.py run --daily
../.venv/bin/python3 run.py
```

To schedule it, weekdays at 15:35 Chicago (16:35 New York):

```
cp com.kelvinhe.options-collector.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kelvinhe.options-collector.plist
```

`launchctl bootout gui/$(id -u)/com.kelvinhe.options-collector` stops it.
Output goes to `reports/launchd.log`. The job runs `collect.py` then `run.py`,
both idempotent, so a second trigger on the same day is harmless. Market
holidays produce a folder whose quotes match the prior close; the
`underlying.csv` `close_date` column shows which day the prices belong to.

## Limitations

**One snapshot a day, from Yahoo.** No intraday history, no exchange
timestamps, and a vendor whose book is visibly incomplete early in the session.
The 16:35 schedule is a guess at when the closing quotes have settled; if the
first week of `close` manifests show large empty-quote fractions the time
should move later.

**Single names are thin after cleaning.** The trade-recency and volume filters
are right for a surface fit and harsh for a daily series. XOM kept 3 expiries
and UNH 5 even before the intraday penalty. Because the raw chain is stored,
the filters can be re-run on the whole history with `summarise_day(date,
force=True)` when the consumer wants coverage over cleanliness.

**The implied correlation is a sub-basket proxy.** Covered above. It measures
ten names against an index they are a third of.

**American exercise.** SPY, QQQ, IWM and every single name here are
American-style, and the inversion is Black-76 European. On short-dated
out-of-the-money quotes the early-exercise premium is small; on deep
in-the-money puts it is not, which is one reason the in-the-money side is
dropped.

**The rate is stale by construction.** FRED publishes DTB3 with a lag, so the
stored rate is usually two days old and its observation date is recorded next
to it. At current rates that moves a 30-day forward by under a basis point.

**No holiday calendar.** The scheduler fires on every weekday and the
collector does not know a holiday from a trading day. The `close_date` column
is the tell, and a consumer should drop rows whose close date is not the
folder date.

**Nothing is verified against a second source.** Every vol here is mine. The
SVI project's snapshot of SPY and the dispersion project's basket numbers are
the only cross-checks, and both come from the same Yahoo feed.
