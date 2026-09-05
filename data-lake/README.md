# Data lake: one store, one loader

Every project in this repo so far fetched its own data and cached it its own
way: `momentum/` a CSV of prices, `pit-universe/` another, `nelson-siegel/` a
FRED pull, `kalshi-desk/` a SQLite ledger. Each was right for its project and
none of them can be read by the next one without going through that project's
code. This folder is the fix: one Parquet store under `store/`, one loader,
and a set of collectors on launchd that keep it filled.

```python
import sys; sys.path.insert(0, "../data-lake")
import lake
px = lake.load("equities_daily", "2015-01-01", "2020-12-31", universe="sp500")
```

`universe="sp500"` keeps a row only if the ticker was in the S&P 500 on that
row's date, by the `pit-universe/` membership spells, so a backtest that
loads through the lake cannot hold a name before it joined. `as_of=` is a hard
ceiling on the time column: no row after it is ever returned, whatever `end`
says. Both are tested in `check.py`. `as_of` is an instant: a date-only `end` on an
intraday set means the end of that day, but a date-only `as_of` means midnight.

## What is in the store

Numbers from `run.py` on 2026-09-05 01:15 UTC, after each collector ran once
in the foreground (`reports/inventory.csv`, `reports/summary.md`).

| dataset | what | rows | keys | span | cadence | source |
|---|---|---|---|---|---|---|
| `equities_daily` | raw OHLCV plus Yahoo's adjusted close | 3,355,162 | 680 tickers | 2003-06-02 to 2026-09-03 | daily | yfinance, universe from `pit-universe/reports/membership_intervals.csv` plus SPY |
| `fred` | 20 series: fed funds, SOFR, bills, the CMT curve 1m to 30y, 10y-2y and 10y-3m, CPI and core CPI, payrolls, unemployment, VIX, HY OAS, the target rate | 192,858 | 20 series | 1939-01-01 to 2026-09-03 | daily | fredgraph.csv |
| `kalshi_markets` | every open market's listing: bid, ask, last, volume, OI, liquidity, times | 5,000 | 5,000 tickers | one snapshot so far | hourly | Kalshi public API through `kalshi-desk/kalshi.py` |
| `kalshi_books` | full depth, both sides, on the 200 highest 24h-volume markets | 5,318 | 199 tickers | one snapshot so far | hourly | same |
| `edgar_8k` | full text of every 8-K by a universe company: body plus EX-99 exhibits, items, an earnings flag | 37 | 35 tickers | 2026-09-02 to 2026-09-03 | daily | SEC daily form index, complete submission files |
| `crypto_l2` | top-20 bids and asks, once a second, BTCUSDT and ETHUSDT | 336 (first 3 minutes; the daemon is running) | 2 symbols | from 2026-09-05 01:10 UTC | 1 s | Bybit spot websocket, OKX as fallback |
| `crypto_klines_1m` | 1-minute spot candles | 83,520 | 2 symbols | 2026-08-06 to 2026-09-03 | 1 min | data.binance.vision |
| `crypto_bookdepth` | USD-M perp depth at 1 to 5% from mid, both sides, about once a minute | 2,004,480 | 2 symbols | 2026-08-06 to 2026-09-03 | ~1 min | data.binance.vision |
| `options` | the chains `options-collector/` snapshots | 36,406 | 13 tickers | 2026-09-04 | daily | read through that project's `data.py`; not copied |

Store size after the first run: 183 MB, 150 of it equities.

Equity coverage against the point-in-time panel is the number that matters
most and it is not good early on: at June 30 of 2004 the panel has 493 members
and 300 of them have a price row; 2010, 351 of 499; 2016, 419 of 504; 2026,
501 of 503 (`reports/coverage_by_year.csv`, `reports/coverage.png`). That is
Yahoo not serving delisted names, the same lower bound `pit-universe/` reported
(257 of 390 deleted names have no prices). The lake makes the hole visible; it
does not fill it.

## Layout

```
store/<dataset>/<part>.parquet
```

A part is one ticker or series for `equities_daily` and `fred`, one calendar
day for the Kalshi and EDGAR sets, one month for the Binance archives and one
hour (`<date>/<HH>.parquet`) for the L2 stream. `lake.load` opens only the
parts a request can touch, so a 12-name load over one year reads 12 files.
`lake.write(dataset, part, df, keys)` is the only write path: it reads the part
if it exists, drops duplicate keys keeping the new row, sorts, and replaces the
file through a rename, so a collector killed mid-write leaves the old part
intact. Every row carries `collected_at`.

Time columns: `date` (naive, daily), `ts` (UTC), `filed` (EDGAR), `snapshot`
(options). Bounds you pass are read as UTC when the column is tz-aware and as
plain dates otherwise; a date-only `end` on an intraday set means the end of
that day.

## Collectors and schedule

`collect.py <feed>` runs one feed and prints one line. All of them are
idempotent: a rerun rewrites the same parts with the same keys.

| feed | what one run does | schedule |
|---|---|---|
| `equities` | yfinance `auto_adjust=False` for the 894-name universe; incremental from seven days before the newest stored date; `--full` backfills from 2003-06-01 for names with no part yet, so an interrupted first run resumes | daily 21:30 Chicago, weekdays |
| `fred` | full history of each series; a revised value replaces the stored one | same job |
| `edgar` | for each of the last three filing days: the daily form index, 8-K and 8-K/A rows whose CIK is in the universe map, then the complete submission `.txt`, split into documents, keeping the 8-K body and every EX-99 exhibit, HTML stripped; items parsed from the first 20,000 characters; `earnings` is true when 2.02 is among them | same job |
| `archive` | the last 30 daily zips of spot 1m klines and USD-M `bookDepth` from data.binance.vision, cached under `source-material/data-lake/` | same job |
| `kalshi` | five pages of open markets, then depth on the 200 with the most 24h volume | hourly |
| `crypto` | websocket L2, one top-20 snapshot per symbol per second, flushed to the hour part every 60 s; reconnects with backoff on any error; after three consecutive failures on a venue moves to the next | 24/7, `KeepAlive` |

`collect.py daily` runs equities, fred, edgar, archive and then `quality.py`;
`collect.py hourly` runs kalshi and `quality.py`. A feed that raises is logged
and the others still run.

Venues, recorded on 2026-09-05 from this machine: Binance.com REST and
websocket both answer 451 (restricted location); data.binance.vision serves
the archives; Bybit REST is behind a CloudFront block but its spot websocket
connects and delivers `orderbook.50` at about 27 messages a second; OKX
connects both ways and its `books` channel delivers 400 levels. The daemon
tries Bybit first. Both parsers are unit-tested on planted snapshots and
deltas, and both ran live for the first parts (89 s Bybit, 19 s OKX; top-of-book
spread 0.0126 bps on BTC and 0.0407 on ETH at both, which is one tick). The
`venue` column says which wrote each row.

### launchd

Three jobs, plists written in this folder (gitignored, machine paths) and
copied to `~/Library/LaunchAgents/`. Loaded 2026-09-05:

```
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kelvinhe.lake-daily.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kelvinhe.lake-hourly.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kelvinhe.lake-crypto.plist
```

To stop any of them:

```
launchctl bootout gui/$(id -u)/com.kelvinhe.lake-daily
launchctl bootout gui/$(id -u)/com.kelvinhe.lake-hourly
launchctl bootout gui/$(id -u)/com.kelvinhe.lake-crypto
```

`lake-crypto` is `KeepAlive`, so launchd restarts it within ten seconds if it
exits; `bootout` is the only way to stop it. All three log to
`reports/launchd.log`.

## Quality

`quality.py` writes `reports/status.json`:

```json
{"generated_at": "...", "ok": true, "summary": "all feeds fresh",
 "datasets": {"crypto_l2": {"parts": 1, "rows": 336, "last": "2026-09-05 01:15:35+00:00",
                            "stale": false, "gaps": [{"after": "...", "before": "...", "missing": 30}],
                            "dupes": 0}, ...}}
```

`stale` is the newest row being older than the feed's allowance (3 minutes
for the L2 stream, 2 hours for Kalshi, 4 days for the daily sets, 3 days for
the archives, which lag a day). `gaps` are the ten largest holes in the recent
parts at the feed's cadence, from `lake.gaps`, which knows that a weekend is
not a gap in a business-day series and a holiday is; SPY currently shows
Juneteenth and July 3. `dupes` counts repeated keys. `ok` is no feed stale and
no duplicates; gaps are reported, not judged, because a real holiday and a
missed day look the same. The morning brief reads `summary`.

## Checks

`check.py` runs against a temporary store and never touches the real one:
round trip through `write` and `load` with window, universe and sort;
`write` deduping and replacing on keys; `as_of` on three dates and `as_of`
beating `end`; the PIT universe excluding a name before it joined, inside its
spell only, and after it left; tz-aware bounds; the gap detector on a planted
30-second hole, hourly holes with counts, business-day holes across a weekend,
a bare weekend, and empty input; the CIK map (zero padding, class-share dash,
unknown ticker, two tickers on one CIK); 8-K parsing (document types kept and
dropped, items, HTML and entities); Bybit and OKX book deltas (delete, insert,
size update, padding to depth); tz-aware bounds and `as_of` as an instant on intraday rows; and the staleness verdicts. 42 checks, exits 1
on any failure.

## Limitations

- **Adjusted closes go stale.** `adj_close` is Yahoo's dividend-and-split
  adjusted series as of the day the row was fetched. An incremental run only
  refetches the last seven days, so after a dividend the older rows keep the
  old adjustment. `close` and `volume` are raw and do not have this problem.
  Rebuild with `collect.py equities --full` after moving `store/equities_daily`
  aside, or adjust from raw with a corporate-actions table this store does
  not yet have.
- **Yahoo's Sep 4 bars were not there at 21:15 ET.** 660 of the 662 names
  with a 2026-09-03 bar had no 2026-09-04 bar at the first run; the two that
  did, CSRA and SPLS, are delisted names Yahoo serves phantom rows for. The
  seven-day overlap picks the real bars up on the next run, and the PIT
  filter drops the phantoms.
- **`universe="sp500"` ends where the membership file ends.** Every current
  member's spell closes on the date `pit-universe/` was last built (2026-08-31
  today), so a PIT load past that date returns no rows at all, not today's
  index. Rebuild `pit-universe/` before loading recent dates through the filter.
- **`collected_at` is last-seen, not first-seen.** The seven-day overlap
  rewrites recent rows on every run and the stamp moves with them, so it
  cannot be used to reconstruct when a bar first appeared.
- **One L2 writer at a time.** `write` is read-merge-rename with no lock; a
  foreground `collect.py crypto` while `lake-crypto` is loaded races it on the
  current hour part and one flush loses the other's rows.
- **FRED has no vintages here.** A revised CPI or payroll print overwrites
  the earlier value; `collected_at` tells you when the current value was
  seen but not what it was before. ALFRED would fix that and is the obvious
  next collector.
- **Ticker reuse in the CIK map.** SEC's `company_tickers.json` is today's
  mapping. A ticker that left the index and was reassigned (STI, once
  SunTrust) maps to whoever holds it now, so `edgar_8k` can carry a filing
  by a company that was never in the panel under a ticker that was. The
  `cik` column is the truth; the ticker is a convenience.
- **EDGAR's daily index lags.** The index for day D was not published at 21:07
  ET on D (the server answers 403 until it is), so D's filings land on D+1's
  run. The three-day lookback is what makes that safe.
- **The L2 store is one venue at a time.** Keys are `(ts, symbol)`, so if the
  daemon fell back to OKX and Bybit came back in the same second, one would
  overwrite the other. In practice a fallback lasts until the next
  disconnect. Sampling is wall-clock, not exchange time, and the row is the
  book as of the last message before the second ticked, with no sequence
  numbers or checksum verification, so a dropped delta corrupts the book
  until the next reconnect. A 5-second silence forces one.
- **No L2 backfill exists.** data.binance.vision has no order-book archive;
  `bookDepth` (depth at 1 to 5% bands, USD-M perps, about once a minute) is
  the closest thing and is a different instrument from the spot books the
  stream records. Klines are there for completeness.
- **Kalshi depth is 200 books an hour**, chosen by 24h volume, and the
  listing snapshot is the only record of the other 4,800.
- **Options are read, not copied.** `lake.load("options")` calls
  `options-collector/data.py`, so it is as current as that job.
