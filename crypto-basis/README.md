# Crypto perp-spot basis

Binance's BTCUSDT and ETHUSDT perpetuals against Binance spot, 5-minute bars from January 2024 through July 2026, plus one month of OKX minute prints against the same Binance minutes. The question was whether the perp-spot basis dislocates by enough to trade it back to its mean after fees, funding and a bar of latency, and whether the same coin prints far enough apart on two venues to be worth carrying inventory on both.

The one-line honest answer: no, not at any fee tier I can get. A 3-sigma dislocation of the basis against its trailing day is about 3 bps wide and gives back 1.2 bps per round trip before costs; a taker round trip on both legs costs 34 bps, a maker one 28, and the best published VIP tier still costs 9. The mean-reversion rule scores gross Sharpe 4.5 on the research window and loses 87% a year net on the untouched window at taker fees, 41% a year at VIP fees. Cross-venue, the OKX-Binance gap never once exceeded a round trip in 44,160 minutes of July 2026. The basis is real, persistent and small, and everything here is a measurement of how small.

## Data

Everything comes through `data-lake/` (`lake.load`), and `data.py` is the collector that fills it:

- `crypto_klines_1m`: Binance spot 1-minute klines, monthly zips from `data.binance.vision`, 2024-01 to 2026-07 (the lake's own collector already carried August and September from the daily zips)
- `crypto_perp_klines_1m`: the USD-M perpetual's 1-minute klines, same source and months
- `crypto_funding`: realised 8-hour funding settlements, 2,829 per symbol
- `okx_trades_1m`: OKX publishes daily trade dumps and no candles, so `data.okx_minutes` builds last, size-weighted vwap, volume and count per minute from every trade on BTC-USDT, BTC-USDT-SWAP, ETH-USDT and ETH-USDT-SWAP for July 2026 (124 files, 1.3 GB of zips cached under `source-material/crypto-basis/`)

The three new datasets are registered on `lake.DATASETS` at import in `data.py` rather than by editing the lake, so `lake.load` on them works only after `import data`; that line belongs in `data-lake/lake.py` when the lake next changes. Binance's timestamp column switches from milliseconds to microseconds during 2025 and the newer files carry a header row; both are handled and checked. So is a third quirk that the first version of this project missed: 991 of the 2,829 settlements per symbol are stamped one millisecond after the hour (`HH:00:00.001`), and a plain reindex onto the bar grid drops them. `funding_accrual` rounds settlement stamps to the minute and `check.py` plants a 1 ms late settlement to hold it there. 271,585 5-minute bars, no gaps, close hash `105cc57660dbee8c`.

## The instrument

The research harness trades prices, so the trade is written as one: P = spot / perp, multiplied by (1 + funding) on the bar that closes at each settlement. One unit is a dollar of spot long against a dollar of perp short; P's return is spot minus perp plus what the short collects. A rich basis (perp above spot) is a low P. `check.py` plants a settlement and shows the jump, and shows P's return equals spot minus perp to first order.

Costs, per unit, one way: spot taker 10 bps, perp taker 5, a basis point of slippage on each leg, 17 in total and 34 a round trip. Two other tiers are carried through the untouched read: maker (10 + 2 + 2 = 14 a side) and Binance's top VIP tier (1.5 + 1 + 2 = 4.5 a side, 9 a round trip, most of it the slippage assumption). Latency is one bar: every feature is published at lag 1, so a decision at bar t uses the close of t - 1 and fills at the close of t, five minutes later. Returns are on the spot notional and ignore the margin parked against the perp, so a fully collateralised version earns half of every number here.

## Basis distribution

5-minute basis in bps, perp over spot minus one:

| | BTCUSDT | ETHUSDT |
|---|---|---|
| mean | -2.87 | -2.82 |
| median | -4.26 | -4.24 |
| std | 3.97 | 4.14 |
| 1st / 99th pct | -6.69 / 10.29 | -6.96 / 10.44 |
| min / max | -40.7 / 77.8 | -62.1 / 95.5 |
| share negative | 85% | 85% |
| autocorr at 1h / 1d | 0.95 / 0.90 | 0.94 / 0.90 |

By year: 2024 mean -0.3 bps with std 5.3; 2025 mean -4.3 with std 1.2; 2026 to July mean -4.8 with std 0.9. The top panel of `reports/basis.png` shows what that is: a basis that spent the first half of 2024 between +5 and +13 bps while funding was high, and has sat at about -4.5 bps with sub-basis-point noise since. The persistent negative level is the difference between Binance's spot USDT print and the multi-venue index the perp is marked to, not a free carry: funding averaged 0.64 bps per 8 hours (16% of settlements negative) over the whole period and funds the level away.

The distribution, by-year, exceedance and reversion tables run on all 271,585 bars, untouched window included; they describe the data and pick nothing, and the grid and rule were frozen before any of them was computed. Dislocation is the basis minus its trailing one-day mean. Share of bars where |dislocation| exceeds a threshold:

| threshold | what it is | BTC | ETH |
|---|---|---|---|
| 1 bp | | 29.7% | 30.7% |
| 2 bps | | 5.8% | 6.0% |
| 5 bps | | 0.47% | 0.52% |
| 9 bps | VIP round trip | 0.08% | 0.10% |
| 20 bps | | 0.012% | 0.015% |
| 28 bps | maker round trip | 0.004% | 0.010% |
| 34 bps | taker round trip | 0.002% | 0.007% |

That last row is 6 bars out of 271,585 for BTC and 19 for ETH, nearly all in the first weeks of 2024. After a 10 bp excursion begins (60 BTC episodes, 94 ETH), 38 to 40% is given back within one bar, 61 to 69% within an hour and 79 to 83% within four hours, so the reversion is real and fast; it is the size that is missing.

## The trade and the walk-forward

`BasisMR(window, entry)` buys P when the z-score of the basis against its trailing window crosses above `entry`, sells it below `-entry`, and holds until |z| falls under 0.5. Frozen before any result: grid windows of one day, four hours and one hour (288, 48, 12 bars) by entries of 2 and 3 sigma, six variants, baseline the one-day window at 2 sigma; four walk-forward folds on the bars before the untouched window with the harness's default purge; chosen variant the one picked most often, ties to the baseline; untouched window the last fifth of bars, locked to the data hash before the walk-forward ran.

Walk-forward, 17 bps a side (`reports/walk_forward.csv`):

| fold | train | test | picked | train Sharpe | test Sharpe |
|---|---|---|---|---|---|
| 1 | 2024-01 to 2024-05 | 2024-05-30 to 2024-10-28 | w=288, z=3 | -25.9 | -27.7 |
| 2 | 2024-01 to 2024-10 | 2024-10-28 to 2025-03-28 | w=288, z=3 | -26.1 | -25.4 |
| 3 | 2024-01 to 2025-03 | 2025-03-28 to 2025-08-26 | w=288, z=3 | -26.1 | -24.6 |
| 4 | 2024-01 to 2025-08 | 2025-08-26 to 2026-01-23 | w=288, z=3 | -26.1 | -25.9 |

Fold Sharpes are in the same unit as every other Sharpe here: bar returns compounded to UTC days, flat days dropped, sqrt(252). The harness's default fold score is the Sharpe of the nonzero bars, which on 5-minute bars is neither daily nor annual, so `run.py` passes `basis.daily_sharpe` as the fold score; `check.py` shows it agrees with the registry's Sharpe to the last digit. Selected-in-sample OOS Sharpe: **-25.8**. Every fold picks the slowest, widest variant because it trades least; the fast 2-sigma variants lose everything inside three months (middle panel of the chart). Chosen: w=288, z=3.

Pre-window, 2024-01-01 to 2026-01-23, chosen variant, daily returns, registry convention (252-day annualisation on a 365-day market, so multiply Sharpes by 1.2 for the calendar-year figure): net Sharpe -25.7, 1,418 round trips. Gross of costs the same positions score Sharpe 4.5 and capture 1.25 bps per round trip. A round trip costs 34.

## Untouched window

2026-01-24 to 2026-08-01, 54,433 bars, 190 days, opened on 2026-09-05 and stored in `reports/untouched.json`; the same call read three fee tiers, which is one look at the window for three cost models, and the registry counted 7 trials before it. This is the second read of these bars, see Limitations:

| fee tier | round trip | Sharpe | annual return (365d) | max drawdown | DSR |
|---|---|---|---|---|---|
| taker | 34 bps | -25.3 | -87.5% | -66.0% | 0.00 |
| maker | 28 bps | -25.3 | -81.8% | -58.7% | 0.00 |
| VIP | 9 bps | -24.5 | -40.8% | -23.8% | 0.00 |

323 round trips, in a position 2.4% of the time. The window lands within half a Sharpe point of the walk-forward's -25.8, which is what a strategy with no signal-to-cost ratio looks like: the P&L is the fee schedule.

Registry: `reports/registry/runs.jsonl`, 8 trials (six grid variants, the stitched walk-forward stream, the untouched read), chosen variant's entry `e5b4be1b512157c1`. The DSR column is 0.00 because a Sharpe of -25 has no chance of being positive, and the gross pre-window DSR (1.3e-266) is uninformative the other way: the variance of daily Sharpes across trials is 3.4 because five of the eight are fee massacres, and the DSR deflates against that spread. Model `5b78134c526b` in `reports/models/index.jsonl` holds the chosen parameters and feature lags with the untouched taker Sharpe as its score.

## Cross-venue monitor

July 2026, one-minute prints, OKX last trade over the Binance close of the same minute (`reports/venue_monitor.csv`, bottom panel of the chart):

| pair | mean | std | 1st / 99th pct | autocorr 1m | round trip | minutes over cost |
|---|---|---|---|---|---|---|
| BTC spot OKX/Binance | +0.34 bps | 0.74 | -1.49 / 2.04 | 0.55 | 22 bps | 0 of 44,160 |
| BTC perp OKX/Binance | -0.06 | 0.77 | -1.87 / 1.60 | 0.73 | 12 | 0 |
| ETH spot OKX/Binance | +0.11 | 0.75 | -1.61 / 1.89 | 0.38 | 22 | 0 |
| ETH perp OKX/Binance | -0.06 | 0.78 | -1.82 / 1.65 | 0.69 | 12 | 0 |
| BTC basis, OKX | -4.74 | 0.75 | -6.45 / -2.92 | 0.59 | 34 | 0 |
| BTC basis, Binance | -4.34 | 0.75 | -5.97 / -2.65 | 0.54 | 34 | 0 |

The round trip for a cross-venue trade assumes inventory already sits on both venues (a taker and a basis point of slippage on each of the two legs: 22 bps for spot, 12 for perp), because an on-chain transfer takes minutes and the gaps last one. Three minutes in the month crossed 10 bps, none of them a cross-venue spot pair: two on the Binance BTC basis (worst 11.4 bps, 2026-07-21 17:46) and one on the ETH perp OKX/Binance gap (11.8 bps, 2026-07-02 13:31), all inside their round trips. The worst BTC spot print was 4.9 bps, the worst BTC perp print 7.7. OKX spot prints 0.34 bps above Binance on average, which is inside the tick, and OKX's own basis sits 0.4 bps below Binance's, so the two venues' perps are marked to the same index and their spot prices differ by less than a taker fee on either.

## Limitations

- **The window was read twice.** The first read (2026-09-04, kept as `reports/untouched_first_read.json`, hash `b887eacece990a5e`) was on an instrument that silently dropped the 991 settlements per symbol stamped 1 ms after the hour, so the short collected 65% of its funding. An audit found it, `funding_accrual` was fixed, and because the lock is keyed to the data hash the window had to be relocked on the corrected instrument and read again on 2026-09-05. Nothing about the rule, the grid or the chosen variant changed between the reads and the fix could not have been chosen against the window: on the same Sharpe convention it moves the taker Sharpe by less than 0.01 (-25.34 before and after) and the annual return from -87.48% to -87.47%. The first read's file shows -25.43 because it was written before the harness fix below to `Registry.dsr`. It is still a second look at the same 54,433 bars, and any further change judged against them is in-sample. The gross pre-window numbers were added to `run.py` after the first read and are in-sample by construction.
- **Bars, not books.** Fills are at the 5-minute close on both legs, slippage is an assumed basis point a leg. The lake's L2 stream (`crypto_l2`) began in September 2026 and could replace the assumption with a walked book, but only for the days it covers.
- **One venue's basis.** The mean-reversion trade is Binance perp against Binance spot only. OKX's basis is measured for one month as a monitor and not traded, and the two bases move together (both about -4.5 bps with 0.75 bps of noise), so a cross-venue basis trade would be a trade on their 0.4 bp difference.
- **Funding in the index is realised, not predicted.** The short collects the settled rate; a rule that watched the premium index to time settlements is a different strategy.
- **One Sharpe convention now.** `Registry.dsr` in the shared harness computed its Sharpe with ddof=0 while `Registry.record` and `validate.stats` used ddof=1, so the same stream carried two Sharpes (-25.43 and -25.34 on the first read). Fixed at the source in `research/registry/experiments.py`; every number here is ddof=1 and `check.py` holds the two paths equal to 1e-12. Other projects' stored `untouched.json` files predate the fix and are lower by a factor of sqrt(n/(n-1)) on their n_obs.
- **252-day annualisation** throughout the registry, as in every other project here; crypto trades 365 days, so the calendar Sharpe is 1.2 times the table.
- **Walk-forward purge is in business days.** The harness adds `BDay(horizon)` to bar timestamps, so each fold loses about 16 calendar days of training before its test window rather than 12 bars. Harmless for a rule with no fit, noted because the fold table shows the gap.
- **No engine path.** The premia book's engine runs daily; a 5-minute sleeve does not fit it, so there is no `Sleeve` parity check here and the numbers are the research path only.

## Files

- `data.py`: collectors and the bar panels; `python data.py` fills the lake (about six minutes)
- `basis.py`: the index, the z feature, `BasisMR`, the exceedance, reversion and venue-gap functions
- `run.py`: everything above, about 16 minutes, deterministic; writes `reports/`
- `check.py`: 24 offline checks on planted series, exits 1 on failure
- `reports/summary.json`, `basis_distribution.csv`, `basis_by_year.csv`, `exceedance.csv`, `reversion.csv`, `walk_forward.csv`, `venue_monitor.csv`, `untouched.json`, `untouched_first_read.json`, `untouched_returns.csv`, `registry/`, `models/`, `basis.png`
