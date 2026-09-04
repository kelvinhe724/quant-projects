# Cross-exchange price gaps in BTC and ETH: how often, how big, how long, and what survives fees

Four spot venues polled together every three seconds for about an hour, plus a
month of hourly and two years of daily candles from two of them. The question
is whether the price of the same coin differs across venues by more than it
costs to trade it, and if so for how long.

The short answer: the books are crossed by a basis point or two almost all the
time, and by more than five basis points almost never. Nothing in the window
clears a round trip of taker fees at any tier a person can actually get.

## Results

Window: 2026-09-04 03:33:55Z to 04:28:54Z, 55.0 minutes, 1,099 ticks, 8,792 quotes,
zero failed requests. Within-tick timestamp skew across venues: median 187 ms,
p95 215 ms, max 261 ms. Coinbase answered in 36 ms median, Binance.US in 230 ms.
Kraken's USDT/USD mid sat between 0.99992 and 0.99996 the whole time.

**Best executable gap per sample, bps** (the largest sell-bid-minus-buy-ask
across the twelve ordered venue pairs):

| | median | p95 | p99 | max | share > 0 | share > 20 (pro round trip) | share > 50 |
|---|---|---|---|---|---|---|---|
| BTC | 1.66 | 3.74 | 4.94 | 5.78 | 100% | 0% | 0% |
| ETH | 0.97 | 2.31 | 2.98 | 7.90 | 99.4% | 0% | 0% |

The share above zero is not a free lunch: it is the max of twelve numbers that
hover around zero, and it measures how often one venue's tick moved a fraction
of a second before another's. The mid-to-mid spread between any two venues had
a median absolute value of 0.4 to 1.4 bps and a 95th percentile of 1.3 to 3.6 bps.
No pair was ever more than 5 bps apart on mids.

**Share of samples where the best gap beats twice a flat per-leg fee:**

| fee per leg, bps | 0.5 | 1 | 1.5 | 2 | 2.5 | 5 | 10 | 40 |
|---|---|---|---|---|---|---|---|---|
| BTC | 79% | 34% | 11% | 3.7% | 0.3% | 0% | 0% | 0% |
| ETH | 48% | 9.5% | 1.0% | 0.4% | 0.3% | 0% | 0% | 0% |

Breakeven is somewhere around 2 bps per leg. The cheapest published taker fee
on these venues is OKX's 10 bps; Coinbase's lowest tier is 60. So the fraction
of observed gaps that exceed round-trip fees, at any fee a person can get, is zero.

**Persistence** (episodes of the best gap above a threshold, 3-second samples):

| threshold | BTC share of time | BTC episodes | median / p90 / max | ETH share | ETH episodes | median / p90 / max |
|---|---|---|---|---|---|---|
| > 2 bps | 34% | 60 | 12 s / 37 s / 186 s | 9.5% | 52 | 3 s / 12 s / 30 s |
| > 5 bps | 0.3% | 2 | 4.5 s / 6 s / 6 s | 0.3% | 3 | 3 s / 3 s / 3 s |
| > 10 bps | 0 | 0 | | 0 | 0 | |

Anything worth more than 5 bps lasted one or two samples. A BTC deposit takes
tens of minutes to confirm.

**Simulations** ($10,000 lots, 5 lots of cash and coin per venue, signal at
t, fill at t+1). The minimum-edge grid on the formation half was degenerate at
pro fees, no threshold produced a single signal, so the chosen edge was 0 and
the evaluation half also produced nothing:

| scenario | BTC trades | BTC net | ETH trades | ETH net |
|---|---|---|---|---|
| pro fees (10 bps/leg), prepositioned, eval half | 0 | $0 | 0 | $0 |
| retail fees, prepositioned, eval half | 0 | $0 | 0 | $0 |
| pro fees, coin must transfer, eval half | 0 | $0 | 0 | $0 |
| zero fees, prepositioned, full window | 96 (1,003 blocked) | +$107 | 240 (852 blocked) | +$131 |

The zero-fee row is the upper bound on what the gaps were worth: about 1.1 bps
a trade on BTC and 0.5 bps on ETH, with 13 and 66 losing fills respectively
because the gap had closed by t+1. Even with no fees, five lots a side ran out
almost immediately: 91% of BTC signals and 78% of ETH signals were blocked for
lack of coin at the rich venue or cash at the cheap one, because the direction
is persistent.
Binance.US was the rich venue and OKX the cheap one for most of the BTC
window, which is as likely to be the USDT conversion and Binance.US's thinner
book as anything tradeable.

**Longer horizon**, Coinbase minus Kraken candle closes, bps:

| | n | median abs | p95 abs | max abs | share abs > 20 | AR(1) |
|---|---|---|---|---|---|---|
| BTC hourly, 30 days | 720 | 0.90 | 3.16 | 6.39 | 0% | 0.10 |
| ETH hourly, 30 days | 720 | 0.97 | 3.89 | 8.99 | 0% | 0.04 |
| BTC daily, 2 years | 720 | 1.50 | 6.05 | 10.68 | 0% | 0.05 |
| ETH daily, 2 years | 720 | 1.54 | 6.37 | 16.36 | 0% | 0.05 |

The close gap has essentially no autocorrelation (half-life well under one
period), which says the two venues are one market at the hourly scale and the
close-to-close differences are which venue printed last before the boundary.
The 2-year daily max of 16 bps on ETH is the widest thing in the whole study,
and it is still below a single Coinbase taker fee. Closes on two venues are not
simultaneous, so these are noisier than the live panel, not cleaner.

## Data

Public REST tickers, no keys. I wanted Binance, Coinbase, Kraken, Bybit and OKX.
Binance.com returns HTTP 451 (restricted location) and Bybit's CloudFront edge
returns 403 from a US address, so the panel is:

| Venue | Endpoint | Pair | Note |
|---|---|---|---|
| Coinbase | `api.exchange.coinbase.com/products/{sym}-USD/ticker` | BTC-USD, ETH-USD | |
| Kraken | `api.kraken.com/0/public/Ticker` | XBT/USD, ETH/USD, USDT/USD | one call for all three |
| Binance.US | `api.binance.us/api/v3/ticker/bookTicker` | BTCUSD, ETHUSD | US affiliate, thinner than Binance.com |
| OKX | `okx.com/api/v5/market/ticker` | BTC-USDT, ETH-USDT | converted to USD at Kraken's USDT/USD mid |

OKX has a BTC-USD book but it trades about 4% of the volume of its USDT book,
so I sample the liquid one and convert, storing the raw USDT price and the
conversion rate in the CSV so the choice can be undone. Every tick fires all
requests concurrently from a thread pool, stamps each with the local
send/receive midpoint, and appends to `source-material/crypto-arbitrage/quotes.csv`.
Failed requests are logged and skipped; the loop stops cleanly on the timer or
Ctrl-C.

Longer horizon: hourly candles for 30 days and daily candles for 720 days from
Coinbase (paged, 300 per call) and Kraken (last 720 per call), cached as CSV.

## Method

**Alignment.** `arb.align` puts every venue on a 3-second grid using its last
quote at or before each grid time (`merge_asof`, backward). No quote from after
the grid time is ever used. A venue silent for more than 6 seconds drops the
row. Each row also carries the age of every quote, so the sync error is
measurable rather than assumed.

**Two spreads.** The mid-to-mid spread is the descriptive one:
`(mid_a - mid_b) / avg_mid` in bps, signed. The executable gap is the one you
can trade: `(bid_a - ask_b) / ask_b`, sell at a's bid, buy at b's ask, for every
ordered pair. Its maximum across the twelve ordered pairs is the "best gap" per
sample. A negative best gap means the books overlap normally and crossing them
costs money.

**Persistence.** Run lengths of the best gap above 0, 2, 5, 10, 20 and 50 bps,
in samples, times the 3-second interval.

**Fees.** Two taker fees, one per leg. Retail tier as I read the fee pages in
September 2026: Coinbase 60, Kraken 40, Binance.US 40, OKX 10 bps. "Pro" is a
flat 10 bps per leg. The breakeven sweep asks, for each flat per-leg fee from
0 to 60 bps, what share of samples has a best gap above twice that fee.

**Simulation.** Signal on sample t when the best gap beats both fees plus a
minimum edge; fill at sample t+1's quotes; $10,000 lots. Prepositioned mode
starts with five lots of cash and five lots of coin at every venue, and a trade
that cannot be funded on both sides is counted as blocked. Transfer mode starts
with cash only, withdraws the coin after buying (BTC 30 minutes and 0.0002 BTC,
ETH 5 minutes and 0.001 ETH) and sells at the first sample after it lands.

**Formation / evaluation.** The one free parameter, minimum edge over fees, is
chosen from {0, 1, 2, 5} bps on the first half of the window and applied once
to the second half.

`check.py` verifies the alignment on a staggered synthetic feed, detects a
planted 20 bps gap with the right sign, pair and duration, reproduces a hand
fee calculation to the cent, confirms fills come from t+1, caps trades at the
inventory, resolves transfers at the right time, and produces zero trades on a
zero-gap feed. It exits nonzero on any failure.

## Look-ahead audit

The alignment only ever uses quotes stamped at or before the grid time. The
simulator reads the gap at t and fills at the quotes of t+1, so a gap that
blinked for one sample is filled after it closed, which `check.py` confirms by
planting one and asserting the trade loses money. The one tuned parameter is
chosen on the first half and applied to the second. Nothing in the reports
comes from a quantity computed on the row it is scored against.

## Limitations

- **One hour, one regime.** Thursday evening in the US, early Friday UTC, a
  quiet tape. The gaps that
  people remember (May 2021, the FTX week, the March 2023 USDC depeg) happened
  under stress, and none of that is in the window. The daily series widens the
  horizon to two years but on non-synchronous closes, which can only overstate
  the gap, and even that overstatement tops out at 16 bps.
- **Missing the biggest book.** Binance.com and Bybit are geo-blocked from a
  US address, so the panel is the US-reachable market. Binance.US is a
  different, far thinner venue and quoted a wider, jumpier book than the
  other three all night.
- **REST polling at 3 seconds** with a 190 ms p95 skew between venues. The
  crossed slivers I measure at 1 to 2 bps are of the same order as what that
  skew produces on a moving price, so the small end of the distribution is
  partly measurement, not market. WebSocket feeds with exchange sequence
  numbers are the right instrument at sub-second resolution.
- **Top of book only.** The quotes carry no depth, and I assume a $10,000 lot
  fills at the displayed price. The sampler does not store size, but in the
  probes I ran before starting it Binance.US showed 0.0001 to 0.007 BTC at the
  touch, so a $10,000 lot there would have walked the book.
- **OKX is a USDT book** converted at Kraken's USDT/USD mid. That removes most
  of the stablecoin basis but adds Kraken's USDT half-spread to OKX's noise,
  and OKX being the "cheap" venue for most of the BTC window is probably that
  residual rather than an opportunity.
- **Fees are approximate.** The retail table is what the fee pages said when
  I read them; tiers, maker rebates and promotions change. The breakeven sweep
  is there so the conclusion does not depend on any one number: the answer is
  the same at every fee above about 2.5 bps per leg.
- **Survivorship.** Only venues that were up and reachable for the whole hour
  are in the panel, and an exchange outage is exactly when gaps open. There is
  no equivalent of delisting here, but a study that only sees venues on their
  good days will underestimate the tail for the same reason a stock universe
  built from today's index does.
- **The transfer case never fired** because nothing cleared fees to begin with.
  The machinery is tested on synthetic data, not exercised on real data.

## Files

- `data.py` samples the four venues (`python3 data.py 55`) and fetches candles
- `arb.py` alignment, spreads, persistence, simulation
- `check.py` offline planted-truth tests
- `run.py` full pipeline, writes charts and CSVs to `reports/` and `reports/run_log.txt`
- `reports/gaps_{BTC,ETH}.png` venue mids vs average, best gap over time, histogram
- `reports/breakeven_{BTC,ETH}.png` share of samples with an arb, by fee level
- `reports/close_gap_history.png` hourly and daily Coinbase minus Kraken close gap

## How to run

```
python3 data.py 55      # about an hour, writes quotes.csv
python3 check.py
python3 run.py
```
