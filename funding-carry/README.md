# Crypto funding rate carry

Perpetual futures never expire, so exchanges tether them to spot with a funding
payment every eight hours. When the perp trades above spot, longs pay shorts.
Long the spot and short the perp and you collect that payment while carrying no
view on the price. This measures how big the payment actually was on Binance
from 2020 to 2026, and what is left of it after fees, margin and the days the
trade breaks.

Two results are worth the whole project.

The first is that most of the carry is not a market premium at all. Binance's
funding formula is `premium + clamp(0.01% - premium, ±0.05%)`, so whenever the
perp trades within five basis points of spot the rate snaps to a hard-coded
0.01% per eight hours, which is 10.95% a year. That happens on 34% of all
settlements. BTCUSDT averaged 11.86% annualised over six and a half years, so
92% of the average carry is that constant and 0.91% a year is what the market
was genuinely paying.

The second is that the trade stopped beating cash. Net of costs the BTCUSDT
carry earned 12.6% over T-bills in 2020 and 23.7% in 2021. Since then: +0.7%,
+0.3%, +3.6%, -0.7%, and -3.1% in 2026 to July. It is not that the strategy
broke. It is that the funding rate fell to roughly the T-bill rate at the same
time the T-bill rate went to 5%, and being paid the risk-free rate to take
exchange, liquidation and stablecoin risk is not a trade.

## Data

Binance's REST endpoints (`fapi.binance.com`, `api.binance.com`) return
`"Service unavailable from a restricted location"` from a US IP. The monthly
archives at `data.binance.vision` are the same exchange data, are not blocked,
and are what this uses:

- `futures/um/monthly/fundingRate/{sym}/` realised 8h funding settlements
- `futures/um/monthly/klines/{sym}/8h/` USD-M perpetual bars
- `spot/monthly/klines/{sym}/8h/` spot bars
- `spot/monthly/klines/USDCUSDT/8h/` for the stablecoin question
- `^IRX` from Yahoo for the 3-month T-bill

BTCUSDT and ETHUSDT, 2020-01-01 to 2026-07-31, 7,211 funding periods each, every
one at an 8-hour interval. The archives start at 2020-01 for both perpetuals and
the 2026-08 spot file was not published yet, which sets both ends of the window.
Everything is cached to `source-material/crypto/` as CSV, so only the first run
touches the network.

Two things about the raw files that will silently corrupt a naive loader.
Timestamps switched from milliseconds to microseconds partway through 2025, in
the same column, so a fixed `unit="ms"` puts half the history in the year 58000.
And funding files carry a header row while kline files do not.

Funding settling at time T is joined to the bar spanning `[T-8h, T)`, whose
close is the price at T and whose high is the worst point a short passed through
during the period.

## Funding

| | BTCUSDT | ETHUSDT |
|---|---|---|
| mean 8h | 1.08 bps | 1.28 bps |
| median 8h | 0.94 bps | 1.00 bps |
| std 8h | 2.11 bps | 2.75 bps |
| annualised, compounded | 12.59% | 15.08% |
| share negative | 14.3% | 13.9% |
| exactly at the 0.01% floor | 34.1% | 34.3% |
| 1st percentile | -1.70 bps | -1.64 bps |
| 99th percentile | 10.62 bps | 13.57 bps |

Annualised by year, against the cash rate over the same year:

| year | BTC funding | BTC % neg | ETH funding | ETH % neg | 3m T-bill |
|---|---|---|---|---|---|
| 2020 | 18.79% | 14.2% | 31.55% | 2.6% | 0.34% |
| 2021 | 35.79% | 7.3% | 45.53% | 4.1% | 0.03% |
| 2022 | 4.25% | 22.1% | 0.79% | 34.2% | 2.00% |
| 2023 | 8.18% | 10.1% | 8.61% | 9.1% | 5.05% |
| 2024 | 12.66% | 8.4% | 13.84% | 4.2% | 4.95% |
| 2025 | 5.26% | 12.9% | 5.05% | 16.2% | 4.05% |
| 2026 to Jul | 1.95% | 32.7% | 0.97% | 36.8% | 3.62% |

2021 against 2023 is a factor of four. 2021 against 2026 is a factor of eighteen.
On a simple, uncompounded basis (the table above compounds) 2021 funding was
30.61% and 2026 was 1.94%. Split against the fixed 10.95% interest term, the
market premium was +19.66% in 2021 and -9.01% in 2026, so a third of the way
through this sample the perp stopped paying a premium and started charging one.

Funding is highly persistent: autocorrelation 0.80 at one period, 0.70 at a day,
0.47 at a week, 0.33 at a month. Regressed on trailing 30-day spot momentum and
30-day realised vol with Newey-West errors at 90 lags, both load positive and
significant (momentum t = 6.2, vol t = 2.3 for BTC) for an R² of 0.24. Funding
is a leveraged-long-positioning gauge, which is why it peaks in bull markets and
inverts in liquidation cascades.

## Basis

The premium is tiny and, at the settlement snapshot, usually negative: BTCUSDT
mean -1.5 bps, median -3.8 bps, positive only 26.7% of the time. That is not a
contradiction with positive funding, it is the mechanism. Because the formula
pays 1 bp per period at zero premium, arbitrageurs push the perp to a small
discount until the total cost of being long is fair. The carry is being paid by
Binance's hard-coded interest assumption, and the market claws part of it back
through the basis.

Pushing the observed basis through the exchange's own formula reproduces the
funding it charged: correlation 0.67 for BTC and 0.62 for ETH, regression slope
0.57 on a HAC fit. It is not 1.0 because funding is set on the time-weighted
average premium over the eight hours, and one closing snapshot is a noisy
estimate of that average. `reports/basis_vs_funding.png` shows the cloud sitting
on the 45-degree line with the clamp visible as a dense bar at 1 bp.

## The trade

Capital is 1. It splits into spot notional and a perp margin buffer of 50% of
that notional, so 1 unit of capital carries 0.67 units of the trade. Both legs
are resized to current equity every 30 days, and sooner if the margin account
has burned 60% of the buffer it started the cycle with. Costs are Binance retail
taker fees with no VIP tier or BNB discount: 10 bps spot, 5 bps perp, plus 1 bp
of slippage per fill. Liquidation is tested against the high inside each period.

| | BTCUSDT | ETHUSDT |
|---|---|---|
| total return, 6.6y | +70.2% | +88.6% |
| annualised, net | 8.41% | 10.11% |
| annualised, gross | 8.68% | 10.44% |
| annual vol | 1.03% | 1.16% |
| Sharpe, net, raw returns | 7.86 | 8.35 |
| Sharpe, net, over 3m T-bill | 5.09 | 5.86 |
| max drawdown | -1.13% | -1.22% |
| funding collected | 0.728 | 0.923 |
| price P&L | -0.003 | -0.007 |
| fees paid | 0.022 | 0.030 |
| skew | +1.69 | +2.61 |
| excess kurtosis | 537 | 51 |

The P&L decomposition is the point: 73 cents of funding on a starting dollar,
against a third of a cent of price drift and two cents of fees. The hedge works.
Over 7,211 periods the two legs cancelled to within 0.7 cents on the dollar.

Do not read the Sharpe of 7.9 as a quality score. It is computed on raw
returns; the position ties up cash, and against the T-bill rate that cash would
have earned it drops to 5.1. It is marked every eight hours on a position whose
volatility is almost entirely basis noise, the excess kurtosis is 537, and the
worst single period, 21 December 2020, was 37 standard deviations away. A
statistic that assumes the second moment describes the distribution does not
describe this one.

By year, net of costs, against the cash the position ties up:

| year | BTC carry | ETH carry | T-bill | BTC over cash | ETH over cash |
|---|---|---|---|---|---|
| 2020 | +12.95% | +21.03% | +0.34% | +12.62% | +20.70% |
| 2021 | +23.71% | +30.16% | +0.03% | +23.67% | +30.13% |
| 2022 | +2.69% | +0.47% | +2.00% | +0.69% | -1.53% |
| 2023 | +5.33% | +5.61% | +5.05% | +0.29% | +0.57% |
| 2024 | +8.53% | +9.34% | +4.95% | +3.58% | +4.39% |
| 2025 | +3.37% | +3.10% | +4.05% | -0.68% | -0.95% |
| 2026 to Jul | +0.51% | +0.21% | +3.62% | -3.10% | -3.41% |

Fees barely matter, which is the one comfortable finding here. This is a
buy-and-hold position rebalanced monthly, so the entry cost amortises over
years:

| spot / perp fee | BTC annual | BTC Sharpe | ETH annual | ETH Sharpe |
|---|---|---|---|---|
| 0 / 0 bps | 8.65% | 8.08 | 10.40% | 8.58 |
| 5 / 2 bps | 8.54% | 7.98 | 10.27% | 8.47 |
| 10 / 5 bps | 8.41% | 7.86 | 10.11% | 8.35 |
| 20 / 10 bps | 8.18% | 7.60 | 9.83% | 8.07 |
| 40 / 20 bps | 7.72% | 7.00 | 9.22% | 7.38 |

Quadrupling the fee assumption costs 69 bps a year. A pairs book turning over
nine times a year would be dead; this one is not fee-constrained, it is
funding-constrained.

Turning the trade off when the trailing 7-day funding mean is negative makes it
worse: BTC goes from 8.41% to 7.23% and pays 12.7% of capital in fees over the
sample instead of 2.2%. Funding is persistent enough to look forecastable and
mean-reverting enough that acting on it at this frequency just buys round trips.

## What actually kills it

**The short leg gets liquidated.** This is the real risk and it is a function of
one number, the margin buffer.

| buffer | BTC annual | BTC max DD | liquidated | equity then |
|---|---|---|---|---|
| 10% | -1.06% | -12.41% | 2020-03-13 | 0.93 |
| 20% | +6.63% | -4.04% | 2023-10-24 | 1.53 |
| 30% | +9.31% | -1.44% | survived | |
| 50% | +8.41% | -1.13% | survived | |
| 100% | +6.45% | -1.12% | survived | |

A 10% buffer died on the 13 March 2020 bounce. The book had run up to 1.065 on
the crash, because the perp fell to a 74 bp discount to spot and the short leg
booked it. Eight hours later spot printed a high 10.6% above the last rebalance,
the margin was gone, and unwinding the now-naked spot leg at the close cost
about another 3%. That is 12.0% of capital in one eight-hour period and 12.4%
from the peak two periods earlier. A 20%
buffer survived that and died three and a half years later on the October 2023
ETF-approval rally, handing back 4% of a 59% cumulative gain instantly. Note the
non-monotonicity: 30% beat 50% beat 100%, because past the point where you stop
getting liquidated, extra margin is just idle capital.

The buffer is not the only variable. Binance liquidates on a mark price built
from a spot index, not on its own order book, and during a squeeze the perp
prints far above the index. On 26 July 2021 the BTCUSDT perp wicked to 48,168
while spot high was 39,800 and the perp closed at 38,157. Marking on the perp
instead of the index kills every BTCUSDT buffer up to 50%:

| buffer | mark = spot index | mark = perp print |
|---|---|---|
| 20% | liquidated 2023-10-24 | liquidated 2021-07-26 |
| 30% | survived | liquidated 2021-07-26, -17.5% |
| 50% | survived | liquidated 2021-07-26, -21.0% |
| 100% | survived | survived |

The index-marked column is what Binance actually does, so the headline results
use it. The perp column is what happens on an exchange with a worse mark, or if
you are marked by a broker who is looking at last trade.

**Funding goes negative for a long time.** The longest unbroken negative run was
24 periods (8 days) for BTC starting 12 March 2020, costing 1.04% of notional,
and 25 periods for ETH starting 8 September 2022, costing 1.35%. Worse than the
runs is the sag: the worst trailing 30-day window annualises to -15.96% for BTC
and -21.63% for ETH, and the worst trailing 90-day to -6.81% for ETH. There is
no stop-loss that helps, because the position is not losing money on price.

**The 2022 deleveraging.** The mechanical answer is that funding did not go
deeply negative for long; it went flat and stayed flat. BTC funding fell from
35.79% in 2021 to 4.25% in 2022 with 22.1% of settlements negative, and ETH from
45.53% to 0.79% with 34.2% negative. Through the individual episodes the carry
was small but not disastrous:

| episode | BTC funding, ann. | BTC % neg | BTC carry P&L | ETH funding, ann. | ETH % neg | ETH carry P&L |
|---|---|---|---|---|---|---|
| Mar 2020 covid crash | -15.03% | 58.3% | -0.29% | +8.41% | 6.9% | +0.32% |
| May 2021 China ban | +3.79% | 30.0% | +0.33% | +7.73% | 14.0% | +1.04% |
| Jun 2022 3AC / Celsius | +5.29% | 17.5% | +0.56% | +1.98% | 33.9% | +0.19% |
| Nov 2022 FTX | +1.37% | 24.6% | +0.14% | +1.21% | 35.1% | +0.14% |

March 2020 was the violent one and 2022 was the slow one. The way 2022 hurt was
not a loss, it was the disappearance of the reason to be in the trade: 2.69% on
BTC for a year of exchange risk, while T-bills paid 2.00% and were about to pay
5%. The carry never recovered to 2021 levels and, apart from 2024, never
meaningfully beat cash again.

**Tails.** The return distribution is nothing like normal.

| | BTCUSDT | ETHUSDT |
|---|---|---|
| skew | +1.69 | +2.61 |
| excess kurtosis | 537 | 51 |
| worst 8h | -1.13% | -0.45% |
| worst day | -1.04% | -0.67% |
| worst month | -0.83% | -1.18% |
| 1% quantile | -4.2 bps | -5.7 bps |
| 1% quantile under a normal | -6.5 bps | -7.3 bps |
| periods beyond 5 sd | 0.28% | 0.62% |

The 1% quantile is *smaller* than a normal would predict, and 0.28% of periods
land beyond five standard deviations against the 0.00006% a normal allows. That
is the shape: most of the time the book grinds out well under a basis point (the
median 8h return is 0.4 bps), and the risk lives entirely in a handful of
periods that a variance-based measure cannot see.
`reports/tails.png` is the picture.

**Stablecoin depeg.** The whole position is denominated in USDT, on both legs
and in the margin. USDC/USDT on Binance, restricted to bars with meaningful
volume, has a median close of 0.9999 but ranges from 0.9128 to 1.0203. The worst
deviation was 8.72% on 11 March 2023, the weekend Silicon Valley Bank failed and
USDC broke first; 0.26% of 8-hour bars are more than 0.5% off par and 0.08% are
more than 2% off. An 8.7% move in the unit of account is six years of this
strategy's excess return over cash, and it says nothing about USDT itself, only
about the pair.

**Exchange counterparty risk.** Unquantified, and the largest of them. The entire
position, both legs plus the margin, sits inside one venue. FTX paid competitive
funding until the week it did not return anyone's collateral. There is no number
in this repo for that and I do not think an honest one exists.

## Limitations

**One exchange, two symbols.** Binance is the deepest perp venue, which is
exactly why its funding is likely the lowest available. A cross-venue study
would find higher rates on smaller exchanges, and the right reading of that is
that the extra yield is compensation for worse counterparty risk, not alpha.
This study cannot separate those.

**Marked every eight hours.** Everything between settlements is invisible.
Liquidation is checked against the period high, but the volatility, drawdown and
Sharpe figures are computed on 8-hour closes, so they understate what a
continuously marked book would show. This flatters the Sharpe.

**Funding notional uses the perp close, not the mark price.** Binance charges
funding on the mark price at settlement, which is the index plus a smoothed
basis. The difference against the perp close averages a couple of basis points
of a payment that is itself a basis point, so it is third-order, but it is an
approximation.

**Idealised fills.** Every rebalance transacts at the close with 1 bp of
slippage and no market impact. A real position of size would move a perp book
during exactly the stress periods when it most needs to rebalance, which is when
slippage stops being 1 bp.

**No borrow or financing on the spot leg.** The spot is bought outright with
cash, which is the retail-accessible version. It also means the strategy ties up
1.5 units of capital per unit of trade. A prime-brokered or portfolio-margin
version would need much less capital and would show a very different return on
equity, at the cost of more ways to get liquidated.

**Survivorship in the symbol choice.** BTC and ETH were picked because they have
the longest continuous history, which is the same thing as picking the two
perpetuals that never got delisted. The result for a random 2021-vintage
altcoin perp, which paid far higher funding and then stopped existing, is not in
here.

**The liquidation model is a simplification.** It closes the short at the
maintenance-margin threshold, zeroes the margin account and unwinds the spot at
the next close. Real liquidation is partial, tiered by position size, charges a
liquidation fee and can go through into an insurance fund. The model is roughly
right in direction and understates the cost.

## Files

- `data.py` monthly archive downloads, timestamp-unit handling, funding/spot/perp join, caching
- `carry.py` funding statistics, the basis and the exchange's formula, the delta-neutral simulator, metrics
- `check.py` offline checks on synthetic panels with hand-computed answers
- `run.py` the full study, tables and charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline. `run.py` downloads about 400 monthly archives on the
first run (roughly four minutes) and reads the cache afterwards.
