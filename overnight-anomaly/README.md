# Overnight vs intraday returns

Kelly and Clifton (2016) and Lou, Polk and Skouras (2019) report that the
equity premium is earned almost entirely while the market is closed. Split each
day's close-to-close return into an overnight leg (prior close to today's open)
and an intraday leg (open to close), and the intraday leg is roughly zero. This
repo tests that on yfinance daily OHLC for SPY, QQQ, IWM, the nine original
sector SPDRs and twelve large caps that have traded since 2000, then asks
whether anyone can actually trade it.

Short version: the decomposition holds, is statistically strong on the index
and sector ETFs, and is worth nothing after costs on every asset but one.

## Design

Returns are simple, per day, from adjusted OHLC:

```
overnight_t = Open_t / Close_{t-1} - 1
intraday_t  = Close_t / Open_t - 1
(1 + overnight_t)(1 + intraday_t) = 1 + close_to_close_t
```

`run.py` confirms the identity holds to 2.2e-16 across every asset-day.

yfinance's `auto_adjust=True` scales Open and Close by the same factor on a given
day, so the two ratios are unaffected. It does mean dividends land in the
overnight leg, because the ex-dividend gap happens between one close and the
next open. That is a fair reflection of what an overnight holder receives, but
it means part of the ETF "overnight premium" is just the yield, and I size that
below.

**Significance.** Each leg's mean daily return is tested against zero with a
Newey-West t-stat (5 lags). Annual return is compounded, not arithmetic.

**Formation and evaluation.** The papers were written on data ending around
2014 to 2017. I split at the end of 2015: 2000-2015 is the period the finding
was discovered on, 2016-2026 is post-publication and is the honest test of
whether it survived being known. A post-2010 cut is reported separately because
the brief asked for it.

**Look-ahead.** There is no signal, so there is nothing to lag. Holding
overnight means buying at the close of t-1 and selling at the open of t, which
is a fixed rule that uses no information. The only place a future leak could
enter is the data itself, and that is the stale-open problem below.

**Costs.** The overnight-only strategy is two trades a day, every day, forever.
Net daily return is `(1 + overnight)(1 - c)^2 - 1` for a one-way cost `c`. The
breakeven one-way cost is where that equals zero, `c = 1 - (1 + m)^{-1/2}` for
mean daily return `m`, about half the daily edge. Buy and hold pays nothing.

**Data hygiene.** Yahoo has stretches where the open field is the previous
close copied forward, which zeroes the overnight leg and dumps the whole move
into intraday. Any ticker-year where more than 25% of opens are an exact match
to the prior close is dropped. That only fired on KO 2000-2001 (61% and 44%
stale), 500 days. Every other ticker is under 6% stale, which is consistent
with a quiet large cap genuinely opening unchanged a few times a month.

**Survivorship.** The twelve stocks are names I know traded continuously from
2000 to today, so they are survivors by construction. GE and INTC are in the
list precisely because they are survivors that did badly, and the stock panel
should be read as illustrative, not as a cross-section. The ETFs have no
survivorship problem.

## Results

All numbers from `run.py`. Data runs 2000-01-03 to 2026-08-31, 6,704 trading
days.

### The decomposition, full sample, no costs

| | Overnight /yr | Intraday /yr | Total /yr | Overnight t | Intraday t |
|---|---|---|---|---|---|
| SPY | +7.1% | +1.1% | +8.3% | 3.77 | 0.86 |
| QQQ | +11.4% | -2.5% | +8.6% | 4.71 | 0.02 |
| IWM | +13.4% | -4.1% | +8.8% | 5.49 | -0.62 |
| EW 9 sector SPDRs | +10.1% | -0.7% | +9.2% | 5.12 | 0.14 |
| EW 12 large caps | +6.7% | +6.0% | +13.0% | 3.53 | 2.69 |

On the ETFs the finding is intact. SPY compounds at 8.3%/yr close to close,
and 7.1 of that is earned while the market is shut. QQQ and IWM go further: the
intraday leg is negative over 26 years. Overnight t-stats are 3.8 to 5.5;
intraday t-stats are all inside plus or minus one. Overnight also carries less
risk: SPY's overnight leg has 11.2% annual vol against 15.5% intraday, so the
overnight Sharpe is 0.67 and the intraday Sharpe is 0.15.

Across all 24 assets, overnight out-earns intraday on 19, overnight t > 2 on
18, intraday t > 2 on 6 and intraday t < -2 on none.

The large caps are where it gets messy. AAPL, MSFT, JPM, GE, INTC and PFE look
like the ETFs. AMZN, XOM, JNJ, WMT and above all PG go the other way: PG earns
-7.2%/yr overnight (t = -2.80) and +14.8%/yr intraday (t = 4.96). Twelve names
are far too few to say anything about the cross-section, and I do not try to.
What they do show is that the effect is a market-level phenomenon that
individual stocks can sit on either side of, which is consistent with Lou, Polk
and Skouras finding that overnight and intraday returns are driven by different
clienteles.

Dividends explain some but not most of the ETF overnight leg: SPY's average
yield over the sample is 1.74%/yr, which is 24% of its 7.1% overnight return.
For QQQ it is 6% and for IWM 9%.

### Stability by period

SPY:

| Period | Overnight /yr | Intraday /yr | Overnight t | Intraday t |
|---|---|---|---|---|
| 2000-2015 (pre-publication) | +5.5% | -1.4% | 2.47 | 0.00 |
| 2016-2026 (post-publication) | +9.6% | +5.0% | 2.87 | 1.61 |
| 2010-2026 | +8.9% | +5.0% | 3.61 | 2.01 |

QQQ: overnight +10.6% then +12.6%; intraday -8.1% (t -0.90) then +6.5% (t 1.63).
IWM: overnight +12.2% then +15.3%; intraday -4.3% then -3.9% (t -0.47).
Equal-weight sectors: overnight +11.4% then +8.2%; intraday -4.2% then +4.9%.

Two things happened after publication. The overnight leg did not weaken at
all, and its t-stats held up on ten years of fresh data. But the intraday leg
stopped being zero on SPY, QQQ and the sectors: it went from negative to
+5%/yr, with t-stats around 1.6 to 2.0. Post-2010, SPY's intraday leg is
significant at the 5% level. The strong form of the claim, that intraday
returns are nothing, is a 2000-2015 fact. The weak form, that overnight earns
more per unit of risk, is still true everywhere. IWM is the exception where the
strong form survives.

By calendar year on SPY (`reports/spy_by_year.csv`), overnight beats intraday
in 18 of 27 years, overnight is positive in 22, intraday in 18. The year-level
t-stats are almost never above 2 on either side, which is what 250 daily
observations at 1% vol buys you. 2000 stands out with overnight +25.5% and
intraday -27.4%, and I do not fully trust Yahoo's SPY opens that far back;
starting in 2002 instead moves the full-sample overnight number from 7.1% to
7.4% and intraday from 1.1% to 2.4%, which does not change the reading.

`reports/spy_rolling_t.png` shows the 3-year rolling t-stat. Overnight sits
above 2 for most of the sample except 2001-2003 and 2008-2009; intraday
oscillates around zero and spends 2020-2024 near or above 2.

### The cost reality

Net annual return of holding only the overnight leg, at a one-way cost per
trade, full sample:

| One-way cost | 0 | 0.25bps | 0.5bps | 1bp | 2bps | 5bps | Buy and hold | Breakeven |
|---|---|---|---|---|---|---|---|---|
| SPY | +7.1% | +5.8% | +4.5% | +1.9% | -3.2% | -16.8% | +8.3% | 1.49bps |
| QQQ | +11.4% | +10.0% | +8.6% | +5.9% | +0.7% | -13.4% | +8.6% | 2.34bps |
| IWM | +13.4% | +12.0% | +10.6% | +7.9% | +2.6% | -11.8% | +8.8% | 2.68bps |

Two trades a day at 1bp each is roughly 5%/yr of drag, and 5%/yr is most of
the edge. SPY's breakeven one-way cost is 1.5bps, QQQ's 2.3, IWM's 2.7. For
reference, SPY's quoted spread is about 1 cent on a $600 price, 0.2bps, so
crossing the spread twice a day on SPY at the close and open auctions is
survivable in principle. Everything else in the universe has a wider spread, and
the open auction in particular is where a retail order pays up.

The right comparison is not against zero but against buy and hold, since
holding overnight only means giving up the intraday leg for free. On that
test:

| One-way cost | 0 | 0.25bps | 0.5bps | 1bp | 2bps | 5bps |
|---|---|---|---|---|---|---|
| Assets where overnight-only beats buy and hold | 11 of 24 | 10 | 7 | 2 | 1 | 0 |

At zero cost the overnight-only strategy beats buy and hold on fewer than half
the assets, because on the large caps the intraday leg is often positive. At
1bp one-way it beats buy and hold on two. At 2bps, one: IWM.

Post-publication, at 1bp one-way:

| 2016-2026 | Overnight gross | Overnight net | Buy and hold |
|---|---|---|---|
| SPY | +9.6% | +4.2% | +15.1% |
| QQQ | +12.6% | +7.1% | +19.9% |
| IWM | +15.3% | +9.6% | +10.8% |

Holding SPY overnight only, at a cost most people cannot get, would have
returned 4.2%/yr against 15.1% for doing nothing. The anomaly is real as a
description of where returns come from. As a strategy it loses to the index it
is built from, on every asset but IWM, and IWM only wins because its intraday
leg happens to have been negative for 26 years.

The intraday-only strategy is worse everywhere: SPY breakeven is 0.46bps,
QQQ 0.01bps, IWM zero. Buying the open and selling the close is a way to pay
two spreads for a leg with no mean.

### Where it is real and where costs erase it

Real: the ETF-level decomposition, in and out of sample. Overnight carries the
premium at lower vol, with t-stats that stayed above 2.5 for a decade after
publication.

Erased: any attempt to harvest it by trading. Breakeven costs are 1.5 to 2.7bps
one-way on the liquid ETFs, and the strategy has to beat buy and hold, not
zero, which it fails to do at any realistic cost on SPY and QQQ. The usable
version of this finding is not a strategy but a fact about timing: if you are
going to rebalance anyway, the evidence says do it at the open, not the close,
and if you are going to hedge intraday, the cost of the hedge is not lost
premium.

## Files

- `data.py` universe, download and per-ticker cache, stale-open detection, dividend yields
- `decompose.py` the two legs, Newey-West test, single-leg strategy with costs, breakeven
- `check.py` 25 offline checks on synthetic OHLC with a planted overnight drift
- `run.py` full pipeline, tables and charts to `reports/`, console copy in `reports/run_log.txt`

Charts in `reports/`: `cumulative_spy.png`, `cumulative_qqq.png`,
`cumulative_iwm.png`, `by_asset.png`, `spy_by_year.png`, `spy_rolling_t.png`,
`cost_curve.png`. Tables: `by_asset.csv`, `by_period.csv`, `spy_by_year.csv`,
`cost_sensitivity.csv`, `data_hygiene.csv`.

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is fully offline. `run.py` downloads once to
`../source-material/overnight-anomaly/` and is offline after that. No
dependencies beyond the shared `requirements.txt`.

## Limitations

**Yahoo opens.** The whole study rests on the open print, which is the least
reliable field in free daily data. I filter exact copies of the prior close and
report the stale fraction per ticker, but a slightly wrong open that is not an
exact copy passes the filter. The 2000 SPY year looks off and I have not been
able to verify it against another source. Official auction prices from the
exchange would be the fix.

**The open is not a tradeable price.** The daily open is the opening auction
print. A market order into the open auction gets that price plus whatever the
auction imbalance does, and a limit order might not fill. The same goes for the
close. My cost grid starts at zero and the breakeven is under 3bps, so the
result is sensitive to exactly the execution detail I cannot model from daily
data.

**Dividends sit in the overnight leg.** Sized above at 24% of SPY's overnight
return. An overnight-only holder does receive them, so this is not an error, but
anyone reading the overnight premium as a pure price effect should subtract it.

**No cross-section.** Twelve hand-picked survivors is not a test of Lou, Polk
and Skouras' main result, which is about the cross-sectional persistence of
overnight versus intraday returns across thousands of stocks. I test the
market-level claim only.

**No explanation tested.** The candidates in the literature are clientele
differences, the overnight premium as compensation for holding through the
period when you cannot trade, and short-sale or margin constraints that bind
differently by session. I measure the effect and do not try to attribute it.

**Newey-West with a fixed 5 lags.** Daily returns have little autocorrelation,
so the choice barely matters, but I did not run a bandwidth selection rule, and
the t-stats on the calendar-year table are on 250 observations and should be
read as noise.
