# G10 FX carry: does uncovered interest parity hold, and what does it cost to bet against it

Borrow the three lowest-yielding G10 currencies, lend the three highest, roll
monthly. Uncovered interest parity says the high-yielders should depreciate by
exactly their rate advantage and the trade should earn nothing. This project
tests that claim directly on 24 years of daily data, then runs the trade and
looks closely at the months where it blew up.

The short version: the data does not behave the way UIP says. The pooled Fama
slope is 0.05 where UIP needs 1, so high-rate currencies do not depreciate to
offset their carry, although with a month-clustered standard error of 0.67 the
regression on its own cannot reject UIP at the 5% level. The trade's P&L is the
sharper test: it collects about 3.2% a year in interest, gives back 0.3% of it
in spot, and compounds to 2.6% a year at a Sharpe of 0.35 before and after
costs. It pays for that with negative skew and a 31.7% drawdown that started in
July 2007 and was not recovered until May 2024.

## Data

All from FRED, no key needed, cached to `../source-material/fx-carry/` on first
run so later runs are offline.

| Currency | Spot (daily, H.10 noon NY) | Quoted as | 3-month rate (monthly) |
|---|---|---|---|
| USD | | | IR3TIB01USM156N |
| EUR | DEXUSEU | USD per EUR | IR3TIB01EZM156N |
| JPY | DEXJPUS | JPY per USD | IR3TIB01JPM156N |
| GBP | DEXUSUK | USD per GBP | IR3TIB01GBM156N |
| CAD | DEXCAUS | CAD per USD | IR3TIB01CAM156N |
| AUD | DEXUSAL | USD per AUD | IR3TIB01AUM156N |
| NZD | DEXUSNZ | USD per NZD | IR3TIB01NZM156N |
| CHF | DEXSZUS | CHF per USD | IR3TIB01CHM156N |
| NOK | DEXNOUS | NOK per USD | IR3TIB01NOM156N |
| SEK | DEXSDUS | SEK per USD | IR3TIB01SEM156N |

The rate series are the OECD Main Economic Indicators 3-month interbank rates
(monthly average, percent per annum) as republished by FRED. I inverted the five
series quoted as foreign per USD so every spot column is USD per one unit of
foreign currency and a rise always means the foreign currency strengthened.

The Japanese rate series starts in April 2002, which fixes the start of the
sample. The euro and sterling rates currently stop at January 2026 and the yen
at May 2026 while the others run to June 2026; a rate is carried forward for at
most 8 months, so the sample ends 2026-06-30. The last five rebalances use a
stale EUR and GBP rate,
and it matters: at the June 2026 rebalance EUR (2.03) sits 8bps above SEK
(1.95) at the edge of the short leg, and GBP (3.71) sits 6bps below USD (3.77)
at the edge of the long leg. Whether those five months held the right
currencies depends on rate moves I do not have. They are 5 of 291 months and I
report them as they are rather than truncating the sample, but nothing in 2026
should be read as precise. One other gap: the USD series has no April 2020
observation, so that month's rank uses the March 2020 rate. USD was top-ranked
either way.

## Design

| Window | Dates | Use |
|---|---|---|
| In-sample | 2002-04-30 to 2015-12-31 | Build the pipeline, diagnostics, cost choice |
| Final test | 2016-01-01 to 2026-06-30 | Run once at the end, untouched |

There is nothing to fit in a top-3 / bottom-3 carry sort, so the split is less
about protecting a fitted parameter and more about checking whether the effect
that was documented on the first window is still there on the second.
Parameters were fixed before any result was looked at: three currencies a leg,
equal weight, monthly rebalance, one-day execution lag, 5bps one-way cost.

**Signal.** On the last trading day of each month, rank the ten currencies by
their 3-month rate for that month. Long the top three at +1/3 each, short the
bottom three at -1/3 each. The USD is ranked like any other currency, so the
book can hold the dollar and end up net long or short foreign currency. A
second construction leaves the USD out of the sort and ranks only the nine
foreign currencies, which forces the foreign notionals to cancel; I report both.

**Returns.** The daily excess return of being long a currency against USD is
its spot change plus the rate differential accrued over the calendar days since
the last quote. Rates are percent per annum on an actual/365 accrual. The USD
column is identically zero, so a USD position is just the absence of a foreign
position and carries no cost.

**Look-ahead.** Weights are stamped with the date the signal was observed and
shifted forward one trading day in `backtest.run`, in one place, so a book set
at the close of month end t first earns the return of the next trading day.
`check.py` confirms the weight earning day t's return is the weight set at t-1
and that a spot spike at t leaves every earlier excess return unchanged.

**One caveat on timing.** The OECD rate for month M is a monthly average, so it
includes the last days of M and is strictly published a little after the month
end where I use it. Because the sort changes rarely (turnover 0.6x a year) the
effect on which currencies are held is small, but it is a look-ahead of a few
basis points in the level of the rate and I am not going to pretend otherwise.

**UIP test.** For each foreign currency, regress next month's log spot change
on the forward premium (i_usd - i_fx)/12 implied by covered parity. UIP says the
slope is 1. Fama's 1984 finding is that it is negative. Per-currency standard
errors are Newey-West with 6 lags. The pooled row stacks all nine currencies,
and because every one is quoted against the dollar its standard error is
clustered by month; treating the stack as nine independent series would shrink
the SE to 0.45 and manufacture a rejection the data does not support.

## Results

Everything below is produced by `run.py`. Net is after 5bps one-way on traded
foreign notional.

### UIP

| Window | Months | Pooled slope | SE | t against UIP (slope = 1) |
|---|---|---|---|---|
| In-sample | 165 | 0.44 | 0.71 | -0.78 |
| Final test | 126 | -0.02 | 1.02 | -1.00 |
| Full | 291 | **0.05** | 0.67 | **-1.41** |

Per currency on the full sample the slopes run from -0.50 (SEK) to +0.54 (JPY),
with standard errors of 1.1 to 2.5, so none is individually distinguishable from
anything. Pooled, the slope is 0.05, 1.4 standard errors below the UIP value of
1, so the regression leans against UIP without rejecting it at conventional
levels. Nine currencies against one dollar is closer to one noisy series than
to nine, and 24 years of monthly data does not pin a slope down. The point
estimate says high-rate currencies do not depreciate to offset their carry; on
average over this sample they do not move in either direction in response to
the differential.

That is not the classic Fama result, which is a slope well below zero. The
in-sample slope of 0.44 is in line with what others have found since 2008: the
forward premium puzzle weakened after the crisis and the slope has drifted
towards zero or positive. The out-of-sample slope of -0.02 sits at zero. The
reading that survives both windows is "spot does not offset the differential",
which is what carry needs, rather than "high-rate currencies appreciate", which
is a stronger claim this data does not make.

### Carry

| | USD ranked | Dollar neutral |
|---|---|---|
| **In-sample, 2002-04 to 2015-12** | | |
| Annual return, gross / net | +2.99% / +2.93% | +2.95% / +2.90% |
| Annual vol | 9.77% | 9.40% |
| Sharpe, gross / net | 0.35 / 0.35 | 0.36 / 0.35 |
| Skew, monthly / daily | -0.52 / -0.93 | -0.66 / -0.96 |
| Max drawdown | -31.68% | -30.55% |
| Worst month | -10.68% (2008-10) | -10.68% (2008-10) |
| **Final test, 2016-01 to 2026-06** | | |
| Annual return, gross / net | +2.14% / +2.07% | +2.05% / +1.99% |
| Annual vol | 5.68% | 6.04% |
| Sharpe, gross / net | 0.40 / 0.39 | 0.37 / 0.36 |
| Skew, monthly / daily | -0.46 / -0.37 | -0.48 / -0.41 |
| Max drawdown | -9.56% | -13.41% |
| Worst month | -3.97% (2020-03) | -5.35% (2020-03) |
| **Full sample** | | |
| Annual return, gross / net | +2.62% / +2.56% | +2.56% / +2.50% |
| Sharpe, gross / net | **0.35 / 0.35** | 0.35 / 0.34 |
| Skew, monthly / daily | **-0.53 / -0.92** | -0.63 / -0.91 |
| Excess kurtosis, monthly | 2.57 | 2.88 |
| Max drawdown | **-31.68%**, 2007-07-25 to 2009-02-02 | -30.55%, same dates |
| Recovered | 2024-05-16 | 2013-02-14 |
| Hit rate, monthly | 58.4% | 55.3% |
| Annual turnover | 0.62x | 0.55x |
| Cost drag | 0.06%/yr | 0.06%/yr |

Where the gross return comes from, per year, USD ranked:

| | In-sample | Final test | Full |
|---|---|---|---|
| Interest accrual | +3.83% | +2.48% | +3.24% |
| Spot | -0.40% | -0.20% | -0.32% |
| Gross, arithmetic | +3.42% | +2.28% | +2.93% |

This is the UIP result stated as money. The book collects 3.24% a year in
interest and gives back 0.32% of it in spot. UIP said it would give back all of
it.

Three things to say about the Sharpe. It is 0.35 gross and net, because the
book trades 0.6x its one-sided notional a year (1.2x counting both legs) and at
5bps that is six basis points a year, which does not matter. It is lower than the 0.6 to 0.9 that the pre-2008 literature reports,
because this sample starts in 2002, contains 2008 in full, and then spends a
decade in near-zero rates where the average absolute differential the book was
collecting fell from 3.8% to 2.5% a year. And with 24 years of data the standard
error on a Sharpe ratio is about 0.20, so 0.35 is a little under two standard
errors from zero. The final-test Sharpe of 0.39 is consistent with the
in-sample 0.35 and neither is a precise number.

The dollar-neutral version is the same trade with the same numbers, except for
one thing: it recovered its 2007 peak in February 2013 while the USD-ranked
book took until May 2024. The USD-ranked book was short the dollar from 2009 to
2014 (the Fed was at zero, so USD ranked in the bottom three) and then long it
from 2016 onwards, and the 2011 to 2015 dollar rally cost it the difference. On
66% of rebalance dates the USD sits in one of the tails, so the two books are
different more often than not, but the average net dollar exposure is +0.02, so
it nets out to nothing over the sample and shows up only in the path.

### Per year, net

| 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| +8.9% | +17.7% | +5.5% | +12.8% | +1.4% | +4.1% | **-22.3%** | +21.5% | +2.0% | +1.6% | +6.5% | -5.2% |

| 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -1.5% | -5.2% | +7.7% | -3.8% | +1.8% | +4.6% | -5.3% | +5.0% | -1.0% | +1.8% | +8.1% | -1.8% | +5.6% |

2002 to 2007 is the carry trade everyone remembers, +8.8% a year compounded with
the worst year at +1.4%. Everything after 2009 is a different regime: 17 years
averaging +1.2% with the sign flipping every year or two.

### Holdings

Share of rebalance dates each currency was held, USD ranked:

| | USD | EUR | JPY | GBP | CAD | AUD | NZD | CHF | NOK | SEK |
|---|---|---|---|---|---|---|---|---|---|---|
| Long | 36% | 0% | 0% | 30% | 10% | 73% | 90% | 0% | 62% | 0% |
| Short | 30% | 45% | 58% | 0% | 9% | 0% | 0% | 100% | 1% | 55% |

CHF is in the short leg on every single rebalance. NZD is long 90% of the time,
AUD 73%. The trade is, for most of its life, long the antipodeans and Norway
against the franc, the yen and whichever of EUR or SEK is lowest. That is worth
knowing before reading the crisis section, because it means the book's risk is
concentrated in a handful of currency pairs and the "diversified G10 sort" is
mostly a name.

### Crisis autopsies

**2008.** Entering July 2008 the book was long AUD, NZD, NOK and short JPY, CAD,
CHF. Over July 2008 to March 2009 it lost 17.7% net; peak to trough inside the
window was -27.9% from 2008-07-22 to 2009-02-02. All three longs fell 25 to 27%
against the dollar and contributed -8.6%, -7.6% and -7.8%. The short JPY leg
lost another 2.3% as the yen rose 6.6%. The only things that helped were the
CAD and CHF shorts, which fell with everything else. The three worst days were
2008-10-06 (-6.1%), 2008-10-24 (-5.5%) and 2008-10-08 (-4.9%), and on each the
damage was almost entirely the long leg. October 2008 was -10.7%, the worst
month in the sample. This is the carry crash in its standard form: the funding
currencies rally and the investment currencies collapse at the same time, so a
book that looks hedged because it is long three things and short three things
is in fact one bet on risk appetite.

**January 2015.** The SNB dropped the EUR/CHF floor on 15 January. The book was
short CHF, as it always is, at -1/3. CHF rose 7.9% against the dollar over the
month and the position lost 2.9%; on the day itself the book lost 3.3%, all of
it on the short leg. The month was -3.8% net. A one-third weight in a currency
that can gap 15% in a morning is the tail that the -0.92 daily skew is
describing.

**August 2015.** China devalued on 11 August and the commodity currencies sold
off. The book was long AUD, NZD, NOK and short EUR, CHF, SEK. Over August and
September it lost 4.5% net, with -6.2% peak to trough from 2015-08-10 to
2015-08-24. The three longs lost 1.3%, 1.0% and 1.5%; the EUR and SEK shorts
lost too as both rose. 24 August, the day of the equity flash crash, cost 3.1%
split evenly across both legs.

**March 2020.** Long USD, CAD, NOK, short EUR, CHF, SEK going in. Lost 4.5% net
over mid-February to April with a -7.7% drawdown to 2020-03-20. NOK fell 9.8%
and contributed -3.1%, CAD -2.1%. The USD position, one third of the long leg,
contributed exactly nothing, which is the price of ranking the dollar: when
your safe-haven currency is the base currency, being long it is being flat.

The pattern across all four is the same. Losses come from the long leg's spot
move, the short leg's funding currencies rally at the same time, and there is
no month in which the interest accrual (about 27bps a month) is large enough to
matter against a 5% spot move.

### Costs

Net Sharpe by one-way cost:

| Cost | USD ranked IS | USD ranked test | Dollar neutral IS | Dollar neutral test | Annual cost |
|---|---|---|---|---|---|
| 0bps | 0.35 | 0.40 | 0.36 | 0.37 | 0.000% |
| 5bps | 0.35 | 0.39 | 0.35 | 0.36 | 0.062% |
| 10bps | 0.34 | 0.38 | 0.35 | 0.35 | 0.123% |
| 20bps | 0.33 | 0.35 | 0.33 | 0.33 | 0.247% |
| 40bps | 0.31 | 0.30 | 0.31 | 0.29 | 0.494% |

Costs are not what decides this trade. Even at 40bps one-way, eight times what
G10 majors actually cost, the Sharpe loses 0.05. The reason is that the ranking
barely moves: in 2011, 2015, 2016 and 2018 the book did not trade at all.

## Files

- `data.py` FRED download, unit conversion, month-end alignment of monthly rates to daily spot
- `carry.py` differential, excess returns, the sort, both weight constructions, the UIP regression
- `backtest.py` execution lag, costs, drawdown, skew and the other metrics
- `check.py` 41 offline checks on synthetic currencies with planted differentials
- `run.py` full pipeline, tables and charts to `reports/`

Charts in `reports/`: `equity_curve.png`, `drawdown.png`, `monthly_histogram.png`,
`crisis_2008.png`, `by_year.png`, `holdings.png`, `uip_scatter.png`.

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is fully offline and exits nonzero if anything fails. `run.py`
downloads nineteen FRED series once into `../source-material/fx-carry/` and is
offline after that. Delete that folder to refresh. No dependencies beyond the
shared `requirements.txt`.

## Limitations

**Interbank rates, not forward points.** The carry a real trader earns is the
forward discount, which equals the interbank differential only if covered
interest parity holds. It held closely before 2008 and has not since; the
cross-currency basis for JPY and CHF against USD has run at tens of basis points
for most of the final-test window. My accrual therefore overstates the carry on
the yen and franc shorts by roughly that amount. The sign of the error is
against the strategy and the size is a fraction of a percent a year, not enough
to change the conclusion but enough that the 3.24% accrual figure is an upper
bound.

**Monthly average rates used at month end.** Covered above. A daily rate series
would remove it, and OECD does not publish one. The ranking barely notices it
because differentials between the tails are usually a full percentage point or
more.

**No survivorship in the usual sense, but a selected universe.** No G10 currency
disappeared, so nothing dropped out of the sample. The selection is the choice
of G10 itself, which is a set of currencies picked precisely because they have
had liquid, uninterrupted markets. The carry trades that ended worst (Russia
1998, the Asian currencies in 1997, Turkey and Argentina more recently) are all
outside it, and the 1998 yen carry unwind is before the sample starts. A G10
carry Sharpe is the Sharpe of the currencies that were safe enough to survive.

**Flat cost, no roll, no spread widening.** 5bps one-way on a change in
weights, nothing for rolling a forward each month, and nothing for the fact
that spreads widen in exactly the weeks the book is losing. Since costs do not
decide the result this matters less here than usual, but it is still a
best-case cost model.

**The book is not what it looks like.** Three long, three short, equal weight
sounds diversified. The holdings table says the book is long NZD and AUD and
short CHF and JPY for most of 24 years. It has been a risk-on bet on two small
commodity economies against two safe havens, and the four crisis autopsies all
show the same failure mode because it is one exposure, not six.

**No volatility scaling, no crash conditioning.** The published fixes are to
scale positions by trailing FX volatility or to cut carry when the VIX or a
risk gauge is elevated. I did not build either, because both add parameters and
the point of this project was to measure the raw premium and its tail, not to
engineer them away.

**Sharpe 0.35 on 24 years is not a precise number.** Standard error about 0.20.
The trade probably has a positive premium; the size of it is not something this
sample pins down.
