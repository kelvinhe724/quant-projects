# Rates carry and rolldown

A bond premium that is not equity beta, not trend and not FX carry: the excess return from lending long over funding short, measured on the US Treasury curve since 1962 and then held through Treasury ETFs in the same engine and cost model as the other sleeves.

For each tenor on each month end I compute

- **carry** = tenor yield minus the 3-month funding rate,
- **rolldown** = the yield a T-year bond gives up by becoming a (T - 1/12)-year bond on an unchanged curve, turned into a return by multiplying by duration and annualising,
- **expected excess return** = carry + rolldown,

and hold the two tenors with the highest positive expected excess return, sized so the book carries 7 years of duration. The interpolation between quoted tenors comes from the Nelson-Siegel fit in `../nelson-siegel/`, imported, with the fit used for the shape between quoted points and never for their level.

The short version: the term premium is real but has changed sign twice in sixty years; carry does predict next month's excess return (pooled t 3.2), rolldown on its own does not (t 0.95); the rule holds up through 2022 slightly better than a fixed IEF/TLT position; and through the book's engine it is a 0.53 net Sharpe bond position with a beta of 0.75 to IEF+TLT and an alpha of 1.4% a year that is not significant (t 1.66).

## Data

- Curve: FRED constant-maturity yields (DGS1MO to DGS30), through the NS project's loader, 1962-01-02 to 2026-09-01. The 1y, 3y, 5y, 10y and 20y run from 1962; 7y from 1969; 2y from 1976; 30y from 1977; the bills from 1981; 1m from 2001. The loader forward-fills, so I mask each tenor back to the raw series: DGS20 was not published from 1987 to 1993 and would otherwise have been a constant for six years.
- Funding: DGS3MO from 1982, DTB3 (3-month bill, secondary market) before that.
- ETFs: SHY, IEI, IEF, TLH, TLT from yfinance, 2002-07-30 on (IEI and TLH from 2007-01-11), through `framework.engine.load_yfinance`.

## The curve study, 1962-2026

776 month-end Nelson-Siegel fits, RMSE mean 8.3 bp, median 4.7 bp. The 1980s are the bad decade (mean 26 bp, max 118 bp in September 1981): a four-parameter curve cannot hold a 15% bill rate and a 15% bond rate with a hump in between. That is why the level at each quoted tenor is always the observed yield and the fit only supplies the slope to the tenor one month shorter.

### Term premium by decade

Realised excess return of a par bond bought at the month end, sold one month later, over 3-month funding, percent a year; each row is grouped by the month the bond was bought. Calendar-year figures below use the month the return landed in. `reports/term_premium_by_decade.csv` has the t-statistics.

| | 2y | 5y | 10y | 20y | 30y |
|---|---|---|---|---|---|
| 1960s | | -1.6 | -2.9 | -3.9 | |
| 1970s | -1.2 | +0.4 | -0.7 | -2.3 | -9.8 (1977-79 only) |
| 1980s | +2.1 | +2.9 | +3.8 | +7.5 | +5.0 |
| 1990s | +1.6 | +2.5 | +3.2 | +0.8 (gap to 1993) | +4.6 |
| 2000s | +2.0 | +3.8 | +4.7 | +5.6 | +5.7 |
| 2010s | +0.8 | +2.5 | +4.6 | +6.9 | +8.2 |
| 2020s | -1.0 | -1.9 | -3.3 | -5.9 | -7.3 |
| 1962-2026 | +1.1 (t 2.9) | +1.5 (t 2.3) | +1.7 (t 1.8) | +1.5 (t 1.1) | +3.2 (t 1.7) |

The sign flips with the direction of rates. 1962-1981 was a rising-rate era and long bonds lost to bills for two decades; 1982-2020 was the great disinflation and every tenor paid, the long end most; the 2020s so far are the reversal. Over the whole sample the 10y earned 1.7% a year over bills with a t of 1.8, which is the honest size of the premium: positive, and not something sixty years can pin down.

### 2022

Going in, carry plus rolldown said every tenor should earn about 1.2-2.1% a year over funding. What happened, excess of funding, one-month-rolled par bonds:

| | 2y | 5y | 10y | 20y | 30y |
|---|---|---|---|---|---|
| expected at end 2021, %/yr | +1.2 | +1.8 | +1.9 | +2.1 | +2.0 |
| realised 2022, % | -5.6 | -10.9 | -17.8 | -27.2 | -33.3 |

The 10y's -17.8% is the worst calendar year in this sample by six points (next 1999 and 1980, both -11.8%); the 30y's -33.3% is the worst by sixteen (next 1999, -17.0%). Two things made it that bad: the level of yields was 1.5% so durations were at their maximum (22.7 years on the 30y against 17 at a 4% yield), and there was no coupon to cushion it. The sleeve held the 20y and 30y on the way in, because that is where the expected excess return was highest, and lost 7.4% through the engine against 9.4% for a fixed IEF/TLT position; the difference is that the rule went flat in October 2022 once the curve inverted and every expected excess return turned negative, and stayed flat through all of 2023.

### Does carry plus rolldown predict the realised excess return?

Next month's realised excess return (percent a year) regressed on the expected one, Newey-West per tenor, pooled with errors clustered by month.

| | slope | se | t | alpha | R2 | n |
|---|---|---|---|---|---|---|
| 2y | 1.24 | 0.43 | 2.9 | -0.1 | 0.020 | 602 |
| 5y | 1.78 | 0.50 | 3.6 | -1.1 | 0.019 | 775 |
| 10y | 2.30 | 0.66 | 3.5 | -2.0 | 0.018 | 775 |
| 20y | 3.19 | 0.93 | 3.4 | -3.8 | 0.018 | 693 |
| 30y | 3.41 | 1.16 | 2.9 | -3.3 | 0.014 | 594 |
| pooled | 2.51 | 0.79 | 3.2 | -2.1 | 0.015 | 3439 |

Three readings. It predicts: every tenor's t is about 3, and the pooled estimate holds in both halves of the sample (slope 3.1, t 2.4 to 1994; 2.1, t 2.2 after). The slope is above one and grows with maturity: a percent of expected return brings more than a percent of realised, which is the Campbell-Shiller result that a steep curve is followed by falling long yields rather than the rising ones the expectations hypothesis needs, so carry earns its yield pickup and a capital gain on top. And the R2 is 1.5%: monthly bond returns are almost all rate shock, and this signal explains the same small slice of it that the term-spread literature reports.

Split into its two pieces, pooled: carry alone has slope 3.2, t 3.5; rolldown alone has slope 1.7, t 0.94, R2 0.001. The rolldown term is small (mean 0.3% a year at the 10y, 0.05% at the 30y, where the curve is flat) and its month-to-month variation carries no detectable information about next month's return. Everything the signal knows is in the level of the curve over funding.

### The rule on the curve itself, no costs

Monthly, 7 years of duration, excess of funding, 1962-2026:

| | ann. return | ann. vol | Sharpe | max drawdown |
|---|---|---|---|---|
| sleeve: top 2 long only, flat when nothing pays | +2.4% | 7.2% | 0.36 | -39% (1981) |
| top 2 vs bottom 2, duration neutral | +0.4% | 4.7% | 0.10 | -33% |
| always long the 10y at the same duration | +1.5% | 7.8% | 0.24 | -49% (1981) |

The sleeve is a long-duration position with a switch. It has beta 0.80 to the constant 10y and 1.2% a year of alpha, holds the 20y or 30y in over half of all months, and is flat in 10% of them. The duration-neutral version, which is what carry looks like once you take the term premium out of it, has a Sharpe of 0.10 over 64 years: the cross-section of tenors carries almost nothing once the level of the curve is removed. By decade the sleeve beats the 10y in every decade except the 1960s and 2020s, when both lose, because the switch turns off in inverted curves and inverted curves precede the bull markets.

## Through the book's engine, 2002-2026

Same engine, cost model and overlay as `../framework/book/`: 5 bps commission, 2 bps half spread, sqrt impact, 50 bps borrow, next-open fills, 10% vol target, 3x gross cap, half size past a 15% drawdown, 10% position buffer, kill off, cash earning nothing. Each fund is scaled by its tenor's par duration over the fund's fact-sheet duration so the position carries the duration the rule asked for; `check.py` holds that arithmetic, and the fact-sheet durations sit within 11% of a par bond at 4% for every fund (SHY 2%, IEI 3%, IEF 11%, TLH 7%, TLT 8%). Empirical durations from daily fund returns on daily yield changes agree with the fact sheets to within 0.2 years except TLH (11.3 measured against 12.6 quoted, `reports/etf_durations.csv`).

| | Sharpe gross | Sharpe net | ann. return | ann. vol | max drawdown | 2022 return | 2022 drawdown | turnover/yr | trades |
|---|---|---|---|---|---|---|---|---|---|
| **RatesCarry (sleeve)** | 0.56 | **0.53** | +4.2% | 8.9% | -21.6% (2022-11) | -7.4% | -9.3% | 3.6x | 295 |
| RatesCarry, long/short duration neutral | 0.06 | -0.10 | -0.6% | 4.5% | -40.3% | +3.2% | -2.7% | 2.7x | 479 |
| hold IEF+TLT 1/N | 0.48 | 0.47 | +3.8% | 8.8% | -24.7% (2023-10) | -9.4% | -10.5% | 1.1x | 411 |
| hold all five funds 1/N | 0.58 | 0.57 | +4.8% | 9.0% | -25.2% | -9.7% | -10.9% | 1.7x | 996 |
| sleeve at 2x costs | 0.57 | 0.51 | | | -22.0% | | | | |
| sleeve at 4x costs | 0.54 | 0.42 | | | -22.8% | | | | |

Gross is the same path with each day's costs added back, the book's convention. Cost drag is 0.03 of Sharpe at the base model: a monthly rule in five of the most liquid ETFs in the world does not have a cost problem.

### Attribution

Sleeve net returns regressed on owning its benchmark, using `framework/book/validate.py::attribution` so the numbers read like the book's table. Alpha is annualised with a Newey-West t; residual Sharpe is the sleeve minus beta times the benchmark.

| benchmark | benchmark Sharpe | sleeve Sharpe | alpha / yr | t | beta | residual Sharpe |
|---|---|---|---|---|---|---|
| IEF+TLT, 1/N | 0.39 | 0.50 | +1.4% | 1.66 | 0.75 | 0.33 |
| IEF+TLT, 1/N vol-targeted | 0.46 | 0.50 | +0.8% | 1.04 | 0.78 | 0.19 |
| all five funds, 1/N | 0.48 | 0.50 | +0.8% | 0.92 | 1.09 | 0.18 |
| all five funds, 1/N vol-targeted | 0.55 | 0.50 | +0.1% | 0.14 | 0.78 | 0.03 |

(The attribution rows keep the zero-return days of 2023 in the sample, so the sleeve's Sharpe there is 0.50 against 0.53 in the performance table, which drops them the way the book does.)

Against IEF+TLT the sleeve has 1.4% a year of alpha at t 1.66, positive and not significant, and its residual Sharpe of 0.33 is below the benchmark's own 0.39. Against a vol-targeted 1/N of all five funds the alpha is 0.1% a year. The sleeve is a duration position with a switch that has turned off twice in 24 years (late 2006 to 2007, and October 2022 through 2023), and the switch is where the 0.06 of Sharpe over IEF+TLT comes from. That is what carry on one curve is: the term premium, held only when the curve says it is there.

## What I take from it

1. The term premium exists and is not stable. 1.7% a year at the 10y over 64 years, t 1.8, with two decades of the wrong sign at each end of the sample.
2. Carry predicts. A t of about 3 at every tenor, a slope above one, and it survives a split of the sample. R2 of 1.5% is the size of that.
3. Rolldown does not, on its own. Its variation is too small next to carry to detect. Carry plus rolldown and carry alone are the same signal here.
4. The cross-section of tenors is almost empty. Duration neutral, 64 years, Sharpe 0.10 on the curve and -0.10 through the engine. The premium is the level of the curve, not the shape.
5. Through the engine the sleeve is a 0.53 net Sharpe bond position, beta 0.75 to IEF+TLT, with no significant alpha over just owning them. It belongs in a book as a different beta from equities, trend and FX carry, not as a source of skill.

## Files

- `data.py` curve through the NS project's loader, funding rate, ETF bars, and the Bars object with the yields attached
- `carry.py` bond arithmetic, curve fit, carry, rolldown, expected and realised excess returns, the position rule, the ETF map, the regression, the engine strategy
- `check.py` 47 offline checks on hand-built curves; exits 1 on any failure
- `run.py` everything above; tables and charts to `reports/`, console to `reports/run_log.txt`

Charts: `term_premium_by_decade.png`, `predictive_regression.png`, `study_equity.png`, `engine_equity.png`, plus the engine's own `reports/engine/`.

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline. `run.py` downloads once (FRED into `reports/treasury_curve.csv` and the framework's FRED cache, ETFs into the framework's yfinance cache) and takes about 45 seconds after that. It does not append to `framework/book/reports/trials.csv`: the sleeve is not in the book and has not been through `validate.py`, so it is not a counted trial there.

## Limitations

**One curve, one premium.** Everything is the US Treasury curve. The paper version of this trade is a cross-country carry book in ten bond markets, where the cross-section is the point; here the cross-section is five tenors on one curve and it carries almost nothing.

**The ETFs are not the tenors.** IEF holds 7-10y paper and runs 7.2 years of duration against 8.1 for a 10y par bond; TLH is 11.3 measured against 13.6 for the 20y. I scale by fact-sheet duration, which is a snapshot; a fund's duration moves with the level of yields and with its holdings, and the map is only as good as that number. The empirical durations in `reports/etf_durations.csv` are the check.

**Par-bond arithmetic with annual coupons and a closed-form price at fractional maturity.** Real Treasuries pay semiannually and the CMT curve is a par curve with its own construction. The realised returns here are those of a synthetic par bond on the fitted curve, not of any bond that traded, and the exact month-end-to-month-end day count is used. The reconciliation check in `check.py` shows the realised return on an unchanged curve equals the expectation to within convexity.

**The 1980s fit.** RMSE of 26 bp on average, 118 bp at the worst. Anchoring the level at each quoted tenor to the observed yield keeps that out of carry and out of the realised return, but the one-month rolldown slope in those years is a four-parameter curve's opinion of a shape it cannot hold.

**Funding at the bill rate.** The 3-month CMT (a bond-equivalent yield) from 1982, the bill discount rate before it; no repo, no financing spread, and the engine's version of the sleeve earns nothing on cash, which is the book's convention and is harsh on a rule that sits in cash 10% of the time.

**Two parameters chosen without a search, but chosen.** Top 2 of 5 and 7 years of duration were written down before the run and not varied. I have not shown the result is insensitive to them, and the long-only rule's positive-expectation switch is a third choice: the flat months are where it beats the constant 10y, and 10% of months over 64 years is a small number of episodes.

**Monthly on the curve, daily through the engine.** The curve study fills at the month-end close; the engine fills at the next open and rebalances into a 10% vol target with a buffer. The two are not the same trade, which is why the study and the engine sections are reported separately and not reconciled.
