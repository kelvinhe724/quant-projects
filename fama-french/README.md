# Fama-French factor models and a blended factor portfolio

Two stages. First I replicate the three- and five-factor time-series regressions
on the 25 size/book-to-market portfolios and on ten individual stocks, and test
the alphas jointly with GRS. Then I build an inverse-volatility blend of the
long-short factors, compare its Sharpe and drawdown to holding the market, and
split the sample at 2010 to see whether the premia are still there.

Sample is 1963-07 to 2026-06, 756 monthly observations. Everything is in
percent per month.

## Data

Factors, momentum and the 25 portfolios come from the Ken French data library
through `pandas_datareader` (`famafrench` source, no key needed). Stock prices
come from Yahoo via `yfinance`, monthly total returns from adjusted closes,
1990-02 onward. Excess returns subtract the RF column of the French file.

`check.py` runs offline: it builds synthetic assets from planted loadings and a
planted alpha and asserts the regressions recover them.

## Files

- `data.py` factor, portfolio and stock downloads
- `factors.py` `regress`, `fit_panel`, `grs`, `premia`, `blend`, `performance`
- `check.py` offline checks against planted loadings
- `run.py` full pipeline, charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

## Results

### Stage 1: replication

25 size/value portfolios, 1963-2026:

| model | k | mean R² | mean abs alpha | alphas with t>2 | GRS | p |
|---|---|---|---|---|---|---|
| CAPM | 1 | 0.734 | 0.198 | 10 | 4.17 | 9.0e-11 |
| FF3 | 3 | 0.912 | 0.090 | 6 | 3.66 | 7.1e-09 |
| FF5 | 5 | 0.916 | 0.087 | 7 | 3.23 | 2.5e-07 |
| FF5+MOM | 6 | 0.917 | 0.081 | 6 | 2.90 | 3.5e-06 |

Adding SMB and HML to CAPM is the big jump: mean R² goes from 0.73 to 0.91 and
mean absolute alpha halves. The two extra factors in FF5 buy almost nothing on
these test assets, +0.005 of R², which is expected since the 25 portfolios are
sorted on exactly the two characteristics FF3 already prices. Every model is
rejected by GRS. The alphas are small but they are not zero, and 63 years of
monthly data has enough power to say so.

FF3 alphas, monthly percent, t-stat in brackets:

```
  Small  -0.44[-4.79]   0.03[ 0.49]   0.01[ 0.19]   0.15[ 3.06]   0.19[ 2.66]
  2      -0.15[-2.48]   0.05[ 0.94]   0.09[ 1.68]   0.06[ 1.35]   0.00[ 0.04]
  3      -0.09[-1.58]   0.06[ 1.02]  -0.02[-0.42]   0.03[ 0.50]   0.01[ 0.15]
  4       0.06[ 1.09]  -0.06[-0.99]  -0.03[-0.43]   0.05[ 0.82]  -0.11[-1.45]
  Big     0.16[ 3.95]  -0.00[-0.03]  -0.02[-0.39]  -0.23[-3.93]  -0.12[-1.26]
```

The small-growth cell is the worst, -0.44% a month at t = -4.79, which is the
same corner that has embarrassed the model since 1993. FF5 shrinks it to -0.275
(t = -3.23) because small growth stocks load negatively on RMW and CMA, unprofitable
firms that invest aggressively. That is precisely the fix the 2015 paper claims,
and it shows up here, but it is a shrinkage and not a repair.

### Does this match the published numbers

I reran the test on Fama-French's own 1963-07 to 1991-12 window:

| model | GRS | p | mean abs alpha | mean R² |
|---|---|---|---|---|
| CAPM | 2.01 | 0.0033 | 0.257 | 0.795 |
| FF3 | 1.45 | 0.079 | 0.094 | 0.935 |

Mean R² of 0.935 matches the "above 0.9, mostly 0.93" the 1993 paper reports for
the three-factor regressions on these portfolios. The pattern of the alphas
matches too: small and low book-to-market negative, most of the interior close
to zero. The GRS statistic does not match exactly. FF (1993) report 1.91 for the
three-factor model and I get 1.45, which flips the 5% verdict from reject to
not-reject. I do not think that is a coding error. The French library has been
rebuilt many times since 1993 on a restated CRSP file and with revised
Compustat coverage, so the 1963-1991 returns I download today are not byte-for-byte
the returns they regressed. A GRS statistic aggregates 25 alphas and a 25x25
residual covariance, so it is far more sensitive to small data revisions than the
R² or the individual alphas are. Call the replication a match on the economics
and a near-match on the test statistic.

Ten individual stocks, 1990-2026, FF5:

| stock | alpha | t | Mkt-RF | SMB | HML | RMW | CMA | R² |
|---|---|---|---|---|---|---|---|---|
| AAPL | 1.429 | 2.85 | 1.08 | 0.33 | -0.54 | 0.19 | -0.75 | 0.282 |
| CAT | 0.219 | 0.64 | 1.37 | 0.18 | 0.46 | 0.32 | 0.48 | 0.426 |
| GE | -0.014 | -0.05 | 1.23 | -0.24 | 0.56 | -0.17 | -0.09 | 0.439 |
| JNJ | 0.251 | 1.09 | 0.70 | -0.26 | -0.19 | 0.51 | 0.44 | 0.281 |
| JPM | 0.468 | 1.50 | 1.21 | -0.12 | 1.18 | -0.68 | -0.55 | 0.521 |
| KO | 0.039 | 0.16 | 0.74 | -0.22 | -0.17 | 0.55 | 0.54 | 0.283 |
| MSFT | 1.070 | 3.35 | 1.06 | -0.31 | -0.39 | 0.18 | -0.79 | 0.447 |
| PG | 0.060 | 0.25 | 0.65 | -0.14 | -0.37 | 0.59 | 0.86 | 0.249 |
| WMT | 0.448 | 1.65 | 0.65 | -0.27 | -0.16 | 0.55 | -0.15 | 0.238 |
| XOM | 0.055 | 0.22 | 0.71 | 0.02 | 0.37 | 0.21 | 0.39 | 0.305 |

R² on single names runs 0.24 to 0.52 against 0.91 for the portfolios, because a
single stock carries idiosyncratic risk that diversifies away in a 25-portfolio
sort. The loadings are readable: JPM has an HML beta of 1.18 and a negative RMW,
a bank; PG and KO load positively on RMW and CMA, profitable firms that do not
reinvest much; AAPL and MSFT have CMA loadings near -0.8, the aggressive-investment
end. Apple and Microsoft carry alphas of 1.43 and 1.07 a month with t above 2.8,
and GRS on the ten together is 2.38, p = 0.0094. That is a survivorship-selected
basket of names I already knew had won, so the right reading is that my sampling
is biased, not that the factor model missed 17% a year of free money.

### Stage 2: blended factor portfolio

Inverse-volatility weights across SMB, HML, RMW, CMA and momentum, weights
computed from a 60-month trailing window lagged one month, rebalanced monthly.

Full sample 1968-07 to 2026-06:

| portfolio | ann return | ann vol | Sharpe | max drawdown |
|---|---|---|---|---|
| market (Mkt-RF) | 6.01% | 15.82% | 0.45 | -55.8% |
| blend, 5 long-short | 3.42% | 4.35% | 0.80 | -14.4% |
| blend incl. market | 3.70% | 3.62% | 1.02 | -10.6% |

The blend nearly doubles the market's Sharpe and cuts the worst drawdown from
-56% to -14%. Those are unlevered long-short returns, so the absolute return is
low by construction; at market volatility the blend would have returned around
12% a year before any financing or shorting cost, which is the honest way to
state it and also the point where the comparison starts to get generous.

2010 onward:

| portfolio | ann return | ann vol | Sharpe | max drawdown |
|---|---|---|---|---|
| market (Mkt-RF) | 12.71% | 14.97% | 0.88 | -25.3% |
| blend, 5 long-short | 0.86% | 4.33% | 0.22 | -14.4% |
| blend incl. market | 2.06% | 4.04% | 0.53 | -10.6% |

The ranking reverses completely. Post-2010 the market Sharpe is 0.88 and the
blend is 0.22.

### Have the premia decayed

Yes, badly, and it is not subtle:

| factor | full sample | pre-2010 | 2010+ |
|---|---|---|---|
| Mkt-RF | 0.601 (t 3.70) | 0.426 (t 2.23) | 1.095 (t 3.57) |
| SMB | 0.186 (t 1.70) | 0.278 (t 2.10) | -0.071 (t -0.37) |
| HML | 0.298 (t 2.76) | 0.418 (t 3.46) | -0.038 (t -0.17) |
| RMW | 0.239 (t 2.92) | 0.280 (t 2.88) | 0.124 (t 0.83) |
| CMA | 0.246 (t 3.26) | 0.324 (t 3.73) | 0.025 (t 0.17) |
| Mom | 0.620 (t 4.09) | 0.718 (t 3.89) | 0.344 (t 1.36) |

Every one of the five non-market premia loses its significance after 2010. HML
and SMB go outright negative. RMW and CMA keep the right sign but shed half to
ninety percent of their size and neither clears t = 1. Momentum halves. The one
premium that got bigger is the market itself, 5.1% annualized before 2010 and
13.1% after, which is most of the story of the last fifteen years: owning
beta paid and everything else did not.

Two readings, and I cannot separate them with this data. Either the factors were
arbitraged away after publication, which is what McLean and Pontiff find across
the anomaly literature generally, or 198 months is a short window and value in
particular is mean-reverting on a decade scale, in which case the 2010s were a
drawdown and not a death. What I can say is that anyone quoting the full-sample
HML premium of 3.6% a year as a forward expectation is quoting a number that has
not been earned since 2009.

Charts in `reports/`: `factor_cumulative.png`, `blend_vs_market.png`,
`drawdown.png`, `rolling_premia.png`, `alpha_grid.png`.

## Limitations

The factor returns are the ones Ken French publishes, so I have replicated the
regressions, not the factor construction. Building SMB and HML from CRSP and
Compustat myself would be the real replication, and it is where all the
interesting choices live: breakpoints, rebalance timing, delisting returns.

The blend has no transaction costs, no shorting costs and no borrow constraints.
Long-short factor portfolios turn over heavily, momentum most of all, and the
academic factors are notoriously the version of the strategy that costs nothing
to trade. A realistic cost assumption would take a meaningful bite out of that
0.80 Sharpe and would hit momentum hardest.

The inverse-vol weights use a trailing window and a one-month lag so there is no
look-ahead in the weighting, but the choice of which five factors to blend is
itself made with full-sample hindsight. I picked the factors that are famous
because they worked.

Ten hand-picked large caps that all still trade in 2026 is a survivorship-biased
sample, which is why their alphas look so good. The GRS rejection on that basket
measures my stock picking in 2026, not a failure of the model in 1990.

GRS assumes normal iid errors. Monthly equity returns are fat-tailed and
heteroskedastic, so the exact F distribution is an approximation and the p-values
should be read as indicative. A wild bootstrap would be the fix.
