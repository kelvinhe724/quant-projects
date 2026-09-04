# Cross-hedging commodity futures

Screens 78 pairs of commodity futures for return co-movement, converts the
regression slopes into whole contracts using each exchange's real lot size and
quote unit, tests whether the ratios hold up, and then applies ratios estimated
on 2015-2023 to an untouched 2024-2026 window.

Two things came out of it that I did not expect. The first is how badly price
levels lie: regressing raw prices on raw prices gives a mean R-squared of 0.452
across the 78 pairs and a Durbin-Watson of 0.02, and copper explains 70% of the
variation in corn prices. On returns those same 78 pairs average 0.091, and
copper-corn drops to 0.020. The second is that out-of-sample hedge effectiveness
did not decay for the pairs that mattered: WTI hedged with Brent reduced variance
86.1% in training and 83.4% in the test window. Hedging is not forecasting, and
that difference shows up in the numbers.

## Data, and how it differs from the brief

The brief specifies thirteen named December-2025 contracts pulled from Bloomberg.
I do not have a Bloomberg terminal, so I used **continuous front-month series**
for the same thirteen underlyings from Yahoo Finance via `yfinance`, daily closes
from 2015-01-02 to 2026-08-31, cached to `source-material/commodities/`.

What that substitution changes:

- **Roll returns are inside the series.** A continuous front-month contract rolls
  from one expiry to the next; unless the price is back-adjusted, the roll day
  shows a jump that is a change of instrument rather than a market move. Yahoo's
  series are not back-adjusted, so a handful of my daily returns per contract per
  year are roll artifacts. This inflates measured volatility slightly and adds
  noise that is uncorrelated across contracts with different roll schedules, so
  it biases R-squared *down*, not up.
- **No expiry calendar.** Sections 3 and 3.1 of the brief are largely about
  building one, and with a continuous series there is nothing to build: the
  contract never dies. I lose the exercise, and I also lose the ability to say
  anything about basis behaviour into notice period, which is the part of the
  brief that is genuinely about a specific contract.
- **No single-contract basis.** Everything here is front-month against
  front-month. A desk hedging a December inventory with a March contract has a
  calendar-spread exposure that this study cannot see at all.
- **A much longer sample.** The brief's design gets at most 11 months of daily
  data on a December contract; I get 2,926 trading days. That is a real advantage
  for the stability tests, and it is why the block bootstrap intervals here are
  narrow enough to be worth reading.

If a Bloomberg export of the actual December-2025 contracts arrives, drop it at
`source-material/commodities/bloomberg_dec2025.csv` with one column per ticker
and set `USE_BLOOMBERG = True` in `data.py`. Everything downstream works on a
price frame and does not care where it came from, with one exception: the expiry
handling in `get_panel` currently only drops stale settlement runs, and a
fixed-expiry file would also need the last-trading-date cut described in the
brief.

One thing yfinance gets wrong that I did not fix: volume for the metals
continuous series is nonsense (median 2 contracts a day for platinum, 48 for
silver). I report it in the coverage table because it is what the source gives,
and I do not use it for anything.

### Instruments

| ticker | contract | venue | quote | lot | $ per 1.00 price move |
|---|---|---|---|---|---|
| CL=F | WTI crude | NYMEX | USD/barrel | 1,000 bbl | 1,000 |
| BZ=F | Brent crude | ICE Europe | USD/barrel | 1,000 bbl | 1,000 |
| NG=F | Henry Hub gas | NYMEX | USD/MMBtu | 10,000 MMBtu | 10,000 |
| HO=F | NY Harbor ULSD | NYMEX | USD/gallon | 42,000 gal | 42,000 |
| RB=F | RBOB gasoline | NYMEX | USD/gallon | 42,000 gal | 42,000 |
| GC=F | Gold | COMEX | USD/troy oz | 100 oz | 100 |
| SI=F | Silver | COMEX | USD/troy oz | 5,000 oz | 5,000 |
| HG=F | Copper | COMEX | USD/pound | 25,000 lb | 25,000 |
| PL=F | Platinum | NYMEX | USD/troy oz | 50 oz | 50 |
| KC=F | Coffee C | ICE US | cents/pound | 37,500 lb | 375 |
| ZC=F | Corn | CBOT | cents/bushel | 5,000 bu | 50 |
| ZW=F | SRW wheat | CBOT | cents/bushel | 5,000 bu | 50 |
| ZS=F | Soybeans | CBOT | cents/bushel | 5,000 bu | 50 |

Lot sizes and tick sizes are from the CME Group contract specification pages for
every contract except Brent (ICE Futures Europe) and Coffee C (ICE Futures U.S.),
checked 2026-09-03. The last column is `lot x quote_conversion`, where the
conversion is 1.0 for a dollar quote and 0.01 for a US-cent quote. That factor of
100 is the whole reason coffee at 200 cents/lb is a $75,000 contract and not a
$7.5m one, and `check.py` asserts it for coffee, corn and ULSD directly.

Substituted from the brief's list: platinum and soybeans replace Dubai crude and
low-sulphur gasoil, neither of which has a free continuous series. The RBOB
ticker in the brief (XBN6) is not a December-2025 symbol under the standard month
codes; the continuous series sidesteps the question.

The common window is the intersection of dates where all thirteen printed, which
is 2,926 of 2,933 possible days. Settlement runs of four or more identical prints
are set to missing first; on this universe that flags nothing. Seven days drop out
because at least one contract did not print, and one of them is 2020-04-20, the day
WTI settled at -$37.63: a non-positive price is set to missing, so that day is
excluded for every contract and the return spanning it is a two-day return. The
same is true of the other six gaps; no missing print is filled forward.

## Periods

| window | dates | days | used for |
|---|---|---|---|
| training | 2015-01-05 to 2023-12-29 | 2,256 | screening, hedge ratios, integer choice |
| test | 2024-01-02 to 2026-08-31 | 669 | the out-of-sample numbers, nothing else |

Ratios are estimated on the training window, frozen, and applied to the test
window. The integer contract count is chosen from the training ratio, never by
searching the test window for the integer that happens to work.

## The spurious regression, and why everything here is in returns

This is the trap the project is built around, so I ran it deliberately rather
than just avoiding it.

Regressing price levels on price levels across all 78 pairs, training window:

| | price levels | daily returns |
|---|---|---|
| mean R-squared | 0.452 | 0.091 |
| median R-squared | 0.500 | 0.023 |
| max R-squared | 0.984 | 0.786 |
| pairs with R-squared > 0.5 | 39 of 78 | 5 of 78 |
| mean Durbin-Watson | 0.020 | 2.048 |
| pairs with \|t\| > 1.96 on the slope | 75 of 78 | 72 of 78 |

The Durbin-Watson row is the tell. A value of 0.02 against a null of 2 means the
level residuals are almost perfectly autocorrelated: the regression is not
explaining anything, it is fitting one trending series to another and leaving a
residual that is itself a random walk. The standard errors under that residual
structure are meaningless, which is why 75 of 78 slopes look significant.

The pairs where this is most obviously nonsense:

| pair | levels R-squared | levels DW | returns R-squared |
|---|---|---|---|
| copper vs corn | 0.704 | 0.02 | 0.020 |
| RBOB vs corn | 0.649 | 0.03 | 0.022 |
| coffee vs soybeans | 0.643 | 0.02 | 0.033 |
| WTI vs corn | 0.626 | 0.02 | 0.031 |

Copper does not explain 70% of corn. Both series went up over 2015-2023.

The control is the same code run on 200 pairs of *independent simulated random
walks*, 2,257 observations each (the length of the training price series). In levels they average R-squared 0.239 with a
Durbin-Watson of 0.008, and the slope is significant at 5% in 87% of pairs. In
first differences they average R-squared 0.00047 and reject at 4%, which is the
nominal rate. There is nothing to find in those series, and the level regression
finds it anyway 87% of the time.

`reports/spurious_regression.png` shows both distributions side by side.
`check.py` asserts the whole thing: high level R-squared and near-zero difference
R-squared on independent walks, plus a check that a hedge fitted to two fake
crude series delivers no out-of-sample variance reduction.

## Hedge ratios

I compute the ratio two ways and reconcile them, because they should agree and
the size of the gap is diagnostic.

**Return route.** Regress target returns on hedge returns, then scale by the
notional ratio: `h = beta * (P_i * mult_i) / (P_j * mult_j)`, evaluated at the
last training date. HAC standard errors with 5 lags.

**P&L route.** Convert prices to per-contract dollar P&L (`price change x
multiplier`) and regress directly. This is the minimum-variance ratio
`Cov(dV_i, dV_j) / Var(dV_j)` and it carries the contract sizes automatically.

The brief's worked examples reproduce exactly: a 0.92 WTI-on-Brent beta at $70
and $74 gives 0.870 Brent contracts, and a 0.75 WTI-on-ULSD beta at $70 and
$2.30/gal gives 0.544 ULSD contracts. Both are asserted in `check.py`.

On real data the two routes disagree by more than I expected:

| target | hedge | return beta (HAC se) | h via notionals | h via P&L | gap |
|---|---|---|---|---|---|
| WTI | Brent | 1.072 (0.050) | 0.997 | 0.919 | +8.4% |
| Brent | ULSD | 0.825 (0.044) | 0.593 | 0.434 | +36.6% |
| gold | silver | 0.404 (0.014) | 0.699 | 0.611 | +14.4% |
| WTI | ULSD | 0.908 (0.066) | 0.607 | 0.413 | +46.9% |
| RBOB | Brent | 0.815 (0.027) | 0.934 | 1.043 | -10.5% |

The reason is in the brief's section 14: the notional conversion is evaluated at
one reference date, but the return regression was fitted over nine years during
which the notional ratio moved a lot. WTI/ULSD is the worst case because the
crack spread widened substantially over the sample, so the single-date notional
ratio is not representative of the average. **The P&L regression is the one I
use**, precisely because it does not depend on picking a date. The return route
is kept because a gap this large is a useful alarm: it would also fire on a
cents-versus-dollars scale error, which is the failure mode I most wanted to
catch.

## Results

Leading hedges by training-window variance reduction, one long target contract.
The first eight are the top of the ranking; the last two are the best the
agricultural block produces, included because a table of nothing but energy and
metals would hide how thin the rest of the universe is.

| target | hedge | returns R-squared | h (contracts) | train | test |
|---|---|---|---|---|---|
| WTI crude | Brent crude | 0.786 | short 0.920 | +86.1% | +83.4% |
| Brent crude | WTI crude | 0.786 | short 0.936 | +86.1% | +83.5% |
| silver | gold | 0.610 | short 0.985 | +60.2% | +51.5% |
| gold | silver | 0.610 | short 0.611 | +60.2% | +39.2% |
| RBOB gasoline | Brent crude | 0.519 | short 1.043 | +59.4% | +62.6% |
| RBOB gasoline | WTI crude | 0.485 | short 1.042 | +58.2% | +59.7% |
| ULSD | Brent crude | 0.633 | short 1.123 | +48.7% | +60.3% |
| ULSD | WTI crude | 0.525 | short 1.088 | +45.0% | +62.3% |
| corn | soybeans | 0.291 | short 0.278 | +27.0% | +33.7% |
| SRW wheat | corn | 0.288 | short 0.789 | +22.3% | +31.6% |

Every relationship worth using is positive-correlation, so every hedge is a
short. The ranking by effectiveness is not the ranking by R-squared: WTI/ULSD has a higher return R-squared (0.525) than WTI/RBOB (0.485) but lower
training effectiveness, because effectiveness is measured in dollar-P&L space
where the contract multipliers and the changing notional ratio both matter and
R-squared is scale-free.

Across the top 15 ordered pairs the mean training effectiveness is 55.7% and the mean test
effectiveness is 59.5%, so the average pair got *better* out of sample. That is
not a mistake and it is not evidence of a good model; it is the difference
between hedging and forecasting. A minimum-variance hedge ratio is a description
of a contemporaneous covariance, not a prediction, so it does not decay the way a
trading signal does. The 2024-2026 window also happened to be a period of higher
crude co-movement, which flatters the energy pairs specifically.

Where it fails is the other 141 ordered pairs. **80 of the 156 ordered
target-hedge directions have negative test effectiveness**: the hedge increased
variance. Those are the pairs whose training R-squared was in the noise to begin
with: Henry Hub gas has no usable hedge anywhere in this universe (best training
effectiveness 1.1%, against corn/crude/anything), and coffee's best candidate is
copper at 3.3% training and 2.2% test, which is not a hedge, it is a coin flip
with margin requirements.

The pairs that survive are the ones with a physical story: same barrel (WTI,
Brent), same refinery output (ULSD, RBOB), same monetary-metal demand (gold,
silver, platinum), same acreage (corn, wheat, soybeans). Nothing crossed a group
boundary and survived.

### Risk detail on the leading pair

One long WTI contract hedged with 0.919 Brent, test window, dollars per day:

| | unhedged | hedged, continuous | hedged, 1 whole contract |
|---|---|---|---|
| variance reduction | | 83.4% | 82.2% |
| volatility reduction | | 59.2% | 57.8% |
| daily sd | $2,162 | $881 | $913 |
| 95% VaR | -$2,830 | -$811 | -$860 |
| 95% expected shortfall | -$5,288 | -$1,751 | -$1,784 |
| worst day | -$18,540 | -$6,621 | -$7,070 |
| days abs(hedged) < abs(unhedged) | | 89.4% | 88.6% |

Variance reduction of 83.4% is a volatility reduction of only 59.2%, which is
worth stating plainly because the first number is the one everyone quotes.

Walk-forward, re-estimating on a 60-day trailing window every day and applying it
the next day, gives 80.5% against 83.2% for the frozen ratio scored on the same 609
days (the walk-forward cannot score the first 60 test days). The frozen ratio wins
here, which says the relationship is stable enough that a rolling estimator is
just adding estimation noise. Mean daily change in the walk-forward ratio is
0.0064 contracts, so a desk rebalancing on a tolerance band would trade rarely.

### Integer contracts

WTI hedged with Brent, continuous 0.9195, evaluated on the test window:

| target contracts | continuous | whole | effectiveness (continuous) | effectiveness (whole) | residual notional |
|---|---|---|---|---|---|
| 1 | 0.92 | 1 | 83.4% | 82.2% | +$6,204 |
| 2 | 1.84 | 2 | 83.4% | 82.2% | +$12,408 |
| 5 | 4.60 | 5 | 83.4% | 82.2% | +$31,020 |
| 10 | 9.19 | 9 | 83.4% | 83.5% | -$15,000 |
| 25 | 22.99 | 23 | 83.4% | 83.4% | +$1,019 |
| 100 | 91.95 | 92 | 83.4% | 83.4% | +$4,078 |

At one contract, rounding 0.92 up to 1 costs 1.2 points of variance reduction. At
25 contracts the rounding error is 0.01 contracts and costs nothing measurable.
The 9-contract row is above the continuous ratio, which is not a free lunch: the
true test-window ratio is slightly below the frozen training ratio, so rounding
down happened to move toward it. That is luck, and `check.py` asserts the general
rule on planted data instead: rounding can never beat the continuous ratio
in-sample, and the amount it gives up is bounded by
`(n_int - n_cont)^2 * Var(dV_hedge) / (n_target^2 * Var(dV_target))`.

## Stability

60-day rolling, full sample:

| pair | h (train) | h median | h IQR | h CV | R-squared median | R-squared 5th pct | windows above 0.25 |
|---|---|---|---|---|---|---|---|
| WTI / Brent | 0.919 | 0.933 | 0.127 | 0.095 | 0.889 | 0.698 | 100.0% |
| ULSD / Brent | 1.123 | 1.061 | 0.202 | 0.191 | 0.732 | 0.261 | 95.6% |
| gold / silver | 0.611 | 0.672 | 0.182 | 0.222 | 0.633 | 0.428 | 98.9% |
| RBOB / Brent | 1.043 | 1.029 | 0.204 | 0.150 | 0.639 | 0.222 | 94.2% |
| RBOB / WTI | 1.042 | 1.006 | 0.205 | 0.151 | 0.614 | 0.235 | 94.7% |

No sign flips in any of these across the full sample. WTI/Brent never drops below
0.25 R-squared in any 60-day window in eleven years, which is as stable as
anything in this dataset gets.

Three equal subperiods, re-estimated on the full sample, tell a different story
about metals (`run.py` prints both directions of each leading pair):

| pair | 2015-2018 | 2018-2022 | 2022-2026 |
|---|---|---|---|
| WTI on Brent, h | 0.840 | 0.938 | 0.905 |
| WTI on Brent, R-squared | 0.765 | 0.881 | 0.853 |
| gold on silver, h | 0.674 | 0.585 | 0.391 |
| gold on silver, R-squared | 0.615 | 0.601 | 0.585 |

The gold-silver ratio fell 42% across the sample while its R-squared barely
moved. This is the case the brief warns about in section 17: relationship
strength and position size are separate stability questions, and here one is
stable and the other is not. A desk that estimated 0.674 in 2016 and left it
alone would be under-hedged by 40% by 2024, with an R-squared that never gave a
warning. It is also why gold-on-silver is the worst decay in the top table
(60.2% train to 39.2% test) while silver-on-gold holds up better: the drift is in
the direction that matters for the gold-target ratio.

Block bootstrap, 10-day blocks, 1,000 replications, training window:

| pair | h 2.5% | h 50% | h 97.5% | R-squared 2.5% | R-squared 97.5% |
|---|---|---|---|---|---|
| WTI / Brent | 0.887 | 0.921 | 0.950 | 0.823 | 0.892 |
| silver / gold | 0.899 | 0.987 | 1.072 | 0.556 | 0.646 |
| RBOB / Brent | 0.989 | 1.045 | 1.104 | 0.516 | 0.664 |
| RBOB / WTI | 0.987 | 1.043 | 1.096 | 0.497 | 0.662 |

Blocks rather than individual days because commodity returns cluster in
volatility. None of these intervals crosses zero or spans a whole contract, so
the ratios are precisely estimated; the risk in them is drift, not sampling
error, and the bootstrap cannot see drift.

## Limitations

**Roll returns.** The continuous series contain roll jumps that are not market
moves. Contracts with different roll schedules have those jumps on different
days, so the contamination is uncorrelated across contracts and pushes measured
R-squared down. I have not quantified it, because doing so needs the individual
contract series I do not have, but it means the numbers here are conservative, not
inflated, and it is the single biggest gap between this and the specified study.

**One train/test split, and the test window is short.** 669 days is under three
years, and it is a specific three years: crude co-moved unusually tightly in
2024-2026. The finding that effectiveness did not decay rests on that one window.
A rolling-origin evaluation across many splits would be the honest version and I
did not build it.

**Selection was not corrected for multiple testing.** 78 pairs screened, and I
report the top ones. With 2,256 observations the strong pairs are far outside
anything chance would produce, so I do not think the top of the table is noise,
but the marginal ones, copper/platinum at R-squared 0.156 and wheat/soybeans at
0.119, are exactly where a Bonferroni or Benjamini-Hochberg correction would
start biting, and I have not applied one.

**No costs, no liquidity, no margin.** Every number here is a variance ratio.
There is no bid-ask spread, no market impact, no exchange fee, no initial margin
and no variation-margin funding. Platinum in particular is thin enough that the
spread would matter; I cannot check it because the volume data for the metals
continuous series is unusable.

**Linear and contemporaneous.** Everything is a same-day OLS on daily closes. US
and European settlements are not simultaneous (Brent settles at 19:30 London,
WTI at 14:30 New York), so the WTI/Brent relationship is measured across a timing
mismatch, and some of the residual is a lead-lag effect rather than basis risk. I
did not run the lead-lag diagnostics the brief asks for in section 7.2.

**No structural-break test.** Section 18.4 asks for CUSUM or Chow. I have rolling
estimates and subperiod splits, which show the gold-silver drift clearly, but I
did not run a formal break test with a p-value.

## Files

- `data.py` contract specs with multipliers and ticks, cached download, stale-print filter, train/test split
- `hedging.py` HAC regressions, R-squared screen, minimum-variance ratios, unit conversion, rolling and bootstrap stability, spurious-regression demonstration
- `check.py` offline checks on a planted two-asset system
- `run.py` full pipeline, tables and charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

`check.py` is offline and needs no data. `run.py` downloads once to
`source-material/commodities/`, then reads the cache.
