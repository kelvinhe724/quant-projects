# SVI volatility surface on a live SPY option chain

Fits the raw SVI parameterisation

```
w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
```

one slice per expiry, to a cleaned SPY option chain. `w` is total implied
variance and `k` is log-moneyness against the forward. The calibration targets
mid total variance, weights each quote by its bid-ask width, and enforces both
static arbitrage conditions: Durrleman's butterfly condition within each slice,
and monotone total variance across slices.

## Data

Real option chains from Yahoo Finance via `yfinance` (`Ticker.option_chain`),
cached to `reports/chain_SPY_<date>_<time>.csv` so a rerun does not re-hit the
network. The spec this project came from asks for SPX chains out of a Bloomberg
OMON export; I do not have a terminal, so this is Yahoo SPY instead. SPY options
are American on a dividend-paying ETF, where SPX options are European on an
index, and that difference is the largest single approximation in the project.
See Limitations.

The snapshot below is 2026-09-03 03:42 UTC, SPY at 765.16, eight listed
expiries from 1 week to 1 year.

### Cleaning

| filter | quotes left |
|---|---|
| raw calls and puts, 8 expiries | 2,456 |
| bid > 0 | 2,278 |
| ask > bid | 2,278 |
| last trade within 1 day | 1,136 |
| non-zero volume | 1,136 |
| spread ≤ 25% of mid, or ≤ $0.10 | 725 |
| a parity forward exists for the expiry | 725 |
| \|k\| ≤ 0.5 | 718 |
| out-of-the-money side only | 413 |
| ≥ 12 quotes in the slice | 407 |

**407 clean quotes across 7 expiries.** The one-week expiry is dropped by the
last filter: after the liquidity cuts it has 6 usable strikes, which is not
enough to identify five parameters.

The stale-quote filter is the one that bites hardest, removing half the chain.
Yahoo returns every listed contract with a live bid-ask whether or not anyone
has traded it this month, and `lastTradeDate` is the only staleness signal in
the feed.

Keeping only the out-of-the-money side halves the data again. In-the-money
quotes carry the same information through put-call parity but trade wider, and
including both would double-count every strike.

### Forwards and discounting

No dividend yield is assumed anywhere. For each expiry the forward comes out of
put-call parity, `F = K + (C - P) / D`, taken as the median over strikes within
5% of spot. The discount factor is `exp(-rT)` off the 13-week bill (`^IRX`,
3.77% on the snapshot date), so the only rate assumption is a flat curve.

The forward basis `F/S - 1` runs from 0.10% at one week to 3.65% at one year,
which is the rate minus the dividend yield compounding out. It is not quite
monotone at the front: the three-week expiry (0.06%) sits below the one-week
(0.10%) because it straddles SPY's mid-September ex-dividend date and the
one-week does not. That dip is the dividend showing up in the parity forward,
which is what it is there for. I first tried backing out both `F` and `D` by regressing `C - P` on `K`;
the slope came back at 0.9998 for a one-year expiry, implying a zero rate.
Two free parameters were not identified from mid quotes this noisy, so I fixed
the discount factor externally.

## My implied vols vs the yfinance column

Every implied vol is computed here: Black-76 on the parity forward, inverted
with Brent's method on `[1e-4, 5.0]`, with quotes outside the no-arbitrage price
bounds returned as NaN.

Against Yahoo's own `impliedVolatility` column, on the same 407 quotes: mean
difference **-0.04 vol points**, RMSE **1.41**, max absolute difference **4.34**.

The mean agreement is close to perfect and the dispersion is not, which points
at inputs rather than at the solver. Yahoo appears to invert `lastPrice` rather
than the mid, so its number is as stale as the last trade, and it uses its own
rate and dividend assumption instead of a parity forward. The disagreement is
widest on the deep wings, where a single tick of price is worth several vol
points. Everything downstream of the vol inherits that noise, which is the
argument for computing it yourself.

## Static arbitrage before fitting

Butterfly, straight off the mid quotes: for each adjacent strike triple the
butterfly `(K3-K2)/(K3-K1)·C1 + (K2-K1)/(K3-K1)·C3 - C2` must be non-negative.

| T | negative butterflies | worst |
|---|---|---|
| 0.062 | 9 / 43 | -0.50 |
| 0.081 | 22 / 84 | -0.40 |
| 0.158 | 39 / 124 | -0.45 |
| 0.292 | 14 / 35 | -0.27 |
| 0.407 | 11 / 36 | -0.75 |
| 0.574 | 5 / 26 | -0.51 |
| 1.040 | 9 / 45 | -0.83 |

Roughly a quarter of quoted butterflies are negative. These are not free money:
they are mid prices assembled from three different bid-ask spreads that are
never all crossable at once, plus quotes struck at slightly different instants.
It is exactly why fitting the mid needs a no-arbitrage constraint bolted on
rather than trusting the data to be consistent.

Calendar, on interpolated market total variance: **0 of 6** adjacent expiry
pairs violate. The term structure in the raw data is already monotone.

## Results

Constrained fit, inverse-spread weighting, ridge `lambda = 0.01`, bid-ask
penalty `eta = 0` (see below for why):

| expiry | T | n | a | b | rho | m | sigma | RMSE (vol pts) | outside band |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | 0.062 | 45 | -0.0078 | 0.050 | -0.168 | 0.015 | 0.171 | 0.30 | 4 |
| 2026-10-02 | 0.081 | 86 | -0.0082 | 0.054 | -0.168 | 0.018 | 0.171 | 0.35 | 7 |
| 2026-10-30 | 0.158 | 126 | -0.0093 | 0.068 | -0.171 | 0.031 | 0.169 | 0.20 | 7 |
| 2026-12-18 | 0.292 | 37 | -0.0104 | 0.092 | -0.174 | 0.038 | 0.165 | 0.25 | 3 |
| 2027-01-29 | 0.407 | 38 | -0.0092 | 0.099 | -0.176 | 0.048 | 0.164 | 0.24 | 2 |
| 2027-03-31 | 0.574 | 28 | -0.0072 | 0.109 | -0.179 | 0.066 | 0.161 | 0.16 | 0 |
| 2027-09-17 | 1.040 | 47 | -0.0006 | 0.130 | -0.188 | 0.121 | 0.161 | 0.26 | 3 |

Mean RMSE **0.250 vol points**, worst slice 0.35. The typical bid-ask width on
these quotes is one to two vol points, so the fit sits well inside the spread:
26 of 407 fitted vols land outside their own bid-ask band, 18 of them on the
three shortest expiries.

### What arbitrage enforcement costs

| | butterfly violations | calendar violations | mean RMSE |
|---|---|---|---|
| unconstrained | 7 / 7 slices | 5 / 6 pairs | 0.152 vol pts |
| constrained | 0 / 7 slices | 0 / 6 pairs | 0.250 vol pts |

This is the headline. An unconstrained least-squares SVI fit to real SPY quotes
produces a surface that is arbitrageable **on every single slice** and across
five of six expiry pairs, and it is the *better-fitting* surface, by 0.10 vol
points. Chasing the mid without constraints buys a tenth of a vol point of fit
and a surface you cannot quote off.

Per slice, unconstrained to constrained, in vol points:
0.261→0.301, 0.149→0.349, 0.118→0.197, 0.125→0.249, 0.160→0.240,
0.069→0.156, 0.180→0.259.

The short end pays the most. The Durrleman panel in `reports/residuals.png`
shows why: `g(k)` for the two shortest slices sits on zero around k = 0.2, so
the butterfly constraint binds right where the call wing has almost no quotes to
argue with it.

The calendar arbitrage is created by the fitting, not present in the data. The
market total variances are monotone across all six pairs; seven independently
fitted slices cross each other once each extrapolates past its own strike range,
which differs slice to slice. Fitting in maturity order with the previous slice
as a floor removes it, and costs almost nothing, because the constraint only
binds outside the quoted strikes.

### The bid-ask penalty does not earn its place

The spec calls for penalising fitted vols that escape the bid-ask band. I
implemented it (`eta`) and swept it, with and without the inverse-spread
weighting of the mid errors:

| eta | weighted RMSE | outside band | unweighted RMSE | outside band |
|---|---|---|---|---|
| 0 | 0.250 | 26 / 407 | 0.281 | 47 / 407 |
| 1 | 0.264 | 36 / 407 | 0.263 | 42 / 407 |
| 10 | 0.249 | 31 / 407 | 0.251 | 41 / 407 |
| 100 | 0.248 | 30 / 407 | 0.238 | 31 / 407 |

Unweighted, the penalty does exactly what it is supposed to: band violations
fall from 47 to 31 and RMSE from 0.281 to 0.238 as `eta` goes from 0 to 100.
Weighted by inverse squared band width, `eta = 0` already gives 26 violations,
better than anything the penalty achieves unweighted, and turning `eta` up from
there does nothing but shuffle the optimiser between local minima.

So the two mechanisms are substitutes. Weighting each quote by
`1/band^2` is a soft version of the same instruction, applied at every quote
rather than only at the ones that breach, and it is better conditioned because
it is smooth. I ship `eta = 0` and keep the weighting. The penalty stays in the
code because it is the right tool if the errors are ever fitted unweighted.

### Ridge across the expiry ladder

| lambda | mean RMSE | parameter churn |
|---|---|---|
| 0 | 0.263 | 1.381 |
| 0.01 | 0.250 | 0.230 |
| 0.1 | 0.485 | 0.200 |
| 1.0 | 1.416 | 0.095 |

Churn is the summed absolute parameter change between adjacent expiries.
`lambda = 0.01` cuts it by 83%, and on this chain the fit gets slightly better
too, because the `lambda = 0` run lands in a worse local minimum on two of the
slices. Past 0.1 the ridge eats the term structure and the fit falls apart.

What the ridge reveals matters more than what it costs. Fitting each slice
independently, `rho` runs +0.37, +0.42, +0.04, +0.36, -0.51, +0.38, +0.70 down
the ladder with no pattern, `m` from -0.10 to +0.56 and `sigma` from 0.18 to
0.53, while the fitted
*curves* barely move. Constrain the slices but leave the ridge off and `rho`
still wanders between -0.01 and -0.22 and `sigma` from 0.15 to 0.44, with
neither moving monotonically in maturity. Add
`lambda = 0.01` and `rho` sits between -0.168 and -0.188 at every maturity,
`sigma` between 0.161 and 0.171, `m` rising smoothly from 0.015 to 0.121.

Those swings were not term structure. They are the standard degeneracy between
`b`, `rho`, `m` and `sigma`, which can produce nearly the same smile from very
different parameter vectors, and a small ridge is enough to pick the stable
branch. A related symptom: with `lambda = 0` the fit is multi-modal enough that
rewriting a constraint that never binds (the wing-slope bound, which sits an
order of magnitude above any fitted `b`) changed the optimiser's path and moved
mean RMSE from 0.21 to 0.26 and churn from 1.70 to 1.38. At `lambda = 0.01` the
same change moved nothing at three decimals. Given a fixed cached chain and a
fixed code path, every number here reproduces exactly.

## Files

- `data.py` chain download and cache, parity forwards, cleaning funnel
- `svi.py` Black-76 and the implied-vol solver, raw SVI, Durrleman and calendar
  checks, constrained multi-start calibration
- `check.py` offline checks against planted parameters and a planted arbitrage
- `run.py` full pipeline, tables and charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py    # offline, 23 checks
../.venv/bin/python3 run.py      # cached chain, ~2 min including both sweeps
../.venv/bin/python3 run.py --refresh   # pull a new snapshot first
```

`run.py` reuses the newest cached chain in `reports/`, so it never touches the
network unless you pass `--refresh` or there is no cache yet.

Charts land in `reports/`: `smiles.png` (market mid, bid-ask band and fit per
expiry), `surface.png` (3D fitted surface plus the total variance slices),
`residuals.png` (residuals and the Durrleman function), `params.png` (parameter
term structure, free vs constrained), `iv_comparison.png` (my vols against
Yahoo's).

## Limitations

**SPY is not SPX.** SPY options are American on a dividend-paying ETF; I price
them with a European formula. Early exercise is worth close to nothing for OTM
options, which is all I fit, so the error is small, but it is not zero and it is
one-sided. SPX or SPXW quotes would remove the approximation entirely.

**One snapshot.** The spec's ridge penalty is meant to run across time
snapshots, keeping parameters stable from one repricing to the next. With a
single chain there is no time axis, so my ridge runs across the expiry ladder
instead. It answers a related question, whether the parameter path is smooth in
maturity, and not the one about repricing stability. Multiple snapshots would
need the run scheduled through a trading day, and the cache is already keyed by
timestamp so that the panel accumulates.

**The call wing is thin.** SPY OTM calls barely trade at these strikes, so most
slices are fitted almost entirely on puts, with k running to -0.45 and only to
+0.1 or so. `rho` and the right-wing slope `b(1+rho)` are weakly identified by
the data and are effectively being set by the butterfly constraint and the
ridge rather than by quotes. The left wing is the trustworthy half of every
fitted smile, and `rho` should not be read as a standalone skew number.

**Residuals are structured, not noise.** `reports/residuals.png` shows a
consistent hump: the fit sits below the market around k = -0.05 and above it in
both wings, on nearly every slice. Five parameters cannot bend that far. This is
the usual argument for eSSVI or a two-regime fit if the surface has to be
tighter than a quarter vol point.

**Enforcement is by penalty, not by construction.** Butterfly and calendar are
imposed as large quadratic penalties on a 301-point grid over k in [-1.5, 1.5],
and verified afterwards on the same grid, which grades the fit on the same points
it was trained to satisfy. A parameterisation that is arbitrage-free by
construction (Gatheral-Jacquier SSVI) would not need a grid at all. Between grid
points, and outside |k| = 1.5, nothing is guaranteed.

**The forward carries a flat-rate assumption.** One 13-week bill rate is used to
discount every expiry out to a year. At current rate levels the error in `D` is
a few tens of basis points at the long end, which moves the forward by a similar
amount and the fitted vols by well under a tenth of a vol point. Small, but it
grows with maturity and would matter on a longer surface.
