# Market making with informed order flow

A simulated market with an efficient price, exponentially decaying fill intensity, and
order flow split between uninformed traders and traders who can see where the price is
going. Three quoting strategies run over the same 1,000 sessions, and every session's
P&L is split into three parts: the spread captured, the cost of being adversely selected,
and what the leftover inventory did.

The point of the project is the gap between those parts. Every fill in the simulator looks
like a winner at the moment it happens, because the maker always buys below the efficient
price and sells above it. The naive maker earns exactly 0.850 of spread on every one of
its 56 fills a session. It still ends up with a Sharpe of 3.65 against 7.15 for
Avellaneda-Stoikov, because 4.81 dollars a session go straight back out to the people who
knew something, and because it holds inventory it never chose.

## The market

One session is a horizon of 1 split into 500 steps. Parameters follow the numerical
example in Avellaneda and Stoikov (2008) section 3.3 so the closed form is being run near
where its authors ran it.

| | value | meaning |
|---|---|---|
| `s0` | 100 | starting efficient price |
| `sigma` | 2.0 | dollar volatility over the session, 2% of price |
| `n_steps` | 500 | |
| `arrival_rate` (A) | 140 | order intensity at zero quote distance |
| `decay` (kappa) | 1.5 | how fast flow dries up with distance |
| `informed_frac` | 0.30 | share of arriving orders that are informed |
| `informed_lag` | 100 steps | how far ahead the informed traders can see |
| `position_limit` | 20 | hard risk backstop, quotes are pulled on one side at the limit |

The efficient price is geometric Brownian motion in logs. A jump component is available
and off in the base case; the stress table below turns on four jumps a session of 1.5%
log size.

Order flow is a Poisson process on each side with intensity `lambda = A * exp(-kappa * delta)`,
where `delta` is the distance from my quote to the efficient price. Each arriving order is
informed with probability `informed_frac`. An uninformed order trades in a random direction.
An informed order knows the efficient price 100 steps ahead and only trades if that price is
on the far side of my quote: it lifts my ask only when the price is heading above the ask,
and hits my bid only when the price is heading below the bid. At `informed_frac = 0` this
is exactly the Avellaneda-Stoikov flow model.

The informed horizon is not arbitrary. `sigma * sqrt(h/T) = 0.89` against a half-spread of
0.85, so an informed trader's information is worth about one half-spread. Shorter and they
essentially never trade, because they cannot clear the spread; longer and every fill is a
pick-off. That calibration is the single most consequential parameter in the project and I
sweep the whole informed fraction from 0 to 1 rather than defending one point on it.

## The strategies

All three quote one lot a side and see the efficient price exactly.

**Naive.** Fixed 0.85 either side, no reaction to inventory.

**Inventory-skewed.** Both quotes shift by `-0.15 * q`. Hand-tuned, no derivation.

**Avellaneda-Stoikov.** The closed form from Avellaneda and Stoikov, *High-frequency trading
in a limit order book*, Quantitative Finance 8(3), 2008:

```
reservation price  r = s - q * gamma * sigma^2 * (T - t)
optimal spread       = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)
```

with `gamma = 0.1`. Their price process is arithmetic Brownian motion; mine is GBM, so I
read their `sigma` as a local dollar volatility and rescale it by `price / s0`. I have not
re-derived the result and I do not claim it is optimal in my market, which differs from
theirs in exactly one way that matters: theirs has no informed flow.

Both baselines are tuned in their own favour, so the comparison is not rigged. Naive's 0.85
is the top of its own half-spread sweep (37.88 at 0.50, 42.67 at 0.85, 25.46 at 1.70), and
0.15 is near the top of the skew sweep.

## P&L decomposition

For a fill of signed size `q` at price `p` when the efficient price is `S_t`, the value of
that fill at the end of the session is `q * (S_T - p)`, and

```
S_T - p = (S_t - p) + (S_{t+h} - S_t) + (S_T - S_{t+h})
```

Summed over fills, the three terms are the spread captured at the moment of the trade, the
markout over the informed trader's horizon `h`, and everything the remaining position did
after that. The split is an algebraic identity, so the parts sum to the total exactly rather
than approximately. `check.py` asserts it to 1e-9 on every strategy, and asserts separately
that the total equals cash plus inventory marked at the final price.

The markout horizon is a choice. Moving `h` moves money between the second and third terms
and leaves the total alone. I set `h` to the informed traders' horizon because that is the
window over which their information is realised, which is the quantity the decomposition
is meant to isolate.

## Results

1,000 sessions per strategy, same price paths and same order flow for all three, so the
differences are not sampling noise.

| | naive | skewed | avellaneda-stoikov |
|---|---|---|---|
| mean P&L | 42.69 | 44.17 | 44.17 |
| P&L sd | 11.70 | 6.51 | 6.18 |
| Sharpe (per session) | 3.65 | 6.78 | 7.15 |
| 5th percentile P&L | 22.87 | 33.78 | 34.51 |
| mean max drawdown | -7.20 | -1.73 | -1.87 |
| worst drawdown | -49.10 | -6.56 | -11.90 |
| inventory sd | 2.85 | 1.31 | 1.38 |
| mean max abs inventory | 8.85 | 3.55 | 4.15 |
| worst abs inventory | 20 | 6 | 12 |
| fills | 56.2 | 58.8 | 70.3 |
| steps at the position limit | 0.72 | 0 | 0 |

Decomposition, mean dollars per session:

| | naive | skewed | avellaneda-stoikov |
|---|---|---|---|
| spread captured | +47.77 | +45.97 | +46.21 |
| adverse selection | -4.81 | -1.77 | -2.10 |
| inventory P&L | -0.27 | -0.03 | +0.05 |
| **total** | **42.69** | **44.17** | **44.17** |
| spread per fill | 0.850 | 0.781 | 0.658 |
| adverse selection per fill | -0.086 | -0.030 | -0.030 |

Three things in that table.

The spread captured is nearly identical across strategies. Every maker gets its quoted
edge on every fill, always. That is the easy half of the job and no strategy is better at
it than any other.

Naive pays 2.7 times the adverse selection cost of the skewed maker on 5% fewer fills, 2.3
times Avellaneda-Stoikov's on 20% fewer, and its per-fill cost is nearly three times either. That is not because it gets picked off more
often on any single quote. It is because once informed flow has pushed it onto one side of
the market, it keeps quoting the same price and gets picked off repeatedly in the same
direction. The inventory-skewed makers move away after the first hit.

The mean P&L difference is small (1.47 dollars, t = 4.68 against skewed and 4.58 against
Avellaneda-Stoikov on paired sessions) and the risk difference is large. Naive's P&L
standard deviation is 1.8 times the skewed maker's and 1.9 times Avellaneda-Stoikov's, and
its worst drawdown is 7.5 and 4.1 times worse. `reports/pnl_distribution.png` shows the
same mean with a much fatter left tail.

### Jump market

Four jumps a session of 1.5% log size, everything else unchanged. Informed traders see the
jumps coming; the naive maker does not.

| | naive | skewed | avellaneda-stoikov |
|---|---|---|---|
| mean P&L | 37.96 | 43.12 | 42.50 |
| Sharpe (per session) | 1.81 | 5.50 | 5.42 |
| 5th percentile P&L | -0.73 | +30.30 | +29.77 |
| losing sessions | 5.0% | 0.0% | 0.0% |
| worst drawdown | -120.27 | -22.62 | -40.03 |
| adverse selection | -10.78 | -4.00 | -4.75 |
| steps at the position limit | 1.38 | 0 | 0 |

This is where the naive maker actually breaks. Its Sharpe falls to 1.81, one session in
twenty loses money, and its worst session drew down 120 dollars against a 38 dollar mean.
`reports/sample_session.png` is one of those sessions, the 0.5th percentile of the naive
distribution, with both makers on identical prices and identical flow: the price trends up,
naive is sold to repeatedly on the way, it spends the last third of the session at or
within two lots of its -20 position limit, takes a jump against the position, and ends
-45.4. The
Avellaneda-Stoikov maker on the same flow never exceeds 5 lots and ends +41.0.

### The informed-fraction sweep

400 sessions per point, base market otherwise.

| informed | naive P&L | skew P&L | AS P&L | naive Sharpe | skew Sharpe | AS Sharpe | naive adv sel | AS adv sel |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 63.42 | 61.52 | 61.10 | 5.01 | 8.43 | 8.83 | -0.34 | +0.01 |
| 0.1 | 56.54 | 55.97 | 55.56 | 4.81 | 7.88 | 8.41 | -1.81 | -0.77 |
| 0.2 | 49.40 | 49.90 | 49.77 | 4.27 | 7.40 | 7.73 | -3.47 | -1.53 |
| 0.3 | 42.67 | 44.16 | 44.10 | 3.89 | 6.81 | 6.99 | -4.92 | -2.17 |
| 0.4 | 35.83 | 38.18 | 38.19 | 3.28 | 6.17 | 6.31 | -6.35 | -2.84 |
| 0.6 | 22.10 | 25.35 | 25.67 | 2.07 | 4.67 | 4.88 | -9.50 | -4.72 |
| 0.8 | 8.67 | 11.96 | 12.14 | 0.81 | 2.73 | 2.81 | -12.52 | -7.14 |
| 1.0 | -4.82 | -2.51 | -2.98 | -0.47 | -0.62 | -0.75 | -15.53 | -10.76 |

I expected the naive maker to collapse here while Avellaneda-Stoikov degraded gracefully.
That is not what happened, and the distinction matters.

On mean P&L, all three decline nearly linearly and nearly together. Nobody collapses and
nobody is graceful. At 30% informed flow the entire advantage of the closed form over doing
nothing about inventory is 1.47 dollars on 44, about 3%. Below roughly 15% informed flow
naive earns slightly *more* than either alternative, because it quotes wider per fill and
there is not enough adverse selection yet to punish it for standing still.

On risk-adjusted return the collapse is real. Naive's Sharpe falls from 5.01 at no informed
flow to 0.81 at 80%, an 84% fall. Avellaneda-Stoikov falls from 8.83 to 2.81, a 68% fall.
The ratio between them widens the whole way: 1.8x at no informed flow, 1.9x at 40%, 3.5x at
80%. Naive's adverse selection cost also grows about twice as fast per unit of informed
flow, -15.53 against -10.76 at the extreme.

At 100% informed flow every maker loses money, which is the sanity check. When the only
counterparties are people who know where the price is going and who only trade when they
profit net of the spread, no quoting rule earns anything; the only winning move is to not
quote. `check.py` asserts this for all three makers.

The honest summary is that inventory skew is a risk-management tool, not an alpha source.
It converts the same mean P&L into half the variance. That is a large improvement for a
real desk, which is levered and cares about drawdown, and a small one on a mean P&L table.

### Does the closed form beat a guess?

Almost not at all. The skewed maker and the Avellaneda-Stoikov maker have mean P&Ls of
44.167 and 44.168 in the base market, and Sharpes of 6.78 and 7.15. The paired difference
between them is inside noise.

Avellaneda-Stoikov gets there differently. It quotes narrower (0.658 of spread per fill
against 0.781), so it takes 19% more fills for the same money. It also has a `(T - t)`
term, which the hand-tuned skew does not: its inventory penalty shrinks to zero at the
close, which is why its worst inventory over 1,000 sessions is 12 against the skewed
maker's 6. The model is deliberately indifferent to a position it has no time left to lose
money on. That is correct under its assumptions and wrong in any market that reopens
tomorrow.

What the closed form actually buys is that its two free parameters are `gamma` and `kappa`,
both of which mean something and one of which is estimable from your own fill data, whereas
the skew of 0.15 was found by running a sweep on the answer.

## Limitations

**The maker sees the efficient price exactly.** This is the biggest unrealism in the
project and it makes every strategy look better than it should. A real maker infers a mid
from a noisy order book, and its estimate is worst precisely when the market is moving,
which is when being wrong costs most. Adding observation noise would hurt the naive maker
most, since it has no other mechanism for noticing it is on the wrong side.

**The Sharpes are not real numbers.** 3.65 to 7.15 per session annualises to 58 to 113,
which no market maker has ever run. The simulator has one maker with no competition, no
exchange fees or rebates, no queue position, no latency, no minimum tick, one-lot fills,
and no capital charge. The levels here are meaningless. Only the comparisons between
strategies, which face all the same missing frictions, carry information.

**This is a single-asset equity model, not an options book.** An options market maker's
inventory is not a share count, it is a vector of greeks, and the analogue of skewing your
quotes is hedging your delta and pricing your vega inventory into the next quote. The
adverse selection story carries over directly and gets worse, since option flow contains
volatility information as well as direction. None of that is here.

**Informed traders have perfect foresight over a fixed horizon.** They know `S_{t+h}`
exactly and trade only when it clears my quote. That is the strongest possible form of
adverse selection per fill, and it is also why total volume falls as the informed fraction
rises: informed traders decline the trades they would lose on, so raising the informed
fraction removes uninformed flow without fully replacing it. Part of the naive maker's
P&L decline across the sweep is that mechanical volume effect rather than pick-off, and the
decomposition separates them: for the naive maker, going from 0% to 100% informed drops
fills from 74.9 to 11.9 and spread captured from 63.67 to 10.08, while adverse selection
rises from 0.34 to 15.53. Most of the P&L decline is lost volume, not pick-off, and only
the decomposition makes that visible.

**The position limit flatters the naive maker.** It binds only for naive, on 0.72 steps a
session in the base market and 1.38 in the jump market, and it binds exactly in its worst
sessions. Naive's worst drawdown of -49 would be worse without it. `check.py` runs the
inventory comparison with the limit effectively disabled so it is testing the quoting rule
rather than the backstop.

**Fill intensity uses one distance for both a passive quote and an aggressive one.** If a
strategy quotes through the efficient price the intensity formula gives a fill probability
above the arrival rate, which is the right sign but is not calibrated to anything. This
matters for the Avellaneda-Stoikov maker at large inventory, where the reservation price
shift can exceed the half-spread.

**No cost of capital and no inventory financing.** Holding 20 lots overnight is free here.
That understates the cost of the naive maker's inventory specifically.

## Files

- `sim.py` market parameters, price path, order flow, fills, P&L decomposition
- `maker.py` the three quoting strategies behind one `quote()` interface
- `check.py` offline assertions on planted truth
- `run.py` full comparison, tables and charts to `reports/`

## How to run

```
../.venv/bin/python3 check.py
../.venv/bin/python3 run.py
```

Both are offline. `check.py` takes about 20 seconds, `run.py` about 15.

## Reference

Marco Avellaneda and Sasha Stoikov, *High-frequency trading in a limit order book*,
Quantitative Finance 8(3), 217-224, 2008.
