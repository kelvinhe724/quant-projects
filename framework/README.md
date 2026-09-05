# Premia book

A paper-traded risk-premia book on one daily engine, run once per NYSE session into two books (the engine's own shadow fills and an Alpaca paper account, or a simulated stand-in while there are no paper keys) through a broker layer that carries the kill switch and the limits. Built to the plan in `outputs/2026-09/2026-09-04_trading-framework-research/PLAN.md`; every number below is recomputed by `book/validate.py` and lives in `book/reports/`.

Three versions so far. v1 was three sleeves at 1/N: ETF trend, FX carry through currency ETFs, long/flat crypto trend. v2 added a continuous EWMAC trend rule as a candidate. v3 (2026-09-04) moved the position buffer into the engine, ran EWMAC through the untouched window, compared four books on that window and put the winner live: **`beta+alpha`**, a vol-targeted 1/N of the 14 ETFs as the core and EWMAC on top, 1/2 each. 2026-09-05: a cost-aware mean-variance allocator (`book/optimizer.py`) was run as a candidate through the same four-book comparison and lost on every book; it stays documented below and 1/N stays live.

The one-line honest summary: the live book's net Sharpe is 0.81 over 23 years (DSR 0.63 after deflating for the 60 configurations tried), and 0.79 on the untouched last fifth of the panel against 0.40 for the v1 book. Most of that is the core: on the untouched window the ETF universe held 1/N scores 0.98 by itself, and EWMAC alone scores 0.41 there and fails its own promotion test. The book is live because the rule written down in advance says the out-of-sample winner goes live; the rule does not say the winner has skill, and this one mostly has beta.

## Sleeves

- **TrendETF** ports `trend-following/trend.py` (12-1 sign, 40% per-asset vol, class-balanced, month-end rebalance). That specification was frozen before results were seen and the 14-ETF universe was the project's own cross-check universe, so it is v1 as written. The project's own numbers are in `trend-following/README.md` (class-balanced Sharpe 0.41 gross / 0.37 net on its futures universe over 2005-2026; 0.52 net on the 14-ETF cross-check that this book trades, `book/reports/trend_etf_vs_project.csv`).
- **FXCarryETF** ports `fx-carry/carry.py::carry_weights` onto the six CurrencyShares ETFs Alpaca lists (FXE, FXY, FXB, FXA, FXC, FXF), ranked with USD on the same OECD 3-month rates from FRED, top-2 / bottom-2. USD in a leg is cash. The project (`fx-carry/README.md`) got Sharpe 0.35 on ten currencies with explicit rate differentials; see the cash convention below for why this port does not.
- **CryptoTrend** is the same 12-1 rule long/flat on BTC and ETH from Alpaca's crypto bars (2021 on). Alpaca cannot short crypto.
- **EWMAC** (v2) is a continuous-forecast trend rule on the same 14 ETFs: fast minus slow EWMA over a 35-day price vol, three speeds (16/64, 32/128, 64/256) combined with a diversification multiplier, forecast scalars fit on trailing data only, cap 20, forecast / 10 times TrendETF's 40% per-asset target with class balance, sent every day.
- **ETFBeta** (v3) is 1/N of the ETFs with a price that day, sent every day; the overlay sizes it to the 10% vol target. It exists because the attribution table below kept saying that owning the universe beat the rules run on it.
- **Books.** `strategies.BOOKS` names four: `v1` (TrendETF, FXCarryETF, CryptoTrend), `v1+ewmac`, `ewmac-for-trend` (EWMAC in TrendETF's place) and `beta+alpha` (ETFBeta, EWMAC). Every book is 1/N of its sleeves, static. `allocate.LIVE_BOOK` names the one the daemon runs.
- **Allocator.** `portfolio-construction/` found nothing that beat 1/N across 14 ETFs at p < 0.10, so the plan's rule is that ERC across *sleeves* (skfolio `RiskBudgeting`, Ledoit-Wolf, three-year window, quarterly refit) has to beat 1/N out of sample before it goes live. It has not for any book (v1: 0.27 vs 0.28; beta+alpha: 0.86 vs 0.87), so the live allocator is equal weight. A cost-aware mean-variance allocator (`book/optimizer.py`, below) was run as a candidate on 2026-09-05 through the four-book untouched-window comparison and lost (0.66 vs 0.79 on the live book); `allocate.LIVE_ALLOCATOR = "rule"` keeps the ERC-or-1/N rule live.

Not in the book: `funding-carry/` (needs a perp venue) and `vol-risk-premium/` (a daily-hedged option book is a separate daemon). No futures (PLAN risk #3).

Candidate sleeves are run through the same promotion rule before they touch the book: net Sharpe above zero on the untouched window, above the incumbent sleeve there, and a positive alpha t against the ETF universe. `../rates-carry/` (carry and rolldown on the Treasury curve, 1962-2026) was tested on 2026-09-04 and did not pass. Against IEF+TLT it has +1.4% a year of alpha at t 1.66 and a residual Sharpe of 0.33 against the benchmark's own 0.39, so it stays a standalone project and is not a counted trial here.

## Engine and cost model

Shadow cost model throughout: 5 bps commission, 2 bps half spread, sqrt impact (k = 0.1), 50 bps a year borrow on shorts, next-open fills. Overlay per sleeve: 10% vol target, 3x gross cap, half size past a 15% drawdown, kill at 25% (off for backtests), and since v3 a **10% position buffer on the final weights**: after the overlay has scaled a sleeve's targets, an instrument trades only when its held weight is outside target x (1 +/- 0.1); a gross or per-instrument breach, a change in the drawdown cut or a kill sends every target regardless. The Alpaca bridge applies the same band against the paper account's positions. Sharpe is on daily returns, days before a sleeve is live excluded. **Gross** is the same path with each day's trade costs and borrow added back; a separate zero-cost run is no longer the same path, because the buffer trades off held weights and those depend on what costs have done to equity (a zero-cost TrendETF run lands at 0.41 against 0.46 net for that reason, which is noise, not a cost).

## Headline

Sleeves alone, kill off, buffered engine, whole panel:

| | Sharpe gross | Sharpe net | ann. return net | ann. vol | max drawdown | live from | trades |
|---|---|---|---|---|---|---|---|
| TrendETF | 0.55 | 0.46 | +3.6% | 8.4% | -20.3% | 2004-07 | 2,564 |
| FXCarryETF | -0.07 | -0.16 | -1.2% | 6.2% | -38.7% | 2006-07 | 440 |
| CryptoTrend | 0.10 | 0.09 | +0.4% | 9.4% | -16.3% | 2022-02 | 50 |
| EWMAC (from 2005-06) | 0.84 | 0.68 | +6.6% | 10.2% | -19.4% | 2005-06 | 27,050 |
| ETFBeta | 0.68 | 0.66 | +5.3% | 8.4% | -26.5% | 2003-06 | 3,770 |
| v1 book, 1/3 each, static | 0.36 | 0.25 | +1.0% | 4.3% | -12.7% | 2004-07 | 3,051 |
| **beta+alpha book, 1/2 each, static (live)** | 0.92 | **0.81** | +5.7% | 7.2% | -12.1% | 2003-06 | 30,820 |

Book honesty (`book/reports/validation.md`, run 2026-09-05 02:59): live book PSR 1.00, **DSR 0.63** with n_trials = 60 (every configuration `validate.py` has run on the real panel, before and after the buffer moved into the engine, identical return streams counted once; 0.57 if the 20 placebo draws are counted too), MinBTL 8.4 years against 23.2 in hand, MinTRL 4.3 years. Those last three are what a 0.8 Sharpe over 23 years buys, and for a book whose core is 0.67 beta to its universe they are statements about the universe. The v1 book on the same engine: 0.25 net, PSR 0.88, DSR 0.02 at 47 trials (the count when it was the live book, from the run before this one). Cost stress on the live book: 2x costs 0.73, 4x 0.55. A one-bar signal delay does nothing (0.82; +5 bars 0.76), which is what a book that is mostly a buy-and-hold core looks like.

Two results from the falsification battery that matter more than any headline:

- **Vol scaling without the signal beats the signal.** The trend sleeve with every sign forced long scores 0.62 against 0.46 with the 12-1 sign. Over this sample the sizing rule is where the return came from; the trend signal cost Sharpe and bought a shallower drawdown (-20.3% vs -23.4%). ETFBeta is this finding turned into a sleeve.
- **The carry sleeve is negative net** and would have hit the 25% kill on 2010-08-31. It stayed in the v1 book because the v1 spec was frozen and 1/N never drops a sleeve on its own record. It is out of the live book now, not because of its record but because the book it sits in lost on the untouched window.

**Cash convention.** The shadow book pays nothing on cash, which is what Alpaca paper pays. That is harsh on the carry sleeve in particular: when USD is in a leg, the strategy holds cash where a T-bill would earn the rate it is ranking on, and the fx-carry project credited that rate explicitly. With cash swept at the 3-month USD rate (`Config(financing="USD")`, 50 bps spread on negative cash) the carry sleeve is +0.31, the v1 book 0.57 / +2.3% / -9.7% and the live book 0.85 / +6.3% / -10.8%. Those figures include the interest itself, so they compare to T-bills, not to zero; they are the upside if the live account earns a sweep rate, and they are not the headline.

## v3

### The buffer

Before v3 the 10% buffer lived inside `EWMAC.on_bar` on the raw targets: on any day one instrument left its band the whole sleeve was re-sent and the vol overlay rescaled every position, so the rule traded on 99.9% of instrument-days. The fix is in `engine/risk.py::RiskManager.apply`: the buffer applies to the final weights after the overlay, per instrument, against what the book holds, for every strategy; the drift path (a monthly book corrected between rebalances) and any breach or state change bypass it. `engine/check.py` has five checks for it, each mutation-tested: 3% daily wobble trades once and never again; a 15% step trades every instrument on the next bar and nothing else; a constant target under the vol overlay trades only when the overlay has moved it more than 10%; a gross breach sends every instrument, in band or not; a drawdown cut inside the band still trades. Turning the buffer off fails four of them plus two in `book/check.py`; removing the breach override fails two; removing the state-change override fails one.

What it did, same window (2005-06-09 to 2026-08-31), same cost model:

| | turnover / yr before | after | trades before | after | Sharpe net before | after |
|---|---|---|---|---|---|---|
| TrendETF | 6.8x | 6.5x | 3,900 | 2,506 | 0.38 | 0.43 |
| EWMAC, three speeds | 19.3x | 18.5x | 72,860 | 27,050 | 0.71 | 0.68 |

The expected result was that a buffer on final positions would cut EWMAC's turnover materially. It did not. The trade count fell 63% and the notional traded fell 4%: the trades the buffer removes are the small ones, and EWMAC's turnover is in the large daily moves of its own targets (a 16/64 forecast on a 35-day vol moves more than 10% of its level on most days for most instruments), which no band relative to the current target catches. With a proportional cost model those small trades were nearly free, so removing them saved nearly nothing, and the 10% tracking error the buffer introduces cost 0.03 of Sharpe. TrendETF gained 0.05 from skipping month-end rebalances smaller than 10% of a position. EWMAC's cost drag, measured on the same path, is 0.16 of Sharpe (0.84 gross to 0.68 net); a band sized off the average position (Carver's own formulation) or trading to the band edge rather than the target are the untested next candidates, and either would be a new trial.

### EWMAC on the untouched window

The untouched window is the last 20% of the panel's sessions, 2021-12-31 to 2026-08-31, the retired `WalkForward(holdout=0.2)` convention from the archived metrics module. Nothing was fit or picked on it; EWMAC's speeds and multiplier are pysystemtrade's defaults and its scalars are trailing. Its return was inside the one full-panel look reported in v2, so this confirms rather than discovers. The promotion rule was fixed before the run: net Sharpe above zero on the window, above TrendETF's on the same window, and a positive alpha t against the ETF universe there.

| rule, untouched window | Sharpe gross | Sharpe net | ann. return | ann. vol | max drawdown | turnover / yr | trades |
|---|---|---|---|---|---|---|---|
| EWMAC | 0.56 | 0.41 | +2.9% | 7.6% | -19.4% | 12.7x | 6,196 |
| TrendETF | 0.98 | 0.87 | +5.9% | 6.9% | -8.0% | 6.8x | 614 |

Against its universe on the window: alpha +2.8% a year, t 0.75, beta 0.04, benchmark Sharpe 0.98, residual Sharpe 0.37. **EWMAC does not clear the rule**: positive, but below TrendETF and with no significant alpha. The walk-forward over the four EWMAC variants on the sessions before the window (4 folds of 3 years) picks the combined rule in three folds and 32/128 in one, 0.72 selected against 0.66 frozen, so nothing there either. The full-panel 0.68 (t 3.1 on alpha) and the untouched-window 0.41 (t 0.75) are both real numbers; the second is the one that was never looked at while anything was being decided.

### Four books, one window

All four at 1/N of their sleeves, kill off, buffered engine, full cost model, from 2005-06-09 (the first day every book is live). In-sample columns are the whole window; OOS is the untouched window. DSR uses the final run's trial count (60; the run that made the decision had 47, which gave 0.02 / 0.27 / 0.30 / 0.66). Every book is a logged trial.

| book | Sharpe gross | Sharpe net | DSR | ann. return | ann. vol | max drawdown | turnover / yr | OOS Sharpe | OOS max drawdown |
|---|---|---|---|---|---|---|---|---|---|
| (a) v1, 3 sleeves | 0.33 | 0.22 | 0.01 | +0.9% | 4.3% | -12.7% | 3.8x | 0.40 | -9.2% |
| (b) v1 + EWMAC, 4 sleeves | 0.70 | 0.54 | 0.18 | +3.0% | 5.8% | -13.5% | 9.7x | 0.46 | -13.5% |
| (c) EWMAC for TrendETF | 0.72 | 0.55 | 0.20 | +3.1% | 5.8% | -14.4% | 10.8x | 0.40 | -14.4% |
| **(d) beta + alpha** | 0.88 | **0.76** | 0.54 | +5.4% | 7.3% | -12.1% | 10.5x | **0.79** | -11.7% |

**Rule, written before the run: the live book is the winner on the untouched window; if nothing beats (a) there, (a) stays.** (d) wins, 0.79 against 0.40, and beats (a) on drawdown too. ERC vs 1/N on (d)'s two sleeves, quarterly refits, out of sample: 0.86 vs 0.87, so 1/N. `allocate.LIVE_BOOK = "beta+alpha"`, weights ETFBeta 0.5 / EWMAC 0.5.

What the choice means, stated plainly. (b) and (c) show that EWMAC adds 0.3 of in-sample Sharpe to the v1 book but almost nothing out of sample (0.46 and 0.40 against 0.40). (d) wins because of ETFBeta: the ETF universe held 1/N and vol-targeted scores 0.98 on the untouched window, which was a good four years to own everything. The book is a beta position with a trend overlay whose out-of-sample contribution is small and whose own promotion test failed. The DSR of 0.53 is the highest in this directory and it deflates a Sharpe that is mostly the market's; (a), (b) and (c) are all under 0.2, which is where a rule with no beta and this much selection lands. The untouched window is 4.7 years, and (d) was written after reading the full-panel attribution table, which includes that window: nothing was fit on it, but the idea of what to test came from a look that covered it. Anyone reading this as evidence of skill is reading it wrong; it is evidence that this ETF universe went up over 2022-2026 and that a rule-based book of it did not get in the way.

With the live 25% kill on, ETFBeta would have been flattened on 2016-01-08 (the 2014-2016 commodity fall, inside a 1/N of 14 ETFs sized to 10% vol) and EWMAC never. Live, that parks half the book until it is restarted by hand; the backtests above run with the kill off.

### Candidate allocator: cost-aware mean-variance (2026-09-05)

`book/optimizer.py` is the third allocator, and it is a candidate, not live. It maximises mu'w - w'Sw / 2 - c'|w - w_prev| over long-only sleeve weights: mu is each sleeve's trailing three-year mean net return, S is the covariance (Ledoit-Wolf, or `../risk-model/risk_model.py::covariance` if that project exists at run time; it did not for this run), c is each sleeve's realised cost per unit of traded notional times its gross exposure, both from trades before the refit. Constraints: book vol at most 10%, gross and net at most 1.5, any one sleeve at most 0.75, and a turnover budget of 0.5 per refit; then a half-Kelly cap on the leverage of the chosen mix. Risk aversion 1 is the growth-optimal objective, so the Kelly, vol and gross caps are what size the book. A vol or Kelly cut that the budget would block is not budgeted, the same convention as the engine's buffer. Drawdown-scaled sizing is not repeated at the book level; it stays in the engine's overlay on every sleeve. Every parameter was written down before the run and none was tuned; the configuration is one trial per book.

The comparison is `allocate.walk`'s: quarterly refits on a three-year window, weights held for the next quarter. Its realised stream also pays the move on the first day after each refit at the costs known then and, when the sleeve weights sum past 1, financing on the excess at the 3-month bill from the data lake (`lake.load("fred", ..., universe=["DTB3"])`) plus 50 bps. Each book's candidate stream is a logged trial in `trials.csv` and a run in a research registry under `reports/registry/` (`research/registry/experiments.py`, keyed on the parameters). Rule, written before the run: the candidate goes live only if its best book's Sharpe on the untouched window beats the best 1/N book's there; otherwise 1/N stays and the candidate is a documented trial.

| book | walk Sharpe mvo | walk Sharpe 1/N | DSR | untouched Sharpe mvo | untouched Sharpe 1/N (engine) | mean leverage | turnover / refit | vol cap bound / refits |
|---|---|---|---|---|---|---|---|---|
| (a) v1 | 0.05 | 0.24 | 0.00 | **-0.44** | 0.40 | 0.67 | 0.24 | 1 / 86 |
| (b) v1 + EWMAC | 0.45 | 0.52 | 0.09 | **-0.27** | 0.46 | 1.08 | 0.24 | 42 / 86 |
| (c) EWMAC for TrendETF | 0.44 | 0.52 | 0.10 | **-0.26** | 0.40 | 0.91 | 0.17 | 1 / 82 |
| (d) beta + alpha | 0.73 | 0.87 | 0.48 | **0.66** | **0.79** | 1.13 | 0.10 | 36 / 90 |

**Not promoted.** The candidate's best book on the untouched window is (d) at 0.66 against 0.79 for the same book at 1/N, and on the three v1-based books it is negative where 1/N is 0.40 to 0.46. It also loses on the whole 22-year walk for every book (0.73 vs 0.87 on the live book, where ERC scored 0.86). `allocate.LIVE_ALLOCATOR = "rule"`, so the live weights are still ETFBeta 0.5 / EWMAC 0.5. The candidate's weight path for the live book is in `reports/mvo_weights.csv`: it sits at the 0.75 sleeve caps on both sleeves (1.5x, the gross cap) in a fifth of the 94 refits, has one sleeve at zero in 26 of them, and averages 1.08x. On the untouched window it cut ETFBeta from 0.75 to 0.08 by September 2022 and to zero at the end of that year, after the 2022 fall, and rebuilt it to 0.75 by March 2024, after the recovery; it cut EWMAC to zero from mid-2025 to the end of that year. The fit at the last date is ETFBeta 0.75 / EWMAC 0.59 (`reports/allocations.json`, under `candidate`).

What the result says. A three-year mean of a 10%-vol sleeve has a standard error of about 5.8% a year, about the size of the sleeve means themselves (ETFBeta +5.3%, EWMAC +6.6%), so the tilts are mostly noise, and the noise has a sign: the trailing mean falls after a drawdown and rises after a recovery, so the rule sells what just lost and buys what just won, a quarter late. The cost penalty and the turnover budget did what they were built to do (turnover 0.10 per refit on the live book, the budget binding 7 times in 90, the vol cap 36 times, the Kelly cap never) and none of it helps a rule whose expected-return input is noise. The books with the carry sleeve lose more because the candidate also times the crypto sleeve on two to three years of its own history. This is the same thing `portfolio-construction/` found across instruments, now found across sleeves: with two to four return streams and this much estimation error, 1/N is hard to beat and the gap is not small.

Two honesty notes. The untouched window was already read once, for the v3 book decision; this is its second read, on a rule fixed before the run, and the README's own limitation below applies: a promotion on it would have been weaker than v3's. And the live book's candidate walk was run once on the real panel before `validate.py`, as a smoke test of the code path, with the same parameters and the same result; it is the same trial, counted once.

## Signal vs beta

Each sleeve's net returns regressed on its own universe: the daily-rebalanced 1/N return of its instruments, and that return scaled to the sleeves' 10% vol target (60-day EWMA vol, one-day lag, 3x cap). Alpha is annualised with a Newey-West t; residual Sharpe is the Sharpe of the sleeve minus beta times the benchmark. If the residual Sharpe is below the benchmark's own Sharpe, the rule is not adding return beyond owning the universe. Whole panel, buffered engine:

| sleeve | benchmark | benchmark Sharpe | sleeve Sharpe | alpha / yr | t | beta | residual Sharpe |
|---|---|---|---|---|---|---|---|
| TrendETF | 1/N | 0.63 | 0.46 | +2.8% | 1.63 | 0.15 | 0.34 |
| TrendETF | 1/N vol-targeted | 0.61 | 0.46 | +2.4% | 1.45 | 0.23 | 0.30 |
| FXCarryETF | 1/N | 0.08 | -0.16 | -0.9% | -0.71 | -0.09 | -0.15 |
| FXCarryETF | 1/N vol-targeted | 0.02 | -0.16 | -1.0% | -0.73 | -0.08 | -0.16 |
| CryptoTrend | 1/N | 0.44 | 0.07 | -2.1% | -0.96 | 0.11 | -0.44 |
| CryptoTrend | 1/N vol-targeted | 0.52 | 0.07 | -2.7% | -1.24 | 0.63 | -0.59 |
| ETFBeta | 1/N | 0.67 | 0.66 | +0.7% | 0.76 | 0.67 | 0.16 |
| ETFBeta | 1/N vol-targeted | 0.65 | 0.66 | +0.4% | 0.57 | 0.77 | 0.13 |
| EWMAC | 1/N | 0.60 | 0.68 | +6.6% | 3.07 | 0.05 | 0.65 |
| EWMAC | 1/N vol-targeted | 0.58 | 0.68 | +6.2% | 2.89 | 0.13 | 0.61 |

TrendETF's alpha is positive but not significant and its residual Sharpe sits under the universe's own; the monthly sign has not added return beyond holding the 14 ETFs. FXCarryETF's universe has no return and the sleeve is below it. CryptoTrend held BTC and ETH worse than holding them. EWMAC is the only rule with alpha beyond its universe over the whole panel (t 3.1), and the section above is what happened to that alpha on the window nothing was decided on (t 0.75).

**Port fidelity.** `TrendETF` reproduces the trend project's `build()` targets at all 266 month ends to 0.00e+00 (`book/reports/trend_etf_vs_project.csv`). With the project's own settings (5 bps, no overlay) the daily engine scores 0.43 where the project's monthly engine scores 0.52 on the same panel: the monthly pipeline fills at the signal close and collects the overnight gap after month end (mean +16 bps on equities), the engine fills at the next open and does not, and the 2 bps half spread costs another 0.02.

## Stack

| Job | What | Notes |
|---|---|---|
| Simulator | `engine/` (own, 730 lines) | As-of `Bars` view that raises `LookAheadError`, next-open fills, commission + half spread + sqrt impact with a participation cap, per-strategy books, borrow and financing accrual, vol / gross / drawdown / kill overlay that also runs between rebalances, per-instrument position buffer on the final weights. `check.py` has 46 checks including a mutation test; the five buffer checks were mutation-tested (buffer off, breach override off, state-change override off: 6, 2 and 1 failures). |
| Sleeve allocation | [skfolio 1.0.3](https://skfolio.org) | `RiskBudgeting` with `EmpiricalPrior(covariance_estimator=LedoitWolf())`. |
| Candidate allocator | cvxpy 1.9.2 (already installed under skfolio), sklearn `LedoitWolf` | `book/optimizer.py`; the financing rate through `data-lake/lake.py`, the trial log through `research/registry/experiments.py`. |
| Walk-forward, DSR, PSR, MinBTL, MinTRL | [purgedcv 0.1.5](https://pypi.org/project/purgedcv/) (pinned) | Cross-checked against the retired `archive/framework-metrics-honesty.py` on the best-of-100-random case: DSR 0.6855 vs 0.6848, PSR identical. |
| Tearsheet | [quantstats 0.0.81](https://github.com/ranaroussi/quantstats) | `book/reports/tearsheet.html`, SPY benchmark passed as a Series. |
| Broker and crypto bars | [alpaca-py 0.44.0](https://github.com/alpacahq/alpaca-py), `book/broker.py` | Paper endpoint only; crypto bars need no key. The layer's kill switch, limits, reconciliation, simulated account, L2 fill model and dormant live adapters are described under Broker layer below. |
| ETF bars | yfinance via `engine.load_yfinance` | Cached under `source-material/framework/yf/`; `auto_adjust=True`, so a re-pull can rewrite history (the daemon logs a fixed-window panel hash to make that visible). |
| Rates | FRED (OECD `IR3TIB01*`) | FRED stopped updating the EUR and GBP series in 2026-01; `universe.RATE_FALLBACK` splices monthly averages of ECBDFR and SONIA on after each series' last print. The daemon re-downloads every series each run. |

**GPL note.** The EWMAC rule in `book/strategies.py` follows pysystemtrade (GPL-3.0): `systems/provided/rules/ewmac.py`, `sysquant/estimators/forecast_scalar.py` and `sysquant/estimators/vol.py`, linked above the code. It is written from the equations (fast minus slow EWMA over a 35-day EWMA price vol floored at its trailing 5th percentile, scalar to a median |forecast| of 10 after 500 days, cap 20, equal rule weights times a diversification multiplier, forecast / 10 times vol-target sizing); no code is copied. The position buffer in `engine/risk.py` is Carver's band as a trigger; it trades back to the target, not to the band edge. The falsification battery follows the shape of engineerinvestor's `falsification.py` (MIT) and is its own code.

## Run

All commands from `quant-projects/` with its venv (Python 3.14; do not install backtrader, bt, zipline, pysystemtrade or qlib into it, see PLAN risk #1).

```
.venv/bin/python3 framework/check.py                 # engine, 46 checks, offline, ~40 s
.venv/bin/python3 framework/book/check.py            # sleeves, EWMAC, books, attribution, broker layer, L2 fills, daemon, offline
.venv/bin/python3 -m framework.book.universe         # panel summary and hash
.venv/bin/python3 -m framework.book.allocate         # ERC vs 1/N on the live book, writes reports/allocations.json (with the candidate's fit)
.venv/bin/python3 -m framework.book.optimizer        # candidate allocator vs 1/N on the live book's sleeves, ~4 min
.venv/bin/python3 -m framework.book.validate         # everything in validation.md, ~25 min (seven daily EWMAC runs)
.venv/bin/python3 -m framework.book.daemon --dry-run # one session: targets and the Alpaca order list, nothing written
.venv/bin/python3 -m framework.book.daemon           # the same, submitted and logged to reports/ledger.csv
.venv/bin/python3 -m framework.book.dashboard        # regenerate reports/index.html from the ledger
```

A backtest in code:

```python
from framework.book import universe
from framework.book.strategies import sleeves, book_config
from framework.engine import run
bars = universe.load_bars()
res = run(sleeves("beta+alpha"), bars, config=book_config(kill=False, allocations={"ETFBeta": 0.5, "EWMAC": 0.5}))
res.metrics; res.by_year(); res.report("some/dir")
```

Paper account: put `ALPACA_PAPER_KEY` and `ALPACA_PAPER_SECRET` in `framework/.env` (paper keys only; the file is gitignored and currently holds empty placeholders). With keys `AlpacaBroker` is in **paper** mode: it constructs with `paper=True`, checks the client is in sandbox mode and that the base URL host is `paper-api.alpaca.markets`, and raises otherwise. Without keys it is in **simulated** mode, says so on every run, and the ledger rows carry `simulated:` in their note. **As of 2026-09-05 the keys are empty, so the mode that is actually running is simulated.**

Schedule: `book/com.kelvinhe.premia-book.plist` fires 15:45 Central Mon-Fri (16:45 ET). Same label and command as before the broker layer; the plist was edited in place (a comment block documents the layer and the KILL file) and reloaded on 2026-09-05.

```
launchctl bootout gui/$(id -u)/com.kelvinhe.premia-book        # if loaded
cp framework/book/com.kelvinhe.premia-book.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kelvinhe.premia-book.plist
launchctl list | grep premia
```

To stop all trading now: `touch framework/KILL`. The next run stops before it fetches a bar, and in paper mode cancels anything still queued. Delete the file to restart; the session that was skipped is picked up by the next run.

The daemon refuses to run unless the newest bar is the last completed NYSE session, skips a session whose shadow and alpaca rows are both in the ledger, cancels any orders still queued before submitting new ones, and rebuilds the shadow book by replaying the engine from `LIVE_START` (2026-08-31) on every run, so a missed day is caught up by the next run rather than by a state file. Its targets are what each sleeve holds with the engine's fresh sends laid over, so an instrument the buffer left alone is not re-traded at Alpaca.

## Broker layer

`book/broker.py`, added 2026-09-05. One set of targets from the shadow book, one `AlpacaBroker` that is either the paper account or a simulated one, and the safety in the layer rather than in any strategy. `book/check.py` covers all of it offline; `walk.py` (gitignored) is a press-Enter tour on synthetic bars.

**Modes.** `AlpacaBroker()` never raises for want of keys any more. With paper keys in `.env` it is `mode == "paper"` and talks to `paper-api.alpaca.markets` only (host and sandbox checked at construction). Without them it is `mode == "simulated"`: a `SimulatedAccount` rebuilt on every run from the ledger's own `simulated:` alpaca rows, the way the shadow book is rebuilt from the engine. An order recorded on session D's row fills at the open of the next session in the bars (Alpaca paper's own convention for a market order queued after the close), a limit fills at the better of the open and the limit if the day's range touched it and otherwise stays pending, and a crypto order fills at the second it was submitted through the L2 model below. No commission, no spread, no partials, no borrow, cash can go negative: it is a stand-in for Alpaca paper, which charges none of that either, and it is more optimistic than Alpaca paper because it has no random partial fills. There is no live mode in this class and no key it could read that would make one.

**Order routing.** `build_orders` turns weights into alpaca request objects: market by default, `LimitOrderRequest` for any name given a `limit_prices` entry, longs by notional, shorts in whole shares, sign flips closed first, Carver's 10% buffer and a $25 floor. The daemon sends market orders only.

**Kill switch.** `framework/KILL`. `killed()` is checked at the top of `run_once` before any data is fetched, and again inside every adapter's `submit`. Paper mode cancels queued orders on the way out. Every halt the layer raises (`Halted`) appends its reason and a timestamp to the same file, so a limit breach, a reconciliation failure and a hand-placed kill all leave the same trace and need the same restart: read it, delete it, rerun. The daemon then writes no alpaca row for that session, so a rerun trades it.

**Limits** (`Limits`, enforced in `submit`, not in the strategy): gross at most 3.0x equity, any one name at most 0.5x, and equity no more than 3% below the previous run's ledger equity. A breach halts before anything is sent, and in paper mode the cancel of anything still queued runs before the check, so a halt leaves no order from an earlier crashed run waiting for the open. `submit` refuses orders given without targets, so nothing leaves the layer unchecked. The engine's own overlay (10% vol target, 3x gross, drawdown cuts) is what keeps the book well inside these; the layer's numbers are the fence, not the target.

**Reconciliation.** Every run, before ordering, the account's weights are checked against what the previous ledger row of the same mode implies: a name that run sent an order for should sit at that run's target, everything else at what was held. Any name off by more than 5% of equity halts, with the list in the KILL file. Drift from one day's price move is well inside that; an Alpaca partial fill of a large position is not, on purpose. The first run has nothing to reconcile against and says so.

**Dormant live adapters.** `AlpacaLive`, `IBKRBroker` (ib_insync, TWS paper port 7497 by default, 7496 is live) and `KalshiLive` (wrapping `kalshi-desk/kalshi.py`'s client in `prod`) all call `live_gate` on their first line, which raises unless `LIVE_ENABLED=true` is in `framework/.env` (or the process environment; `read_env` lets the environment win) **and** the caller passed `confirm="LIVE YYYY-MM-DD"` for today **and** no KILL file exists. Only then is a live key read (`read_env` takes the key names it may return, and skips every other line before parsing its value). The daemon never constructs any of them; there is no flag that would make it. Checks: each of the three refuses in four configurations (no `LIVE_ENABLED`, no token, yesterday's token, `LIVE_ENABLED=false`) with the spy showing only `LIVE_ENABLED` was read, and refuses with both locks open while KILL exists. With both open, `AlpacaLive` reads the live key and constructs against `api.alpaca.markets`; `IBKRBroker` gets past the gate and stops at the import (ib_insync is not installed, on purpose); `KalshiLive` stops at kalshi-desk's own `LIVE_TRADING` lock, so that one has three locks. None of the three has been run against a real endpoint; `IBKRBroker`'s order mapping is written to the ib_insync docs and untested.

**Simulated crypto fills** (`book/l2fill.py`). Against the data lake's `crypto_l2` stream (top 20 levels, once a second, Bybit spot with OKX fallback, from 2026-09-05). A market order walks the far side of the first snapshot within 5 s of the order time; if the visible 20 levels run out the remainder is priced at the last level and flagged. A limit order that crosses the touch fills at once up to its price; otherwise it joins behind whatever rests at its price and is worked through the following snapshots for an hour: a drop in resting size at the price counts as executions ahead of it (never as cancels, so fills come early), and the far touch reaching the price fills the rest. Both are tested on planted books, and both ran against the real stream on 2026-09-05 (0.5 BTC market buy at 79,575.40 against a 79,575.30 bid; a 0.2 BTC limit at the touch filled in 45 s). `SimulatedAccount` routes BTC/USD and ETH/USD orders through it, falling back to the daily bar's open when the lake has no row for that second. The live book (`beta+alpha`) has no crypto sleeve, so this path is wired and idle until one is promoted.

## Daily report

`book/report.py` writes the desk summary the morning brief reads (`agents/morning.md`, Task 10 and the DESK section). It is read-only: it opens `ledger.csv`, the bar files the daemon already cached under `source-material/framework/`, `kalshi-desk/ledger.sqlite` on a `mode=ro` connection, `data-lake/reports/status.json`, `framework/KILL` and `launchctl list`. It never imports `read_env` or `AlpacaBroker` (a check enforces this), never downloads, never writes to a ledger.

```
.venv/bin/python3 -m framework.book.report     # about 2 s; prints the three headline lines
```

Output, every run: `book/reports/daily/<date>.md` (the full report), `book/reports/daily/<date>.json` and `book/reports/latest.json` (the same compact JSON, fixed path for the brief) and `book/reports/alerts.json`. `<date>` is the run date in Chicago; the book numbers inside are for the last session in the ledger.

What is in it:

- **Book.** Equity, day and since-`LIVE_START` P&L, drawdown from the equity path, gross against the overlay's 3x cap, the largest name against the broker's 50% per-name limit, the day's loss against its 3% daily-loss limit, held and target weights, the account row if there is one. **Per-sleeve P&L, gross and net, drawdown and cap usage come from replaying the shadow book** from `LIVE_START` on the cached bars of the ledger's last session, the same replay the daemon does; gross is net plus that day's commission and slippage for the sleeve (borrow stays in both). If any bar file for that session is missing the replay is skipped and the report says so; it does not download. Before the replay exists (or when it is skipped) equity comes from the ledger rows, and rows before 2026-09-04 are the v1 book, so a drawdown read across that boundary compares two books.
- **Risk.** One-day parametric VaR (95, 99) of the held weights under a Ledoit-Wolf covariance (`sklearn`) on the last 250 cached daily returns, in fraction of equity and dollars, with the model's annualised vol against the 10% target and the realised 20-day vol of the same weights. Gaussian, so a scale rather than a bound. `../risk-model/` does not exist yet; when it does, swap it in here.
- **Kalshi shadow desk.** Per book with any fills: fills, settled, open contracts and dollars at risk, P&L gross and net on dollars staked, mean payoff and t, Brier, through `kalshi-desk/paper.py`'s own `stats` and `exposure` so the numbers match `paper.py stats`. The backtest line it is being tested against sits beside it.
- **Data lake.** `status.json`'s verdict and age, stale and duplicated datasets, the last row per dataset.
- **Daemons.** `launchctl list` for the six `com.kelvinhe.*` jobs (pid, last exit status), the ledger's last run and last session against the last completed NYSE session, `daemon.log`'s mtime, the KILL file, and the broker layer's reconciliation rerun on the ledger (the last account row against what the row before it said would be held, `RECON_TOL` 5%).

Alerts (`alerts.json`, and the third headline line), in this order: `kill` (KILL file present, with the reason written in it), `drawdown` (book or sleeve past the 15% half-size line or 25% kill, a sleeve over 3x gross, a name over 50%, a day past the 3% loss limit), `daemon` (the last NYSE session not in the ledger by 19:00 ET, a job unloaded or with a nonzero exit status, no Kalshi mark for 6 hours), `feed` (lake not ok, a stale dataset, `status.json` older than 3 hours or missing), `reconcile` (an account row off the ledger by more than 5% on a name, the panel hash moving between the last two shadow rows).

`book/com.kelvinhe.desk-report.plist` fires the report at 07:30 local every day (optional: the brief's Task 10 runs it itself, so the job only matters if you want the files before the brief). Not loaded by default; `cp` to `~/Library/LaunchAgents/` and `launchctl bootstrap gui/$(id -u) ...` as for the book.

Checks live at the end of `book/check.py`: synthetic ledgers with a peak and drop, a missing session, a broken and an unloaded launchd job, stale and old and missing `status.json`, account rows inside and outside the reconciliation tolerance, a moved panel hash, a KILL file, a day past the loss limit, VaR on white noise (1.645 sigma, linear in the weight, zero with no positions, an uncached name dropped), a temporary Kalshi ledger with one settled bet, and the written files.

Limitations of the report itself:

- The drawdown is peak-to-now on the shadow equity path since `LIVE_START`, five sessions old at the time of writing; it says nothing about the backtest's -12% until the live path is long enough to have one.
- VaR uses close-to-close returns of the cached files and ignores the crypto sleeve unless a coin is held; there is no stress or historical VaR, and the covariance is on 250 days, which is one regime.
- Daemon health is inferred from files and `launchctl`, not from the process: a job that runs and writes nothing (the daemon's own "already in the ledger" path) looks the same as one that never fired until the next session is missed.
- The panel-hash alert fires on the first run after any change to `HASH_START` or the hash window, not only on rewritten history; the two ledger rows at the time of writing differ for that reason.
- The Kalshi shadow book's P&L is the optimistic fill (README there); the report prints it as recorded.

## Limitations

From the plan's risk register, still true:

- **#3, no futures.** Yahoo `=F` continuous contracts are unadjusted front-month splices; there is no free back-adjusted series. The book is ETFs only until an IB paper account and a proper roll exist. That means no short-vol, no ags beyond DBA, no rates beyond IEF/TLT.
- **#4, ETF proxies.** USO and DBC carry contango roll drag and USO restructured in 2020; TLT is not ZB duration; the currency ETFs charge 0.40% and CurrencyShares has delisted several (FXS, FXSG gone); nothing here is leveraged or inverse. The trend project's own ETF-vs-futures gap discussion applies unchanged.
- **#5, Alpaca paper flatters.** NBBO fills, no slippage or impact, no borrow on shorts, random 10% partial fills, no crypto shorts, no fractional shorts. The shadow book charges all of that; read Alpaca-minus-shadow in `reports/index.html` as the optimism gap, not as skill. A partial fill large enough to leave a name 5% of equity off its target halts the book (see Broker layer); smaller ones are re-reconciled on the next run. The simulated account flatters more still: every order fills in full at the open.

Also:

- Cash earns nothing in the shadow book (see the cash convention above).
- Rates reach the strategy with a lag: month M's average is dated the first of M+1, so a month-end rebalance ranks on the previous month's average.
- The kill switch is off in the backtests. Live it is on (25%), and a killed sleeve stays flat until restarted by hand.
- Ledger rows dated before 2026-09-04 are the v1 book; the 2026-09-03 row was also written before the rate fix below. They are left in place because the ledger is append-only. The first live-book run replays `beta+alpha` from `LIVE_START` and moves the paper account to it in one session.
- Yahoo's daily bar is not always final at 16:45 ET; a bar that later changes shows up as a panel-hash change in the ledger, not as a corrected fill.
- The untouched window has been read twice now: once for the v3 book decision, once on 2026-09-05 for the candidate allocator, on a rule fixed before that run. It is spent: any further change to the book that is judged on it is in-sample.
- The candidate allocator's comparison is on sleeve return streams recombined outside the engine, daily-rebalanced at no cost between refits; the engine's static 1/N drifts. The two agree to 0.07 of Sharpe on the untouched window for the live book (0.86 walk, 0.79 engine) and the rule compares against the engine's number. If the candidate ever went live it would run as a static allocation refit quarterly, like ERC would have, so the daily-rebalanced walk is an upper bound on it.
- If `../risk-model/` appears, `optimizer.covariance` will use it and the candidate's numbers here no longer describe the code; rerun `validate.py` and that is a new trial.
- The broker layer's limits are checked on the post-trade book once a day, when the daemon runs. Nothing watches the account intraday; a 3% daily loss is caught at 16:45 ET, not when it happens. That is what a once-a-session daemon can do.
- The daily loss limit compares to the previous ledger row's equity, so a weekend or a missed run stretches "daily" to whatever the gap was.
- Reconciliation reads weights, not shares, and its tolerance is per name; a small error on every name adds up to nothing it would catch.
- The simulated account has no market impact, no partials and no rejected orders, so it cannot surface any of the things the paper account exists to surface. It keeps the ledger, the reconciliation and the kill path exercised until keys exist, and nothing more.
- The queue model in `l2fill` is optimistic by construction (drops in resting size are trades, never cancels) and the stream has no sequence numbers, so a dropped delta corrupts the book until the next reconnect (`data-lake/README.md`). Its numbers are an upper bound on fill quality.

## Audit, 2026-09-04

Adversarial pass against PLAN.md before publishing. Fixed: `Bars` dropped any attached series observation dated on a non-session (36% of the monthly rate rows since 2005 landed on a weekend or holiday and were lost, so carry ranked on two-month-old rates in those months); `AlpacaBroker`'s host check parsed the string form of a `BaseURL` enum and always raised, so the paper book could never trade; a crypto short target skipped the coin instead of flattening it; a full close of a long sold by notional and could oversell after a gap; the daemon's panel hash covered a moving window and changed every day; the daemon never refreshed rates and two of the FRED series had gone dead; DSR counted 11 trials with duplicated return streams and left the allocator, cost, delay and kill variants out. Every check file gained a check for the fix, and the checks were mutation-tested (planted look-ahead in `on_bar`, dropped commission, disabled drift, same-bar fills: 12 failures across the four, sources restored byte-identical).

v3, later the same day: the buffer moved from `EWMAC.on_bar` to the engine (above); `ShadowBroker.targets()` would have dropped every instrument the buffer left alone and let Alpaca close it, fixed to lay fresh sends over held weights, with a check; the gross column changed from a zero-cost rerun to the same path with costs added back, because with the buffer a zero-cost run is a different path; trials are tagged with the engine settings so both engines' runs count.
