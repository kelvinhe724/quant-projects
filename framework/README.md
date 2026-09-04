# Premia book

A paper-traded, three-sleeve risk-premia book on one daily engine: ETF trend, FX carry through currency ETFs, and long/flat crypto trend, weighted 1/N, run once per NYSE session into two books (the engine's own shadow fills and an Alpaca paper account). Built to the plan in `outputs/2026-09/2026-09-04_trading-framework-research/PLAN.md`; every number below is recomputed by `book/validate.py` and lives in `book/reports/`.

The one-line honest summary: the book's 22-year net Sharpe is 0.26, the probability that it is above zero after deflating for the 22 configurations tried is 0.06, and the carry sleeve has lost money net of costs. This is a paper book because it is not yet evidence of anything. The one thing in this directory with a real signal is the v2 EWMAC candidate (net 0.71, alpha t 3.2 over its own universe), and it is not in the live book yet; see below for why.

## Why these sleeves

- **TrendETF** ports `trend-following/trend.py` (12-1 sign, 40% per-asset vol, class-balanced, month-end rebalance). That specification was frozen before results were seen and the 14-ETF universe was the project's own cross-check universe, so it is v1 as written. The project's own numbers are in `trend-following/README.md` (class-balanced Sharpe 0.41 gross / 0.37 net on its futures universe over 2005-2026; 0.52 net on the 14-ETF cross-check that this book trades, `book/reports/trend_etf_vs_project.csv`).
- **FXCarryETF** ports `fx-carry/carry.py::carry_weights` onto the six CurrencyShares ETFs Alpaca lists (FXE, FXY, FXB, FXA, FXC, FXF), ranked with USD on the same OECD 3-month rates from FRED, top-2 / bottom-2. USD in a leg is cash. The project (`fx-carry/README.md`) got Sharpe 0.35 on ten currencies with explicit rate differentials; see the cash convention below for why this port does not.
- **CryptoTrend** is the same 12-1 rule long/flat on BTC and ETH from Alpaca's crypto bars (2021 on). Alpaca cannot short crypto.
- **Allocator.** `portfolio-construction/` found nothing that beat 1/N across 14 ETFs at p < 0.10, so the plan's rule is that ERC across *sleeves* (skfolio `RiskBudgeting`, Ledoit-Wolf, three-year window, quarterly refit) has to beat 1/N out of sample before it goes live. It did not (0.23 vs 0.23), so the live allocator is equal weight.

Not in the book: `funding-carry/` (needs a perp venue) and `vol-risk-premium/` (a daily-hedged option book is a separate daemon). No futures (PLAN risk #3).

## Headline

Shadow cost model throughout: 5 bps commission, 2 bps half spread, sqrt impact (k = 0.1), 50 bps a year borrow on shorts, next-open fills, 10% vol target and 3x gross cap per sleeve, drawdown cut at 15%, kill switch off for backtests. Sharpe is on daily returns, days before a sleeve is live excluded. Gross = the same run with every cost coefficient at zero.

| | Sharpe gross | Sharpe net | ann. return net | ann. vol | max drawdown | live from | trades |
|---|---|---|---|---|---|---|---|
| TrendETF | 0.43 | 0.41 | +3.1% | 8.3% | -19.8% | 2004-07 | 3,972 |
| FXCarryETF | -0.08 | -0.17 | -1.2% | 6.2% | -39.3% | 2006-07 | 866 |
| CryptoTrend | 0.08 | 0.06 | +0.1% | 9.5% | -16.4% | 2022-02 | 75 |
| **Book, 1/3 each, static** | 0.32 | **0.26** | +1.0% | 4.3% | -13.5% | 2004-07 | |

Book honesty (`book/reports/validation.md`, run 2026-09-04): PSR 0.89, **DSR 0.06** with n_trials = 22 (every configuration `validate.py` runs on the real panel, the five EWMAC runs included, identical return streams counted once; 0.03 if the 20 placebo draws are counted too), MinBTL 55 years against 22 in hand, MinTRL 40 years. The move from 0.21 to 0.06 since the v1 run is the six new trials (five EWMAC, one gross TrendETF): the extra count alone takes it to 0.16, and the rest is the spread of Sharpes tried, which DSR deflates by and which EWMAC widens. Cost stress: 2x costs takes the book to 0.13, 4x to -0.09. A one-bar signal delay takes it to 0.13.

ERC vs 1/N, out of sample, quarterly refits from 2005-07: Sharpe 0.2275 vs 0.2327, drawdown -15.0% vs -14.8%. ERC's current weights would be Trend 0.32 / Carry 0.42 / Crypto 0.26; its risk contributions under the fitted covariance are 0.3333 / 0.3334 / 0.3333 and the weights match the project's SLSQP solver to 6e-5, so the library is doing what the cross-check says.

Two results the falsification battery turned up that matter more than the headline:

- **Vol scaling without the signal beats the signal.** The trend sleeve with every sign forced long scores 0.59 against 0.41 with the 12-1 sign. Over this sample the sizing rule is where the return came from; the trend signal cost Sharpe and bought a shallower drawdown (-19.8% vs -23.4%).
- **The carry sleeve is negative net** and would have hit the 25% kill on 2011-08-08. It stays in the book because the v1 spec was frozen and 1/N never drops a sleeve on its own record; removing it after seeing the result would be selection and would have to be logged as a trial.

**Cash convention.** The shadow book pays nothing on cash, which is what Alpaca paper pays. That is harsh on the carry sleeve in particular: when USD is in a leg, the strategy holds cash where a T-bill would earn the rate it is ranking on, and the fx-carry project credited that rate explicitly. With cash swept at the 3-month USD rate (`Config(financing="USD")`, 50 bps spread on negative cash) the carry sleeve is +0.30 and the book 0.56 / +2.3% / -9.5%. Those figures include the interest itself, so they compare to T-bills, not to zero; they are the upside if the live account earns a sweep rate, and they are not the headline.

## v2 candidate: EWMAC

`book/strategies.py::EWMAC` is a continuous-forecast trend rule on the same 14 ETFs: fast minus slow EWMA over a 35-day price vol, three speeds (16/64, 32/128, 64/256) combined with a diversification multiplier, forecast scalars fit on trailing data only, cap 20, forecast / 10 times TrendETF's 40% per-asset target with class balance, daily, with a 10% position buffer. Same cost model and overlay as the book. The window is 2005-06-09 to 2026-08-31, where both rules are live, so TrendETF's numbers here differ from the sleeve table above. Gross is the same run with every cost coefficient at zero; it was run for the combined rule only, so the single speeds have no gross column.

| | Sharpe gross | Sharpe net | ann. return net | ann. vol | max drawdown | turnover / yr | trades |
|---|---|---|---|---|---|---|---|
| TrendETF (same window) | 0.40 | 0.38 | +2.8% | 8.2% | -19.8% | 6.8x | 3,900 |
| **EWMAC, three speeds combined** | 0.84 | **0.71** | +7.0% | 10.3% | -20.1% | 19.3x | 72,860 |
| EWMAC 16/64 | | 0.54 | +4.9% | 9.8% | -17.7% | 26.7x | 72,867 |
| EWMAC 32/128 | | 0.70 | +7.2% | 10.6% | -16.7% | 18.8x | 72,867 |
| EWMAC 64/256 | | 0.57 | +5.4% | 10.2% | -19.5% | 13.9x | 72,810 |

Post-2010 the combined rule is 0.65 net, against 0.26 for the sleeve in the book. Costs take 0.13 of Sharpe (0.84 to 0.71), and that is the problem with it: turnover is 19x equity a year, three times TrendETF's, because the buffer is defeated. It sits on the raw per-instrument targets, and on any day one instrument leaves its band the whole sleeve is re-sent and the vol overlay re-scales every position, so the rule traded on 72,860 of 72,951 instrument-days in the window (99.9%) and on 5,338 of 5,340 days. A buffer applied to final positions after the overlay would cut most of that.

**EWMAC is a candidate, not in the live book.** The daemon runs the three v1 sleeves at 1/N. Two things would promote it: a confirmation on the untouched out-of-sample window (its 0.71 was seen once, on the whole panel, with the speeds and multiplier taken from pysystemtrade's defaults rather than fit here, but the number is still in-sample in the sense that nothing was held back), and the buffer fix, so that the net figure is not paying 19x turnover. Five EWMAC rows are in `trials.csv` (four net, one gross) and the DSR above already deflates for them.

## Signal vs beta

Each sleeve's net returns regressed on its own universe: the daily-rebalanced 1/N return of its instruments, and that return scaled to the sleeves' 10% vol target (60-day EWMA vol, one-day lag, 3x cap). Alpha is annualised with a Newey-West t; residual Sharpe is the Sharpe of the sleeve minus beta times the benchmark. If the residual Sharpe is below the benchmark's own Sharpe, the rule is not adding return beyond owning the universe.

| sleeve | benchmark | benchmark Sharpe | sleeve Sharpe | alpha / yr | t | beta | residual Sharpe |
|---|---|---|---|---|---|---|---|
| TrendETF | 1/N | 0.63 | 0.41 | +2.5% | 1.43 | 0.14 | 0.30 |
| TrendETF | 1/N vol-targeted | 0.61 | 0.41 | +2.1% | 1.25 | 0.21 | 0.26 |
| FXCarryETF | 1/N | 0.08 | -0.17 | -1.0% | -0.76 | -0.09 | -0.16 |
| FXCarryETF | 1/N vol-targeted | 0.02 | -0.17 | -1.0% | -0.78 | -0.08 | -0.17 |
| CryptoTrend | 1/N | 0.44 | 0.05 | -2.3% | -1.04 | 0.11 | -0.47 |
| CryptoTrend | 1/N vol-targeted | 0.52 | 0.05 | -2.9% | -1.32 | 0.63 | -0.62 |
| EWMAC | 1/N | 0.60 | 0.71 | +7.0% | 3.20 | 0.05 | 0.68 |
| EWMAC | 1/N vol-targeted | 0.57 | 0.71 | +6.5% | 3.02 | 0.13 | 0.64 |

What the table says, sleeve by sleeve:

- **TrendETF**: alpha is positive but not significant (t 1.4), and the residual Sharpe of 0.30 is under the universe's own 0.63. The monthly sign has not added return beyond holding the 14 ETFs; this is the same finding as the long-only falsification run above, seen from the other side.
- **FXCarryETF**: the universe itself has no return (1/N Sharpe 0.08), the sleeve is below it, and the beta is near zero. The carry ranking has neither picked up the currencies' drift nor found anything of its own.
- **CryptoTrend**: BTC and ETH held 1/N over 2022-2026 score 0.44; the long/flat rule scores 0.05 with a residual Sharpe of -0.47. The rule's time out of the market cost more than the drawdowns it sidestepped. On the vol-targeted benchmark the beta is 0.63, so most of what the sleeve does is hold scaled crypto, and it does that worse than the benchmark. (The 0.05 here counts the sleeve's flat days over 1,149 sessions; the sleeve table above drops zero-return days and reads 0.06.)
- **EWMAC**: alpha +7.0% a year, t 3.2, beta 0.05, residual Sharpe 0.68 against the universe's 0.60. This is the only sleeve where the daily signal adds return beyond owning the instruments.

**Port fidelity.** `TrendETF` reproduces the trend project's `build()` targets at all 266 month ends to 0.00e+00 (`book/reports/trend_etf_vs_project.csv`). With the project's own settings (5 bps, no overlay) the daily engine scores 0.43 where the project's monthly engine scores 0.52 on the same panel: the monthly pipeline fills at the signal close and collects the overnight gap after month end (mean +16 bps on equities), the engine fills at the next open and does not, and the 2 bps half spread costs another 0.02. Post-2010 that comparison run is 0.34, over the plan's "about 0.3" gate; the sleeve as it sits in the book (overlay, full cost model) is 0.26 post-2010, under it.

## Stack

| Job | What | Notes |
|---|---|---|
| Simulator | `engine/` (own, 760 lines) | As-of `Bars` view that raises `LookAheadError`, next-open fills, commission + half spread + sqrt impact with a participation cap, per-strategy books, borrow and financing accrual, vol / gross / drawdown / kill overlay that also runs between rebalances. `check.py` has 41 checks including a mutation test. |
| Sleeve allocation | [skfolio 1.0.3](https://skfolio.org) | `RiskBudgeting` with `EmpiricalPrior(covariance_estimator=LedoitWolf())`. |
| Walk-forward, DSR, PSR, MinBTL, MinTRL | [purgedcv 0.1.5](https://pypi.org/project/purgedcv/) (pinned) | Cross-checked against the retired `archive/framework-metrics-honesty.py` on the best-of-100-random case: DSR 0.6855 vs 0.6848, PSR identical. |
| Tearsheet | [quantstats 0.0.81](https://github.com/ranaroussi/quantstats) | `book/reports/tearsheet.html`, SPY benchmark passed as a Series. |
| Broker and crypto bars | [alpaca-py 0.44.0](https://github.com/alpacahq/alpaca-py) | Paper endpoint only; crypto bars need no key. |
| ETF bars | yfinance via `engine.load_yfinance` | Cached under `source-material/framework/yf/`; `auto_adjust=True`, so a re-pull can rewrite history (the daemon logs a fixed-window panel hash to make that visible). |
| Rates | FRED (OECD `IR3TIB01*`) | FRED stopped updating the EUR and GBP series in 2026-01; `universe.RATE_FALLBACK` splices monthly averages of ECBDFR and SONIA on after each series' last print. The daemon re-downloads every series each run. |

**GPL note.** The v2 EWMAC rule in `book/strategies.py` follows pysystemtrade (GPL-3.0): `systems/provided/rules/ewmac.py`, `sysquant/estimators/forecast_scalar.py` and `sysquant/estimators/vol.py`, linked above the code. It is written from the equations (fast minus slow EWMA over a 35-day EWMA price vol floored at its trailing 5th percentile, scalar to a median |forecast| of 10 after 500 days, cap 20, equal rule weights times a diversification multiplier, forecast / 10 times vol-target sizing, 10% buffer); no code is copied. The falsification battery follows the shape of engineerinvestor's `falsification.py` (MIT) and is its own code.

## Run

All commands from `quant-projects/` with its venv (Python 3.14; do not install backtrader, bt, zipline, pysystemtrade or qlib into it, see PLAN risk #1).

```
.venv/bin/python3 framework/check.py                 # engine, 41 checks, offline, ~30 s
.venv/bin/python3 framework/book/check.py            # sleeves, EWMAC, attribution, broker, daemon, 46 checks, offline
.venv/bin/python3 -m framework.book.universe         # panel summary and hash
.venv/bin/python3 -m framework.book.allocate         # ERC vs 1/N, writes reports/allocations.json
.venv/bin/python3 -m framework.book.validate         # everything in validation.md, ~20 min (five daily EWMAC runs)
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
res = run(sleeves(), bars, config=book_config(kill=False, allocations={"TrendETF": 1/3, "FXCarryETF": 1/3, "CryptoTrend": 1/3}))
res.metrics; res.by_year(); res.report("some/dir")
```

Paper account: put `ALPACA_PAPER_KEY` and `ALPACA_PAPER_SECRET` in `framework/.env` (paper keys only; the file is gitignored and currently holds empty placeholders). `AlpacaBroker` constructs with `paper=True`, checks the client is in sandbox mode and that the base URL host is `paper-api.alpaca.markets`, and raises otherwise. Without keys the daemon runs the shadow book and skips Alpaca.

Schedule: `book/com.kelvin.premia-book.plist` fires 15:45 Central Mon-Fri (16:45 ET).

```
cp framework/book/com.kelvin.premia-book.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.kelvin.premia-book.plist
launchctl list | grep premia
```

The daemon refuses to run unless the newest bar is the last completed NYSE session, skips a session already in the ledger, cancels any orders still queued before submitting new ones, and rebuilds the shadow book by replaying the engine from `LIVE_START` (2026-08-31) on every run, so a missed day is caught up by the next run rather than by a state file.

## Limitations

From the plan's risk register, still true:

- **#3, no futures.** Yahoo `=F` continuous contracts are unadjusted front-month splices; there is no free back-adjusted series. The book is ETFs only until an IB paper account and a proper roll exist. That means no short-vol, no ags beyond DBA, no rates beyond IEF/TLT.
- **#4, ETF proxies.** USO and DBC carry contango roll drag and USO restructured in 2020; TLT is not ZB duration; the currency ETFs charge 0.40% and CurrencyShares has delisted several (FXS, FXSG gone); nothing here is leveraged or inverse. The trend project's own ETF-vs-futures gap discussion applies unchanged.
- **#5, Alpaca paper flatters.** NBBO fills, no slippage or impact, no borrow on shorts, random 10% partial fills, no crypto shorts, no fractional shorts. The shadow book charges all of that; read Alpaca-minus-shadow in `reports/index.html` as the optimism gap, not as skill. Partial fills are re-reconciled on the next run.

Also:

- Cash earns nothing in the shadow book (see the cash convention above).
- Rates reach the strategy with a lag: month M's average is dated the first of M+1, so a month-end rebalance ranks on the previous month's average.
- The kill switch is off in the backtests. Live it is on (25%), and a killed sleeve stays flat until restarted by hand.
- The 2026-09-03 ledger row was written before this audit's rate fix (below); its carry weights differ from what the same code produces now. It is left in place because the ledger is append-only.
- Yahoo's daily bar is not always final at 16:45 ET; a bar that later changes shows up as a panel-hash change in the ledger, not as a corrected fill.

## Audit, 2026-09-04

Adversarial pass against PLAN.md before publishing. Fixed: `Bars` dropped any attached series observation dated on a non-session (36% of the monthly rate rows since 2005 landed on a weekend or holiday and were lost, so carry ranked on two-month-old rates in those months); `AlpacaBroker`'s host check parsed the string form of a `BaseURL` enum and always raised, so the paper book could never trade; a crypto short target skipped the coin instead of flattening it; a full close of a long sold by notional and could oversell after a gap; the daemon's panel hash covered a moving window and changed every day; the daemon never refreshed rates and two of the FRED series had gone dead; DSR counted 11 trials with duplicated return streams and left the allocator, cost, delay and kill variants out. Every check file gained a check for the fix, and the checks were mutation-tested (planted look-ahead in `on_bar`, dropped commission, disabled drift, same-bar fills: 12 failures across the four, sources restored byte-identical).
