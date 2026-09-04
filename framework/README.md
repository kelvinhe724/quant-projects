# Premia book

A paper-traded, three-sleeve risk-premia book on one daily engine: ETF trend, FX carry through currency ETFs, and long/flat crypto trend, weighted 1/N, run once per NYSE session into two books (the engine's own shadow fills and an Alpaca paper account). Built to the plan in `outputs/2026-09/2026-09-04_trading-framework-research/PLAN.md`; every number below is recomputed by `book/validate.py` and lives in `book/reports/`.

The one-line honest summary: the book's 22-year net Sharpe is 0.26, the probability that it is above zero after deflating for the 16 configurations tried is 0.21, and the carry sleeve has lost money net of costs. This is a paper book because it is not yet evidence of anything.

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

Book honesty (`book/reports/validation.md`, run 2026-09-04): PSR 0.89, **DSR 0.21** with n_trials = 16 (every configuration `validate.py` runs on the real panel, identical return streams counted once; 0.11 if the 20 placebo draws are counted too), MinBTL 47 years against 22 in hand, MinTRL 40 years. Cost stress: 2x costs takes the book to 0.13, 4x to -0.09. A one-bar signal delay takes it to 0.13.

ERC vs 1/N, out of sample, quarterly refits from 2005-07: Sharpe 0.2275 vs 0.2327, drawdown -15.0% vs -14.8%. ERC's current weights would be Trend 0.32 / Carry 0.42 / Crypto 0.26; its risk contributions under the fitted covariance are 0.3333 / 0.3334 / 0.3333 and the weights match the project's SLSQP solver to 6e-5, so the library is doing what the cross-check says.

Two results the falsification battery turned up that matter more than the headline:

- **Vol scaling without the signal beats the signal.** The trend sleeve with every sign forced long scores 0.59 against 0.41 with the 12-1 sign. Over this sample the sizing rule is where the return came from; the trend signal cost Sharpe and bought a shallower drawdown (-19.8% vs -23.4%).
- **The carry sleeve is negative net** and would have hit the 25% kill on 2011-08-08. It stays in the book because the v1 spec was frozen and 1/N never drops a sleeve on its own record; removing it after seeing the result would be selection and would have to be logged as a trial.

**Cash convention.** The shadow book pays nothing on cash, which is what Alpaca paper pays. That is harsh on the carry sleeve in particular: when USD is in a leg, the strategy holds cash where a T-bill would earn the rate it is ranking on, and the fx-carry project credited that rate explicitly. With cash swept at the 3-month USD rate (`Config(financing="USD")`, 50 bps spread on negative cash) the carry sleeve is +0.30 and the book 0.56 / +2.3% / -9.5%. Those figures include the interest itself, so they compare to T-bills, not to zero; they are the upside if the live account earns a sweep rate, and they are not the headline.

**Port fidelity.** `TrendETF` reproduces the trend project's `build()` targets at all 266 month ends to 0.00e+00 (`book/reports/trend_etf_vs_project.csv`). The daily engine scores 0.44 where the project's monthly engine scores 0.52 on the same panel: the monthly pipeline fills at the signal close and collects the overnight gap after month end (mean +16 bps on equities), the engine fills at the next open and does not, and the 2 bps half spread costs another 0.02. Post-2010 Sharpe 0.34 against the plan's "about 0.3" gate.

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

**GPL note.** The plan cites pysystemtrade (GPL-3.0) as the reference for a v2 continuous-forecast (EWMAC) trend rule. Nothing from it is in this repository; v1 has no EWMAC, and if v2 is built the rule is re-implemented from the description, not copied. The falsification battery follows the shape of engineerinvestor's `falsification.py` (MIT) and is its own code.

## Run

All commands from `quant-projects/` with its venv (Python 3.14; do not install backtrader, bt, zipline, pysystemtrade or qlib into it, see PLAN risk #1).

```
.venv/bin/python3 framework/check.py                 # engine, 41 checks, offline, ~30 s
.venv/bin/python3 framework/book/check.py            # sleeves, broker, daemon, 36 checks, offline
.venv/bin/python3 -m framework.book.universe         # panel summary and hash
.venv/bin/python3 -m framework.book.allocate         # ERC vs 1/N, writes reports/allocations.json
.venv/bin/python3 -m framework.book.validate         # everything in validation.md, ~5 min
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
