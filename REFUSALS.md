# Refusals

A backtest that flatters is easy to produce, and I produced several of them here. The skill I was actually trying to build is the one that tells me which of my own results are fake, so this file is the ledger of what this repository rejected: strategies that failed the test I wrote for them before I ran it, candidates the promotion rule turned down, audit findings that stopped a publish, and the biases I measured instead of citing. Every number below points at a file in this repository that reproduces it, and where a fix has a commit, the hash is given so you can read the diff rather than take my word.

---

## 1. Strategies that failed their own test

| project | what it claimed | what killed it | verify |
|---|---|---|---|
| `pairs-trading/` | in-sample net Sharpe **3.01**, out of sample **0.08** | the 2015-2019 formation window chose the 35 pairs out of 1,480, estimated every hedge ratio, and then graded the result; the parameter grid took the max of 54 configurations on that same Sharpe. **99 pairs clear p<0.05 against 74 expected from 1,480 coin flips; under Bonferroni (p<3.4e-05) zero survive.** The out-of-sample number goes negative above about 17 bps of round-trip cost. | `pairs-trading/README.md` §Multiple testing, `reports/performance.csv`, `reports/parameter_grid.csv`; `pairs-trading/DEFENSE.md` Q1 |
| `momentum/` | 12-1 long-short on 190 US large caps | net Sharpe **-0.03** in-sample and **0.12** out of sample; the chained walk-forward never switched variant because every candidate's training Sharpe was negative, returning -1.33%/yr. The one attractive version was contaminated: a $5 price floor applied to split-adjusted closes removed NVDA for 3,495 stock-days, KLAC for 2,612, AAPL and AMZN for over 1,000 each, in exactly the years they were the biggest winners. The floor was deleted, not kept. | `momentum/README.md` §Design and §Results, commit `68117bd` |
| `betting-markets/` | one betting rule made **+1.27%** | it made it by assuming every selection filled at the best price across up to eleven books. The identical selections filled at the consensus price return **-6.06% at t = -5.33**. There is no information in the selection, only in the price shopping. The favourite-longshot gradient likewise falls from t = -3.43 at quoted odds to t = -1.36 de-vigged. | `betting-markets/README.md` §Betting rules table (rows "best price beats de-vigged Pinnacle" and "same selections, filled at consensus price") |
| `prediction-markets/` + `kalshi-desk/` | fade the fitted longshot curve, **+2.4c per dollar at the mid** | the median quoted Kalshi spread is **3 cents** and 59.4% of markets are wider than the 2 cents the spec suggested assuming. Charged the spread actually quoted, the strategy is **-1.2c to -1.4c** (t -1.28 / -1.45). Restricted to books 5c wide or tighter the bias itself is 0.8c at t 1.9. The desk was built to forward-test that loss, not to trade it. | `prediction-markets/README.md` §Out-of-sample fade table; `kalshi-desk/README.md` lines 10-20 |
| `kalshi-model/` | supervised calibration should beat the bin curve | the mid wins. Untouched window of 2,318 markets: log-loss **0.5097 market vs 0.5136 best model vs 0.5139 bin curve**. Walk-forward picked the published bin curve over every estimator. After spread and fee the bin curve is -6.5c/$ (t -2.95) and the model -5.6c (t -2.33). Models beat the mid only in books wider than 10c, where the rule never trades. | `kalshi-model/README.md`, DSR 0.001 over 20 trials |
| `funding-carry/` | BTCUSDT carry **11.86%/yr** | Binance's funding formula is `premium + clamp(0.01% - premium, +/-0.05%)`, so the rate snaps to a hard-coded 0.01% per eight hours on **34.1%** of all settlements. **92% of the average carry is that constant**; 0.91%/yr is what the market was genuinely paying. Excess over T-bills by year after 2021: +0.7%, +0.3%, +3.6%, -0.7%, -3.1%. | `funding-carry/README.md` first two findings, §Distribution table row "exactly at the 0.01% floor" |
| `equity-xs/` | LightGBM walk-forward OOS Sharpe **0.29** net, rank IC 0.029 (t 2.3) | **-0.56 net on the untouched 2022-2026 window** (IC -0.008, t -0.4), while plain 12-1 momentum does the reverse. A post-hoc ablation puts the walk-forward number on three macro series that are constant within a date, so the trees were separating dates, not stocks: without them the walk-forward falls from 0.27 to -0.09. Worse, the point-in-time fix made it worse, not better: the same frozen spec went from **-0.15 to -0.56** once 4.5 years of back-cast membership rows were removed from training. | `equity-xs/README.md` §The window and §Limitations; `reports/untouched.json`, `untouched-first-open.json`, `untouched-second-open.json` |
| `crypto-basis/` | gross Sharpe **4.5** pre-window | **net -25.3** on the untouched window, -87.5%/yr at taker fees and -40.8% at the top VIP tier. A 3-sigma dislocation is about 3 bps wide and gives back 1.2 bps a round trip gross against 34 bps of taker cost; **only 6 of 271,585 BTC bars ever exceed a taker round trip**. DSR 0.00 over 8 trials. | `crypto-basis/README.md`, `reports/untouched.json` |
| `vol-risk-premium/` | 9.55%/yr net, **Sharpe 1.74** | the shape. **Skew -5.16, excess kurtosis 75.4**, worst day -6.49% which is 18.7 standard deviations, and a second day above 17 sigma. Deleting the worst 1% of days lifts the Sharpe to 3.75; **deleting the ten worst days of 4,180 removes 23% of the total P&L**. Separately, the VIX is not what a straddle sells at: a four-point basis takes the whole thing negative. | `vol-risk-premium/README.md` §Limitations; `vol-risk-premium/DEFENSE.md` Q3 |

Also failed, in one line each, with the same structure: `nlp-events/` (FinBERT reads the release at IC 0.094 t 4.6 and nothing survives to the first tradable open; untouched-year net Sharpe -1.84 / -0.77 / -1.09, DSR 0.004-0.028 over 13 trials); `deep-lob/` (the network is the only model whose argmax leaves the flat class, gross edge +0.13 bps a trade, breakeven fee 0.09-0.13 bps a side, so at 1 bp every variant loses, -17,180 bps over two days); `kalman-pairs/` (adaptive hedge ratios score net OOS -0.16 against 0.08 for the frozen OLS beta; the best of the state-noise grid finds gross 0.51 and spends all of it on turnover); `crypto-arbitrage/` (median best executable gap 1.66 bps BTC, **0% of 1,099 samples clear any published taker fee**); `overnight-anomaly/` (SPY +7.1%/yr overnight vs +1.1% intraday, but breakeven one-way cost is 1.49 bps, so the trade is worth nothing after costs on every asset but one); `portfolio-construction/` (nothing beats 1/N at p<0.10; unconstrained tangency runs 20-23x gross leverage); `dispersion/` (full-sample net Sharpe swings from +0.91 to -1.19 across plausible values of two unobservable proxy constants, so no single Sharpe from the project is reportable); `regime-hmm/` (the HMM lifts net Sharpe 0.86 to 0.90 out of sample while plain 15% vol targeting gets 0.97; the in-sample smoothed version shows 1.88, which is the trap).

---

## 2. Candidates the promotion rule rejected

The rule, written into `framework/README.md` before any of these were run: a candidate sleeve goes into the book only if its net Sharpe on the untouched window is above zero, above the incumbent's on the same window, and carries a positive alpha t against the ETF universe there. The allocator rule was written the same way, before its run. In all three cases below the live book was left exactly as it was.

| candidate | rule outcome | numbers | verify |
|---|---|---|---|
| **EWMAC** (16/64, 32/128, 64/256 and the combined rule) | **failed** | On the full panel it looked like the best sleeve in the repository: net 0.68, alpha t 3.1. On the untouched 2021-12-31 to 2026-08-31 window it scores **net 0.41 against TrendETF's 0.87**, with alpha +2.79% at **t 0.75** against its own 1/N universe. It is in the live book only as half of the beta+alpha pair chosen by the separate book comparison, and the README says so. | `framework/book/reports/validation.md` §EWMAC: walk-forward and the untouched window ("Clears the rule: no."), commit `886e0d0` |
| **Cost-aware mean-variance allocator** (`optimizer.py`) | **failed** | Rule fixed before the run: it goes live only if its best book beats the best 1/N book on the untouched window net of reallocation cost and financing. Best candidate book **0.660 against 0.789** for the same book at 1/N, and negative on all three v1-based books where 1/N is 0.40 to 0.46. It also loses on the full 22-year walk (0.726 vs 0.870). `allocate.LIVE_ALLOCATOR` stayed `"rule"`. | `framework/book/reports/validation.md` §Candidate allocator ("Promoted: no."), commit `4b4e095` |
| **`rates-carry/` as a book sleeve** | **failed** | Through the book's own engine and cost model the sleeve is net Sharpe 0.53 with beta 0.75 to IEF+TLT and **alpha +1.4%/yr at t 1.66**, positive but not significant, and its residual Sharpe of 0.33 is below the benchmark's own 0.39, so the sleeve does not add anything beyond simply holding IEF+TLT. It stays a standalone project and is not even counted as a book trial. | `rates-carry/README.md` §Through the engine, `framework/README.md` line 21, commit `890c72f` |
| **`regimes/` HMM overlay on the live book** | **not adopted** | Untouched window: bare book **0.79 (DSR 0.87 over 65 trials)** against **0.53 (DSR 0.71)** with the chosen K=3 overlay, for a shallower drawdown but 22% more trades. Walk-forward picked the overlay in 3 of 4 folds and its stitched OOS Sharpe was still 0.50 against 0.59 for leaving the book alone. | `regimes/README.md` |
| **`vol-models/` GJR timing on the short-straddle sleeve** | **rejected** | GJR wins the forecast horse race (QLIKE 0.223, DM t -9.3) and then makes the trade worse: untouched-window Sharpe **0.37 on capital against 1.41 always-on**, 15% time in market, DSR 0.52 against 0.83. The harness's active-day Sharpe had picked a rule that never traded in three of four walk-forward test windows. | `vol-models/README.md` |

---

## 3. Audit findings that blocked a publish

| finding | what was wrong | fix commit |
|---|---|---|
| **The GARCH VaR backtest graded itself** | `var_backtest` scaled conditional volatility by the empirical quantile of the model's own standardised residuals, which pins the hit rate to the target level by construction and tests nothing. Fixed to use the fitted student-t quantile with the mean included. **The conclusion reversed**: the published numbers went from 4.16-4.67% observed against 5% expected ("the model over-reserves slightly") to **6.12-6.84%**, meaning the model under-reserves and the fitted tails are too thin. | `d576ffb` |
| **Four `check.py` files always exited 0** | `capm/check.py`, `garch/check.py`, `gradient-boosting/check.py` and `nelson-siegel/check.py` printed a pass/fail line and then returned success either way, so any CI or scripted run would report green on a failing check. Each gained `raise SystemExit(0 if checks and all(checks) else 1)`. Two weak checks were replaced in the same pass (`fit_all` sorted-by-beta on two tickers, which cannot fail; a pairs position-lag check that only tested the first element). | `d576ffb` |
| **Cross-validation leakage in the credit model** | `cv_auc` used the scored fold itself as LightGBM's early-stopping set, so the reported CV number was tuned on the data it was grading. Fixed by carving a 20% early-stopping split out of each training fold. Published CV AUC moved from **0.7864 +/- 0.0080 to 0.7851 +/- 0.0084**. | `d576ffb` |
| **`crypto-basis/` silently dropped 35% of funding settlements** | Binance stamps 991 of 2,829 settlements per symbol at `HH:00:00.001`; `funding.reindex(bar_index)` turned those into NaN then 0, so the short leg collected 65% of its funding while the README claimed the timestamp quirks were "handled and checked". `check.py` could not catch it because it planted settlements on exact bar stamps. Fixed in `basis.py::funding_accrual` with a discriminating regression check that fails on the old code. The audit's verdict was **NOT SAFE TO PUBLISH** and the project was published only after the fix, with the pre-fix lock kept as `reports/untouched_first_read.json`. | `265f245`, first-read reports archived in `e60e578`; audit at `outputs/2026-09/w1-audits/crypto-basis.md` |
| **`equity-xs/` three defects, verdict NOT SAFE TO PUBLISH** | (1) the untouched lock was keyed to a live lake that a nightly cron rewrote, so `run.py` raised `LockError` and the "reruns reproduce every number" claim was false; (2) the first 4.5 years of training rows carried back-cast S&P membership while the README called the panel point-in-time; (3) the walk-forward purge was 19-20 sessions, not 21, because `evaluation_times = cal + BDay(21)` counts business days rather than trading sessions. All three fixed (pinned snapshot, membership NaN before 2007-11-29, purge counted on the calendar), which forced a refit and a third window read that moved the result **against** the model. | `265f245`; purge fix `e60e578`; audit at `outputs/2026-09/w1-audits/equity-xs.md` |
| **`options-collector/` shipped two checks that could not fail** | Horizon interpolation tested on a flat surface, and a stale flag tested on an empty summary. Both replaced with checks that can fail. A README VIX reference and an `svi/` quote count were corrected in the same pass, and a warning added when quotes are filed under a folder that is not today's date. | `321b4fb` |
| **`rates-carry/` calendar shift and a 365x weight** | Calendar-year returns were shifted by a month, so 2022 on the 10y read -13.4% when it is **-17.8%**; and a one-day trailing month was entering the predictive regression at 365x weight. | `e784b88` |
| **`framework/` adversarial pass before publishing** | `Bars` dropped any attached series observation dated on a non-session, so **36% of monthly rate rows since 2005 were lost and carry ranked on two-month-old rates**; `AlpacaBroker`'s host check always raised, so the paper book could never have traded; a crypto short target skipped the coin instead of flattening it; a full close could oversell after a gap; the daemon's panel hash covered a moving window and changed every day; two FRED series had gone dead unnoticed; **the deflated Sharpe was counting 11 trials with duplicated return streams and omitting the allocator, cost, delay and kill variants**, which is why the published count is now 60. Every fix got a check, and the checks were mutation-tested (12 planted failures across four files). | `framework/README.md` §Audit 2026-09-04, `350fc87` and `886e0d0` |
| **`pit-universe/` under-counted recoveries** | Seven renames that Yahoo backfills under the successor symbol (KFT, DISCA, WLTW, PX, FLT, ARNC, BHI) were being scored as missing, plus two new checks on the membership lag and the gap filler. | `e92ae86` |
| **`prediction-markets/` reports predated their own purge** | `results.txt` was generated before the 24-hour test-window purge and was regenerated rather than left to disagree with the README. | `ff29c83` |
| **`momentum/` universe look-ahead** | The split-adjusted $5 price floor described in section 1, plus survivorship membership lagged by a day. Found in the audit and removed before the project was published. | `68117bd` |

---

## 4. Windows and trials, disclosed

**Untouched-window reads.** Every one of these is a re-read of a window that was supposed to be opened once, and each is recorded in the project it belongs to rather than quietly absorbed.

- **`equity-xs/`, read three times.** First read; second read; then a third on 2026-09-05 after the audit above changed the training rows, which forced the frozen spec to be refit. The README's own words: "the training-row rule was changed after the window's first result was known, which is the thing the lock exists to make impossible, and a reader should treat 2022-2026 as a window this project has used rather than one it holds in reserve." All three locks are committed (`untouched.json`, `untouched-first-open.json`, `untouched-second-open.json`).
- **`crypto-basis/`, read twice.** The first read (hash `b887eacece990a5e`) was on the instrument that dropped a third of its funding. The fix changed the data hash, so the lock had to be recreated and the same 54,433 bars read again. Nothing about the rule, the grid or the chosen variant changed, and the fix moves the taker Sharpe by less than 0.01, but it is still a second look, and the README says any further change judged against those bars is in-sample.
- **`framework/`, the book's window read twice.** Once for the v3 book decision, once on 2026-09-05 for the cost-aware allocator, on a rule fixed before that run. "It is spent: any further change to the book that is judged on it is in-sample."
- **`vol-models/`, two lock files.** The first open stored a result whose integer dict keys JSON turned into strings, so the lock's hash of its own contents failed on reread. The first file is kept as written. The window was read twice by the code and once by me, and the second read reproduced the first to four decimals. Recorded because the lock exists to record exactly this.
- **`regimes/`** reads the book's window again, for a different decision, through its own lock, and states that the window is therefore not untouched in the strict sense for anything touching the book.
- **`research/` and `risk-model/`** each read their window once but read it for two things in the same call (two cost models; the estimator race and the VaR coverage), which is disclosed rather than counted as one look.

**Trial counts behind each deflated Sharpe.** These are the denominators. A DSR is only as honest as the count fed to it, and these counts are read off logged registries, not estimated.

| project | trials | DSR |
|---|---|---|
| `framework/` live book | **60** (every configuration `validate.py` ran on the real panel; `trials.csv` holds 188 rows, identical return streams counted once; 0.571 if the 20 placebo draws are counted too) | 0.632 |
| `framework/` v1 book | 47 (the count when it was live) | 0.02 |
| `regimes/` | 65 (11 of its own plus the book's 56, since the overlay sits on a book that was itself selected) | 0.87 bare / 0.71 overlay |
| `kalshi-model/` | 20 | 0.001 |
| `vol-models/` | 16 | 0.52 timed / 0.83 untimed |
| `deep-lob/` | 14 | 0.000 |
| `nlp-events/` | 13 | 0.004 / 0.028 / 0.014 |
| `risk-model/` | 12 | 0.93 |
| `equity-xs/` | 8 distinct registry keys, and the README states the true number of looks is higher because the earlier reads were the same keys on different training rows | 0.00 |
| `crypto-basis/` | 8 | 0.00 |
| `research/` | 5 | 0.59 |

**The plain statement.** A deflated Sharpe of 0.63 at 60 trials is not the same claim as a raw Sharpe of 0.81. The raw number says what one configuration did on one panel; the deflated number is the probability that the *selected* configuration's true Sharpe is above zero, given that 60 were tried and the variance of their daily Sharpes was 3.94e-04. The book's own README makes the harder version of the point: MinBTL is 8.4 years against 23.2 in hand, and for a book whose core is 0.67 beta to its universe, those are statements about the universe rather than about skill. Note one internal inconsistency I have not resolved: the root `README.md` still prints "DSR 0.63 over 56 trials" for the live book while `framework/README.md` and `validation.md` print 60. The 60 is current; the root table is stale.

---

## 5. Measured biases

Survivorship was the largest unquantified caveat in every equity project here. `pit-universe/` exists to replace the caveat with a number, by rebuilding S&P 500 membership from Wikipedia change rows and 39 dated list snapshots (893 names, 2003-2026) and rerunning the same code on both universes with the same dates and costs.

| bias | measured value | direction |
|---|---|---|
| Equal-weight level bias from using today's members | **+3.97%/yr in-sample, +4.23%/yr out of sample** (root README rounds this to +4.0%/yr), splitting roughly 80/20 between holding future winners before they joined and dropping deleted names | flatters |
| 12-1 momentum gross annual return | overstated **+2.2%/yr in-sample, +6.3%/yr on the 2021+ test, +3.2%/yr over the full 21 years**, all of it from the long leg | flatters |
| Pairs screen | **-0.66%/yr net and -0.07 in Sharpe**, i.e. the survivor universe made it look slightly *worse*, and that is inside the noise on a 6.7-year Sharpe (standard error about 0.4) | negligible |

**Coverage holes, stated rather than hidden.** 257 of the 390 names ever removed from the index return nothing from Yahoo, and the ones that do are mostly demotions to the S&P 400 rather than bankruptcies or takeouts, so every bias number above is a **lower bound**. The point-in-time universe is **60% of the true index in 2005 and 90% in 2020**. 2003 to November 2007 is a back-cast from late-2007 membership with about 100 unknown changes missing, and 154 of the 169 recovered changes sit in 2008-2010 with up to six months of snapshot-timing error, so the crisis years carry an error the later years do not. Delisting returns are zero throughout. In `equity-xs/` the same hole shows as a median of **433 eligible names per month end against roughly 500 in the index**, and the missing names are disproportionately the losers a short leg wants.

**Projects built on simulated or proxied data, labelled as such in their own READMEs.**

- `market-making/` is a simulator, and says so before its results: "The Sharpes are not real numbers. 3.65 to 7.15 per session annualises to 58 to 113, which no market maker has ever run." Only the comparisons between strategies carry information, because the missing frictions are the same for all three.
- `monte-carlo/`'s headline match to Black-Scholes is admitted to be circular in its own DEFENSE: the volatility was backed out with Black-Scholes and fed to an engine that converges to Black-Scholes, so the test proves the plumbing and nothing else. The non-circular comparison is the 1.33 gap between realised-vol and implied-vol pricing.
- `dispersion/`'s historical implied side is a proxy with two constants measured on a single day. The full-sample net Sharpe moves from **+0.91 to -1.19** across plausible values of them, so the calibration table is the output and no single Sharpe from the project is reportable.
- `deep-lob/` had 0.8 hours of real L2, so it runs on seven days of aggTrades with the touch reconstructed from prints and no book at all. Stated in the root table, not buried.
- `orderbook-imbalance/`'s t = 12.7 is on a **50-minute sample**, and `crypto-arbitrage/`'s conclusion rests on 55 minutes of polling.
- Every project's `check.py` fits its estimators to simulated data with planted parameters. That is a test of the code, and it is never reported as a result.

---

## 6. What is still wrong

Pulled from the Limitations sections of the largest projects. These are unfixed, and each is written where the result that depends on it is published.

**`framework/` (the live book).** (1) The book is mostly beta: on the untouched window the ETF universe held 1/N scores 0.98 by itself, so the 0.79 is not evidence of skill, and the README says "anyone reading this as evidence of skill is reading it wrong." (2) No futures, so no short vol, no ags beyond DBA, no rates beyond IEF/TLT; the ETF proxies carry contango roll drag and USO restructured in 2020. (3) Nothing watches the account intraday: a 3% daily loss is caught at 16:45 ET by a once-a-session daemon, and the simulated account has no impact, no partials and no rejected orders, so it cannot surface the things a paper account exists to surface.

**`equity-xs/`.** (1) Three macro series constant within a date let the trees fit regimes rather than stocks, and a fair version needs hand-built interactions or one training row per date. (2) The 2008-2010 membership corrections are in every training set. (3) The results reproduce from a 94 MB local snapshot, not from the repository alone, so a clone raises `LockError` rather than silently re-locking.

**`risk-model/`.** (1) The pre-declared rule picked `pca` on the S&P panel and `pca` is second-worst on the 14 ETFs the book actually holds; the headline VaR is reported on the estimator the rule named anyway. (2) Every stress number holds today's weights fixed, so it is not the risk of the book as it would actually trade. (3) The factor count is biased low, and parametric-normal VaR already fails Kupiec at a 2.1% hit rate pre-window.

**`research/` (the platform everything else runs on).** (1) The peek audit perturbs one session, so a feature peeking only at unperturbed rows would pass. (2) The registry keys on config, not code: changing an `Alpha`'s `signal` under the same config replaces the row instead of counting a new trial. (3) Research and engine numbers are not comparable head to head, 10 bps flat against the book's full cost stack.

**`data-lake/`.** Yahoo's adjusted close is a snapshot from the fetch date and incremental runs only refresh seven days, so after a dividend the older rows keep the old factor. Every project on the lake inherits this.

**`vol-risk-premium/`.** Sixteen years with one true crash and no 1987 or 2008; the sample's worst day is a hard lower bound, not an estimate, and there is no margin model, so the -13.6% drawdown is optimistic in a way with no bound.

**`crypto-basis/`.** The reports in `reports/` and the code disagree in one place the audit chose not to close: the walk-forward fold Sharpes are on 5-minute bar returns times sqrt(252), a different unit from every other Sharpe in the repository (the four test folds are -27.6, -25.6, -24.7, -25.9 on daily returns). Selection is internally consistent so the pick is unaffected, and the README carries the caveat.

**`pit-universe/`.** The 54 hand renames are one person's reading of corporate history and any of them could be wrong; two ticker-inheritance cases (WM, AGN) are known-broken and were not special-cased; sector labels for the pairs screen are frozen at one date across two GICS reshuffles.

**`diamonds/`.** Selection and inference are separated but not perfectly: the candidate 18 terms were fixed before any fitting and inference runs on a split selection never touched, but the transformations come from EDA on a well-known dataset, so the p-values are cleaner than post-selection OLS and not as clean as a pre-registration.

---

## How to check any of this

Every project has an offline `check.py` that fits the same estimators to simulated data with planted parameters, and a `run.py` that regenerates `reports/`. Locks live in `reports/untouched.json`, trial registries in `reports/registry/runs.jsonl`, and the book's full validation run in `framework/book/reports/validation.md`. The two audit reports that produced a NOT SAFE TO PUBLISH verdict are at `outputs/2026-09/w1-audits/`. Commit hashes above are from this repository's `git log`.
