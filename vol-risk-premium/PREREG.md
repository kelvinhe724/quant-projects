# Pre-registration — VRP sleeve (short delta-hedged ATM straddle, XSP)

Status: **SIGNED 2026-10-06** by Kelvin He, in session, before any backtest was run against this document. The commit that adds this file is the lock; any later change to it is a second look and goes in `REFUSALS.md §4`.

Grill record: `outputs/2026-09/2026-09-11_trading-desk-project-handover.md` → session 2026-09-11 / 09-17 / 10-06, decisions Q1–Q9.

## 0. Identity
- **Name:** `VRP` · project dir `quant-projects/vol-risk-premium/` (existing research, 2010–2026, README Sharpe 1.74 *at the VIX*, which the README itself says is not the price)
- **Author / reviewer:** C2 builds · Kelvin reviews and signs
- **Date signed:** 2026-10-06 · **Commit hash:** the commit that adds this file (`git log --follow PREREG.md`)

## 1. Hypothesis
Index implied volatility exceeds subsequently realised volatility by enough that a short ATM straddle on a 1/10-size cash-settled index option, delta-hedged daily, earns more than its spread, hedge slippage and the ATM-vs-VIX basis, at a size an account of $35k can hold through a VIX-80 day without a forced liquidation.

The part that is genuinely in doubt is not the premium (NW t 8.35 over 16 years) but whether any of it is left after the basis: the project's own table puts the Sharpe at 0.73 with a 2-point basis and 0.23 at 3; the collector measures 2.4.

## 2. Universe and instrument
- **Traded:** XSP (Cboe Mini-SPX, 1/10 SPX, cash-settled, European, Section 1256). ATM straddle, nearest monthly expiry 25–40 calendar days out at entry. Hedge leg: SPY shares (XSP has no deliverable).
- **Backtest data:** SPY and VIX daily closes 2010-01-05 → 2026-09-01 (`vol-risk-premium/reports/daily.csv`, 4,189 days). The simulator sells at `VIX − basis` and marks options at that day's VIX; the mismatch with XSP is the basis, frozen in §4.
- **Forward-track data:** own option-chain snapshots from `options-collector/` (15:35 CT Mon–Fri, raw, never overwritten, in the Parquet lake dataset `options`). **XSP is added to the collector universe at signature** — until then the forward track cannot start. SPY snapshots since 2026-09-04 (12 sessions as of 10-05) are the basis measurement, not the track.

## 3. Signal and position
- **Entry:** on each cycle start (every 21 trading days from the first signed-off session; calendar offset is a logged trial, §5), sell 1 ATM straddle per unit, strike = listed strike nearest spot, expiry per §2. **Entry blocked when VIX close > 30** (§6).
- **Hedge:** rehedge to zero delta at each close in SPY, Black-Scholes delta at the sold vol; zero rate, zero dividend (put-call parity forward measured within 0.09% of spot at 30 days).
- **Exit:** hold to expiry (cash settlement). Flatten rule in §6.
- **Sizing:** units = min(4, floor(account / 7,500), vol-target units) where vol-target units bring the sleeve's trailing-63-day realised P&L vol to **15% of the sleeve's capital**. Hard cap **4 straddles**. At $35k: ≤ 4 × ~$77 × 100 ≈ $31k notional. Capital at paper: the sleeve runs in the paper book at the equal-allocator weight and is scored on that path.
- **The scored path is the capped, vol-targeted one.** Constant-notional P&L (the README's object) is reported but is not the number in §5.
- **Never:** no strangles, no spreads, no intraday hedging, no second cycle open at once, no roll before expiry, no change to strike selection after signature.

## 4. Cost model — frozen numbers
| item | number | source | re-measure |
|---|---|---|---|
| ATM-vs-VIX basis | **2.4 vol points** | collector SPY ATM 30d IV vs VIX close, 12 sessions 2026-09-04 → 10-05 (ATM 12.7% vs VIX 15.1 on 10-05; horizons_latest.csv) | once, at **60 SPY collector sessions** (≈ 2026-11-25 if no gaps), then frozen. Re-measure uses the mean over all sessions to date. If XSP ATM IV differs from SPY's by > 0.3 pt on the first 10 XSP sessions, XSP's number replaces it at the same re-measure date. |
| Straddle spread given up at entry | **0.6% of mid (SPY, median)** → XSP: **to be measured, provisionally 1.5%** | 12 SPY sessions, ATM 30d straddle, range 0.2–1.7% | XSP measured over its first 10 collector sessions; whichever is larger of measured and 1.5% is frozen |
| Hedge slippage | 1 bp one-way per SPY share | project default; SPY is the most liquid listed equity | none |
| Commission | $0.65/contract/leg + $0 SPY | Alpaca options paper schedule | none |
| Margin | **not modelled in the backtest** (disclosed §8); forward track records broker-reported margin each mark | — | — |

## 5. The test
**Prior layer — backtest.** `vrp.simulate(iv_offset=0.024, option_spread=<§4>, cost_bps=1.0)` on the 2010–2026 panel, then the capped vol-targeted path on top of it. Untouched window: last fifth of sessions (`HOLDOUT = 0.2`, same rule as the book, ≈ 2023-05 → 2026-09), locked with `research.alpha.base.Untouched` **before** any in-sample run. Walk-forward on the pre-window with the project's existing `walk.py`.
- **Clears if:** untouched-window net Sharpe of the capped path **> 0**.
- **If not:** one row in `REFUSALS.md §2` with the numbers, and this candidate stops. No second basis, no second spread.

**Gate layer — forward paper track.** XSP straddles sold at real listed strikes at real snapshot mids, hedged daily at the SPY close, marked at the 15:35 collector snapshot, run inside the paper book at the equal-allocator weight. **Minimum 6 months (≥ 126 sessions, ≥ 6 full cycles) from the first XSP cycle.** Earliest possible start: the first collector session after signature + XSP add; earliest possible gate decision: ~April 2027.

**Comparator and promotion rule (verbatim, binding):**
> VRP is promoted into the live book only if, on the forward track of at least 126 sessions, (a) its own net Sharpe is above 1.0, (b) the book including VRP at the equal-allocator weight has a higher net Sharpe than the book without it over the same sessions, and (c) the 1/N attribution t-statistic of the VRP stream is positive. Any one failing is a refusal, logged in REFUSALS.md §2.

**Trial count, enumerated now:** cycle offsets 0–20 (21 trials, the project's existing luck test) × basis {2.4} × spread {§4} × cap {4} = **21 logged trials** before the window is opened, plus the capped-path variant = 22. Every one goes through `research.registry.experiments.Registry` so `trials()` counts it. The DSR reported uses that count, never fewer.

## 6. Kill criteria
- **No new cycle** when the VIX close > 30 at the entry session. (Backtest worst days: VIX 82.7, 75.5, 48.0, 40.8, 37.3 — all above 30 at the time, but entry was in the prior calm month; this rule limits *adding* exposure into stress, not the loss on the cycle already open. That is disclosed, not hidden.)
- **Flatten the sleeve** when a single day's sleeve P&L is worse than **−10% of the sleeve's open notional**, at the next close; no re-entry until the next scheduled cycle start and VIX ≤ 30.
- **Book limits unchanged:** 3% daily loss halt, 15% drawdown half-size, 25% kill, 3× gross, 50% per name (`framework/book/broker.py` Limits).
- **Retire the candidate** when: the prior layer fails (§5), or the forward track shows ≥ 2 flatten events in any 126 sessions, or the re-measured basis at 60 sessions exceeds **3.0 points** (the README's own table says the strategy is ~0 there and we do not run tracks on strategies we already believe are zero).

## 7. What would make me wrong
1. The premium is paid by the OTM put strip, not ATM: forward-track ATM straddle P&L ≈ 0 while the collector's 25-delta skew premium stays wide (skew series already in `horizons_latest.csv`). Number: 6-month ATM Sharpe < 0.3 with mean skew > 4 pts.
2. Costs on XSP eat it: measured XSP straddle spread > 2.5% of mid → the project's sensitivity table puts net annual P&L near zero.
3. The 2010–2026 sample flatters short vol (one crash, long bull): a forward drawdown deeper than −14% of notional inside the first 6 months on < 4 units.

## 8. Disclosures
- Margin and forced liquidation are **not** in the backtest; the forward track records broker margin so the live-sizing rule can be set from data rather than guessed.
- The backtest sells SPY-at-VIX-minus-basis and settles physically; the traded instrument is XSP, cash-settled. The basis is a scalar standing in for a time-varying quantity.
- Discrete daily hedging at the close, 1 bp; a VIX-80 close does not fill at the close.
- Sixteen years, one market; the sample worst day is a lower bound, not an estimate. The −14%-of-account single-day figure that ruled out full SPY contracts is from that sample.
- The paper book's own track has a 4-session gap (2026-09-12/14/15/16, DNS failures) and a 13-session pause (09-18 → 10-05, travel KILL); the VRP forward track starts clean after both.
- Author's prior, stated so it can be held against the result: **I expect the prior layer to clear narrowly (Sharpe 0.3–0.7 on the untouched window) and the forward track to fail the 1.0 bar.** If that is the outcome it is still worth 6 months, because the collector's chain history is the asset either way.

---
*Signature:* Kelvin He · 2026-10-06 (in session, chat record) · "I have read §5 and §6 and agree the result is binding either way."
