# Strategy pre-registration — template

One page, written and signed **before** the first backtest runs. A field left blank is a field that will be filled in to fit the result; every field is required. The signed copy is committed to `quant-projects/<project>/PREREG.md` and its git hash is the lock. Anything changed after that hash is disclosed in `REFUSALS.md §4` as a second look.

Why this exists: `quant-projects/REFUSALS.md` lists two published claims that were false and seven windows that were read twice. Every one of them was a decision made after the number was known. This page moves those decisions to before.

Version 2 (2026-10-08) adds §2's tradability line and §2a–2b after the first two candidates failed on exactly those questions after being built: one on an instrument the broker will not trade, one on a window its own project had already published.

---

## 0. Identity
- **Name / project dir:**
- **Author / reviewer:** C2 builds · Kelvin reviews and signs
- **Date signed:** · **Commit hash of this file:**

## 1. Hypothesis (one sentence, falsifiable)
*Who is paying this premium and why does it survive costs for an account this size?*

## 2. Universe and instrument
- Instruments actually traded (exact contract, not the proxy):
- Data used for the backtest, and where it differs from the traded instrument:
- Data used for the forward track:
- **Can the account trade every leg?** For each leg: the venue, the account type, and the broker's own
  flag or rule that permits it, checked on a named date (Alpaca `get_asset(...).shortable`, the options
  level and the strategies it allows, Kalshi KYC status). A leg the account cannot place is a leg the
  backtest may not profit from. *VRP (2026-10) was pre-registered on a straddle Alpaca allows at no
  level; the book shorted FXE for a month in backtests before the broker refused the first live short.*

## 2a. Window provenance — who has already read this data
- Every commit, README table, or report that published a number covering any part of the untouched
  window, with the date. One hit means the window is a second look, and the pre-registration says so.
- The last session anyone in this repo has read for this instrument set. The untouched window must
  start after it, or the test is on history already seen. *Both 2026-10 candidates were locked on windows this repo had already published a month earlier, one by its own README and one by the project whose rule it reused (REFUSALS §4).*

## 2b. Power — what the test can tell apart
- The forward track's minimum length, the target net Sharpe, and the probability that a strategy with
  **no** edge passes the bar over that length (normal approximation, then the return shape: for a short
  option book state the share of P&L in the worst 1% of days). If the no-edge pass rate is above 25%,
  say what else decides at the end of the track: a longer track, a shrunk estimate, or Kelvin's judgment,
  named now.

## 3. Signal and position
- Entry rule, exit rule, holding period, rebalance schedule — exact:
- Sizing rule (vol target, caps), and the *capped* path is what gets scored:
- What the strategy never does (explicit non-actions):

## 4. Cost model — frozen numbers
- Spread, commission, slippage, financing, margin — each with its source and the date measured:
- Rule for re-measuring (once, at a pre-set date), after which the number is frozen:

## 5. The test
- **Prior layer:** backtest window, untouched window (per `research.alpha.base.Untouched` — locked by hash, opened once), the number that must clear and what happens if it does not:
- **Gate layer:** forward paper track — venue, mark source, minimum length, start date:
- **Comparator:** what it must beat (the book, not a zero-Sharpe null):
- **Promotion rule, verbatim** (same shape as `framework/book/validate.py:328`):
- **Trial count:** every configuration that will be run, enumerated now; the DSR uses this count plus whatever else gets logged, never less.

## 6. Kill criteria — written before, applied without appeal
- Entry blocked when:
- Position flattened when:
- Strategy retired (goes to `REFUSALS.md`) when:

## 7. What would make me wrong
Three observable outcomes, each with the number that would show it.

## 8. Disclosures
Known optimism in the test (margin not modelled, proxy data, survivorship of the sample, discrete hedging), each one named.

---
*Signature line:* Kelvin He · date · "I have read §5 and §6 and agree the result is binding either way."
