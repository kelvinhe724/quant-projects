# Strategy pre-registration — template

One page, written and signed **before** the first backtest runs. A field left blank is a field that will be filled in to fit the result; every field is required. The signed copy is committed to `quant-projects/<project>/PREREG.md` and its git hash is the lock. Anything changed after that hash is disclosed in `REFUSALS.md §4` as a second look.

Why this exists: `quant-projects/REFUSALS.md` lists two published claims that were false and seven windows that were read twice. Every one of them was a decision made after the number was known. This page moves those decisions to before.

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
