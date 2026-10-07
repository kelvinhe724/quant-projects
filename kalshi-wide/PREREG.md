# Pre-registration — KALSHI-WIDE

Status: **SIGNED 2026-10-07** by Kelvin He, in session ("sign both"), before any backtest ran against it. The commit that adds this file is the lock; later changes are second looks for `REFUSALS.md §4`. Template: `research/PREREG-TEMPLATE.md`. Bar and gate shape: `vol-risk-premium/PREREG.md` §5.

## `KALSHI-WIDE`: model fade in wide books only

**Hypothesis.** `kalshi-model/` found the market mid beats every model where books are tight, and the models beat the mid only where the quoted spread exceeds 10c — exactly where the published rule refused to trade because the spread ate the edge. The untested question is whether a **limit order at the model's price inside a wide book** (maker, not taker) gets filled often enough to earn the gap.
**Instrument.** Kalshi binaries, shadow book then demo exchange (no production flag). Books ≥ 10c wide, ≥ 24h to settlement, the categories the model was trained on.
**Rule.** Post a resting order at model-P minus a 2c margin on the side the model favours; cancel at 60 min or on a 3c move; size per the existing Kelly cap and the desk's hard dollar caps.
**Costs.** Kalshi maker fee schedule as of signature; fill assumption = **only touches recorded in the 15-min book snapshots** (the lake's `kalshi_books`), which understates fills — the conservative side.
**Test.** Prior: replay 2026-08-25 → today from `kalshi_books` with the frozen `kalshi-model` calibrator (no refit); clears if net P&L per $ staked > 0 with t > 2 on ≥ 300 fills. Gate: 6-month shadow track with the same fill rule; the bar's 1.0 Sharpe measured on daily P&L.
**Kills.** Pause on any day < −5% of the Kalshi sleeve capital; retire if fills < 100 in the first 60 days (the edge exists but can't be reached).
**Author's prior.** Fill rate is the whole story; I expect < 30% fills and a net near zero. Cheap to find out: the data is already collected and the scanner exists.
**Trials.** Margin {2c} × cancel {60 min} = 1 logged configuration.

