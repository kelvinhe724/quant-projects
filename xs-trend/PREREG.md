# Pre-registration — XS-TREND

Status: **SIGNED 2026-10-07** by Kelvin He, in session ("sign both"), before any backtest ran against it. The commit that adds this file is the lock; later changes are second looks for `REFUSALS.md §4`. Template: `research/PREREG-TEMPLATE.md`. Bar and gate shape: `vol-risk-premium/PREREG.md` §5.

## `XS-TREND`: 50/200 long-only on a point-in-time large-cap universe

**Hypothesis.** Time-series trend in US large caps, long-only, survives costs and a point-in-time universe: the published 1.02 on 2021–26 was earned on 190 names that are known to have lasted, and `pit-universe/` measured that bias at +4.0%/yr on equal weight. If half the Sharpe survives the PIT fix, it still clears the bar's untouched test; if it doesn't, the published number was survivorship.
**Instrument.** Shares (Alpaca paper; fractional fine), top-N by trailing 63-day dollar volume from the PIT S&P 500 panel, N frozen at 100. Long only; weight 1/N over names with 50d MA > 200d MA, cash otherwise.
**Rebalance.** Month-end, signals at t−1 close, fills next open (the book's convention).
**Costs.** 10 bps one-way + the engine's spread/impact model; zero borrow (long only).
**Sizing.** Scored at 15% vol target on the sleeve; gross ≤ 1.0; no leverage.
**Test.** Prior: `research.alpha` walk-forward 2005–2020, untouched 2021–2026 locked first; clears if untouched net Sharpe > 0 **and** > buy-and-hold of the same PIT universe at the same vol (this one must beat beta, not zero). Gate: 6-month paper track as a sleeve of the book; promotion rule as VRP's — raises the book's untouched Sharpe, 1/N attribution t > 0.
**Kills.** Flatten to cash when the sleeve's drawdown passes 20%; retire if the untouched window fails or the PIT rerun loses more than half the published Sharpe.
**Author's prior.** Untouched Sharpe 0.4–0.7 after PIT and costs; beta ~0.6 to SPY, so the attribution test is the one it fails.
**Trials.** N ∈ {100} × MA pair ∈ {50/200} = 1, plus the published variant for reference = 2. No grid.

