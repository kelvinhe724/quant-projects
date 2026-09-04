"""Offline checks on synthetic panels: accrual arithmetic, neutrality, costs, liquidation.

Run: python3 check.py
"""
import os

import numpy as np
import pandas as pd

import carry
from carry import (PERIODS_PER_YEAR, annualise, funding_filter, implied_funding,
                   metrics, simulate)

rng = np.random.default_rng(7)
REPORTS = os.path.join(os.path.dirname(__file__), "reports")

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def make_panel(spot, perp=None, rate=0.0, high=None):
    """Build a panel with explicit price paths, for arithmetic that can be done by hand."""
    spot = np.asarray(spot, float)
    perp = spot.copy() if perp is None else np.asarray(perp, float)
    idx = pd.date_range("2021-01-01", periods=len(spot), freq="8h", tz="UTC")
    return pd.DataFrame({
        "rate": np.full(len(spot), rate) if np.isscalar(rate) else np.asarray(rate, float),
        "interval_hours": 8,
        "spot": spot,
        "perp": perp,
        "perp_high": perp if high is None else np.asarray(high, float),
        "perp_low": perp,
        "spot_high": spot if high is None else np.asarray(high, float),
        "spot_low": spot,
    }, index=idx)


FREE = dict(spot_fee_bps=0.0, perp_fee_bps=0.0, slip_bps=0.0)
STATIC = dict(rebalance_every=10 ** 9, topup_at=0.0, **FREE)

# Funding accrual by hand. Capital 1 splits 1/1.5 into spot notional at 50,000,
# so the position is 1/(1.5*50000) coins. Funding settles at 51,000, 52,000 and
# 53,000 at 0.01% a period, all of it received because the position is short.
prices = [50000.0, 51000.0, 52000.0, 53000.0]
qty = (1 / 1.5) / 50000.0
expected = qty * 0.0001 * (51000 + 52000 + 53000)
book = simulate(make_panel(prices, rate=0.0001), **STATIC)
check(f"funding accrual matches the hand-computed {expected:.9f} "
      f"(got {book['funding'].sum():.9f})",
      np.isclose(book["funding"].sum(), expected, rtol=1e-12))
check("with zero basis and no fees, equity is exactly 1 plus the funding",
      np.isclose(book["equity"].iloc[-1], 1 + expected, rtol=1e-12))
check("no funding is booked in the entry period",
      book["funding"].iloc[0] == 0.0)

# Neutrality. A violent price path with no basis and no funding should leave the
# book flat: every dollar the spot leg makes, the perp leg loses.
path = 40000 * np.exp(np.cumsum(rng.normal(0, 0.03, 900)))
flat = simulate(make_panel(path), rebalance_every=30, **FREE)
check(f"delta-neutral book is flat through a {path.max() / path.min():.1f}x price "
      f"swing (drift {flat['equity'].iloc[-1] - 1:.2e})",
      np.allclose(flat["equity"], 1.0, atol=1e-12))

# Same path, but with a realistic wobbling basis so returns are not identically
# zero. The book should still have no beta to the underlying.
prem = 0.001 + 0.0015 * rng.standard_normal(len(path))
noisy = simulate(make_panel(path, perp=path * (1 + prem), rate=0.0001),
                 rebalance_every=30, **FREE)
spot_ret = pd.Series(path, index=noisy.index).pct_change().fillna(0.0)
beta = np.polyfit(spot_ret[1:], noisy["ret"][1:], 1)[0]
long_only = metrics(pd.DataFrame({"equity": path / path[0], "ret": spot_ret,
                                  "funding": 0.0, "price": 0.0, "fees": 0.0}))
check(f"beta to the underlying is ~0 (got {beta:.5f}, spot vol "
      f"{long_only['annual_vol']:.0%})", abs(beta) < 0.02)
check(f"carry vol is a small fraction of spot vol "
      f"({metrics(noisy)['annual_vol']:.1%} vs {long_only['annual_vol']:.0%})",
      metrics(noisy)["annual_vol"] < 0.25 * long_only["annual_vol"])

# Costs.
panel = make_panel(path, perp=path * 1.001, rate=0.0001)
terminal = [simulate(panel, rebalance_every=30, spot_fee_bps=f, perp_fee_bps=f / 2,
                     slip_bps=0.0)["equity"].iloc[-1] for f in (0.0, 5.0, 10.0, 20.0)]
check(f"fees reduce the terminal equity, monotonically "
      f"({', '.join(f'{t:.4f}' for t in terminal)})",
      all(a > b for a, b in zip(terminal, terminal[1:])))
check("zero fees means zero fees paid",
      simulate(panel, **FREE)["fees"].sum() == 0.0)
check("rebalancing more often costs more",
      simulate(panel, rebalance_every=3)["fees"].sum()
      > simulate(panel, rebalance_every=300)["fees"].sum())

# Negative funding is a cost to the short, not a gain.
neg = simulate(make_panel(prices, rate=-0.0001), **STATIC)
check("negative funding is paid out, not received",
      neg["funding"].sum() < 0 and neg["equity"].iloc[-1] < 1.0)

# Liquidation. A steady rally with no rebalancing has to take out a thin buffer
# and leave a thick one alone.
rally = 30000 * np.linspace(1.0, 1.8, 400)
thin = simulate(make_panel(rally), margin_frac=0.2, **STATIC)
thick = simulate(make_panel(rally), margin_frac=2.0, **STATIC)
check(f"a 20% margin buffer is liquidated on an 80% rally "
      f"(at {thin.attrs['liquidated_at']})", thin.attrs["liquidated_at"] is not None)
check("a 200% buffer survives the same rally",
      thick.attrs["liquidated_at"] is None)
check("liquidation is triggered by the intraperiod high, not the close",
      simulate(make_panel(rally * 0 + 30000, high=np.where(np.arange(400) == 200,
                                                           60000, 30000)),
               margin_frac=0.3, **STATIC).attrs["liquidated_at"] is not None)
check("rebalancing prevents the liquidation the same path would otherwise cause",
      simulate(make_panel(rally), margin_frac=0.2, rebalance_every=5,
               **FREE).attrs["liquidated_at"] is None)
check("topping up the margin prevents it too, with no calendar rebalance",
      simulate(make_panel(rally), margin_frac=0.2, rebalance_every=10 ** 9,
               topup_at=0.4, **FREE).attrs["liquidated_at"] is None)
check("the position is closed and stays closed after a liquidation",
      (thin["qty"].loc[thin.attrs["liquidated_at"]:] == 0).all())

# A perp that wicks far above the spot index is survivable, because Binance
# liquidates on the index rather than on its own order book.
wick = make_panel(np.full(400, 30000.0), perp=np.full(400, 30000.0))
wick.loc[wick.index[200], "perp_high"] = 60000.0
check("a perp wick with a calm index does not liquidate a spot-marked position",
      simulate(wick, margin_frac=0.3, **STATIC).attrs["liquidated_at"] is None)
check("the same wick does liquidate if the mark follows the perp",
      simulate(wick, margin_frac=0.3, mark="perp",
               **STATIC).attrs["liquidated_at"] is not None)

# Determinism, and reproducibility from the cached CSVs.
check("two runs on the same input are bit-identical",
      simulate(panel, rebalance_every=30).equals(simulate(panel, rebalance_every=30)))

saved = os.path.join(REPORTS, "equity_BTCUSDT.csv")
if os.path.exists(saved):
    import data
    stored = pd.read_csv(saved, index_col=0, parse_dates=True)
    rerun = simulate(data.panel("BTCUSDT"))
    check("the saved BTCUSDT equity curve reproduces from the cached data",
          np.allclose(stored["equity"], rerun["equity"], rtol=1e-10))
else:
    print("SKIP  reproducibility of the saved equity curve (run.py has not run)")

# Formula plumbing.
check("zero premium implies the 0.01% interest rate",
      np.isclose(implied_funding(0.0), carry.INTEREST))
check("a 0.10% premium is damped to 0.05% by the clamp",
      np.isclose(implied_funding(0.001), 0.0005))
check("a 2% premium is capped at 0.75%",
      np.isclose(implied_funding(0.02), carry.RATE_CAP))
check(f"0.01% per 8h annualises to {annualise(0.0001):.2%}",
      np.isclose(annualise(0.0001), 0.0001 * PERIODS_PER_YEAR))

# The trailing filter must not see the period it is deciding on.
rates = pd.Series(rng.normal(-0.0002, 0.0003, 500))
spiked = rates.copy()
spiked.iloc[300] = 0.05
base_panel = make_panel(np.full(500, 30000.0), rate=rates.to_numpy())
spike_panel = make_panel(np.full(500, 30000.0), rate=spiked.to_numpy())
f_base = funding_filter(base_panel)
f_spike = funding_filter(spike_panel)
check("the funding filter at t ignores the rate at t",
      f_base.iloc[:301].equals(f_spike.iloc[:301]))
check("one large positive rate turns the filter on for the next 21 periods",
      f_spike.iloc[301:322].all() and not f_base.iloc[301:322].any())

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
