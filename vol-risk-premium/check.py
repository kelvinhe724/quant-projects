"""Offline checks: known Black-Scholes values, planted volatility, zero-premium drift, costs.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from vrp import (TRADING_DAYS, bs_delta, bs_price, bs_vega, drawdown, forward_realised_vol,
                 implied_vol, metrics, realised_vol, simulate, straddle_delta,
                 straddle_price)

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


# Black-Scholes reference values, S=100 K=100 T=1 sigma=0.20 r=q=0.
# At-the-forward the call and put are equal and price to 2*(N(sigma/2)-0.5)*S.
call = bs_price(100, 100, 1.0, 0.20, "C")
put = bs_price(100, 100, 1.0, 0.20, "P")
check(f"ATM call price 7.9656 (got {call:.4f})", abs(call - 7.965567) < 1e-5)
check("put-call parity holds at zero rate", abs(call - put) < 1e-10)
check(f"ATM call delta 0.5398 (got {bs_delta(100, 100, 1.0, 0.20, 'C'):.4f})",
      abs(bs_delta(100, 100, 1.0, 0.20, "C") - 0.539828) < 1e-5)
check("call delta minus put delta is 1",
      abs(bs_delta(100, 100, 1.0, 0.20, "C") - bs_delta(100, 100, 1.0, 0.20, "P") - 1) < 1e-12)
check(f"ATM vega 39.695 (got {bs_vega(100, 100, 1.0, 0.20):.3f})",
      abs(bs_vega(100, 100, 1.0, 0.20) - 39.695255) < 1e-3)

deep_call = bs_price(150, 100, 0.5, 0.20, "C")
check(f"deep ITM call is worth about intrinsic ({deep_call:.4f} vs 50)",
      49.99 < deep_call < 50.05)
check("deep ITM call delta is 1", bs_delta(300, 100, 0.5, 0.20, "C") > 0.9999)
check("deep OTM put delta is 0", abs(bs_delta(300, 100, 0.5, 0.20, "P")) < 1e-4)
check("delta matches a finite difference of price",
      abs(bs_delta(105, 100, 0.4, 0.25, "C")
          - (bs_price(105.001, 100, 0.4, 0.25, "C")
             - bs_price(104.999, 100, 0.4, 0.25, "C")) / 0.002) < 1e-5)
check("vega matches a finite difference of price",
      abs(bs_vega(105, 100, 0.4, 0.25)
          - (bs_price(105, 100, 0.4, 0.2501, "C")
             - bs_price(105, 100, 0.4, 0.2499, "C")) / 0.0002) < 1e-2)
check("expiry price is intrinsic", bs_price(107, 100, 0.0, 0.2, "C") == 7.0
      and bs_price(107, 100, 0.0, 0.2, "P") == 0.0)
check("implied vol inverts the price",
      abs(implied_vol(bs_price(100, 95, 0.3, 0.27, "P"), 100, 95, 0.3, "P") - 0.27) < 1e-6)

check("straddle price is call plus put",
      abs(straddle_price(100, 100, 0.25, 0.2)
          - bs_price(100, 100, 0.25, 0.2, "C") - bs_price(100, 100, 0.25, 0.2, "P")) < 1e-12)
check("ATM straddle delta is near zero",
      abs(straddle_delta(100, 100, 1 / 12, 0.15)) < 0.03)
check("straddle delta is positive above the strike and negative below",
      straddle_delta(110, 100, 1 / 12, 0.15) > 0.95
      and straddle_delta(90, 100, 1 / 12, 0.15) < -0.95)

# Realised volatility estimator on a planted volatility.
rng = np.random.default_rng(7)
TRUE_VOL = 0.18
n = 6000
daily = TRUE_VOL / np.sqrt(TRADING_DAYS)
ret = pd.Series(rng.normal(0, daily, n), index=pd.bdate_range("2005-01-03", periods=n))
est = realised_vol(ret)
check(f"realised vol recovers the planted {TRUE_VOL} (got {est:.4f})",
      abs(est - TRUE_VOL) < 0.005)

fwd = forward_realised_vol(ret, horizon=21)
check(f"forward realised vol averages to the planted level (got {fwd.mean():.4f})",
      abs(fwd.mean() - TRUE_VOL) < 0.01)
check("forward realised vol at t uses only returns after t",
      np.isclose(fwd.iloc[100], np.sqrt(TRADING_DAYS * (ret.iloc[101:122] ** 2).mean())))
spiked = ret.copy()
spiked.iloc[500] *= 20
check("a spike at t=500 leaves the forward vol at t=500 untouched",
      np.isclose(fwd.iloc[500], forward_realised_vol(spiked, 21).iloc[500]))
check("the spike does move the forward vol of earlier dates",
      not np.isclose(fwd.iloc[490], forward_realised_vol(spiked, 21).iloc[490]))
check("forward realised vol is nan at the end of the sample", fwd.iloc[-1] != fwd.iloc[-1])

# A path whose realised volatility equals the implied volatility it is sold at
# should produce a delta-hedged P&L with no drift. This is the core no-arbitrage
# statement of the simulator, so it is the check that matters most.
SOLD_VOL = 0.20
n = 21 * 400
path_ret = rng.normal(0, SOLD_VOL / np.sqrt(TRADING_DAYS), n)
idx = pd.bdate_range("2000-01-03", periods=n)
fair = pd.DataFrame({"spy": 100 * np.exp(np.cumsum(path_ret)),
                     "vix": SOLD_VOL * 100}, index=idx)

free = simulate(fair, cost_bps=0.0, option_spread=0.0)
m = metrics(free["pnl"])
per_cycle = free.groupby("cycle")["pnl"].sum()
tstat = per_cycle.mean() / per_cycle.std() * np.sqrt(len(per_cycle))
check(f"zero-premium hedged P&L has no drift (t = {tstat:.2f} over "
      f"{len(per_cycle)} cycles)", abs(tstat) < 2.5)
check(f"total P&L is small against its own noise ({m['total_pnl']:+.4f} of notional)",
      abs(m["total_pnl"]) < 3 * per_cycle.std() * np.sqrt(len(per_cycle)))
check("hedged P&L is far smaller than the unhedged option payoff",
      per_cycle.std() < 0.3 * free["premium"][free["premium"] > 0].mean())

# Selling at a volatility above what the path realises has to make money.
rich = fair.copy()
rich["vix"] = SOLD_VOL * 100 * 1.25
rich_pnl = simulate(rich, cost_bps=0.0, option_spread=0.0)["pnl"]
check(f"selling 25% rich is profitable ({rich_pnl.sum():+.3f} of notional)",
      rich_pnl.sum() > 0)
cheap = fair.copy()
cheap["vix"] = SOLD_VOL * 100 * 0.75
check("selling 25% cheap loses money", simulate(cheap, cost_bps=0.0,
                                                option_spread=0.0)["pnl"].sum() < 0)

# Costs.
costly = simulate(fair, cost_bps=5.0, option_spread=0.0)["pnl"]
check(f"hedge costs reduce P&L ({free['pnl'].sum():+.4f} -> {costly.sum():+.4f})",
      costly.sum() < free["pnl"].sum())
spread_only = simulate(fair, cost_bps=0.0, option_spread=0.02)["pnl"]
check(f"the option bid-ask reduces P&L ({spread_only.sum():+.4f})",
      spread_only.sum() < free["pnl"].sum())
check("a wider hedge spread costs more than a narrow one",
      simulate(fair, cost_bps=10.0, option_spread=0.0)["pnl"].sum() < costly.sum())
check("the option spread charge equals the stated fraction of premium",
      np.isclose(free["pnl"].sum() - spread_only.sum(),
                 0.02 * free["premium"][free["premium"] > 0].sum(), rtol=1e-9))

check("every day belongs to exactly one cycle", (free["cycle"] >= 0).all())
check("the hedge is unwound at the end of the sample", free["hedge"].iloc[-1] == 0.0)
check("cycles are consecutive and horizon days apart",
      (np.diff(sorted(free["cycle"].unique())) == 21).all())

# Metrics.
toy = pd.Series([0.01, -0.02, 0.03, -0.01, 0.005])
check("drawdown is zero at a new high", drawdown(toy).iloc[0] == 0)
check("drawdown is negative after a loss", drawdown(toy).min() < 0)
mm = metrics(pd.Series(rng.normal(0.001, 0.01, 3000)))
check(f"metrics recovers a planted Sharpe near 1.6 (got {mm['sharpe']:.2f})",
      abs(mm["sharpe"] - 0.001 / 0.01 * np.sqrt(TRADING_DAYS)) < 0.5)
left = pd.Series(np.concatenate([rng.normal(0.001, 0.004, 990), rng.normal(-0.05, 0.01, 10)]))
ml = metrics(left)
check(f"metrics detects negative skew ({ml['skew']:.2f})", ml["skew"] < -3)
check("trimming the worst 1% raises the Sharpe of a left-tailed series",
      ml["sharpe_ex_worst_1pct"] > ml["sharpe"])

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
