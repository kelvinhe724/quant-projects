"""Offline checks on planted series: the funding accrual, the z feature's lag, a planted dislocation
traded through the research harness, the exceedance and reversion counts, and the OKX minute builder.

Run: ../.venv/bin/python check.py
"""
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from research.alpha.base import backtest, positions  # noqa: E402
from research.features.store import Feature, FeatureStore, PeekError, Raw  # noqa: E402

import data  # noqa: E402
from research.registry.experiments import Registry  # noqa: E402
from basis import (BasisMR, basis, daily, daily_sharpe, dislocation, exceedance, gap_runs, reversion,  # noqa: E402
                   tr_index, venue_gap, zscore_feature)

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


idx = pd.date_range("2025-01-01", periods=2000, freq="5min", tz="UTC")
rng = np.random.default_rng(0)
spot = pd.DataFrame({"BTC": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx)))),
                     "ETH": 10 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx))))}, index=idx)

# a flat basis of -3 bps, one planted +20 bp excursion lasting 30 bars, then back
# a slow 0.5 bp ripple keeps the rolling std away from zero without ever reaching 2 sigma
ripple = 0.5e-4 * np.sin(2 * np.pi * np.arange(len(idx)) / 20)
b = pd.DataFrame({"BTC": -3e-4 + ripple, "ETH": -3e-4 + ripple}, index=idx)
b.loc[idx[1000]:idx[1029], "BTC"] += 20e-4
perp = spot * (1 + b)
settle = idx[(idx.hour % 8 == 0) & (idx.minute == 0)]
funding = pd.DataFrame(0.0, index=settle, columns=spot.columns)
funding.iloc[3] = [1e-4, -2e-4]

P = tr_index(spot, perp, funding)
jump = P.loc[settle[3]] / (spot / perp).loc[settle[3]] / (P.loc[settle[2]] / (spot / perp).loc[settle[2]])
check("funding at a settlement multiplies the index by 1 + rate, for both signs",
      np.allclose(jump.to_numpy(), [1 + 1e-4, 1 - 2e-4]))
check("no accrual on a bar with no settlement",
      np.allclose((P / (spot / perp)).iloc[1:10].to_numpy(), 1.0))
late = funding.copy()
late.index = late.index + pd.Timedelta(milliseconds=1)
check("a settlement stamped 1 ms after the hour (as Binance does) still lands on its bar",
      np.allclose(tr_index(spot, perp, late).to_numpy(), P.to_numpy()))
r_spot, r_perp = spot.pct_change(), perp.pct_change()
r_idx = (spot / perp).pct_change()
check("the index return is spot minus perp to first order",
      np.allclose(r_idx.iloc[1:], (r_spot - r_perp).iloc[1:], atol=1e-5))
check("basis is perp over spot minus one", np.allclose(basis(spot, perp), b))

raw = Raw({"close": P, "basis": b, "open": P, "high": P, "low": P, "volume": P * 0 + 1})
store = FeatureStore([zscore_feature(50)])
check("the z feature passes the peek audit", store.audit(raw))
peek = Feature("z_peek", lambda r: r.frames["basis"].rolling(50, center=True).mean(), lag=1)
try:
    FeatureStore([peek]).audit(raw)
    check("a centred window is caught by the audit", False)
except PeekError:
    check("a centred window is caught by the audit", True)
panel = store.build(raw)
z = panel["z_50"]["BTC"]
first = idx[1000]
check("the z at the excursion bar does not yet see the excursion (lag 1)",
      abs(z.loc[first]) < 1 and z.loc[idx[1001]] > 3)

alpha = BasisMR(window=50, entry=2.0, exit=0.5)
pos = positions(alpha, panel, idx)
pb = pos["BTC"]
check("enters long the index one bar after the basis jumps rich",
      pb.loc[idx[1000]] == 0 and pb.loc[idx[1001]] == 1.0)
check("holds through the excursion while |z| stays above the exit band", (pb.loc[idx[1001]:idx[1030]] == 1.0).all())
check("flat again after the basis comes back and never trades ETH",
      pb.iloc[1100:].eq(0).all() and (pos["ETH"] == 0).all())
check("no position while z is between exit and entry once flat",
      pb.iloc[:1000].eq(0).all())

# the trade earns the excursion's give-back: it enters after +20 bp and exits after
# the basis has fallen back, so before costs it makes about 20 bp of P, minus 2 x cost
r0 = backtest(pos, P, cost_bps=0.0)
r17 = backtest(pos, P, cost_bps=17.0)
gross = (1 + r0).prod() - 1
check("gross P&L of the planted trade is close to the 20 bp excursion", 15e-4 < gross < 25e-4)
check("net of 17 bp a side the same trade loses about 34 bp against gross",
      np.isclose(((1 + r17).prod() - 1) - gross, -2 * 17e-4, atol=2e-5))

d = dislocation(b, window=50)
ex = exceedance(d, [5, 15, 25])
check("exceedance counts the planted excursion at 5 and 15 bp and nothing at 25",
      ex.loc[5, "BTC"] > 0 and ex.loc[15, "BTC"] > 0 and ex.loc[25, "BTC"] == 0 and (ex["ETH"] == 0).all())
rv = reversion(d, 15, [1, 40])
check("reversion sees one episode, nothing given back after 1 bar, most of it after 40",
      rv.loc["BTC", "episodes"] == 1 and abs(rv.loc["BTC", "given_back_1"]) < 0.1
      and rv.loc["BTC", "given_back_40"] > 0.9)

dr = daily(pd.Series([0.01, 0.01, -0.005], index=pd.to_datetime(["2025-01-01 01:00", "2025-01-01 02:00",
                                                                     "2025-01-02 03:00"], utc=True)))
check("daily compounds the bars of one UTC day", np.allclose(dr.to_numpy(), [1.01 * 1.01 - 1, -0.005]))
with tempfile.TemporaryDirectory() as tmp:
    reg = Registry(tmp)
    rb = pd.Series(rng.normal(0, 1e-3, len(idx)), index=idx)
    rb.iloc[288 * 2:288 * 3] = 0.0  # one flat day, dropped by both
    ds = daily_sharpe(rb)
    check("the walk-forward fold score is the registry's daily Sharpe to the last digit",
          np.isclose(ds, reg.dsr(daily(rb))["sharpe"], rtol=0, atol=1e-12) and abs(ds) < 30)
    check("the registry's Sharpe uses the same ddof as its logged sharpe_daily",
          np.isclose(reg.record("x", {}, ["BTC"], ("a", "b"), daily(rb))["sharpe_daily"] * np.sqrt(252),
                     reg.dsr(daily(rb))["sharpe"], rtol=0, atol=1e-12))

# OKX minute builder on planted trades
csv = ("instrument_name,trade_id,side,price,size,created_time\n"
       "BTC-USDT,1,buy,100,1,1700000000000\n"
       "BTC-USDT,2,sell,110,3,1700000030000\n"
       "BTC-USDT,3,buy,120,1,1700000060000\n")
m = data.okx_minutes(csv.encode(), "BTC-USDT").set_index("ts")
check("okx minute bars: last, size-weighted vwap, volume and count",
      len(m) == 2 and m["last"].iloc[0] == 110 and np.isclose(m["vwap"].iloc[0], 107.5)
      and m["volume"].iloc[0] == 4 and m["trades"].iloc[0] == 2 and m["last"].iloc[1] == 120)
check("binance stamps: milliseconds and microseconds land on the same instant",
      data.to_utc(pd.Series([1700000000000]))[0] == data.to_utc(pd.Series([1700000000000000]))[0])

minutes = pd.date_range("2026-07-01", periods=100, freq="min", tz="UTC")
pa = pd.Series(100.0, index=minutes)
pb2 = pd.Series(100.0, index=minutes)
pb2.iloc[40:47] = 100.0 * (1 + 30e-4)
pb2.iloc[80:82] = 100.0 * (1 - 12e-4)
g = venue_gap(pa, pb2.drop(minutes[5]))
check("venue gap is a over b minus one on the shared minutes",
      len(g) == 99 and np.isclose(g.iloc[41], 1 / 1.003 - 1) and g.iloc[0] == 0)
runs = gap_runs(g, 10)
check("gap runs above 10 bp: one of 7 minutes and one of 2",
      sorted(runs.tolist()) == [2, 7])
check("no runs above a threshold nothing crosses", len(gap_runs(g, 50)) == 0)

print(f"\n{sum(checks)}/{len(checks)} passed")
sys.exit(0 if all(checks) else 1)
