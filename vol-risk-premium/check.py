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


# --- prior.py: the pre-registered kill rules, sizing and lock (PREREG.md §3, §5, §6) ---
import json, os, tempfile
import prior

days = pd.bdate_range("2020-01-01", periods=120)
calm = pd.DataFrame({"spy": 300.0 + np.arange(120) * 0.01, "vix": 15.0}, index=days)
calm["ret"] = np.log(calm["spy"]).diff().fillna(0)
sim = simulate(calm, horizon=21, cost_bps=0.0, option_spread=0.0, iv_offset=0.0, offset=0)
spike = calm.copy(); spike.loc[days[21], "vix"] = 35.0
k = prior.kills(spike, simulate(spike, horizon=21, cost_bps=0.0, option_spread=0.0, offset=0))
check("kill: a cycle whose entry day has VIX above 30 is never opened",
      (k.loc[days[21]:days[41], "cycle"] == -1).all() and (k.loc[days[21]:days[41], "pnl"] == 0).all()
      and (k.loc[days[0]:days[20], "cycle"] == 0).all())
crash = calm.copy(); crash.loc[days[5]:, "spy"] = 300.0 * 0.80  # a 20% gap on day 5 of the first cycle
cs = simulate(crash, horizon=21, cost_bps=0.0, option_spread=0.0, offset=0)
kc = prior.kills(crash, cs)
check("kill: a day worse than -10% of notional flattens the cycle, nothing booked after it",
      cs.loc[days[5], "pnl"] <= prior.FLATTEN and (kc.loc[days[6]:days[21], "pnl"] == 0).all()
      and kc.loc[days[5], "pnl"] == cs.loc[days[5], "pnl"])
sz = prior.capped(calm, sim, account=35_000, cap=4, per_unit_account=7_500, target=0.15)
check("sizing: units are whole contracts, fixed within a cycle, never above the cap",
      (sz.units == sz.units.round()).all() and sz.units.max() <= 4
      and all(g.units.nunique() == 1 for _, g in sz.join(sim.cycle).groupby("cycle")))
sz1 = prior.capped(calm, sim, account=10_000, cap=4, per_unit_account=7_500, target=0.15)
check("sizing: the account-per-unit rule binds before the cap (10k account -> at most 1 unit)", sz1.units.max() <= 1)
tiny = prior.capped(calm, sim, account=35_000, cap=4, per_unit_account=7_500, target=0.0001)
check("sizing: a vol target below one contract's vol sizes to zero, not to a fraction", (tiny.units == 0).all())
with tempfile.TemporaryDirectory() as tmp:
    old = prior.LOCK, prior.FROZEN, prior.REGISTRY
    prior.LOCK, prior.FROZEN, prior.REGISTRY = (type(prior.LOCK)(tmp) / "u.json", type(prior.LOCK)(tmp) / "f.json",
                                                type(prior.LOCK)(tmp) / "reg")
    try:
        check("prior: the spread is provisional until spread-frozen.json exists", prior.spread() == (prior.SPREAD_PROVISIONAL, False))
        json.dump({"spread": 0.021}, open(prior.FROZEN, "w"))
        check("prior: a frozen spread is read from the file", prior.spread() == (0.021, True))
    finally:
        prior.LOCK, prior.FROZEN, prior.REGISTRY = old
lock = prior.Untouched(str(prior.LOCK))
_vb = lock.read() if os.path.exists(prior.LOCK) else {}  # read() raises if the hash no longer matches
check("prior: the real untouched window is locked on 2023-04-28, its hash verifies, and if it has been opened "
      "it was opened once with a frozen spread and a recorded result",
      _vb.get("start") == "2023-04-28" and (not lock.opened or (os.path.exists(prior.FROZEN) and _vb.get("result") is not None)))

# forward track: a 0/0 quoted leg is "no quote" (nan), never a worthless straddle; 2026-10-21 booked +$1,599 on one
import forward as _fwd
_chain = pd.DataFrame({"snapshot": ["2026-10-21 20:38:21"] * 2, "expiry": [pd.Timestamp("2026-11-02")] * 2,
                       "strike": [780.0, 780.0], "cp": ["C", "P"], "bid": [0.0, 0.0], "ask": [0.0, 0.0],
                       "T": [0.03, 0.03], "spot": [775.7, 775.7]})
_mid0 = _fwd.mark(_chain, 780.0, pd.Timestamp("2026-11-02"))
check("forward mark: a 0/0 quoted straddle is no quote (nan), not a mid of 0", np.isnan(_mid0[1]) and not _mid0[3])
_mid1 = _fwd.mark(_chain.assign(bid=[9.0, 5.0], ask=[9.5, 5.5]), 780.0, pd.Timestamp("2026-11-02"))
check("forward mark: a quoted straddle marks at the mid", abs(_mid1[1] - 14.5) < 1e-9)

# branch r1-basis-clock: basis.py clock (usable sessions, lost-session reasons, session-60 date, XSP clause) on planted summaries
import basis as _bs


def _plant(root, rows, vix):
    """rows {date: {ticker: iv30 or None}} -> <root>/<date>/ day folders + summary/<date>_horizons.csv; vix {date: close}."""
    import pathlib
    root = pathlib.Path(root)
    (root / "summary").mkdir(parents=True, exist_ok=True)
    for d, tk in rows.items():
        (root / d).mkdir(exist_ok=True)
        if tk:
            pd.DataFrame([{"date": d, "ticker": k, "iv30": np.nan if v is None else v / 100, "stale": False}
                          for k, v in tk.items()]).to_csv(root / "summary" / f"{d}_horizons.csv", index=False)
    pd.DataFrame({"date": pd.to_datetime(list(vix)), "series": "VIXCLS", "value": list(vix.values()),
                  "collected_at": pd.Timestamp("2026-12-31")}).to_parquet(root / "vix.parquet")
    return root, root / "vix.parquet"


with tempfile.TemporaryDirectory() as _tmp:
    _signed = [("2026-09-04", 11.5943, 14.53), ("2026-09-09", 13.3449, 16.46), ("2026-09-10", 14.3907, 17.84),
               ("2026-09-11", 12.5395, 15.84), ("2026-09-17", 12.4779, 15.44), ("2026-09-29", 13.4068, 16.04),
               ("2026-09-30", 13.6189, 16.34), ("2026-10-02", 12.6602, 15.31), ("2026-10-05", 12.7401, 15.52),
               ("2026-10-06", 12.5583, 15.01)]
    _r, _v = _plant(_tmp + "/a", {d: {"SPY": iv} for d, iv, _ in _signed}, {d: x for d, _, x in _signed})
    _t, _ = _bs.table(_r, _v, "2026-09-04")
    _u = _t[_t.reason == "ok"]
    check(f"basis: the 10 signed rows give n == 10 and mean 2.900 +/- 0.005 (n {len(_u)}, mean {_u.basis.mean():.4f})",
          len(_u) == 10 and abs(_u.basis.mean() - 2.900) < 0.005)

    # Thanksgiving week: 11-25 has no SPY row, 11-26 is a holiday that still has a row, 11-27 is NaN iv30, 12-01 has no folder
    _good = {"2026-11-23": 13.0, "2026-11-24": 14.5, "2026-11-30": 12.0, "2026-12-02": 16.0}
    _rows = {d: {"SPY": iv, "QQQ": 20.0} for d, iv in _good.items()}
    _rows.update({"2026-11-25": {"QQQ": 20.0}, "2026-11-26": {"SPY": 15.0}, "2026-11-27": {"SPY": None}})
    _vix = {d: iv + 3.0 for d, iv in _good.items()}
    _vix.update({"2026-11-25": 18.0, "2026-11-26": 18.0, "2026-11-27": 18.0})
    _r, _v = _plant(_tmp + "/b", _rows, _vix)
    _t, _f = _bs.table(_r, _v, "2026-11-23")
    _u = _t[_t.reason == "ok"]
    _tally = {k: int((_t.reason == k).sum()) for k in _bs.REASONS}
    check("basis: the holiday-dated row, the missing-SPY row, the NaN iv30 and the folderless session are all excluded",
          list(_u.index) == list(_good) and "2026-11-26" not in _t.index
          and _tally == {"no-folder": 1, "empty": 0, "no-SPY": 2, "no-rate": 0})
    check(f"basis: VIX = iv30 + 3.0 gives a mean of +3.00 (got {_u.basis.mean():.4f})", abs(_u.basis.mean() - 3.0) < 1e-9)
    # hand count: 4 usable, last folder 12-02, so sessions 5..8 are 12-03, 12-04, 12-07, 12-08
    check("basis: the projected session-8 date is the hand-counted 2026-12-08", _bs.project(_t, _f[-1], target=8) == "2026-12-08")

    _d = pd.bdate_range("2026-11-02", periods=10).strftime("%Y-%m-%d")
    for _gap, _want in ((0.4, True), (0.1, False)):
        _r, _v = _plant(_tmp + f"/x{_gap}", {d: {"SPY": 13.0, "^XSP": 13.0 + _gap} for d in _d}, {d: 16.0 for d in _d})
        _t, _ = _bs.table(_r, _v, "2026-11-02")
        _x = _bs.xsp(_t, _r)
        check(f"basis: XSP {_gap} above SPY over {_x['n']} sessions flags {_want} (mean gap {_x['mean']:.3f})",
              _x["n"] == 10 and abs(_x["mean"] - _gap) < 1e-9 and _x["flag"] is _want)

    # pending cover: 3 ok sessions then 4 no-rate (no VIX yet) sessions; the target-th session must land inside the planted range
    _d = list(pd.bdate_range("2026-11-02", periods=7).strftime("%Y-%m-%d"))
    _r, _v = _plant(_tmp + "/p", {d: {"SPY": 13.0} for d in _d}, {d: 16.0 for d in _d[:3]})
    _t, _f = _bs.table(_r, _v, "2026-11-02")
    _p = {k: _bs.project(_t, _f[-1], target=k) for k in (4, 7, 8)}
    check(f"basis: pending days cover the target: ok+1 -> first pending day, ok+pending -> last pending day, beyond -> after the range ({_p})",
          (_t.reason == "no-rate").sum() == 4 and _p[4] == _d[3] and _p[7] == _d[6] and _p[8] > _d[6] and _p[8] < "2026-12-31")
    # the reviewer's live case: 27 then 28 with pending covering both must be in order and before session 60
    _d = list(pd.bdate_range("2026-09-01", periods=33).strftime("%Y-%m-%d"))
    _r, _v = _plant(_tmp + "/q", {d: {"SPY": 13.0} for d in _d}, {d: 16.0 for d in _d[:25]})
    _t, _f = _bs.table(_r, _v, "2026-09-01")
    _a, _b, _c = (_bs.project(_t, _f[-1], target=k) for k in (27, 28, 60))
    check(f"basis: target 27 < 28 < 60 with pending covering 27 and 28 ({_a}, {_b}, {_c})",
          _d[0] <= _a < _b <= _d[-1] < _c)

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
