"""Planted-panel checks for prior.py: each one fails if a reported number would be wrong.

Run: ../.venv/bin/python3 check.py   (offline; the last check reads reports/untouched.json, so run prior.py first)
"""
import os

import numpy as np
import pandas as pd

import prior
from research.alpha.base import month_ends, positions
from research.features import Feature, FeatureStore, PeekError, Raw
from research.features.store import cross_section

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


# --- a planted panel: ten risers, one faller, one late joiner, one planted crossover ------------------
N = 760
cal = pd.bdate_range("2010-01-04", periods=N)
t = np.arange(N, dtype=float)
names = [f"U{i}" for i in range(10)] + ["D0", "L0"]
close = pd.DataFrame({f"U{i}": 100 + 0.1 * (i + 1) * t for i in range(10)}, index=cal)
close["U0"] = np.where(t < 400, 100.0, 100.0 + (t - 400))     # flat, then rising: the crossover is planted at 401
close["D0"] = 200 - 0.1 * t                                      # always below its 200-day average once warm
close["L0"] = 50 + 0.05 * t                                      # rising, but in the index only from JOIN
close = close[names]
JOIN = cal[500]
frames = {"close": close, "open": close * 0.999, "raw_close": close, "volume": pd.DataFrame(1e6, index=cal, columns=names)}
member_pit = pd.DataFrame(True, index=cal, columns=names)
member_pit.loc[:JOIN - pd.Timedelta(days=1), "L0"] = False
member_ref = pd.DataFrame(True, index=cal, columns=names)
raw = Raw(frames)
store = prior.features(member_pit, member_ref)
panel = store.build(raw, audit=True)
dates = month_ends(cal)
warm = dates[dates > cal[260]]                                   # 252 prints and the 200-day average both complete

check("the peek audit passes on the planted panel with every feature at its declared lag", True)
try:
    FeatureStore([Feature("peek", lambda r: r.frames["close"].shift(-1), lag=1)]).build(raw, audit=True)
    check("the audit in this pipeline catches a feature that reads tomorrow", False)
except PeekError:
    check("the audit in this pipeline catches a feature that reads tomorrow", True)

# --- the signal: 50/200 from momentum's code, one session stale through the store ---------------------
raw_ma = prior.mom.ma_signal(close, prior.MA_SHORT, prior.MA_LONG)
store_ma = panel["ma"]
raw_flip = raw_ma["U0"].eq(1).idxmax()
store_flip = store_ma["U0"].eq(1).idxmax()
check(f"the planted crossover flips the raw signal on {raw_flip.date()} and the store's on the next session "
      f"({store_flip.date()})", cal.get_loc(store_flip) == cal.get_loc(raw_flip) + 1 and raw_flip > cal[395])
check("every store 'ma' row equals momentum's ma_signal one session earlier",
      store_ma.iloc[1:].equals(raw_ma.shift(1).iloc[1:]))
check("the faller is never in an uptrend once warm; the risers always are",
      (store_ma.loc[warm[0]:, "D0"] == -1).all() and (store_ma.loc[warm[0]:, ["U1", "U5", "U9"]] == 1).all().all())

# --- the universe: point-in-time membership, eligibility, top-N ---------------------------------------
alpha = prior.XSTrend("member_pit", prior.TOP_N)
before_join, at_join = cal[cal.get_loc(JOIN) - 1], JOIN
check(f"the PIT filter excludes L0 on {before_join.date()}, the session before it joins, and includes it on "
      f"{at_join.date()}", not alpha.universe(cross_section(panel, before_join))["L0"]
      and alpha.universe(cross_section(panel, at_join))["L0"])
pos = positions(alpha, panel, warm)
check("L0 carries no weight at any rebalance before it joins and is held after",
      (pos.loc[:before_join, "L0"] == 0).all() and (pos.loc[at_join:, "L0"] > 0).all())
xs = cross_section(panel, warm[-1]).copy()
xs["dv63"] = xs["dv63"] * 0 + 1e6                                 # below the $10m floor: nobody is eligible
check("a cross-section below the dollar-volume floor holds nothing", (alpha.signal(xs) == 0).all())
xs = cross_section(panel, warm[-1]).copy()
xs.loc[["U3", "U4"], "dv63"] = 1e12
check("top-N keeps the N most liquid names and drops the rest",
      set(prior.XSTrend("member_pit", 2).universe(xs)[lambda s: s].index) == {"U3", "U4"})
xs = cross_section(panel, warm[-1]).copy()
xs.loc[[f"U{i}" for i in range(4)], "ma"] = -1                    # seven left in an uptrend, one short of MIN_NAMES
check(f"fewer than {prior.MIN_NAMES} names in an uptrend is cash, as published", (alpha.signal(xs) == 0).all())

# --- the weights: long only, 1/N over the uptrend names, never above 1 ------------------------------
check("long only: no weight is ever negative", (pos >= 0).all().all())
check("weights sum to at most 1 at every rebalance, and to exactly 1 whenever anything is held",
      (pos.sum(axis=1) <= 1 + 1e-12).all() and np.allclose(pos.sum(axis=1)[pos.sum(axis=1) > 0], 1.0))
late = warm[warm > cal[520]]                                     # every name eligible and a member from here
ref = prior.XSTrend("member_ref", None)
mine = positions(ref, panel, late)
published = prior.mom.long_only_weights(raw_ma, pd.DataFrame(True, index=cal, columns=names))
check("the Alpha's weights equal momentum.long_only_weights of the previous session at every late rebalance",
      np.allclose(mine.to_numpy(), published.shift(1).loc[late, names].to_numpy(), atol=1e-12))

# --- the engine: next-open fills, costs, the vol target --------------------------------------------
flat = pd.DataFrame(100.0, index=cal[:40], columns=["A", "B"])
one = pd.DataFrame(0.0, index=[cal[10]], columns=["A", "B"])
one.loc[cal[10], "A"] = 0.6
p = prior.path(one, flat, flat, bps=17.0)
check("a planted 0.6 trade at 17 bps costs exactly 0.6 x 17 bps on the fill day and nothing else",
      np.isclose(p["ret"].loc[cal[11]], -0.6 * 17e-4, atol=1e-15) and (p["ret"].drop(cal[11]) == 0).all()
      and p["traded"].loc[cal[11]] == 0.6)
adv = pd.DataFrame(1e6, index=cal[:40], columns=["A", "B"])
sig = pd.DataFrame(0.02, index=cal[:40], columns=["A", "B"])
p = prior.path(one, flat, flat, adv, sig)
expect = prior.COST_BPS + prior.COSTS.commission_bps + prior.COSTS.half_spread_bps \
    + 1e4 * prior.COSTS.impact_coef * 0.02 * np.sqrt(0.6 * prior.CAPITAL / 1e6)
check(f"the book's cost model on that trade is {expect:.3f} bps (10 + 5 + 2 + sqrt impact at $100k on $1m ADV)",
      np.isclose(p["ret"].loc[cal[11]], -0.6 * expect * 1e-4, atol=1e-15))
px = flat.copy()
px.loc[cal[11]:, "A"] = 102.0
px.loc[cal[12]:, "A"] = 103.0
op = flat.copy()
op.loc[cal[11], "A"] = 100.0                                      # fill at 100, close 102: the increment earns 2%
p = prior.path(one, px, op, bps=0.0)
check("the fill-day increment earns open-to-close and a held weight earns close-to-close",
      np.isclose(p["ret"].loc[cal[11]], 0.6 * 0.02) and np.isclose(p["ret"].loc[cal[12]], 0.6 * (103 / 102 - 1)))
gap = px.copy()
gap.loc[cal[20], "A"] = np.nan                                    # no print at t: the returns of t and t+1 are undefined
p_gap = prior.path(one, gap, op, bps=0.0)
check("a held name with no print earns zero on the days its return is undefined, and both days are counted",
      p_gap["missing_held"] == 2 and p_gap["ret"].loc[cal[20]] == 0 and p_gap["ret"].loc[cal[21]] == 0)
rng = np.random.default_rng(0)
hot = pd.Series(rng.normal(0, 0.30 / np.sqrt(252), 400), index=cal[:400])
cold = pd.Series(rng.normal(0, 0.10 / np.sqrt(252), 400), index=cal[:400])
s_hot, s_cold = prior.vol_scale(hot, cal[:400]), prior.vol_scale(cold, cal[:400])
check(f"the vol target halves a 30%-vol sleeve (scale {s_hot.iloc[100:].mean():.2f}) and never levers a 10%-vol one",
      abs(s_hot.iloc[100:].mean() - 0.5) < 0.05 and (s_cold.iloc[100:] == 1.0).all() and s_hot.max() <= 1.0)
check("no vol estimate yet means no position, not the cap", (s_hot.iloc[:prior.VOL_DAYS] == 0).all())
scaled = pos.mul(prior.vol_scale(hot.reindex(cal).fillna(0.0), warm), axis=0)
check("scored weights stay long only and sum to at most 1", (scaled >= 0).all().all() and (scaled.sum(axis=1) <= 1 + 1e-12).all())
rt = pd.Series(np.r_[np.zeros(5), -0.25, np.zeros(5), 0.4, -0.1, -0.12], index=cal[:14])  # -25%, new high, -20.8%
check("the drawdown-kill diagnostic dates the first crossing of each episode",
      prior.kill_dates(rt) == [cal[5].date(), cal[13].date()])

# --- the lock ---------------------------------------------------------------------------------------
lock = prior.Untouched(prior.LOCK)
body = lock.read() if os.path.exists(prior.LOCK) else {}
# Opened once on 2026-10-07 and refused (REFUSALS.md section 1); the lock must stay intact and say so.
check(f"the untouched window {body.get('start')} -> {body.get('end')} is intact (hash verifies), was opened once, "
      "and records the refusal: sleeve below the equal-weight basket",
      body.get("start") == "2021-01-04" and body.get("end") == prior.PANEL_END and lock.opened
      and body.get("result", {}).get("clears") is False
      and body["result"]["pit"]["sharpe"] < body["result"]["bh"]["sharpe"])

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if checks and all(checks) else 1)
