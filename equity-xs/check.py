"""Offline checks on synthetic panels with planted cross-sectional effects.

Two fake universes. In the first, next month's excess return is a U in
trailing momentum: both tails go up, the middle goes down. A linear model
cannot see it and a tree can, so LightGBM must recover it and ridge and
plain 12-1 momentum must not. In the second the drift is persistent, so
all three must find momentum. Membership, ranking, the decile book and the
cost model are checked by hand. Exits 1 on any failure.

Run: ../.venv/bin/python3 check.py
"""
import os
import tempfile

import numpy as np
import pandas as pd

from xs import (DECILE, MACRO, XSModel, decile_spread, eligible, features, framework_net, ic_summary,
                importance_stability, load_raw, membership, rank_ic, ranks, scores)
from framework.book.strategies import CAPITAL, book_config
from research.alpha import backtest, forward_returns, positions, walk_forward
from research.alpha.base import month_ends
from research.features import FeatureStore, PeekError, Raw
from research.features.store import cross_section, long_panel

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


rng = np.random.default_rng(5)
N, T = 100, 2000
IDX = pd.bdate_range("2012-01-02", periods=T)
NAMES = [f"S{i:02d}" for i in range(N)]


def simulate(kind, noise=0.012, beta=0.006):
    """Daily returns whose drift is a planted function of each stock's own trailing 12-1 momentum."""
    rets = np.zeros((T, N))
    drift = rng.normal(0, 0.0006, N)
    px = np.full(N, 50.0)
    hist = [px.copy()]
    for t in range(T):
        if kind == "ushape" and t > 252:
            mom = hist[-21] / hist[-252] - 1
            r = pd.Series(mom).rank(pct=True).to_numpy()
            s = np.abs(r - 0.5) - 0.25
        elif kind == "momentum":
            drift = 0.995 * drift + rng.normal(0, 0.00008, N)
            s = drift / 0.004
        else:
            s = np.zeros(N)
        rets[t] = beta * s + rng.normal(0, noise, N)
        px = px * np.exp(rets[t])
        hist.append(px.copy())
    close = pd.DataFrame(np.array(hist[1:]), index=IDX, columns=NAMES)
    open_ = close.shift(1).fillna(50.0) * np.exp(rng.normal(0, 0.002, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    vol = pd.DataFrame(rng.lognormal(13, 0.3, (T, N)), index=IDX, columns=NAMES)
    series = {k: pd.Series(20 + rng.normal(0, 1, T).cumsum() * 0.1, index=IDX) for k in MACRO}
    return Raw({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, series)


INTERVALS = pd.DataFrame({"ticker": NAMES, "start": [IDX[0]] * N, "end": [IDX[-1]] * N})
INTERVALS.loc[INTERVALS["ticker"] == "S98", "start"] = IDX[900]
INTERVALS = INTERVALS[INTERVALS["ticker"] != "S99"]
STORE = FeatureStore(features(INTERVALS))
DATES = month_ends(IDX)


def oos_ic(raw, model, split=1400, **kw):
    panel = STORE.build(raw)
    X, y = long_panel(panel), long_panel(pd.concat({"y": forward_returns(raw.frames["close"], 21)}, axis=1,
                                                   names=["feature", "instrument"]))["y"]
    train = IDX[:split]
    a = XSModel(model, dates=DATES, **kw).fit(X.loc[train], y.loc[train])
    test = DATES[(DATES > IDX[split + 21]) & (DATES < IDX[-22])]
    sc = scores(a, panel, test)
    fwd = forward_returns(raw.frames["close"], 21)
    return a, panel, ic_summary(rank_ic(sc, fwd)), decile_spread(sc, fwd).mean()


print("FEATURE STORE AND MEMBERSHIP\n")

raw_u = simulate("ushape")
try:
    panel_u = STORE.build(raw_u, audit=True)
    check("the store's peek audit passes with the lag-0 membership feature attached", True)
except PeekError as e:
    panel_u = STORE.build(raw_u)
    check("the store's peek audit passes with the lag-0 membership feature attached", False, str(e))
m = panel_u["member"]
check("a name is a member only inside its spell and a name with no spell never is",
      m["S98"].loc[:IDX[899]].eq(0).all() and m["S98"].loc[IDX[900]:].eq(1).all() and m["S99"].eq(0).all()
      and m["S00"].eq(1).all())
check("membership frame equals the feature the store built", membership(INTERVALS, IDX, NAMES).equals(m))
m_k = membership(INTERVALS, IDX, NAMES, known_from=IDX[300])
panel_k = FeatureStore(features(INTERVALS, IDX[300])).build(raw_u, audit=True)
check("before known_from membership is NaN for every name, not 0, so nothing there is eligible or a non-member",
      m_k.loc[:IDX[299]].isna().all().all() and m_k.loc[IDX[300]:].equals(m.loc[IDX[300]:])
      and not eligible(cross_section(panel_k, IDX[299])).any() and eligible(cross_section(panel_k, IDX[300])).sum() == 98
      and panel_k["member"].equals(m_k))
try:
    FeatureStore(features(INTERVALS) + [type(features(INTERVALS)[0])("peek", lambda r: r.frames["close"].shift(-1), 1)]
                 ).build(raw_u, audit=True)
    check("the audit still catches a peeking feature next to the membership feature", False)
except PeekError:
    check("the audit still catches a peeking feature next to the membership feature", True)

X = long_panel(panel_u)
R = ranks(X)
d = DATES[20]
by_date = ranks(cross_section(panel_u, d))
check("ranks on the long panel equal ranks on the cross-section at every date",
      np.allclose(R.loc[d].sort_index().to_numpy(), by_date.sort_index().to_numpy(), equal_nan=True))
check("ranks are centred percentiles: top name at +0.5, bottom near -0.5, NaN kept as NaN",
      by_date["mom_12_1"].max() == 0.5 and abs(by_date["mom_12_1"].min() + 0.5 - 1 / by_date["mom_12_1"].count()) < 1e-12
      and R["mom_12_1"].isna().sum() == X["mom_12_1"].isna().sum())
el = eligible(cross_section(panel_u, IDX[10]))
check("a name without 21 days of history is not eligible", not el.any())

print("\nPLANTED U-SHAPE: TREES SEE IT, LINES DO NOT\n")

lg, panel, ic_lg, sp_lg = oos_ic(raw_u, "lgbm")
_, _, ic_rd, sp_rd = oos_ic(raw_u, "ridge")
_, _, ic_mo, sp_mo = oos_ic(raw_u, "mom")
check("LightGBM's out-of-sample rank IC recovers the planted U",
      ic_lg["mean"] > 0.15 and ic_lg["t"] > 3, f"IC {ic_lg['mean']:.3f}, t {ic_lg['t']:.1f}")
check("ridge, a line through a U, sees nothing", abs(ic_rd["mean"]) < 0.08, f"IC {ic_rd['mean']:.3f}")
check("12-1 momentum sees nothing either", abs(ic_mo["mean"]) < 0.08, f"IC {ic_mo['mean']:.3f}")
check("LightGBM's decile spread is positive and larger than momentum's",
      sp_lg > 0.005 and sp_lg > sp_mo, f"{100 * sp_lg:.2f}%/mo vs {100 * sp_mo:.2f}%/mo")
check("gain importance puts mom_12_1 first on this panel",
      lg.importance["lgbm"].idxmax() == "mom_12_1", lg.importance["lgbm"].idxmax())
last_me = DATES[DATES <= IDX[1399]][-1]
check("a training slice that ends mid-month trains on the calendar's month ends only, no partial last cross-section",
      IDX[1399] not in DATES and lg.train_dates.isin(DATES).all() and lg.train_dates[-1] == last_me,
      f"slice ends {IDX[1399].date()}, last training cross-section {lg.train_dates[-1].date()}")
try:
    XSModel("ridge").fit(X.loc[IDX[:1400]], long_panel(pd.concat({"y": forward_returns(raw_u.frames["close"], 21)},
                                                                    axis=1, names=["feature", "instrument"]))["y"])
    check("fit without the calendar's rebalance dates refuses rather than inventing month ends", False)
except ValueError:
    check("fit without the calendar's rebalance dates refuses rather than inventing month ends", True)

print("\nPLANTED MOMENTUM: EVERY MODEL FINDS IT\n")

raw_m = simulate("momentum")
_, _, ic_lg2, _ = oos_ic(raw_m, "lgbm")
_, _, ic_rd2, _ = oos_ic(raw_m, "ridge")
_, _, ic_mo2, _ = oos_ic(raw_m, "mom")
check("all three score a positive out-of-sample IC on persistent drift",
      min(ic_lg2["mean"], ic_rd2["mean"], ic_mo2["mean"]) > 0.05,
      f"lgbm {ic_lg2['mean']:.3f}, ridge {ic_rd2['mean']:.3f}, mom {ic_mo2['mean']:.3f}")
_, _, ic_nm, _ = oos_ic(raw_m, "lgbm", macro=False)
check("switching the macro series off still finds it", ic_nm["mean"] > 0.05, f"IC {ic_nm['mean']:.3f}")

X_m = long_panel(STORE.build(raw_m))
y_m = long_panel(pd.concat({"y": forward_returns(raw_m.frames["close"], 21)}, axis=1,
                           names=["feature", "instrument"]))["y"]
shift = pd.Series(rng.normal(0, 0.05, T), index=IDX).reindex(y_m.index.get_level_values(0)).to_numpy()
a1 = XSModel("ensemble", dates=DATES).fit(X_m.loc[IDX[:1400]], y_m.loc[IDX[:1400]])
a2 = XSModel("ensemble", dates=DATES).fit(X_m.loc[IDX[:1400]], (y_m + shift).loc[IDX[:1400]])
xs = cross_section(STORE.build(raw_m), DATES[55])
check("a per-date shift of the label (the market) changes no prediction: the label is cross-sectionally demeaned",
      np.allclose(a1.score(xs).to_numpy(), a2.score(xs).to_numpy(), equal_nan=True))

print("\nTHE PURGE ON A CALENDAR WITH HOLIDAYS\n")

hol = IDX[::9]
raw_h = Raw({k: v.drop(hol) for k, v in raw_u.frames.items()}, {k: v.drop(hol) for k, v in raw_u.series.items()})
cal_h = raw_h.frames["close"].index
tab_h, _, _ = walk_forward(lambda **p: XSModel(dates=month_ends(cal_h), **p), [{"model": "mom"}],
                           STORE.build(raw_h), raw_h.frames["close"], n_splits=3, test_size=200)
gaps = [cal_h.get_loc(pd.Timestamp(r.test_start)) - cal_h.get_loc(pd.Timestamp(r.train_end)) for r in tab_h.itertuples()]
check("every fold's training rows end at least 21 sessions (not business days) before its test window",
      min(gaps) >= 21, f"session gaps {gaps}")

print("\nTHE DECILE BOOK\n")

w = lg.signal(cross_section(panel_u, DATES[-3]))
n_el = eligible(cross_section(panel_u, DATES[-3])).sum()
k = int(n_el * DECILE)
check("long top decile, short bottom decile, equal weight: dollar neutral at 200% gross",
      abs(w.sum()) < 1e-12 and abs(w.abs().sum() - 2.0) < 1e-12)
check("exactly int(n * 0.1) names on each side", (w > 0).sum() == k and (w < 0).sum() == k, f"{k} a side")
check("non-members get zero weight even when scored history exists", w["S99"] == 0.0)
w_late = positions(lg, panel_u, DATES[DATES < IDX[900]])
check("S98 is never held before it joins", (w_late["S98"] == 0).all())
check("a cross-section with fewer than 20 eligible names sends no positions",
      lg.signal(cross_section(panel_u, DATES[-3]).iloc[:15]).abs().sum() == 0)

print("\nTHE COST MODEL\n")

cfg = book_config()
close2 = pd.DataFrame({"A": [100.0, 101.0, 102.0, 101.0, 103.0, 102.0], "B": [50.0, 50.5, 50.0, 49.0, 49.5, 50.0]},
                      index=pd.bdate_range("2020-01-01", periods=6))
pos2 = pd.DataFrame({"A": [0.0, 0.5, 0.5, 0.0, 0.0, 0.0], "B": [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]}, index=close2.index)
adv2 = pd.DataFrame(2e5, index=close2.index, columns=["A", "B"])
sig2 = pd.DataFrame(0.02, index=close2.index, columns=["A", "B"])
net2 = framework_net(pos2, close2, adv2, sig2)
gross2 = backtest(pos2, close2, cost_bps=0.0)
part = 0.5 * CAPITAL / 2e5
bps = cfg.costs.commission_bps + cfg.costs.slippage_bps(part, 0.02)
day2 = gross2.iloc[2] - 2 * 0.5 * bps / 1e4 - 0.5 * cfg.borrow_bps / 1e4 / 252
check("the day after the trade: net = gross minus commission + spread + sqrt impact on both legs minus a day of borrow",
      abs(net2.iloc[2] - day2) < 1e-14, f"bps per leg {bps:.2f}")
check("impact uses the book's participation: doubling capital raises the cost",
      framework_net(pos2, close2, adv2, sig2, capital=2 * CAPITAL).iloc[2] < net2.iloc[2])
check("no trade and no short: net equals gross", abs(net2.iloc[5] - gross2.iloc[5]) < 1e-15 and net2.iloc[5] == 0.0)
check("cost_scale=2 doubles the cost gap", abs((gross2.iloc[2] - framework_net(pos2, close2, adv2, sig2, cost_scale=2).iloc[2])
                                                - 2 * (gross2.iloc[2] - net2.iloc[2])) < 1e-14)

print("\nDIAGNOSTICS\n")

fwd = forward_returns(close2, 1)
sc = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[close2.index[0]])
check("rank IC of a perfectly ordered score is 1", rank_ic(pd.DataFrame(np.tile(np.arange(30.0), (1, 1)),
      index=[IDX[0]], columns=NAMES[:30]), pd.DataFrame(np.tile(np.arange(30.0), (1, 1)), index=[IDX[0]],
      columns=NAMES[:30])).iloc[0] == 1.0)
sc30 = pd.DataFrame([np.arange(30.0)], index=[IDX[0]], columns=NAMES[:30])
check("decile spread is top-decile mean minus bottom-decile mean", decile_spread(sc30, sc30).iloc[0] == (28.0 - 1.0))
same = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=list("abc"), dtype=float)
flip = pd.DataFrame([[1, 2, 3], [3, 2, 1]], columns=list("abc"), dtype=float)
check("importance stability is 1 for identical folds and -1 for reversed ones",
      importance_stability(same) == 1.0 and importance_stability(flip) == -1.0)


class StubLake:
    def load(self, dataset, start, end, universe=None):
        if dataset == "equities_daily":
            rows = []
            for i, d in enumerate(IDX[:5]):
                rows.append({"date": d, "ticker": "A", "open": 10.0, "high": 12.0, "low": 9.0, "close": 10.0,
                             "adj_close": 5.0, "volume": 100})
            return pd.DataFrame(rows)
        return pd.DataFrame({"date": list(IDX[:5]) * 3, "series": sorted(list(MACRO) * 5), "value": 1.0})


r = load_raw(StubLake(), None, None)
check("load_raw uses the adjusted close and scales the open, high and low by the same factor",
      r.frames["close"].iloc[0, 0] == 5.0 and r.frames["high"].iloc[0, 0] == 6.0 and r.frames["low"].iloc[0, 0] == 4.5
      and set(r.series) == set(MACRO))
snap = tempfile.mkdtemp()
r1 = load_raw(StubLake(), None, None, snapshot=snap)
r2 = load_raw(None, None, None, snapshot=snap)
check("with a snapshot directory the second load never touches the lake and returns the same frames",
      all(r1.frames[k].equals(r2.frames[k]) for k in r1.frames) and all(r1.series[k].equals(r2.series[k]) for k in r1.series)
      and sorted(os.listdir(snap)) == ["equities_daily.parquet", "fred.parquet"])

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
