"""Offline checks of the three sleeves on synthetic data. Exits nonzero on any failure.

Run: ../../.venv/bin/python3 check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from framework.book.strategies import CryptoTrend, FXCarryETF, TrendETF, carry_weights, month_ends
from framework.engine import Bars, Config, CostModel, run, synthetic

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


NAMES = ("A", "B", "C", "D", "E", "F")
CODES = dict(zip(NAMES, ("EUR", "JPY", "GBP", "AUD", "CAD", "CHF")))
CLASSES = dict(zip(NAMES, ("equities", "equities", "bonds", "commodities", "fx", "fx")))
COSTS = Config(costs=CostModel(commission_bps=5, half_spread_bps=2))

base = synthetic(n_days=252 * 6, instruments=NAMES, seed=3)
cal = base.calendar
rng = np.random.default_rng(4)
rates = pd.DataFrame(rng.normal(0, 0.3, (len(cal), 7)).cumsum(axis=0) + 3.0, index=cal,
                     columns=[*CODES.values(), "USD"])
bars = Bars({f: base.field(f) for f in ("open", "high", "low", "close", "volume")},
            {c: rates[c] for c in rates.columns})
cut = cal[700]


def mutate(bars, rates):
    """Rewrite every price and rate after the cut."""
    frames = {f: bars.field(f).copy() for f in ("open", "high", "low", "close", "volume")}
    later = cal > cut
    for f in ("open", "high", "low", "close"):
        frames[f].loc[later] *= np.exp(rng.normal(0, 0.05, (later.sum(), len(NAMES))))
    r = rates.copy()
    r.loc[later] += rng.normal(0, 1.0, (later.sum(), r.shape[1]))
    return Bars(frames, {c: r[c] for c in r.columns})


mutated = mutate(bars, rates)
sleeves = {"TrendETF": TrendETF(classes=CLASSES), "CryptoTrend": CryptoTrend(instruments=["A", "B"]),
           "FXCarryETF": FXCarryETF(etfs=CODES)}
results = {}
for name, strat in sleeves.items():
    print(f"\n{name}\n")
    res = run(strat, bars, config=COSTS)
    results[name] = res
    tr = res.trades
    prev_is_month_end = tr.date.map(lambda d: bars.is_month_end(cal[cal.get_loc(d) - 1]))
    check("trades exist and every one is dated the bar after a month end",
          len(tr) > 0 and prev_is_month_end.all(), f"{len(tr)} trades")
    first = tr.iloc[0]
    check("the first fill is at that bar's open",
          np.isclose(first.fill, bars.open.loc[first.date, first.instrument] * (1 + 2 / 1e4 * np.sign(first.quantity))))
    mut = run(strat, mutated, config=COSTS)
    t0, t1 = tr[tr.date <= cut], mut.trades[mut.trades.date <= cut]
    check("mutation test: rewriting prices and rates after the cut leaves every trade up to the cut identical",
          len(t0) == len(t1) and np.allclose(t0.quantity, t1.quantity) and np.allclose(t0.fill, t1.fill),
          f"{len(t0)} trades compared")
    a, b = tr[tr.date > cut].quantity.head(30), mut.trades[mut.trades.date > cut].quantity.head(30)
    check("mutation test: trades after the cut did change", len(a) != len(b) or not np.allclose(a, b))
    check("equity up to the cut identical", np.allclose(res.equity.loc[:cut], mut.equity.loc[:cut]))

print("\nSLEEVE RULES\n")
t = month_ends(cal)[-3]
w = sleeves["TrendETF"].on_bar(t, bars.upto(t))
ends = month_ends(cal)
mpx = bars.close.loc[ends]
manual = np.sign(mpx.loc[ends[-4]] / mpx.loc[ends[-15]] - 1)
check("TrendETF: sign of each weight is the sign of P(t-1 month) / P(t-12 months) - 1",
      all(np.sign(w[n]) == manual[n] for n in NAMES))
check("TrendETF: nothing is emitted away from month end", sleeves["TrendETF"].on_bar(cal[701], bars.upto(cal[701])) is None)
mid = cal[cal.get_loc(month_ends(cal)[-5]) + 5]
ahead = pd.bdate_range(mid + pd.Timedelta(days=1), periods=40)
live = Bars({f: bars.field(f).loc[:mid].reindex(cal[:0].union(bars.field(f).loc[:mid].index).union(ahead))
             for f in ("open", "high", "low", "close", "volume")}, end=cal.get_loc(mid) + 1)
check("TrendETF: a live view whose newest bar is mid-month, with the calendar known ahead, emits nothing",
      live.asof == mid and sleeves["TrendETF"].on_bar(mid, live) is None)
check("CryptoTrend: never short", (results["CryptoTrend"].weights >= -1e-12).all().all())
r = rates.loc[t]
w = sleeves["FXCarryETF"].on_bar(t, bars.upto(t))
order = r.sort_values()
check("FXCarryETF: long the two highest rates, short the two lowest, USD as cash",
      all(w[n] == (0.5 if CODES[n] in order.index[-2:] else -0.5 if CODES[n] in order.index[:2] else 0.0)
          for n in NAMES))
check("carry_weights: fewer than 2 * n_leg quoted rates gives a flat book",
      (carry_weights(pd.Series({"EUR": 1.0, "JPY": np.nan, "USD": 2.0, "GBP": 3.0})) == 0).all())

print("\nEWMAC\n")
from framework.book.strategies import (EWMAC, FORECAST_CAP, buffered, combined_forecast, ewmac,
                                       forecast_scalar, robust_vol)
from framework.book.validate import attribution

trending = synthetic(n_days=252 * 6, instruments=("UP", "DOWN", "FLAT"), drift=(0.6, -0.6, 0.0), vol=0.15, seed=5)
fc = combined_forecast(trending.close)
tail = fc.iloc[-500:]
check("EWMAC: forecast is positive on a planted uptrend and negative on a planted downtrend on 90%+ of days",
      (tail["UP"] > 0).mean() > 0.9 and (tail["DOWN"] < 0).mean() > 0.9,
      f"up {(tail['UP'] > 0).mean():.2f}, down {(tail['DOWN'] < 0).mean():.2f}")
check("EWMAC: a view shorter than 500 days has no forecast, so nothing is backfilled",
      combined_forecast(trending.close.iloc[:499]).isna().all().all()
      and combined_forecast(trending.close.iloc[:600]).iloc[-1].notna().all())
raw = ewmac(base.close, robust_vol(base.close.diff()), 16, 64)
check("forecast scalar: scaling a raw forecast by 3 divides the scalar by 3 and the scaled |median| stays 10",
      np.isclose(forecast_scalar(raw * 3), forecast_scalar(raw) / 3)
      and np.isclose((raw * forecast_scalar(raw)).abs().median(axis=1).mean(), 10.0))
check("forecast scalar: fewer than 500 days gives NaN", np.isnan(forecast_scalar(raw.iloc[:400])))
check("forecast cap: every combined forecast is within +/-20 and the cap binds somewhere on a planted trend",
      fc.abs().max().max() <= FORECAST_CAP and (fc.abs() == FORECAST_CAP).any().any())
check("buffer: inside 10% of the target no trade, outside it moves to the band edge",
      buffered(1.0, 0.95) is None and buffered(1.0, 0.5) == 0.9 and buffered(1.0, 1.5) == 1.1
      and buffered(-1.0, 0.0) == -0.9)
ew = EWMAC(classes={"UP": "a", "DOWN": "b", "FLAT": "c"})
res_ew = run(ew, trending, config=COSTS)
w_last = res_ew.weights.iloc[-1]
check("EWMAC through the engine: long the uptrend, short the downtrend at the end of the sample",
      w_last["UP"] > 0 and w_last["DOWN"] < 0, f"{w_last.round(3).to_dict()}")
later = np.where(np.asarray(trending.calendar > cut)[:, None], 1.1, 1.0)
mut_ew = run(EWMAC(classes={"UP": "a", "DOWN": "b", "FLAT": "c"}),
             Bars({f: trending.field(f) * (later if f != "volume" else 1.0)
                   for f in ("open", "high", "low", "close", "volume")}), config=COSTS)
t0, t1 = res_ew.trades[res_ew.trades.date <= cut], mut_ew.trades[mut_ew.trades.date <= cut]
a0, a1 = res_ew.trades[res_ew.trades.date > cut].quantity.head(30), mut_ew.trades[mut_ew.trades.date > cut].quantity.head(30)
check("EWMAC mutation test: rewriting prices after the cut leaves every trade up to the cut identical and changes later ones",
      len(t0) == len(t1) and np.allclose(t0.quantity, t1.quantity) and (len(a0) != len(a1) or not np.allclose(a0, a1)),
      f"{len(t0)} trades compared")

bench = synthetic(n_days=252 * 8, instruments=("X",), vol=0.15, seed=7)
x = bench.close["X"].pct_change().fillna(0.0)
planted = 0.0003 + 0.7 * x + rng.normal(0, 0.003, len(x))
a = attribution(pd.Series(planted, index=x.index), bench, ["X"])["1/N"]
check("attribution: regression recovers a planted alpha of 7.6% a year and beta of 0.7",
      abs(a["alpha"] - 0.0003 * 252) < 0.02 and abs(a["beta"] - 0.7) < 0.05,
      f"alpha {a['alpha']:.3f} t {a['t_alpha']:.1f} beta {a['beta']:.3f}")
b = attribution(pd.Series(0.3 * x.to_numpy() + rng.normal(0, 0.001, len(x)), index=x.index), bench, ["X"])["1/N"]
check("attribution: pure beta scores near-zero alpha and a residual Sharpe near zero",
      abs(b["alpha"]) < 0.01 and abs(b["residual_sharpe"]) < 0.3, f"alpha {b['alpha']:.3f} resid {b['residual_sharpe']:.2f}")

print("\nBROKER AND DAEMON\n")
import json
import tempfile

import framework.book.broker as broker_mod
from framework.book.broker import AlpacaBroker, ShadowBroker
from framework.book.daemon import COLUMNS, append_ledger, last_session
from framework.book.universe import sessions

for k in ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET"):
    os.environ.pop(k, None)
broker_mod.ENV = os.path.join(tempfile.mkdtemp(), "missing.env")
try:
    AlpacaBroker()
    check("AlpacaBroker refuses without keys", False)
except RuntimeError as e:
    check("AlpacaBroker refuses without keys and points at the signup page",
          "https://app.alpaca.markets/signup" in str(e))

dummy = AlpacaBroker("dummy-key", "dummy-secret")  # no request is made until an order or account call
check("AlpacaBroker with keys passes the paper host check and is on the paper endpoint",
      dummy.client._sandbox and getattr(dummy.client._base_url, "value", "") == f"https://{broker_mod.PAPER_HOST}")

rc = object.__new__(AlpacaBroker).reconcile
prices = {"SPY": 500.0, "FXE": 100.0, "BTC/USD": 50000.0, "GLD": 300.0}
orders = rc({"SPY": 0.5, "FXE": -0.1, "BTC/USD": 0.1}, 100_000, prices,
            held={"SPY": (50.0, 25000.0), "GLD": (10.0, 3000.0)})
got = [(o.symbol, o.side.value, o.qty, o.notional, o.time_in_force.value) for o in orders]
check("reconcile: buy the long by notional, short whole shares, close what is not a target by quantity, crypto GTC",
      got == [("BTCUSD", "buy", None, 10000.0, "gtc"), ("FXE", "sell", 100, None, "day"),
              ("GLD", "sell", 10.0, None, "day"), ("SPY", "buy", None, 25000.0, "day")], str(got))
trim = [(o.side.value, o.qty, o.notional) for o in rc({"SPY": 0.2}, 100_000, prices, held={"SPY": (50.0, 25000.0)})]
check("reconcile: a partial trim sells by notional", trim == [("sell", None, 5000.0)], str(trim))
flat = [(o.symbol, o.side.value, o.qty) for o in rc({"BTC/USD": -0.1}, 100_000, prices, held={"BTC/USD": (0.2, 10000.0)})]
check("reconcile: a short target on a held coin sells it to flat", flat == [("BTCUSD", "sell", 0.2)], str(flat))
check("reconcile: a position inside the 10% buffer is left alone",
      rc({"SPY": 0.5}, 100_000, prices, held={"SPY": (95.0, 47500.0)}) == [])
check("reconcile: a move under $25 is skipped", rc({"SPY": 0.0001}, 100_000, prices, held={}) == [])
flip = [(o.side.value, o.qty, o.notional) for o in rc({"FXE": 0.05}, 100_000, prices, held={"FXE": (-100.0, -10000.0)})]
check("reconcile: a sign flip closes first, then opens", flip == [("buy", 100.0, None), ("buy", None, 5000.0)], str(flip))
check("reconcile: crypto is never shorted", rc({"BTC/USD": -0.1}, 100_000, prices, held={}) == [])

path = os.path.join(tempfile.mkdtemp(), "ledger.csv")
row = dict(zip(COLUMNS, ["2026-09-02", "shadow", 100000.0, 0.5, "{}", "[]", "{}", "abc", "t", ""]))
append_ledger([row], path)
append_ledger([row], path)
n = append_ledger([dict(row, date="2026-09-03")], path)
check("ledger append is idempotent per (date, book)", len(pd.read_csv(path)) == 2 and n == 1)

from framework.book.universe import panel_hash

h_fixed = panel_hash(bars, cal[100], cal[400])
check("panel hash over a fixed window does not move when bars are appended, does when the window changes",
      panel_hash(bars.upto(cal[500]), cal[100], cal[400]) == h_fixed and panel_hash(bars, cal[100], cal[401]) != h_fixed)
check("NYSE calendar: Good Friday and observed Independence Day 2026 are not sessions",
      not sessions("2026-04-01", "2026-07-10").isin([pd.Timestamp("2026-04-03"), pd.Timestamp("2026-07-03")]).any())
ls = lambda s: last_session(pd.Timestamp(s, tz="America/New_York")).date().isoformat()
check("last session: before the close it is yesterday, after the close today, Saturday and Labor Day roll back",
      (ls("2026-09-04 10:00"), ls("2026-09-04 16:30"), ls("2026-09-05 10:00"), ls("2026-09-07 17:00"))
      == ("2026-09-03", "2026-09-04", "2026-09-04", "2026-09-04"))

me = month_ends(cal)[-4]
shadow = ShadowBroker([TrendETF(classes=CLASSES)], {"TrendETF": 1.0}, month_ends(cal)[-8])
shadow.run(bars.upto(me))
pend = shadow.results.books["TrendETF"]["pending"]
check("shadow targets on a month end are the sleeve's fresh targets",
      pend and shadow.targets() == {k: v for k, v in pend.items() if abs(v) > 1e-9})
shadow.run(bars.upto(cal[cal.get_loc(me) + 3]))
check("shadow targets on a quiet day are what the book holds",
      shadow.results.books["TrendETF"]["pending"] is None and shadow.targets() == shadow.positions())

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
