"""Offline checks of the three sleeves on synthetic data. Exits nonzero on any failure.

Run: ../../.venv/bin/python3 check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from framework.book.strategies import (BOOKS, CryptoTrend, ETFBeta, FXCarryETF, TrendETF, carry_weights,
                                       month_ends)
from framework.book.strategies import sleeves as book_sleeves
from framework.engine import BUFFER, Bars, Config, CostModel, OrderRules, RiskConfig, run, synthetic

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
from framework.book.strategies import EWMAC, FORECAST_CAP, combined_forecast, ewmac, forecast_scalar, robust_vol
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
ew = EWMAC(classes={"UP": "a", "DOWN": "b", "FLAT": "c"})
BUFFERED = Config(costs=COSTS.costs, risk=RiskConfig(buffer=BUFFER))
res_ew = run(ew, trending, config=BUFFERED)
w_last = res_ew.weights.iloc[-1]
check("EWMAC through the engine: long the uptrend, short the downtrend at the end of the sample",
      w_last["UP"] > 0 and w_last["DOWN"] < 0, f"{w_last.round(3).to_dict()}")
live_days = int(combined_forecast(trending.close).notna().any(axis=1).sum())
check("EWMAC sends a target every live day and the engine's buffer trades under a third of those instrument-days",
      ew.on_bar(trending.asof, trending) is not None and len(res_ew.trades) < live_days,
      f"{len(res_ew.trades)} trades on {live_days * 3} instrument-days")
later = np.where(np.asarray(trending.calendar > cut)[:, None], 1.1, 1.0)
mut_ew = run(EWMAC(classes={"UP": "a", "DOWN": "b", "FLAT": "c"}),
             Bars({f: trending.field(f) * (later if f != "volume" else 1.0)
                   for f in ("open", "high", "low", "close", "volume")}), config=BUFFERED)
t0, t1 = res_ew.trades[res_ew.trades.date <= cut], mut_ew.trades[mut_ew.trades.date <= cut]
a0, a1 = res_ew.trades[res_ew.trades.date > cut].quantity.head(30), mut_ew.trades[mut_ew.trades.date > cut].quantity.head(30)
check("EWMAC mutation test: rewriting prices after the cut leaves every trade up to the cut identical and changes later ones",
      len(t0) == len(t1) and np.allclose(t0.quantity, t1.quantity) and (len(a0) != len(a1) or not np.allclose(a0, a1)),
      f"{len(t0)} trades compared")

beta = ETFBeta(instruments=["UP", "DOWN", "FLAT", "NONE"]).on_bar(trending.asof, trending)
check("ETFBeta: 1/N of the instruments with a price, nothing for one without",
      beta == {"UP": 1 / 3, "DOWN": 1 / 3, "FLAT": 1 / 3})
check("every candidate book builds distinct sleeves",
      all(len({str(s) for s in book_sleeves(b)}) == len(BOOKS[b]) for b in BOOKS)
      and [str(s) for s in book_sleeves()] == ["TrendETF", "FXCarryETF", "CryptoTrend"])

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
from framework.book.broker import (AlpacaBroker, AlpacaLive, Halted, IBKRBroker, KalshiLive, Limits, LiveDisabled,
                                   ShadowBroker, SimulatedAccount, check_limits, expected_weights, filled_value,
                                   live_token, mode_change, order_outcome, order_record, unfilled,
                                   verify_positions)
import framework.book.daemon as daemon_mod
from framework.book.daemon import COLUMNS, append_ledger, last_session, previous_row, sim_rows
from framework.book.strategies import CAPITAL
from framework.book.universe import sessions

for k in ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET", "LIVE_ENABLED"):
    os.environ.pop(k, None)
TMP = tempfile.mkdtemp()
broker_mod.ENV = os.path.join(TMP, "missing.env")
broker_mod.KILL = os.path.join(TMP, "KILL")
sim = AlpacaBroker(bars=bars.upto(cal[-2]))
check("AlpacaBroker without keys runs the clearly labelled SIMULATED account at CAPITAL and points at the signup page",
      sim.mode == "simulated" and "https://app.alpaca.markets/signup" in sim.why and sim.equity() == CAPITAL
      and sim.positions() == {})

dummy = AlpacaBroker("dummy-key", "dummy-secret")  # no request is made until an order or account call
check("AlpacaBroker with keys is in paper mode, passes the host check and is on the paper endpoint",
      dummy.mode == "paper" and dummy.client._sandbox
      and getattr(dummy.client._base_url, "value", "") == f"https://{broker_mod.PAPER_HOST}")

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
lim = rc({"SPY": 0.5}, 100_000, prices, held={}, limit_prices={"SPY": 495.0})
check("reconcile: a name with a limit price routes as a limit order, qty = notional / limit, market elsewhere",
      type(lim[0]).__name__ == "LimitOrderRequest" and lim[0].limit_price == 495.0
      and np.isclose(float(lim[0].qty), 50_000 / 495.0)
      and type(rc({"SPY": 0.5}, 100_000, prices, held={})[0]).__name__ == "MarketOrderRequest")

print("\nONE ORDER MODEL, TWO PATHS\n")
from framework.book.broker import book_symbol, build_orders
from framework.book.strategies import LIVE_RULES
from framework.engine.execution import Executor
from framework.engine.portfolio import Portfolio


class OneBar:
    """The smallest bars view Executor needs: one fill price per name, no volume."""
    volume = None

    def __init__(self, prices, date):
        self.frame = pd.DataFrame([prices], index=[date])

    def field(self, _name):
        return self.frame


def backtest_end(weights, equity, prices, held, rules=LIVE_RULES):
    """Where the BACKTEST path lands: Executor.execute, zero costs so only quantities matter."""
    day = pd.Timestamp("2026-09-09")
    book = Portfolio(equity - sum(q * prices[n] for n, q in held.items()))
    book.positions, book.marks = dict(held), dict(prices)
    Executor(rules=rules).execute(day, OneBar(prices, day), weights, book)
    return {n: q for n, q in book.positions.items() if abs(q) > 1e-9}


def live_end(weights, equity, prices, held):
    """Where the LIVE path lands: build_orders, its requests replayed back onto `held`."""
    out = dict(held)
    for o in build_orders(weights, equity, prices, {n: (q, q * prices[n]) for n, q in held.items()}):
        name = book_symbol(o.symbol)
        dq = float(o.qty) if o.qty is not None else float(o.notional) / prices[name]
        out[name] = out.get(name, 0.0) + (dq if o.side.value == "buy" else -dq)
    return {n: q for n, q in out.items() if abs(q) > 1e-9}


def same_book(weights, equity, prices, held, tol=0.02):
    """True when both paths hold the same dollars of every name (notional rounds to cents)."""
    a, b = backtest_end(weights, equity, prices, held), live_end(weights, equity, prices, held)
    return all(abs(a.get(n, 0.0) - b.get(n, 0.0)) * prices[n] <= tol for n in set(a) | set(b)), a, b


PX = {"SPY": 500.0, "FXE": 100.0, "GLD": 300.0, "BTC/USD": 50000.0, "IEF": 95.0,
      "TLT": 90.0, "QQQ": 400.0, "EFA": 80.0, "VNQ": 33.33}
HELD = {"SPY": 50.0, "GLD": 10.0, "BTC/USD": 0.2, "IEF": 95.0, "TLT": -100.0, "QQQ": 0.05, "EFA": 1.0}
W = {"SPY": 0.5, "FXE": -0.1, "GLD": 0.0, "BTC/USD": -0.1, "IEF": 0.09, "TLT": 0.05,
     "QQQ": 0.0001, "EFA": 0.0001, "VNQ": 0.02}
ok, a, b = same_book(W, 100_000.0, PX, HELD)
check("one order model: backtest and live reach the same book for one set of weights "
      "(buy, whole-share short, flatten, unshortable crypto, buffer, sign flip, sub-$25)",
      ok, f"backtest {sorted(a)} vs live {sorted(b)}")
check("that set of weights exercises every rule: a buffered hold, a flatten, a whole-share short, "
      "a flat coin and an untouched sub-$25 move",
      np.isclose(a["IEF"], 95.0) and "GLD" not in a and "BTC/USD" not in a
      and a["FXE"] == -100.0 and np.isclose(a["QQQ"], 0.05) and "EFA" not in a, str(a))
unshaped = backtest_end(W, 100_000.0, PX, HELD, rules=OrderRules())
check("closing the fork moved the backtest: with the rules off the engine reaches a book the broker cannot",
      unshaped["BTC/USD"] < 0 and "BTC/USD" not in a          # a short the broker cannot place
      and not np.isclose(unshaped["IEF"], a["IEF"])            # a move the buffer stops
      and "EFA" in unshaped and "EFA" not in a,                # a position under the $25 minimum
      f"unshaped {({k: round(v, 3) for k, v in unshaped.items()})}")

rng_fuzz = np.random.default_rng(20260909)
names = list(PX)
bad = 0
for _ in range(300):
    px = {n: float(rng_fuzz.uniform(5, 600)) if n != "BTC/USD" else float(rng_fuzz.uniform(2e4, 8e4)) for n in names}
    eq = float(rng_fuzz.uniform(2e4, 5e5))
    w = {n: float(rng_fuzz.choice([0.0, 1.0], p=[0.25, 0.75]) * rng_fuzz.uniform(-0.3, 0.3)) for n in names}
    h = {n: float(rng_fuzz.choice([0.0, 1.0], p=[0.4, 0.6]) * rng_fuzz.uniform(-0.3, 0.3)) * eq / px[n] for n in names}
    h = {n: q for n, q in h.items() if abs(q) > 1e-9}
    if not same_book(w, eq, px, h)[0]:
        bad += 1
check("one order model: 300 random (weights, equity, prices, held) draws land both paths on the same book",
      bad == 0, f"{bad} disagreements")


path = os.path.join(tempfile.mkdtemp(), "ledger.csv")
row = dict(zip(COLUMNS, ["2026-09-02", "shadow", 100000.0, 0.5, "{}", "[]", "{}", "abc", "t", "", None, ""]))
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
held = shadow.results.books["TrendETF"]["weights"].iloc[-1].to_dict()
check("shadow targets on a month end are the sleeve's fresh targets over what the buffer left alone",
      pend and len(pend) < len(held) and shadow.targets() == {k: v for k, v in {**held, **pend}.items() if abs(v) > 1e-9},
      f"{len(pend)} sent, {len(held)} held")
shadow.run(bars.upto(cal[cal.get_loc(me) + 3]))
check("shadow targets on a quiet day are what the book holds",
      shadow.results.books["TrendETF"]["pending"] is None and shadow.targets() == shadow.positions())


# ---------------------------------------------------------------- report.py on synthetic ledgers
import json
import sqlite3
import tempfile

from framework.book import report

tmp = tempfile.mkdtemp()
NOW = pd.Timestamp("2026-09-05 07:30", tz="America/Chicago")


def ledger_rows(equities, alpaca=None, hashes=None):
    """Synthetic ledger.csv: one shadow row per session, optional alpaca row on the last."""
    days = [d.strftime("%Y-%m-%d") for d in sessions("2026-08-31", "2026-09-04")[:len(equities)]]
    held = {"SPY": 0.15, "TLT": -0.07}
    rows = []
    for i, (d, e) in enumerate(zip(days, equities)):
        rows.append({"date": d, "book": "shadow", "equity": e, "gross": 0.22, "positions": json.dumps(held),
                     "fills": "[]", "targets": json.dumps(held), "panel_hash": (hashes or ["h"] * len(days))[i],
                     "run_at": f"{d}T18:49:00-04:00", "note": "synthetic"})
    if alpaca is not None:
        rows.append({**rows[-1], "book": "alpaca", "positions": json.dumps(alpaca), "note": "submitted"})
    path = os.path.join(tmp, f"ledger{len(os.listdir(tmp))}.csv")
    pd.DataFrame(rows, columns=report.read_ledger.__globals__["COLUMNS"]).to_csv(path, index=False)
    return report.read_ledger(path)


def status_json(ok=True, stale=(), age_h=0.5):
    d = {"generated_at": (NOW - pd.Timedelta(hours=age_h)).tz_convert("UTC").isoformat(), "ok": ok,
         "summary": "all feeds fresh" if ok else "stale", "datasets": {"fred": {"stale": False, "dupes": 0, "last": "x"}}}
    for k in stale:
        d["datasets"][k] = {"stale": True, "dupes": 0, "last": "x"}
    path = os.path.join(tmp, f"status{len(os.listdir(tmp))}.json")
    json.dump(d, open(path, "w"))
    return path


LC = "28729\t0\tcom.kelvinhe.lake-crypto\n-\t0\tcom.kelvinhe.premia-book\n-\t0\tcom.kelvinhe.kalshi-shadow\n" \
     "-\t0\tcom.kelvinhe.lake-daily\n-\t0\tcom.kelvinhe.lake-hourly\n-\t0\tcom.kelvinhe.options-collector\n" \
     "-\t0\tcom.kelvinhe.desk-report\n"
good = ledger_rows([100000, 100500, 101000, 100800, 100900])
book = report.book_summary(good)
check("book summary: day P&L, since-start P&L and drawdown from the ledger's equity path",
      abs(book["pnl_day"] - 100) < 1e-6 and abs(book["pnl_since_start_pct"] - 0.009) < 1e-9
      and abs(book["drawdown"] - (100900 / 101000 - 1)) < 1e-9 and book["session"] == "2026-09-04")
lake_ok = report.lake_summary(status_json(), NOW)
health_ok = report.daemon_health(good, NOW, LC, log_path=os.path.join(tmp, "none"))
none = report.build_alerts(book, lake_ok, health_ok, report.reconciliation(good), {"last_mark": NOW.isoformat()}, NOW)
check("alerts: a healthy desk raises none", none == [], str(none))

dd_book = report.book_summary(ledger_rows([100000, 100000, 84000, 83000, 83500]))
a = report.build_alerts(dd_book, lake_ok, health_ok, {"mismatches": []}, {}, NOW)
check("alerts: drawdown past the 15% half-size line", [x["kind"] for x in a] == ["drawdown"], str(a))

late = report.daemon_health(ledger_rows([100000, 100500, 101000]), NOW, LC, log_path=os.path.join(tmp, "none"))
a = report.build_alerts(book, lake_ok, late, {"mismatches": []}, {}, NOW)
check("alerts: daemon dead when the last NYSE session is not in the ledger the next morning",
      late["missed"] and late["expected_session"] == "2026-09-04" and any("not in the ledger" in x["msg"] for x in a))
early = report.daemon_health(ledger_rows([100000, 100500, 101000]), pd.Timestamp("2026-09-04 17:00", tz="America/New_York"), LC)
check("alerts: no daemon alert before the 19:00 ET grace on the day itself", not early["missed"])
broken = report.daemon_health(good, NOW, LC.replace("-\t0\tcom.kelvinhe.lake-daily", "-\t1\tcom.kelvinhe.lake-daily")
                              .replace("-\t0\tcom.kelvinhe.premia-book\n", ""))
a = report.build_alerts(book, lake_ok, broken, {"mismatches": []}, {}, NOW)
check("alerts: launchd exit status and an unloaded job are daemon alerts",
      sorted(x["msg"] for x in a) == ["com.kelvinhe.lake-daily last exit status 1", "com.kelvinhe.premia-book is not loaded in launchd"], str(a))

a = report.build_alerts(book, report.lake_summary(status_json(ok=False, stale=["crypto_l2"]), NOW), health_ok, {"mismatches": []}, {}, NOW)
check("alerts: a stale feed", [x["kind"] for x in a] == ["feed"] and "crypto_l2" in a[0]["msg"], str(a))
a = report.build_alerts(book, report.lake_summary(status_json(age_h=5), NOW), health_ok, {"mismatches": []}, {}, NOW)
check("alerts: an old status.json means the quality job itself is dead", [x["kind"] for x in a] == ["feed"] and "5h old" in a[0]["msg"], str(a))
a = report.build_alerts(book, report.lake_summary(os.path.join(tmp, "missing.json"), NOW), health_ok, {"mismatches": []}, {}, NOW)
check("alerts: a missing status.json is a feed alert, not a crash", len(a) == 2 and all(x["kind"] == "feed" for x in a))

def account_rows(held_now):
    """Two alpaca rows in simulated mode: the first sent an order for TLT to its target, the second reports what is held."""
    led = ledger_rows([100000, 100500])
    prev = {**led.iloc[0].to_dict(), "book": "alpaca", "positions": json.dumps({"SPY": 0.15, "TLT": 0.0}),
            "fills": json.dumps([{"symbol": "TLT", "side": "sell", "qty": 1, "notional": None}]),
            "targets": json.dumps({"SPY": 0.15, "TLT": -0.07}), "note": "simulated: submitted"}
    cur = {**led.iloc[1].to_dict(), "book": "alpaca", "positions": json.dumps(held_now), "note": "simulated: submitted"}
    path = os.path.join(tmp, f"ledger{len(os.listdir(tmp))}.csv")
    pd.concat([led, pd.DataFrame([prev, cur])]).to_csv(path, index=False)
    return report.read_ledger(path)


rec = report.reconciliation(account_rows({"SPY": 0.15, "TLT": 0.0}))
check("reconciliation: the broker's own check, held against what the previous row said would be held, past 5% of equity",
      rec["against"] == "2026-08-31" and [m["instrument"] for m in rec["mismatches"]] == ["TLT"], str(rec))
rec_ok = report.reconciliation(account_rows({"SPY": 0.17, "TLT": -0.06}))
check("reconciliation: inside the tolerance is not a mismatch", rec_ok["alpaca_rows"] and rec_ok["mismatches"] == [])
rec_first = report.reconciliation(ledger_rows([100000, 100500], alpaca={"SPY": 0.15, "TLT": -0.07}))
check("reconciliation: a first account row has nothing to reconcile against", rec_first["alpaca_rows"] and "against" not in rec_first)
rec_h = report.reconciliation(ledger_rows([100000, 100500], hashes=["aaa", "bbb"]))
a = report.build_alerts(book, lake_ok, health_ok, rec_h, {}, NOW)
check("reconciliation: a moved panel hash is an alert", rec_h["panel_hash_changed"] and [x["kind"] for x in a] == ["reconcile"])
a = report.build_alerts(book, lake_ok, health_ok, rec, {}, NOW)
check("alerts: reconciliation mismatch names the instrument", any(x["kind"] == "reconcile" and "TLT" in x["msg"] for x in a))

kill_path = os.path.join(tmp, "KILL")
check("kill switch: no file means armed and no alert", not report.kill_status(kill_path)["present"]
      and report.build_alerts(book, lake_ok, health_ok, {"mismatches": []}, {}, NOW, report.kill_status(kill_path)) == [])
open(kill_path, "w").write("2026-09-04T18:49:00 gross 3.20 above limit 3.0\n")
a = report.build_alerts(book, lake_ok, health_ok, {"mismatches": []}, {}, NOW, report.kill_status(kill_path))
check("kill switch: a KILL file is the first alert and carries the reason written into it",
      a[0]["kind"] == "kill" and "gross 3.20 above limit" in a[0]["msg"], str(a))
loss = report.book_summary(ledger_rows([100000, 100000, 96500]))
a = report.build_alerts(loss, lake_ok, health_ok, {"mismatches": []}, {}, NOW)
check("alerts: a day past the broker's 3% daily loss limit", any("daily loss limit" in x["msg"] for x in a), str(a))

rng = np.random.default_rng(7)
R = pd.DataFrame({"X": rng.normal(0, 0.01, 1000), "Y": rng.normal(0, 0.02, 1000)})
v1 = report.ledoit_wolf_var({"X": 1.0}, R, 100000)
v2 = report.ledoit_wolf_var({"X": 2.0}, R, 100000)
v0 = report.ledoit_wolf_var({}, R, 100000)
check("VaR: Ledoit-Wolf 95% one-day VaR is 1.645 sigma and scales with the weight",
      abs(v1["var"]["95"] / (1.6449 * 0.01) - 1) < 0.1 and abs(v2["var"]["95"] / v1["var"]["95"] - 2) < 1e-6
      and abs(v1["var_usd"]["95"] - v1["var"]["95"] * 100000) < 1e-6, f"{v1['var']['95']:.4%}")
check("VaR: no positions or no history gives zero, not an error", v0["var"]["95"] == 0.0 and report.ledoit_wolf_var({"X": 1.0}, R.head(5), 1)["var"]["95"] == 0.0)
check("VaR: an instrument with no cached returns is dropped, not padded",
      report.ledoit_wolf_var({"X": 1.0, "Q": 5.0}, R, 1)["instruments"] == 1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "kalshi-desk"))
import paper
kdb = os.path.join(tmp, "kalshi.sqlite")
db = sqlite3.connect(kdb)
db.executescript(paper.SCHEMA)
db.execute("insert into orders (order_id, ticker, side, price, mid, book) values ('o1', 'T1', 'no', 0.55, 0.455, 'shadow')")
db.execute("insert into fills (fill_id, order_id, ticker, side, cost, count, fee, ts, book) values ('f1', 'o1', 'T1', 'no', 0.55, 4, 0, '2026-09-04T00:00:00+00:00', 'shadow')")
db.execute("insert into fills (fill_id, order_id, ticker, side, cost, count, fee, ts, book) values ('f2', 'o1', 'T2', 'no', 0.50, 2, 0, '2026-09-04T00:00:00+00:00', 'shadow')")
db.execute("insert into settlements values ('T1', 'no', '2026-09-04T12:00:00+00:00')")
db.execute("insert into marks values ('2026-09-05T11:00:00+00:00', 'T2', 0.5)")
db.commit(); db.close()
k = report.kalshi_summary(kdb)
sb = k["books"]["shadow"]
check("kalshi: settled P&L through paper.stats on a read-only connection, open contracts and dollars at risk counted",
      sb["n"] == 1 and abs(sb["pnl"] - 0.45 * 4) < 1e-9 and sb["fills"] == 2 and sb["open"] == 1 and abs(sb["at_risk"] - 1.0) < 1e-9
      and "demo" not in k["books"], str(sb))
check("kalshi: a missing ledger is reported, not raised", "error" in report.kalshi_summary(os.path.join(tmp, "no.sqlite")))

r = report.build(NOW, ledger_path=os.path.join(tmp, "ledger0.csv"), kalshi_path=kdb, lake_path=status_json(),
                 launchctl_text=LC, do_replay=False, kill_path=os.path.join(tmp, "no-KILL"))
md = report.write(r, os.path.join(tmp, "reports"))
latest = json.load(open(os.path.join(tmp, "reports", "latest.json")))
alerts_file = json.load(open(os.path.join(tmp, "reports", "alerts.json")))
check("write: dated md and json, latest.json and alerts.json, three headline lines, alerts as a list",
      os.path.exists(md) and md.endswith("2026-09-05.md") and os.path.exists(os.path.join(tmp, "reports", "daily", "2026-09-05.json"))
      and len(latest["headline"]) == 3 and latest["headline"][2].startswith("no alerts") and alerts_file["alerts"] == []
      and "## Alerts" in open(md).read(), str(latest["headline"]))
check("report never touches keys: read_env and AlpacaBroker are not imported into report.py",
      "read_env" not in dir(report) and "AlpacaBroker" not in dir(report))

print("\nOPTIMIZER\n")
from framework.book import allocate, optimizer
from framework.book.optimizer import Params, covariance, optimize, walk

S_NAMES = ["S1", "S2", "S3"]
mu = pd.Series([0.08, 0.04, 0.02], index=S_NAMES)
cov = pd.DataFrame(np.diag([0.01, 0.01, 0.01]), S_NAMES, S_NAMES)
zero = pd.Series(0.0, index=S_NAMES)
loose = dict(max_sleeve=50.0, max_gross=50.0, max_net=50.0, target_vol=1.0, kelly=100.0)
w_free, _ = optimize(mu, cov, zero, zero, Params(**loose))
check("optimizer: with every cap loose the solution is mu / var, the full-Kelly mix (8, 4, 2)",
      np.allclose(w_free, [8, 4, 2], atol=1e-3), str(w_free.round(3).to_dict()))
w_cap, _ = optimize(mu, cov, zero, zero, Params(**{**loose, "max_sleeve": 0.4}))
check("sleeve cap binds: every sleeve that wanted more sits at 0.4", np.allclose(w_cap, 0.4, atol=1e-4))
w_g, _ = optimize(mu, cov, zero, zero, Params(**{**loose, "max_gross": 1.0}))
w_n, _ = optimize(mu, cov, zero, zero, Params(**{**loose, "max_net": 1.2}))
check("gross and net caps bind: weights sum to 1.0 and 1.2 where unconstrained they sum to 14",
      np.isclose(w_g.sum(), 1.0, atol=1e-4) and np.isclose(w_n.sum(), 1.2, atol=1e-4))
w_v, iv = optimize(mu, cov, zero, zero, Params(**{**loose, "target_vol": 0.05}))
check("vol cap binds: ex-ante vol of the solution is 5% (unconstrained 92%)",
      np.isclose(np.sqrt(w_v @ cov @ w_v), 0.05, atol=1e-3) and iv["vol_bound"])
w_k, ik = optimize(mu, cov, zero, zero, Params(**{**loose, "kelly": 0.5}))
u = w_k / w_k.sum()
check("Kelly cap: leverage is half the Kelly leverage of the chosen mix (7 of 14)",
      np.isclose(w_k.sum(), 0.5 * (mu @ u) / (u @ cov @ u), atol=1e-3) and np.isclose(w_k.sum(), 7.0, atol=0.01)
      and ik["kelly_scaled"], f"{w_k.sum():.3f}")
prev = pd.Series(0.2, index=S_NAMES)
w_t, it = optimize(mu, cov, prev, zero, Params(**loose), turnover=0.3)
check("turnover budget respected and binding: |w - w_prev|_1 = 0.3 where the free move is 13.4",
      (w_t - prev).abs().sum() <= 0.3 + 1e-5 and it["budget_bound"] and (w_free - prev).abs().sum() > 0.3,
      f"{(w_t - prev).abs().sum():.4f}")
moves = [(optimize(mu, cov, prev, pd.Series(c, index=S_NAMES), Params(**loose))[0] - prev).abs().sum()
         for c in (0.0, 0.01, 0.03, 0.1, 1.0)]
check("cost penalty monotone: the move shrinks as the cost rises and is zero at a prohibitive cost",
      all(a >= b - 1e-6 for a, b in zip(moves, moves[1:])) and moves[0] > moves[-1] and moves[-1] < 1e-5,
      " > ".join(f"{m:.3f}" for m in moves))
sing = pd.DataFrame([[0.01, 0.01, 0.0], [0.01, 0.01, 0.0], [0.0, 0.0, 0.01]], S_NAMES, S_NAMES)
w_s, i_s = optimize(mu, sing, zero, zero, Params())
neg = pd.DataFrame([[0.01, 0.0, 0.0], [0.0, -1e-4, 0.0], [0.0, 0.0, 0.01]], S_NAMES, S_NAMES)
w_neg, i_neg = optimize(mu, neg, zero, zero, Params())
w_0, i_0 = optimize(mu, cov * 0.0, zero, zero, Params())
w_nan, i_nan = optimize(mu, cov * np.nan, prev, zero, Params())
check("degenerate covariance: singular, indefinite and all-zero matrices solve within the caps; NaN returns the held weights",
      all(i["status"] == "optimal" and np.isfinite(w).all() and w.sum() <= Params().max_gross + 1e-6 and (w >= 0).all()
          for w, i in ((w_s, i_s), (w_neg, i_neg), (w_0, i_0)))
      and i_nan["status"] == "skipped" and w_nan.equals(prev),
      f"singular {w_s.round(3).to_dict()} zero {w_0.round(3).to_dict()}")
w_neg_mu, _ = optimize(-mu, cov, prev, zero, Params())
check("a negative trailing mean is not bought: long-only, the book goes flat", (w_neg_mu <= 1e-6).all())

R = allocate.returns_frame(results)
fin = pd.Series(0.03, index=R.index)
oos, path, log = walk(R, results, fin)
W = pd.DataFrame({d: p["mvo"] for d, p in path.items()}).T
moved = W.diff().abs().sum(axis=1).iloc[1:]
fits = log.set_index("date").loc[W.index]
check("walk: out-of-sample streams for the candidate and 1/N on the same days, weights inside every cap",
      len(oos) > 252 and set(oos.columns) == {"mvo", "equal"} and (W <= Params().max_sleeve + 1e-6).all().all()
      and (W.sum(axis=1) <= Params().max_gross + 1e-6).all() and (W >= 0).all().all(),
      f"{len(oos)} days, {len(W)} refits, max weight {W.max().max():.3f}, max sum {W.sum(axis=1).max():.3f}, min {W.min().min():.3f}")
check("walk: every budgeted refit moves at most the budget; only the first fit and a risk cut are unbudgeted",
      ((moved <= Params().turnover + 1e-6) | ~fits["budgeted"].iloc[1:].astype(bool)
       | fits["budget_dropped"].iloc[1:].astype(bool)).all()
      and fits["budgeted"].astype(bool).sum() == (fits["status"] == "optimal").sum() - 1,
      f"max {moved.max():.3f}, {int(fits['budgeted'].astype(bool).sum())} budgeted of {int((fits['status'] == 'optimal').sum())}")
d = W.index[-1]
w_re, _, _ = optimizer.fit(R, d, optimizer.sleeve_costs(results, d), w_prev=W.iloc[-2])
check("walk: a refit recomputed by hand from the same window and held weights matches the path", np.allclose(w_re, W.iloc[-1], atol=1e-6))
first = oos.index[0]
q = W.index[W.index < first][-1]
check("walk: the first out-of-sample day is charged the reallocation from zero at each sleeve's realised cost",
      np.isclose(oos.loc[first, "mvo"], float(R.loc[first] @ W.loc[q]) - float((W.loc[q] * optimizer.sleeve_costs(results, q)).sum())
                 - max(W.loc[q].sum() - 1, 0) * 0.03 / 252, atol=1e-9))
_, src = covariance(R.tail(300))
import tempfile
stub = tempfile.mkdtemp()
with open(os.path.join(stub, "risk_model.py"), "w") as fh:
    fh.write("import numpy as np\ndef covariance(R):\n    return np.eye(R.shape[1]) * 2.0\n")
optimizer.RISK_MODEL, keep = stub, optimizer.RISK_MODEL
cov_rm, src_rm = covariance(R.tail(300))
optimizer.RISK_MODEL = keep
check("covariance: Ledoit-Wolf when risk-model/ is absent, the project's own estimator when it exists",
      src == "ledoit-wolf" and src_rm == "risk-model" and np.allclose(cov_rm, np.eye(3) * 2.0 * 252))
out_rule, *_ = allocate.allocate(bars, results, write=False)
allocate.LIVE_ALLOCATOR = "mvo"
out_mvo, *_ = allocate.allocate(bars, results, write=False)
allocate.LIVE_ALLOCATOR = "rule"
check("allocate: the rule picks ERC or 1/N as before and records the candidate's weights; LIVE_ALLOCATOR = 'mvo' makes them live",
      out_rule["allocator"] in ("erc", "equal") and out_rule["candidate"]["allocator"] == "mvo"
      and out_mvo["allocator"] == "mvo" and out_mvo["weights"] == out_rule["candidate"]["weights"]
      and json.dumps(out_mvo), f"rule {out_rule['allocator']} {out_rule['weights']}, candidate {out_rule['candidate']['weights']}")

print("\nSIMULATED ACCOUNT\n")
d0, d1 = cal[-3], cal[-2]
ord_rows = [{"symbol": "A", "side": "buy", "qty": None, "notional": 10000.0, "limit": None},
            {"symbol": "B", "side": "sell", "qty": 5, "notional": None, "limit": None}]
row = {"date": str(d0.date()), "book": "alpaca", "fills": json.dumps(ord_rows), "run_at": f"{d0.date()}T16:45:00-04:00",
       "note": "simulated: submitted"}
acct0 = SimulatedAccount([row], bars.upto(d0))
check("orders recorded on session D are pending until the next session's bar exists",
      acct0.pending == ord_rows and acct0.qty == {} and acct0.cash == CAPITAL)
acct1 = SimulatedAccount([row], bars.upto(d1))
oa, ob = float(bars.open.loc[d1, "A"]), float(bars.open.loc[d1, "B"])
check("then fill at D+1's open: the notional buy as notional / open, the short as whole shares, cash moved, no cost",
      np.isclose(acct1.qty["A"], 10000.0 / oa) and acct1.qty["B"] == -5.0
      and np.isclose(acct1.cash, CAPITAL - 10000.0 + 5 * ob) and not acct1.pending, f"{acct1.qty}")
px1 = bars.close.loc[d1].to_dict()
check("equity is cash plus positions at the close, and the replay is deterministic",
      np.isclose(acct1.equity(px1), acct1.cash + acct1.qty["A"] * px1["A"] - 5 * px1["B"])
      and SimulatedAccount([row], bars.upto(d1)).qty == acct1.qty)
lo_c, op_c = float(bars.low.loc[d1, "C"]), float(bars.open.loc[d1, "C"])
lim_rows = [{"symbol": "C", "side": "buy", "qty": 3, "notional": None, "limit": lo_c * 0.9},
            {"symbol": "D", "side": "buy", "qty": 3, "notional": None, "limit": float(bars.open.loc[d1, "D"]) * 1.05}]
acct2 = SimulatedAccount([dict(row, fills=json.dumps(lim_rows))], bars.upto(d1))
check("a limit below the day's low never fills and stays pending; a limit above the open fills at the open",
      "C" not in acct2.qty and len(acct2.pending) == 1 and acct2.qty.get("D") == 3.0
      and acct2.fills[0]["fill"] == round(float(bars.open.loc[d1, "D"]), 4))
sim1 = AlpacaBroker(sim_rows=[row], bars=bars.upto(d1))
check("AlpacaBroker in simulated mode reports that account's equity and weights",
      np.isclose(sim1.equity(), acct1.equity(px1)) and np.isclose(sim1.weights()["B"], -5 * px1["B"] / sim1.equity()))

print("\nRECONCILIATION\n")
prev = {"date": str(d0.date()), "equity": 100000.0, "positions": json.dumps({"A": 0.10, "B": -0.05, "C": 0.20}),
        "targets": json.dumps({"A": 0.10, "B": 0.0, "C": 0.30}),
        "fills": json.dumps([{"symbol": "B", "side": "buy", "qty": 5, "notional": None, "limit": None},
                             {"symbol": "C", "side": "buy", "qty": None, "notional": 10000.0, "limit": None}])}
check("expected weights: a name the previous run sent an order for sits at its target, the rest at what was held",
      expected_weights(prev) == {"A": 0.10, "B": 0.0, "C": 0.30})
check("verify: drift inside 5% of equity on every name passes and writes no KILL file",
      verify_positions({"A": 0.11, "C": 0.28}, prev) == {} and not os.path.exists(broker_mod.KILL))
try:
    verify_positions({"A": 0.10, "C": 0.20}, prev)
    check("verify: a name 10% of equity off halts", False)
except Halted as e:
    check("verify: a name 10% of equity off halts and the KILL file names it",
          "C held +0.2000 expected +0.3000" in str(e) and "reconciliation" in open(broker_mod.KILL).read())
os.remove(broker_mod.KILL)
r1 = dict(row, positions=json.dumps({}), targets=json.dumps({"A": 0.1, "B": -5 * ob / CAPITAL}), equity=CAPITAL)
check("end to end: the simulated account replayed from a row reconciles against that same row",
      verify_positions(sim1.weights(), r1) == {})
check("first run: nothing to reconcile against", verify_positions({"A": 0.5}, None) == {})

# ---- reconciling against the broker's own order records, not against intent


class FakeOrder:
    def __init__(self, oid, symbol, side, status, filled_qty=0, filled_avg_price=None, qty=None, notional=None):
        self.id, self.symbol, self.side, self.status = oid, symbol, side, status
        self.filled_qty, self.filled_avg_price, self.qty, self.notional = filled_qty, filled_avg_price, qty, notional


class FakeOrderClient:
    """Only what order_outcomes touches: look an order up by the id the ledger stored."""

    def __init__(self, orders):
        self.orders = {o.id: o for o in orders}
        self.asked = []

    def get_order_by_id(self, oid):
        self.asked.append(oid)
        return self.orders[oid]


def fake_broker(orders):
    b = object.__new__(AlpacaBroker)
    b.mode, b.client = "paper", FakeOrderClient(orders)
    return b


sent_rows = [{"symbol": "A", "side": "buy", "qty": None, "notional": 20000.0, "limit": None, "id": "o1"},
             {"symbol": "B", "side": "sell", "qty": 50.0, "notional": None, "limit": None, "id": "o2"},
             {"symbol": "C", "side": "buy", "qty": None, "notional": 5000.0, "limit": None, "id": "o3"}]
prev2 = {"date": "2026-09-08", "equity": 100000.0, "mode": "paper",
         "positions": json.dumps({"B": 0.20, "C": 0.05}),
         "targets": json.dumps({"A": 0.20, "B": 0.0, "C": 0.10}),
         "fills": json.dumps(sent_rows)}
br = fake_broker([FakeOrder("o1", "A", "buy", "partially_filled", filled_qty=20, filled_avg_price=500.0, notional=20000.0),
                  FakeOrder("o2", "B", "sell", "rejected", qty=50.0),
                  FakeOrder("o3", "C", "buy", "filled", filled_qty=10, filled_avg_price=500.0, notional=5000.0)])
outs = br.order_outcomes(sent_rows)
check("order ids stored on the ledger row are polled at the broker and classified",
      br.client.asked == ["o1", "o2", "o3"] and [o["status"] for o in outs] == ["partial", "rejected", "filled"],
      str([o["status"] for o in outs]))
check("a partial fill is valued at the broker's own average price, a reject at nothing",
      [o["filled_value"] for o in outs] == [10000.0, 0.0, 5000.0], str([o["filled_value"] for o in outs]))
check("unfilled() names exactly the orders that did not fully fill",
      [(o["instrument"], o["status"]) for o in unfilled(outs)] == [("A", "partial"), ("B", "rejected")])
actual = {"A": 0.10, "B": 0.20, "C": 0.10}   # half of A bought, B never sold, C filled
check("PARTIAL FILL + REJECT: the account matches the broker's record, so reconciliation passes",
      verify_positions(actual, prev2, filled_value(outs), 100000.0) == {}
      and not os.path.exists(broker_mod.KILL))
try:
    verify_positions(actual, prev2)
    check("PARTIAL FILL + REJECT: reconciling against intent instead would have false-halted", False)
except Halted as e:
    check("PARTIAL FILL + REJECT: reconciling against intent instead false-halts on A and B (the old blindness)",
          "A held" in str(e) and "B held" in str(e))
    os.remove(broker_mod.KILL)
try:
    verify_positions({"A": 0.20, "B": 0.0, "C": 0.10}, prev2, filled_value(outs), 100000.0)
    check("PARTIAL FILL: an account that is NOT where the broker's fills put it halts", False)
except Halted as e:
    check("PARTIAL FILL: an account that is NOT where the broker's fills put it halts and names both legs",
          "A held +0.2000 expected +0.1000" in str(e) and "B held +0.0000 expected +0.2000" in str(e), str(e))
    os.remove(broker_mod.KILL)
check("a simulated broker has no order records to poll, so it falls back to the target expectation",
      sim.order_outcomes(sent_rows) == [])
check("order_record carries the broker's id for the response, None when nothing was sent",
      order_record(orders_ex := rc({"SPY": 0.5}, 100_000, prices, held={})[0],
                   FakeOrder("xyz", "SPY", "buy", "new"))["id"] == "xyz"
      and order_record(orders_ex)["id"] is None)

# ---- a mode change is a loud halt, never a silently skipped check
paper_prev = dict(prev2, mode="paper")
sim_prev = dict(prev2, mode="simulated")
check("MODE CHANGE: same mode reconciles against the previous row as usual",
      mode_change(paper_prev, "paper") is paper_prev and mode_change(None, "paper") is None)
try:
    mode_change(sim_prev, "paper")
    check("MODE CHANGE: a simulated -> paper flip halts instead of skipping reconciliation", False)
except Halted as e:
    check("MODE CHANGE: a simulated -> paper flip halts instead of skipping reconciliation, and says so",
          "simulated -> paper" in str(e) and "ack-mode-change" in str(e)
          and "mode changed" in open(broker_mod.KILL).read())
    os.remove(broker_mod.KILL)
check("MODE CHANGE: acknowledged by hand, the run goes on with no cross-account comparison",
      mode_change(sim_prev, "paper", ack=True) is None and not os.path.exists(broker_mod.KILL))
check("MODE CHANGE: a dry run reports it but never arms the kill switch for the live book",
      "ack_mode_change or dry_run" in open(os.path.join(os.path.dirname(__file__), "daemon.py")).read())

print("\nKILL SWITCH AND LIMITS\n")


class FakeClient:
    def __init__(self):
        self.sent, self.cancelled = [], 0

    def cancel_orders(self):
        self.cancelled += 1

    def submit_order(self, o):
        self.sent.append(o)
        return o


dummy.client = FakeClient()
orders = rc({"SPY": 0.5}, 100_000, prices, held={})
with open(broker_mod.KILL, "w") as fh:
    fh.write("planted\n")
try:
    dummy.submit(orders, {"SPY": 0.5}, 100_000)
    check("kill switch: paper submit refuses while the KILL file exists", False)
except Halted as e:
    check("kill switch: paper submit refuses while the KILL file exists, cancels what is queued, sends nothing",
          dummy.client.sent == [] and dummy.client.cancelled == 1 and "KILL" in str(e))
try:
    sim.submit(orders, {"SPY": 0.5}, 100_000)
    check("kill switch: simulated submit refuses too", False)
except Halted:
    check("kill switch: simulated submit refuses too", True)
os.remove(broker_mod.KILL)
check("kill switch: deleting the file re-enables submit, which sends the orders",
      dummy.submit(orders, {"SPY": 0.5}, 100_000) == orders and dummy.client.sent == orders)
check("limits: a book inside every limit passes and returns its gross",
      np.isclose(check_limits({"A": 0.4, "B": -0.3}, 100_000, 100_000), 0.7) and not os.path.exists(broker_mod.KILL))
for name, kw in (("gross 3.5 above 3.0", dict(targets={"A": 2.0, "B": -1.5}, equity=1e5)),
                 ("one name at 60% above 50%", dict(targets={"A": 0.6}, equity=1e5)),
                 ("equity down 4% on the previous run past the 3% daily loss", dict(targets={"A": 0.1}, equity=96_000, ref_equity=1e5))):
    try:
        check_limits(**kw)
        check(f"limits: {name} halts", False)
    except Halted:
        check(f"limits: {name} halts and writes the KILL file", os.path.exists(broker_mod.KILL))
        os.remove(broker_mod.KILL)
check("limits: equity down 2% passes", check_limits({"A": 0.1}, 98_000, 1e5) == 0.1)
check("limits: tighter limits can be passed in", Limits(max_per_name=0.1).max_per_name == 0.1)
dummy.client = FakeClient()
try:
    dummy.submit(orders, {"SPY": 0.6}, 100_000)
    check("limits are enforced in submit", False)
except Halted:
    check("limits are enforced in the layer: a breach in submit sends nothing, cancels what was queued and writes the KILL file",
          dummy.client.sent == [] and dummy.client.cancelled == 1 and os.path.exists(broker_mod.KILL))
    os.remove(broker_mod.KILL)
dummy.client = FakeClient()
try:
    dummy.submit(orders)
    check("limits cannot be skipped: orders without targets are refused", False)
except ValueError:
    check("limits cannot be skipped: orders without targets are refused, nothing sent, no KILL file",
          dummy.client.sent == [] and not os.path.exists(broker_mod.KILL))
check("an empty submit without targets (the daemon's kill path) is still allowed", dummy.submit([]) == [])
check("daemon: after a halt the kill file gates the next run before any data is fetched",
      "killed()" in open(os.path.join(os.path.dirname(__file__), "daemon.py")).read().split("def run_once")[1].split("live_bars")[0])

print("\nDORMANT LIVE ADAPTERS\n")
asked = []
real_read_env = broker_mod.read_env


def spy(keys=broker_mod.PAPER_KEYS, path=None):
    asked.extend(keys)
    return real_read_env(keys, path)


broker_mod.read_env = spy
today = pd.Timestamp("2026-09-05").date()


def env(text):
    with open(broker_mod.ENV, "w") as fh:
        fh.write(text)


LIVE_KEYS = "ALPACA_LIVE_KEY=live-secret-key\nALPACA_LIVE_SECRET=live-secret\n"
adapters = {"AlpacaLive": lambda c: AlpacaLive(confirm=c, today=today),
            "IBKRBroker": lambda c: IBKRBroker(confirm=c, today=today),
            "KalshiLive": lambda c: KalshiLive(confirm=c, today=today)}
cases = [("no LIVE_ENABLED in .env, token given", LIVE_KEYS, live_token(today)),
         ("LIVE_ENABLED=true, no token", "LIVE_ENABLED=true\n" + LIVE_KEYS, None),
         ("LIVE_ENABLED=true, yesterday's token", "LIVE_ENABLED=true\n" + LIVE_KEYS, live_token(today - pd.Timedelta(days=1))),
         ("LIVE_ENABLED=false, today's token", "LIVE_ENABLED=false\n" + LIVE_KEYS, live_token(today))]
for cls, make in adapters.items():
    for label, text, token in cases:
        env(text)
        asked.clear()
        try:
            make(token)
            check(f"{cls} refuses: {label}", False)
        except LiveDisabled:
            check(f"{cls} refuses: {label}; only LIVE_ENABLED was read, no live key",
                  set(asked) == {"LIVE_ENABLED"}, str(set(asked)))
        except Exception as e:  # noqa: BLE001
            check(f"{cls} refuses: {label}", False, f"{type(e).__name__}: {e}")
    env("LIVE_ENABLED=true\n" + LIVE_KEYS)
    with open(broker_mod.KILL, "w") as fh:
        fh.write("planted\n")
    try:
        make(live_token(today))
        check(f"{cls} refuses with both locks open while the KILL file exists", False)
    except Halted:
        check(f"{cls} refuses with both locks open while the KILL file exists", True)
    os.remove(broker_mod.KILL)
env("LIVE_ENABLED=true\n" + LIVE_KEYS)
asked.clear()
al = AlpacaLive(confirm=live_token(today), today=today)
check("AlpacaLive with both locks open reads the live key and constructs against api.alpaca.markets, not sandbox",
      "ALPACA_LIVE_KEY" in asked and not al.client._sandbox
      and getattr(al.client._base_url, "value", "") == f"https://{broker_mod.LIVE_HOST}")
try:
    IBKRBroker(confirm=live_token(today), today=today)
    check("IBKRBroker with both locks open gets past the gate", False, "connected?")
except ImportError as e:
    check("IBKRBroker with both locks open gets past the gate and stops at the missing ib_insync (not installed)",
          "ib_insync" in str(e))
try:
    KalshiLive(confirm=live_token(today), today=today)
    check("KalshiLive with both locks open stops at kalshi-desk's own LIVE_TRADING lock", False, "constructed")
except (LiveDisabled, RuntimeError) as e:
    check("KalshiLive with both locks open stops at kalshi-desk's own LIVE_TRADING lock or missing prod key",
          "LIVE_TRADING" in str(e) or "key" in str(e).lower())
broker_mod.read_env = real_read_env
os.remove(broker_mod.ENV)
check("read_env only returns the keys asked for", real_read_env(("X",), os.path.join(TMP, "none.env")) == {})

print("\nL2 FILL MODEL\n")
from framework.book.l2fill import queue, walk


def snap(t, bids, asks):
    r = {"ts": t}
    for i in range(1, 21):
        for side, lv in (("bid", bids), ("ask", asks)):
            r[f"{side}_px_{i}"], r[f"{side}_sz_{i}"] = lv[i - 1] if i <= len(lv) else (np.nan, np.nan)
    return r


book = snap(0, [(100.0, 1.0), (99.0, 2.0), (98.0, 5.0)], [(101.0, 1.0), (102.0, 2.0), (103.0, 5.0)])
f = walk(book, "buy", qty=2.5)
check("market buy walks the asks: 1 at 101 and 1.5 at 102 average 101.6", np.isclose(f["price"], 101.6) and not f["exhausted"])
f = walk(book, "sell", qty=10.0)
check("market sell past the visible book is flagged exhausted and priced the rest at the last level",
      f["exhausted"] and np.isclose(f["price"], (100 + 2 * 99 + 5 * 98 + 2 * 98) / 10))
check("market buy by notional sizes off the touch", np.isclose(walk(book, "buy", notional=202.0)["price"], 101.5))
rows = [book,
        snap(1, [(100.0, 1.0), (99.0, 1.0)], [(101.0, 1.0)]),   # 1.0 traded ahead: 1.0 still ahead
        snap(2, [(100.0, 1.0), (99.0, 0.2)], [(101.0, 1.0)]),   # 0.8 more: 0.2 ahead
        snap(3, [(100.0, 1.0), (99.0, 0.5)], [(101.0, 1.0)]),   # 0.3 joined behind: nothing
        snap(4, [(100.0, 1.0)], [(101.0, 1.0)]),                # 0.5 traded: 0.2 ahead then 0.3 of ours
        snap(5, [(98.0, 1.0)], [(99.0, 1.0)])]                  # ask touches 99: the rest fills
q = queue(rows, "buy", 1.0, 99.0)
check("limit buy at 99 behind 2.0: fills 0.3 when trades pass the queue, the rest when the ask reaches 99",
      np.isclose(q["filled"], 1.0) and q["remaining"] == 0 and q["price"] == 99.0 and q["ts"] == 5, str(q))
q = queue(rows[:5], "buy", 1.0, 99.0)
check("the same order over the first five snapshots is 0.3 filled", np.isclose(q["filled"], 0.3) and np.isclose(q["remaining"], 0.7))
q = queue(rows, "buy", 2.5, 102.0)
check("a marketable limit fills at once up to its price like a market order", np.isclose(q["price"], 101.6) and q["remaining"] == 0)
q = queue(rows, "buy", 3.0, 101.0)
check("a marketable limit bigger than the size at its price takes what is there and rests for the remainder",
      np.isclose(q["filled"], 3.0) and np.isclose(q["price"], 101.0), str(q))
q = queue([book, snap(1, [(100.0, 1.0)], [(101.0, 1.0)])], "sell", 1.0, 103.0)
check("a sell that never trades stays unfilled with its full remainder", q["filled"] == 0 and q["remaining"] == 1.0)

print("\nDAEMON HELPERS\n")
def led_row(date, book, equity, mode, note):
    return dict(zip(COLUMNS, [date, book, equity, 0, "{}", "[]", "{}", "h", "t", mode, None, note]))


led = pd.DataFrame([led_row("2026-09-03", "shadow", 1.0, "", "replay"),
                    led_row("2026-09-03", "alpaca", 2.0, "simulated", "misleading: note"),
                    led_row("2026-09-04", "alpaca", 3.0, "paper", "submitted")])
led["date"] = pd.to_datetime(led.date)
check("sim_rows keys on the mode column, not the note text",
      [r["equity"] for r in sim_rows(led)] == [2.0])
check("previous_row keys on (date, book) alone: it finds the last alpaca row whatever the note or the mode says",
      previous_row(led, "2026-09-05")["equity"] == 3.0 and previous_row(led, "2026-09-04")["equity"] == 2.0
      and previous_row(led, "2026-09-03") is None)

mig = os.path.join(tempfile.mkdtemp(), "old-ledger.csv")
OLD = ["date", "book", "equity", "gross", "positions", "fills", "targets", "panel_hash", "run_at", "note"]
pd.DataFrame([dict(zip(OLD, ["2026-09-04", "alpaca", 4.0, 0, "{}", "[]", "{}", "h", "t", "simulated: submitted"]))],
             columns=OLD).to_csv(mig, index=False)
back = daemon_mod.read_ledger(mig)
check("a ledger written before the mode column reads back with mode backfilled from the old note prefix",
      list(back.columns) == COLUMNS and back["mode"].tolist() == ["simulated"])
append_ledger([led_row("2026-09-08", "alpaca", 5.0, "paper", "submitted")], mig)
again = pd.read_csv(mig)
check("appending to it rewrites the header once so the new columns line up, and nothing is lost",
      list(again.columns) == COLUMNS and again["mode"].tolist() == ["simulated", "paper"] and len(again) == 2)

print("\nCRYPTO VENDOR OUTAGE\n")
import shutil
import tempfile

from framework.book import universe as uni

tmp = tempfile.mkdtemp()
cpanel = synthetic(n_days=40, instruments=uni.CRYPTO, seed=11)
CSTART, CEND = "2010-01-01", "2010-03-01"


def write_crypto_cache(d):
    for sym in uni.CRYPTO:
        df = pd.DataFrame({f: cpanel.field(f)[sym] for f in
                           ("open", "high", "low", "close", "volume")})
        df.index = pd.bdate_range("2010-01-01", periods=len(df))
        df.index.name = "date"
        # a different date key, as a previous session would have left it
        df.to_csv(os.path.join(d, f"{sym.replace('/', '-')}_daily_2010-02-26.csv"))


real_alpaca = uni._alpaca_bars


def failing_alpaca(bad):
    def _f(sym, start, end, **kw):
        if sym in bad:
            return None
        df = pd.DataFrame({f: cpanel.field(f)[sym] for f in
                           ("open", "high", "low", "close", "volume")})
        df.index = pd.bdate_range("2010-01-01", periods=len(df))
        df.index.name = "date"
        return df
    return _f


try:
    uni._alpaca_bars = failing_alpaca({"BTC/USD"})
    write_crypto_cache(tmp)
    out = uni.load_crypto(CSTART, CEND, cache=tmp, max_age_days=10_000)
    check("one flaky crypto pair falls back to cache and the panel stays full",
          list(out["close"].columns) == uni.CRYPTO and out["close"]["BTC/USD"].notna().any(),
          f"{list(out['close'].columns)}")

    shutil.rmtree(tmp); os.makedirs(tmp)
    write_crypto_cache(tmp)
    uni._alpaca_bars = failing_alpaca(set(uni.CRYPTO))
    try:
        uni.load_crypto(CSTART, CEND, cache=tmp, max_age_days=10_000)
        raised = False
    except RuntimeError:
        raised = True
    check("every crypto pair missing raises even though both are cached", raised)

    shutil.rmtree(tmp); os.makedirs(tmp)
    uni._alpaca_bars = failing_alpaca({"BTC/USD"})
    try:
        uni.load_crypto(CSTART, CEND, cache=tmp)
        raised = False
    except RuntimeError:
        raised = True
    check("a missing crypto pair with no cache at all raises", raised)

    write_crypto_cache(tmp)
    try:
        uni.load_crypto(CSTART, CEND, cache=tmp, max_age_days=1)
        raised = False
    except RuntimeError:
        raised = True
    check("crypto cache staler than max_age_days is refused, not silently traded", raised)
finally:
    uni._alpaca_bars = real_alpaca
    shutil.rmtree(tmp, ignore_errors=True)


# --- rate vintage ------------------------------------------------------------
# One cross-sectional carry rank must compare rates from one month. Mixed
# vintages (a current EUR against a three-month-old AUD) bias the sort.
_rates = uni.load_rates()
_ends = {c: _rates[c].last_valid_index() for c in _rates}
check("every currency's rate series ends on the same month",
      len(set(_ends.values())) == 1,
      ", ".join(f"{c} {d.date()}" for c, d in _ends.items()))
check("rate panel has no trailing all-NaN row", _rates.iloc[-1].notna().all())


print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
