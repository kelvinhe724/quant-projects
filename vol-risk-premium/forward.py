"""The gate layer of PREREG.md: the forward paper track on the collector's own option snapshots.

Each run reads every snapshot session in the lake for the ticker, opens a cycle
when one is due (21 sessions after the last start, VIX at or below 30, a listed
strike nearest spot at the nearest monthly expiry 25-40 days out), sells the
straddle at the bid, marks it at the mid every session, hedges in SPY at the
same snapshot's SPY spot with the Black-Scholes delta at the sold vol, and
settles at intrinsic on the expiry session. Append-only CSVs; a session already
marked is never marked twice. Per unit: sizing is a separate, pre-registered rule.

Run:  ../.venv/bin/python3 forward.py                 the XSP track -> reports/forward/
      ../.venv/bin/python3 forward.py --ticker SPY --dry   mechanics on SPY snapshots -> reports/forward-dry/
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data-lake"))
sys.path.insert(0, os.path.dirname(HERE))
import lake  # noqa: E402
from framework.book.universe import sessions as nyse_sessions  # noqa: E402
from vrp import straddle_delta  # noqa: E402

HORIZON = 21
VIX_ENTRY_MAX = 30.0
MIN_DAYS, MAX_DAYS = 25, 40
MULTIPLIER = 100
HEDGE_BPS = 1.0
UNITS = 1                      # per-unit track; PREREG.md §3 sizing is applied in reporting, not here
COMMISSION = 0.65              # per contract per leg, Alpaca options paper schedule

CYCLE_COLS = ["start", "expiry", "strike", "spot", "units", "entry_bid", "entry_mid", "spread_pct", "iv", "vix", "monthly"]
MARK_COLS = ["date", "start", "spot", "spy", "straddle_mid", "hedge_shares", "pnl_option", "pnl_hedge", "cost",
             "pnl", "status"]


def vix_closes(start, end):
    import yfinance as yf
    v = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)["Close"]
    v = v.iloc[:, 0] if isinstance(v, pd.DataFrame) else v
    return v.dropna()


def sessions(ticker, start="2026-09-01"):
    """{day: chain} for every snapshot session of `ticker`, plus SPY spot by day for the hedge."""
    end = str((pd.Timestamp.today() + pd.Timedelta(days=1)).date())
    for i in range(6):                 # the collector may still be writing today's manifest; wait, don't crash
        try:
            df = lake.load("options", start, end)
            break
        except FileNotFoundError as e:
            if i == 5:
                raise
            print(f"lake not ready ({e}); retry {i + 1}/5 in 5 min")
            import time; time.sleep(300)
    df["day"] = pd.to_datetime(df.snapshot).dt.normalize()
    spy = df[df.ticker == "SPY"].groupby("day").spot.first()
    own = df[df.ticker == ticker]
    live = set(nyse_sessions(own.day.min(), own.day.max())) if len(own) else set()
    return {d: g for d, g in own.groupby("day") if d in live}, spy   # a holiday snapshot is stale, not a session


def pick(chain):
    """The straddle the sleeve sells on this session, or None."""
    g = chain[(chain["T"] * 365 >= MIN_DAYS) & (chain["T"] * 365 <= MAX_DAYS)]
    if g.empty:
        return None
    exp = pd.to_datetime(g.expiry)
    monthly = g[(exp.dt.weekday == 4) & (exp.dt.day >= 15) & (exp.dt.day <= 21)]
    src = monthly if not monthly.empty else g
    e = src.sort_values("T").expiry.iloc[0]
    c = g[g.expiry == e]
    spot = float(c.spot.iloc[0])
    ks = set(c[c.cp == "C"].strike) & set(c[c.cp == "P"].strike)
    if not ks:
        return None
    k = min(ks, key=lambda x: abs(x - spot))
    call = c[(c.cp == "C") & (c.strike == k)].iloc[0]
    put = c[(c.cp == "P") & (c.strike == k)].iloc[0]
    bid, mid = call.bid + put.bid, (call.bid + call.ask + put.bid + put.ask) / 2
    if bid <= 0 or mid <= 0:
        return None
    return {"expiry": pd.Timestamp(e), "strike": float(k), "spot": spot, "entry_bid": float(bid), "entry_mid": float(mid),
            "spread_pct": float((mid - bid) * 2 / mid), "iv": float((call.impliedVolatility + put.impliedVolatility) / 2),
            "monthly": not monthly.empty, "T": float(c["T"].iloc[0])}


def mark(chain, strike, expiry):
    """Mid of the open straddle on this session; intrinsic if the expiry has arrived."""
    spot = float(chain.spot.iloc[0])
    day = pd.to_datetime(chain.snapshot.iloc[0]).normalize()
    if day >= expiry.normalize():
        return spot, abs(spot - strike), 0.0, True
    c = chain[(chain.expiry == expiry) & (chain.strike == strike)]
    if len(c) < 2:
        return spot, np.nan, np.nan, False
    call, put = c[c.cp == "C"].iloc[0], c[c.cp == "P"].iloc[0]
    return spot, float((call.bid + call.ask + put.bid + put.ask) / 2), float(c["T"].iloc[0]), False


def run(ticker="^XSP", out=None, units=UNITS):
    out = out or os.path.join(HERE, "reports", "forward")
    os.makedirs(out, exist_ok=True)
    cpath, mpath = os.path.join(out, "cycles.csv"), os.path.join(out, "marks.csv")
    cycles = pd.read_csv(cpath, parse_dates=["start", "expiry"]) if os.path.exists(cpath) else pd.DataFrame(columns=CYCLE_COLS)
    marks = pd.read_csv(mpath, parse_dates=["date", "start"]) if os.path.exists(mpath) else pd.DataFrame(columns=MARK_COLS)
    days, spy = sessions(ticker)
    if not days:
        print(f"no {ticker} snapshots in the lake")
        return cycles, marks
    vix = vix_closes(min(days) - pd.Timedelta(days=7), max(days) + pd.Timedelta(days=1))
    done = set(marks.date) if len(marks) else set()
    new_c, new_m = [], []
    for day in sorted(days):
        if day in done:
            continue
        chain = days[day]
        open_c = [c for c in (list(cycles.to_dict("records")) + new_c) if c.get("status", "open") == "open"]
        if open_c:
            c = open_c[-1]
            spot, smid, T, settled = mark(chain, c["strike"], pd.Timestamp(c["expiry"]))
            prev = [m for m in (list(marks.to_dict("records")) + new_m) if pd.Timestamp(m["start"]) == pd.Timestamp(c["start"])]
            last_mid = prev[-1]["straddle_mid"] if prev else c["entry_mid"]
            last_sh = prev[-1]["hedge_shares"] if prev else 0.0
            last_spy = prev[-1]["spy"] if prev else spy.get(pd.Timestamp(c["start"]), np.nan)
            spy_now = spy.get(day, np.nan)
            if np.isnan(smid) or np.isnan(spy_now):
                new_m.append({"date": day, "start": c["start"], "spot": spot, "spy": spy_now, "straddle_mid": smid,
                              "hedge_shares": last_sh, "pnl_option": np.nan, "pnl_hedge": np.nan, "cost": np.nan,
                              "pnl": np.nan, "status": "no mark"})
                continue
            pnl_opt = (last_mid - smid) * MULTIPLIER * units
            pnl_hdg = last_sh * (spy_now - last_spy)
            sh_new = 0.0 if settled else -(-straddle_delta(spot, c["strike"], max(T, 1e-6), c["iv"])) * MULTIPLIER * units * (spot / spy_now)
            cost = abs(sh_new - last_sh) * spy_now * HEDGE_BPS / 1e4 + (2 * COMMISSION * units if settled else 0.0)
            new_m.append({"date": day, "start": c["start"], "spot": spot, "spy": spy_now, "straddle_mid": smid,
                          "hedge_shares": sh_new, "pnl_option": pnl_opt, "pnl_hedge": pnl_hdg, "cost": cost,
                          "pnl": pnl_opt + pnl_hdg - cost, "status": "settled" if settled else "open"})
            if settled:
                c["status"] = "settled"
            continue
        starts = [pd.Timestamp(c["start"]) for c in (list(cycles.to_dict("records")) + new_c)]
        # the cycle clock runs on NYSE sessions, not on snapshot sessions: a missed collector day still counts
        since = len(nyse_sessions(max(starts) + pd.Timedelta(days=1), day)) if starts else HORIZON
        if since < HORIZON:
            new_m.append({"date": day, "start": pd.NaT, "spot": float(chain.spot.iloc[0]), "spy": spy.get(day, np.nan),
                          "straddle_mid": np.nan, "hedge_shares": 0.0, "pnl_option": 0.0, "pnl_hedge": 0.0, "cost": 0.0,
                          "pnl": 0.0, "status": "flat"})
            continue
        v = float(vix.asof(day)) if len(vix) else np.nan
        p = pick(chain)
        if p is None or np.isnan(v) or v > VIX_ENTRY_MAX:
            why = "no straddle" if p is None else ("no VIX" if np.isnan(v) else f"VIX {v:.1f} > {VIX_ENTRY_MAX:.0f}")
            new_m.append({"date": day, "start": pd.NaT, "spot": float(chain.spot.iloc[0]), "spy": spy.get(day, np.nan),
                          "straddle_mid": np.nan, "hedge_shares": 0.0, "pnl_option": 0.0, "pnl_hedge": 0.0, "cost": 0.0,
                          "pnl": 0.0, "status": f"no entry: {why}"})
            continue
        spy_now = spy.get(day, np.nan)
        sh = -(-straddle_delta(p["spot"], p["strike"], p["T"], p["iv"])) * MULTIPLIER * units * (p["spot"] / spy_now)
        cost = abs(sh) * spy_now * HEDGE_BPS / 1e4 + 2 * COMMISSION * units
        new_c.append({"start": day, "expiry": p["expiry"], "strike": p["strike"], "spot": p["spot"], "units": units,
                      "entry_bid": p["entry_bid"], "entry_mid": p["entry_mid"], "spread_pct": p["spread_pct"], "iv": p["iv"],
                      "vix": v, "monthly": p["monthly"], "status": "open"})
        # sold at the bid, marked at the mid: the spread is paid on day one
        new_m.append({"date": day, "start": day, "spot": p["spot"], "spy": spy_now, "straddle_mid": p["entry_mid"],
                      "hedge_shares": sh, "pnl_option": (p["entry_bid"] - p["entry_mid"]) * MULTIPLIER * units,
                      "pnl_hedge": 0.0, "cost": cost,
                      "pnl": (p["entry_bid"] - p["entry_mid"]) * MULTIPLIER * units - cost, "status": "open"})
    cycles = pd.concat([cycles, pd.DataFrame(new_c)], ignore_index=True) if new_c else cycles
    if len(cycles):
        settled_starts = {pd.Timestamp(m["start"]) for m in new_m if m["status"] == "settled"}
        cycles["status"] = [("settled" if pd.Timestamp(s) in settled_starts else st)
                            for s, st in zip(cycles.start, cycles.get("status", pd.Series(["open"] * len(cycles))))]
    marks = pd.concat([marks, pd.DataFrame(new_m)], ignore_index=True) if new_m else marks
    cycles.to_csv(cpath, index=False)
    marks.to_csv(mpath, index=False)
    print(f"{ticker}: {len(new_m)} sessions marked ({len(new_c)} cycles opened), {len(cycles)} cycles total, "
          f"P&L to date ${marks.pnl.sum():,.2f} per {units} unit(s) -> {out}")
    return cycles, marks


if __name__ == "__main__":
    a = sys.argv[1:]
    t = a[a.index("--ticker") + 1] if "--ticker" in a else "^XSP"
    run(t, out=os.path.join(HERE, "reports", "forward-dry") if "--dry" in a else None)
