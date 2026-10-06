"""Measure the XSP ATM straddle spread from the collector's own snapshots (PREREG.md §4).

Prints the per-session table. Once ten XSP sessions exist it writes
reports/spread-frozen.json with the larger of the measured median and the 1.5%
provisional number, and refuses to overwrite it afterwards: the number is
frozen the first time it is written.

Run:  ../.venv/bin/python3 measure_spread.py [--ticker ^XSP] [--min-sessions 10]
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data-lake"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lake  # noqa: E402
from framework.book.universe import sessions as nyse_sessions  # noqa: E402

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
FROZEN = os.path.join(REPORTS, "spread-frozen.json")
PROVISIONAL = 0.015
MIN_DAYS, MAX_DAYS = 25, 40    # §2: the expiry the sleeve sells


def atm_straddles(ticker, start="2026-09-01", end=None):
    """One row per snapshot session: the ATM straddle at the nearest monthly expiry 25-40 days out."""
    end = end or str((pd.Timestamp.today() + pd.Timedelta(days=1)).date())
    df = lake.load("options", start, end)
    df = df[df.ticker == ticker].copy()
    if df.empty:
        return pd.DataFrame()
    df["day"] = pd.to_datetime(df.snapshot).dt.normalize()
    df = df[df.day.isin(nyse_sessions(df.day.min(), df.day.max()))]   # a holiday snapshot is stale, not a session
    rows = []
    for day, g in df.groupby("day"):
        g = g[(g["T"] * 365 >= MIN_DAYS) & (g["T"] * 365 <= MAX_DAYS)]
        if g.empty:
            continue
        exp = pd.to_datetime(g.expiry)
        monthly = g[(exp.dt.weekday == 4) & (exp.dt.day >= 15) & (exp.dt.day <= 21)]
        pick = monthly if not monthly.empty else g
        e = pick.sort_values("T").expiry.iloc[0]
        chain = g[g.expiry == e]
        spot = chain.spot.iloc[0]
        strikes = set(chain[chain.cp == "C"].strike) & set(chain[chain.cp == "P"].strike)
        if not strikes:
            continue
        k = min(strikes, key=lambda x: abs(x - spot))
        c = chain[(chain.cp == "C") & (chain.strike == k)].iloc[0]
        p = chain[(chain.cp == "P") & (chain.strike == k)].iloc[0]
        mid = (c.bid + c.ask + p.bid + p.ask) / 2
        spr = (c.ask - c.bid) + (p.ask - p.bid)
        if mid <= 0 or spr < 0:
            continue
        rows.append({"day": day.date(), "expiry": pd.Timestamp(e).date(), "days": round(float(chain["T"].iloc[0] * 365)),
                     "monthly": not monthly.empty, "spot": round(float(spot), 2), "strike": k, "mid": round(float(mid), 2),
                     "spread": round(float(spr), 2), "spread_pct": float(spr / mid),
                     "iv": float((c.impliedVolatility + p.impliedVolatility) / 2)})
    return pd.DataFrame(rows)


def main(ticker="^XSP", min_sessions=10):
    tab = atm_straddles(ticker)
    if tab.empty:
        print(f"no {ticker} snapshots in the lake yet")
        return
    pd.set_option("display.width", 140)
    print(tab.assign(spread_pct=lambda t: (t.spread_pct * 100).round(2), iv=lambda t: (t.iv * 100).round(1)).to_string(index=False))
    med = float(tab.spread_pct.median())
    print(f"\n{len(tab)} sessions, median spread {med:.2%} of mid, mean {tab.spread_pct.mean():.2%}, "
          f"max {tab.spread_pct.max():.2%}; provisional {PROVISIONAL:.1%}")
    if os.path.exists(FROZEN):
        print(f"already frozen: {open(FROZEN).read().strip()}")
        return
    if len(tab) < min_sessions:
        print(f"not frozen: {min_sessions - len(tab)} more {ticker} sessions needed")
        return
    body = {"ticker": ticker, "sessions": int(len(tab)), "first": str(tab.day.min()), "last": str(tab.day.max()),
            "measured_median": med, "provisional": PROVISIONAL, "spread": max(med, PROVISIONAL),
            "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    with open(FROZEN, "w") as fh:
        json.dump(body, fh, indent=2)
    print(f"FROZEN at {body['spread']:.2%} -> {FROZEN}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[a.index("--ticker") + 1] if "--ticker" in a else "^XSP",
         int(a[a.index("--min-sessions") + 1]) if "--min-sessions" in a else 10)
