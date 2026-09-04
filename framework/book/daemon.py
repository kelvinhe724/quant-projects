"""One run per NYSE session: fresh bars, shadow replay, Alpaca reconcile, ledger row, dashboard.

The run refuses unless the newest bar is the last completed NYSE session, so
a fire before Yahoo finalises the day's bar does nothing. A session already in
the ledger is not logged or traded twice. The shadow book is a replay of the
engine from LIVE_START, so a missed day is caught up by the next run rather
than by a state file.

Run: ../../.venv/bin/python3 -m framework.book.daemon [--dry-run]
"""
import json
import os
import sys

import pandas as pd

from framework.book import dashboard, universe
from framework.book.allocate import LIVE_BOOK, REPORTS, load_allocations
from framework.book.broker import AlpacaBroker, ShadowBroker
from framework.book.strategies import sleeves
from framework.book.universe import sessions

LIVE_START = "2026-08-31"
HASH_START = "2025-08-31"  # panel hash covers this fixed year, so it moves only when history is rewritten
HISTORY_YEARS = 4
LEDGER = os.path.join(REPORTS, "ledger.csv")
ET = "America/New_York"
COLUMNS = ["date", "book", "equity", "gross", "positions", "fills", "targets", "panel_hash", "run_at", "note"]


def last_session(now=None):
    """Last NYSE session whose 16:00 ET close is at or before `now`."""
    now = pd.Timestamp.now(tz=ET) if now is None else pd.Timestamp(now).tz_convert(ET)
    local = now.tz_localize(None)
    days = sessions(local - pd.Timedelta(days=14), local)
    closes = days.tz_localize(ET) + pd.Timedelta(hours=16)
    return days[closes <= now][-1]


def live_bars(session):
    start = (session - pd.DateOffset(years=HISTORY_YEARS)).date()
    rates = universe.load_rates(refresh=True)
    newest = rates.apply(lambda s: s.last_valid_index() - pd.DateOffset(months=1))
    print("rates: newest month " + ", ".join(f"{c} {d:%Y-%m}" for c, d in newest.items()))
    return universe.load_bars(str(start), str((session + pd.Timedelta(days=1)).date()), rates)


def read_ledger(path=LEDGER):
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, parse_dates=["date"])


def append_ledger(rows, path=LEDGER):
    """Append rows whose (date, book) is not already there. Returns how many were written."""
    have = read_ledger(path)
    keys = set(zip(have.date.astype(str).str[:10], have.book))
    new = [r for r in rows if (r["date"], r["book"]) not in keys]
    if new:
        pd.DataFrame(new, columns=COLUMNS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return len(new)


def run_once(dry_run=False, session=None, now=None):
    now = pd.Timestamp.now(tz=ET) if now is None else pd.Timestamp(now)
    last = last_session(now)
    session = last if session is None else pd.Timestamp(session)
    if session > last:
        raise ValueError(f"{session.date()} has not closed yet; last session is {last.date()}")
    bars = live_bars(session)
    if bars.asof != session:
        print(f"refused: newest bar is {bars.asof.date()}, last NYSE session is {session.date()}")
        return None
    if session < last:
        print(f"catch-up run for {session.date()}; last session is {last.date()}")
    if not dry_run and session.strftime("%Y-%m-%d") in set(read_ledger().date.astype(str).str[:10]):
        print(f"{session.date()} already in the ledger; nothing to do")
        dashboard.render()
        return []

    alloc = load_allocations()
    live = {k: v for k, v in alloc["weights"].items() if v > 0}
    shadow = ShadowBroker([s for s in sleeves(LIVE_BOOK) if str(s) in live], live, LIVE_START)
    shadow.run(bars)
    targets = shadow.targets()
    prices = bars.close.iloc[-1].to_dict()
    stamp = pd.Timestamp.now(tz=ET).isoformat(timespec="seconds")
    phash = universe.panel_hash(bars, HASH_START, LIVE_START)
    rows = [{"date": session.strftime("%Y-%m-%d"), "book": "shadow", "equity": round(shadow.equity, 2),
             "gross": round(sum(abs(v) for v in shadow.positions().values()), 4),
             "positions": json.dumps(shadow.positions()), "fills": json.dumps(shadow.fills(session)),
             "targets": json.dumps(targets), "panel_hash": phash, "run_at": stamp,
             "note": f"replay from {LIVE_START}, {alloc['allocator']} allocator"}]
    print(f"{session.date()}  shadow equity {shadow.equity:,.2f}  positions {shadow.positions()}")
    print(f"targets for next open: {targets}")

    try:
        alpaca = AlpacaBroker()
    except RuntimeError as e:
        print(f"Alpaca skipped: {e}")
    else:
        equity = alpaca.equity()
        orders = alpaca.reconcile(targets, equity, prices)
        print(f"Alpaca equity {equity:,.2f}, {len(orders)} orders:")
        for o in orders:
            print(f"  {o.side.value:4s} {o.symbol:8s} " + (f"qty {o.qty}" if o.qty else f"${o.notional}"))
        if not dry_run:
            alpaca.submit(orders)
        rows.append({"date": session.strftime("%Y-%m-%d"), "book": "alpaca", "equity": round(equity, 2),
                     "gross": round(sum(abs(v) for _, v in alpaca.positions().values()) / equity, 4),
                     "positions": json.dumps({k: round(v / equity, 4) for k, (_, v) in alpaca.positions().items()}),
                     "fills": json.dumps([{"symbol": o.symbol, "side": o.side.value, "qty": o.qty,
                                           "notional": o.notional} for o in orders]),
                     "targets": json.dumps(targets), "panel_hash": phash, "run_at": stamp,
                     "note": "submitted" if not dry_run else "dry run"})
    if dry_run:
        print("dry run: nothing written")
        return rows
    n = append_ledger(rows)
    print(f"ledger: {n} rows appended -> {LEDGER}")
    print(f"dashboard -> {dashboard.render()}")
    return rows


if __name__ == "__main__":
    run_once(dry_run="--dry-run" in sys.argv)
