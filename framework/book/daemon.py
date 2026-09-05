"""One run per NYSE session: fresh bars, shadow replay, broker reconcile, ledger row, dashboard.

The run refuses unless the newest bar is the last completed NYSE session, so
a fire before Yahoo finalises the day's bar does nothing. A session already in
the ledger is not logged or traded twice. The shadow book is a replay of the
engine from LIVE_START, so a missed day is caught up by the next run rather
than by a state file; the simulated Alpaca account is a replay of the ledger's
own order rows in the same way.

Order of the safety checks, every run: KILL file (before any data is fetched),
positions reconciled against the previous ledger row (halts past 5% of equity
on any name), then limits on the post-trade book inside broker.submit. A halt
writes framework/KILL and the alpaca row is not written, so the session is
retried once a human deletes the file.

Run: ../../.venv/bin/python3 -m framework.book.daemon [--dry-run]
"""
import json
import os
import sys

import pandas as pd

from framework.book import dashboard, universe
from framework.book.allocate import LIVE_BOOK, REPORTS, load_allocations
from framework.book.broker import KILL, AlpacaBroker, Halted, ShadowBroker, killed, order_record, verify_positions
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


def sim_rows(ledger):
    """Ledger rows the simulated account replays: alpaca rows written in simulated mode."""
    if not len(ledger):
        return []
    rows = ledger[(ledger.book == "alpaca") & ledger.note.astype(str).str.startswith("simulated")]
    return [dict(r, date=r["date"].strftime("%Y-%m-%d")) for r in rows.to_dict("records")]


def previous_row(ledger, mode, session):
    """The last alpaca row before `session` written in the same mode, or None."""
    if not len(ledger):
        return None
    rows = ledger[(ledger.book == "alpaca") & ledger.note.astype(str).str.startswith(mode)
                  & (ledger.date < pd.Timestamp(session))]
    return rows.sort_values("date").iloc[-1].to_dict() if len(rows) else None


def run_once(dry_run=False, session=None, now=None):
    if killed():
        print(f"KILL file present at {KILL}; nothing fetched, nothing traded. Delete it to restart.")
        try:
            AlpacaBroker().submit([])  # paper mode cancels anything still queued, then raises
        except Halted:
            pass
        return None
    now = pd.Timestamp.now(tz=ET) if now is None else pd.Timestamp(now)
    last = last_session(now)
    session = last if session is None else pd.Timestamp(session)
    if session > last:
        raise ValueError(f"{session.date()} has not closed yet; last session is {last.date()}")
    day = session.strftime("%Y-%m-%d")
    ledger = read_ledger()
    done = set(zip(ledger.date.astype(str).str[:10], ledger.book)) if len(ledger) else set()
    if not dry_run and {(day, "shadow"), (day, "alpaca")} <= done:
        print(f"{session.date()} already in the ledger; nothing to do")
        dashboard.render()
        return []
    bars = live_bars(session)
    if bars.asof != session:
        print(f"refused: newest bar is {bars.asof.date()}, last NYSE session is {session.date()}")
        return None
    if session < last:
        print(f"catch-up run for {session.date()}; last session is {last.date()}")

    alloc = load_allocations()
    live = {k: v for k, v in alloc["weights"].items() if v > 0}
    shadow = ShadowBroker([s for s in sleeves(LIVE_BOOK) if str(s) in live], live, LIVE_START)
    shadow.run(bars)
    targets = shadow.targets()
    prices = bars.close.iloc[-1].to_dict()
    stamp = pd.Timestamp.now(tz=ET).isoformat(timespec="seconds")
    phash = universe.panel_hash(bars, HASH_START, LIVE_START)
    rows = [{"date": day, "book": "shadow", "equity": round(shadow.equity, 2),
             "gross": round(sum(abs(v) for v in shadow.positions().values()), 4),
             "positions": json.dumps(shadow.positions()), "fills": json.dumps(shadow.fills(session)),
             "targets": json.dumps(targets), "panel_hash": phash, "run_at": stamp,
             "note": f"replay from {LIVE_START}, {alloc['allocator']} allocator"}]
    print(f"{session.date()}  shadow equity {shadow.equity:,.2f}  positions {shadow.positions()}")
    print(f"targets for next open: {targets}")

    broker = AlpacaBroker(sim_rows=sim_rows(ledger), bars=bars)
    prev = previous_row(ledger, broker.mode, session)
    if broker.mode == "simulated":
        print(f"SIMULATED account (no paper keys): {broker.why}")
    try:
        equity = broker.equity()
        held_w = broker.weights()
        verify_positions(held_w, prev)
        print(f"{broker.mode} equity {equity:,.2f}, reconciled against the ledger row of "
              + (str(prev["date"])[:10] if prev else "nothing (first run)"))
        orders = broker.reconcile(targets, equity, prices)
        for o in orders:
            print(f"  {o.side.value:4s} {o.symbol:8s} " + (f"qty {o.qty}" if o.qty else f"${o.notional}")
                  + (f" limit {o.limit_price}" if getattr(o, "limit_price", None) else ""))
        if not dry_run:
            broker.submit(orders, targets, equity, prev["equity"] if prev else None)
    except Halted as e:
        print(f"HALTED: {e}\nKILL file written at {KILL}; the alpaca row for {day} is not written, delete the file and rerun")
        orders = None
    if orders is not None:
        rows.append({"date": day, "book": "alpaca", "equity": round(equity, 2),
                     "gross": round(sum(abs(v) for v in held_w.values()), 4),
                     "positions": json.dumps({k: round(v, 4) for k, v in held_w.items()}),
                     "fills": json.dumps([order_record(o) for o in orders]),
                     "targets": json.dumps(targets), "panel_hash": phash, "run_at": stamp,
                     "note": f"{broker.mode}: " + ("submitted" if not dry_run else "dry run")})
    if dry_run:
        print("dry run: nothing written")
        return rows
    n = append_ledger(rows)
    print(f"ledger: {n} rows appended -> {LEDGER}")
    print(f"dashboard -> {dashboard.render()}")
    return rows


if __name__ == "__main__":
    run_once(dry_run="--dry-run" in sys.argv)
