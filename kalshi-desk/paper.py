"""Demo-exchange book: place capped fades, poll fills, settle, keep a SQLite ledger.

Usage:
  paper.py place [N]   scan, place the top N fades on the demo exchange under caps
  paper.py poll        pull fills, order status, settlements and marks
  paper.py stats       realised P&L per book with the backtest's statistics

With no demo key configured, place and poll fall through to shadow.py.
"""

import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import scanner  # sets sys.path for the research modules below
from calibration import brier  # noqa: E402
from kelly import per_bet_edge  # noqa: E402

import kalshi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "ledger.sqlite")
EXPIRE_SECONDS = 600
DEFAULT_N = 5

# The quoted-spread row of prediction-markets/reports/results.txt. That is the
# scenario this book reproduces: taker fills at the touch, 24h-ish horizon.
BACKTEST = {"scenario": "quoted spread", "n": 4223, "mean_payoff": -0.0120,
            "t": -1.28, "brier": 0.1527}

# book is shadow, demo or live. Settlements are market facts, shared by all three.
SCHEMA = """
create table if not exists orders (
  order_id text primary key, client_order_id text, env text, ticker text,
  side text, price real, count real, remaining real, cost real, mid real,
  p_model real, edge real, f real, status text, placed_at text, note text,
  book text default 'demo');
create table if not exists fills (
  fill_id text primary key, order_id text, ticker text, side text,
  cost real, count real, fee real, ts text, book text default 'demo');
create table if not exists settlements (
  ticker text primary key, result text, settled_at text);
create table if not exists marks (ts text, ticker text, mid real);
"""
BOOKS = ("shadow", "demo", "live")


def open_ledger(path=LEDGER):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    for t in ("orders", "fills"):  # ledgers made before the book column existed
        if "book" not in [c[1] for c in db.execute(f"pragma table_info({t})")]:
            db.execute(f"alter table {t} add column book text default 'demo'")
    return db


def book_for(client):
    return "live" if client.env == "prod" else "demo"


def _book_clause(book, alias=""):
    if not book:
        return "", ()
    return f" and {alias}book = ?", (book,)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def exposure(db, book=None):
    """Dollars at risk in one book (or all): resting remainders plus unsettled fills."""
    w, args = _book_clause(book)
    resting = db.execute(
        "select coalesce(sum(cost * remaining), 0) from orders "
        "where status in ('resting', 'pending')" + w, args).fetchone()[0]
    w, args = _book_clause(book, "f.")
    held = db.execute(
        "select coalesce(sum(f.cost * f.count), 0) from fills f "
        "left join settlements s on s.ticker = f.ticker where s.ticker is null" + w,
        args).fetchone()[0]
    return float(resting) + float(held)


def size_under_caps(ranked, open_dollars, max_order, max_open, n=DEFAULT_N):
    """Trim the ranked list to orders that respect both dollar caps.

    Each order is at most max_order dollars, and the running total plus what is
    already at risk never exceeds max_open. Contracts are whole numbers.
    """
    room = max_open - open_dollars
    picks = []
    for r in ranked:
        if len(picks) >= n or room <= 0:
            break
        dollars = min(r["stake"], max_order, room)
        contracts = math.floor(dollars / r["cost"])
        if contracts < 1:
            continue
        cost_total = contracts * r["cost"]
        room -= cost_total
        picks.append({**r, "contracts": contracts, "dollars": cost_total})
    return picks


def order_args(pick):
    """Translate a pick into the exchange's YES-leg bid/ask vocabulary."""
    if pick["side"] == "yes":
        return "bid", pick["ask"]
    return "ask", pick["bid"]


def record_order(db, env, pick, resp, side_book, price, book="demo"):
    remaining = float(resp.get("remaining_count", pick["contracts"]))
    db.execute(
        "insert or replace into orders values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (resp["order_id"], resp.get("client_order_id"), env, pick["ticker"],
         pick["side"], price, pick["contracts"], remaining, pick["cost"],
         pick["mid"], pick["p_model"], pick["edge"], pick["f"],
         resp.get("status") or ("resting" if remaining > 0 else "executed"),
         now_iso(), resp.get("note") or f"book {side_book}", book))
    db.commit()


def place(client, db, n=DEFAULT_N, settings=None, log=print, book=None):
    settings = settings or kalshi.load_env()
    book = book or book_for(client)
    max_order = float(settings["MAX_ORDER_DOLLARS"])
    max_open = float(settings["MAX_OPEN_DOLLARS"])
    try:
        bankroll = float(client.balance()["balance_dollars"])
    except kalshi.NoKey as e:
        print(e)
        return []
    # Scoring reads the public production book. Demo prices can be nonsense,
    # so the signal comes from prod and only the fill goes to demo.
    ranked = scanner.run(kalshi.Client("prod", settings), settings, bankroll,
                         quiet=True)
    held = {r["ticker"] for r in db.execute(
        "select distinct ticker from orders where status != 'canceled' and book = ?",
        (book,))}
    ranked = [r for r in ranked if r["ticker"] not in held]
    picks = size_under_caps(ranked, exposure(db, book), max_order, max_open, n)
    if not picks:
        log("nothing to place: no positive-edge candidate fits under the caps")
        return []
    placed = []
    for p in picks:
        side_book, price = order_args(p)
        log(f"{book} order: {p['ticker']} buy {p['side']} x{p['contracts']} "
            f"at cost {p['cost']:.2f} (book {side_book} @ {price:.2f}) "
            f"edge {p['edge']:+.3f} f {p['f']:.3f}")
        resp = client.place_order(
            p["ticker"], side_book, price, p["contracts"],
            expiration_ts=time.time() + EXPIRE_SECONDS)
        record_order(db, client.env, p, resp, side_book, price, book)
        placed.append(resp["order_id"])
        log(f"  placed {resp['order_id']} filled {resp.get('fill_count')} "
            f"remaining {resp.get('remaining_count')}")
    return placed


def settle(client, db, book=None):
    """Settle held contracts on the public market result, mark the rest at the mid.

    Idempotent: settlements is keyed by ticker, so a second pass changes nothing.
    """
    w, args = _book_clause(book, "f.")
    settled = marked = 0
    for (ticker,) in db.execute(
            "select distinct f.ticker from fills f left join settlements s "
            "on s.ticker = f.ticker where s.ticker is null" + w, args).fetchall():
        m = client.market(ticker)
        if m.get("result") in ("yes", "no"):
            db.execute("insert or ignore into settlements values (?,?,?)",
                       (ticker, m["result"], m.get("settlement_ts") or now_iso()))
            settled += 1
        else:
            q = kalshi.quote(m)
            if q:
                db.execute("insert into marks values (?,?,?)",
                           (now_iso(), ticker, (q[0] + q[1]) / 2))
                marked += 1
    db.commit()
    return settled, marked


def poll(client, db, log=print):
    """Sync fills, order status, settlements and marks for this client's book."""
    book = book_for(client)
    ours = {r["order_id"]: r for r in db.execute(
        "select * from orders where book = ?", (book,))}
    if not ours:
        log("ledger is empty")
        return
    new_fills = 0
    for f in client.fills():
        if f["order_id"] not in ours or db.execute(
                "select 1 from fills where fill_id = ?", (f["fill_id"],)).fetchone():
            continue
        side = f.get("outcome_side") or ("yes" if f.get("book_side") == "bid" else "no")
        cost = float(f["yes_price_dollars"] if side == "yes" else f["no_price_dollars"])
        db.execute("insert into fills values (?,?,?,?,?,?,?,?,?)",
                   (f["fill_id"], f["order_id"], f["ticker"], side, cost,
                    float(f["count_fp"]), float(f.get("fee_cost") or 0),
                    f.get("created_time") or now_iso(), book))
        new_fills += 1
    for oid, row in ours.items():
        if row["status"] in ("resting", "pending"):
            o = client.get(f"/portfolio/orders/{oid}", auth=True)["order"]
            db.execute("update orders set status = ?, remaining = ? where order_id = ?",
                       (o["status"], float(o["remaining_count_fp"]), oid))
    db.commit()
    settled, marked = settle(client, db, book)
    log(f"{new_fills} new fills, {settled} settlements, {marked} marks")


def settled_bets(db, book=None):
    """One row per settled fill in the backtest's bet vocabulary."""
    w, args = _book_clause(book, "f.")
    rows = db.execute(
        "select f.ticker, f.side, f.cost, f.count, f.fee, o.mid, s.result, s.settled_at "
        "from fills f join settlements s on s.ticker = f.ticker "
        "left join orders o on o.order_id = f.order_id where 1" + w +
        " order by s.settled_at", args).fetchall()
    bets = pd.DataFrame([dict(r) for r in rows])
    if bets.empty:
        return bets
    win = bets["side"] == bets["result"]
    bets["outcome"] = (bets["result"] == "yes").astype(int)
    bets["payoff"] = ((1 - bets["cost"]) / bets["cost"]).where(win, -1.0)
    bets["pnl"] = bets["payoff"] * bets["cost"] * bets["count"]
    bets["pnl_net"] = bets["pnl"] - bets["fee"]
    return bets


def stats(db, book=None):
    bets = settled_bets(db, book)
    out = {"n": 0, "mean_payoff": float("nan"), "t": float("nan"),
           "brier": float("nan"), "pnl": 0.0, "pnl_net": 0.0, "staked": 0.0}
    if bets.empty:
        return out
    edge = per_bet_edge(bets)
    out.update(n=int(edge["n"]), mean_payoff=float(edge["mean_payoff"]),
               t=float(edge["t"]), pnl=float(bets["pnl"].sum()),
               pnl_net=float(bets["pnl_net"].sum()),
               staked=float((bets["cost"] * bets["count"]).sum()))
    if len(bets) >= 2 and bets["mid"].notna().all():
        out["brier"] = float(brier(bets["mid"], bets["outcome"])["brier"])
    return out


def equity_curve(db, book=None):
    bets = settled_bets(db, book)
    if bets.empty:
        return []
    cum = bets.groupby("settled_at")["pnl_net"].sum().cumsum()
    return [{"ts": t, "equity": float(v)} for t, v in cum.items()]


def print_stats(db, books=BOOKS):
    for b in books:
        s = stats(db, b)
        print(f"{b:<8} n={s['n']}  mean payoff per $ {s['mean_payoff']:+.4f}  "
              f"t={s['t']:+.2f}  Brier {s['brier']:.4f}  "
              f"P&L {s['pnl']:+.2f} gross / {s['pnl_net']:+.2f} net on "
              f"{s['staked']:.2f} staked")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "stats"
    settings = kalshi.load_env()
    db = open_ledger()
    demo = kalshi.Client("demo", settings)
    if cmd in ("place", "poll") and not demo.has_key():
        import shadow
        print("no demo key configured: running the shadow book instead "
              "(public data, simulated fills)")
        n = int(argv[2]) if len(argv) > 2 else DEFAULT_N
        shadow.place(db, settings, n) if cmd == "place" else shadow.settle(db, settings)
    elif cmd == "place":
        n = int(argv[2]) if len(argv) > 2 else DEFAULT_N
        place(demo, db, n, settings)
    elif cmd == "poll":
        poll(demo, db)
    else:
        print_stats(db)
        print(f"backtest n={BACKTEST['n']}  mean payoff per $ "
              f"{BACKTEST['mean_payoff']:+.4f}  t={BACKTEST['t']:+.2f}  "
              f"Brier {BACKTEST['brier']:.4f}  ({BACKTEST['scenario']})")


if __name__ == "__main__":
    main(sys.argv)
