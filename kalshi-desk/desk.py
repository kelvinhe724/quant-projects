"""Local dashboard on :6161. Stdlib http.server, one JSON endpoint, one page."""

import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import kalshi
import paper
import scanner

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 6161


def account(settings):
    client = kalshi.Client(settings["KALSHI_ENV"], settings)
    try:
        bal = client.balance()
        return {
            "ok": True,
            "env": client.env,
            "balance": float(bal["balance_dollars"]),
            "portfolio_value": float(bal.get("portfolio_value", 0)) / 100,
            "positions": [{
                "ticker": p["ticker"],
                "position": float(p["position_fp"]),
                "exposure": float(p["market_exposure_dollars"]),
                "realized": float(p["realized_pnl_dollars"]),
            } for p in client.positions()],
            "fills": [{
                "ticker": f["ticker"],
                "side": f.get("outcome_side"),
                "count": float(f["count_fp"]),
                "price": float(f["yes_price_dollars"] if f.get("outcome_side") == "yes"
                               else f["no_price_dollars"]),
                "ts": f.get("created_time"),
            } for f in client.fills(limit=20)][:20],
        }
    except kalshi.NoKey as e:
        return {"ok": False, "env": client.env, "hint": str(e)}
    except Exception as e:
        return {"ok": False, "env": client.env, "hint": f"Kalshi error: {e}"}


def book_stats(db, book):
    # NaN is not JSON; the browser's parser rejects it.
    st = {k: (None if isinstance(v, float) and math.isnan(v) else v)
          for k, v in paper.stats(db, book).items()}
    st["orders"] = db.execute("select count(*) from orders where book = ?", (book,)).fetchone()[0]
    st["fills"] = db.execute("select count(*) from fills where book = ?", (book,)).fetchone()[0]
    st["exposure"] = paper.exposure(db, book)
    st["equity"] = paper.equity_curve(db, book)
    return st


def state():
    settings = kalshi.load_env()
    db = paper.open_ledger()
    rows = lambda q: [dict(r) for r in db.execute(q)]
    books = {b: book_stats(db, b) for b in paper.BOOKS}
    # Shadow is the default book until a demo key exists; then demo, with shadow
    # running beside it as the no-impact control.
    book = "demo" if kalshi.Client("demo", settings).has_key() else "shadow"
    return {
        "live": settings["LIVE_TRADING"].strip().lower() == "true",
        "env": settings["KALSHI_ENV"],
        "book": book,
        "books": books,
        "caps": {"order": float(settings["MAX_ORDER_DOLLARS"]),
                 "open": float(settings["MAX_OPEN_DOLLARS"]),
                 "k": float(settings["KELLY_FRACTION"])},
        "account": account(settings),
        "scan": scanner.latest_scan(),
        "ledger": {
            "orders": rows("select * from orders order by placed_at desc"),
            "fills": rows("select * from fills order by ts desc"),
            "settlements": rows("select * from settlements order by settled_at desc"),
            "exposure": books[book]["exposure"],
            "equity": books[book]["equity"],
        },
        "forward": books[book],
        "backtest": paper.BACKTEST,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(state(), default=str).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                body = fh.read()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


if __name__ == "__main__":
    print(f"kalshi desk on http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
