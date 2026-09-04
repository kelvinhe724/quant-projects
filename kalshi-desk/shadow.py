"""Shadow book: the same strategy with no account. Public data in, simulated fills out.

Scan, sizing and caps are paper.place unchanged. The only simulated part is the
fill: the public orderbook is re-read at placement, the touch is recorded, and a
fill is written at the quoted ask (YES) or one minus the quoted bid (NO) when the
order would have crossed the touch at its full size. Settlement is the public
market result. Nothing here needs a key.

Usage:
  shadow.py place [N]        scan, simulate the top N fades, record the touch
  shadow.py settle           settle held contracts on the public result, mark the rest
  shadow.py stats
  shadow.py run --every 15m  place then settle, forever, sleeping in between
"""

import sys
import time
import uuid

import kalshi
import paper

BOOK = "shadow"


def fill(side, price, count, touch):
    """Cost per contract if a limit at price for count crosses the touch, else None.

    side is the exchange's: bid buys YES at the ask, ask buys NO at 1 - bid.
    ponytail: all-or-nothing at the touch size. Walk the book if partials matter.
    """
    if touch is None:
        return None
    if side == "bid":
        ok = touch["ask"] <= price + 1e-9 and touch["ask_size"] >= count
        cost = touch["ask"]
    else:
        ok = touch["bid"] >= price - 1e-9 and touch["bid_size"] >= count
        cost = 1 - touch["bid"]
    return cost if ok else None


class Book(kalshi.Client):
    """Quacks like an exchange client for paper.place; fills against the public book."""

    def __init__(self, db, settings=None):
        super().__init__("prod", settings)
        self.db = db

    def _headers(self, method, url):
        # The shadow book only reads public endpoints. Even with a prod key in
        # .env it must never sign a request, so the signing path is cut here.
        raise RuntimeError("shadow book never signs a request")

    def balance(self):
        return {"balance_dollars": self.settings["SHADOW_BANKROLL"]}

    def fills(self, **params):
        return []

    def place_order(self, ticker, side, price, count, client_order_id=None,
                    expiration_ts=None):
        touch = kalshi.touch(self.orderbook(ticker, depth=5))
        ts = paper.now_iso()
        order_id = f"shadow-{ticker}-{uuid.uuid4().hex[:8]}"
        cost = fill(side, price, count, touch)
        note = "touch " + (f"{touch['bid']:.2f}x{touch['bid_size']:.0f} / "
                           f"{touch['ask']:.2f}x{touch['ask_size']:.0f}" if touch else "none")
        if cost is not None:
            # One fill per ticker per book, ever. The primary key makes a re-run a no-op.
            self.db.execute("insert or ignore into fills values (?,?,?,?,?,?,?,?,?)",
                            (f"shadow-{ticker}", order_id, ticker,
                             "yes" if side == "bid" else "no", cost, count, 0.0, ts, BOOK))
        return {"order_id": order_id, "client_order_id": client_order_id,
                "fill_count": count if cost is not None else 0,
                "remaining_count": 0,
                "status": "executed" if cost is not None else "canceled",
                "note": note if cost is not None else "not marketable, " + note}


def place(db, settings=None, n=paper.DEFAULT_N, log=print):
    settings = settings or kalshi.load_env()
    return paper.place(Book(db, settings), db, n, settings, log=log, book=BOOK)


def settle(db, settings=None, log=print):
    settled, marked = paper.settle(kalshi.Client("prod", settings), db, BOOK)
    log(f"shadow: {settled} settlements, {marked} marks")
    return settled, marked


def seconds(spec):
    unit = {"s": 1, "m": 60, "h": 3600}.get(spec[-1], None)
    return int(spec[:-1]) * unit if unit else int(spec)


def run(every, n=paper.DEFAULT_N):
    settings = kalshi.load_env()
    db = paper.open_ledger()
    while True:
        try:
            place(db, settings, n)
            settle(db, settings)
        except Exception as e:  # a bad cycle must not stop the accumulation
            print(f"cycle failed: {type(e).__name__}: {e}", file=sys.stderr)
        time.sleep(every)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "stats"
    if cmd == "run":
        every = seconds(argv[argv.index("--every") + 1]) if "--every" in argv else 900
        print(f"shadow book every {every}s, ctrl-c to stop")
        run(every)
        return
    settings = kalshi.load_env()
    db = paper.open_ledger()
    if cmd == "place":
        place(db, settings, int(argv[2]) if len(argv) > 2 else paper.DEFAULT_N)
    elif cmd == "settle":
        settle(db, settings)
    else:
        paper.print_stats(db, (BOOK,))


if __name__ == "__main__":
    main(sys.argv)
