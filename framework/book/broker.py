"""Two books from one set of targets: the engine's simulated fills and Alpaca paper.

ShadowBroker replays the engine from a fixed start on every run, so its
positions are the Executor's next-open fills at the shadow cost model.
AlpacaBroker turns the same target weights into market orders on the paper
endpoint and nothing else; it refuses to construct against anything but
paper-api.alpaca.markets.
"""
import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from framework.book import universe
from framework.book.strategies import CAPITAL, book_config
from framework.engine import run

ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
SIGNUP = "Alpaca paper accounts need only an email: https://app.alpaca.markets/signup"
PAPER_HOST = "paper-api.alpaca.markets"
MIN_ORDER = 25.0
BUFFER = 0.10


def read_env(path=None):
    """KEY=VALUE lines from framework/.env; the process environment wins."""
    path = ENV if path is None else path
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                k, _, v = line.strip().partition("=")
                if k and not k.startswith("#"):
                    out[k] = v.strip()
    for k in ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


class ShadowBroker:
    def __init__(self, strategies, allocations, start, capital=CAPITAL):
        self.strategies, self.allocations, self.start, self.capital = strategies, allocations, start, capital
        self.results = None

    def run(self, bars):
        """Replay every live sleeve from `start` through the last bar of `bars`."""
        self.results = run(self.strategies, bars, start=self.start,
                           config=book_config(allocations=self.allocations, capital=self.capital))
        return self.results

    @property
    def equity(self):
        return float(self.results.equity.iloc[-1])

    def positions(self):
        """Weights of total equity held at the last bar, nonzero only."""
        w = self.results.weights.iloc[-1]
        return {k: float(v) for k, v in w.items() if abs(v) > 1e-9}

    def fills(self, date):
        t = self.results.trades
        t = t[t.date == pd.Timestamp(date)]
        return [{"instrument": r.instrument, "quantity": round(float(r.quantity), 4),
                 "fill": round(float(r.fill), 4), "strategy": r.strategy} for r in t.itertuples()]

    def targets(self):
        """What the whole book should hold tomorrow, as weights of total equity.

        A sleeve's fresh targets override what it holds; anything the buffer
        left alone, and a quiet sleeve, contribute what they hold, so nothing
        is re-traded that the engine did not trade.
        """
        total = self.equity
        out = {}
        for k, b in self.results.books.items():
            share = float(b["equity"].iloc[-1]) / total
            w = {**b["weights"].iloc[-1].to_dict(), **(b["pending"] or {})}
            for n, x in w.items():
                out[n] = out.get(n, 0.0) + float(x) * share
        return {n: x for n, x in out.items() if abs(x) > 1e-9}


def alpaca_symbol(name):
    return name.replace("/", "") if name in universe.CRYPTO else name


def book_symbol(symbol):
    for c in universe.CRYPTO:
        if symbol == c.replace("/", ""):
            return c
    return symbol


class AlpacaBroker:
    def __init__(self, key=None, secret=None):
        env = read_env()
        key, secret = key or env.get("ALPACA_PAPER_KEY"), secret or env.get("ALPACA_PAPER_SECRET")
        if not key or not secret:
            raise RuntimeError(f"ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET not set in {ENV}. {SIGNUP}")
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(key, secret, paper=True)
        base = self.client._base_url  # a BaseURL enum; str() of it is the member name, not the URL
        host = urlparse(getattr(base, "value", base)).hostname
        if not self.client._sandbox or host != PAPER_HOST:
            raise RuntimeError(f"refusing to trade against {host}; paper only")

    def equity(self):
        return float(self.client.get_account().equity)

    def positions(self):
        """{name: (qty, market value)} in the book's instrument names."""
        return {book_symbol(p.symbol): (float(p.qty), float(p.market_value))
                for p in self.client.get_all_positions()}

    def reconcile(self, targets, equity, prices, held=None):
        """Market orders that move `held` to `targets` (weights of `equity`).

        Longs trade by notional so fractional shares are allowed; any short
        leg trades whole shares, which Alpaca requires. A position that must
        change sign is closed first and reopened by a second order. Moves below
        $25 or below 10% of the target position are skipped, Carver's buffer.
        """
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        held = self.positions() if held is None else held
        orders = []

        def order(name, side, qty=None, notional=None):
            tif = TimeInForce.GTC if name in universe.CRYPTO else TimeInForce.DAY
            kw = {"qty": qty} if qty is not None else {"notional": round(notional, 2)}
            orders.append(MarketOrderRequest(symbol=alpaca_symbol(name), side=side, time_in_force=tif, **kw))

        for name in sorted(set(targets) | set(held)):
            px = prices.get(name, np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            want = targets.get(name, 0.0) * equity
            if want < 0 and name in universe.CRYPTO:
                want = 0.0  # unshortable: a short target flattens the coin, never leaves it long
            qty, have = held.get(name, (0.0, 0.0))
            if abs(want - have) < max(MIN_ORDER, BUFFER * abs(want)):
                continue
            if qty and np.sign(want) not in (0, np.sign(qty)):
                order(name, OrderSide.BUY if qty < 0 else OrderSide.SELL, qty=abs(qty))
                qty = have = 0.0
                if abs(want) < MIN_ORDER:
                    continue
            if want >= 0 and qty >= 0:
                delta = want - have
                if delta > 0:
                    order(name, OrderSide.BUY, notional=delta)
                elif want < MIN_ORDER and qty:
                    order(name, OrderSide.SELL, qty=qty)  # full close by quantity, so a gap down cannot oversell
                elif -delta >= MIN_ORDER:
                    order(name, OrderSide.SELL, notional=-delta)
            else:
                target_qty = -int(abs(want) // px)
                dq = target_qty - qty
                if dq < 0:
                    order(name, OrderSide.SELL, qty=int(-dq))
                elif dq > 0:
                    order(name, OrderSide.BUY, qty=int(dq))
        return orders

    def submit(self, orders):
        """Cancel anything still queued (after-hours orders wait for the open), then submit.

        A rerun after a crash between submit and the ledger write therefore
        replaces the earlier orders instead of doubling them.
        """
        self.client.cancel_orders()
        return [self.client.submit_order(o) for o in orders]
