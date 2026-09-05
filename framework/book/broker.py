"""The broker layer: one set of targets, several books, safety in the layer.

ShadowBroker replays the engine from a fixed start on every run, so its
positions are the Executor's next-open fills at the shadow cost model.
AlpacaBroker turns the same target weights into orders on the paper endpoint;
without paper keys it runs a clearly labelled SIMULATED account instead,
rebuilt from the ledger's own order rows on every run (no state file).

Safety lives here, not in the strategy:
  * KILL file (framework/KILL): submit() refuses while it exists, and every
    halt the layer raises writes the file, so trading stays off until a human
    deletes it.
  * Limits (max gross, max per name, max daily loss) are checked on the
    post-trade book before anything is sent.
  * Live adapters (AlpacaLive, IBKRBroker, KalshiLive) are dormant: they
    raise before any live key is read unless framework/.env has
    LIVE_ENABLED=true AND the caller passes today's confirmation token.
    The daemon never constructs them.
"""
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from framework.book import universe
from framework.book.strategies import CAPITAL, book_config
from framework.engine import run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV = os.path.join(ROOT, ".env")
KILL = os.path.join(ROOT, "KILL")
SIGNUP = "Alpaca paper accounts need only an email: https://app.alpaca.markets/signup"
PAPER_HOST = "paper-api.alpaca.markets"
LIVE_HOST = "api.alpaca.markets"
PAPER_KEYS = ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET")
MIN_ORDER = 25.0
BUFFER = 0.10
RECON_TOL = 0.05  # weight of equity, per instrument, before a discrepancy halts the book


@dataclass(frozen=True)
class Limits:
    max_gross: float = 3.0        # sum |weight|, the engine's own gross cap
    max_per_name: float = 0.5     # |weight| of any one instrument
    max_daily_loss: float = 0.03  # equity below (1 - this) x the previous run's equity halts


class Halted(RuntimeError):
    """Trading is off: the KILL file exists or a limit wrote it."""


class LiveDisabled(RuntimeError):
    """A live adapter was asked for without both locks open."""


def read_env(keys=PAPER_KEYS, path=None):
    """Only the named KEY=VALUE lines from framework/.env; the process environment wins.

    A line whose key is not asked for is skipped before its value is parsed,
    so a live key on disk is never held in memory by a paper run.
    """
    path = ENV if path is None else path
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                k = line.partition("=")[0].strip()
                if k in keys:
                    out[k] = line.strip().partition("=")[2].strip()
    for k in keys:
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


def killed(path=None):
    return os.path.exists(KILL if path is None else path)


def halt(reason, path=None):
    """Write the KILL file and raise. A human deletes the file to restart."""
    path = KILL if path is None else path
    with open(path, "a") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {reason}\n")
    raise Halted(reason)


def check_limits(targets, equity, ref_equity=None, limits=Limits()):
    """Halt if the post-trade book or today's loss breaks a limit. Returns the gross."""
    if targets is None:
        raise ValueError("orders without targets cannot be limit-checked; nothing sent")
    gross = sum(abs(v) for v in targets.values())
    if gross > limits.max_gross + 1e-9:
        halt(f"gross {gross:.2f} above limit {limits.max_gross}")
    for name, w in targets.items():
        if abs(w) > limits.max_per_name + 1e-9:
            halt(f"{name} weight {w:+.2f} above per-name limit {limits.max_per_name}")
    if ref_equity and equity < ref_equity * (1 - limits.max_daily_loss):
        halt(f"equity {equity:,.2f} is {equity / ref_equity - 1:+.2%} on {ref_equity:,.2f}, "
             f"past the daily loss limit {limits.max_daily_loss:.0%}")
    return gross


def expected_weights(prev_row):
    """What the account should hold now, from the ledger row of the previous run.

    A name the previous run sent an order for should sit at that run's target;
    everything else at what the account held then.
    """
    held = json.loads(prev_row["positions"])
    targets = json.loads(prev_row["targets"])
    sent = {book_symbol(f["symbol"]) for f in json.loads(prev_row["fills"])}
    return {**held, **{n: targets.get(n, 0.0) for n in sent}}


def discrepancies(held, expected, tol=RECON_TOL):
    """{name: (held, expected)} for every name off by more than tol, weights of equity."""
    out = {}
    for n in sorted(set(held) | set(expected)):
        h, e = held.get(n, 0.0), expected.get(n, 0.0)
        if abs(h - e) > tol:
            out[n] = (round(h, 4), round(e, 4))
    return out


def verify_positions(held, prev_row, tol=RECON_TOL):
    """Reconcile the account against the ledger; halt if anything is off by more than tol."""
    if prev_row is None:
        return {}
    off = discrepancies(held, expected_weights(prev_row), tol)
    if off:
        halt("position reconciliation failed: " + ", ".join(f"{n} held {h:+.4f} expected {e:+.4f}"
                                                            for n, (h, e) in off.items()))
    return off


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


def order_record(o):
    """The ledger's row for one order request: what the next run replays or reconciles."""
    return {"symbol": o.symbol, "side": o.side.value, "qty": o.qty, "notional": o.notional,
            "limit": getattr(o, "limit_price", None)}


def build_orders(targets, equity, prices, held, limit_prices=None):
    """Orders that move `held` ({name: (qty, market value)}) to `targets` (weights of `equity`).

    Longs trade by notional so fractional shares are allowed; any short leg
    trades whole shares, which Alpaca requires. A position that must change
    sign is closed first and reopened by a second order. Moves below $25 or
    below 10% of the target position are skipped, Carver's buffer. A name in
    `limit_prices` goes as a limit order at that price; everything else is a
    market order. Nothing here talks to a broker.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

    limit_prices = limit_prices or {}
    orders = []

    def order(name, side, qty=None, notional=None):
        tif = TimeInForce.GTC if name in universe.CRYPTO else TimeInForce.DAY
        kw = {"qty": qty} if qty is not None else {"notional": round(notional, 2)}
        if name in limit_prices:
            if qty is None:  # limit orders need a quantity
                kw = {"qty": round(notional / limit_prices[name], 6)}
            orders.append(LimitOrderRequest(symbol=alpaca_symbol(name), side=side, time_in_force=tif,
                                            limit_price=round(limit_prices[name], 2), **kw))
        else:
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


class SimulatedAccount:
    """Stand-in for the paper account when there are no keys.

    Rebuilt from the ledger's simulated rows on every run, the way the shadow
    book is rebuilt from the engine: an order recorded on the row for session
    D fills at the open of the next session in `bars` (Alpaca paper's own
    convention for a market order queued after the close, without its random
    partials); a crypto order fills when it was submitted (Alpaca crypto is
    24/7 and a market order fills at once), through l2fill against the lake's
    L2 stream at that second, or the daily bar's open if the lake has no row.
    No commission, no spread, no borrow: that is what Alpaca paper charges.
    """

    def __init__(self, rows, bars, capital=CAPITAL):
        self.bars, self.cash, self.qty, self.pending, self.fills = bars, float(capital), {}, [], []
        cal = bars.index  # the view's own bars, not the calendar known ahead
        for row in sorted(rows, key=lambda r: str(r["date"])):
            date = pd.Timestamp(row["date"])
            later = cal[cal > date]
            for o in json.loads(row["fills"]) if isinstance(row["fills"], str) else row["fills"]:
                name = book_symbol(o["symbol"])
                sign = 1.0 if o["side"] == "buy" else -1.0
                if name in universe.CRYPTO:
                    px, q = self._crypto_fill(name, o, sign, row.get("run_at"))
                elif len(later):
                    px, q = self._etf_fill(name, o, sign, later[0])
                else:
                    self.pending.append(o)
                    continue
                if q == 0:
                    self.pending.append(o)  # a limit that never traded
                    continue
                self.qty[name] = self.qty.get(name, 0.0) + q
                self.cash -= q * px
                self.fills.append({"date": str(date.date()), "instrument": name, "quantity": round(q, 6), "fill": round(px, 4)})
        self.qty = {k: v for k, v in self.qty.items() if abs(v) > 1e-9}

    def _etf_fill(self, name, o, sign, day):
        opn = float(self.bars.open.loc[day, name])
        if o.get("limit"):
            lim = float(o["limit"])
            lo, hi = float(self.bars.low.loc[day, name]), float(self.bars.high.loc[day, name])
            # ponytail: a day bar cannot say when the limit traded; fill at the better of open and limit if touched
            if (sign > 0 and lo > lim) or (sign < 0 and hi < lim):
                return opn, 0.0
            opn = min(opn, lim) if sign > 0 else max(opn, lim)
        q = float(o["qty"]) if o.get("qty") else float(o["notional"]) / opn
        return opn, sign * q

    def _crypto_fill(self, name, o, sign, run_at):
        from framework.book import l2fill

        side = "buy" if sign > 0 else "sell"
        ts = pd.Timestamp(run_at) if run_at else None
        if ts is not None and ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        res = None
        if ts is not None:
            try:
                if o.get("limit"):
                    res = l2fill.limit_fill(name, side, float(o["qty"]), float(o["limit"]), ts)
                else:
                    res = l2fill.market_fill(name, side, ts, qty=o.get("qty"), notional=o.get("notional"))
            except (FileNotFoundError, KeyError, ValueError, ImportError):
                res = None
        if res is None or res["filled"] == 0:
            day = self.bars.index[self.bars.index >= pd.Timestamp(str(run_at)[:10])]
            if not len(day):
                return 0.0, 0.0
            px = float(self.bars.open.loc[day[0], name])
            q = float(o["qty"]) if o.get("qty") else float(o["notional"]) / px
            return px, sign * q
        return res["price"], sign * res["filled"]

    def positions(self, prices):
        return {n: (q, q * float(prices[n])) for n, q in self.qty.items() if n in prices}

    def equity(self, prices):
        return self.cash + sum(v for _, v in self.positions(prices).values())


class _AlpacaAccount:
    """The read and submit calls shared by the paper and (dormant) live adapters."""

    def _connect(self, key, secret, paper, host):
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(key, secret, paper=paper)
        base = self.client._base_url  # a BaseURL enum; str() of it is the member name, not the URL
        got = urlparse(getattr(base, "value", base)).hostname
        if bool(self.client._sandbox) != paper or got != host:
            raise RuntimeError(f"refusing to trade against {got}; expected {host}")

    def equity(self):
        return float(self.client.get_account().equity)

    def positions(self):
        """{name: (qty, market value)} in the book's instrument names."""
        return {book_symbol(p.symbol): (float(p.qty), float(p.market_value))
                for p in self.client.get_all_positions()}

    def _send(self, orders):
        """Submit in order. submit() has already cancelled what was queued."""
        return [self.client.submit_order(o) for o in orders]


class AlpacaBroker(_AlpacaAccount):
    """Paper account when framework/.env has paper keys, SIMULATED account otherwise. Never live."""

    def __init__(self, key=None, secret=None, sim_rows=None, bars=None, limits=None):
        env = read_env(PAPER_KEYS)
        key, secret = key or env.get("ALPACA_PAPER_KEY"), secret or env.get("ALPACA_PAPER_SECRET")
        self.limits = limits or Limits()
        if key and secret:
            self.mode = "paper"
            self._connect(key, secret, True, PAPER_HOST)
        else:
            self.mode = "simulated"
            self.why = f"ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET not set in {ENV}. {SIGNUP}"
            self.sim = SimulatedAccount(sim_rows or [], bars) if bars is not None else None
            self.prices = bars.close.iloc[-1].to_dict() if bars is not None else {}

    def equity(self):
        return self.sim.equity(self.prices) if self.mode == "simulated" else super().equity()

    def positions(self):
        return self.sim.positions(self.prices) if self.mode == "simulated" else super().positions()

    def weights(self):
        eq = self.equity()
        return {n: v / eq for n, (_, v) in self.positions().items()}

    def reconcile(self, targets, equity, prices, held=None, limit_prices=None):
        """Orders that move the account to `targets`; see build_orders."""
        return build_orders(targets, equity, prices, self.positions() if held is None else held, limit_prices)

    def submit(self, orders, targets=None, equity=None, ref_equity=None):
        """The only way out of the layer. Cancel what is queued, kill file, limits, then the wire.

        The cancel comes first so that a kill or a limit halt leaves nothing
        queued from a run that crashed between submit and the ledger write
        (after-hours orders wait for the open), and a clean rerun replaces those
        orders instead of doubling them. Orders without targets are refused:
        nothing leaves the layer unchecked. In simulated mode nothing is sent:
        the returned orders go into the ledger row and fill on the next run's replay.
        """
        paper = self.mode == "paper"
        if paper:
            self.client.cancel_orders()
        if killed():
            raise Halted(f"KILL file present at {KILL}; nothing submitted")
        if orders or targets is not None:
            check_limits(targets, equity, ref_equity, self.limits)
        return self._send(orders) if paper else orders


# Dormant live adapters. Two locks, both checked before any live key or socket is touched.

def live_token(today=None):
    today = dt.date.today() if today is None else today
    return f"LIVE {today:%Y-%m-%d}"


def live_gate(confirm, today=None):
    """Raise unless LIVE_ENABLED=true in framework/.env AND `confirm` is today's token AND no KILL file."""
    if killed():
        raise Halted(f"KILL file present at {KILL}")
    if read_env(("LIVE_ENABLED",)).get("LIVE_ENABLED", "").strip().lower() != "true":
        raise LiveDisabled(f"live trading is OFF: LIVE_ENABLED is not true in {ENV}")
    if confirm != live_token(today):
        raise LiveDisabled(f"no per-run confirmation: pass confirm={live_token(today)!r} (valid today only)")


class AlpacaLive(_AlpacaAccount):
    """Real-money Alpaca. Dormant: the gate runs before the live key is read."""

    def __init__(self, confirm=None, today=None, limits=None):
        live_gate(confirm, today)
        env = read_env(("ALPACA_LIVE_KEY", "ALPACA_LIVE_SECRET"))
        if not env.get("ALPACA_LIVE_KEY") or not env.get("ALPACA_LIVE_SECRET"):
            raise RuntimeError(f"ALPACA_LIVE_KEY / ALPACA_LIVE_SECRET not set in {ENV}")
        self.mode, self.limits = "live", limits or Limits()
        self._connect(env["ALPACA_LIVE_KEY"], env["ALPACA_LIVE_SECRET"], False, LIVE_HOST)

    def submit(self, orders, targets=None, equity=None, ref_equity=None):
        self.client.cancel_orders()  # first, as in AlpacaBroker.submit: a halt leaves nothing queued
        if killed():
            raise Halted(f"KILL file present at {KILL}")
        if orders or targets is not None:
            check_limits(targets, equity, ref_equity, self.limits)
        return self._send(orders)


class IBKRBroker:
    """Interactive Brokers through ib_insync, TWS/Gateway paper port 7497 by default (7496 live).

    Dormant and untested against a gateway: ib_insync is not installed in the
    venv, and the import happens only after the gate. Written to the ib_insync
    API as documented; treat the first connection as a test.
    """

    def __init__(self, confirm=None, today=None, host="127.0.0.1", port=7497, client_id=7, limits=None):
        live_gate(confirm, today)
        try:
            import ib_insync
        except ImportError as e:
            raise ImportError("ib_insync is not installed; `.venv/bin/pip install ib_insync` when a gateway exists") from e
        self.ib, self.limits, self.mode = ib_insync.IB(), limits or Limits(), "live" if port == 7496 else "ib-paper"
        self.ib.connect(host, port, clientId=client_id)

    def equity(self):
        return float(next(v.value for v in self.ib.accountValues() if v.tag == "NetLiquidation" and v.currency == "USD"))

    def positions(self):
        return {p.contract.symbol: (float(p.position), float(p.position) * float(p.avgCost)) for p in self.ib.positions()}

    def submit(self, orders, targets=None, equity=None, ref_equity=None):
        """`orders` are the alpaca request objects from build_orders; mapped to SMART/USD stocks."""
        import ib_insync

        if killed():
            self.ib.reqGlobalCancel()
            raise Halted(f"KILL file present at {KILL}")
        if orders or targets is not None:
            check_limits(targets, equity, ref_equity, self.limits)
        out = []
        for o in orders:
            qty = float(o.qty) if o.qty else None
            if qty is None:
                raise ValueError(f"IBKR needs a quantity, not notional, for {o.symbol}")
            side = o.side.value.upper()
            lim = getattr(o, "limit_price", None)
            order = ib_insync.LimitOrder(side, qty, float(lim)) if lim else ib_insync.MarketOrder(side, qty)
            out.append(self.ib.placeOrder(ib_insync.Stock(o.symbol, "SMART", "USD"), order))
        return out


class KalshiLive:
    """Production Kalshi through kalshi-desk's client. Dormant: three locks, since the desk has its own."""

    def __init__(self, confirm=None, today=None, limits=None):
        live_gate(confirm, today)
        desk = os.path.join(os.path.dirname(ROOT), "kalshi-desk")
        if desk not in sys.path:
            sys.path.insert(0, desk)
        import kalshi

        settings = kalshi.load_env()
        if settings.get("LIVE_TRADING", "false").strip().lower() != "true":
            raise LiveDisabled("kalshi-desk/.env LIVE_TRADING is not true")
        self.client, self.limits, self.mode = kalshi.Client("prod", settings), limits or Limits(), "live"
        if not self.client.has_key():
            raise RuntimeError(kalshi.setup_hint("prod"))

    def equity(self):
        return float(self.client.balance().get("balance", 0)) / 100.0

    def positions(self):
        return {p["ticker"]: (float(p.get("position", 0)), float(p.get("market_exposure", 0)) / 100.0)
                for p in self.client.positions().get("market_positions", [])}

    def submit(self, orders, targets=None, equity=None, ref_equity=None):
        """`orders` are dicts {ticker, side, price, count}; GTC limits through the desk's client."""
        if killed():
            raise Halted(f"KILL file present at {KILL}")
        if orders or targets is not None:
            check_limits(targets, equity, ref_equity, self.limits)
        return [self.client.place_order(o["ticker"], o["side"], o["price"], o["count"]) for o in orders]
