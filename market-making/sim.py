"""Simulated market: efficient price, exponential fill intensity, informed and uninformed flow.

One session is a fixed horizon split into n_steps. At each step the efficient price
moves, the maker posts a two-sided quote, and each side draws an order arrival whose
intensity decays exponentially in the distance from the efficient price.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Market:
    """Parameters of the simulated market. Defaults follow Avellaneda-Stoikov (2008) sec. 3.3."""
    s0: float = 100.0
    sigma: float = 2.0            # dollar volatility per unit time, quoted at s0
    horizon: float = 1.0
    n_steps: int = 500
    arrival_rate: float = 140.0   # A in lambda = A * exp(-kappa * delta)
    decay: float = 1.5            # kappa
    informed_frac: float = 0.0
    informed_lag: int = 10        # steps of foresight the informed traders have
    jump_rate: float = 0.0
    jump_sd: float = 0.0
    position_limit: int = 50

    @property
    def dt(self):
        return self.horizon / self.n_steps


@dataclass
class Draws:
    """Random input for one session, held apart from the strategy so runs can be paired."""
    price: np.ndarray
    arrival: np.ndarray
    informed: np.ndarray


@dataclass
class Session:
    """Everything one simulated session produced."""
    market: Market
    price: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    fills: list = field(default_factory=list)
    limit_steps: int = 0

    @property
    def equity(self):
        return self.cash + self.inventory * self.price

    @property
    def total_pnl(self):
        return float(self.equity[-1])


def price_path(market, rng):
    """Draw one efficient price path: GBM in logs plus optional compound Poisson jumps."""
    n, dt = market.n_steps, market.dt
    vol = market.sigma / market.s0
    steps = -0.5 * vol ** 2 * dt + vol * math.sqrt(dt) * rng.normal(size=n)
    if market.jump_rate > 0.0 and market.jump_sd > 0.0:
        hit = rng.random(n) < market.jump_rate * dt
        steps = steps + hit * rng.normal(0.0, market.jump_sd, n)
    return market.s0 * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))


def draw_session(market, seed):
    """Pre-draw the price path and the flow randomness for one session.

    Column 0 is the bid side, column 1 the ask side. Drawing up front lets every
    strategy face the same prices and the same order flow, so differences between
    them are not sampling noise.
    """
    rng = np.random.default_rng(seed)
    return Draws(price_path(market, rng),
                 rng.random((market.n_steps, 2)),
                 rng.random((market.n_steps, 2)))


def _intensity(market, delta):
    """Order arrival intensity at signed distance delta from the efficient price."""
    # A quote through the efficient price gives delta < 0; the clamp stops a pathological
    # strategy from overflowing the exponential.
    return market.arrival_rate * math.exp(-market.decay * max(delta, -20.0))


def simulate(maker, market, draws):
    """Run one session and return the price, quote, inventory and fill record."""
    s = draws.price
    n, dt = market.n_steps, market.dt
    inventory = np.zeros(n + 1)
    cash = np.zeros(n + 1)
    bids = np.full(n, np.nan)
    asks = np.full(n, np.nan)
    fills = []
    position = 0
    balance = 0.0
    limit_steps = 0

    for i in range(n):
        mid = s[i]
        bid, ask = maker.quote(mid, position, market.horizon - i * dt, market)
        if position >= market.position_limit:
            bid = None
            limit_steps += 1
        elif position <= -market.position_limit:
            ask = None
            limit_steps += 1
        bids[i] = np.nan if bid is None else bid
        asks[i] = np.nan if ask is None else ask

        for side, (quote, qty) in enumerate(((bid, 1), (ask, -1))):
            if quote is None:
                continue
            delta = (mid - quote) if qty == 1 else (quote - mid)
            if draws.arrival[i, side] >= -math.expm1(-_intensity(market, delta) * dt):
                continue
            if draws.informed[i, side] < market.informed_frac:
                ahead = s[min(i + market.informed_lag, n)]
                # An informed seller only hits my bid if the price is heading below it.
                if qty == 1 and ahead >= quote:
                    continue
                if qty == -1 and ahead <= quote:
                    continue
            position += qty
            balance -= qty * quote
            fills.append((i, qty, quote, mid))

        inventory[i + 1] = position
        cash[i + 1] = balance

    return Session(market, s, inventory, cash, bids, asks, fills, limit_steps)


def decompose(session, lag=None):
    """Split total P&L into spread capture, adverse selection cost and residual inventory P&L.

    For a fill of signed size q at price p when the efficient price is S_t, the terminal
    value of that fill is q*(S_T - p), and

        S_T - p = (S_t - p) + (S_{t+h} - S_t) + (S_T - S_{t+h}).

    The three pieces are the edge at the moment of the trade, what the efficient price
    did over the informed trader's horizon h, and everything after. The identity is exact,
    so the parts sum to the total by construction rather than by approximation.
    """
    market = session.market
    n = market.n_steps
    lag = max(market.informed_lag, 1) if lag is None else max(lag, 1)
    s = session.price
    spread = adverse = residual = 0.0
    for i, qty, price, mid in session.fills:
        ahead = s[min(i + lag, n)]
        spread += qty * (mid - price)
        adverse += -qty * (ahead - mid)
        residual += qty * (s[n] - ahead)
    return {"spread": spread, "adverse_selection": adverse, "inventory": residual,
            "total": spread - adverse + residual}


def max_drawdown(equity):
    """Largest peak-to-trough fall of the mark-to-market equity path, in dollars."""
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def summarise(session):
    """Reduce one session to the numbers run.py aggregates across sessions."""
    parts = decompose(session)
    inv = session.inventory
    buys = sum(1 for f in session.fills if f[1] == 1)
    return {"pnl": session.total_pnl,
            "max_drawdown": max_drawdown(session.equity),
            "inventory_std": float(inv.std()),
            "max_abs_inventory": float(np.abs(inv).max()),
            "end_inventory": float(inv[-1]),
            "fills": len(session.fills),
            "buys": buys,
            "sells": len(session.fills) - buys,
            "limit_steps": session.limit_steps,
            **parts}


def run_sessions(maker, market, n_sessions, seed=0):
    """Simulate n_sessions with matched randomness and return the per-session summaries."""
    return [summarise(simulate(maker, market, draw_session(market, seed + k)))
            for k in range(n_sessions)]
