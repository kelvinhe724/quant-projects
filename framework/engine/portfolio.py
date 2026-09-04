"""Positions, cash, mark-to-market and carry costs for one book."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


class Portfolio:
    def __init__(self, capital, borrow_bps=0.0, financing_spread_bps=0.0, earn_on_cash=True):
        self.cash = float(capital)
        self.positions = {}
        self.marks = {}
        self.borrow_bps = borrow_bps
        self.financing_spread_bps = financing_spread_bps
        self.earn_on_cash = earn_on_cash
        self.equity_history = []
        self.pnl_history = []
        self.carry_history = []
        self._last_value = {}
        self._trade_cash = {}

    @property
    def equity(self):
        return self.cash + sum(q * self.marks.get(n, 0.0) for n, q in self.positions.items())

    def weights(self):
        eq = self.equity
        return {n: q * self.marks.get(n, 0.0) / eq for n, q in self.positions.items() if q}

    @property
    def gross(self):
        return sum(abs(w) for w in self.weights().values())

    @property
    def net(self):
        return sum(self.weights().values())

    def apply(self, trade):
        """Book a fill: cash moves by the fill value plus commission."""
        n = trade.instrument
        self.positions[n] = self.positions.get(n, 0.0) + trade.quantity
        spent = trade.quantity * trade.fill + trade.commission
        self.cash -= spent
        self._trade_cash[n] = self._trade_cash.get(n, 0.0) + spent
        if abs(self.positions[n]) < 1e-12:
            del self.positions[n]
        self.marks[n] = trade.price

    def accrue(self, date, rate_pct):
        """Charge one day of financing on negative cash and borrow on shorts; credit cash if enabled."""
        r = (rate_pct if rate_pct is not None and np.isfinite(rate_pct) else 0.0) / 100 / TRADING_DAYS
        carry = 0.0
        if self.cash < 0:
            carry += self.cash * (r + self.financing_spread_bps / 1e4 / TRADING_DAYS)
        elif self.earn_on_cash:
            carry += self.cash * r
        short_value = sum(-q * self.marks.get(n, 0.0) for n, q in self.positions.items() if q < 0)
        carry -= short_value * self.borrow_bps / 1e4 / TRADING_DAYS
        self.cash += carry
        self.carry_history.append((date, carry))
        return carry

    def mark(self, date, closes):
        """Mark to `closes`; record equity and each instrument's P&L net of what its trades cost."""
        for n in self.positions:
            px = closes.get(n, np.nan)
            if np.isfinite(px):
                self.marks[n] = px
        value = {n: q * self.marks[n] for n, q in self.positions.items()}
        names = set(value) | set(self._last_value) | set(self._trade_cash)
        pnl = {n: value.get(n, 0.0) - self._last_value.get(n, 0.0) - self._trade_cash.get(n, 0.0)
               for n in names}
        self._last_value, self._trade_cash = value, {}
        self.equity_history.append((date, self.equity))
        self.pnl_history.append((date, pnl))
        return self.equity

    def frames(self):
        equity = pd.Series(dict(self.equity_history), name="equity")
        pnl = pd.DataFrame({d: p for d, p in self.pnl_history}).T.reindex(equity.index).fillna(0.0)
        carry = pd.Series(dict(self.carry_history), name="carry").reindex(equity.index).fillna(0.0)
        return equity, pnl, carry
