"""Overlay applied to every target before it reaches execution.

Order: vol target, gross and per-instrument limits, drawdown control, kill
switch, then the position buffer against what the book already holds.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class RiskConfig:
    target_vol: float = None
    vol_window: int = 60
    max_leverage: float = 10.0
    max_gross: float = None
    max_weight: float = None
    dd_threshold: float = None
    dd_scale: float = 0.5
    dd_recover: float = None
    kill_dd: float = None
    buffer: float = None


class RiskManager:
    def __init__(self, config=None):
        self.cfg = config or RiskConfig()
        self.reduced = False
        self.killed = False
        self.last_scale = 1.0
        self.targets = None

    def ex_ante_vol(self, weights, bars):
        """Annualised portfolio vol from the trailing covariance of daily returns."""
        names = [n for n, w in weights.items() if w]
        if not names:
            return 0.0
        rets = bars.close[names].pct_change().tail(self.cfg.vol_window).dropna(how="all")
        if len(rets) < max(5, self.cfg.vol_window // 3):
            return np.nan
        cov = rets.cov().fillna(0.0).to_numpy()
        w = np.array([weights[n] for n in names])
        return float(np.sqrt(max(w @ cov @ w, 0.0) * TRADING_DAYS))

    def drawdown(self, equity):
        if not equity:
            return 0.0
        peak = max(e for _, e in equity)
        return equity[-1][1] / peak - 1

    def update_state(self, equity_history):
        """Move the drawdown-cut and kill-switch flags from the book's equity so far."""
        c = self.cfg
        dd = self.drawdown(equity_history)
        if c.dd_threshold is not None:
            recover = c.dd_recover if c.dd_recover is not None else c.dd_threshold / 2
            if dd <= -c.dd_threshold:
                self.reduced = True
            elif dd >= -recover:
                self.reduced = False
        if c.kill_dd is not None and dd <= -c.kill_dd:
            self.killed = True

    def breach(self, held):
        """True when the held weights are through the gross or per-instrument limit."""
        c = self.cfg
        gross = sum(abs(x) for x in held.values())
        return ((c.max_gross is not None and gross > c.max_gross)
                or (c.max_weight is not None and any(abs(x) > c.max_weight for x in held.values())))

    def apply(self, targets, bars, equity_history, held=None):
        """Return the weights to send. equity_history is the book's (date, equity) list so far.

        With a buffer and `held` (the book's current weights), an instrument
        whose held weight is inside target * (1 +/- buffer) is left out, so the
        overlay can rescale every target daily without churning positions that
        are close enough. A gross or per-instrument breach, a change in the
        drawdown cut or a kill sends every target.
        """
        c = self.cfg
        self.targets = dict(targets)
        before = (self.reduced, self.killed)
        w = dict(targets)
        if c.target_vol:
            vol = self.ex_ante_vol(w, bars)
            scale = 1.0 if not np.isfinite(vol) or vol == 0 else min(c.target_vol / vol, c.max_leverage)
            self.last_scale = scale
            w = {n: x * scale for n, x in w.items()}
        if c.max_weight is not None:
            w = {n: float(np.clip(x, -c.max_weight, c.max_weight)) for n, x in w.items()}
        gross = sum(abs(x) for x in w.values())
        if c.max_gross is not None and gross > c.max_gross:
            w = {n: x * c.max_gross / gross for n, x in w.items()}
        self.update_state(equity_history)
        if self.reduced:
            w = {n: x * c.dd_scale for n, x in w.items()}
        if self.killed:
            w = {n: 0.0 for n in w}
        if held is not None and c.buffer and (self.reduced, self.killed) == before and not self.breach(held):
            w = {n: x for n, x in w.items() if abs(held.get(n, 0.0) - x) > c.buffer * abs(x)}
        return w

    def drift(self, held, bars, equity_history):
        """Re-check the overlay on a day the strategy left its targets alone.

        Returns corrective targets when the held book has drifted through a
        gross or per-instrument limit, or when the drawdown cut or kill switch
        changed state; otherwise None. This is what lets a monthly strategy be
        killed on a Tuesday. The buffer does not apply to a correction.
        """
        if self.targets is None:
            return None
        before = (self.reduced, self.killed)
        self.update_state(equity_history)
        if self.breach(held) or (self.reduced, self.killed) != before:
            return self.apply(self.targets, bars, equity_history)
        return None
