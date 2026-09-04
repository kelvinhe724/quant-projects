"""Fills at the next bar with an explicit cost model.

Cost per trade in bps of traded notional is commission + half spread + impact,
where impact is k * sigma_daily * sqrt(participation) and participation is the
trade's share of the instrument's trailing average daily dollar volume. Trades
above `participation_cap` of that volume are cut to the cap and the remainder
is left for the next day.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    impact_coef: float = 0.0
    participation_cap: float = 1.0
    adv_window: int = 20

    def slippage_bps(self, participation, sigma_daily):
        """Price slippage in bps for one trade; excludes commission."""
        impact = 1e4 * self.impact_coef * sigma_daily * np.sqrt(participation)
        return self.half_spread_bps + impact


@dataclass
class Trade:
    date: pd.Timestamp
    instrument: str
    quantity: float
    price: float
    fill: float
    notional: float
    commission: float
    slippage: float
    slippage_bps: float
    participation: float
    capped: bool
    sigma: float = 0.0
    strategy: str = ""

    @property
    def cost(self):
        return self.commission + self.slippage


class Executor:
    def __init__(self, costs=None, fill="open"):
        if fill not in ("open", "close"):
            raise ValueError("fill must be 'open' or 'close'")
        self.costs = costs or CostModel()
        self.fill = fill

    def fill_prices(self, bars, date):
        return bars.field(self.fill).loc[date]

    def liquidity(self, bars, date):
        """Trailing average dollar volume and daily close-to-close vol per instrument.

        Uses only bars strictly before `date`: the fill happens at this bar's
        open, when its own close and volume are not yet known.
        """
        w = self.costs.adv_window
        vol = bars.volume
        if vol is None:
            return None, None
        vols = vol.loc[vol.index < date].tail(w)
        if vols.isna().all().all():
            return None, None
        closes = bars.close.loc[bars.close.index < date].tail(w + 1)
        adv = (vols * closes.iloc[1:]).mean()
        sigma = closes.pct_change().std().fillna(0.0)
        return adv, sigma

    def execute(self, date, bars, targets, portfolio, strategy=""):
        """Trade toward `targets` (weights of current equity) at this bar's fill price.

        Returns the trades done and the residual targets that were capped.
        """
        prices = self.fill_prices(bars, date)
        adv, sigma = self.liquidity(bars, date)
        equity = portfolio.equity
        trades, residual = [], {}
        for name, w in targets.items():
            px = prices.get(name, np.nan)
            if not np.isfinite(px) or px <= 0:
                residual[name] = w
                continue
            want = w * equity / px
            qty = want - portfolio.positions.get(name, 0.0)
            if abs(qty) * px < 1e-9:
                continue
            participation, capped = 0.0, False
            if adv is not None and np.isfinite(adv.get(name, np.nan)) and adv[name] > 0:
                participation = abs(qty) * px / adv[name]
                if participation > self.costs.participation_cap:
                    qty = np.sign(qty) * self.costs.participation_cap * adv[name] / px
                    participation, capped = self.costs.participation_cap, True
                    residual[name] = w
            s_daily = float(sigma[name]) if sigma is not None and name in sigma else 0.0
            bps = self.costs.slippage_bps(participation, s_daily)
            fill = px * (1 + np.sign(qty) * bps / 1e4)
            notional = abs(qty) * px
            t = Trade(date, name, qty, px, fill, notional,
                      commission=notional * self.costs.commission_bps / 1e4,
                      slippage=notional * bps / 1e4, slippage_bps=bps,
                      participation=participation, capped=capped, sigma=s_daily, strategy=strategy)
            portfolio.apply(t)
            trades.append(t)
        return trades, residual
