"""Fills at the next bar with an explicit cost model, and the order-shaping rules.

Cost per trade in bps of traded notional is commission + half spread + impact,
where impact is k * sigma_daily * sqrt(participation) and participation is the
trade's share of the instrument's trailing average daily dollar volume. Trades
above `participation_cap` of that volume are cut to the cap and the remainder
is left for the next day.

`OrderRules.target_quantities` is the ONE implementation of what a broker will
actually do with a target weight: the position buffer, the minimum order, whole
shares on a short leg, and instruments that cannot be shorted at all. The
backtest reaches it through `Executor.execute`; the live path reaches it through
`book.broker.build_orders`. Both get the same end quantities for the same
(weights, equity, prices, held), which is what `book/check.py` asserts. The
default `OrderRules()` is every rule off, so the bare engine is unchanged.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


BUFFER = 0.10     # Carver's position buffer, the single definition in the repo
MIN_ORDER = 25.0  # smallest dollar move worth an order


@dataclass(frozen=True)
class OrderRules:
    """What a broker does to a target weight before it becomes a position.

    min_order          a move (and a target) below this many dollars is not sent
    buffer             an instrument inside target x (1 +/- buffer) is left alone
    whole_share_shorts a short leg rounds toward zero to whole shares
    unshortable        names whose short target flattens instead of going short
    """
    min_order: float = 0.0
    buffer: float = 0.0
    whole_share_shorts: bool = False
    unshortable: tuple = ()

    def target_quantities(self, targets, equity, prices, held):
        """{name: end quantity} for every name in `targets` with a usable price.

        `held` is {name: quantity}. A name left alone by the buffer maps to what
        it already holds, so a caller can diff against `held` and get nothing.
        """
        out = {}
        for name, w in targets.items():
            px = prices.get(name, np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            have_qty = float(held.get(name, 0.0))
            want = float(w) * equity
            if want < 0 and name in self.unshortable:
                want = 0.0
            if abs(want - have_qty * px) < max(self.min_order, self.buffer * abs(want)):
                out[name] = have_qty
            elif abs(want) < self.min_order:
                out[name] = 0.0
            elif want < 0 and self.whole_share_shorts:
                out[name] = -float(int(abs(want) // px))
            else:
                out[name] = want / px
        return out


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
    def __init__(self, costs=None, fill="open", rules=None):
        if fill not in ("open", "close"):
            raise ValueError("fill must be 'open' or 'close'")
        self.costs = costs or CostModel()
        self.fill = fill
        self.rules = rules or OrderRules()

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

        `self.rules` decides the end quantity for each name, so the position this
        reaches is one the broker layer could also reach; see OrderRules. Returns
        the trades done and the residual targets that were capped.
        """
        prices = self.fill_prices(bars, date)
        adv, sigma = self.liquidity(bars, date)
        equity = portfolio.equity
        shaped = self.rules.target_quantities(targets, equity, prices, portfolio.positions)
        trades, residual = [], {}
        for name, w in targets.items():
            px = prices.get(name, np.nan)
            if not np.isfinite(px) or px <= 0:
                residual[name] = w
                continue
            have = portfolio.positions.get(name, 0.0)
            qty = shaped.get(name, have) - have
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
