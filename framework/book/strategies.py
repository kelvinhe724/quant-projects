"""The three v1 sleeves as engine strategies.

TrendETF and CryptoTrend port trend-following/trend.py, FXCarryETF ports
fx-carry/carry.py::carry_weights. Each on_bar returns target weights only at
month end, built from data through that day; the engine fills them at the
next open, so no function here shifts anything.
"""
import numpy as np
import pandas as pd

from framework.book import universe
from framework.engine import Strategy

LOOKBACK = 12
SKIP = 1
TARGET_VOL = 0.40
VOL_COM = 60
TRADING_DAYS = 252
N_LEG = 2


def month_ends(index):
    """Last trading day of each month in `index`."""
    days = pd.Series(index, index=index)
    return pd.DatetimeIndex(sorted(days.groupby([index.year, index.month]).max()))


def trend_signal(monthly_px, lookback=LOOKBACK, skip=SKIP):
    """+1 where the return from t-lookback to t-skip months is positive, -1 where negative."""
    if lookback <= skip:
        raise ValueError(f"lookback {lookback} must exceed skip {skip}")
    ret = monthly_px.shift(skip) / monthly_px.shift(lookback) - 1
    return np.sign(ret).where(ret.notna())


def ex_ante_vol(daily_returns, com=VOL_COM):
    """Annualised exponentially weighted standard deviation of daily returns."""
    return daily_returns.ewm(com=com, min_periods=com).std() * np.sqrt(TRADING_DAYS)


def positions(signal, vol, target=TARGET_VOL):
    """Size each asset's sign so that its ex-ante volatility equals the target."""
    return signal * target / vol.reindex(signal.index)


def portfolio_weights(pos, classes, scheme="class"):
    """Combine per-asset positions into one book.

    "class" gives each live asset class an equal share and splits it equally
    among the class's live assets; "equal" gives every live asset 1/N.
    """
    live = pos.notna()
    if scheme == "equal":
        return pos.div(live.sum(axis=1), axis=0).fillna(0.0)
    n_classes = live.T.groupby(classes).any().sum()
    out = pd.DataFrame(0.0, index=pos.index, columns=pos.columns)
    for names in classes.groupby(classes).groups.values():
        names = list(names)
        n = live[names].sum(axis=1)
        out[names] = pos[names].div(n.where(n > 0), axis=0).div(n_classes, axis=0).fillna(0.0)
    return out


def carry_weights(rates, n_leg=N_LEG):
    """Long the n_leg highest-rate currencies and short the n_leg lowest, 1/n_leg each.

    Zero everywhere if fewer than 2 * n_leg currencies have a rate.
    """
    r = rates.dropna()
    if len(r) < 2 * n_leg:
        return pd.Series(0.0, index=rates.index)
    rank = r.rank(method="first")
    w = (rank > len(r) - n_leg).astype(float) - (rank <= n_leg).astype(float)
    return (w / n_leg).reindex(rates.index).fillna(0.0)


class TrendETF(Strategy):
    """12-1 sign per ETF, 40% per-asset vol target, class-balanced, month-end rebalance."""

    def __init__(self, classes=None, lookback=LOOKBACK, skip=SKIP, target=TARGET_VOL, scheme="class"):
        self.classes = pd.Series(universe.ETFS if classes is None else classes)
        self.lookback, self.skip, self.target, self.scheme = lookback, skip, target, scheme

    def on_bar(self, asof, bars):
        if not bars.is_month_end(asof):
            return None
        names = [n for n in self.classes.index if n in bars.instruments]
        px = bars.close[names]
        sig = trend_signal(px.loc[month_ends(px.index)], self.lookback, self.skip)
        pos = positions(sig, ex_ante_vol(px.pct_change()), self.target)
        return portfolio_weights(pos, self.classes[names], self.scheme).iloc[-1].to_dict()


class CryptoTrend(Strategy):
    """Long or flat 12-1 on each coin, sized to the vol target, split equally. Alpaca cannot short crypto."""

    def __init__(self, instruments=None, lookback=LOOKBACK, skip=SKIP, target=TARGET_VOL):
        self.instruments = list(universe.CRYPTO if instruments is None else instruments)
        self.lookback, self.skip, self.target = lookback, skip, target

    def on_bar(self, asof, bars):
        if not bars.is_month_end(asof):
            return None
        names = [n for n in self.instruments if n in bars.instruments]
        px = bars.close[names]
        sig = trend_signal(px.loc[month_ends(px.index)], self.lookback, self.skip).clip(lower=0.0)
        pos = positions(sig, ex_ante_vol(px.pct_change()), self.target)
        return portfolio_weights(pos, pd.Series("crypto", index=names), "equal").iloc[-1].to_dict()


class FXCarryETF(Strategy):
    """Long the highest-rate currencies, short the lowest, through their ETFs.

    USD is ranked like any other currency; when it is in a leg that leg is
    cash, so the ETF book can be net long or short the dollar. The rate series
    must be attached to the Bars under the currency codes.
    """

    def __init__(self, etfs=None, n_leg=N_LEG):
        self.etfs = {t: c for t, (c, _) in universe.FX_ETFS.items()} if etfs is None else dict(etfs)
        self.n_leg = n_leg

    def on_bar(self, asof, bars):
        if not bars.is_month_end(asof):
            return None
        live = bars.close.iloc[-1]
        codes = {t: c for t, c in self.etfs.items() if t in bars.instruments and np.isfinite(live[t])}
        rates = pd.Series({c: bars.series(c).iloc[-1] for c in [*codes.values(), "USD"]})
        w = carry_weights(rates, self.n_leg)
        return {t: float(w[c]) for t, c in codes.items()}


CAPITAL = 100_000.0


def sleeves():
    """The three v1 sleeves with their frozen parameters."""
    return [TrendETF(), FXCarryETF(), CryptoTrend()]


def book_config(allocations=None, kill=True, cost_scale=1.0, capital=CAPITAL):
    """One Config for backtest, validation and the live shadow book.

    Costs are 5 bps commission, 2 bps half spread, sqrt impact and 50 bps a
    year on shorts, on purpose harsher than Alpaca paper's NBBO fills. Each
    sleeve runs behind the same overlay: 10% vol target, 3x gross cap, half
    size past a 15% drawdown, flat past 25%. kill=False drops the last rule for
    long backtests, where one 2008 breach would zero a sleeve forever; live, a
    kill is a human decision to restart.
    """
    from framework.engine import Config, CostModel, RiskConfig

    return Config(capital=capital,
                  costs=CostModel(commission_bps=5 * cost_scale, half_spread_bps=2 * cost_scale,
                                  impact_coef=0.1 * cost_scale),
                  borrow_bps=50 * cost_scale,
                  risk=RiskConfig(target_vol=0.10, max_gross=3.0, dd_threshold=0.15,
                                  kill_dd=0.25 if kill else None),
                  allocations=allocations)
