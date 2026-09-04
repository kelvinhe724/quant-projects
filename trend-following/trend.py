"""Time-series momentum: the sign of each asset's own past return, sized to a volatility target.

Every function returns a value at month-end t built only from prices up to and
including t. Nothing here shifts for execution; the one-month lag is applied in
backtest.run, in one place.
"""
import numpy as np
import pandas as pd

from data import month_ends

LOOKBACK = 12
SKIP = 1
TARGET_VOL = 0.40
VOL_COM = 60
TRADING_DAYS = 252


def trend_signal(monthly_px, lookback=LOOKBACK, skip=SKIP):
    """Return +1 where the return from t-lookback to t-skip months is positive, -1 where negative."""
    if lookback <= skip:
        raise ValueError(f"lookback {lookback} must exceed skip {skip}")
    ret = monthly_px.shift(skip) / monthly_px.shift(lookback) - 1
    return np.sign(ret).where(ret.notna())


def ex_ante_vol(daily_returns, com=VOL_COM):
    """Annualise an exponentially weighted standard deviation of daily returns."""
    return daily_returns.ewm(com=com, min_periods=com).std() * np.sqrt(TRADING_DAYS)


def positions(signal, vol, target=TARGET_VOL):
    """Size each asset's sign so that its ex-ante volatility equals the target."""
    return signal * target / vol.reindex(signal.index)


def portfolio_weights(pos, classes, scheme="class"):
    """Combine per-asset positions into one book.

    "class" gives each live asset class an equal share of the book and splits that
    share equally among the class's live assets, so five commodities do not
    outweigh three bonds. "equal" gives every live asset 1/N.
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


def build(px, daily_returns, classes, lookback=LOOKBACK, skip=SKIP, target=TARGET_VOL,
          scheme="class"):
    """Run the whole signal chain from daily prices to month-end target weights."""
    ends = month_ends(px.index)
    sig = trend_signal(px.loc[ends], lookback, skip)
    pos = positions(sig, ex_ante_vol(daily_returns), target)
    return portfolio_weights(pos, classes, scheme)
