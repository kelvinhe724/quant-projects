"""Split close-to-close returns into an overnight and an intraday leg, test each, and trade each."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 252
NW_LAGS = 5


def legs(ohlc):
    """Return a frame of overnight, intraday and close-to-close simple returns.

    overnight_t = Open_t / Close_{t-1} - 1
    intraday_t  = Close_t / Open_t - 1
    (1 + overnight_t)(1 + intraday_t) = Close_t / Close_{t-1} = 1 + close_to_close_t
    """
    out = pd.DataFrame({
        "overnight": ohlc["Open"] / ohlc["Close"].shift(1) - 1,
        "intraday": ohlc["Close"] / ohlc["Open"] - 1,
        "close_to_close": ohlc["Close"] / ohlc["Close"].shift(1) - 1,
    })
    return out.dropna()


def annualise(daily):
    """Return the compounded annual return of a daily return series."""
    n = len(daily)
    if n == 0:
        return np.nan
    return (1 + daily).prod() ** (TRADING_DAYS / n) - 1


def nw_tstat(daily, lags=NW_LAGS):
    """Return (mean, Newey-West t-stat) for the hypothesis that the mean daily return is zero."""
    y = np.asarray(daily, dtype=float)
    if len(y) < lags + 2:
        return np.nan, np.nan
    fit = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])


def summarise(leg_frame):
    """Return annual return, annual vol, Sharpe, mean daily bps and NW t-stat for each leg."""
    rows = {}
    for col in leg_frame.columns:
        s = leg_frame[col]
        mean, t = nw_tstat(s)
        rows[col] = {
            "annual_return": annualise(s),
            "annual_vol": s.std() * np.sqrt(TRADING_DAYS),
            "sharpe": s.mean() / s.std() * np.sqrt(TRADING_DAYS) if s.std() > 0 else np.nan,
            "mean_daily_bps": mean * 1e4,
            "nw_tstat": t,
            "days": len(s),
        }
    return pd.DataFrame(rows).T


def strategy(leg_frame, leg, cost_bps):
    """Return net daily returns from holding only one leg, paying cost_bps one-way on each trade.

    Holding the overnight leg means buying at every close and selling at every
    open, so two trades per day and a round trip of 2 * cost_bps. The buy-and-hold
    reference is close-to-close with no trades at all.
    """
    if leg == "close_to_close":
        return leg_frame[leg].copy()
    c = cost_bps / 1e4
    return (1 + leg_frame[leg]) * (1 - c) ** 2 - 1


def breakeven_cost_bps(daily_leg):
    """Return the one-way cost in bps at which the leg's mean daily return is fully consumed.

    Two trades a day means the round trip costs about 2c per day, so the leg
    breaks even when c = mean / 2. Solved exactly from (1+m)(1-c)^2 = 1.
    """
    m = daily_leg.mean()
    if m <= 0:
        return 0.0
    return (1 - (1 + m) ** -0.5) * 1e4


def by_year(leg_frame):
    """Return annual compounded return of each leg by calendar year."""
    return leg_frame.groupby(leg_frame.index.year).apply(lambda g: (1 + g).prod() - 1)
