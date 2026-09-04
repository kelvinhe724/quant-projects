"""Performance statistics for one return series.

Honesty corrections (deflated and probabilistic Sharpe, bootstrap CI) and the
walk-forward splitter moved to purgedcv on 2026-09-04; the old code is kept in
archive/framework-metrics-honesty.py.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(r, periods=TRADING_DAYS):
    r = pd.Series(r).dropna()
    s = r.std()
    return float(r.mean() / s * np.sqrt(periods)) if s > 0 else np.nan


def drawdown(returns):
    """Underwater curve and the peak, trough and recovery dates of the worst fall."""
    equity = (1 + returns.fillna(0.0)).cumprod()
    under = equity / equity.cummax() - 1
    trough = under.idxmin()
    peak = equity.loc[:trough].idxmax()
    after = equity.loc[trough:] >= equity.loc[peak]
    return under, {"max_drawdown": float(under.min()), "dd_peak": peak, "dd_trough": trough,
                   "dd_recovery": after.idxmax() if after.any() else pd.NaT}


def summary(returns, periods=TRADING_DAYS, turnover=None):
    """Headline statistics for one periodic return series."""
    r = pd.Series(returns).dropna()
    if r.empty or r.std() == 0:
        return {}
    years = len(r) / periods
    equity = (1 + r).cumprod()
    down = r[r < 0].std()
    out = {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(equity.iloc[-1] ** (1 / years) - 1),
        "annual_vol": float(r.std() * np.sqrt(periods)),
        "sharpe": sharpe(r, periods),
        "sortino": float(r.mean() / down * np.sqrt(periods)) if down > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n_periods": len(r),
    }
    out.update(drawdown(r)[1])
    out["calmar"] = out["annual_return"] / abs(out["max_drawdown"]) if out["max_drawdown"] else np.nan
    if turnover is not None:
        out["annual_turnover"] = float(pd.Series(turnover).reindex(r.index).fillna(0.0).sum() / years)
    return out


def rolling_sharpe(returns, window, periods=TRADING_DAYS):
    roll = returns.rolling(window)
    return roll.mean() / roll.std() * np.sqrt(periods)


def by_year(returns, periods=TRADING_DAYS):
    rows = {}
    for year, chunk in returns.groupby(returns.index.year):
        rows[year] = {"return": (1 + chunk).prod() - 1, "vol": chunk.std() * np.sqrt(periods),
                      "sharpe": sharpe(chunk, periods),
                      "max_drawdown": drawdown(chunk)[1]["max_drawdown"], "periods": len(chunk)}
    return pd.DataFrame(rows).T
