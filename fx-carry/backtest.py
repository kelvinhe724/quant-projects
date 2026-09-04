"""Backtest engine: execution lag, transaction costs, metrics.

Target weights carry the date the signal was observed. The engine lags them one
trading day before they earn anything, so a book set on the close of t first
sees the return of t+1. That shift lives here and nowhere else.
"""
import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
COST_BPS = 5.0


def run(weights, returns, cost_bps=COST_BPS, base="USD"):
    """Score a target-weight panel, returning daily gross and net return series.

    cost_bps is a one-way charge on traded notional in the foreign legs. A
    position in the base currency is the absence of a foreign position, so it
    is never charged.
    """
    weights = weights.reindex(columns=returns.columns)
    held = weights.reindex(returns.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    foreign = held.drop(columns=base, errors="ignore")
    traded = foreign.diff().abs().sum(axis=1)
    cost = traded * cost_bps / 1e4
    return pd.DataFrame({
        "gross": gross,
        "net": gross - cost,
        "long": (held.clip(lower=0) * returns).sum(axis=1),
        "short": (held.clip(upper=0) * returns).sum(axis=1),
        "traded": traded,
        "cost": cost,
        "net_dollar": -foreign.sum(axis=1),
    })


def max_drawdown(returns):
    """Return the worst peak-to-trough decline and the dates that bracket it."""
    equity = (1 + returns).cumprod()
    underwater = equity / equity.cummax() - 1
    trough = underwater.idxmin()
    peak = equity.loc[:trough].idxmax()
    recovered = equity.loc[trough:] >= equity.loc[peak]
    return {
        "max_drawdown": underwater.min(),
        "dd_peak": peak,
        "dd_trough": trough,
        "dd_recovery": recovered.idxmax() if recovered.any() else pd.NaT,
    }


def monthly(returns):
    """Compound daily returns into calendar months."""
    return returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def metrics(book, column="net"):
    """Summarise one backtest frame into the performance table."""
    r = book[column].dropna()
    if r.empty or r.std() == 0:
        return {}
    years = len(r) / TRADING_DAYS
    equity = (1 + r).cumprod()
    m = monthly(r)
    out = {
        "annual_return": equity.iloc[-1] ** (1 / years) - 1,
        "annual_vol": r.std() * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "skew_daily": stats.skew(r),
        "skew_monthly": stats.skew(m),
        "kurtosis_monthly": stats.kurtosis(m),
        "hit_rate_monthly": (m > 0).mean(),
        "worst_month": m.min(),
        "worst_month_date": m.idxmin(),
        "annual_turnover": book.loc[r.index, "traded"].sum() / 2 / years,
        "cost_drag": book.loc[r.index, "cost"].sum() / years,
        "n_months": len(m),
    }
    out.update(max_drawdown(r))
    out["calmar"] = out["annual_return"] / abs(out["max_drawdown"])
    return out


def summary(book):
    """Gross and net metrics for one backtest."""
    out = {f"gross_{k}": v for k, v in metrics(book, "gross").items()
           if k in ("annual_return", "sharpe", "max_drawdown")}
    out.update(metrics(book, "net"))
    return out


def by_year(book):
    """Per-calendar-year gross, net, Sharpe, drawdown and turnover."""
    rows = {}
    for year, chunk in book.groupby(book.index.year):
        net = chunk["net"]
        rows[year] = {
            "gross": (1 + chunk["gross"]).prod() - 1,
            "net": (1 + net).prod() - 1,
            "sharpe": net.mean() / net.std() * np.sqrt(TRADING_DAYS),
            "max_dd": max_drawdown(net)["max_drawdown"],
            "turnover": chunk["traded"].sum() / 2,
        }
    return pd.DataFrame(rows).T
