"""Monthly backtest engine: one-month execution lag, costs on traded notional, metrics.

Target weights carry the month-end on which the signal was observed. The engine
shifts them one month before they earn anything, so a book set at the close of
month t earns the return of month t+1.
"""
import numpy as np
import pandas as pd

COST_BPS = 5.0
MONTHS = 12


def run(weights, returns, cost_bps=COST_BPS):
    """Score month-end target weights on monthly asset returns.

    cost_bps is a one-way charge on traded notional. Weights are notional
    exposures relative to capital, so a vol-scaled bond position of 4x is
    charged on the full 4x when it flips.
    """
    weights = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    contrib = held * returns.fillna(0.0)
    traded = held.diff().abs().sum(axis=1)
    traded.iloc[0] = held.iloc[0].abs().sum()
    gross = contrib.sum(axis=1)
    out = pd.DataFrame({
        "gross": gross,
        "net": gross - traded * cost_bps / 1e4,
        "traded": traded,
        "leverage": held.abs().sum(axis=1),
    })
    out.attrs["contrib"] = contrib
    out.attrs["held_returns_missing"] = int(((held != 0) & returns.isna()).sum().sum())
    return out


def max_drawdown(returns):
    """Return the worst peak-to-trough decline and the dates that bracket it."""
    equity = (1 + returns).cumprod()
    underwater = equity / equity.cummax() - 1
    trough = underwater.idxmin()
    return {"max_drawdown": underwater.min(),
            "dd_peak": equity.loc[:trough].idxmax(), "dd_trough": trough}


def metrics(r, book=None):
    """Summarise one monthly return series; add turnover and leverage if a book is given."""
    r = r.dropna()
    if r.empty:
        return {}
    years = len(r) / MONTHS
    out = {
        "annual_return": (1 + r).prod() ** (1 / years) - 1,
        "annual_vol": r.std() * np.sqrt(MONTHS),
        "sharpe": r.mean() / r.std() * np.sqrt(MONTHS) if r.std() > 0 else np.nan,
        "hit_rate": (r > 0).mean(),
        "n_months": len(r),
    }
    out.update(max_drawdown(r))
    if book is not None:
        b = book.loc[r.index]
        out["annual_turnover"] = b["traded"].sum() / years
        out["avg_leverage"] = b["leverage"].mean()
    return out


def summary(book):
    """Gross and net metrics for one book."""
    gross, net = metrics(book["gross"]), metrics(book["net"], book)
    out = {f"gross_{k}": v for k, v in gross.items()
           if k in ("annual_return", "sharpe", "max_drawdown")}
    out.update(net)
    return out


def by_year(book):
    """Per-calendar-year gross return, net return and Sharpe."""
    rows = {}
    for year, chunk in book.groupby(book.index.year):
        net = chunk["net"]
        rows[year] = {"gross": (1 + chunk["gross"]).prod() - 1,
                      "net": (1 + net).prod() - 1,
                      "sharpe": net.mean() / net.std() * np.sqrt(MONTHS) if net.std() > 0 else np.nan,
                      "months": len(net)}
    return pd.DataFrame(rows).T


def beta_alpha(strategy, benchmark):
    """Regress monthly strategy returns on the benchmark; return annualised alpha and beta."""
    both = pd.concat([strategy, benchmark], axis=1).dropna()
    beta = both.cov().iloc[0, 1] / both.iloc[:, 1].var()
    alpha = (both.iloc[:, 0] - beta * both.iloc[:, 1]).mean() * MONTHS
    return {"beta": beta, "alpha": alpha}
