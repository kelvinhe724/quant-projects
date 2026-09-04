"""Backtest engine shared by both strategies: execution lag, costs, metrics.

The engine takes a panel of target weights and a panel of stock returns. Target
weights carry the date the signal was observed; the engine lags them one trading
day before they earn anything, so a weight set from the close of t first sees the
return of t+1. Doing that here rather than in the signal code means both
strategies get the same trading clock by construction.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST_BPS = 10.0
BORROW_BPS = 50.0


def run(weights, returns, cost_bps=COST_BPS, borrow_bps=BORROW_BPS):
    """Score a target-weight panel, returning daily gross and net return series.

    cost_bps is a one-way charge on traded notional; borrow_bps is an annual rate
    accrued daily on short market value.
    """
    weights = weights.reindex(columns=returns.columns)
    held = weights.reindex(returns.index).ffill().fillna(0.0).shift(1).fillna(0.0)

    stale = ((held != 0) & returns.isna()).sum().sum()
    gross = (held * returns).sum(axis=1)

    traded = held.diff().abs().sum(axis=1)
    traded.iloc[0] = held.iloc[0].abs().sum()
    trading_cost = traded * cost_bps / 1e4
    borrow_cost = held.clip(upper=0).abs().sum(axis=1) * borrow_bps / 1e4 / TRADING_DAYS

    out = pd.DataFrame({
        "gross": gross,
        "net": gross - trading_cost - borrow_cost,
        "long": (held.clip(lower=0) * returns).sum(axis=1),
        "short": (held.clip(upper=0) * returns).sum(axis=1),
        "traded": traded,
        "gross_exposure": held.abs().sum(axis=1),
        "net_exposure": held.sum(axis=1),
        "cost": trading_cost + borrow_cost,
    })
    out.attrs["held_returns_missing"] = int(stale)
    return out


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


def metrics(book, column="net"):
    """Summarise one backtest frame into the performance table."""
    r = book[column].dropna()
    if r.empty or r.std() == 0:
        return {}
    years = len(r) / TRADING_DAYS
    equity = (1 + r).cumprod()
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    out = {
        "total_return": equity.iloc[-1] - 1,
        "annual_return": equity.iloc[-1] ** (1 / years) - 1,
        "annual_vol": r.std() * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "sortino": r.mean() / r[r < 0].std() * np.sqrt(TRADING_DAYS),
        "hit_rate_monthly": (monthly > 0).mean(),
        "hit_rate_daily": (r > 0).mean(),
        "annual_turnover": book.loc[r.index, "traded"].sum() / 2 / years,
        "cost_drag": book.loc[r.index, "cost"].sum() / years,
        "avg_gross_exposure": book.loc[r.index, "gross_exposure"].mean(),
        "avg_net_exposure": book.loc[r.index, "net_exposure"].mean(),
        "n_months": len(monthly),
    }
    out.update(max_drawdown(r))
    out["calmar"] = (out["annual_return"] / abs(out["max_drawdown"])
                     if out["max_drawdown"] else np.nan)
    return out


def summary(book):
    """Gross and net metrics plus per-leg annual returns for one backtest."""
    net, gross = metrics(book, "net"), metrics(book, "gross")
    out = {f"gross_{k}": v for k, v in gross.items() if k in
           ("annual_return", "sharpe", "annual_vol", "max_drawdown")}
    out.update(net)
    for leg in ("long", "short"):
        r = book[leg].dropna()
        years = len(r) / TRADING_DAYS
        out[f"{leg}_annual_return"] = (1 + r).cumprod().iloc[-1] ** (1 / years) - 1
    return out


def by_year(book):
    """Per-calendar-year gross return, net return, Sharpe, drawdown and turnover."""
    rows = {}
    for year, chunk in book.groupby(book.index.year):
        net = chunk["net"]
        rows[year] = {
            "gross": (1 + chunk["gross"]).prod() - 1,
            "net": (1 + net).prod() - 1,
            "sharpe": net.mean() / (net.std() or np.nan) * np.sqrt(TRADING_DAYS),
            "max_dd": max_drawdown(net)["max_drawdown"],
            "turnover": chunk["traded"].sum() / 2,
            "days": len(chunk),
        }
    return pd.DataFrame(rows).T


def beta_alpha(book, benchmark, column="net"):
    """Regress strategy returns on the benchmark; return annualised alpha and beta."""
    both = pd.concat([book[column], benchmark], axis=1).dropna()
    both.columns = ["strategy", "market"]
    beta = both.cov().iloc[0, 1] / both["market"].var()
    alpha = (both["strategy"] - beta * both["market"]).mean() * TRADING_DAYS
    return {"beta": beta, "alpha": alpha}
