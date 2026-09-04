"""Rolling out-of-sample evaluation, turnover accounting and performance metrics.

Weights for month t+1 are estimated from the M months ending at t and earn the
return of t+1. The rebalance happens at the close of t, so the traded notional is
the gap between the new target and the position the previous target has drifted
to over month t.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from portfolio import weights as rule_weights

MONTHS = 12
COST_BPS = 10.0


def drift(w, r):
    """Where a weight vector ends up after one period of returns, renormalised to sum one."""
    grown = w * (1 + r)
    return grown / grown.sum()


def run(rule, excess, window, cost_bps=COST_BPS, total=None):
    """Score one rule with rolling estimation. Returns a frame of monthly results and the weight path.

    `excess` is used for estimation and for the reported returns; `total` (simple
    returns) is only needed to drift the previous weights. If absent, excess
    returns drift them, which is a second-order approximation.
    """
    x = excess.to_numpy()
    tot = excess.to_numpy() if total is None else total.to_numpy()
    n_periods, n_assets = x.shape
    rows, w_path = [], {}
    held = np.zeros(n_assets)
    for t in range(window - 1, n_periods - 1):
        target = rule_weights(rule, x[t - window + 1:t + 1])
        turnover = np.abs(target - held).sum()
        gross = target @ x[t + 1]
        rows.append({
            "gross": gross,
            "net": gross - turnover * cost_bps / 1e4,
            "turnover": turnover,
            "gross_exposure": np.abs(target).sum(),
            "max_weight": np.abs(target).max(),
        })
        w_path[excess.index[t]] = target
        held = drift(target, tot[t + 1])
    out = pd.DataFrame(rows, index=excess.index[window:])
    return out, pd.DataFrame(w_path, index=excess.columns).T


def max_drawdown(returns):
    """Worst peak-to-trough fall of the compounded return series."""
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1).min()


def metrics(book, column="net"):
    """Summarise one backtest frame. Returns are already in excess of the risk-free rate."""
    r = book[column]
    return {
        "mean_excess": r.mean() * MONTHS,
        "vol": r.std() * np.sqrt(MONTHS),
        "sharpe": r.mean() / r.std() * np.sqrt(MONTHS),
        "max_drawdown": max_drawdown(r),
        "turnover": book["turnover"].mean(),
        "gross_exposure": book["gross_exposure"].mean(),
        "max_weight": book["max_weight"].max(),
        "months": len(r),
    }


def sharpe_diff_pvalue(a, b):
    """Two-sided p-value for equal Sharpe ratios by the Jobson-Korkie test with Memmel's correction."""
    a, b = np.asarray(a), np.asarray(b)
    t = len(a)
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    var = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / t
    if var <= 0:  # identical series: no difference to test
        return 1.0
    z = (sa - sb) / np.sqrt(var)
    return float(2 * (1 - norm.cdf(abs(z))))
