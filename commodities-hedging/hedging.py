"""Pairwise screening, minimum-variance hedge ratios and the tests that decide whether to trust them."""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import SPECS


def ols_hac(y, x, lags=5):
    """Regress y on x with an intercept and Newey-West standard errors.

    Returns slope, its HAC standard error, intercept, R-squared, residual standard
    deviation, the Durbin-Watson statistic and the observation count.
    """
    y, x = pd.Series(y).align(pd.Series(x), join="inner")
    mask = y.notna() & x.notna()
    y, x = y[mask], x[mask]
    fit = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    resid = fit.resid
    return {
        "beta": fit.params[1],
        "beta_se": fit.bse[1],
        "alpha": fit.params[0],
        "alpha_se": fit.bse[0],
        "r2": fit.rsquared,
        "resid_sd": resid.std(ddof=2),
        "dw": float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)),
        "n": int(len(y)),
    }


def screen(returns):
    """Rank every unordered pair by R-squared on returns, keeping the correlation sign."""
    rows = []
    for a, b in itertools.combinations(returns.columns, 2):
        corr = returns[a].corr(returns[b])
        rows.append({"a": a, "b": b, "n": int(len(returns)),
                     "corr": corr, "r2": corr ** 2})
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def r2_matrix(returns):
    """Symmetric R-squared matrix. For simple OLS with an intercept this is the squared correlation."""
    return returns.corr() ** 2


def min_var_ratio(pnl_target, pnl_hedge):
    """Minimum-variance hedge ratio in contract-P&L space, Cov(dVi, dVj) / Var(dVj)."""
    a, b = pd.Series(pnl_target).align(pd.Series(pnl_hedge), join="inner")
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    return float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))


def effectiveness(pnl_target, pnl_hedge, h, n_target=1.0):
    """Variance reduction from holding n_target contracts of i against h*n_target of j."""
    a, b = pd.Series(pnl_target).align(pd.Series(pnl_hedge), join="inner")
    mask = a.notna() & b.notna()
    a, b = a[mask] * n_target, b[mask]
    hedged = a - h * b
    unhedged_var = float(np.var(a, ddof=1))
    if unhedged_var == 0:
        return float("nan")
    return 1.0 - float(np.var(hedged, ddof=1)) / unhedged_var


def hedged_pnl(pnl_target, pnl_hedge, h, n_target=1.0):
    """Daily P&L of the hedged book: long n_target of i, short h*n_target of j."""
    a, b = pd.Series(pnl_target).align(pd.Series(pnl_hedge), join="inner")
    return n_target * a - h * b


def contracts_from_return_beta(beta, price_target, price_hedge, target, hedge):
    """Convert a return beta into hedge contracts per target contract using notionals.

    beta comes from a regression of target returns on hedge returns; the notional ratio
    rescales it from percentage space into contract space. Sign convention: the number
    returned is the size of the SHORT hedge position for one long target contract.
    """
    n_i = price_target * SPECS[target]["multiplier"]
    n_j = price_hedge * SPECS[hedge]["multiplier"]
    return beta * n_i / n_j


def integer_hedge(continuous, n_target=1):
    """Candidate whole-contract positions around the continuous ratio."""
    raw = continuous * n_target
    return sorted({int(np.floor(raw)), int(np.round(raw)), int(np.ceil(raw))})


def rolling_ratio(pnl_target, pnl_hedge, window):
    """Rolling minimum-variance ratio and R-squared, both using only the trailing window."""
    a, b = pnl_target.align(pnl_hedge, join="inner")
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    corr = a.rolling(window).corr(b)
    return pd.DataFrame({"h": cov / var, "r2": corr ** 2}).dropna()


def block_bootstrap(pnl_target, pnl_hedge, block=10, n_reps=1000, seed=0):
    """Resample overlapping blocks of paired days and re-estimate the ratio each time.

    Blocks rather than single days because commodity returns cluster in volatility, so
    independent resampling would understate the sampling error.
    """
    rng = np.random.default_rng(seed)
    a, b = pnl_target.align(pnl_hedge, join="inner")
    a, b = a.to_numpy(), b.to_numpy()
    n = len(a)
    n_blocks = int(np.ceil(n / block))
    out = np.empty((n_reps, 2))
    for rep in range(n_reps):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        ai, bi = a[idx], b[idx]
        out[rep, 0] = np.cov(ai, bi, ddof=1)[0, 1] / np.var(bi, ddof=1)
        out[rep, 1] = np.corrcoef(ai, bi)[0, 1] ** 2
    return pd.DataFrame(out, columns=["h", "r2"])


def subperiod_table(pnl, target, hedge, n_periods=3):
    """Re-estimate the ratio and R-squared inside each of n_periods equal-length blocks."""
    edges = np.linspace(0, len(pnl), n_periods + 1).astype(int)
    rows = []
    for k in range(n_periods):
        chunk = pnl.iloc[edges[k]:edges[k + 1]]
        h = min_var_ratio(chunk[target], chunk[hedge])
        rows.append({"period": f"{chunk.index[0].date()} to {chunk.index[-1].date()}",
                     "n": len(chunk), "h": h,
                     "r2": chunk[target].corr(chunk[hedge]) ** 2,
                     "effectiveness": effectiveness(chunk[target], chunk[hedge], h)})
    return pd.DataFrame(rows)


def walk_forward(pnl_target, pnl_hedge, window=60):
    """Re-estimate the ratio on the trailing window each day and apply it to the next day."""
    ratios = rolling_ratio(pnl_target, pnl_hedge, window)["h"].shift(1).dropna()
    a, b = pnl_target.align(pnl_hedge, join="inner")
    a, b = a.loc[ratios.index], b.loc[ratios.index]
    return pd.DataFrame({"h": ratios, "unhedged": a, "hedged": a - ratios * b})


def risk_report(unhedged, hedged):
    """Variance, volatility, tail and hit-rate comparison for a hedged versus unhedged book."""
    var_u, var_h = np.var(unhedged, ddof=1), np.var(hedged, ddof=1)
    return {
        "effectiveness": 1.0 - var_h / var_u,
        "vol_reduction": 1.0 - np.std(hedged, ddof=1) / np.std(unhedged, ddof=1),
        "sd_unhedged": float(np.std(unhedged, ddof=1)),
        "sd_hedged": float(np.std(hedged, ddof=1)),
        "var95_unhedged": float(np.percentile(unhedged, 5)),
        "var95_hedged": float(np.percentile(hedged, 5)),
        "es95_unhedged": float(unhedged[unhedged <= np.percentile(unhedged, 5)].mean()),
        "es95_hedged": float(hedged[hedged <= np.percentile(hedged, 5)].mean()),
        "worst_unhedged": float(np.min(unhedged)),
        "worst_hedged": float(np.min(hedged)),
        "hit_rate": float((np.abs(hedged) < np.abs(unhedged)).mean()),
    }


def spurious_demo(n=2000, n_pairs=200, seed=1):
    """Regress independent random walks on each other, in levels and in differences.

    Two unrelated random walks share a stochastic trend by construction, so a level
    regression finds a high R-squared and a slope many standard errors from zero. The
    same series differenced find nothing. This is why every regression in this project
    is run on returns.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_pairs):
        x = np.cumsum(rng.normal(0, 1, n)) + 100
        y = np.cumsum(rng.normal(0, 1, n)) + 100
        lev = ols_hac(pd.Series(y), pd.Series(x))
        dif = ols_hac(pd.Series(np.diff(y)), pd.Series(np.diff(x)))
        rows.append({"r2_levels": lev["r2"], "t_levels": abs(lev["beta"] / lev["beta_se"]),
                     "dw_levels": lev["dw"],
                     "r2_diffs": dif["r2"], "t_diffs": abs(dif["beta"] / dif["beta_se"])})
    return pd.DataFrame(rows)
