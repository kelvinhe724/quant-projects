"""Time-series factor regressions, the GRS test, and factor-blend backtests."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def regress(y, factors):
    """Regress one excess return series on factors. Returns a flat dict."""
    df = pd.concat([y.rename("y"), factors], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df[factors.columns])).fit()
    out = {
        "alpha": res.params["const"],
        "t_alpha": res.tvalues["const"],
        "r2": res.rsquared,
        "adj_r2": res.rsquared_adj,
        "n": int(res.nobs),
    }
    for f in factors.columns:
        out[f] = res.params[f]
        out["t_" + f] = res.tvalues[f]
    out["resid"] = res.resid
    return out


def fit_panel(returns, factors):
    """Run `regress` on every column. Returns (table, residual frame)."""
    rows, resid = [], {}
    for name in returns.columns:
        d = regress(returns[name], factors)
        resid[name] = d.pop("resid")
        rows.append({"asset": name, **d})
    table = pd.DataFrame(rows).set_index("asset")
    return table, pd.DataFrame(resid)


def grs(alphas, resid, factors):
    """Gibbons-Ross-Shanken test that all alphas are jointly zero.

    Uses the maximum-likelihood covariance estimates (divided by T), as in the
    1989 paper, so the prefactor is (T-N-K)/N and the statistic is exactly
    F(N, T-N-K) under normal iid errors. The (T/N)((T-N-K)/(T-K-1)) prefactor
    seen in textbooks goes with the unbiased residual covariance; pairing it with
    the MLE one overstates the statistic by T/(T-K-1).
    """
    resid = resid.dropna()
    f = factors.loc[resid.index]
    a = np.asarray(alphas, float)
    T, N, K = len(resid), len(a), f.shape[1]
    if T - N - K <= 0:
        raise ValueError(f"need T > N + K, got T={T} N={N} K={K}")
    sigma = resid.to_numpy().T @ resid.to_numpy() / T
    mu = f.mean().to_numpy()
    omega = np.atleast_2d(np.cov(f.to_numpy(), rowvar=False, bias=True))  # K=1 collapses to a scalar
    sharpe_sq = mu @ np.linalg.solve(omega, mu)
    stat = ((T - N - K) / N) * (a @ np.linalg.solve(sigma, a)) / (1 + sharpe_sq)
    return {
        "grs": stat,
        "p_value": stats.f.sf(stat, N, T - N - K),
        "mean_abs_alpha": np.abs(a).mean(),
        "T": T,
        "N": N,
        "K": K,
    }


def premia(factors, split=None):
    """Mean monthly premium and t-stat per factor, optionally per subsample."""
    def block(df, label):
        m, s, n = df.mean(), df.std(ddof=1), len(df)
        return pd.DataFrame({
            "sample": label,
            "months": n,
            "mean": m,
            "annualized": m * 12,
            "t_stat": m / (s / np.sqrt(n)),
        })

    if split is None:
        return block(factors, "full")
    early, late = factors[factors.index < split], factors[factors.index >= split]
    return pd.concat([block(factors, "full"),
                      block(early, f"pre-{split}"),
                      block(late, f"{split}+")])


def blend(factors, columns, lookback=60):
    """Inverse-volatility blend of long-short factors, rebalanced monthly.

    Weights use only volatility observed through last month, so nothing
    peeks ahead. Months before the lookback fills are dropped.
    """
    f = factors[columns].dropna()
    vol = f.rolling(lookback).std().shift(1)
    w = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
    return (w * f).sum(axis=1).loc[vol.dropna().index]


def performance(r, periods=12):
    """Annualized return, vol, Sharpe and max drawdown of a percent-return series."""
    r = r.dropna()
    curve = (1 + r / 100).cumprod()
    peak = curve.cummax().clip(lower=1.0)  # the first month can itself be a drawdown
    dd = curve / peak - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods)
    return {
        "ann_return": (curve.iloc[-1] ** (periods / len(r)) - 1) * 100,
        "ann_vol": ann_vol,
        "sharpe": np.nan if ann_vol == 0 else r.mean() * periods / ann_vol,
        "max_drawdown": dd.min() * 100,
        "months": len(r),
    }
