"""GARCH(1,1) fits, index comparison, and an in-sample VaR check."""
import pandas as pd
from arch import arch_model


def fit_garch(returns, dist="normal"):
    """Fit GARCH(1,1) to one return series (already scaled x100).

    Returns omega/alpha/beta/persistence/aic/bic plus the fitted result
    object under "res" for plots and forecasts.
    """
    r = returns.dropna()
    res = arch_model(r, vol="GARCH", p=1, q=1, dist=dist).fit(disp="off")
    p = res.params
    return {
        "omega": p["omega"],
        "alpha": p["alpha[1]"],
        "beta": p["beta[1]"],
        "persistence": p["alpha[1]"] + p["beta[1]"],
        "aic": res.aic,
        "bic": res.bic,
        "res": res,
    }


def fit_all(rets, dist="normal"):
    """One GARCH(1,1) per column. Returns (table, {name: fitted result})."""
    rows, fits = [], {}
    for name in rets.columns:
        d = fit_garch(rets[name], dist)
        fits[name] = d.pop("res")
        d["index"] = name
        rows.append(d)
    cols = ["index", "omega", "alpha", "beta", "persistence", "aic", "bic"]
    return pd.DataFrame(rows)[cols], fits


def var_backtest(returns, res, level=0.05):
    """Fraction of days the loss beat the level-VaR. Should land near `level`."""
    r = returns.dropna()
    q = res.std_resid.quantile(level)
    var = res.conditional_volatility * q
    hit_rate = (r < var).mean()
    return {"expected": level, "observed": hit_rate, "n_days": len(r)}
