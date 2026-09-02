"""CAPM regressions: per-stock alpha/beta and a rolling beta."""
import pandas as pd
import statsmodels.api as sm


def fit_capm(stock_excess, market_excess):
    """Regress one stock's excess return on the market's.

    Returns alpha_annual, beta, r2, alpha_pvalue. Alpha comes out of the
    regression daily; x252 to report it per year.
    """
    df = pd.concat([stock_excess, market_excess], axis=1).dropna()
    y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1])
    fit = sm.OLS(y, X).fit()
    # params is indexed by name (const/x1), so positional .iloc, not [0]
    return {
        "alpha_annual": fit.params.iloc[0] * 252,
        "beta": fit.params.iloc[1],
        "r2": fit.rsquared,
        "alpha_pvalue": fit.pvalues.iloc[0],
    }


def fit_all(excess, market_excess):
    """One regression per column of `excess`, sorted by beta."""
    rows = []
    for name in excess.columns:
        d = fit_capm(excess[name], market_excess)
        d["ticker"] = name
        rows.append(d)
    return pd.DataFrame(rows).sort_values("beta")


def rolling_beta(stock_excess, market_excess, window=60):
    """Beta re-estimated in a sliding window, as cov/var."""
    return stock_excess.rolling(window).cov(market_excess) / market_excess.rolling(window).var()
