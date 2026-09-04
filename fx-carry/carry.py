"""Carry signal, portfolio weights, currency excess returns and the UIP regression.

Every function returns a value at date t built only from data up to t. The
execution lag is applied once, in backtest.run.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

N_LEG = 3
BASE = "USD"


def differential(rates, base=BASE):
    """Interest rate of each currency minus the base rate, in percent per annum."""
    return rates.sub(rates[base], axis=0)


def excess_returns(spot, rates, base=BASE):
    """Daily excess return of holding each currency against the base.

    Spot change plus the interest differential accrued over the calendar days
    since the previous quote. The base currency's own column is identically zero.
    """
    diff = differential(rates, base) / 100.0
    days = spot.index.to_series().diff().dt.days.fillna(0) / 365.0
    out = spot.pct_change().reindex(columns=diff.columns).fillna(0.0)
    return out + diff.shift(1).mul(days, axis=0).fillna(0.0)


def carry_weights(rates, dates, n_leg=N_LEG, base=BASE, dollar_neutral=False):
    """Long the n highest-rate currencies and short the n lowest on each date.

    With dollar_neutral the base currency is left out of the sort, so the long and
    short foreign notionals always cancel. Without it the base is ranked like any
    other currency and the book can end up net long or short the dollar.
    """
    r = rates.loc[dates]
    if dollar_neutral:
        r = r.drop(columns=base)
    rank = r.rank(axis=1, method="first")
    n = r.notna().sum(axis=1)
    longs = rank.gt(n - n_leg, axis=0)
    shorts = rank <= n_leg
    w = (longs.astype(float) - shorts.astype(float)) / n_leg
    w = w.where(n >= 2 * n_leg, 0.0)
    return w.reindex(columns=rates.columns).fillna(0.0)


def uip_regression(spot, rates, dates, base=BASE, lags=6):
    """Fama regression of next month's log spot change on the current differential.

    s is log(base per foreign). Covered parity puts the forward premium at
    (i_base - i_foreign)/12 per month, and uncovered parity says the spot change
    should equal it on average, so the slope on it is 1 under UIP.
    Per-currency slopes get Newey-West standard errors. The pooled row stacks all
    nine currencies; every one is quoted against the base, so residuals in the
    same month share the dollar's move and the pooled standard error is
    clustered by month rather than treated as nine independent series.
    """
    s = np.log(spot.loc[dates])
    ds = s.shift(-1) - s
    fp = -differential(rates.loc[dates], base).drop(columns=base) / 100.0 / 12.0
    rows = {}
    pooled = []
    for ccy in ds.columns:
        both = pd.concat([ds[ccy], fp[ccy]], axis=1).dropna()
        fit = sm.OLS(both.iloc[:, 0], sm.add_constant(both.iloc[:, 1])).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags})
        rows[ccy] = {"slope": fit.params.iloc[1], "se": fit.bse.iloc[1],
                     "alpha_monthly": fit.params.iloc[0], "n": int(fit.nobs)}
        pooled.append(both.set_axis(["y", "x"], axis=1))
    stacked = pd.concat(pooled)
    months = pd.factorize(stacked.index)[0]
    fit = sm.OLS(stacked["y"].values, sm.add_constant(stacked["x"].values)).fit(
        cov_type="cluster", cov_kwds={"groups": months})
    rows["pooled"] = {"slope": fit.params[1], "se": fit.bse[1],
                      "alpha_monthly": fit.params[0], "n": int(fit.nobs)}
    table = pd.DataFrame(rows).T
    table["t_vs_uip"] = (table["slope"] - 1) / table["se"]
    table["t_vs_zero"] = table["slope"] / table["se"]
    return table
