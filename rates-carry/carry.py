"""Carry, rolldown, expected and realised excess returns on a fitted curve, the
position rule, and the map from curve tenors onto Treasury ETFs.

Units: yields and rates in percent per annum, durations in years, returns in
percent per annum. Every function reads data up to its date only; the engine
applies the execution lag.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import data
from framework.engine import Strategy

ns = data.ns

HOLD = 1 / 12
N_LEG = 2
TARGET_DURATION = 7.0
MIN_POINTS = 4
SLEEVE_TENORS = sorted(set(data.ETF_TENOR.values()))


def par_duration(y, T):
    """Modified duration of a par bond with annual coupons at yield y percent."""
    r = np.asarray(y, float) / 100
    T = np.asarray(T, float)
    safe = np.where(r == 0, 1.0, r)
    return np.where(r == 0, T, (1 - (1 + safe) ** -T) / safe)


def bond_price(coupon, y, T):
    """Price per unit face of a bond paying `coupon` percent a year, at yield y percent, T years out.

    Exact for whole years of annual coupons; the same closed form is used for
    fractional T so a month-old bond prices smoothly.
    """
    c, r = coupon / 100, y / 100
    if r == 0:
        return 1 + c * T
    disc = (1 + r) ** -T
    return c / r * (1 - disc) + disc


def fit_curve(tau, y):
    """Fit Nelson-Siegel to one day's curve; return it as a callable, or None with too few points."""
    tau, y = np.asarray(tau, float), np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < MIN_POINTS:
        return None
    r = ns.fit_static(tau[ok], y[ok])
    b0, b1, b2, lam = r["beta0"], r["beta1"], r["beta2"], r["lam"]

    def curve(t):
        return ns.ns_curve(t, b0, b1, b2, lam)

    curve.rmse = r["rmse"]
    curve.params = (b0, b1, b2, lam)
    curve.observed = pd.Series(y[ok], index=tau[ok])
    return curve


def yield_at(curve, T):
    """The observed yield at a quoted tenor, the fitted one anywhere else."""
    obs = curve.observed
    return float(obs[T]) if T in obs.index else float(curve(T))


def interpolate(curve, T, hold=HOLD):
    """Yield at T - hold: the observed T yield shifted by the fitted curve's slope between the two.

    The fit is used for the shape between quoted points, not for their level,
    so a fit error at the tenor does not flow into carry or into the realised
    return.
    """
    return yield_at(curve, T) + float(curve(T - hold) - curve(T))


def carry(y, funding):
    """Yield pickup over the funding rate."""
    return y - funding


def rolldown_yield(curve, T, hold=HOLD):
    """Yield change from a T-year bond becoming a (T - hold)-year bond on an unchanged curve."""
    return curve(T) - curve(T - hold)


def rolldown(curve, T, hold=HOLD):
    """Rolldown as an annualised return: duration times the yield rolled, per year of holding."""
    y = np.array([yield_at(curve, x) for x in np.atleast_1d(T)]) if np.ndim(T) else yield_at(curve, T)
    return par_duration(y, T) * rolldown_yield(curve, T, hold) / hold


def expected_excess(curve, funding, tenors, hold=HOLD):
    """Per tenor: yield, carry, rolldown, duration and their sum, the expected excess return.

    A tenor FRED did not quote that day is NaN throughout, not filled from the fit.
    """
    t = np.asarray(tenors, float)
    y = np.array([curve.observed.get(x, np.nan) for x in t])
    out = pd.DataFrame({"yield": y, "carry": carry(y, funding),
                        "rolldown_bp": rolldown_yield(curve, t, hold) * 100,
                        "rolldown": rolldown(curve, t, hold),
                        "duration": par_duration(y, t)}, index=t)
    out["expected"] = out["carry"] + out["rolldown"]
    return out


def realised_excess(curve0, curve1, funding, T, hold=HOLD):
    """Annualised excess return of buying a T-year par bond on curve0 and selling it `hold` years later on curve1."""
    y0 = yield_at(curve0, T)
    p1 = bond_price(y0, interpolate(curve1, T, hold), T - hold)
    return (p1 - 1 + (y0 - funding) / 100 * hold) / hold * 100


def weights(expected, duration, n_leg=N_LEG, target=TARGET_DURATION, long_short=False):
    """Position per tenor in units of face, each leg carrying `target` years of duration.

    Long only: the n_leg tenors with the highest positive expected excess
    return, flat if none is positive. Long/short: the top n_leg against the
    bottom n_leg regardless of sign, so the book is duration neutral.
    """
    e = expected.dropna()
    w = pd.Series(0.0, index=expected.index)
    longs = e.nlargest(n_leg) if long_short else e[e > 0].nlargest(n_leg)
    if len(longs):
        w[longs.index] = target / duration[longs.index] / len(longs)
    if long_short and len(e) >= 2 * n_leg:
        shorts = e.nsmallest(n_leg)
        w[shorts.index] = -target / duration[shorts.index] / len(shorts)
    return w


def etf_weights(tenor_weights, etf_tenor, etf_duration, tenor_duration):
    """Scale each tenor's face weight by tenor duration over fund duration.

    A fund is a shorter bond than its tenor (TLT holds 20-30y paper and runs
    about 16 years of duration against 17-18 for a 30-year par bond), so the
    same face in the fund would carry less rate exposure than intended.
    """
    out = {}
    for etf, T in etf_tenor.items():
        if T in tenor_weights.index:
            w = tenor_weights[T]
            out[etf] = float(w * tenor_duration[T] / etf_duration[etf]) if w else 0.0
    return out


def empirical_duration(etf_returns, yield_pct):
    """Minus the slope of daily fund returns on daily yield changes, yields in percent."""
    d = pd.DataFrame({"r": etf_returns, "dy": yield_pct.diff() / 100}).dropna()
    fit = sm.OLS(d["r"], sm.add_constant(d["dy"])).fit()
    return -float(fit.params["dy"]), int(fit.nobs)


def monthly_panel(wide, funding, tenors, dates=None):
    """Fit the curve at each month end and compute expected and realised excess returns to the next.

    Returns a dict of date x tenor frames (expected, carry, rolldown,
    duration, realised, yield) and a per-date fit RMSE. `realised` on date t
    is the return earned from t to the next month end, so it lines up with
    the expectation formed at t.
    """
    dates = data.month_ends(wide.index) if dates is None else dates
    f = funding.reindex(wide.index, method="ffill").loc[dates]
    tau = wide.columns.to_numpy(float)
    curves = {}
    for d in dates:
        c = fit_curve(tau, wide.loc[d].to_numpy())
        if c is not None:
            curves[d] = c
    keep = [d for d in dates if d in curves]
    tables = {d: expected_excess(curves[d], f[d], tenors) for d in keep}
    frames = {col: pd.DataFrame({d: t[col] for d, t in tables.items()}).T
              for col in ("yield", "carry", "rolldown", "duration", "expected")}
    realised = pd.DataFrame(np.nan, index=keep, columns=list(tenors), dtype=float)
    dt = pd.Series(np.nan, index=keep)
    for d0, d1 in zip(keep[:-1], keep[1:]):
        dt[d0] = (d1 - d0).days / 365.25
        for T in tenors:
            if T in curves[d0].observed.index and T in curves[d1].observed.index:
                realised.loc[d0, T] = realised_excess(curves[d0], curves[d1], f[d0], T, dt[d0])
    frames["realised"] = realised
    frames["dt"] = dt
    frames["rmse"] = pd.Series({d: c.rmse for d, c in curves.items()})
    return frames


def predictive_regression(expected, realised):
    """Regress realised excess return on the expected one, tenor by tenor and pooled.

    Newey-West standard errors per tenor (monthly returns to the next month
    end do not overlap, but the errors are still autocorrelated through the
    level of rates); the pooled row clusters by month, since every tenor
    shares that month's rate shock. Slope 1 means carry plus rolldown is an
    unbiased forecast; slope 0 means it carries no information.
    """
    rows, pooled = {}, []
    for T in expected.columns:
        d = pd.DataFrame({"y": realised[T], "x": expected[T]}).dropna()
        if len(d) < 24:
            continue
        lags = int(4 * (len(d) / 100) ** (2 / 9))
        fit = sm.OLS(d["y"], sm.add_constant(d["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        rows[f"{T:g}y"] = {"slope": fit.params["x"], "se": fit.bse["x"], "t": fit.tvalues["x"],
                           "alpha": fit.params["const"], "r2": fit.rsquared, "n": int(fit.nobs)}
        pooled.append(d)
    stacked = pd.concat(pooled)
    months = pd.factorize(stacked.index)[0]
    fit = sm.OLS(stacked["y"].to_numpy(), sm.add_constant(stacked["x"].to_numpy())).fit(
        cov_type="cluster", cov_kwds={"groups": months})
    rows["pooled"] = {"slope": fit.params[1], "se": fit.bse[1], "t": fit.tvalues[1],
                      "alpha": fit.params[0], "r2": fit.rsquared, "n": int(fit.nobs)}
    return pd.DataFrame(rows).T


def sleeve_returns(panel, n_leg=N_LEG, target=TARGET_DURATION, long_short=False):
    """Monthly excess return (decimal) of the rule on the curve itself, no costs, and the weights it held."""
    w = pd.DataFrame({d: weights(panel["expected"].loc[d], panel["duration"].loc[d], n_leg, target, long_short)
                      for d in panel["expected"].index}).T
    r = (w * panel["realised"]).sum(axis=1, min_count=1) / 100 * panel["dt"]
    return r, w


class RatesCarryETF(Strategy):
    """Carry plus rolldown on the fitted Treasury curve, held through the ETF that stands in for each tenor.

    At month end: fit the curve to that day's CMT yields, rank the tenors that
    have a live fund by expected excess return, hold the top n_leg at the
    target duration, scale each fund by tenor duration over fund duration.
    The yield series must be attached to the Bars as y2, y5, ... and the
    funding rate as data.FUNDING; a tenor FRED did not publish is inf there.
    """

    def __init__(self, etf_tenor=None, etf_duration=None, n_leg=N_LEG, target=TARGET_DURATION,
                 long_short=False):
        self.etf_tenor = dict(data.ETF_TENOR if etf_tenor is None else etf_tenor)
        self.etf_duration = dict(data.ETF_DURATION if etf_duration is None else etf_duration)
        self.n_leg, self.target, self.long_short = n_leg, target, long_short
        self.name = "RatesCarryLS" if long_short else "RatesCarry"

    def on_bar(self, asof, bars):
        if not bars.is_month_end(asof):
            return None
        live = bars.close.iloc[-1]
        etfs = {t: T for t, T in self.etf_tenor.items() if t in bars.instruments and np.isfinite(live[t])}
        if not etfs:
            return None
        y = np.array([bars.series(f"y{t:g}").iloc[-1] if f"y{t:g}" in bars.series_names else np.nan
                      for t in data.TENORS])
        curve = fit_curve(data.TENORS, y)
        if curve is None:
            return None
        tab = expected_excess(curve, bars.series(data.FUNDING).iloc[-1], sorted(set(etfs.values())))
        w = weights(tab["expected"], tab["duration"], self.n_leg, self.target, self.long_short)
        return etf_weights(w, etfs, self.etf_duration, tab["duration"])
