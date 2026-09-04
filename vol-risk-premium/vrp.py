"""Black-Scholes greeks, realised-vol estimators, and a delta-hedged short-straddle simulator."""
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
HORIZON = 21


def _d1(S, K, T, sigma):
    return (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))


def bs_price(S, K, T, sigma, cp):
    """Black-Scholes price with zero rate and zero dividend. cp is 'C' or 'P'."""
    if T <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if cp == "C":
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, sigma, cp):
    """Delta of a European option under the same zero-rate assumption."""
    if T <= 0:
        step = 1.0 if S > K else 0.0
        return step if cp == "C" else step - 1.0
    d1 = _d1(S, K, T, sigma)
    return norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1.0


def bs_vega(S, K, T, sigma):
    """Vega per unit of volatility, identical for the call and the put."""
    if T <= 0:
        return 0.0
    return S * norm.pdf(_d1(S, K, T, sigma)) * np.sqrt(T)


def straddle_price(S, K, T, sigma):
    return bs_price(S, K, T, sigma, "C") + bs_price(S, K, T, sigma, "P")


def straddle_delta(S, K, T, sigma):
    return bs_delta(S, K, T, sigma, "C") + bs_delta(S, K, T, sigma, "P")


def implied_vol(price, S, K, T, cp, lo=1e-4, hi=5.0, tol=1e-8):
    """Invert Black-Scholes by bisection. Returns nan outside the no-arbitrage band."""
    if T <= 0 or price <= bs_price(S, K, T, lo, cp) or price >= bs_price(S, K, T, hi, cp):
        return np.nan
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, cp) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def realised_vol(returns, annualise=TRADING_DAYS):
    """Annualised close-to-close volatility, zero-mean estimator."""
    r = np.asarray(returns, dtype=float)
    return float(np.sqrt(annualise * np.mean(r ** 2)))


def forward_realised_vol(returns, horizon=HORIZON, annualise=TRADING_DAYS):
    """Vol realised over the `horizon` days strictly after each date."""
    sq = returns ** 2
    return np.sqrt(annualise * sq.rolling(horizon).mean().shift(-horizon))


def garch_forecast(returns, horizon=HORIZON, refit_every=HORIZON, min_obs=500):
    """Rolling GARCH(1,1) forecast of annualised vol over the next `horizon` days.

    Refits on an expanding window every `refit_every` days and holds the
    parameters between refits, which is what a desk running this daily would do
    and keeps the run to a couple of hundred fits.
    """
    from arch import arch_model

    r = returns.dropna() * 100
    out = pd.Series(np.nan, index=r.index)
    omega = alpha = beta = mu = var = eps = None
    for i in range(min_obs, len(r)):
        if omega is None or (i - min_obs) % refit_every == 0:
            fit = arch_model(r.iloc[:i + 1], vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
            omega = float(fit.params["omega"])
            alpha = float(fit.params["alpha[1]"])
            beta = float(fit.params["beta[1]"])
            mu = float(fit.params["mu"])
            var = float(fit.conditional_volatility.iloc[-1] ** 2)
            eps = float(fit.resid.iloc[-1])
        else:
            # recurse the conditional variance by hand between refits
            var = omega + alpha * eps ** 2 + beta * var
            eps = float(r.iloc[i]) - mu

        # the fit lands on the IGARCH boundary often enough to divide by zero
        persist = min(alpha + beta, 0.9999)
        long_run = omega / (1 - persist)
        one_step = omega + alpha * eps ** 2 + beta * var
        path = [long_run + persist ** h * (one_step - long_run) for h in range(horizon)]
        out.iloc[i] = np.sqrt(TRADING_DAYS * np.mean(path)) / 100
    return out


def simulate(df, horizon=HORIZON, cost_bps=1.0, option_spread=0.01,
             iv_offset=0.0, offset=0, iv_col="vix"):
    """Roll a short delta-hedged ATM straddle and return daily P&L per $1 of notional.

    A cycle opens every `horizon` trading days: sell the straddle struck at the
    close, hedge to zero delta, rehedge at every close, hold to expiry. Options
    are marked with that day's VIX so the daily marks telescope to the
    hold-to-expiry P&L exactly. `option_spread` is the fraction of mid premium
    given up on entry, `cost_bps` is one-way slippage on every share traded, and
    `iv_offset` subtracts vol points from the VIX to stand in for the gap between
    the VIX and an actual at-the-money quote. `offset` shifts the whole calendar
    of cycle start dates, which is how much of the result is timing luck.
    """
    spot = df["spy"].to_numpy(float)
    iv = np.maximum(df[iv_col].to_numpy(float) / 100.0 - iv_offset, 0.01)
    n = len(df)
    cost = cost_bps / 1e4

    pnl = np.zeros(n)
    hedge = np.zeros(n)
    cycle = np.full(n, -1)
    premium = np.zeros(n)

    for start in range(offset, n - horizon, horizon):
        S0, K = spot[start], spot[start]
        T0 = horizon / TRADING_DAYS
        mid = straddle_price(S0, K, T0, iv[start])
        h_prev = straddle_delta(S0, K, T0, iv[start])
        v_prev = mid

        pnl[start] += (-option_spread * mid - abs(h_prev) * S0 * cost) / S0
        hedge[start] = h_prev
        cycle[start] = start
        premium[start] = mid / S0

        for i in range(start + 1, start + horizon + 1):
            T = (start + horizon - i) / TRADING_DAYS
            v = straddle_price(spot[i], K, T, iv[i])
            day = (v_prev - v) + h_prev * (spot[i] - spot[i - 1])

            h_new = 0.0 if T <= 0 else straddle_delta(spot[i], K, T, iv[i])
            day -= abs(h_new - h_prev) * spot[i] * cost

            pnl[i] += day / S0
            hedge[i] = h_new
            cycle[i] = start
            v_prev, h_prev = v, h_new

    out = pd.DataFrame({"pnl": pnl, "hedge": hedge, "cycle": cycle,
                        "premium": premium}, index=df.index)
    return out[out["cycle"] >= 0]


def drawdown(pnl):
    """Drawdown path of a cumulative P&L series on constant notional."""
    equity = pnl.cumsum()
    return equity - equity.cummax()


def metrics(pnl, tail_pct=0.01):
    """Performance and tail statistics for a daily P&L series on constant notional."""
    p = pnl.dropna()
    years = len(p) / TRADING_DAYS
    cut = p.quantile(tail_pct)
    trimmed = p[p > cut]
    return {
        "total_pnl": p.sum(),
        "annual_pnl": p.sum() / years,
        "annual_vol": p.std() * np.sqrt(TRADING_DAYS),
        "sharpe": p.mean() / p.std() * np.sqrt(TRADING_DAYS),
        "sharpe_ex_worst_1pct": trimmed.mean() / trimmed.std() * np.sqrt(TRADING_DAYS),
        "skew": p.skew(),
        "excess_kurtosis": p.kurtosis(),
        "worst_day": p.min(),
        "best_day": p.max(),
        "max_drawdown": drawdown(p).min(),
        "hit_rate": (p > 0).mean(),
        "days": len(p),
    }


def hac_mean(x, lags=None):
    """Mean of an overlapping series with a Newey-West standard error and t-stat."""
    import statsmodels.api as sm

    y = np.asarray(x.dropna(), dtype=float)
    lags = lags or int(1.5 * HORIZON)
    fit = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean": float(fit.params[0]), "se": float(fit.bse[0]),
            "t": float(fit.tvalues[0]), "n": len(y)}
