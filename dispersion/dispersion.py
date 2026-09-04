"""Black-Scholes pricing and inversion, ATM vol extraction, implied and realised
correlation, and the vega-weighted dispersion book."""
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
MONTH = 21

# Half bid-ask spread crossed at entry, in vol points of the ATM straddle, and a
# monthly allowance for delta hedging slippage per leg, same units. Index options
# are much tighter than single names. Both are assumptions, not measurements.
SPREAD_INDEX = 0.25
SPREAD_SINGLE = 0.75
HEDGE_COST = 0.10


def bs_price(S, K, T, r, sigma, cp):
    """Price a European option; cp is +1 for a call, -1 for a put."""
    S, K, T, sigma, cp = map(np.asarray, (S, K, T, sigma, cp))
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return cp * (S * norm.cdf(cp * d1) - K * np.exp(-r * T) * norm.cdf(cp * d2))


def implied_vol(price, S, K, T, r, cp, lo=1e-4, hi=5.0, steps=60):
    """Invert bs_price by bisection on [lo, hi]; returns nan where no vol fits.

    Bisection is slower than Newton but cannot diverge on far-from-the-money
    quotes, which is the failure mode a single bad print would trigger.
    """
    price, S, K, T, cp = np.broadcast_arrays(*map(np.asarray, (price, S, K, T, cp)))
    lo = np.full(price.shape, lo, dtype=float)
    hi = np.full(price.shape, hi, dtype=float)
    intrinsic = np.maximum(cp * (S - K * np.exp(-r * T)), 0.0)
    ok = (price > intrinsic) & (price < bs_price(S, K, T, r, hi, cp))
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        too_high = bs_price(S, K, T, r, mid, cp) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def atm_vol(chain, r):
    """Return one ATM implied vol per expiry from a single ticker's quotes.

    ATM is the strike nearest the forward. Call and put mids at that strike are
    inverted separately and averaged, which cancels most of the parity error
    that a stale dividend or rate assumption introduces.
    """
    rows = []
    for expiry, q in chain.groupby("expiry"):
        T, spot = float(q["T"].iloc[0]), float(q["spot"].iloc[0])
        fwd = spot * np.exp(r * T)
        strike = q.loc[(q["strike"] - fwd).abs().idxmin(), "strike"]
        at = q[q["strike"] == strike]
        vols = {}
        for cp, sign in (("C", 1), ("P", -1)):
            leg = at[at["cp"] == cp]
            if leg.empty or leg["bid"].iloc[0] <= 0:
                continue
            mid = 0.5 * (leg["bid"].iloc[0] + leg["ask"].iloc[0])
            vols[cp] = float(implied_vol(mid, spot, strike, T, r, sign))
        if vols:
            rows.append({"expiry": expiry, "T": T, "strike": strike,
                         "call_iv": vols.get("C", np.nan), "put_iv": vols.get("P", np.nan),
                         "atm_iv": np.nanmean(list(vols.values()))})
    return pd.DataFrame(rows).set_index("expiry")


def vol_at_horizon(atm, days=30):
    """Interpolate ATM vols to a fixed horizon, linearly in total variance."""
    target = days / 365
    atm = atm.dropna(subset=["atm_iv"]).sort_values("T")
    var = atm["atm_iv"] ** 2 * atm["T"]
    if target <= atm["T"].iloc[0]:
        return float(atm["atm_iv"].iloc[0])
    if target >= atm["T"].iloc[-1]:
        return float(atm["atm_iv"].iloc[-1])
    return float(np.sqrt(np.interp(target, atm["T"], var) / target))


def implied_correlation(index_vol, vols, weights):
    """Solve sigma_I^2 = sum w_i^2 s_i^2 + rho * sum_{i!=j} w_i w_j s_i s_j for rho."""
    vols, weights = np.asarray(vols, float), np.asarray(weights, float)
    ws = weights * vols
    own = np.sum(ws ** 2)
    cross = np.sum(ws) ** 2 - own
    return (index_vol ** 2 - own) / cross


def basket_vol(vols, weights, rho):
    """Invert implied_correlation: the index vol a basket has at correlation rho."""
    ws = np.asarray(weights, float) * np.asarray(vols, float)
    own = np.sum(ws ** 2)
    return np.sqrt(own + rho * (np.sum(ws) ** 2 - own))


def realised_vol(returns, window):
    """Annualised trailing standard deviation of log returns, zero-mean."""
    return np.sqrt((returns ** 2).rolling(window).mean() * TRADING_DAYS)


def realised_correlation(returns, index_returns, weights, window):
    """Trailing correlation of the basket, weighted the same way as the implied one.

    Uses the realised index variance against the realised single-name variances,
    which is exactly the quantity implied_correlation backs out of option prices.
    """
    vols = realised_vol(returns, window)
    idx = realised_vol(index_returns, window)
    ws = vols * weights.values
    own = (ws ** 2).sum(axis=1)
    cross = ws.sum(axis=1) ** 2 - own
    return (idx ** 2 - own) / cross


def average_pairwise_correlation(returns, window):
    """Trailing mean of the off-diagonal entries of the correlation matrix."""
    n = returns.shape[1]
    out = pd.Series(np.nan, index=returns.index)
    x = returns.to_numpy()
    for i in range(window - 1, len(x)):
        c = np.corrcoef(x[i - window + 1:i + 1].T)
        out.iloc[i] = (c.sum() - n) / (n * (n - 1))
    return out


def forward_realised_vol(returns, window):
    """Annualised vol of the `window` returns starting the day after the index date."""
    fwd = (returns ** 2).rolling(window).mean().shift(-window)
    return np.sqrt(fwd * TRADING_DAYS)


def parallel_vega(index_vol, vols, weights):
    """Change in the index vol per unit parallel shift in every single-name vol.

    Scaling the single-name legs by this makes the book flat to a market-wide
    vol move and leaves only the correlation exposure, which is what desks call
    correlation-weighted dispersion. It is close to the square root of the
    implied correlation.
    """
    vols, weights = np.asarray(vols, float), np.asarray(weights, float)
    ws = weights * vols
    rho = implied_correlation(index_vol, vols, weights)
    return (np.sum(weights * ws) + rho * (np.sum(ws) - np.sum(weights * ws))) / index_vol


def book_pnl(index_implied, index_realised, single_implied, single_realised, weights,
             scale=1.0, spread_index=SPREAD_INDEX, spread_single=SPREAD_SINGLE,
             hedge=HEDGE_COST):
    """P&L of one index vega unit of dispersion, in vol points.

    Short one vega of the index straddle, long scale * w_i vega of each name's
    straddle: scale 1 is vega-weighted, scale = parallel_vega is
    correlation-weighted. A delta-hedged straddle held to expiry pays roughly
    vega times realised minus implied vol, so the book pays the weighted
    single-name spread minus the index spread. Costs are the entry half-spread
    on every leg plus the hedging allowance, charged per vega unit.
    """
    single = scale * (weights.values * (single_realised - single_implied)).sum(axis=1)
    index = index_realised - index_implied
    gross = single - index
    cost = spread_index + hedge + scale * weights.sum() * (spread_single + hedge)
    return pd.DataFrame({"gross": gross, "single_leg": single, "index_leg": -index,
                         "cost": cost, "net": gross - cost})


def simulate(returns, index_returns, vix, weights, premium, entry_dates,
             lookback=MONTH, hold=MONTH, wedge=0.0, corr_weighted=False, **cost_kw):
    """Run the monthly dispersion book with a one-day gap between decision and fill.

    The decision date is `entry_dates`; the trade is filled at the next close using
    that day's VIX and trailing vols, and the straddles realise over the `hold`
    returns after the fill. `premium` scales each name's trailing realised vol
    into a stand-in for its implied vol, and `wedge` is subtracted from VIX to
    turn a variance-swap level into an ATM straddle vol. Both are the
    approximations the README flags: option history is not available here.
    """
    pos = returns.index.get_indexer(entry_dates) + 1
    pos = pos[(pos > 0) & (pos < len(returns))]
    fill = returns.index[pos]

    single_implied = realised_vol(returns, lookback).mul(premium, axis=1).loc[fill] * 100
    index_implied = vix.reindex(fill) - wedge
    single_realised = forward_realised_vol(returns, hold).loc[fill] * 100
    index_realised = forward_realised_vol(index_returns, hold).loc[fill] * 100

    scale = 1.0
    if corr_weighted:
        # clipped because the subset bias the README describes can push the
        # implied correlation below zero, and nobody shorts single-name vol on it
        scale = pd.Series([parallel_vega(index_implied.loc[d], single_implied.loc[d], weights)
                           for d in fill], index=fill).clip(0.0, 1.0)
    book = book_pnl(index_implied, index_realised, single_implied, single_realised,
                    weights, scale, **cost_kw)
    book.index.name = "fill"
    book["scale"] = scale
    book["decision"] = returns.index[pos - 1]
    book["implied_corr"] = [implied_correlation(index_implied.loc[d] / 100,
                                                single_implied.loc[d] / 100, weights)
                            for d in fill]
    book["realised_corr"] = [implied_correlation(index_realised.loc[d] / 100,
                                                 single_realised.loc[d] / 100, weights)
                             for d in fill]
    return book.dropna()


def metrics(pnl, periods_per_year=12):
    """Sharpe, drawdown and tail statistics of a monthly P&L stream in vol points."""
    equity = pnl.cumsum()
    under = equity - equity.cummax()
    trough = under.idxmin()
    peak = equity.loc[:trough].idxmax()
    return {
        "months": int(len(pnl)),
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year)),
        "total": float(pnl.sum()),
        "hit_rate": float((pnl > 0).mean()),
        "worst_month": float(pnl.min()),
        "worst_month_date": pnl.idxmin(),
        "p5": float(pnl.quantile(0.05)),
        "p1": float(pnl.quantile(0.01)),
        "skew": float(pnl.skew()),
        "max_drawdown": float(under.min()),
        "dd_peak": peak,
        "dd_trough": trough,
    }
