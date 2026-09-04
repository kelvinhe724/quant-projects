"""Kelly sizing for binary contracts and a daily-rebalanced equity simulation."""

import numpy as np
import pandas as pd

# Kalshi markets settle on weekends and holidays and the daily series below is
# one row per calendar day, so annualisation uses 365, not the 252 of an
# exchange calendar.
DAYS_PER_YEAR = 365


def kelly_fraction(p_model, cost):
    """Exact Kelly stake for a binary contract bought at `cost`, paying 1 on a win.

    b = (1 - cost) / cost, so f* = (b*p - q)/b = p - (1 - p) * cost / (1 - cost).
    Returns the fraction of bankroll to put at risk; negative means no bet.
    """
    p_model = np.asarray(p_model, dtype=float)
    cost = np.asarray(cost, dtype=float)
    return p_model - (1 - p_model) * cost / (1 - cost)


def build_bets(df, p_model, spread=0.0, min_edge=0.0):
    """Decide a side and a Kelly stake for every market.

    Buying YES costs price + spread/2, buying NO costs 1 - price + spread/2.
    Both sides are priced, and whichever has the larger positive Kelly stake wins.
    """
    price = df["price"].values
    p = np.asarray(p_model, dtype=float)

    cost_yes = np.clip(price + spread / 2, 1e-4, 1 - 1e-4)
    cost_no = np.clip(1 - price + spread / 2, 1e-4, 1 - 1e-4)

    f_yes = kelly_fraction(p, cost_yes)
    f_no = kelly_fraction(1 - p, cost_no)

    take_yes = f_yes >= f_no
    f = np.where(take_yes, f_yes, f_no)
    cost = np.where(take_yes, cost_yes, cost_no)
    win = np.where(take_yes, df["outcome"].values == 1, df["outcome"].values == 0)

    bets = pd.DataFrame({
        "date": df["close_time"].dt.date.values,
        "side": np.where(take_yes, "yes", "no"),
        "price": price,
        "p_model": p,
        "cost": cost,
        "f": f,
        "payoff": np.where(win, (1 - cost) / cost, -1.0),
    })
    return bets[bets["f"] > min_edge].reset_index(drop=True)


def simulate(bets, k=0.25, max_exposure=1.0):
    """Compound a bankroll daily, splitting it across the bets closing that day.

    Single-contract Kelly stakes are derived one bet at a time, so on a day with
    many open positions they sum to far more than the bankroll. The day's stakes
    are therefore held in Kelly proportion but rescaled to a total exposure of
    k * min(sum f, max_exposure). Below the cap this reduces to plain fractional
    Kelly; above it, k is the fraction of the bankroll at risk that day.
    """
    if len(bets) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    daily = []
    for date, g in bets.groupby("date", sort=True):
        f = g["f"].values
        total = f.sum()
        if total <= 0:
            continue
        w = f / total * k * min(total, max_exposure)
        daily.append((date, float(np.dot(w, g["payoff"].values))))
    returns = pd.Series(
        [r for _, r in daily], index=pd.to_datetime([d for d, _ in daily])
    )
    returns = returns.clip(lower=-0.999)
    equity = (1 + returns).cumprod()
    return returns, equity


def stats(returns, equity):
    """Annualised Sharpe, total growth and max drawdown of a daily return series."""
    if len(returns) < 2:
        return {"sharpe": np.nan, "log_growth": np.nan, "total_return": np.nan,
                "max_drawdown": np.nan, "days": len(returns),
                "mean_daily": np.nan}
    sd = returns.std(ddof=1)
    sharpe = np.sqrt(DAYS_PER_YEAR) * returns.mean() / sd if sd > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    return {
        "sharpe": sharpe,
        # Annualised log growth. This, not the Sharpe, is what Kelly maximises:
        # daily returns scale linearly in k, so the arithmetic Sharpe is
        # invariant to k while the growth rate peaks and then falls.
        "log_growth": DAYS_PER_YEAR * np.log1p(returns).mean(),
        "total_return": equity.iloc[-1] - 1,
        "max_drawdown": drawdown.min(),
        "days": len(returns),
        "mean_daily": returns.mean(),
    }


def sharpe_ci(returns, draws=2000, seed=0):
    """Bootstrap a confidence interval for the annualised Sharpe.

    The test window here is a few weeks, so the point estimate is built on a few
    dozen daily observations and its sampling error is large.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(draws, len(r)))
    sample = r[idx]
    sd = sample.std(axis=1, ddof=1)
    sh = np.where(sd > 0, np.sqrt(DAYS_PER_YEAR) * sample.mean(axis=1) / sd, np.nan)
    return tuple(np.nanpercentile(sh, [2.5, 97.5]))


def per_bet_edge(bets):
    """Mean payoff per dollar staked, with a t-statistic against zero."""
    if len(bets) == 0:
        return {"n": 0, "mean_payoff": np.nan, "t": np.nan}
    p = bets["payoff"].values
    se = p.std(ddof=1) / np.sqrt(len(p))
    return {
        "n": len(p),
        "mean_payoff": p.mean(),
        "t": p.mean() / se if se > 0 else np.nan,
    }


def sweep(bets, ks=(0.1, 0.25, 0.5, 0.75, 1.0)):
    rows = []
    for k in ks:
        r, e = simulate(bets, k=k)
        s = stats(r, e)
        s["k"] = k
        rows.append(s)
    return pd.DataFrame(rows)[
        ["k", "sharpe", "log_growth", "total_return", "max_drawdown", "days",
         "mean_daily"]
    ]
