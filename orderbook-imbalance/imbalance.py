"""Order-book imbalance, forward mid-price moves, and a threshold strategy.

Every function takes plain numpy arrays indexed by snapshot. Nothing here shifts
for execution: the strategy takes the signal at snapshot t and fills at t+1, and
that lag is applied in exactly one place, `backtest`.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

HORIZONS_S = (1, 2, 5, 10, 30)
DEPTHS = (1, 5, 20)
THRESHOLDS = (0.3, 0.5, 0.7)
BPS = 1e4


def mid_price(bid_px, ask_px):
    """Midpoint of the best quotes."""
    return (bid_px[:, 0] + ask_px[:, 0]) / 2


def half_spread_bps(bid_px, ask_px):
    """Half the quoted spread as basis points of the mid."""
    return (ask_px[:, 0] - bid_px[:, 0]) / 2 / mid_price(bid_px, ask_px) * BPS


def imbalance(bid_sz, ask_sz, depth):
    """(bid depth - ask depth) / (bid depth + ask depth) over the top `depth` levels, in [-1, 1]."""
    b = bid_sz[:, :depth].sum(axis=1)
    a = ask_sz[:, :depth].sum(axis=1)
    total = b + a
    return np.where(total > 0, (b - a) / np.where(total > 0, total, 1), 0.0)


def forward_index(ts_ms, horizon_s):
    """Index of the first snapshot at least `horizon_s` after each snapshot, or -1 if none."""
    j = np.searchsorted(ts_ms, ts_ms + horizon_s * 1000, side="left")
    return np.where(j < len(ts_ms), j, -1)


def forward_move_bps(price, ts_ms, horizon_s):
    """Change in `price` from t to the first snapshot >= horizon_s later, in bps; nan past the end."""
    j = forward_index(ts_ms, horizon_s)
    out = np.full(len(price), np.nan)
    ok = j >= 0
    out[ok] = (price[j[ok]] - price[ok]) / price[ok] * BPS
    return out


def regress(x, y, horizon_s, interval_s=1.0):
    """OLS of forward move on imbalance with Newey-West errors for the overlapping horizon."""
    ok = ~(np.isnan(x) | np.isnan(y))
    X = sm.add_constant(x[ok])
    lags = max(1, int(round(horizon_s / interval_s)))
    fit = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"slope": fit.params[1], "t_stat": fit.tvalues[1], "r2": fit.rsquared,
            "n": int(ok.sum()), "intercept": fit.params[0]}


def binned_expectation(x, y, n_bins=10):
    """Mean forward move in each equal-count bin of imbalance, with the bin centre and a standard error."""
    ok = ~(np.isnan(x) | np.isnan(y))
    frame = pd.DataFrame({"x": x[ok], "y": y[ok]})
    frame["bin"] = pd.qcut(frame["x"], n_bins, labels=False, duplicates="drop")
    g = frame.groupby("bin")
    return pd.DataFrame({"imbalance": g["x"].mean(), "move_bps": g["y"].mean(),
                         "se_bps": g["y"].std() / np.sqrt(g["y"].count()), "n": g["y"].count()})


def hit_rate(x, y):
    """Fraction of nonzero moves whose sign matches the imbalance sign."""
    ok = ~(np.isnan(x) | np.isnan(y)) & (y != 0) & (x != 0)
    return float((np.sign(x[ok]) == np.sign(y[ok])).mean()) if ok.any() else np.nan


def backtest(imb, bid_px, ask_px, ts_ms, horizon_s, threshold, fee_bps=0.0):
    """Trade when |imbalance| > threshold at t, fill at t+1, unwind after horizon_s.

    Positions do not overlap: no new signal is read until the open one has closed.
    Per trade the table carries the mid-to-mid move (gross), the move less one
    half-spread (aggressive entry, passive exit at mid), and the move less the full
    spread and fees (aggressive both ways).
    """
    mid = (bid_px[:, 0] + ask_px[:, 0]) / 2
    exit_at = forward_index(ts_ms, horizon_s)
    rows = []
    t, n = 0, len(imb)
    while t < n - 1:
        if abs(imb[t]) <= threshold:
            t += 1
            continue
        side = 1 if imb[t] > 0 else -1
        entry = t + 1
        close = exit_at[entry]
        if close < 0:
            break
        m0, m1 = mid[entry], mid[close]
        half_in = (ask_px[entry, 0] - bid_px[entry, 0]) / 2 / m0 * BPS
        half_out = (ask_px[close, 0] - bid_px[close, 0]) / 2 / m1 * BPS
        gross = side * (m1 - m0) / m0 * BPS
        rows.append({"signal_t": t, "entry": entry, "exit": close, "side": side,
                     "imbalance": imb[t], "gross_bps": gross,
                     "net_half_bps": gross - half_in,
                     "net_full_bps": gross - half_in - half_out - 2 * fee_bps})
        t = close
    return pd.DataFrame(rows)


def summarise(trades):
    """Per-trade means, hit rate among trades where the mid moved, and a t-stat, for one backtest."""
    if trades.empty:
        return {"n_trades": 0}
    g = trades["gross_bps"]
    moved = g[g != 0]
    return {"n_trades": len(trades),
            "gross_bps": g.mean(),
            "net_half_bps": trades["net_half_bps"].mean(),
            "net_full_bps": trades["net_full_bps"].mean(),
            "hit_rate": float((moved > 0).mean()) if len(moved) else np.nan,
            "flat_share": float((g == 0).mean()),
            "t_stat": g.mean() / g.std() * np.sqrt(len(g)) if g.std() > 0 else np.nan,
            "total_net_half_bps": trades["net_half_bps"].sum(),
            "total_net_full_bps": trades["net_full_bps"].sum()}
