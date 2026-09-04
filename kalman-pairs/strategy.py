"""Static and Kalman spreads run through the pairs project's trading rules.

The z-score, entry/exit logic, trade extraction and metrics are imported from
../pairs-trading so both spreads are traded by the same code. The only thing that
differs between the two is the hedge ratio: frozen from formation, or the filter's
running estimate.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pairs-trading"))
from backtest import TRADING_DAYS, aggregate, metrics, trades  # noqa: E402
from pairs import positions, zscore  # noqa: E402

from kalman import kalman_spread  # noqa: E402

# The configuration the pairs project tuned on formation. Kept fixed here so the
# only thing being tested is the hedge ratio.
RULES = dict(lookback=90, entry_z=1.5, exit_z=0.0, stop_z=4.0, max_hold=60)
COST_BPS = 10.0

# Kalman state-noise grid. 0 is recursive least squares (hedge ratio never stops
# converging); each step up lets the ratio move roughly three times faster.
NOISE_GRID = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def static_spread(log_a, log_b, beta, intercept):
    """Return the frozen-beta spread and a constant beta path."""
    beta_path = pd.Series(beta, index=log_a.index)
    return log_a - beta * log_b - intercept, beta_path


def pair_backtest(log_px, a, b, spread, beta_path, cost_bps=COST_BPS, **rules):
    """Trade one spread with the shared rules; return daily position, gross, net, turnover.

    Signal at t, fill at t+1. The hedge ratio applied to day t's return is the
    estimate available at t-1, the same day the position was decided. Turnover
    counts both legs, so a drifting hedge ratio pays to rebalance while in a trade.
    """
    z = zscore(spread, rules["lookback"])
    held = positions(z, rules["entry_z"], rules["exit_z"], rules["stop_z"],
                     rules["max_hold"]).shift(1).fillna(0.0)
    beta = beta_path.shift(1)
    gross_exposure = 1.0 + beta.abs()

    ret = np.expm1(log_px[[a, b]].diff())
    gross = held * (ret[a] - beta * ret[b]) / gross_exposure

    weight_a = held / gross_exposure
    weight_b = -held * beta / gross_exposure
    turnover = (weight_a.diff().abs() + weight_b.diff().abs()).fillna(0.0)
    cost = turnover * (cost_bps / 2.0) / 1e4
    return pd.DataFrame({"z": z, "position": held, "beta": beta, "gross": gross.fillna(0.0),
                         "net": (gross - cost).fillna(0.0), "turnover": turnover})


def build_spread(log_px, pair, formation_end, noise_ratio=None):
    """Return (spread, beta path) for one pair: static if noise_ratio is None, else Kalman.

    The Kalman observation variance is the OLS residual variance on formation
    data only, so nothing after formation_end shapes the filter's settings.
    """
    la, lb = log_px[pair.a], log_px[pair.b]
    if noise_ratio is None:
        sp, beta = static_spread(la, lb, pair.beta, pair.intercept)
        return sp, beta
    form = log_px.loc[:formation_end]
    resid = form[pair.a] - pair.beta * form[pair.b] - pair.intercept
    sp, beta, _ = kalman_spread(la, lb, noise_ratio, float(resid.var()))
    return sp, beta


def evaluate(log_px, pairs, window, formation_end, noise_ratio=None, cost_bps=COST_BPS):
    """Backtest every pair on the full history and score only the dates in window."""
    legs = {}
    for p in pairs.itertuples():
        sp, beta = build_spread(log_px, p, formation_end, noise_ratio)
        leg = pair_backtest(log_px, p.a, p.b, sp, beta, cost_bps, **RULES)
        legs[(p.a, p.b)] = leg.loc[window[0]:window[1]]
    port = aggregate(legs)
    all_trades = pd.concat([trades(v) for v in legs.values()], ignore_index=True)
    m = metrics(port["net"], all_trades)
    m["gross_sharpe"] = metrics(port["gross"])["sharpe"]
    m["gross_annual_return"] = metrics(port["gross"])["annual_return"]
    m["annual_turnover"] = port["turnover"].sum() / (len(port) / TRADING_DAYS)
    return m, port, legs


def tune_noise(log_px, pairs, formation, grid=NOISE_GRID):
    """Pick the state-noise ratio by formation-window net Sharpe. Never sees later data."""
    form_px = log_px.loc[:formation[1]]
    rows = [{"noise_ratio": q,
             "formation_sharpe": evaluate(form_px, pairs, formation, formation[1], q)[0]["sharpe"]}
            for q in grid]
    table = pd.DataFrame(rows)
    return float(table.loc[table.formation_sharpe.idxmax(), "noise_ratio"]), table
