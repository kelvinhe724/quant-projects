"""Perp-spot basis as a tradeable index, the z-score mean-reversion Alpha, and the venue-gap monitor.

The instrument the research harness trades is P = spot / perp, accrued with
funding at each settlement: one unit is one dollar of spot long against one
dollar of perp short, so P's return is spot minus perp plus the funding the
short collects. A rich basis (perp above spot) is a low P; the Alpha buys P
when the basis is far above its rolling mean and sells it when far below,
and holds until the z-score comes back inside the exit band.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from research.alpha.base import Alpha  # noqa: E402
from research.features.store import Feature  # noqa: E402

# per traded unit, one way: Binance spot taker 10 bps, USD-M perp taker 5 bps, 1 bp slippage each leg
FEES = {"taker": 10 + 5 + 2, "maker": 10 + 2 + 2, "vip": 1.5 + 1 + 2}
COST_BPS = FEES["taker"]
BAR = "5min"
BARS_PER_DAY = 288


def basis(spot, perp):
    """perp / spot - 1, in decimal, dates x symbols."""
    return perp / spot - 1


def funding_accrual(index, funding):
    """1 + funding rate on the bar that closes at each settlement, 1 elsewhere, dates x symbols."""
    f = funding.copy()
    f.index = f.index.round("min")  # Binance stamps a third of settlements 1 ms after the hour; unrounded they miss the bar
    f = f.reindex(index).fillna(0.0)
    return 1.0 + f


def tr_index(spot, perp, funding):
    """Total-return price of long spot, short perp: spot / perp times funding collected by the short."""
    accrual = funding_accrual(spot.index, funding.reindex(columns=spot.columns)).cumprod()
    return spot / perp * accrual


def zscore_feature(window):
    """z of the basis against its trailing window, published one bar late so the fill is the next close."""
    def fn(raw):
        b = raw.frames["basis"]
        return (b - b.rolling(window).mean()) / b.rolling(window).std()
    return Feature(f"z_{window}", fn, lag=1)


class BasisMR(Alpha):
    """Enter when |z| crosses `entry`, against the sign; exit when |z| falls under `exit`."""

    def __init__(self, window=288, entry=2.0, exit=0.5):
        self.window, self.entry, self.exit = window, entry, exit
        self.name = f"BasisMR[w={window},z={entry}]"
        self.pos = {}

    def fit(self, X, y):
        self.pos = {}
        return self

    def signal(self, features):
        z = features[f"z_{self.window}"]
        out = {}
        for inst, v in z.items():
            held = self.pos.get(inst, 0.0)
            if np.isnan(v):
                w = held
            elif v > self.entry:
                w = 1.0
            elif v < -self.entry:
                w = -1.0
            elif abs(v) < self.exit:
                w = 0.0
            else:
                w = held
            self.pos[inst] = w
            out[inst] = w
        return pd.Series(out)


def daily(bar_returns):
    """Compound bar returns into one return per UTC day."""
    r = bar_returns.dropna()
    return (1 + r).groupby(r.index.floor("D")).prod() - 1


def daily_sharpe(bar_returns):
    """Registry convention for a stream of bar returns: compound to UTC days, drop flat days, ddof=1, sqrt(252)."""
    d = daily(bar_returns)
    d = d[d != 0]
    return float(d.mean() / d.std(ddof=1) * np.sqrt(252)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")


def dislocation(b, window=BARS_PER_DAY):
    """Basis minus its trailing mean, in decimal."""
    return b - b.rolling(window).mean()


def exceedance(d, thresholds_bps):
    """Share of bars where |dislocation| is above each threshold, per symbol."""
    rows = {t: (d.abs() * 1e4 > t).mean() for t in thresholds_bps}
    return pd.DataFrame(rows).T.rename_axis("threshold_bps")


def reversion(d, threshold_bps, horizons):
    """After |d| first exceeds the threshold, the mean fraction of the excursion given back by each horizon."""
    out = {}
    for sym in d.columns:
        x = d[sym].dropna()
        hit = (x.abs() * 1e4 > threshold_bps) & ~(x.shift(1).abs() * 1e4 > threshold_bps)
        starts = x.index[hit.to_numpy()]
        loc = x.index.get_indexer(starts)
        row = {"episodes": len(starts)}
        for h in horizons:
            ok = loc + h < len(x)
            now, later = x.to_numpy()[loc[ok]], x.to_numpy()[loc[ok] + h]
            row[f"given_back_{h}"] = float(np.mean(1 - later / now)) if ok.any() else np.nan
        out[sym] = row
    return pd.DataFrame(out).T


def venue_gap(a, b):
    """a / b - 1 per minute, in decimal, on the minutes both venues printed."""
    idx = a.dropna().index.intersection(b.dropna().index)
    return (a.reindex(idx) / b.reindex(idx) - 1)


def gap_runs(g, threshold_bps):
    """Lengths in minutes of the runs where |gap| stays above the threshold."""
    above = (g.abs() * 1e4 > threshold_bps).to_numpy()
    if not above.any():
        return np.array([], dtype=int)
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
