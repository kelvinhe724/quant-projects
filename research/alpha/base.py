"""Alpha: signal(features) -> target weights, plus the adapter, the walk-forward and the untouched window.

The research path is vectorised: positions at each rebalance date from the
cross-section of the feature panel, held until the next, one-day lag,
proportional cost. The book path wraps the same Alpha in Sleeve, a
framework Strategy that rebuilds the features from the engine's as-of view
at each rebalance and hands the engine the same weights; check.py proves
the two paths return identical targets.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from purgedcv import WalkForwardSplit

from framework.engine import Strategy, sharpe
from research.features.store import Raw, cross_section, long_panel

COST_BPS = 10.0


class Alpha:
    """Subclass and implement signal. features is instruments x feature names at one date."""
    name = None

    def fit(self, X, y):
        """Optional. X is (date, instrument) x features, y the aligned forward return."""
        return self

    def signal(self, features):
        """Return a Series of instrument -> weight of equity."""
        raise NotImplementedError

    def __str__(self):
        return self.name or type(self).__name__


def month_ends(index):
    days = pd.Series(index, index=index)
    return pd.DatetimeIndex(sorted(days.groupby([index.year, index.month]).max()))


def rebalance_dates(calendar, rebalance):
    return month_ends(calendar) if rebalance == "month_end" else calendar


def positions(alpha, panel, dates):
    """dates x instruments of alpha targets at each date in `dates`."""
    rows = {d: alpha.signal(cross_section(panel, d)) for d in dates if d in panel.index}
    return pd.DataFrame(rows).T.reindex(columns=panel.columns.get_level_values("instrument").unique()).fillna(0.0)


def backtest(pos, close, cost_bps=COST_BPS):
    """Net daily returns: targets set at t earn t+1, held until the next target, cost on traded weight."""
    held = pos.reindex(close.index).ffill().fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (held.shift(1) * ret).sum(axis=1) - turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    return pnl.rename("return")


def window_returns(alpha, panel, close, start, end, dates, cost_bps=COST_BPS):
    """Net returns from start to end, entering with the last position set before `start`."""
    dates = pd.DatetimeIndex(dates)
    keep = dates[(dates >= start) & (dates <= end)]
    before = dates[dates < start][-1:]
    pos = positions(alpha, panel, before.append(keep))
    first = before[0] if len(before) else start
    return backtest(pos, close.loc[first:end], cost_bps).loc[start:]


def forward_returns(close, horizon):
    """Return from t to t + horizon, dated t; the label for fit()."""
    return close.shift(-horizon) / close - 1


def walk_forward(make_alpha, grid, panel, close, n_splits, test_size, rebalance="month_end", horizon=21,
                 cost_bps=COST_BPS, registry=None, universe=None):
    """Rolling train/test over `grid`: fit every variant on the train window, keep the best train Sharpe, hold it on test.

    Each fold's training rows are purged by `horizon` sessions before the
    test window so a forward-return label cannot cross into it. Returns
    (table, stitched OOS returns, per-variant full-window returns).
    """
    cal = panel.index
    X = long_panel(panel)
    y = long_panel(pd.concat({"y": forward_returns(close, horizon)}, axis=1, names=["feature", "instrument"]))["y"]
    dates = rebalance_dates(cal, rebalance)
    split = WalkForwardSplit(n_splits=n_splits, test_size=test_size, prediction_times=cal,
                             evaluation_times=cal + pd.tseries.offsets.BDay(horizon), purge_horizon="0D")
    rows, picked, full = [], [], {}
    for k, (tr, te) in enumerate(split.split(np.zeros((len(cal), 1)))):
        train, test = cal[tr], cal[te]
        scores = {}
        for i, params in enumerate(grid):
            a = make_alpha(**params).fit(X.loc[train], y.loc[train])
            r = backtest(positions(a, panel, dates[dates.isin(train)]), close.loc[train], cost_bps)
            scores[i] = (sharpe(r[r != 0]), a)
        best = max(scores, key=lambda i: scores[i][0] if np.isfinite(scores[i][0]) else -np.inf)
        a = scores[best][1]
        r = window_returns(a, panel, close, test[0], test[-1], dates, cost_bps)
        picked.append(r)
        rows.append({"fold": k + 1, "train_start": train[0].date(), "train_end": train[-1].date(),
                     "test_start": test[0].date(), "test_end": test[-1].date(), "picked": json.dumps(grid[best]),
                     "picked_train_sharpe": scores[best][0], "picked_test_sharpe": sharpe(r[r != 0])})
    for params in grid:
        a = make_alpha(**params).fit(X, y)
        r = backtest(positions(a, panel, dates), close, cost_bps)
        full[json.dumps(params)] = r
        if registry is not None:
            registry.record(str(a), params, universe or list(close.columns), (cal[0].date(), cal[-1].date()), r,
                            tags={"stage": "walk_forward_grid"})
    oos = pd.concat(picked)
    return pd.DataFrame(rows).set_index("fold"), oos, full


class Sleeve(Strategy):
    """Any Alpha as a framework sleeve: rebuild features from the as-of view at each rebalance, send its weights."""

    def __init__(self, alpha, store, rebalance="month_end", series=None):
        self.alpha, self.store, self.rebalance, self.series = alpha, store, rebalance, series
        self.name = str(alpha)
        self.targets = {}

    def on_bar(self, asof, bars):
        if self.rebalance == "month_end" and not bars.is_month_end(asof):
            return None
        panel = self.store.build(Raw.from_bars(bars, self.series))
        w = self.alpha.signal(cross_section(panel, asof)).reindex(bars.instruments).fillna(0.0)
        self.targets[asof] = w
        return w.to_dict()


class LockError(Exception):
    """The lock file does not describe the window being asked for."""


class UntouchedWindowUsed(Exception):
    """The final window has already been opened."""


class Untouched:
    """The final window: locked by hash on first sight, opened once, refused after that.

    lock() writes start, end, session count and the data hash, plus a hash
    of the whole body (those four, the opened_at stamp and the result), so
    nulling opened_at by hand breaks the hash too. Locking again with
    anything different raises; so does a lock file whose hash no longer
    matches its contents. open(fn) runs fn
    once and stores its result; a second open raises, and the stored
    result is what a rerun reports.
    """

    def __init__(self, path):
        self.path = path

    KEYS = ("start", "end", "n_sessions", "data_hash")

    @classmethod
    def digest(cls, body):
        keys = cls.KEYS + ("opened_at", "result")
        return hashlib.sha256(json.dumps({k: body.get(k) for k in keys}, sort_keys=True,
                                         default=str).encode()).hexdigest()[:16]

    def read(self):
        with open(self.path) as fh:
            body = json.load(fh)
        if body.get("hash") != self.digest(body):
            raise LockError(f"{self.path}: contents do not match their hash")
        return body

    def write(self, body):
        body["hash"] = self.digest(body)
        with open(self.path, "w") as fh:
            json.dump(body, fh, indent=2, default=str)

    def lock(self, calendar, frac=0.2, data_hash=""):
        start = calendar[len(calendar) - int(round(len(calendar) * frac))]
        body = {"start": str(start.date()), "end": str(calendar[-1].date()),
                "n_sessions": int((calendar >= start).sum()), "data_hash": data_hash, "opened_at": None,
                "result": None}
        if os.path.exists(self.path):
            have = self.read()
            if any(have[k] != body[k] for k in self.KEYS):
                raise LockError(f"{self.path} locks {have['start']}..{have['end']} on data {have['data_hash']}, "
                                f"asked for {body['start']}..{body['end']} on {data_hash}")
            return have
        self.write(body)
        return body

    @property
    def window(self):
        b = self.read()
        return pd.Timestamp(b["start"]), pd.Timestamp(b["end"])

    @property
    def opened(self):
        return self.read()["opened_at"] is not None

    @property
    def result(self):
        return self.read()["result"]

    def open(self, fn):
        """Run fn(start, end) once, store its JSON-able result, refuse ever after."""
        body = self.read()
        if body["opened_at"] is not None:
            raise UntouchedWindowUsed(f"{self.path} was opened at {body['opened_at']}; the window is spent")
        body["opened_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        self.write(body)
        body["result"] = fn(pd.Timestamp(body["start"]), pd.Timestamp(body["end"]))
        self.write(body)
        return body["result"]
