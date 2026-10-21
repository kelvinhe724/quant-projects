"""Alpha: signal(features) -> target weights, plus the adapter, the walk-forward and the untouched window.

The research path is vectorised: positions at each rebalance date from the
cross-section of the feature panel, held until the next, one-day lag,
proportional cost. The book path wraps the same Alpha in Sleeve, a
framework Strategy that rebuilds the features from the engine's as-of view
at each rebalance and hands the engine the same weights; check.py proves
the two paths return identical targets.

Every backtest, window_returns and walk_forward call is an evaluation and
logs itself to the attached registry (passed in, or research.registry.use),
return stream included; with none attached they raise UnloggedEvaluation
unless called with log=False. Untouched.lock records which registry the
window belongs to; Untouched.open refuses any other, an empty one, or one
that already holds a row over the window, and records the registry's trial
count and a hash of its runs.jsonl at the moment of opening in the lock.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from purgedcv import WalkForwardSplit

from framework.book.strategies import month_ends
from framework.engine import Strategy, sharpe
from research.features.store import Raw, cross_section, long_panel, panel_hash
from research.registry.experiments import attached, code_hash, file_sha256

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


def rebalance_dates(calendar, rebalance):
    return month_ends(calendar) if rebalance == "month_end" else calendar


def positions(alpha, panel, dates):
    """dates x instruments of alpha targets at each date in `dates`."""
    rows = {d: alpha.signal(cross_section(panel, d)) for d in dates if d in panel.index}
    return pd.DataFrame(rows).T.reindex(columns=panel.columns.get_level_values("instrument").unique()).fillna(0.0)


class UnloggedEvaluation(RuntimeError):
    """An evaluation with no registry attached and no explicit log=False."""


def _registry(registry, log):
    """The registry this evaluation logs to; None only when the caller passed the literal log=False."""
    reg = attached(registry)
    if reg is None and log is not False:
        raise UnloggedEvaluation("no registry attached: pass registry=Registry(path), call "
                                 "research.registry.use(path), or pass log=False to evaluate unlogged")
    return reg


def _returns(pos, close, cost_bps):
    held = pos.reindex(close.index).ffill().fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (held.shift(1) * ret).sum(axis=1) - turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    return pnl.rename("return")


def _log(reg, r, pos, cost_bps, alpha, config, universe, tags):
    """One registry row per evaluation, keyed on the alpha's code, the cost and a hash of the positions it held.

    An empty window (a forward window that has not started) returns its empty stream unlogged: nothing was seen.
    """
    if reg is not None and len(r):
        reg.record(str(alpha) if alpha is not None else "backtest",
                   {**(config or {}), "cost_bps": cost_bps, "positions": panel_hash(pos)},
                   universe, (r.index[0], r.index[-1]), r, tags=tags,
                   code=code_hash(alpha) if alpha is not None else None)
    return r


def backtest(pos, close, cost_bps=COST_BPS, registry=None, log=True, alpha=None, config=None, universe=None,
             tags=None):
    """Net daily returns: targets set at t earn t+1, held until the next target, cost on traded weight.

    Logged to the attached registry, always; with none attached it raises
    unless log=False. `alpha` names the rule and hashes its code; without it
    the row is named "backtest" and keyed on the positions and the calling file.
    """
    reg = _registry(registry, log)
    return _log(reg, _returns(pos, close, cost_bps), pos, cost_bps, alpha, config, universe or list(close.columns),
                tags)


def window_returns(alpha, panel, close, start, end, dates, cost_bps=COST_BPS, registry=None, log=True, config=None,
                   universe=None, tags=None):
    """Net returns from start to end, entering with the last position set before `start`; logged as backtest is."""
    reg = _registry(registry, log)
    dates = pd.DatetimeIndex(dates)
    keep = dates[(dates >= start) & (dates <= end)]
    before = dates[dates < start][-1:]
    pos = positions(alpha, panel, before.append(keep))
    first = before[0] if len(before) else start
    r = _returns(pos, close.loc[first:end], cost_bps).loc[start:]
    return _log(reg, r, pos, cost_bps, alpha, config, universe or list(close.columns), tags)


def forward_returns(close, horizon):
    """Return from t to t + horizon, dated t; the label for fit()."""
    return close.shift(-horizon) / close - 1


def walk_forward(make_alpha, grid, panel, close, n_splits, test_size, rebalance="month_end", horizon=21,
                 cost_bps=COST_BPS, registry=None, universe=None, score=None, log=True):
    """Rolling train/test over `grid`: fit every variant on the train window, keep the best train Sharpe, hold it on test.

    Each fold's training rows are purged by `horizon` sessions before the
    test window so a forward-return label cannot cross into it. `score`
    maps a stream of bar returns to the number a fold is judged and
    reported on; the default is the Sharpe of the nonzero bars annualised
    by sqrt(252), which is only a daily Sharpe when the bars are days, so
    an intraday project passes its own. Returns (table, stitched OOS
    returns, per-variant full-window returns).

    Every evaluation is logged: each variant on each fold's training rows
    (stage walk_forward_train), the pick on each test window
    (walk_forward_test) and every variant on the whole window
    (walk_forward_grid). With no registry attached it raises before any
    work unless log=False.
    """
    reg = _registry(registry, log)
    universe = universe or list(close.columns)
    score = score or (lambda r: sharpe(r[r != 0]))
    cal = panel.index
    X = long_panel(panel)
    y = long_panel(pd.concat({"y": forward_returns(close, horizon)}, axis=1, names=["feature", "instrument"]))["y"]
    dates = rebalance_dates(cal, rebalance)
    # A row's label is known `horizon` sessions later, on the calendar itself, not `horizon` business
    # days later: a holiday inside the horizon would otherwise shorten the purge by a session.
    evaluation = cal[horizon:].append(pd.DatetimeIndex([cal[-1] + pd.tseries.offsets.BDay(k)
                                                        for k in range(1, horizon + 1)]))
    split = WalkForwardSplit(n_splits=n_splits, test_size=test_size, prediction_times=cal,
                             evaluation_times=evaluation, purge_horizon="0D")
    rows, picked, full = [], [], {}
    for k, (tr, te) in enumerate(split.split(np.zeros((len(cal), 1)))):
        train, test = cal[tr], cal[te]
        scores = {}
        for i, params in enumerate(grid):
            a = make_alpha(**params).fit(X.loc[train], y.loc[train])
            pos = positions(a, panel, dates[dates.isin(train)])
            r = _log(reg, _returns(pos, close.loc[train], cost_bps), pos, cost_bps, a, params, universe,
                     {"stage": "walk_forward_train", "fold": k + 1})
            scores[i] = (score(r), a)
        best = max(scores, key=lambda i: scores[i][0] if np.isfinite(scores[i][0]) else -np.inf)
        a = scores[best][1]
        r = window_returns(a, panel, close, test[0], test[-1], dates, cost_bps, registry=reg, log=log,
                           config=grid[best], universe=universe, tags={"stage": "walk_forward_test", "fold": k + 1})
        picked.append(r)
        rows.append({"fold": k + 1, "train_start": train[0].date(), "train_end": train[-1].date(),
                     "test_start": test[0].date(), "test_end": test[-1].date(), "picked": json.dumps(grid[best]),
                     "picked_train_sharpe": scores[best][0], "picked_test_sharpe": score(r)})
    for params in grid:
        a = make_alpha(**params).fit(X, y)
        pos = positions(a, panel, dates)
        full[json.dumps(params)] = _log(reg, _returns(pos, close, cost_bps), pos, cost_bps, a, params, universe,
                                        {"stage": "walk_forward_grid"})
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

    lock() writes start, end, session count and the data hash, the real path
    of the registry the window belongs to and a sha256 of its runs.jsonl at
    that moment, plus a hash of the whole body (those, the opened_at stamp,
    the result and what open() adds), so nulling opened_at by hand breaks the
    hash too. Locking again with anything different raises; so does a lock
    file whose hash no longer matches its contents. open(fn) runs fn once and
    stores its result; a second open raises, and the stored result is what a
    rerun reports.
    """

    def __init__(self, path):
        self.path = path

    KEYS = ("start", "end", "n_sessions", "data_hash")
    # fields added after the first locks were written: hashed when present, so a lock without them still verifies
    LATER = ("n_trials", "registry", "registry_rel", "registry_sha256_at_lock", "registry_bytes_at_lock",
             "registry_sha256_at_open")

    @classmethod
    def digest(cls, body):
        keys = cls.KEYS + ("opened_at", "result") + tuple(k for k in cls.LATER if k in body)
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

    def lock(self, calendar, frac=0.2, data_hash="", registry=None):
        """Lock the last `frac` of `calendar`. A new lock needs a registry (passed in or research.registry.use):
        the window is pinned to it, so the trials behind its one read cannot be logged somewhere else."""
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
        reg = _registry(registry, True)
        body.update(self._where(reg), registry_sha256_at_lock=file_sha256(reg.file),
                    registry_bytes_at_lock=os.path.getsize(reg.file) if os.path.exists(reg.file) else 0)
        self.write(body)
        return body

    def _where(self, reg):
        """The registry's real path, and its path from the lock's folder (the one that survives moving the repo)."""
        path = os.path.realpath(reg.path)
        return {"registry": path, "registry_rel": os.path.relpath(path, os.path.realpath(os.path.dirname(self.path)))}

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

    def _check_registry(self, body, reg):
        """Refuse an open on the wrong, an empty, a rewritten or an already-peeking registry."""
        where = self._where(reg)
        path = where["registry"]
        if "registry" in body and path != body["registry"] and where["registry_rel"] != body.get("registry_rel"):
            raise LockError(f"{self.path} belongs to the registry {body['registry']}, not {path}")
        if reg.trials() == 0:
            raise LockError(f"{self.path}: the registry {path} holds no counted trial; an open would freeze "
                            "n_trials 0")
        if body.get("registry_sha256_at_lock"):
            with open(reg.file, "rb") as fh:
                head = fh.read(body["registry_bytes_at_lock"])
            if hashlib.sha256(head).hexdigest() != body["registry_sha256_at_lock"]:
                raise LockError(f"{self.path}: {reg.file} no longer starts with the rows it held at the lock; "
                                "the registry was rewritten, not appended to")
        w0, w1 = pd.Timestamp(body["start"]).date(), pd.Timestamp(body["end"]).date()
        runs = reg.runs()
        for _, row in runs.iterrows():
            a, z = (pd.Timestamp(x).date() for x in row["window"])
            if a <= w1 and z >= w0:
                raise LockError(f"{self.path}: registry row {row['id']} ({row['name']}, {row['window'][0]} to "
                                f"{row['window'][1]}) already evaluated part of the window {w0} to {w1}")

    def open(self, fn, registry=None):
        """Run fn(start, end) once, store its JSON-able result, refuse ever after.

        An open needs a registry (passed in or research.registry.use; there is
        no log=False here): the one the lock names, holding at least one
        counted trial, its runs.jsonl an append of what it held at the lock,
        and no row whose window overlaps the untouched one (rows fn itself
        writes come after this check). A lock written before the registry was
        recorded takes the registry it is opened with. The trial count and a
        sha256 of runs.jsonl at the moment of opening, before fn runs, are
        written into the lock with the stamp, so the DSR denominator of the
        one read is frozen with its result.
        """
        body = self.read()
        if body["opened_at"] is not None:
            raise UntouchedWindowUsed(f"{self.path} was opened at {body['opened_at']}; the window is spent")
        reg = _registry(registry, True)
        self._check_registry(body, reg)
        body.update({k: v for k, v in self._where(reg).items() if k not in body})  # a lock from before lock() named one
        body["opened_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        body["n_trials"] = reg.trials()
        body["registry_sha256_at_open"] = file_sha256(reg.file)
        self.write(body)
        body["result"] = fn(pd.Timestamp(body["start"]), pd.Timestamp(body["end"]))
        self.write(body)
        return body["result"]
