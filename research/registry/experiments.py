"""Experiment registry: every backtest or model run logs what was looked at.

One JSON line per run in runs.jsonl, and the run's return stream in
returns/<id>.parquet, so the DSR of any row logged since 2d0b9a7 can be
recomputed from disk (older rows have no stream file). A run's key is the
hash of its name, config, universe, window, code (code_hash of the rule, or
the file that logged it) and a hash of the return stream itself, so an edited
rule is a new key even when the edit sits in a module the code hash does not
see; rerunning the same key replaces its row and its stream, so the trial
count is the number of distinct keys, and keys with an identical return
stream count once, the same rule as framework/book/validate.py::trials().
Walk-forward fold rows (stages walk_forward_train and walk_forward_test) stay
on disk as the audit trail but are not counted: they are the folds of a rule
already counted once, and their Sharpes on short windows would set the
variance. The deflated Sharpe is validate's deflated_sharpe_ratio (purgedcv)
with that count and the variance of the counted daily Sharpes; a stream too
close to constant to have a Sharpe is logged with a NaN Sharpe, counted, and
left out of the variance.

use(path) attaches a default registry for the process: research.alpha.base's
backtest, window_returns, walk_forward and Untouched.open log to it when they
are not handed one.
"""
import hashlib
import inspect
import json
import os
import sys
import types

import numpy as np
import pandas as pd

from framework.book import validate
from research.features.store import panel_hash

HARNESS = {__name__, "research.alpha.base"}  # frames skipped when looking for the file that logged a run
FOLDS = {"walk_forward_train", "walk_forward_test"}  # logged, not counted
_default = None


def use(path):
    """Attach the registry every harness evaluation in this process logs to; use(None) detaches it."""
    global _default
    _default = None if path is None else Registry(path)
    return _default


def attached(registry=None):
    """The registry passed in, else the one use() attached, else None."""
    return registry if registry is not None else _default


def _source(obj):
    """Source of a class or function; the bytes of its module's file when inspect cannot find the source."""
    try:
        return inspect.getsource(obj).encode()
    except (OSError, TypeError):
        path = getattr(sys.modules.get(obj.__module__), "__file__", None)
        # ponytail: no source and no module file (a REPL class, a builtin) hashes the qualified name only
        if not path:
            return f"{obj.__module__}.{obj.__qualname__}".encode()
        with open(path, "rb") as fh:
            return fh.read()


def code_hash(*objs):
    """16 hex characters of sha256 over the code behind a run.

    An instance stands for its class and a class for every class in its MRO
    (so a signal inherited from a parent counts), a function for its source, a
    module for its file's bytes. Pass several to cover a rule spread over files.
    ponytail: the source of what is passed, not of what it calls; a feature function or a module
    constant edited elsewhere is not seen here (the harness also keys on a hash of the positions).
    """
    h = hashlib.sha256()
    for obj in objs:
        if isinstance(obj, types.ModuleType):
            with open(obj.__file__, "rb") as fh:
                h.update(fh.read())
            continue
        if not isinstance(obj, (type, types.FunctionType)):
            obj = type(obj)
        for part in obj.__mro__ if isinstance(obj, type) else (obj,):
            h.update(_source(part))
    return h.hexdigest()[:16]


def _caller_code():
    """code_hash of the file of the first frame outside this module and the harness: the code that logged."""
    f = sys._getframe(1)
    while f is not None and f.f_globals.get("__name__") in HARNESS:
        f = f.f_back
    path = f.f_globals.get("__file__") if f is not None else None
    if not path:
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def daily_sharpe(r):
    """mean / std (ddof=1); NaN when the stream is too close to constant to have one.

    A constant stream's floating-point std is not zero (0.001 a day for 500 days gives about 4e-19), so a
    plain std > 0 guard lets through a Sharpe of 1e15 that would set every DSR in the registry to 0.
    """
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return float("nan")
    m, sd = float(r.mean()), float(r.std(ddof=1))
    return float("nan") if sd < 1e-12 * abs(m) or sd < 1e-15 else m / sd


def file_sha256(path):
    """sha256 of a file's bytes; None when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class Registry:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.file = os.path.join(path, "runs.jsonl")
        self.streams = os.path.join(path, "returns")

    @staticmethod
    def key(name, config, universe, window, code=None, stream=None):
        body = {"name": name, "config": config, "universe": sorted(universe), "window": [str(w) for w in window]}
        if code is not None:
            body["code"] = code
        if stream is not None:
            body["stream"] = stream
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def record(self, name, config, universe, window, returns, metrics=None, tags=None, code=None):
        """Log one run and store its return stream; returns the entry with the trial count after it.

        code is code_hash() of the rule; left out, it is the hash of the file
        that called record (or called the harness function that did). The key
        also carries a hash of the stream, so a rule edited where neither hash
        looks (a helper or a constant in another module) is a new row, not a
        replaced one.
        """
        code = code if code is not None else _caller_code()
        raw = pd.Series(returns)
        r = raw.dropna()
        r = r[r != 0]
        run_id = self.key(name, config, universe, window, code, panel_hash(raw.rename("return").to_frame()))
        os.makedirs(self.streams, exist_ok=True)  # the stream first: a row is never logged without one
        raw.rename("return").to_frame().to_parquet(os.path.join(self.streams, f"{run_id}.parquet"))
        entry = {"id": run_id, "name": name, "config": config, "code": code,
                 "universe": sorted(universe), "window": [str(w) for w in window],
                 "metrics": metrics or (validate.stats(r) if len(r) > 1 else {}),
                 "sharpe_daily": daily_sharpe(r),
                 "n_obs": int(len(r)), "tags": tags or {},
                 "run_at": pd.Timestamp.now().isoformat(timespec="seconds")}
        with open(self.file, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        entry["n_trials"] = self.trials()
        return entry

    def stream(self, run_id):
        """The return stream last logged under run_id, as it was handed to record()."""
        return pd.read_parquet(os.path.join(self.streams, f"{run_id}.parquet"))["return"]

    def runs(self):
        """Latest row per key."""
        if not os.path.exists(self.file):
            return pd.DataFrame(columns=["id", "name", "sharpe_daily", "n_obs"])
        rows = [json.loads(line) for line in open(self.file) if line.strip()]
        return pd.DataFrame(rows).drop_duplicates("id", keep="last").reset_index(drop=True)

    def counted(self):
        """The rows the trial count and the variance are taken over: latest per key, fold rows left out,
        identical return streams once (same dedupe as validate.trials())."""
        t = self.runs()
        if t.empty:
            return t
        if "tags" in t:
            fold = t["tags"].map(lambda x: isinstance(x, dict) and x.get("stage") in FOLDS)
            t = t[~fold]
        return t.drop_duplicates(["sharpe_daily", "n_obs"])

    def trials(self):
        """Distinct keys, fold rows left out, identical return streams counted once."""
        return int(len(self.counted()))

    def dsr(self, returns, extra_trials=0, n_trials=None):
        """PSR and DSR of one return series against every trial counted here.

        n_trials replaces the registry's count (the count a lock froze at its open, so a read that logs
        itself before calling dsr is deflated by the same number the lock holds); extra_trials is added
        either way. NaN Sharpes count as trials but not in the variance; n_nan_sharpe says how many.
        """
        r = pd.Series(returns).dropna()
        r = r[r != 0].to_numpy()
        t = self.counted()
        n = (len(t) if n_trials is None else int(n_trials)) + extra_trials
        s = t["sharpe_daily"].astype(float) if len(t) else pd.Series(dtype=float)
        var = float(s.var(ddof=1)) if s.notna().sum() > 1 else 0.0
        if not np.isfinite(var):
            var = 0.0
        return {"sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)),  # ddof=1, same as record() and validate.stats
                "psr": float(validate.probabilistic_sharpe_ratio(r, 0.0)),
                "dsr": float(validate.deflated_sharpe_ratio(r, max(n, 1), var)),
                "n_trials": int(n), "var_sharpe": var, "n_obs": int(len(r)), "n_nan_sharpe": int(s.isna().sum())}
