"""Experiment registry: every backtest or model run logs what was looked at.

One JSON line per run in runs.jsonl, and the run's return stream in
returns/<id>.parquet so any DSR can be recomputed from disk. A run's key is
the hash of its name, config, universe, window and code (code_hash of the
rule, or the file that logged it), so an edited rule is a new key; rerunning
the same key replaces its row and its stream, so the trial count is the
number of distinct keys, and keys with an identical return stream count once,
the same rule as framework/book/validate.py::trials(). The deflated Sharpe is
validate's deflated_sharpe_ratio (purgedcv) with that count and the variance
of the logged daily Sharpes.

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

HARNESS = {__name__, "research.alpha.base"}  # frames skipped when looking for the file that logged a run
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


class Registry:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.file = os.path.join(path, "runs.jsonl")
        self.streams = os.path.join(path, "returns")

    @staticmethod
    def key(name, config, universe, window, code=None):
        body = {"name": name, "config": config, "universe": sorted(universe), "window": [str(w) for w in window]}
        if code is not None:
            body["code"] = code
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def record(self, name, config, universe, window, returns, metrics=None, tags=None, code=None):
        """Log one run and store its return stream; returns the entry with the trial count after it.

        code is code_hash() of the rule; left out, it is the hash of the file
        that called record (or called the harness function that did).
        """
        code = code if code is not None else _caller_code()
        raw = pd.Series(returns)
        r = raw.dropna()
        r = r[r != 0]
        run_id = self.key(name, config, universe, window, code)
        os.makedirs(self.streams, exist_ok=True)  # the stream first: a row is never logged without one
        raw.rename("return").to_frame().to_parquet(os.path.join(self.streams, f"{run_id}.parquet"))
        entry = {"id": run_id, "name": name, "config": config, "code": code,
                 "universe": sorted(universe), "window": [str(w) for w in window],
                 "metrics": metrics or (validate.stats(r) if len(r) > 1 else {}),
                 "sharpe_daily": float(r.mean() / r.std()) if len(r) > 1 and r.std() > 0 else float("nan"),
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

    def trials(self):
        """Distinct keys, identical return streams counted once."""
        t = self.runs()
        if t.empty:
            return 0
        return int(len(t.drop_duplicates(["sharpe_daily", "n_obs"])))

    def dsr(self, returns, extra_trials=0):
        """PSR and DSR of one return series against every trial logged here."""
        r = pd.Series(returns).dropna()
        r = r[r != 0].to_numpy()
        t = self.runs().drop_duplicates(["sharpe_daily", "n_obs"])  # same dedupe as validate.trials()
        n = len(t) + extra_trials
        var = float(t["sharpe_daily"].var(ddof=1)) if len(t) > 1 else 0.0
        if not np.isfinite(var):
            var = 0.0
        return {"sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)),  # ddof=1, same as record() and validate.stats
                "psr": float(validate.probabilistic_sharpe_ratio(r, 0.0)),
                "dsr": float(validate.deflated_sharpe_ratio(r, max(n, 1), var)),
                "n_trials": int(n), "var_sharpe": var, "n_obs": int(len(r))}
