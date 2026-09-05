"""Experiment registry: every backtest or model run logs what was looked at.

One JSON line per run in runs.jsonl. A run's key is the hash of its name,
config, universe and window; rerunning the same key replaces its row, so
the trial count is the number of distinct keys, and keys with an identical
return stream count once, the same rule as framework/book/validate.py::trials().
The deflated Sharpe is validate's deflated_sharpe_ratio (purgedcv) with that
count and the variance of the logged daily Sharpes.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from framework.book import validate


class Registry:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.file = os.path.join(path, "runs.jsonl")

    @staticmethod
    def key(name, config, universe, window):
        body = json.dumps({"name": name, "config": config, "universe": sorted(universe),
                           "window": [str(w) for w in window]}, sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def record(self, name, config, universe, window, returns, metrics=None, tags=None):
        """Log one run; returns the entry with the trial count after it."""
        r = pd.Series(returns).dropna()
        r = r[r != 0]
        entry = {"id": self.key(name, config, universe, window), "name": name, "config": config,
                 "universe": sorted(universe), "window": [str(w) for w in window],
                 "metrics": metrics or validate.stats(r),
                 "sharpe_daily": float(r.mean() / r.std()) if len(r) > 1 and r.std() > 0 else float("nan"),
                 "n_obs": int(len(r)), "tags": tags or {},
                 "run_at": pd.Timestamp.now().isoformat(timespec="seconds")}
        with open(self.file, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        entry["n_trials"] = self.trials()
        return entry

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
        return {"sharpe": float(r.mean() / r.std() * np.sqrt(252)),
                "psr": float(validate.probabilistic_sharpe_ratio(r, 0.0)),
                "dsr": float(validate.deflated_sharpe_ratio(r, max(n, 1), var)),
                "n_trials": int(n), "var_sharpe": var, "n_obs": int(len(r))}
