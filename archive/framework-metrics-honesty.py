"""Retired from framework/engine/metrics.py on 2026-09-04.

These were the engine's own honesty corrections and walk-forward splitter.
They are replaced by purgedcv (probabilistic_sharpe_ratio, deflated_sharpe_ratio,
minimum_backtest_length, WalkForwardSplit, CombinatorialPurgedCV). Kept here,
not deleted, so the numbers can be cross-checked against the library.

Deflated and probabilistic Sharpe follow Bailey and Lopez de Prado (2014),
"The Deflated Sharpe Ratio". Sharpe ratios inside those two functions are per
period, not annualised.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

TRADING_DAYS = 252


def sharpe(r, periods=TRADING_DAYS):
    r = pd.Series(r).dropna()
    s = r.std()
    return float(r.mean() / s * np.sqrt(periods)) if s > 0 else np.nan


def probabilistic_sharpe(sr, n, skewness=0.0, kurt=3.0, sr_benchmark=0.0):
    """P(true SR > sr_benchmark) given an observed per-period SR over n observations."""
    var = (1 - skewness * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)
    if var <= 0:
        return np.nan
    return float(norm.cdf((sr - sr_benchmark) / np.sqrt(var)))


def expected_max_sharpe(n_trials, sr_var):
    """Expected maximum per-period SR among n_trials strategies whose true SR is zero."""
    if n_trials < 2:
        return 0.0
    euler = 0.5772156649015329
    z = (1 - euler) * norm.ppf(1 - 1 / n_trials) + euler * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_var) * z)


def deflated_sharpe(returns, n_trials, sr_var=None, periods=TRADING_DAYS):
    """Deflated Sharpe of the best of n_trials candidates, with the pieces used to build it.

    sr_var is the variance of the candidates' per-period Sharpe ratios; if it
    is not supplied the variance under the null (1/(n-1)) is used, which is the
    right choice when the candidates were independent draws.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    sr = r.mean() / r.std()
    if sr_var is None:
        sr_var = 1 / (n - 1)
    sr0 = expected_max_sharpe(n_trials, sr_var)
    sk, ku = float(skew(r)), float(kurtosis(r, fisher=False))
    return {
        "sharpe": sr * np.sqrt(periods),
        "expected_max_sharpe": sr0 * np.sqrt(periods),
        "psr": probabilistic_sharpe(sr, n, sk, ku, 0.0),
        "dsr": probabilistic_sharpe(sr, n, sk, ku, sr0),
        "n_trials": n_trials,
    }


def bootstrap_ci(returns, stat=sharpe, n_boot=1000, block=20, level=0.95, seed=0):
    """Circular block bootstrap confidence interval for `stat` of a return series."""
    r = np.asarray(pd.Series(returns).dropna())
    n = len(r)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        out[b] = stat(r[idx])
    lo, hi = np.nanpercentile(out, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


class WalkForward:
    """Train/test splits over a calendar with a final window nothing may touch until the end.

    `holdout` is the fraction (or number of periods) reserved at the end. folds()
    yields (train_index, test_index) pairs that never reach into the holdout;
    reading `holdout_index` marks it as spent, after which folds() refuses to run.
    """

    def __init__(self, index, n_folds=5, holdout=0.2, mode="expanding", train_periods=None):
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'")
        self.index = pd.DatetimeIndex(index)
        cut = int(holdout if holdout >= 1 else round(len(self.index) * holdout))
        self._work, self._holdout = self.index[:len(self.index) - cut], self.index[len(self.index) - cut:]
        self.n_folds, self.mode, self.train_periods = n_folds, mode, train_periods
        self.holdout_spent = False

    def folds(self):
        if self.holdout_spent:
            raise RuntimeError("holdout already read; no more fitting allowed")
        edges = np.linspace(0, len(self._work), self.n_folds + 2).astype(int)
        for k in range(1, self.n_folds + 1):
            lo = 0 if self.mode == "expanding" else max(0, edges[k] - (self.train_periods or edges[1]))
            yield self._work[lo:edges[k]], self._work[edges[k]:edges[k + 1]]

    @property
    def holdout_index(self):
        self.holdout_spent = True
        return self._holdout

    @property
    def holdout_start(self):
        return self._holdout[0]
