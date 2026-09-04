"""Odds to probability, three de-vigging methods, and calibration measurement.

Decimal odds o pay o units per unit staked including the stake, so 1/o is the
implied probability with the bookmaker's margin still in it. The three implied
probabilities on a football match sum to more than one; how you remove that
excess is the de-vigging choice, and the methods disagree most at the extremes.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

OUTCOMES = ("H", "D", "A")


def implied(odds):
    """Convert decimal odds to raw implied probabilities."""
    return 1.0 / np.asarray(odds, dtype=float)


def booksum(odds):
    """Sum the raw implied probabilities across the outcomes of each match."""
    return implied(odds).sum(axis=-1)


def overround(odds):
    """Bookmaker margin: how far the implied probabilities sum above one."""
    return booksum(odds) - 1.0


def devig_proportional(odds):
    """Scale every implied probability by the same factor so they sum to one."""
    q = implied(odds)
    return q / q.sum(axis=-1, keepdims=True)


def devig_power(odds, iters=60):
    """Logarithmic method: find k with sum(q_i**k) = 1 and return q_i**k.

    Because q_i < 1, raising to k > 1 shrinks long prices proportionally more
    than short ones, so this takes a bigger cut out of the longshot.
    """
    q = np.asarray(implied(odds), dtype=float)
    lo = np.full(q.shape[:-1], 0.2)
    hi = np.full(q.shape[:-1], 8.0)
    for _ in range(iters):
        k = 0.5 * (lo + hi)
        s = (q ** k[..., None]).sum(axis=-1)
        hi = np.where(s > 1.0, hi, k)
        lo = np.where(s > 1.0, k, lo)
    k = 0.5 * (lo + hi)
    p = q ** k[..., None]
    return p / p.sum(axis=-1, keepdims=True)


def devig_shin(odds, iters=80, return_z=False):
    """Shin's method: back out the insider fraction z, then the fair probabilities.

    Shin models the margin as protection against a share z of bettors who know
    the outcome. Solving sum(p_i) = 1 for z gives probabilities that shade the
    longshot down harder than proportional scaling does.
    """
    q = np.asarray(implied(odds), dtype=float)
    total = q.sum(axis=-1, keepdims=True)
    lo = np.zeros(q.shape[:-1])
    hi = np.full(q.shape[:-1], 0.9)
    for _ in range(iters):
        z = 0.5 * (lo + hi)
        s = _shin_probs(q, total, z).sum(axis=-1)
        hi = np.where(s > 1.0, hi, z)
        lo = np.where(s > 1.0, z, lo)
    z = 0.5 * (lo + hi)
    p = _shin_probs(q, total, z)
    p = p / p.sum(axis=-1, keepdims=True)
    return (p, z) if return_z else p


def _shin_probs(q, total, z):
    zc = z[..., None]
    root = np.sqrt(zc ** 2 + 4.0 * (1.0 - zc) * q ** 2 / total)
    return (root - zc) / (2.0 * (1.0 - zc))


def shin_quotes(p, z):
    """Forward Shin map: fair probabilities and an insider share to quoted odds.

    Inverting devig_shin, used by check.py to plant a book with a known z.
    """
    p = np.asarray(p, dtype=float)
    s = np.sqrt(p * ((1.0 - z) * p + z))
    q = s * s.sum(axis=-1, keepdims=True)
    return 1.0 / q


METHODS = {
    "proportional": devig_proportional,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(odds, method="shin"):
    """Remove the margin using the named method."""
    return METHODS[method](odds)


def outcome_matrix(results):
    """One-hot the H/D/A result column in the same column order as the odds."""
    r = pd.Series(results).to_numpy()
    return np.stack([(r == o).astype(float) for o in OUTCOMES], axis=-1)


def brier(p, y):
    """Multi-class Brier score per match: squared error summed over the three outcomes."""
    return ((np.asarray(p) - np.asarray(y)) ** 2).sum(axis=-1)


def log_loss(p, y, floor=1e-12):
    """Negative log probability the book assigned to what actually happened."""
    p = np.clip(np.asarray(p), floor, 1.0)
    return -(np.asarray(y) * np.log(p)).sum(axis=-1)


def wilson(k, n, conf=0.95):
    """Wilson score interval for a binomial proportion, stable at small n."""
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(0.5 + conf / 2.0)
    phat = k / n
    denom = 1.0 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z / denom * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return centre - half, centre + half


DEFAULT_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 1.0]


def calibration(p, y, odds=None, edges=DEFAULT_EDGES):
    """Bin flattened predictions by probability and compare to realised frequency."""
    p = np.asarray(p).ravel()
    y = np.asarray(y).ravel()
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        lo, hi = wilson(k, n)
        row = {
            "bin": f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
            "n": n,
            "mean_p": float(p[m].mean()),
            "realised": k / n,
            "lo": lo,
            "hi": hi,
            "z": _bias_z(p[m], y[m]),
        }
        if odds is not None:
            o = np.asarray(odds).ravel()[m]
            row["roi"] = float((y[m] * o - 1.0).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["bias"] = out["realised"] - out["mean_p"]
    return out


def _bias_z(p, y):
    """Standardised gap between realised wins and the sum of predicted probabilities."""
    var = (p * (1 - p)).sum()
    if var <= 0:
        return np.nan
    return float((y.sum() - p.sum()) / np.sqrt(var))


def calibration_regression(p, y, floor=1e-6, cluster=True):
    """Logistic fit of the outcome on logit(p); slope 1 and intercept 0 means calibrated.

    A slope above one means the book's probabilities are too spread out: it is
    quoting longshots at probabilities higher than they deserve and favourites
    at probabilities lower than they deserve, which is the favourite-longshot bias.

    The three outcomes of a match are mechanically dependent, so standard errors
    are clustered by match. On simulated data the naive standard error runs about
    25% too small, which is enough to turn a null result significant.
    """
    p = np.clip(np.asarray(p), floor, 1 - floor)
    y = np.asarray(y)
    groups = np.repeat(np.arange(len(p)), p.shape[-1]) if p.ndim > 1 else None
    p, y = p.ravel(), y.ravel()
    x = sm.add_constant(np.log(p / (1 - p)))
    model = sm.Logit(y, x)
    if cluster and groups is not None:
        fit = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    else:
        fit = model.fit(disp=0)
    return {
        "intercept": float(fit.params[0]),
        "slope": float(fit.params[1]),
        "slope_se": float(fit.bse[1]),
        "slope_t": float((fit.params[1] - 1.0) / fit.bse[1]),
        "slope_p": float(2 * norm.sf(abs((fit.params[1] - 1.0) / fit.bse[1]))),
        "n": int(len(y)),
    }
