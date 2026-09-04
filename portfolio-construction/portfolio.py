"""Portfolio rules and the estimators that feed them.

Every rule maps a mean vector and a covariance matrix (or just the covariance)
to a weight vector summing to one. Estimation from a return window lives in
`estimate`, so the rules can be tested against known moments.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def equal_weight(n):
    """Hold 1/N in each asset."""
    return np.full(n, 1.0 / n)


def tangency(mu, sigma):
    """Maximise Sharpe with no constraints: w proportional to inv(Sigma) mu, scaled to sum one."""
    raw = np.linalg.solve(sigma, mu)
    return raw / raw.sum()


def min_variance(sigma):
    """Minimise variance subject only to full investment."""
    raw = np.linalg.solve(sigma, np.ones(len(sigma)))
    return raw / raw.sum()


def long_only_tangency(mu, sigma):
    """Maximise Sharpe with weights in [0, 1] summing to one."""
    n = len(mu)

    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w @ sigma @ w)

    res = minimize(neg_sharpe, equal_weight(n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def inverse_vol(sigma):
    """Weight each asset by one over its volatility, ignoring correlations."""
    raw = 1.0 / np.sqrt(np.diag(sigma))
    return raw / raw.sum()


def risk_contributions(w, sigma):
    """Share of portfolio variance attributable to each position."""
    marginal = sigma @ w
    return w * marginal / (w @ marginal)


def erc(sigma):
    """Equal risk contribution: every asset accounts for 1/N of portfolio variance.

    Uses Spinu's convex form, minimise 0.5 w'Sigma w minus the mean log weight, whose
    unique positive minimiser has equal risk contributions once rescaled to sum one.
    """
    n = len(sigma)

    def objective(w):
        return 0.5 * w @ sigma @ w - np.log(w).sum() / n

    def gradient(w):
        return sigma @ w - 1.0 / (n * w)

    res = minimize(objective, inverse_vol(sigma) / np.sqrt(np.trace(sigma) / n), jac=gradient,
                   method="L-BFGS-B", bounds=[(1e-8, None)] * n,
                   options={"maxiter": 1000, "gtol": 1e-10, "ftol": 1e-14})
    return res.x / res.x.sum()


def ledoit_wolf(returns):
    """Shrink the sample covariance toward a scaled identity, intensity chosen by Ledoit-Wolf."""
    lw = LedoitWolf(assume_centered=False).fit(returns)
    return lw.covariance_, lw.shrinkage_


def james_stein(returns, sigma):
    """Shrink sample means toward the grand mean with Jorion's data-driven intensity.

    The intensity rises when the means are close together relative to their
    sampling error, which is exactly when the sample dispersion is mostly noise.
    """
    t, n = returns.shape
    mu = returns.mean(axis=0)
    grand = mu.mean()
    gap = mu - grand
    dist = gap @ np.linalg.solve(sigma, gap)
    intensity = (n + 2) / ((n + 2) + t * dist)
    return (1 - intensity) * mu + intensity * grand, intensity


def estimate(window, shrink=False):
    """Return the mean vector and covariance from a window of returns, shrunk or plain."""
    x = np.asarray(window, dtype=float)
    if not shrink:
        return x.mean(axis=0), np.cov(x, rowvar=False, ddof=1)
    sigma, _ = ledoit_wolf(x)
    mu, _ = james_stein(x, sigma)
    return mu, sigma


# name -> (needs shrinkage, weight function of (mu, sigma))
RULES = {
    "1/N": (False, lambda mu, s: equal_weight(len(mu))),
    "tangency": (False, tangency),
    "tangency long-only": (False, long_only_tangency),
    "min variance": (False, lambda mu, s: min_variance(s)),
    "inverse vol": (False, lambda mu, s: inverse_vol(s)),
    "ERC": (False, lambda mu, s: erc(s)),
    "shrunk tangency": (True, tangency),
    "shrunk min variance": (True, lambda mu, s: min_variance(s)),
}


def weights(rule, window):
    """Estimate moments on the window and apply the named rule."""
    shrink, fn = RULES[rule]
    mu, sigma = estimate(window, shrink=shrink)
    return fn(mu, sigma)
