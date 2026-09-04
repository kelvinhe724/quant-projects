"""Gaussian HMM on daily returns: fit, state ordering, filtered vs smoothed probabilities.

Returns are in percent (100 * log return). EM on raw decimals is numerically
fragile because the variances are ~1e-4.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

TRADING_DAYS = 252


def fit(ret_pct, n_states=2, seed=0):
    """Fit a Gaussian HMM by EM and return it with states sorted by ascending volatility.

    Sorting is what makes state 0 mean the same thing across refits and seeds.
    EM is invariant to relabelling, so without it "state 0" is whatever k-means
    initialisation happened to produce.
    """
    x = np.asarray(ret_pct, dtype=float).reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=500,
                        tol=1e-6, random_state=seed)
    model.fit(x)
    return sort_states(model)


def sort_states(model):
    """Permute a fitted model so state indices go from lowest to highest volatility."""
    order = np.argsort(model.covars_.ravel())
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.means_ = model.means_[order]
    model.covars_ = np.array([np.diag(c) for c in model.covars_[order]]).reshape(-1, 1)
    return model


def describe(model):
    """Per-state table: annualised mean and vol, persistence, expected duration, stationary share."""
    sd = np.sqrt(model.covars_.ravel())
    p_stay = np.diag(model.transmat_)
    return pd.DataFrame({
        "mean_ann": model.means_.ravel() / 100 * TRADING_DAYS,
        "vol_ann": sd / 100 * np.sqrt(TRADING_DAYS),
        "p_stay": p_stay,
        "expected_duration": 1 / (1 - p_stay),
        "stationary_prob": stationary(model.transmat_),
    }, index=[f"state_{i}" for i in range(model.n_components)])


def stationary(transmat):
    """Long-run state distribution: the left eigenvector of the transition matrix with eigenvalue 1."""
    vals, vecs = np.linalg.eig(transmat.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def emission_logpdf(model, x):
    """Log density of each observation under each state, shape (T, K)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    sd = np.sqrt(model.covars_.ravel())
    return norm.logpdf(x, loc=model.means_.ravel(), scale=sd)


def filtered(model, x):
    """Forward-filtered state probabilities P(s_t | x_1..x_t), shape (T, K).

    This is the real-time quantity. Each row uses only observations up to and
    including that row, so it can be acted on the next day. hmmlearn's
    predict_proba is the smoothed P(s_t | x_1..x_T), which reads the future.
    """
    logb = emission_logpdf(model, x)
    A = model.transmat_
    out = np.empty_like(logb)
    alpha = model.startprob_ * np.exp(logb[0] - logb[0].max())
    out[0] = alpha / alpha.sum()
    for t in range(1, len(logb)):
        alpha = (out[t - 1] @ A) * np.exp(logb[t] - logb[t].max())
        out[t] = alpha / alpha.sum()
    return out


def smoothed(model, x):
    """Forward-backward P(s_t | x_1..x_T). Uses the whole sample; for charts and diagnostics only."""
    return model.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))


def bear_state(model):
    """Index of the highest-volatility state, which after sorting is always the last one."""
    return model.n_components - 1
