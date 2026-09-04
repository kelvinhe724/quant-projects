"""Random-walk Kalman filter for a time-varying hedge ratio, written out in numpy.

Model, one pair at a time, with y = log price A and x = log price B:

    state       s_t = [beta_t, alpha_t]
    transition  s_t = s_{t-1} + w_t,          w_t ~ N(0, Q),  Q = noise_ratio * obs_var * I
    observation y_t = beta_t * x_t + alpha_t + v_t,   v_t ~ N(0, obs_var)

noise_ratio = 0 makes the state constant, and the filter collapses to recursive
least squares: with a diffuse prior it reproduces the full-sample OLS fit.
Larger ratios let the hedge ratio drift faster and are what run.py tunes.
"""
import numpy as np
import pandas as pd


def ols(y, x):
    """Fit y = beta * x + alpha by least squares; return (beta, alpha, residual variance)."""
    X = np.column_stack([np.asarray(x, float), np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    resid = np.asarray(y, float) - X @ coef
    return float(coef[0]), float(coef[1]), float(resid.var(ddof=2))


def kalman_filter(y, x, noise_ratio, obs_var, prior_var=1e6):
    """Filter y on x and return a frame of innovation, its variance, and the state path.

    Row t holds the prediction error e_t = y_t - (beta_{t-1} x_t + alpha_{t-1}) made
    before y_t is seen, and the state after y_t is absorbed. Nothing at row t reads
    y or x beyond t.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    state = np.zeros(2)
    P = np.eye(2) * prior_var
    Q = np.eye(2) * noise_ratio * obs_var
    out = np.empty((n, 4))
    for t in range(n):
        H = np.array([x[t], 1.0])
        P = P + Q
        e = y[t] - H @ state
        S = H @ P @ H + obs_var
        K = P @ H / S
        state = state + K * e
        P = P - np.outer(K, H) @ P
        out[t] = (e, S, state[0], state[1])
    return pd.DataFrame(out, columns=["innovation", "innovation_var", "beta", "alpha"])


def kalman_spread(log_a, log_b, noise_ratio, obs_var):
    """Run the filter on a pair and return (spread, beta path) indexed like the inputs.

    The spread is the one-step-ahead prediction error, so it measures how far today's
    price sits from where yesterday's hedge ratio said it should be. Using the
    updated state instead would shrink the spread toward zero by construction.
    """
    f = kalman_filter(log_a.to_numpy(), log_b.to_numpy(), noise_ratio, obs_var)
    f.index = log_a.index
    return f["innovation"], f["beta"], f["alpha"]
