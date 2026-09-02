"""Nelson-Siegel fitting: a static daily benchmark, and sequential warm-start
fits penalised on scaled parameter changes (ridge or smooth L1).

Parameters are carried as phi = (beta0, beta1, beta2, log_lambda) so lambda
stays positive and its penalty acts on proportional moves.
"""
import numpy as np
from scipy.optimize import minimize

BOUNDS = [(-10, 10), (-5, 5), (-5, 5), (np.log(0.5), np.log(36))]


def ns_curve(tau, beta0, beta1, beta2, lam):
    x = np.asarray(tau, dtype=float) / lam
    f = (1 - np.exp(-x)) / x
    return beta0 + beta1 * f + beta2 * (f - np.exp(-x))


def _predict(phi, tau):
    return ns_curve(tau, phi[0], phi[1], phi[2], np.exp(phi[3]))


def _mse(phi, tau, y):
    r = _predict(phi, tau) - y
    return float(np.mean(r * r))


def default_start(tau, y, rng=None):
    """Starting vector read straight off the observed curve."""
    rng = rng or np.random.default_rng(0)
    order = np.argsort(tau)
    long_end, short_end = y[order[-1]], y[order[0]]
    return np.array([long_end + rng.normal(0, 0.01),
                     short_end - long_end + rng.normal(0, 0.01),
                     rng.normal(0, 0.02),
                     np.log(rng.uniform(2, 12))])


def fit_static(tau, y, x0=None, prev=None, penalty=None, strength=0.0,
               scale=None):
    """One daily fit. With prev/penalty set, adds strength * ||S dphi|| on the
    change from prev (squared for ridge, smooth-abs for l1)."""
    if x0 is None:
        x0 = default_start(tau, y)
    if scale is None:
        scale = np.ones(4)

    def obj(phi):
        loss = _mse(phi, tau, y)
        if prev is not None and strength > 0:
            d = (phi - prev) / scale
            if penalty == "l1":
                loss += strength * np.sum(np.sqrt(d * d + 1e-8))
            else:
                loss += strength * np.sum(d * d)
        return loss

    # beta2 and lambda trade off badly, so cold fits try several lambda starts
    starts = [x0]
    if prev is None:
        for lg in (np.log(2), np.log(6), np.log(12)):
            starts.append(np.concatenate([x0[:3], [lg]]))
    res = min((minimize(obj, s, method="L-BFGS-B", bounds=BOUNDS) for s in starts),
              key=lambda r: r.fun)
    phi = res.x
    return {"phi": phi, "beta0": phi[0], "beta1": phi[1], "beta2": phi[2],
            "lam": np.exp(phi[3]),
            "rmse": np.sqrt(_mse(phi, tau, y)),
            "success": bool(res.success), "nit": res.nit}


def fit_all_static(slices, mode="fixed", seed=7):
    """Independent daily fits.

    mode: 'fixed' = same start every day, 'random' = a new bounded random start
    each day, 'warm' = previous solution as the start but no change penalty.
    """
    rng = np.random.default_rng(seed)
    out, prev = [], None
    for d, tau, y, _, _ in slices:
        if mode == "warm" and prev is not None:
            x0 = prev
        elif mode == "random":
            x0 = default_start(tau, y, rng)
        else:
            x0 = default_start(tau, y)
        r = fit_static(tau, y, x0=x0)
        r["date"] = d
        prev = r["phi"]
        out.append(r)
    return out


def fit_sequential(slices, penalty="ridge", strength=1.0, scale=None,
                   burn_in=20, seed=7):
    """Chronological fits, each warm-started from and penalised against the
    previous day's phi. The penalty ramps linearly over the burn-in slices."""
    rng = np.random.default_rng(seed)
    out, prev = [], None
    for i, (d, tau, y, _, _) in enumerate(slices):
        lam_t = strength * min(1.0, (i + 1) / max(burn_in, 1))
        if prev is None:
            r = fit_static(tau, y, x0=default_start(tau, y, rng))
        else:
            r = fit_static(tau, y, x0=prev, prev=prev, penalty=penalty,
                           strength=lam_t, scale=scale)
        r["date"] = d
        prev = r["phi"]
        out.append(r)
    return out


def change_scale(results):
    """Per-parameter scale of daily phi changes, from a benchmark run.

    Median absolute deviation x1.4826, floored so no dimension divides by zero.
    """
    phis = np.array([r["phi"] for r in results])
    d = np.abs(np.diff(phis, axis=0))
    s = np.median(d, axis=0) * 1.4826
    return np.maximum(s, 1e-3)


def path_roughness(results, skip=0):
    """Mean scaled parameter jump per day; the stability metric."""
    phis = np.array([r["phi"] for r in results])[skip:]
    return float(np.mean(np.abs(np.diff(phis, axis=0))))
