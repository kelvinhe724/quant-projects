"""Calibration binning, Wilson intervals, Brier decomposition and the bias test."""

import numpy as np
import pandas as pd
from scipy import stats

# Deciles, but split the two tail bins finer. Almost all of the interesting
# mass in a prediction-market sample sits below 0.10 and above 0.90.
EDGES = np.array(
    [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
     0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0]
)


def wilson(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def bin_calibration(price, outcome, edges=EDGES):
    """Bin by implied probability and return realised frequency and bias per bin.

    bias = mean price - realised frequency. Positive means the market overpriced
    the contract. Its standard error is the binomial se of the realised frequency,
    since the price side of the difference is a conditioning variable, not an
    estimate.
    """
    price = np.asarray(price, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    idx = np.digitize(price, edges[1:-1], right=False)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        successes = outcome[mask].sum()
        mean_price = price[mask].mean()
        freq = successes / n
        lo, hi = wilson(successes, n)

        # Agresti-Coull. The plain binomial se collapses to zero when a bin
        # resolves all-yes or all-no, which would report an infinite z on what
        # is often a handful of contracts.
        p_adj = (successes + 2) / (n + 4)
        se = np.sqrt(p_adj * (1 - p_adj) / (n + 4))

        bias = mean_price - freq
        z = bias / se
        rows.append({
            "bin_lo": edges[b],
            "bin_hi": edges[b + 1],
            "n": n,
            "mean_price": mean_price,
            "freq": freq,
            "freq_lo": lo,
            "freq_hi": hi,
            "bias": bias,
            "bias_se": se,
            "z": z,
            "p_value": 2 * (1 - stats.norm.cdf(abs(z))),
            "outside_ci": not (lo <= mean_price <= hi),
        })
    return pd.DataFrame(rows)


def brier(price, outcome):
    """Brier score with the Murphy reliability / resolution / uncertainty split."""
    price = np.asarray(price, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    score = np.mean((price - outcome) ** 2)
    base = outcome.mean()
    uncertainty = base * (1 - base)

    table = bin_calibration(price, outcome)
    w = table["n"].values / len(price)
    reliability = np.sum(w * (table["mean_price"] - table["freq"]) ** 2)
    resolution = np.sum(w * (table["freq"] - base) ** 2)
    return {
        "brier": score,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposition_residual": score - (reliability - resolution + uncertainty),
        "baseline_brier": np.mean((base - outcome) ** 2),
        "base_rate": base,
    }


def bias_curve(table):
    """Interpolator from market price to estimated true probability.

    Built only from the bins of a fitted table, so it must be fitted on data
    that the strategy is not then tested on.
    """
    x = table["mean_price"].values
    y = table["freq"].values
    order = np.argsort(x)
    x, y = x[order], y[order]

    def f(p):
        return np.clip(np.interp(p, x, y), 1e-4, 1 - 1e-4)

    return f
