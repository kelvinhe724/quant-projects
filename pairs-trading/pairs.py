"""Engle-Granger pair screening, spread construction and z-score trading rules."""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def hedge_ratio(y, x):
    """Fit the OLS hedge ratio of log price y on log price x."""
    fit = sm.OLS(np.asarray(y), sm.add_constant(np.asarray(x))).fit()
    return float(fit.params[1]), float(fit.params[0])


def spread(log_a, log_b, beta, intercept):
    """Build the cointegrating residual log(A) - beta*log(B) - intercept."""
    return log_a - beta * log_b - intercept


def test_pair(log_a, log_b):
    """Run Engle-Granger on one pair and return (p-value, hedge ratio, intercept)."""
    # statsmodels' coint() applies the Phillips-Ouliaris critical values, which
    # account for the hedge ratio being estimated. Feeding OLS residuals to a
    # plain adfuller() overstates significance.
    pval = coint(np.asarray(log_a), np.asarray(log_b), trend="c")[1]
    beta, intercept = hedge_ratio(log_a, log_b)
    return pval, beta, intercept


def screen(log_px, sectors):
    """Test every within-sector pair for cointegration, sorted by p-value."""
    rows = []
    for sector, names in sectors.groupby(sectors).groups.items():
        for a, b in itertools.combinations(sorted(names), 2):
            pval, beta, intercept = test_pair(log_px[a], log_px[b])
            rows.append((a, b, sector, pval, beta, intercept))
    out = pd.DataFrame(rows, columns=["a", "b", "sector", "pvalue", "beta", "intercept"])
    return out.sort_values("pvalue").reset_index(drop=True)


def select(screened, pvalue_max, max_pairs=None, one_use_per_stock=True):
    """Keep pairs under the p-value threshold, greedily avoiding repeated names."""
    hits = screened[screened["pvalue"] <= pvalue_max]
    if not one_use_per_stock:
        return hits.head(max_pairs).reset_index(drop=True)
    used, keep = set(), []
    for row in hits.itertuples():
        if row.a in used or row.b in used:
            continue
        keep.append(row.Index)
        used.update((row.a, row.b))
        if max_pairs and len(keep) >= max_pairs:
            break
    return screened.loc[keep].reset_index(drop=True)


def bonferroni(n_tests, alpha=0.05):
    """Family-wise corrected p-value threshold for n_tests independent tests."""
    return alpha / n_tests


def zscore(series, lookback):
    """Standardise against a trailing window; the value at t uses only data to t."""
    mean = series.rolling(lookback).mean()
    sd = series.rolling(lookback).std()
    return (series - mean) / sd.replace(0.0, np.nan)


def positions(z, entry_z, exit_z, stop_z, max_hold):
    """Turn a z-score path into a target position: +1 long the spread, -1 short."""
    zv = np.asarray(z, dtype=float)
    pos = np.zeros(len(zv))
    state = 0
    held = 0
    for i, zi in enumerate(zv):
        if np.isnan(zi):
            state, held = 0, 0
            continue
        if state == 0:
            if zi > entry_z:
                state, held = -1, 0
            elif zi < -entry_z:
                state, held = 1, 0
        else:
            held += 1
            crossed = (state == -1 and zi <= exit_z) or (state == 1 and zi >= -exit_z)
            if crossed or abs(zi) > stop_z or held >= max_hold:
                state, held = 0, 0
        pos[i] = state
    return pd.Series(pos, index=z.index)
