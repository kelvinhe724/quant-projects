"""Momentum signals and the portfolio weights they imply.

Two signal families: 12-1 cross-sectional momentum ranked across the eligible
universe, and a 50/200 moving-average crossover applied stock by stock. Both end
in a target weight panel that backtest.py can score, so the two are compared on
identical costs, dates and exposure conventions.

Every function here returns a value at date t built only from prices up to and
including t. Nothing shifts for execution: that is the backtest's job, in one
place, so the lag cannot be applied twice or missed.
"""
import numpy as np
import pandas as pd

LOOKBACK = 252
SKIP = 21
MA_SHORT = 50
MA_LONG = 200
TAIL = 0.10
MIN_NAMES = 8


def momentum_score(px, lookback=LOOKBACK, skip=SKIP):
    """Compute L-to-S momentum: the return from t-lookback to t-skip.

    The skip drops the most recent month, which carries short-term reversal that
    works against the twelve-month trend.
    """
    return px.shift(skip) / px.shift(lookback) - 1


def ma_signal(px, short=MA_SHORT, long=MA_LONG):
    """Return +1 where the short moving average sits above the long one, -1 below."""
    fast = px.rolling(short, min_periods=short).mean()
    slow = px.rolling(long, min_periods=long).mean()
    return np.sign(fast - slow).where(fast.notna() & slow.notna())


def leg_weights(longs, shorts, min_names=MIN_NAMES):
    """Equal-weight each side to 100% gross, giving 200% gross and 0% net.

    A side holding fewer than min_names is dropped for that date rather than
    concentrated into a handful of positions, and the other side is dropped with
    it so the book stays dollar neutral.
    """
    longs, shorts = longs.astype(float), shorts.astype(float)
    n_long, n_short = longs.sum(axis=1), shorts.sum(axis=1)
    live = ((n_long >= min_names) & (n_short >= min_names)).astype(float)
    w = longs.div(n_long.where(n_long > 0), axis=0).fillna(0.0) \
        - shorts.div(n_short.where(n_short > 0), axis=0).fillna(0.0)
    return w.mul(live, axis=0)


def cross_sectional_weights(score, eligible, dates, tail=TAIL, min_names=MIN_NAMES):
    """Rank the eligible cross-section on each rebalance date and hold the two tails.

    Eligibility is applied before ranking, so an ineligible name cannot shift the
    percentile cutoffs of the names that are actually tradable.
    """
    score = score.loc[dates].where(eligible.loc[dates])
    # method="first" makes ties resolve by column order instead of by a rank
    # average that would put a tied name in neither tail
    pct = score.rank(axis=1, pct=True, method="first")
    return leg_weights(pct > 1 - tail, pct <= tail, min_names)


def crossover_weights(signal, eligible, dates=None, min_names=MIN_NAMES):
    """Turn the crossover signal into an equal-weighted long-short book."""
    signal = signal.where(eligible)
    if dates is not None:
        signal = signal.loc[dates]
    return leg_weights(signal > 0, signal < 0, min_names)


def long_only_weights(signal, eligible, dates=None, min_names=MIN_NAMES):
    """Equal-weight the names in an uptrend to 100% gross, holding nothing else."""
    signal = signal.where(eligible)
    if dates is not None:
        signal = signal.loc[dates]
    longs = (signal > 0).astype(float)
    n = longs.sum(axis=1)
    return longs.div(n.where(n >= min_names), axis=0).fillna(0.0)


def information_coefficient(score, forward_return, eligible):
    """Spearman rank correlation between the signal and the next period's return."""
    score = score.where(eligible)
    out = {}
    for date in score.index:
        row = pd.concat([score.loc[date], forward_return.loc[date]], axis=1).dropna()
        if len(row) >= 20:
            out[date] = row.iloc[:, 0].corr(row.iloc[:, 1], method="spearman")
    return pd.Series(out, dtype=float)


def quantile_returns(score, forward_return, eligible, n_buckets=5):
    """Average next-period return by signal bucket, to test for monotonicity."""
    score = score.where(eligible)
    rows = []
    for date in score.index:
        row = pd.concat([score.loc[date], forward_return.loc[date]], axis=1).dropna()
        if len(row) < n_buckets * 5:
            continue
        bucket = pd.qcut(row.iloc[:, 0].rank(method="first"), n_buckets,
                         labels=range(1, n_buckets + 1))
        rows.append(row.iloc[:, 1].groupby(bucket, observed=True).mean())
    return pd.DataFrame(rows).mean()
