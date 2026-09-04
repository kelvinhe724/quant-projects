"""Probit of recession-within-12-months on the term spread, in sample and expanding window."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from data import HORIZON

# The label at month s uses USREC through s+12, so consecutive labels share 11
# months of information. HAC with 11 lags is the matching standard error.
HAC_LAGS = HORIZON - 1


def fit_probit(frame, features, label="recession_ahead"):
    """Fit a probit of the label on the features, HAC standard errors for the overlapping label."""
    rows = frame[features + [label]].dropna()
    X = sm.add_constant(rows[features], has_constant="add")
    return sm.Probit(rows[label], X).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def predict(result, frame, features):
    """Recession probability for each row of frame under a fitted probit."""
    X = sm.add_constant(frame[features], has_constant="add")
    return pd.Series(result.predict(X), index=frame.index)


def expanding_forecasts(frame, features, start, announce_lag=0, min_obs=120):
    """Out-of-sample probability for each month from start, refit monthly on what was knowable.

    The label for month s needs USREC through s+HORIZON, which NBER publishes
    announce_lag months after the fact. A forecast made at t therefore trains on
    rows s <= t - HORIZON - announce_lag only. Features at t are used as of t.
    """
    label = "recession_ahead"
    usable = frame[features + [label]]
    positions = np.arange(len(frame))
    start_pos = frame.index.searchsorted(pd.Timestamp(start))
    out = pd.Series(np.nan, index=frame.index, name="p_oos")
    for t in positions[start_pos:]:
        if usable[features].iloc[t].isna().any():
            continue
        train = usable.iloc[: t - HORIZON - announce_lag + 1].dropna()
        if len(train) < min_obs or train[label].nunique() < 2:
            continue
        X = sm.add_constant(train[features], has_constant="add")
        fit = sm.Probit(train[label], X).fit(disp=0)
        x_t = np.r_[1.0, usable[features].iloc[t].to_numpy()]
        out.iloc[t] = fit.predict(x_t.reshape(1, -1))[0]
    return out


def auc(y, p):
    """Area under the ROC curve by the rank statistic; ties count half."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def roc_curve(y, p):
    """False positive and true positive rates at every threshold, for plotting."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(-p)
    tp = np.cumsum(y[order] == 1) / max((y == 1).sum(), 1)
    fp = np.cumsum(y[order] == 0) / max((y == 0).sum(), 1)
    return np.r_[0, fp], np.r_[0, tp]


def brier(y, p):
    """Mean squared error of the probability forecast."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def calibration(y, p, bins=10):
    """Observed recession rate against mean forecast, grouped by forecast decile."""
    rows = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    rows["bin"] = pd.qcut(rows["p"].rank(method="first"), bins, labels=False)
    table = rows.groupby("bin").agg(forecast=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return table.reset_index(drop=True)


def scorecard(y, p):
    """AUC, Brier and the base-rate Brier for one forecast series."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"auc": auc(y, p), "brier": brier(y, p), "n": len(y), "positives": int(y.sum())}


def inversion_episodes(spread, min_months=1):
    """Start and end of each stretch of consecutive months with a negative spread."""
    neg = (spread < 0).astype(int)
    edges = neg.diff().fillna(neg)
    starts = spread.index[edges == 1]
    ends = spread.index[edges == -1]
    if len(ends) < len(starts):
        ends = ends.append(pd.DatetimeIndex([spread.index[-1]]))
    out = pd.DataFrame({"start": starts, "end": ends})
    out["months"] = [(neg.loc[a:b] == 1).sum() for a, b in zip(starts, ends)]
    return out[out["months"] >= min_months].reset_index(drop=True)
