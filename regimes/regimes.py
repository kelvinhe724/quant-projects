"""Regime overlay on the premia book: a Gaussian HMM on SPY, a macro nowcast, one gross scale per session.

The HMM sees two observations a day, the log return in percent and the log
of the trailing 21-day realised vol, and is refit every January on every
session before it. The forward filter gives P(state | data through t), the
quantity that exists at the close of t. The nowcast is the mean z-score of
the curve slope, the monthly payroll change and minus the VIX, smoothed. The
scale is 1 - P(highest-vol state), times min(1, exp(nowcast)) when the macro
leg is on, and it multiplies the book's vol target through RegimeOverlay.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

from framework.engine import Strategy

TRADING_DAYS = 252
VOL_DAYS = 21
MIN_TRAIN = 500
Z_WINDOW = 2520
NOWCAST_HALFLIFE = 10
PAYROLL_LAG_DAYS = 40
# RiskManager treats target_vol=0 as "no vol target" and would send the raw
# weights, so the overlay never takes the target below this share of it.
MIN_SCALE = 0.05


def observations(close):
    """Return-in-percent and log realised vol from a close series; the first VOL_DAYS rows are dropped."""
    r = np.log(close.astype(float)).diff() * 100
    vol = r.rolling(VOL_DAYS).std() * np.sqrt(TRADING_DAYS)
    return pd.DataFrame({"ret": r, "logvol": np.log(vol)}).dropna()


def fit(obs, n_states, seed=0):
    """Fit a full-covariance Gaussian HMM and return it with states sorted by ascending return variance."""
    m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, tol=1e-4, random_state=seed)
    m.fit(np.asarray(obs, dtype=float))
    return sort_states(m)


def sort_states(m):
    """Relabel so state 0 is the calmest; EM does not care which label is which, the rest of the code does."""
    order = np.argsort(m.covars_[:, 0, 0])
    m.startprob_ = m.startprob_[order]
    m.transmat_ = m.transmat_[np.ix_(order, order)]
    m.means_ = m.means_[order]
    m.covars_ = m.covars_[order]
    return m


def filtered(m, x):
    """Forward-filtered P(s_t | x_1..x_t), shape (T, K). predict_proba would be smoothed and read the future."""
    x = np.asarray(x, dtype=float)
    logb = np.column_stack([multivariate_normal.logpdf(x, m.means_[k], m.covars_[k]) for k in range(m.n_components)])
    out = np.empty_like(logb)
    a = m.startprob_ * np.exp(logb[0] - logb[0].max())
    out[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (out[t - 1] @ m.transmat_) * np.exp(logb[t] - logb[t].max())
        out[t] = a / a.sum()
    return out


def walk_forward_probs(obs, n_states, first_year, seed=0):
    """Filtered state probabilities with the model refit each January on everything before it.

    Rows before the first fit are NaN. Returns (dates x states frame, {year: model}).
    """
    probs = pd.DataFrame(np.nan, index=obs.index, columns=range(n_states))
    models = {}
    for y in range(first_year, obs.index[-1].year + 1):
        train = obs.loc[: f"{y - 1}-12-31"]
        if len(train) < MIN_TRAIN:
            continue
        m = fit(train, n_states, seed)
        models[y] = m
        thru = obs.loc[: f"{y}-12-31"]
        mask = thru.index.year == y
        probs.loc[thru.index[mask]] = filtered(m, thru)[mask]
    return probs, models


def describe(m):
    """Per-state table: annualised mean and vol of the return leg, persistence, model-implied duration."""
    p_stay = np.diag(m.transmat_)
    return pd.DataFrame({"mean_ann": m.means_[:, 0] / 100 * TRADING_DAYS,
                         "vol_ann": np.sqrt(m.covars_[:, 0, 0]) / 100 * np.sqrt(TRADING_DAYS),
                         "p_stay": p_stay, "duration": 1 / (1 - p_stay)},
                        index=[f"state_{i}" for i in range(m.n_components)])


def run_lengths(states):
    """Mean length of each state's runs in a label path, and each state's share of days."""
    s = pd.Series(np.asarray(states))
    runs = s.groupby((s != s.shift()).cumsum()).agg(["first", "size"])
    return pd.DataFrame({"empirical_duration": runs.groupby("first")["size"].mean(),
                         "share": s.value_counts(normalize=True)}).sort_index()


def nowcast(slope, payrolls, vix, calendar):
    """Macro score on the session calendar: mean z of curve slope, payroll change and minus VIX, EWMA-smoothed.

    z is against a trailing ten-year mean and std so a row only uses the
    past. PAYEMS is dated the first of its month and released about five
    weeks later, so each print is moved PAYROLL_LAG_DAYS forward before it
    can be seen.
    """
    pay = payrolls.diff()
    pay.index = pay.index + pd.Timedelta(days=PAYROLL_LAG_DAYS)
    parts = {"slope": slope, "payrolls": pay, "vix": -vix}
    z = pd.DataFrame({k: v.dropna().sort_index().reindex(calendar, method="ffill") for k, v in parts.items()})
    roll = z.rolling(Z_WINDOW, min_periods=MIN_TRAIN)
    z = (z - roll.mean()) / roll.std()
    return z.mean(axis=1).ewm(halflife=NOWCAST_HALFLIFE).mean()


def scale(p_high, nowcast=None):
    """Gross scale per session: 1 - P(high-vol state), times min(1, exp(nowcast)) if given; 1 where undefined."""
    s = 1 - p_high
    if nowcast is not None:
        s = s * np.minimum(1.0, np.exp(nowcast.reindex(s.index)))
    return s.fillna(1.0).clip(lower=MIN_SCALE, upper=1.0)


class RegimeOverlay(Strategy):
    """Wrap a sleeve and scale the shared RiskConfig's vol target by the day's regime scale.

    The engine builds one RiskManager per sleeve around the same config.risk
    object and reads target_vol on every apply and drift, so setting it
    here, before the inner sleeve answers, is what scales that sleeve's
    gross exposure for the day. Keeps the inner name so allocations match.
    """

    def __init__(self, inner, risk, scale):
        self.inner, self.risk, self.scale = inner, risk, scale
        self.base = risk.target_vol
        self.name = str(inner)

    def on_bar(self, asof, bars):
        self.risk.target_vol = self.base * float(self.scale.get(asof, 1.0))
        return self.inner.on_bar(asof, bars)


def reduced_days(equity, threshold=0.15, recover=None):
    """Replay the engine's drawdown cut on one sleeve's equity path: True on days the sleeve runs at half size.

    The cut arms when the drawdown from the running peak reaches `threshold`
    and clears when it is back inside `recover` (half the threshold by
    default), the rule in framework/engine/risk.py.
    """
    recover = threshold / 2 if recover is None else recover
    dd = equity / equity.cummax() - 1
    out = pd.Series(False, index=equity.index)
    on = False
    for i, d in enumerate(dd.to_numpy()):
        if d <= -threshold:
            on = True
        elif d >= -recover:
            on = False
        out.iloc[i] = on
    return out
