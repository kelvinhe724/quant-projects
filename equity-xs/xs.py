"""Cross-sectional equity ML on the point-in-time S&P 500: LightGBM + ridge against plain 12-1 momentum.

Features come from the research feature store (PRICE set plus three daily
FRED series), the label is the next 21-session return minus the
cross-sectional mean that day, and a model scores the eligible members at
each month end. Long the top decile, short the bottom, equal weight.
Every model is an Alpha so the research walk-forward, registry and
untouched lock apply unchanged.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data-lake"))
from framework.book.strategies import CAPITAL, book_config
from research.alpha import Alpha
from research.features import PRICE, Feature, Raw, macro
from research.features.store import FIELDS

MACRO = {"VIXCLS": 1, "T10Y3M": 1, "DGS10": 1}
# pit-universe's first list snapshot with tickers is 2007-11-29; before it the spells are a back-cast
# from that snapshot, so membership is unknown there rather than known, and no row is trained or scored.
MEMBER_FROM = pd.Timestamp("2007-11-29")
STOCK = [f.name for f in PRICE]
HORIZON = 21
DECILE = 0.1
MIN_NAMES = 20
RIDGE_ALPHA = 10.0
LGBM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4,
            deterministic=True, force_col_wise=True)


def load_raw(lake, start, end, snapshot=None):
    """Raw from the lake: adjusted close as close, OHL scaled by the same factor, three FRED series.

    snapshot: a directory. The lake rows are written there the first time and
    read back from there after, so a rerun sees the panel the lock was keyed
    to whatever the lake's collectors have rewritten since.
    """
    paths = {k: os.path.join(snapshot, f"{k}.parquet") for k in ("equities_daily", "fred")} if snapshot else {}
    if paths and os.path.exists(paths["equities_daily"]):
        eq, fred = pd.read_parquet(paths["equities_daily"]), pd.read_parquet(paths["fred"])
    else:
        eq = lake.load("equities_daily", start, end)
        fred = lake.load("fred", start, end, universe=list(MACRO))
        if paths:
            os.makedirs(snapshot, exist_ok=True)
            eq.to_parquet(paths["equities_daily"], index=False)
            fred.to_parquet(paths["fred"], index=False)
    wide = {f: eq.pivot(index="date", columns="ticker", values=f).sort_index() for f in FIELDS + ("adj_close",)}
    factor = wide["adj_close"] / wide["close"]
    frames = {f: wide[f] * factor for f in ("open", "high", "low")}
    frames["close"] = wide["adj_close"]
    frames["volume"] = wide["volume"].astype(float)
    series = {s: pd.Series(g["value"].to_numpy(), index=pd.DatetimeIndex(g["date"]), name=s)
              for s, g in fred.groupby("series")}
    return Raw(frames, series)


def membership(intervals, calendar, instruments, known_from=None):
    """1.0 where the ticker was in the index on that date, by the pit-universe spells; NaN before known_from."""
    m = pd.DataFrame(0.0, index=calendar, columns=instruments)
    for r in intervals.itertuples():
        if r.ticker in m.columns:
            m.loc[r.start:r.end, r.ticker] = 1.0
    if known_from is not None:
        m.loc[m.index < pd.Timestamp(known_from)] = np.nan
    return m


def member_feature(intervals, known_from=None):
    """Membership as a lag-0 feature: an index change is announced before it takes effect."""
    return Feature("member", lambda raw: membership(intervals, raw.calendar, raw.instruments, known_from), 0)


def features(intervals, known_from=None):
    return PRICE + macro(MACRO) + [member_feature(intervals, known_from)]


def ranks(X):
    """Stock features as within-date percentile ranks centred on zero; NaN stays NaN."""
    r = X[STOCK]
    if isinstance(X.index, pd.MultiIndex):
        return r.groupby(level=0).rank(pct=True) - 0.5
    return r.rank(pct=True) - 0.5


def eligible(X):
    return (X["member"] == 1.0) & X["ret_21"].notna()


class XSModel(Alpha):
    """model: 'lgbm', 'ridge', 'ensemble' (mean of the two prediction ranks) or 'mom' (mom_12_1, unfitted).

    dates: the calendar's rebalance dates. fit trains on those rows only, so a
    training slice that ends mid-month does not add its last session as an
    extra, partial cross-section.
    """

    def __init__(self, model="ensemble", decile=DECILE, seed=0, macro=True, dates=None):
        self.model, self.decile, self.seed, self.macro = model, decile, seed, macro
        self.dates = None if dates is None else pd.DatetimeIndex(dates)
        self.params = {"model": model} if macro else {"model": model, "macro": False}
        self.name = f"XS[{model}]" if macro else f"XS[{model}, no macro]"
        self.lgbm = self.ridge = None
        self.importance = {}
        self.train_end = None
        self.train_dates = None

    def fit(self, X, y):
        if self.model == "mom":
            return self
        if self.dates is None:
            raise ValueError("XSModel.fit needs the calendar's rebalance dates: XSModel(..., dates=month_ends(calendar))")
        dates = X.index.get_level_values(0)
        rows = X[eligible(X) & dates.isin(self.dates)]
        y = y.reindex(rows.index)
        y = (y - y.groupby(level=0).transform("mean")).dropna()
        rows = rows.loc[y.index]
        R = ranks(rows).fillna(0.0)
        if self.model in ("ridge", "ensemble"):
            self.ridge = Ridge(alpha=RIDGE_ALPHA).fit(R, y)
            self.importance["ridge"] = pd.Series(self.ridge.coef_, index=STOCK)
        if self.model in ("lgbm", "ensemble"):
            D = self.design(R, rows)
            self.lgbm = lgb.LGBMRegressor(random_state=self.seed, **LGBM).fit(D, y)
            gain = self.lgbm.booster_.feature_importance("gain")
            self.importance["lgbm"] = pd.Series(gain / gain.sum(), index=D.columns)
        self.train_end = dates.max()
        self.train_dates = rows.index.get_level_values(0).unique().sort_values()
        return self

    def design(self, R, rows):
        """LightGBM inputs: the ranked stock features, plus the raw macro series unless switched off."""
        return pd.concat([R, rows[list(MACRO)]], axis=1) if self.macro else R

    def score(self, xs):
        """One number per eligible instrument; NaN elsewhere. Higher is better."""
        ok = eligible(xs)
        out = pd.Series(np.nan, index=xs.index)
        if ok.sum() < MIN_NAMES:
            return out
        rows = xs[ok]
        if self.model == "mom":
            out[rows.index] = rows["mom_12_1"]
            return out
        R = ranks(rows).fillna(0.0)
        parts = []
        if self.ridge is not None:
            parts.append(pd.Series(self.ridge.predict(R), index=rows.index).rank(pct=True))
        if self.lgbm is not None:
            D = self.design(R, rows)
            parts.append(pd.Series(self.lgbm.predict(D), index=rows.index).rank(pct=True))
        out[rows.index] = sum(parts) / len(parts)
        return out

    def signal(self, xs):
        s = self.score(xs).dropna()
        w = pd.Series(0.0, index=xs.index)
        k = int(len(s) * self.decile)
        if k < 1:
            return w
        rank = s.rank(method="first")
        w[rank.index] = ((rank > len(s) - k).astype(float) - (rank <= k).astype(float)) / k
        return w


def scores(alpha, panel, dates):
    """dates x instruments of alpha.score at each date."""
    from research.features.store import cross_section
    rows = {d: alpha.score(cross_section(panel, d)) for d in dates if d in panel.index}
    return pd.DataFrame(rows).T


def rank_ic(score, fwd):
    """Spearman correlation of score with the forward return, one value per date."""
    out = {}
    for d in score.index:
        s = score.loc[d].dropna()
        f = fwd.loc[d].reindex(s.index).dropna()
        if len(f) >= MIN_NAMES:
            out[d] = s.reindex(f.index).corr(f, method="spearman")
    return pd.Series(out, dtype=float)


def decile_spread(score, fwd, decile=DECILE):
    """Top-decile mean forward return minus bottom-decile mean, one value per date."""
    out = {}
    for d in score.index:
        s = score.loc[d].dropna()
        f = fwd.loc[d].reindex(s.index).dropna()
        k = int(len(f) * decile)
        if k >= 1:
            order = s.reindex(f.index).sort_values()
            out[d] = f[order.index[-k:]].mean() - f[order.index[:k]].mean()
    return pd.Series(out, dtype=float)


def ic_summary(ic):
    ic = ic.dropna()
    return {"mean": float(ic.mean()), "std": float(ic.std()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
            "hit": float((ic > 0).mean()), "n": int(len(ic))}


def framework_net(pos, close, dollar_vol, sigma_daily, capital=CAPITAL, cost_scale=1.0):
    """Net daily returns of a position path under the book's cost model at `capital` dollars.

    Same clock as research.alpha.backtest: targets set at t earn t + 1,
    traded weight at t is charged at t + 1. Per-name cost in bps is the
    engine's commission + half spread + impact(sigma, participation), with
    participation the traded notional over trailing dollar volume, plus the
    book's borrow rate on short weight.
    """
    cfg = book_config(cost_scale=cost_scale)
    held = pos.reindex(close.index).ffill().fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    dw = held.diff().abs().fillna(0.0)
    adv = dollar_vol.reindex(close.index).reindex(columns=held.columns)
    part = (dw * capital / adv.where(adv > 0)).fillna(0.0)
    bps = cfg.costs.commission_bps + cfg.costs.slippage_bps(
        part, sigma_daily.reindex(close.index).reindex(columns=held.columns).fillna(0.0))
    cost = (dw * bps / 1e4).sum(axis=1)
    borrow = held.clip(upper=0.0).abs().sum(axis=1) * cfg.borrow_bps / 1e4 / 252
    gross = (held.shift(1) * ret).sum(axis=1)
    return (gross - cost.shift(1).fillna(0.0) - borrow.shift(1).fillna(0.0)).rename("return")


def importance_stability(table):
    """Mean pairwise Spearman correlation of importance vectors across folds (rows)."""
    c = table.T.corr(method="spearman")
    n = len(c)
    if n < 2:
        return float("nan")
    return float((c.to_numpy().sum() - n) / (n * (n - 1)))
