"""Realised-vol forecasters (naive, HAR-RV, GARCH, GJR-GARCH, LightGBM), their scoring, and the straddle timing rule.

Every forecaster produces a dates x instruments frame of annualised vol
forecasts for the next `h` sessions, dated the session the forecast is made
on and using nothing after it. Models refit on an expanding window every
REFIT sessions and hold their parameters in between; labels that would
cross the refit date are purged from the training rows.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vol-risk-premium"))
from research.alpha import Alpha
from research.features import Feature
import vrp

TRADING_DAYS = 252
HORIZONS = (5, 21)
MODELS = ("naive", "har", "garch", "gjr", "lgb")
WARMUP = 756
REFIT = 63
LGB_PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1, "n_jobs": 4}
LGB_FEATURES = ["rv_1", "rv_5", "rv_21", "rv_63", "park_21", "ret_1", "ret_21", "vix", "t10y2y", "dtb3"]
LOG_FEATURES = ("rv_1", "rv_5", "rv_21", "rv_63", "park_21", "vix")


def log_returns(raw):
    return np.log(raw.frames["close"]).diff()


def realised(ret, window):
    """Annualised close-to-close vol of the `window` sessions ending at t, zero-mean estimator."""
    return np.sqrt(TRADING_DAYS * (ret ** 2).rolling(window).mean())


def forward_realised(ret, h):
    """Vol realised over the h sessions strictly after t, dated t."""
    return np.sqrt(TRADING_DAYS * (ret ** 2).rolling(h).mean().shift(-h))


def parkinson(raw, window):
    hl = np.log(raw.frames["high"] / raw.frames["low"]) ** 2
    return np.sqrt(TRADING_DAYS * hl.rolling(window).mean() / (4 * np.log(2)))


def features():
    """Inputs to the forecasters, all dated the session they use the close of."""
    return [
        Feature("rv_1", lambda r: log_returns(r).abs() * np.sqrt(TRADING_DAYS), 0),
        Feature("rv_5", lambda r: realised(log_returns(r), 5), 0),
        Feature("rv_21", lambda r: realised(log_returns(r), 21), 0),
        Feature("rv_63", lambda r: realised(log_returns(r), 63), 0),
        Feature("park_21", lambda r: parkinson(r, 21), 0),
        Feature("ret_1", lambda r: log_returns(r), 0),
        Feature("ret_21", lambda r: log_returns(r).rolling(21).sum(), 0),
        Feature("vix", lambda r: r.series["vix"] / 100.0, 0),
        Feature("t10y2y", lambda r: r.series["t10y2y"], 1),
        Feature("dtb3", lambda r: r.series["dtb3"], 1),
    ]


def refit_points(n, warmup=WARMUP, refit=REFIT):
    return range(warmup, n, refit)


def naive(panel, h):
    return panel[f"rv_{h}"].copy()


def har(panel, target, h, warmup=WARMUP, refit=REFIT):
    """Per instrument: log RV_h(t+1..t+h) on log rv_1, rv_5, rv_21 at t, OLS refit every `refit` sessions."""
    out = pd.DataFrame(np.nan, index=panel.index, columns=target.columns)
    for name in target.columns:
        with np.errstate(divide="ignore"):
            x = np.column_stack([np.ones(len(panel))] + [np.log(panel[k][name].to_numpy(float))
                                                         for k in ("rv_1", "rv_5", "rv_21")])
            y = np.log(target[name].to_numpy(float))
        ok_x = np.isfinite(x).all(axis=1)
        ok = ok_x & np.isfinite(y)
        pred = np.full(len(y), np.nan)
        for t in refit_points(len(y), warmup, refit):
            rows = np.where(ok[:t - h + 1])[0]
            if len(rows) < 50:
                continue
            beta = np.linalg.lstsq(x[rows], y[rows], rcond=None)[0]
            sl = slice(t, t + refit)
            pred[sl] = np.where(ok_x[sl], np.exp(x[sl] @ beta), np.nan)
        out[name] = pred
    return out


def garch(ret, horizons=HORIZONS, asymmetric=False, warmup=WARMUP, refit=REFIT, dist="t"):
    """Per instrument GARCH(1,1) or GJR-GARCH(1,1,1), student-t, refit every `refit` sessions.

    Between refits the parameters are frozen and arch's analytic multi-step
    forecast is run from each session, so the forecast dated t uses returns
    through t only. Returns {h: frame} for every horizon in `horizons`.
    """
    from arch import arch_model

    hmax = max(horizons)
    out = {h: pd.DataFrame(np.nan, index=ret.index, columns=ret.columns) for h in horizons}
    for name in ret.columns:
        r = ret[name].fillna(0.0) * 100
        spec = dict(vol="GARCH", p=1, o=int(asymmetric), q=1, dist=dist)
        for t in refit_points(len(r), warmup, refit):
            fit = arch_model(r.iloc[:t + 1], **spec).fit(disp="off")
            fc = arch_model(r, **spec).fix(fit.params.to_numpy()).forecast(horizon=hmax, start=t, reindex=False)
            var = fc.variance.iloc[:refit]
            for h in horizons:
                out[h].loc[var.index, name] = np.sqrt(TRADING_DAYS * var.iloc[:, :h].mean(axis=1)) / 100
    return out


def lgb_table(panel, cols=LGB_FEATURES):
    """(date, instrument) x features for the boosted model, vol features in logs."""
    X = panel[list(cols)].stack("instrument", future_stack=True)
    with np.errstate(divide="ignore"):
        for k in cols:
            if k in LOG_FEATURES:
                X[k] = np.log(X[k])
    return X


def lgb(panel, target, h, warmup=WARMUP, refit=REFIT, params=LGB_PARAMS):
    """One pooled LightGBM across instruments on log RV_h, refit every `refit` sessions; also returns the last fit."""
    import lightgbm

    X = lgb_table(panel)
    y = np.log(target.stack(future_stack=True)).reindex(X.index)
    pos = pd.Series(np.arange(len(panel.index)), index=panel.index).reindex(X.index.get_level_values(0)).to_numpy()
    ok_x = np.isfinite(X.to_numpy(float)).all(axis=1)
    ok = ok_x & np.isfinite(y.to_numpy(float))
    pred = np.full(len(y), np.nan)
    model = None
    for t in refit_points(len(panel.index), warmup, refit):
        train = ok & (pos <= t - h)
        if train.sum() < 500:
            continue
        model = lightgbm.LGBMRegressor(**params).fit(X[train], y[train])
        test = ok_x & (pos >= t) & (pos < t + refit)
        if test.any():
            pred[test] = np.exp(model.predict(X[test]))
    out = pd.Series(pred, index=X.index).unstack("instrument").reindex(index=panel.index, columns=target.columns)
    return out, model


def forecast_all(panel, ret, horizons=HORIZONS, warmup=WARMUP, refit=REFIT):
    """Every model at every horizon: {h: {model: frame}}, plus the targets {h: frame} and the last LightGBM fits."""
    targets = {h: forward_realised(ret, h) for h in horizons}
    g = garch(ret, horizons, asymmetric=False, warmup=warmup, refit=refit)
    gjr = garch(ret, horizons, asymmetric=True, warmup=warmup, refit=refit)
    out, fits = {}, {}
    for h in horizons:
        boosted, fits[h] = lgb(panel, targets[h], h, warmup, refit)
        out[h] = {"naive": naive(panel, h), "har": har(panel, targets[h], h, warmup, refit),
                  "garch": g[h], "gjr": gjr[h], "lgb": boosted}
    return out, targets, fits


def qlike(f, rv):
    z = (rv / f) ** 2
    return z - np.log(z) - 1


def score(f, rv):
    """Accuracy of one forecast frame against realised, pooled over dates and instruments."""
    both = pd.concat({"f": f.stack(future_stack=True), "rv": rv.stack(future_stack=True)}, axis=1).dropna()
    e = both["f"] - both["rv"]
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(e.abs().mean()), "bias": float(e.mean()),
            "qlike": float(qlike(both["f"], both["rv"]).mean()), "corr": float(both["f"].corr(both["rv"])),
            "n": int(len(both))}


def dm(loss_a, loss_b, h):
    """Diebold-Mariano on a loss differential series; negative t means a beats b. HAC with h - 1 lags."""
    d = (loss_a - loss_b).dropna()
    if len(d) < 30 or d.std() == 0:
        return {"mean": float(d.mean()) if len(d) else np.nan, "t": np.nan, "n": int(len(d))}
    r = vrp.hac_mean(d, lags=max(h - 1, 1))
    return {"mean": r["mean"], "t": r["t"], "n": r["n"]}


def dm_table(forecasts, rv, h, loss="qlike"):
    """Pairwise DM t-stats, pooled across instruments (cross-sectional mean loss per date). Rows beat columns when negative."""
    names = list(forecasts)
    losses = {}
    for m in names:
        f = forecasts[m].reindex_like(rv)
        l = qlike(f, rv) if loss == "qlike" else (f - rv) ** 2
        losses[m] = l.where(f.notna() & rv.notna())
    mask = np.logical_and.reduce([losses[m].notna().to_numpy() for m in names])
    tab = pd.DataFrame(np.nan, index=names, columns=names)
    per_date = {m: losses[m].where(mask).mean(axis=1) for m in names}
    for a in names:
        for b in names:
            if a != b:
                tab.loc[a, b] = dm(per_date[a], per_date[b], h)["t"]
    return tab


def dm_by_instrument(forecasts, rv, h, against="naive", loss="qlike"):
    """DM t of every model against `against`, one row per instrument."""
    rows = {}
    base = forecasts[against].reindex_like(rv)
    for m in forecasts:
        if m == against:
            continue
        f = forecasts[m].reindex_like(rv)
        for name in rv.columns:
            ok = f[name].notna() & base[name].notna() & rv[name].notna()
            if loss == "qlike":
                la, lb = qlike(f[name][ok], rv[name][ok]), qlike(base[name][ok], rv[name][ok])
            else:
                la, lb = (f[name][ok] - rv[name][ok]) ** 2, (base[name][ok] - rv[name][ok]) ** 2
            rows.setdefault(name, {})[m] = dm(la, lb, h)["t"]
    return pd.DataFrame(rows).T


def straddle_cycles(spy, vix, starts, cost_bps=1.0, option_spread=0.01):
    """Daily P&L per $1 notional of a short delta-hedged ATM straddle sold at each date in `starts` and held to the next.

    Each cycle is one vrp.simulate call on the slice from its start to the
    next start, so the tenor is the gap between them (month end to month
    end here) instead of vrp's fixed 21 sessions. The entry cost booked on
    the start day is moved to the next session, so a position set at t
    earns and pays from t + 1, which is the research harness's convention.
    """
    df = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
    pos = pd.Series(np.arange(len(df)), index=df.index)
    starts = [s for s in pd.DatetimeIndex(starts) if s in pos.index]
    pnl = pd.Series(0.0, index=df.index)
    for a, b in zip(starts[:-1], starts[1:]):
        i, j = int(pos[a]), int(pos[b])
        if j - i < 2:
            continue
        cyc = vrp.simulate(df.iloc[i:j + 1], horizon=j - i, cost_bps=cost_bps, option_spread=option_spread)["pnl"]
        cyc.iloc[1] += cyc.iloc[0]
        cyc.iloc[0] = 0.0
        pnl.loc[cyc.index] += cyc
    return pnl


class Timer(Alpha):
    """Sell the straddle only when implied minus the model's forecast exceeds a threshold fixed on the training rows.

    The threshold is the q-quantile of the training gap, so q = 0 is
    nearly always on and q = 0.75 trades one month in four. model "none"
    is the always-on sleeve.
    """

    def __init__(self, model="none", q=0.0):
        self.model, self.q = model, float(q)
        self.thr = -np.inf
        self.name = f"Timer[{model},q={q}]"

    def gap(self, X):
        return X["vix"] - X[f"f_{self.model}"]

    def fit(self, X, y):
        if self.model != "none":
            self.thr = float(self.gap(X).dropna().quantile(self.q))
        return self

    def signal(self, xs):
        if self.model == "none":
            return pd.Series(1.0, index=xs.index)
        g = self.gap(xs)
        return ((g > self.thr) & g.notna()).astype(float)
