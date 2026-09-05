"""Factor risk model for the premia book: covariance estimators, factor count, min-variance test, exposures, VaR, stress.

Prices come through the lake, the Fama-French daily factors from the same
French library fama-french/data.py reads monthly, the GARCH(1,1) from
garch/garch.py. Weights are fractions of equity; cash is the remainder and
carries no risk.

    import risk
    risk.exposures(positions, date)
    risk.var(positions, date)
    risk.stress(positions)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "data-lake"), os.path.join(ROOT, "garch")):
    sys.path.insert(0, p)
import lake
from framework.book import universe
from garch import fit_garch
from research.alpha.base import month_ends
from research.features import Feature, FeatureStore, Raw

ETFS = list(universe.ETFS)
CLASSES = universe.CLASSES
FF_CACHE = os.path.join(ROOT, "source-material", "risk-model", "ff_daily.csv")
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
ESTIMATORS = ("sample", "lw", "pca", "ff")
WINDOW = 504
GARCH_WINDOW = 1000
REFIT = 21
TW99 = 2.0234
STRESS = {"2008-09": ("2008-09-01", "2009-03-09"), "2020-03": ("2020-02-19", "2020-03-23"),
          "2022": ("2022-01-03", "2022-12-30")}
_cache = {}


# data

def ensure_etfs(tickers=ETFS, start="2003-06-01"):
    """Write any ETF the lake lacks, or holds stale, through lake.write; yfinance unadjusted, the collector's schema."""
    have = set(lake.parts("equities_daily"))
    last = lambda t: pd.read_parquet(lake.part_path("equities_daily", t), columns=["date"])["date"].max()
    spy = last("SPY")
    groups = {start: [t for t in tickers if t not in have]}
    stale = [t for t in tickers if t in have and last(t) < spy]
    if stale:
        groups[(min(last(t) for t in stale) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")] = stale
    rows = 0
    for since, names in groups.items():
        if not names:
            continue
        import yfinance as yf
        raw = yf.download(names, start=since, auto_adjust=False, progress=False, group_by="column", threads=True)
        for t in names:
            df = pd.DataFrame({"date": raw.index, "ticker": t, "open": raw["Open"][t].to_numpy(),
                               "high": raw["High"][t].to_numpy(), "low": raw["Low"][t].to_numpy(),
                               "close": raw["Close"][t].to_numpy(), "adj_close": raw["Adj Close"][t].to_numpy(),
                               "volume": raw["Volume"][t].to_numpy()}).dropna(subset=["close"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df["collected_at"] = pd.Timestamp.now("UTC").floor("s").tz_localize(None)
            rows += lake.write("equities_daily", t, df)
    return rows


def load_panel(tickers=None, start="2003-06-01", end=None, universe_=None):
    """Wide adjusted close, raw close and volume through the lake."""
    eq = lake.load("equities_daily", start, end, universe=universe_ or tickers)
    piv = lambda c: eq.pivot(index="date", columns="ticker", values=c).sort_index()
    frames = {"close": piv("adj_close"), "raw_close": piv("close"), "volume": piv("volume")}
    if tickers:
        frames = {k: v.reindex(columns=tickers) for k, v in frames.items()}
    return frames


def returns(frames, audit=True):
    """Daily returns on the adjusted close, lag 0, through the feature store's peek audit."""
    raw = Raw({"close": frames["close"], "volume": frames["volume"]})
    store = FeatureStore([Feature("ret", lambda r: r.frames["close"] / r.frames["close"].shift(1) - 1, 0)])
    return store.build(raw, audit=audit)["ret"]


def etf_returns():
    if "etf" not in _cache:
        ensure_etfs()
        _cache["etf"] = returns(load_panel(ETFS))
    return _cache["etf"]


def load_factors(cache=FF_CACHE, max_age_days=7):
    """FF5 plus momentum and RF, daily, decimals; French's library as in fama-french/data.py, cached a week."""
    fresh = os.path.exists(cache) and (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(cache), unit="s")).days < max_age_days
    if not fresh:
        from pandas_datareader import data as pdr
        ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start="2003-01-01")[0]
        mom = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start="2003-01-01")[0]
        mom.columns = [c.strip() for c in mom.columns]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        (ff.join(mom, how="inner") / 100).to_csv(cache)
    f = pd.read_csv(cache, index_col=0, parse_dates=True)
    return f[FACTORS + ["RF"]]


# estimators

def mp_edge(n, t, sigma2=1.0, tw=TW99):
    """Largest eigenvalue pure noise puts in an n-name correlation matrix over t days.

    Marchenko-Pastur edge sigma2 (1 + sqrt(n/t))^2 plus the Tracy-Widom
    99% cushion for the largest eigenvalue (Johnstone 2001), so a noise
    eigenvalue clears it one time in a hundred.
    """
    a, b = np.sqrt(t - 1), np.sqrt(n)
    mu = (a + b) ** 2
    sd = (a + b) * (1 / a + 1 / b) ** (1 / 3)
    return sigma2 * (mu + tw * sd) / t


def n_factors(eigvals, t):
    """Eigenvalues of a correlation matrix above the noise edge at unit noise variance.

    Conservative on purpose: strong factors absorb variance and push the
    real noise level below one, so weak factors near the edge are missed
    rather than noise counted. Re-estimating the noise variance from the
    residual (Laloux et al. 1999) over-counts here because the loadings
    are heterogeneous, which widens the bulk.
    """
    lam = np.asarray(eigvals)
    return int((lam > mp_edge(len(lam), t)).sum())


def cov_pca(R, k=None):
    """Statistical factor covariance: top-k eigenvectors of the correlation matrix plus a diagonal residual."""
    R = np.asarray(R, float)
    sd = R.std(axis=0, ddof=1)
    C = np.corrcoef(R, rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    if k is None:
        k = n_factors(lam, len(R))
    Ck = (V[:, :k] * lam[:k]) @ V[:, :k].T
    C_hat = Ck + np.diag(np.clip(1 - np.diag(Ck), 1e-8, None))
    return C_hat * np.outer(sd, sd), k


def cov_ff(R, F):
    """Observed-factor covariance B Cov(F) B' + D from OLS of each column of R on F over the same rows."""
    R, F = np.asarray(R, float), np.asarray(F, float)
    X = np.column_stack([np.ones(len(F)), F])
    coef = np.linalg.lstsq(X, R, rcond=None)[0]
    B = coef[1:].T
    D = (R - X @ coef).var(axis=0, ddof=X.shape[1])
    Fc = np.cov(F, rowvar=False, ddof=1)
    return B @ Fc @ B.T + np.diag(D), B, D


def estimate(name, R, F=None):
    """Covariance of the columns of R by one estimator; F is the factor frame on R's rows, needed for "ff"."""
    R = np.asarray(R, float)
    if name == "sample":
        return np.cov(R, rowvar=False, ddof=1), {}
    if name == "lw":
        lw = LedoitWolf().fit(R)
        return lw.covariance_, {"shrinkage": float(lw.shrinkage_)}
    if name == "pca":
        S, k = cov_pca(R)
        return S, {"k": k}
    if name == "ff":
        S, B, D = cov_ff(R, F)
        return S, {"B": B, "D": D}
    raise KeyError(name)


def gmv(S):
    """Unconstrained global minimum-variance weights."""
    ones = np.ones(len(S))
    try:
        x = np.linalg.solve(S, ones)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(S) @ ones
    return x / x.sum()


def factor_rows(win, factors):
    """Rows of the window with a factor print; None when fewer than 90% have one."""
    if factors is None:
        return None
    both = win.index.intersection(factors.dropna().index)
    return both if len(both) >= 0.9 * len(win) else None


def minvar_race(ret, factors=None, window=WINDOW, pick=None, estimators=ESTIMATORS):
    """Monthly GMV per estimator on the trailing window, held to the next month end, no costs.

    pick(names, date) narrows the complete names at each rebalance. Returns
    the daily portfolio returns, gross leverage per rebalance and the PCA
    factor count.
    """
    ends = month_ends(ret.index)
    out = {e: [] for e in estimators}
    lev = {e: {} for e in estimators}
    ks = {}
    for i, t in enumerate(ends[:-1]):
        pos = ret.index.get_loc(t)
        if pos + 1 < window:
            continue
        win = ret.iloc[pos - window + 1: pos + 1]
        names = list(win.columns[win.notna().all()])
        if pick is not None:
            names = pick(names, t)
        if len(names) < 2:
            continue
        nxt = ret.iloc[pos + 1: ret.index.get_loc(ends[i + 1]) + 1][names].fillna(0.0)
        rows = factor_rows(win, factors)
        for e in estimators:
            if e == "ff":
                if rows is None:
                    continue
                R = win.loc[rows, names].to_numpy() - factors.loc[rows, "RF"].to_numpy()[:, None]
                S, meta = estimate(e, R, factors.loc[rows, FACTORS])
            else:
                S, meta = estimate(e, win[names].to_numpy())
            w = gmv(S)
            out[e].append(nxt @ w)
            lev[e][t] = float(np.abs(w).sum())
            if "k" in meta:
                ks[t] = meta["k"]
    rets = pd.DataFrame({e: pd.concat(v) for e, v in out.items() if v})
    return {"returns": rets, "leverage": pd.DataFrame(lev), "k": pd.Series(ks, dtype=float)}


def ann_vol(r):
    r = r.dropna()
    return float(r.std() * np.sqrt(252)) if len(r) > 1 else np.nan


# the book

def _window(positions, date, ret, window):
    w = pd.Series(positions, dtype=float)
    hist = ret.loc[:pd.Timestamp(date)].iloc[-window:]
    names = [n for n in w.index if n in hist.columns and hist[n].notna().all()]
    missing = sorted(set(w.index) - set(names))
    if missing:
        raise ValueError(f"no complete {window}-day history through {pd.Timestamp(date).date()} for {missing}")
    return hist[names], w[names]


def _cov(estimator, R, factors):
    """Covariance of a window frame by name; "ff" needs the factor rows and excess returns."""
    if estimator != "ff":
        return estimate(estimator, R.to_numpy())[0]
    rows = factor_rows(R, factors)
    if rows is None:
        raise ValueError(f"fewer than 90% of the window's rows have a factor print through {R.index[-1].date()}")
    Rx = R.loc[rows].to_numpy() - factors.loc[rows, "RF"].to_numpy()[:, None]
    return estimate("ff", Rx, factors.loc[rows, FACTORS])[0]


def exposures(positions, date, ret=None, factors=None, window=WINDOW, estimator="lw", classes=None):
    """Risk of a weight dict as of `date`: by asset, asset class, Fama-French factor and statistical factor.

    Contributions are annualised vol; pct is the share of portfolio variance.
    The Fama-French block uses the rows of the window with a factor print
    and reports the last one.
    """
    ret = etf_returns() if ret is None else ret
    factors = load_factors() if factors is None else factors
    classes = CLASSES if classes is None else classes
    R, w = _window(positions, date, ret, window)
    names, wv = list(R.columns), w.to_numpy()
    S = _cov(estimator, R, factors)
    var_p = float(wv @ S @ wv)
    sig = np.sqrt(var_p)
    mcr = S @ wv / sig
    contrib = wv * mcr
    root = np.sqrt(252)
    assets = pd.DataFrame({"weight": wv, "vol": np.sqrt(np.diag(S)) * root, "mcr": mcr * root,
                           "contribution": contrib * root, "pct": contrib / sig}, index=names)
    by_class = assets.groupby(pd.Series(classes).reindex(names).fillna("other").to_numpy())[
        ["weight", "contribution", "pct"]].sum()
    out = {"date": str(pd.Timestamp(date).date()), "vol": sig * root, "estimator": estimator,
           "assets": assets, "by_class": by_class}
    rows = factor_rows(R, factors)
    if rows is not None:
        Rx = R.loc[rows].to_numpy() - factors.loc[rows, "RF"].to_numpy()[:, None]
        Fm = factors.loc[rows, FACTORS].to_numpy()
        _, B, D = cov_ff(Rx, Fm)
        Fc = np.cov(Fm, rowvar=False, ddof=1)
        x = B.T @ wv
        fpart, rpart = x * (Fc @ x), float(wv @ (D * wv))
        tot = fpart.sum() + rpart
        ff = pd.DataFrame({"exposure": x, "factor_vol": np.sqrt(np.diag(Fc)) * root,
                           "contribution": fpart / np.sqrt(tot) * root, "pct": fpart / tot}, index=FACTORS)
        ff.loc["residual"] = [np.nan, np.nan, rpart / np.sqrt(tot) * root, rpart / tot]
        out.update(ff=ff, ff_through=str(rows[-1].date()), ff_vol=np.sqrt(tot) * root,
                   betas=pd.DataFrame(B, index=names, columns=FACTORS))
    sd = R.std(ddof=1).to_numpy()
    lam, V = np.linalg.eigh(np.corrcoef(R.to_numpy(), rowvar=False))
    lam, V = lam[::-1], V[:, ::-1]
    k = n_factors(lam, len(R))
    y = V[:, :k].T @ (sd * wv)
    fvar = lam[:k] * y ** 2
    resid_var = float(np.sum((1 - np.sum(V[:, :k] ** 2 * lam[:k], axis=1)) * (sd * wv) ** 2))
    tot = fvar.sum() + resid_var
    stat = pd.DataFrame({"eigenvalue": lam[:k], "pct_universe": lam[:k] / len(names), "pct": fvar / tot},
                        index=[f"PC{i + 1}" for i in range(k)])
    stat.loc["residual"] = [np.nan, np.nan, resid_var / tot]
    out.update(stat=stat, k=k, eigenvalues=lam, mp_edge=mp_edge(len(names), len(R)))
    return out


def fhs_draws(R, garch_window=GARCH_WINDOW, window=WINDOW, dist="t"):
    """One GARCH(1,1) per column of R; tomorrow's vol times the last `window` standardised residuals, jointly by date."""
    sims = {}
    for n in R.columns:
        r = R[n].dropna().iloc[-garch_window:] * 100
        res = fit_garch(r, dist=dist)["res"]
        z = (res.resid / res.conditional_volatility).iloc[-window:]
        s1 = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
        sims[n] = s1 * z / 100
    return pd.DataFrame(sims).dropna()


def tail(p, level):
    """Empirical VaR and ES of a return series, as positive losses."""
    v = -np.quantile(p, 1 - level)
    return {"var": float(v), "es": float(-p[p <= -v].mean())}


def var(positions, date, ret=None, method="all", level=0.99, window=WINDOW, estimator="lw",
        garch_window=GARCH_WINDOW, factors=None):
    """1-day VaR and ES as fractions of equity: parametric on the estimator's covariance, historical on the trailing window, filtered historical with a GARCH(1,1) per asset."""
    ret = etf_returns() if ret is None else ret
    full = ret.loc[:pd.Timestamp(date)]
    R, w = _window(positions, date, ret, window)
    wv = w.to_numpy()
    out = {}
    if method in ("parametric", "all"):
        if estimator == "ff" and factors is None:
            factors = load_factors()
        S = _cov(estimator, R, factors)
        sig = float(np.sqrt(wv @ S @ wv))
        z = stats.norm.ppf(level)
        out["parametric"] = {"var": z * sig, "es": sig * stats.norm.pdf(z) / (1 - level), "sigma": sig}
    if method in ("historical", "all"):
        out["historical"] = tail((R @ wv), level)
    if method in ("fhs", "all"):
        draws = fhs_draws(full[list(R.columns)], garch_window, window)
        out["fhs"] = tail(draws @ wv, level)
    return out if method == "all" else out[method]


def stress(positions, ret=None, windows=STRESS):
    """Today's weights held fixed through each window: compounded return, worst day, max drawdown, by-asset contribution."""
    ret = etf_returns() if ret is None else ret
    w = pd.Series(positions, dtype=float)
    out = {}
    for name, (a, b) in windows.items():
        seg = ret.loc[a:b, w.index]
        missing = sorted(seg.columns[seg.isna().any()])
        seg = seg.fillna(0.0)
        p = seg @ w
        eq = (1 + p).cumprod()
        out[name] = {"start": a, "end": b, "days": int(len(p)), "return": float(eq.iloc[-1] - 1),
                     "worst_day": float(p.min()), "worst_date": str(p.idxmin().date()),
                     "max_drawdown": float((eq / eq.cummax() - 1).min()),
                     "by_asset": (seg * w).sum().to_dict(), "missing": missing}
    return out


def kupiec(hits, level=0.99):
    """Unconditional coverage test of a hit series against 1 - level."""
    hits = np.asarray(hits, bool)
    n, x, p = len(hits), int(hits.sum()), 1 - level
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * (x * np.log(p) + (n - x) * np.log(1 - p) - x * np.log(x / n) - (n - x) * np.log(1 - x / n))
    return {"n": n, "hits": x, "rate": x / n, "p_value": float(stats.chi2.sf(lr, 1))}


def var_backtest(ret, positions, level=0.99, window=WINDOW, garch_window=GARCH_WINDOW, refit=REFIT,
                 estimator="lw", dist="t", factors=None):
    """Daily 1-day VaR of fixed weights against the next day's return, three methods.

    Parametric re-estimates the covariance every day on the trailing
    window; historical takes the trailing quantile; filtered historical
    fits one GARCH(1,1) to the portfolio series every `refit` days and
    runs the variance recursion by hand in between. Returns a frame of
    realised return and the three VaRs (positive), dated the day the
    return lands.
    """
    w = pd.Series(positions, dtype=float)
    R = ret[w.index].dropna()
    p = R @ w
    z = stats.norm.ppf(level)
    rows = []
    fit = None
    for i in range(max(window, garch_window), len(p)):
        S = _cov(estimator, R.iloc[i - window: i], factors)
        para = z * float(np.sqrt(w.to_numpy() @ S @ w.to_numpy()))
        histo = -float(np.quantile(p.iloc[i - window: i], 1 - level))
        if fit is None or (i - fit["at"]) >= refit:
            g = fit_garch(p.iloc[i - garch_window: i] * 100, dist=dist)
            res = g["res"]
            zq = float(np.quantile((res.resid / res.conditional_volatility).iloc[-window:], 1 - level))
            fit = {"at": i, "omega": g["omega"], "alpha": g["alpha"], "beta": g["beta"], "mu": float(res.params["mu"]),
                   "sig2": float(res.conditional_volatility.iloc[-1] ** 2),
                   "eps": float(res.resid.iloc[-1]), "zq": zq}
        fit["sig2"] = fit["omega"] + fit["alpha"] * fit["eps"] ** 2 + fit["beta"] * fit["sig2"]
        fhs = -np.sqrt(fit["sig2"]) * fit["zq"] / 100
        fit["eps"] = float(p.iloc[i] * 100 - fit["mu"])
        rows.append((p.index[i], p.iloc[i], para, histo, fhs))
    return pd.DataFrame(rows, columns=["date", "return", "parametric", "historical", "fhs"]).set_index("date")


def coverage(bt, level=0.99):
    """Kupiec table for a var_backtest frame."""
    return {m: kupiec(bt["return"] < -bt[m], level) for m in ("parametric", "historical", "fhs")}
