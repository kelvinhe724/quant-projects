"""Cost-aware mean-variance across sleeves: the candidate allocator, 1/N is the incumbent.

Maximise mu'w - (lambda / 2) w'Sw - c'|w - w_prev| over long-only sleeve
weights, subject to a book vol cap, gross and net caps, a per-sleeve cap
and a turnover budget per refit, then a Kelly-fraction cap on the leverage.
mu is each sleeve's trailing mean net return, S its covariance (the
risk-model project's if ../../risk-model/ exists, else Ledoit-Wolf), c the
realised cost per unit of traded notional times the sleeve's gross
exposure, all from data before the refit. Drawdown-scaled sizing is not
repeated here: it stays in the engine's overlay on every sleeve.

walk() mirrors allocate.walk (quarterly refits, three-year window, weights
held for the next quarter) so the comparison with 1/N is out of sample.
Its realised stream also charges the reallocation on refit days and, when
the sleeve weights sum past 1, financing on the excess at the 3-month bill
from the data lake plus a spread.

Run: ../../.venv/bin/python3 -m framework.book.optimizer   (candidate vs 1/N on the live book)
"""
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from framework.book import allocate, universe
from framework.engine import sharpe

WINDOW = allocate.WINDOW
MIN_LIVE = allocate.MIN_LIVE
RISK_MODEL = os.path.join(universe.ROOT, "risk-model")
TRADING_DAYS = 252
INFO = {"status": "skipped", "budgeted": False, "budget_bound": False, "vol_bound": False, "kelly_scaled": False,
        "budget_dropped": False, "leverage": 0.0, "cov": ""}


@dataclass(frozen=True)
class Params:
    """Fixed before any run and not tuned; one configuration is one trial.

    risk_aversion 1 is the growth-optimal objective, so the half-Kelly cap,
    the vol cap and the gross cap are what size the book.
    """
    risk_aversion: float = 1.0
    target_vol: float = 0.10
    max_gross: float = 1.5
    max_net: float = 1.5
    max_sleeve: float = 0.75
    turnover: float = 0.5
    kelly: float = 0.5
    financing_spread_bps: float = 50.0


def covariance(R):
    """Annualised covariance of the columns of R and where it came from.

    risk-model/risk_model.py::covariance(daily_returns) -> DataFrame is used
    when that project exists; otherwise sklearn's Ledoit-Wolf.
    """
    if os.path.isfile(os.path.join(RISK_MODEL, "risk_model.py")):
        if RISK_MODEL not in sys.path:
            sys.path.insert(0, RISK_MODEL)
        import risk_model

        return pd.DataFrame(risk_model.covariance(R), index=R.columns, columns=R.columns) * TRADING_DAYS, "risk-model"
    from sklearn.covariance import LedoitWolf

    return pd.DataFrame(LedoitWolf().fit(R.to_numpy()).covariance_, R.columns, R.columns) * TRADING_DAYS, "ledoit-wolf"


def psd(S):
    """Symmetrise and clip negative eigenvalues so a degenerate covariance still solves."""
    S = (S + S.T) / 2
    vals, vecs = np.linalg.eigh(S)
    return (vecs * np.clip(vals, 0.0, None)) @ vecs.T


def optimize(mu, cov, w_prev, cost, p=Params(), turnover=None):
    """One refit. mu, cost, w_prev are Series on the same index; cov annualised. Returns (weights, info).

    turnover: the budget for this refit, None for no budget (the first fit).
    A vol cut the budget would block is solved again without the budget, and
    the Kelly cap scales after the solve: a risk cut is not budgeted, the
    same convention as the engine's buffer; info["budget_dropped"] says so.
    """
    import cvxpy as cp

    names = list(mu.index)
    w0 = w_prev.reindex(names).fillna(0.0)
    info = {**INFO, "budgeted": turnover is not None, "leverage": float(w0.sum())}
    if len(names) == 0 or mu.isna().any() or not np.isfinite(cov.to_numpy()).all():
        return w0, info
    S = psd(cov.loc[names, names].to_numpy())
    m, c = mu.to_numpy(), cost.reindex(names).fillna(0.0).to_numpy()
    w = cp.Variable(len(names))
    d = w - w0.to_numpy()
    obj = m @ w - p.risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(S)) - c @ cp.abs(d)
    base = [w >= 0, w <= p.max_sleeve, cp.sum(w) <= p.max_net, cp.norm1(w) <= p.max_gross,
            cp.quad_form(w, cp.psd_wrap(S)) <= p.target_vol ** 2]
    for cons in ([cp.norm1(d) <= turnover] if turnover is not None else [], []):
        prob = cp.Problem(cp.Maximize(obj), base + cons)
        prob.solve()
        if prob.status in ("optimal", "optimal_inaccurate"):
            break
        info["budget_dropped"] = bool(cons)
    else:
        info["status"] = prob.status
        return w0, info
    x = np.clip(np.asarray(w.value).ravel(), 0.0, None)
    x[x < 1e-8] = 0.0
    info["status"] = "optimal"
    info["budget_bound"] = bool(turnover is not None and not info["budget_dropped"]
                                and np.abs(x - w0.to_numpy()).sum() >= turnover - 1e-6)
    info["vol_bound"] = bool(np.sqrt(x @ S @ x) >= p.target_vol - 1e-4)
    L = x.sum()
    if L > 0:
        u = x / L
        var = float(u @ S @ u)
        kelly_L = p.kelly * float(m @ u) / var if var > 0 else np.inf
        if L > kelly_L:
            x = x * max(kelly_L, 0.0) / L
            info["kelly_scaled"] = True
            if turnover is not None and np.abs(x - w0.to_numpy()).sum() > turnover + 1e-6:
                info["budget_dropped"] = True
    info["leverage"] = float(x.sum())
    return pd.Series(x, index=names), info


def sleeve_costs(res, date, window=WINDOW):
    """Cost of moving one unit of sleeve weight: realised cost per traded notional times gross, data to `date`.

    `res` is one multi-sleeve Results or {sleeve: Results} of single runs.
    """
    runs = res if isinstance(res, dict) else {k: res for k in res.books}
    out = {}
    for k, r in runs.items():
        tr = r.trades[(r.trades.strategy == k) & (r.trades.date <= date)]
        frac = float((tr.commission + tr.slippage).sum() / tr.notional.sum()) if len(tr) and tr.notional.sum() > 0 else 0.0
        gross = r.books[k]["weights"].loc[:date].abs().sum(axis=1)
        gross = gross[gross > 0].tail(window)
        out[k] = frac * (float(gross.mean()) if len(gross) else 0.0)
    return pd.Series(out)


def financing(bars, spread_bps=Params.financing_spread_bps):
    """Daily annual rate charged on sleeve leverage above 1: the lake's DTB3 plus the spread, else the panel's USD rate."""
    idx = bars.calendar
    try:
        sys.path.insert(0, os.path.join(universe.ROOT, "data-lake"))
        import lake

        df = lake.load("fred", str(idx[0].date()), str(idx[-1].date()), universe=["DTB3"])
        s = pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["date"])).dropna() / 100
        source = "data-lake fred/DTB3"
    except Exception as e:
        s = bars.series("USD").dropna() / 100
        source = f"panel USD rate ({type(e).__name__})"
    return (s.reindex(idx.union(s.index)).ffill().reindex(idx).fillna(0.0) + spread_bps / 1e4).rename("financing"), source


def fit(R, date, costs, p=Params(), w_prev=None, window=WINDOW, min_live=MIN_LIVE):
    """Candidate and 1/N weights on the window ending at `date`, zero for sleeves not yet eligible.

    Same window and eligibility as allocate.fit; `costs` from sleeve_costs at
    `date`; w_prev None means no fit yet, so no turnover budget.
    """
    hist, start, eligible = allocate.eligible(R, date, window, min_live)
    w_eq = pd.Series(0.0, index=R.columns)
    budget = None if w_prev is None else p.turnover
    w_prev = pd.Series(0.0, index=R.columns) if w_prev is None else w_prev
    info = dict(INFO)
    if not eligible:
        return w_eq.copy(), w_eq, info
    w_eq[eligible] = 1 / len(eligible)
    X = hist.loc[hist.index >= start[eligible].max(), eligible]
    cov, source = covariance(X)
    w, info = optimize(X.mean() * TRADING_DAYS, cov, w_prev, costs.reindex(eligible).fillna(0.0), p, budget)
    info["cov"] = source
    return w.reindex(R.columns).fillna(0.0), w_eq, info


def walk(R, res, fin, p=Params(), window=WINDOW):
    """Quarterly refits; out-of-sample daily returns of the candidate and 1/N, weight paths, refit log.

    Both streams are daily-rebalanced to their weights (the convention of
    allocate.walk) and pay the move into them on the first day after the
    refit at the costs known then; the candidate also pays `fin` on any
    leverage above 1. The first fit is unbudgeted, every later one budgeted.
    """
    refits = R.resample("QE").last().index
    refits = refits[refits.isin(R.index) | (refits > R.index[0])]
    oos, path, log = [], {}, []
    zero = pd.Series(0.0, index=R.columns)
    w, held, prev, costs, started = {"mvo": zero, "equal": zero}, {"mvo": zero, "equal": zero}, None, None, False
    for d in refits:
        if prev is not None and not (R.index > prev).any():
            break
        if prev is not None and (w["equal"] > 0).any():
            nxt = R.loc[(R.index > prev) & (R.index <= d)]
            lev = max(w["mvo"].sum() - 1.0, 0.0)
            r = pd.DataFrame({"mvo": nxt @ w["mvo"] - lev * fin.reindex(nxt.index).fillna(0.0) / TRADING_DAYS,
                              "equal": nxt @ w["equal"]})
            for k in r:
                r.iloc[0, r.columns.get_loc(k)] -= float(((w[k] - held[k]).abs() * costs).sum())
            oos.append(r)
        held = w
        costs = sleeve_costs(res, d, window)
        new_mvo, new_eq, info = fit(R, d, costs, p, w["mvo"] if started else None, window)
        started = started or info["status"] == "optimal"
        log.append({"date": d, **info, "turnover": float((new_mvo - w["mvo"]).abs().sum())})
        w = {"mvo": new_mvo, "equal": new_eq}
        path[d] = w
        prev = d
    oos = pd.concat(oos) if oos else pd.DataFrame(columns=["mvo", "equal"])
    oos = oos[(oos != 0).any(axis=1)]
    return oos, path, pd.DataFrame(log)


def summary(oos, log, hold=None):
    """Sharpe of both streams, whole walk and from `hold`, plus how often each cap bound."""
    out = {"sharpe_mvo": sharpe(oos["mvo"]), "sharpe_equal": sharpe(oos["equal"]),
           "start": str(oos.index[0].date()), "end": str(oos.index[-1].date())}
    if hold is not None:
        h = oos.loc[hold:]
        out["oos_sharpe_mvo"], out["oos_sharpe_equal"] = sharpe(h["mvo"]), sharpe(h["equal"])
    fits = log[log["status"] == "optimal"]
    out.update({"refits": int(len(fits)), "mean_leverage": float(fits["leverage"].mean()) if len(fits) else 0.0,
                "turnover_per_refit": float(fits["turnover"].mean()) if len(fits) else 0.0,
                **{k: int(fits[k].sum()) if len(fits) else 0
                   for k in ("budget_bound", "vol_bound", "kelly_scaled", "budget_dropped")}})
    return out


if __name__ == "__main__":
    from framework.book.strategies import book_config, sleeves
    from framework.engine import run

    bars = universe.load_bars()
    s = sleeves(allocate.LIVE_BOOK)
    res = run(s, bars, config=book_config(kill=False, allocations={str(x): 1 / len(s) for x in s}))
    fin, source = financing(bars)
    oos, path, log = walk(res.returns_by_strategy, res, fin)
    print(f"financing from {source}; params {asdict(Params())}")
    print(pd.Series(summary(oos, log)).to_string())
    print(log.tail(8).to_string())
