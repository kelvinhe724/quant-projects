"""Capital shares across sleeves: ERC on sleeve return streams, 1/N as the null.

Each sleeve is backtested alone, its daily returns form one column, and
skfolio's RiskBudgeting (equal risk contributions, Ledoit-Wolf covariance) is
fit on a trailing three-year window and refit at each quarter end. The fitted
weights are applied to the following quarter, so the ERC-vs-1/N comparison is
out of sample. ERC becomes the live allocator only if it beats 1/N on that
out-of-sample record; otherwise 1/N is live. A sleeve enters the allocator
once it has a year of live returns inside the window; before that its weight
is zero, its returns are never padded with zeros.

Run: ../../.venv/bin/python3 -m framework.book.allocate
"""
import json
import os

import numpy as np
import pandas as pd

from framework.book import universe
from framework.book.strategies import book_config, sleeves
from framework.engine import drawdown, run, sharpe

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
WINDOW = 3 * 252
MIN_LIVE = 252
# The book the daemon runs, one of strategies.BOOKS. A candidate replaces it
# only by beating it on the untouched window in validate.py (README, v3).
# 2026-09-04: beta+alpha 0.79 vs v1 0.40 on 2021-12-31 to 2026-08-31.
LIVE_BOOK = "beta+alpha"


def run_sleeves(bars, book=LIVE_BOOK, **config_kwargs):
    """Backtest every sleeve of `book` alone. Returns {name: Results}."""
    return {str(s): run(s, bars, config=book_config(**config_kwargs)) for s in sleeves(book)}


def returns_frame(results):
    return pd.DataFrame({k: r.returns for k, r in results.items()})


def live_from(R):
    """First date each sleeve has a nonzero return, NaT if never."""
    return pd.Series({c: R[c].ne(0).idxmax() if R[c].ne(0).any() else pd.NaT for c in R})


def erc(X):
    """Equal-risk-contribution weights on the columns of X, Ledoit-Wolf covariance."""
    from skfolio.moments import LedoitWolf
    from skfolio.optimization import RiskBudgeting
    from skfolio.prior import EmpiricalPrior

    m = RiskBudgeting(prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf())).fit(X)
    return pd.Series(m.weights_, index=X.columns)


def fit(R, date, window=WINDOW, min_live=MIN_LIVE):
    """ERC and 1/N weights fit on the window ending at `date`, zero for sleeves not yet eligible."""
    hist = R.loc[:date].tail(window)
    start = live_from(hist)
    eligible = [c for c in R if pd.notna(start[c]) and (hist.index > start[c]).sum() >= min_live]
    w_erc, w_eq = pd.Series(0.0, index=R.columns), pd.Series(0.0, index=R.columns)
    if len(eligible) >= 2:
        X = hist.loc[hist.index >= start[eligible].max(), eligible]
        w_erc[eligible] = erc(X)
        w_eq[eligible] = 1 / len(eligible)
    elif eligible:
        w_erc[eligible] = w_eq[eligible] = 1.0
    return w_erc, w_eq


def walk(R, window=WINDOW):
    """Quarterly refits; returns the out-of-sample daily returns of ERC and 1/N and the weight path."""
    refits = R.resample("QE").last().index
    refits = refits[refits.isin(R.index) | (refits > R.index[0])]
    oos, path = [], {}
    prev = None
    for d in refits:
        if prev is not None and not (R.index > prev).any():
            break
        w_erc, w_eq = fit(R, d, window)
        if prev is not None and (w_erc > 0).any():
            nxt = R.loc[(R.index > prev) & (R.index <= d)]
            oos.append(pd.DataFrame({"erc": nxt @ path[prev]["erc"], "equal": nxt @ path[prev]["equal"]}))
        path[d] = {"erc": w_erc, "equal": w_eq}
        prev = d
    oos = pd.concat(oos) if oos else pd.DataFrame(columns=["erc", "equal"])
    oos = oos[(oos != 0).any(axis=1)]
    return oos, path


def stats(r):
    r = r.dropna()
    years = len(r) / 252
    return {"sharpe": sharpe(r), "annual_return": float((1 + r).prod() ** (1 / years) - 1),
            "annual_vol": float(r.std() * np.sqrt(252)), "max_drawdown": drawdown(r)[1]["max_drawdown"],
            "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def compare(oos):
    """Side-by-side out-of-sample statistics and Sharpe by year."""
    table = pd.DataFrame({k: stats(oos[k]) for k in ("erc", "equal")})
    table = table.map(lambda v: round(v, 4) if isinstance(v, float) else v)
    years = oos.groupby(oos.index.year).agg(lambda s: sharpe(s))
    return table, years


def allocate(bars=None, results=None, write=True):
    """Pick the live allocator and its weights; write reports/allocations.json."""
    bars = universe.load_bars() if bars is None else bars
    results = run_sleeves(bars, kill=False) if results is None else results
    R = returns_frame(results)
    oos, path = walk(R)
    table, years = compare(oos)
    winner = "erc" if table.loc["sharpe", "erc"] > table.loc["sharpe", "equal"] else "equal"
    w_erc, w_eq = fit(R, R.index[-1])
    live = w_erc if winner == "erc" else w_eq
    out = {"book": LIVE_BOOK, "allocator": winner, "weights": {k: round(float(v), 4) for k, v in live.items()},
           "erc_weights": {k: round(float(v), 4) for k, v in w_erc.items()},
           "fit_date": str(R.index[-1].date()), "window_days": WINDOW,
           "oos": {k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in table[k].items()}
                   for k in table},
           "panel_hash": universe.panel_hash(bars)}
    if write:
        os.makedirs(REPORTS, exist_ok=True)
        with open(os.path.join(REPORTS, "allocations.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        pd.DataFrame({d: p["erc"] for d, p in path.items()}).T.to_csv(os.path.join(REPORTS, "erc_weights.csv"))
    return out, table, years, oos, R


def load_allocations(path=os.path.join(REPORTS, "allocations.json")):
    with open(path) as fh:
        return json.load(fh)


if __name__ == "__main__":
    out, table, years, oos, R = allocate()
    print("sleeves live from: " + ", ".join(f"{k} {v.date()}" for k, v in live_from(R).items()))
    print("\nout of sample, quarterly refits, 3y window\n")
    print(table.to_string())
    print("\nSharpe by year\n")
    print(years.round(2).to_string())
    print(f"\nlive allocator: {out['allocator']}  weights {out['weights']}  (ERC would be {out['erc_weights']})")
