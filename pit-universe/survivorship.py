"""Run the momentum and pairs rules on two universes and difference the results.

The rules are imported from the sibling projects, not rewritten. The only
thing that changes between the two runs is which names are allowed on which
date: today's members applied backwards, or the point-in-time panel.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def import_project(project, module):
    """Load `module.py` from a sibling project without polluting this project's namespace."""
    folder = os.path.join(ROOT, project)
    spec = importlib.util.spec_from_file_location(f"{project}.{module}",
                                                  os.path.join(folder, module + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(folder)
    return mod


momentum = import_project("momentum", "momentum")
mom_backtest = import_project("momentum", "backtest")
mom_data = import_project("momentum", "data")
pairs = import_project("pairs-trading", "pairs")
pairs_backtest = import_project("pairs-trading", "backtest")
pairs_data = import_project("pairs-trading", "data")

MOM_WINDOWS = {
    "in-sample": (mom_data.IS_START, mom_data.IS_END),
    "final test": (mom_data.TEST_START, mom_data.TEST_END),
    "full": (mom_data.IS_START, mom_data.TEST_END),
}
PAIRS_FORMATION = (pairs_data.FORMATION_START, pairs_data.FORMATION_END)
PAIRS_OOS = (pairs_data.OOS_START, pairs_data.OOS_END)

# The configuration the pairs project chose on its formation grid (reports/parameter_grid.csv).
PAIRS_RULES = dict(lookback=90, entry_z=1.5, exit_z=0.0, stop_z=4.0, max_hold=60,
                   cost_bps=pairs_backtest.DEFAULT_COST_BPS)
PAIRS_PVALUE = 0.05
PAIRS_MAX = 40


def fill(px):
    """Bridge gaps inside a live series without inventing prices before or after it."""
    return px.where(px > 0).ffill().where(px.bfill().notna())


def momentum_book(px, vol, member, tail=momentum.TAIL, cost_bps=mom_backtest.COST_BPS,
                  borrow_bps=mom_backtest.BORROW_BPS):
    """12-1 decile long-short restricted to `member`, with the momentum project's rules.

    Returns the backtest frame plus the eligibility mask actually used.
    """
    px = fill(px)
    returns = px.pct_change()
    eligible = mom_data.eligibility(px, vol) & member.reindex(index=px.index, columns=px.columns, fill_value=False)
    dates = mom_data.rebalance_dates(px.index, mom_data.IS_START, mom_data.TEST_END)
    w = momentum.cross_sectional_weights(momentum.momentum_score(px), eligible, dates, tail=tail)
    book = mom_backtest.run(w, returns, cost_bps, borrow_bps)
    book.attrs["weights"] = w
    book.attrs["eligible"] = eligible
    return book


def equal_weight(px, member):
    """Daily return of an equal-weighted basket of the names in `member` the day before."""
    returns = fill(px).pct_change()
    tradable = member.reindex(index=returns.index, columns=returns.columns, fill_value=False).shift(1, fill_value=False)
    return returns.where(tradable).mean(axis=1)


def annualised(r):
    r = r.dropna()
    return (1 + r).prod() ** (mom_backtest.TRADING_DAYS / len(r)) - 1


def momentum_row(book, window, today):
    """Metrics for one book over one window, plus how much of it sat outside today's list."""
    m = mom_backtest.summary(book.loc[window[0]:window[1]])
    w = book.attrs["weights"].loc[window[0]:window[1]]
    outside = [c for c in w.columns if c not in today]
    live = w.abs().sum(axis=1) > 0
    elig = book.attrs["eligible"].loc[window[0]:window[1]].sum(axis=1)
    return {
        "gross annual return": m["gross_annual_return"],
        "net annual return": m["annual_return"],
        "gross Sharpe": m["gross_sharpe"],
        "net Sharpe": m["sharpe"],
        "annual vol": m["annual_vol"],
        "max drawdown (net)": m["max_drawdown"],
        "long leg": m["long_annual_return"],
        "short leg": m["short_annual_return"],
        "annual turnover": m["annual_turnover"],
        "eligible names (median)": float(elig.median()),
        "share of long book outside today's list":
            (w[outside].clip(lower=0).sum(axis=1)[live]).mean() if outside else 0.0,
        "share of short book outside today's list":
            (w[outside].clip(upper=0).abs().sum(axis=1)[live]).mean() if outside else 0.0,
    }


def compare(rows_a, rows_b, label_a="today's members", label_b="point-in-time"):
    """Side-by-side table with the difference column that is the bias estimate."""
    out = pd.DataFrame({label_a: rows_a, label_b: rows_b})
    out["difference (a - b)"] = out[label_a] - out[label_b]
    return out


def momentum_comparison(px, vol, panel, today, windows=MOM_WINDOWS):
    """Run 12-1 momentum on both universes; return {window: table} and the two books."""
    today_mask = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    today_mask[[c for c in panel.columns if c in today]] = True
    books = {"today": momentum_book(px, vol, today_mask), "pit": momentum_book(px, vol, panel)}
    tables = {name: compare(momentum_row(books["today"], win, today),
                            momentum_row(books["pit"], win, today))
              for name, win in windows.items()}
    return tables, books


def pairs_universe(px, member_on, sectors, pit):
    """Prices and sectors the pairs screen is allowed to see.

    pit=False reproduces the pairs project: names must be 99% complete over the
    whole sample, formation and out-of-sample alike. pit=True requires
    completeness over formation only, so a name delisted later still gets
    screened and traded until its last print.
    """
    span = px.loc[PAIRS_FORMATION[0]:PAIRS_OOS[1], [c for c in px.columns if member_on.get(c, False)]]
    keep = pairs_data.clean(span.loc[:PAIRS_FORMATION[1]] if pit else span).columns
    keep = [c for c in keep if c in sectors.dropna().index]
    out = fill(span[keep]) if pit else pairs_data.clean(span[keep])
    return out, sectors.reindex(keep)


def pairs_result(px, sectors, rules=PAIRS_RULES, pvalue=PAIRS_PVALUE, max_pairs=PAIRS_MAX):
    """Screen on formation and trade out of sample with the pairs project's code."""
    log_px = np.log(px)
    screened = pairs.screen(log_px.loc[PAIRS_FORMATION[0]:PAIRS_FORMATION[1]].dropna(), sectors)
    chosen = pairs.select(screened, pvalue, max_pairs=max_pairs)
    rows = {}
    for label, window in (("formation", PAIRS_FORMATION), ("out-of-sample", PAIRS_OOS)):
        m, _, _ = pairs_backtest.evaluate(log_px, chosen, window=window, **rules)
        rows[label] = {
            "gross annual return": m["gross_annual_return"],
            "net annual return": m["annual_return"],
            "gross Sharpe": m["gross_sharpe"],
            "net Sharpe": m["sharpe"],
            "max drawdown (net)": m["max_drawdown"],
            "trades": m["n_trades"],
            "hit rate": m["hit_rate"],
            "pairs": m["n_pairs"],
            "names screened": len(sectors),
            "pairs tested": len(screened),
        }
    return rows, chosen, screened


def pairs_comparison(px, panel, today, sectors_today, sectors_pit):
    """Run the within-sector pairs pipeline on both universes."""
    on_today = {c: c in today for c in px.columns}
    pit_day = panel.loc[:PAIRS_FORMATION[1]].iloc[-1]
    on_pit = {c: bool(pit_day.get(c, False)) for c in px.columns}
    px_a, sec_a = pairs_universe(px, on_today, sectors_today, pit=False)
    px_b, sec_b = pairs_universe(px, on_pit, sectors_pit, pit=True)
    rows_a, chosen_a, _ = pairs_result(px_a, sec_a)
    rows_b, chosen_b, _ = pairs_result(px_b, sec_b)
    tables = {w: compare(rows_a[w], rows_b[w]) for w in rows_a}
    return tables, {"today": chosen_a, "pit": chosen_b}
