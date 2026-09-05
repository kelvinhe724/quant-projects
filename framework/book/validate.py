"""Honesty checks on the book: walk-forward selection, deflated Sharpe, falsification, tearsheet.

Every parameter variant this file runs is appended to reports/trials.csv and
never removed, so the deflated Sharpe's trial count only grows. The
falsification battery follows the shape of
https://github.com/engineerinvestor/systematic-trend-following-with-managed-futures/blob/main/src/tf/eval/falsification.py
(MIT): placebo signs, higher costs, later signals, vol scaling without a signal.

Run: ../../.venv/bin/python3 -m framework.book.validate   (about 25 minutes, seven daily EWMAC runs)
"""
import importlib.util
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from purgedcv import (WalkForwardSplit, deflated_sharpe_ratio, min_track_record_length,
                      minimum_backtest_length, probabilistic_sharpe_ratio)
from scipy.stats import kurtosis, skew

from framework.book import allocate, optimizer, universe
from framework.book.strategies import BOOKS, EWMAC, SPEEDS, TrendETF, book_config, ex_ante_vol, sleeves
from framework.engine import Strategy, drawdown, run, sharpe

REPORTS = allocate.REPORTS
TRIALS = os.path.join(REPORTS, "trials.csv")
ARCHIVE = os.path.join(universe.ROOT, "archive", "framework-metrics-honesty.py")
GRID = {"lookback": (3, 6, 9, 12), "target": (0.2, 0.4)}
V1 = {"lookback": 12, "target": 0.4}
PLACEBO_DRAWS = 20
# Last fifth of the panel's sessions is the untouched window, the retired
# WalkForward(holdout=0.2) convention. Nothing is fit or picked on it.
HOLDOUT = 0.2
# Engine settings that change every configuration's return stream; logged
# with each trial so a rerun after an engine change counts as a new look.
ENGINE = {"buffer": book_config().risk.buffer}
PROMOTION = ("net Sharpe above zero on the untouched window, above TrendETF's on the same window, "
             "and a positive alpha t against the ETF universe there")
ALLOCATOR_PROMOTION = ("the candidate allocator goes live only if its best book's Sharpe on the untouched window "
                       "beats the best 1/N book's there, net of reallocation costs and financing; otherwise 1/N "
                       "stays live and the candidate is a documented trial")


class Wrap(Strategy):
    """Transform a sleeve's targets (fn) or hold them back `delay` bars. Keeps the inner name."""

    def __init__(self, inner, fn=None, delay=0):
        self.inner, self.fn, self.delay, self.queue = inner, fn, delay, []
        self.name = str(inner)

    def on_bar(self, asof, bars):
        t = self.inner.on_bar(asof, bars)
        if t is not None and self.fn:
            t = self.fn(t)
        if not self.delay:
            return t
        self.queue.append(t)
        return self.queue.pop(0) if len(self.queue) > self.delay else None


def log_trial(name, params, returns):
    """Append one variant to trials.csv; return the number of distinct variants ever logged."""
    r = returns[returns != 0]
    row = pd.DataFrame([{"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "name": name,
                         "params": json.dumps({**params, **ENGINE}, sort_keys=True), "sharpe_daily": r.mean() / r.std(),
                         "n_obs": len(r)}])
    row.to_csv(TRIALS, mode="a", header=not os.path.exists(TRIALS), index=False)


def trials():
    """Distinct trials ever logged. Same (name, params) keeps its latest run; identical return streams count once.

    The per-asset vol target is undone by the sleeve-level 10% vol target, so
    TrendETF target=0.2 and target=0.4 are the same trial, and so is v1 and
    lookback=12; counting them twice would inflate the deflation.
    """
    t = pd.read_csv(TRIALS).drop_duplicates(["name", "params"], keep="last")
    return t.drop_duplicates(["sharpe_daily", "n_obs"])


def stats(r):
    r = r[r != 0].dropna()
    s = allocate.stats(r)
    s["n_obs"] = len(r)
    return s


def walk_forward(grid_returns, v1_key, n_splits=5):
    """Pick the best train-window variant per fold; stitch its test returns. Returns (table, oos)."""
    R = grid_returns
    R = R.loc[R.index >= R[v1_key].ne(0).idxmax()]
    X = np.zeros((len(R), 1))
    rows, picked = [], []
    for k, (tr, te) in enumerate(WalkForwardSplit(n_splits=n_splits, test_size=756, prediction_times=R.index,
                                                  evaluation_times=R.index).split(X)):
        train, test = R.iloc[tr], R.iloc[te]
        best = train.apply(sharpe).idxmax()
        picked.append(test[best])
        rows.append({"fold": k + 1, "test_start": test.index[0].date(), "test_end": test.index[-1].date(),
                     "picked": best, "picked_train_sharpe": sharpe(train[best]),
                     "picked_test_sharpe": sharpe(test[best]), "v1_test_sharpe": sharpe(test[v1_key])})
    oos = pd.concat(picked)
    table = pd.DataFrame(rows).set_index("fold")
    return table, {"selected_oos_sharpe": sharpe(oos), "v1_oos_sharpe": sharpe(R[v1_key].loc[oos.index])}


def dsr_crosscheck(seed=3, n=100, length=750):
    """Same formula, two implementations, on the best of n random return series (risk #8)."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(0, 0.01, (n, length))
    srs = draws.mean(axis=1) / draws.std(axis=1, ddof=1)
    best = draws[srs.argmax()]
    spec = importlib.util.spec_from_file_location("honesty", ARCHIVE)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    ours = old.deflated_sharpe(best, n, sr_var=float(srs.var(ddof=1)))
    lib = deflated_sharpe_ratio(best, n, float(srs.var(ddof=1)))
    return {"purgedcv": float(lib), "archive": float(ours["dsr"]), "psr_archive": float(ours["psr"]),
            "psr_purgedcv": float(probabilistic_sharpe_ratio(best, 0.0))}


def falsify(bars, alloc):
    """Placebo and vol-scaling-only on the frozen trend rule; cost stress and signal delay on the live book."""
    live = {k: v for k, v in alloc["weights"].items() if v > 0}
    book = allocate.LIVE_BOOK
    rng = np.random.default_rng(0)
    names = list(universe.ETFS)
    placebo = []
    for i in range(PLACEBO_DRAWS):
        signs = dict(zip(names, rng.choice((-1.0, 1.0), len(names))))
        if all(v > 0 for v in signs.values()):
            continue
        r = run(Wrap(TrendETF(), fn=lambda t, s=signs: {n: w * s[n] for n, w in t.items()}), bars,
                config=book_config(kill=False))
        placebo.append(sharpe(r.returns[r.returns != 0]))
    trend = run(TrendETF(), bars, config=book_config(kill=False)).returns
    long_only = run(Wrap(TrendETF(), fn=lambda t: {n: abs(w) for n, w in t.items()}), bars,
                    config=book_config(kill=False)).returns
    log_trial("TrendETF", {"v": 1, "signs": "long_only"}, long_only)
    costs = {}
    for m in (1.0, 2.0, 4.0):
        r = run(sleeves(book), bars, config=book_config(kill=False, cost_scale=m, allocations=live)).returns
        costs[f"{m:g}x"] = stats(r)
        log_trial("book", {"book": book, "alloc": alloc["allocator"], "cost_scale": m}, r)
    delays = {}
    for d in (1, 2, 5):
        r = run([Wrap(s, delay=d) for s in sleeves(book)], bars,
                config=book_config(kill=False, allocations=live)).returns
        delays[f"+{d}"] = stats(r)
        log_trial("book", {"book": book, "alloc": alloc["allocator"], "delay": d}, r)
    p = np.array(placebo)
    return {
        "placebo": {"trend_sharpe": sharpe(trend[trend != 0]), "draws": len(p), "mean": float(p.mean()),
                    "p95": float(np.percentile(p, 95)), "beaten": int((p < sharpe(trend[trend != 0])).sum())},
        "vol_scaling_only": {"trend_signal": stats(trend), "long_only_vol_scaled": stats(long_only)},
        "cost_stress": costs, "signal_delay": delays,
    }


def attribution(returns, bars, instruments, target_vol=0.10):
    """Regress a sleeve's live net returns on owning its universe.

    Two benchmarks: the daily-rebalanced 1/N return of the instruments, and
    that return scaled to `target_vol` by the same 60-day EWMA vol the sleeves
    use, lagged one day and capped at 3x. Per benchmark: annualised alpha,
    its Newey-West t (Newey-West 1994 lag rule), beta, the benchmark Sharpe,
    and the Sharpe of returns minus beta x benchmark, which is alpha plus
    residual, the part of the sleeve that owning the universe does not explain.
    """
    import statsmodels.api as sm

    r = returns.loc[returns.ne(0).idxmax():]
    eq = bars.close[list(instruments)].pct_change().mean(axis=1)
    scale = (target_vol / ex_ante_vol(eq).shift(1)).clip(upper=3.0)
    out = {}
    for name, x in (("1/N", eq), ("1/N vol-targeted", eq * scale)):
        d = pd.DataFrame({"y": r, "x": x.reindex(r.index)}).dropna()
        lags = int(4 * (len(d) / 100) ** (2 / 9))
        fit = sm.OLS(d["y"], sm.add_constant(d["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        a, b = float(fit.params["const"]), float(fit.params["x"])
        out[name] = {"alpha": a * 252, "t_alpha": float(fit.tvalues["const"]), "beta": b,
                     "benchmark_sharpe": sharpe(d["x"]), "residual_sharpe": sharpe(d["y"] - b * d["x"]),
                     "sleeve_sharpe": sharpe(d["y"]), "n_obs": len(d)}
    return out


def gross_returns(res):
    """Net returns with each day's trade costs and borrow added back: the same path at zero cost.

    A separate zero-cost run would take a different path, because the buffer
    trades off the held weights and those depend on what costs have done to
    equity.
    """
    cost = res.trades.groupby("date")[["commission", "slippage"]].sum().sum(axis=1)
    charges = cost.reindex(res.equity.index).fillna(0.0) - res.carry.clip(upper=0.0)
    return (res.returns + charges / res.equity.shift(1)).fillna(0.0)


def window_stats(res, start):
    """stats() of a run from `start` on, with same-path gross Sharpe, turnover a year and trades."""
    r = res.returns.loc[start:]
    s = stats(r)
    g = gross_returns(res).loc[start:]
    s["sharpe_gross"] = sharpe(g[r != 0])
    s["turnover"] = float(res.turnover.loc[start:].sum() / (len(r) / 252))
    s["trades"] = int((res.trades["date"] >= start).sum())
    return s


def v2_candidates(bars, trend):
    """EWMAC at each speed and combined next to TrendETF over the window both are live. Returns (table, runs)."""
    runs = {"TrendETF": trend, "EWMAC": run(EWMAC(), bars, config=book_config(kill=False))}
    for f, s in SPEEDS:
        runs[f"EWMAC {f}/{s}"] = run(EWMAC(speeds=[(f, s)]), bars, config=book_config(kill=False))
    log_trial("EWMAC", {"v": 2, "speeds": "combined"}, runs["EWMAC"].returns)
    for f, s in SPEEDS:
        log_trial("EWMAC", {"v": 2, "speeds": f"{f}/{s}"}, runs[f"EWMAC {f}/{s}"].returns)
    start = max(r.returns.ne(0).idxmax() for r in runs.values())
    return {k: window_stats(res, start) for k, res in runs.items()}, runs


def candidate_books(bars, hold):
    """Every book in BOOKS at 1/N, kill off, from the first day all of them are live; in sample and from `hold` on."""
    res = {}
    for name in BOOKS:
        s = sleeves(name)
        res[name] = run(s, bars, config=book_config(kill=False, allocations={str(x): 1 / len(s) for x in s}))
        log_trial("book", {"book": name, "alloc": "equal"}, res[name].returns)
    start = max(r.returns.ne(0).idxmax() for r in res.values())
    table = {}
    for name, r in res.items():
        row = window_stats(r, start)
        oos = r.returns.loc[hold:]
        row["oos_sharpe"] = sharpe(oos[oos != 0])
        row["oos_max_drawdown"] = drawdown(oos)[1]["max_drawdown"]
        table[name] = row
    return table, res, start


def candidate_allocator(books_res, bars, hold):
    """optimizer.walk on every book's sleeve streams: candidate vs daily-rebalanced 1/N, whole walk and untouched window.

    Each book's candidate stream is a logged trial in trials.csv and a run in
    the research registry under reports/registry/, keyed on the allocator's
    parameters; the registry's own DSR is reported next to the book's.
    """
    from research.registry.experiments import Registry

    reg = Registry(os.path.join(REPORTS, "registry"))
    fin, source = optimizer.financing(bars)
    params = optimizer.asdict(optimizer.Params())
    table, streams, paths = {}, {}, {}
    for name, res in books_res.items():
        oos, path, log = optimizer.walk(res.returns_by_strategy, res, fin)
        row = optimizer.summary(oos, log, hold)
        log_trial("book", {"book": name, "alloc": "mvo", "oos": "quarterly refit", **params}, oos["mvo"])
        entry = reg.record(f"allocator:{name}", {"allocator": "mvo", "cov": str(log["cov"].iloc[-1]), **params},
                           list(res.books), (oos.index[0].date(), oos.index[-1].date()), oos["mvo"],
                           tags={"stage": "candidate allocator"})
        row["registry_id"] = entry["id"]
        table[name], streams[name], paths[name] = row, oos, path
    return table, streams, paths, source, reg


def md_table(d, cols=("sharpe", "annual_return", "annual_vol", "max_drawdown")):
    out = ["| variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, s in d.items():
        out.append(f"| {k} | " + " | ".join(f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    bars = universe.load_bars()
    results = allocate.run_sleeves(bars, kill=False)
    # The v1 sleeves stay in the attribution and kill tables whatever book is live.
    v1 = allocate.run_sleeves(bars, "v1", kill=False) if allocate.LIVE_BOOK != "v1" else {}
    every = {**v1, **results}
    for k, r in every.items():
        log_trial(k, {"alone": True}, r.returns)
    alloc, erc_table, erc_years, oos_alloc, R = allocate.allocate(bars, results)
    live = {k: v for k, v in alloc["weights"].items() if v > 0}

    grid = {}
    for lb, tv in itertools.product(*GRID.values()):
        key = f"TrendETF lookback={lb} target={tv}"
        r = run(TrendETF(lookback=lb, target=tv), bars, config=book_config(kill=False))
        grid[key] = r.returns
        log_trial("TrendETF", {"lookback": lb, "target": tv}, r.returns)
    wf_table, wf = walk_forward(pd.DataFrame(grid), f"TrendETF lookback={V1['lookback']} target={V1['target']}")

    book = run(sleeves(allocate.LIVE_BOOK), bars, config=book_config(kill=False, allocations=live))
    r = book.returns[book.returns != 0]
    head = stats(r)
    hold = bars.index[len(bars.index) - int(round(len(bars.index) * HOLDOUT))]
    for k in ("erc", "equal"):
        log_trial("book", {"book": allocate.LIVE_BOOK, "alloc": k, "oos": "quarterly refit"}, oos_alloc[k])

    kill_dates = {}
    killed = allocate.run_sleeves(bars, kill=True)
    if v1:
        killed = {**allocate.run_sleeves(bars, "v1", kill=True), **killed}
    for k, res in killed.items():
        eq = res.equity
        dd = eq / eq.cummax() - 1
        kill_dates[k] = str(dd.index[(dd <= -0.25).argmax()].date()) if (dd <= -0.25).any() else "never"
        log_trial(k, {"alone": True, "kill": True}, res.returns)

    fals = falsify(bars, alloc)
    xcheck = dsr_crosscheck()
    trend = every["TrendETF"]
    v2_table, v2_runs = v2_candidates(bars, trend)
    attr = {k: attribution(res.returns, bars, universe.SLEEVES[k]) for k, res in every.items()}
    attr["EWMAC"] = attribution(v2_runs["EWMAC"].returns, bars, universe.SLEEVES["TrendETF"])

    ew_R = pd.DataFrame({k: v.returns for k, v in v2_runs.items() if k != "TrendETF"})
    ew_wf_table, ew_wf = walk_forward(ew_R.loc[ew_R.index < hold], "EWMAC", n_splits=4)
    ew_hold = v2_runs["EWMAC"].returns.loc[hold:]
    log_trial("EWMAC", {"v": 2, "speeds": "combined", "window": "untouched"}, ew_hold)
    untouched = {"start": str(hold.date()), "end": str(ew_hold.index[-1].date()),
                 "EWMAC": window_stats(v2_runs["EWMAC"], hold), "TrendETF": window_stats(trend, hold),
                 "attribution": attribution(ew_hold, bars, universe.SLEEVES["TrendETF"]), "rule": PROMOTION}
    untouched["clears"] = bool(untouched["EWMAC"]["sharpe"] > 0
                               and untouched["EWMAC"]["sharpe"] > untouched["TrendETF"]["sharpe"]
                               and untouched["attribution"]["1/N"]["t_alpha"] > 0)

    books_table, books_res, books_start = candidate_books(bars, hold)
    oos_winner = max(books_table, key=lambda k: books_table[k]["oos_sharpe"])
    gate_oos, _ = allocate.walk(books_res[oos_winner].returns_by_strategy)
    gate_table, _ = allocate.compare(gate_oos)
    cand_table, cand_streams, cand_paths, fin_source, reg = candidate_allocator(books_res, bars, hold)

    # Every configuration above is now in trials.csv; count after, not before.
    tr = trials()
    n_trials = len(tr)
    var_sharpe = float(tr["sharpe_daily"].var(ddof=1))
    head["psr"] = float(probabilistic_sharpe_ratio(r.to_numpy(), 0.0))
    head["dsr"] = float(deflated_sharpe_ratio(r.to_numpy(), n_trials, var_sharpe))
    head["dsr_with_placebo"] = float(deflated_sharpe_ratio(r.to_numpy(), n_trials + fals["placebo"]["draws"],
                                                           var_sharpe))
    head["n_trials"] = n_trials
    for name, row in books_table.items():
        br = books_res[name].returns.loc[books_start:]
        row["dsr"] = float(deflated_sharpe_ratio(br[br != 0].to_numpy(), n_trials, var_sharpe))
    for name, row in cand_table.items():
        cr = cand_streams[name]["mvo"]
        cr = cr[cr != 0].to_numpy()
        row["dsr"] = float(deflated_sharpe_ratio(cr, n_trials, var_sharpe))
        row["dsr_registry"] = reg.dsr(cand_streams[name]["mvo"], extra_trials=n_trials - reg.trials())["dsr"]
    cand_best = max(cand_table, key=lambda k: cand_table[k]["oos_sharpe_mvo"])
    cand_promoted = bool(cand_table[cand_best]["oos_sharpe_mvo"] > books_table[oos_winner]["oos_sharpe"])
    pd.DataFrame({d: p["mvo"] for d, p in cand_paths[allocate.LIVE_BOOK].items()}).T.to_csv(
        os.path.join(REPORTS, "mvo_weights.csv"))
    head["n_runs_logged"] = int(len(pd.read_csv(TRIALS)))
    head["min_btl_years"] = float(minimum_backtest_length(n_trials, head["sharpe"]))
    head["backtest_years"] = len(r) / 252
    sr_d = r.mean() / r.std()
    head["min_trl_years"] = float(min_track_record_length(sr_d, 0.0, 0.05, float(skew(r)),
                                                          float(kurtosis(r, fisher=False)))) / 252

    import quantstats as qs
    spy = bars.close["SPY"].pct_change().reindex(r.index).fillna(0.0).rename("SPY")
    qs.reports.html(r.rename("Premia book"), benchmark=spy, output=os.path.join(REPORTS, "tearsheet.html"),
                    title="Premia book (shadow config, net of costs)")

    with open(os.path.join(REPORTS, "validation.json"), "w") as fh:
        json.dump({"headline": head, "live_book": allocate.LIVE_BOOK, "allocator": alloc, "walk_forward": wf,
                   "falsification": fals, "v2_candidates": v2_table, "ewmac_walk_forward": ew_wf,
                   "untouched": untouched, "books": books_table, "books_start": str(books_start.date()),
                   "oos_winner": oos_winner,
                   "winner_erc_gate": {k: {m: v for m, v in gate_table[k].items()} for k in gate_table},
                   "candidate_allocator": {"params": optimizer.asdict(optimizer.Params()), "financing": fin_source,
                                           "books": cand_table, "best": cand_best, "promoted": cand_promoted,
                                           "rule": ALLOCATOR_PROMOTION, "live_allocator": allocate.LIVE_ALLOCATOR,
                                           "registry_trials": reg.trials()},
                   "attribution": attr, "dsr_crosscheck": xcheck, "kill_dates": kill_dates,
                   "panel_hash": universe.panel_hash(bars),
                   "run_at": pd.Timestamp.now().isoformat(timespec="seconds")}, fh, indent=2, default=str)

    lines = [
        "# Validation", "",
        f"Run {pd.Timestamp.now():%Y-%m-%d %H:%M}, panel hash `{universe.panel_hash(bars)}`, "
        f"{head['start']} to {head['end']} ({head['backtest_years']:.1f} years), shadow cost model, "
        f"kill switch off for backtests (see below).", "",
        "## Headline (book, live allocator, net)", "",
        f"| Sharpe | ann. return | ann. vol | max drawdown | PSR | DSR | trials | MinBTL | MinTRL |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {head['sharpe']:.3f} | {head['annual_return']:.2%} | {head['annual_vol']:.2%} | "
        f"{head['max_drawdown']:.1%} | {head['psr']:.3f} | {head['dsr']:.3f} | {n_trials} | "
        f"{head['min_btl_years']:.1f} y (have {head['backtest_years']:.1f}) | {head['min_trl_years']:.1f} y |", "",
        f"Live book `{allocate.LIVE_BOOK}`. DSR uses n_trials = {n_trials}: every configuration this file runs on "
        "the real panel (sleeves, the trend grid, allocators, cost and delay variants, kill on, the EWMAC "
        f"variants, the candidate books, the candidate allocator) is appended to `trials.csv` ({head['n_runs_logged']} rows so far) and "
        f"identical return streams count once; the variance of their daily Sharpes is {var_sharpe:.2e}. Trials "
        f"are tagged with the engine settings ({ENGINE}), so the runs before the buffer moved into the engine "
        "still count: they were looked at. The per-asset vol target is undone by the sleeve-level 10% target, so "
        f"the trend grid is really four lookbacks. The {fals['placebo']['draws']} placebo draws are a null, not "
        f"candidates; counting them too gives DSR {head['dsr_with_placebo']:.3f}. PSR is the probability the true "
        "Sharpe is above zero; DSR the same after deflating for selection. MinTRL is the track length needed to "
        "reject zero at 95%.", "",
        "## Allocator: ERC vs 1/N, out of sample", "",
        "Quarterly refits, three-year window, Ledoit-Wolf covariance, weights applied to the next quarter. "
        f"Live allocator: **{alloc['allocator']}** with weights {alloc['weights']}"
        + (f" (ERC would have been {alloc['erc_weights']})." if alloc["allocator"] == "equal" else "."), "",
        erc_table.to_markdown(), "",
        "Sharpe by year:", "", erc_years.round(2).to_markdown(), "",
        "## Walk-forward over the trend grid", "",
        f"Grid: lookback {GRID['lookback']} months x per-asset vol target {GRID['target']}. purgedcv "
        "WalkForwardSplit, 5 folds of 3 years, the variant with the best training Sharpe is held in the test "
        f"window. Selected-in-sample OOS Sharpe **{wf['selected_oos_sharpe']:.3f}** vs frozen v1 "
        f"**{wf['v1_oos_sharpe']:.3f}** over the same test windows.", "",
        wf_table.round(3).to_markdown(), "",
        "## Falsification", "",
        f"**Placebo.** Trend sleeve with a fixed random sign per ETF, {fals['placebo']['draws']} draws: "
        f"actual Sharpe {fals['placebo']['trend_sharpe']:.3f}, placebo mean {fals['placebo']['mean']:.3f}, "
        f"p95 {fals['placebo']['p95']:.3f}, beats {fals['placebo']['beaten']} of {fals['placebo']['draws']}.", "",
        "**Vol scaling without the signal.** Same sizing, every sign forced long:", "",
        md_table(fals["vol_scaling_only"]), "",
        "**Cost stress.** Book at multiples of the cost model:", "", md_table(fals["cost_stress"]), "",
        "**Signal delay.** Every sleeve's targets held back N extra bars:", "", md_table(fals["signal_delay"]), "",
        "## v2 candidate: EWMAC", "",
        f"EWMAC at {', '.join(f'{f}/{s}' for f, s in SPEEDS)} and the three combined with a forecast "
        "diversification multiplier, forecast scalars estimated on trailing data only (expanding, NaN for the "
        "first 500 days), cap 20, forecast / 10 x TrendETF's 40% per-asset target (35-day vol where TrendETF "
        "uses 60-day) and class balance, sent daily, same universe, cost model and overlay. The engine's 10% "
        "buffer sits on the final weights after the overlay, per instrument: an instrument trades only when its "
        f"held weight is outside target x (1 +/- 0.1). Window {v2_table['EWMAC']['start']} to "
        f"{v2_table['EWMAC']['end']}, where both rules are live. Gross is the same path with each day's trade "
        "costs and borrow added back. Turnover is traded notional over equity per year. Each variant is a "
        "logged trial.", "",
        md_table(v2_table, ("sharpe_gross", "sharpe", "annual_return", "annual_vol", "max_drawdown", "turnover",
                            "trades")), "",
        "## EWMAC: walk-forward and the untouched window", "",
        f"Walk-forward over the four EWMAC variants on the sessions before the untouched window, {len(ew_wf_table)} "
        "folds of 3 years, best training Sharpe held in the test window. Selected-in-sample OOS Sharpe "
        f"**{ew_wf['selected_oos_sharpe']:.3f}** vs the frozen combined rule **{ew_wf['v1_oos_sharpe']:.3f}** "
        "over the same test windows.", "",
        ew_wf_table.round(3).to_markdown(), "",
        f"**Untouched window** {untouched['start']} to {untouched['end']}, the last {HOLDOUT:.0%} of the panel's "
        "sessions. Nothing was fit or picked on it; its return was inside the one full-panel look above, so it "
        "confirms rather than discovers. Promotion rule, fixed before this run: " + PROMOTION + ".", "",
        "| rule | Sharpe gross | Sharpe net | ann. return | ann. vol | max drawdown | turnover | trades |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {k} | {s['sharpe_gross']:.3f} | {s['sharpe']:.3f} | {s['annual_return']:.2%} | {s['annual_vol']:.2%} | "
          f"{s['max_drawdown']:.1%} | {s['turnover']:.1f}x | {s['trades']} |"
          for k, s in ((k, untouched[k]) for k in ("EWMAC", "TrendETF"))], "",
        "EWMAC against its universe on the window: "
        + "; ".join(f"{b}: alpha {s['alpha']:+.2%} (t {s['t_alpha']:.2f}), beta {s['beta']:.2f}, benchmark Sharpe "
                    f"{s['benchmark_sharpe']:.2f}, residual Sharpe {s['residual_sharpe']:.2f}"
                    for b, s in untouched["attribution"].items())
        + f". **Clears the rule: {'yes' if untouched['clears'] else 'no'}.**", "",
        "## Candidate books", "",
        f"Every book in `strategies.BOOKS` at 1/N of its sleeves, kill off, from {books_start.date()} (the first "
        "day all four are live) to the end; the same window, engine and cost model for all four. DSR uses the "
        "trial count above. OOS is the untouched window; the live book is the OOS winner only if it beats `v1` "
        f"there. **OOS winner: `{oos_winner}`** (live book `{allocate.LIVE_BOOK}`).", "",
        md_table(books_table, ("sharpe_gross", "sharpe", "dsr", "annual_return", "annual_vol", "max_drawdown",
                               "turnover", "oos_sharpe", "oos_max_drawdown")), "",
        f"ERC vs 1/N gate on `{oos_winner}`'s sleeves, quarterly refits, out of sample:", "",
        gate_table.to_markdown(), "",
        "## Candidate allocator: cost-aware mean-variance", "",
        "`optimizer.py` on each book's sleeve streams: maximise mu'w - w'Sw / 2 - cost'|dw| over long-only sleeve "
        f"weights, parameters {optimizer.asdict(optimizer.Params())} fixed before the run. mu is the trailing "
        "three-year mean, S Ledoit-Wolf (no `risk-model/` at run time), cost each sleeve's realised cost per traded "
        "notional times its gross. Quarterly refits, weights held for the next quarter, the move charged on its "
        f"first day, leverage above 1 financed at {fin_source} plus the spread. Both columns are daily-rebalanced "
        "to their weights; the 1/N column here is that convention, the books table above is the engine's static "
        "1/N, and the rule compares against the engine's. Every row is a logged trial and a research-registry run "
        f"({reg.trials()} there). Rule, written before the run: " + ALLOCATOR_PROMOTION + ".", "",
        "| book | walk Sharpe mvo | walk Sharpe 1/N | DSR | DSR (registry) | untouched Sharpe mvo | untouched "
        "Sharpe 1/N (walk) | untouched Sharpe 1/N (engine) | mean leverage | turnover / refit | refits | budget "
        "bound | vol bound | Kelly scaled |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {k} | {s['sharpe_mvo']:.3f} | {s['sharpe_equal']:.3f} | {s['dsr']:.3f} | {s['dsr_registry']:.3f} | "
          f"{s['oos_sharpe_mvo']:.3f} | {s['oos_sharpe_equal']:.3f} | {books_table[k]['oos_sharpe']:.3f} | "
          f"{s['mean_leverage']:.2f} | {s['turnover_per_refit']:.2f} | {s['refits']} | {s['budget_bound']} | "
          f"{s['vol_bound']} | {s['kelly_scaled']} |" for k, s in cand_table.items()], "",
        f"Best candidate book on the untouched window: `{cand_best}` at {cand_table[cand_best]['oos_sharpe_mvo']:.3f} "
        f"against the best 1/N book `{oos_winner}` at {books_table[oos_winner]['oos_sharpe']:.3f}. "
        f"**Promoted: {'yes' if cand_promoted else 'no'}.** `allocate.LIVE_ALLOCATOR = \"{allocate.LIVE_ALLOCATOR}\"`. "
        "The live book's candidate weight path is in `mvo_weights.csv`.", "",
        "## Signal vs beta", "",
        "Each sleeve's live net returns regressed on its own universe: the daily-rebalanced 1/N return of its "
        "instruments, and that return scaled to the sleeves' 10% vol target (60-day EWMA vol, one-day lag, 3x "
        "cap). Alpha is annualised with a Newey-West t; residual Sharpe is the Sharpe of the sleeve minus "
        "beta x benchmark. A sleeve whose residual Sharpe is below the benchmark's own Sharpe is not adding "
        "return beyond owning the universe.", "",
        "| sleeve | benchmark | benchmark Sharpe | sleeve Sharpe | alpha | t | beta | residual Sharpe |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {k} | {b} | {s['benchmark_sharpe']:.3f} | {s['sleeve_sharpe']:.3f} | {s['alpha']:+.2%} | "
          f"{s['t_alpha']:.2f} | {s['beta']:.3f} | {s['residual_sharpe']:.3f} |"
          for k, d in attr.items() for b, s in d.items()], "",
        "## Kill switch", "",
        "Backtests above run with `kill_dd=None`. With the live 25% kill on, the sleeves would have been "
        "flattened permanently on: " + ", ".join(f"{k} {v}" for k, v in kill_dates.items()) + ". Live, a kill "
        "stops the sleeve until Kelvin restarts it, which no backtest can model.", "",
        "## purgedcv vs archived formula (risk #8)", "",
        f"Best of 100 random series, 750 days: DSR purgedcv {xcheck['purgedcv']:.4f} vs archive "
        f"{xcheck['archive']:.4f}; PSR {xcheck['psr_purgedcv']:.4f} vs {xcheck['psr_archive']:.4f}. "
        "purgedcv is pinned at 0.1.5 in requirements.txt.", "",
        f"Tearsheet: `reports/tearsheet.html` (quantstats, SPY benchmark). {time.time() - t0:.0f}s.",
    ]
    with open(os.path.join(REPORTS, "validation.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
