"""Covariance estimators raced on the point-in-time S&P 500 and the book's ETFs, then the live book's risk.

Prices through the lake (the 13 ETFs the lake lacked are written through
lake.write first), Fama-French daily factors from French's library, GARCH
from garch/, the untouched last fifth locked and opened once through
research/, every minimum-variance stream logged to the registry.

Run: ../.venv/bin/python3 run.py   (about 20 seconds after the first run's ETF download)
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import risk
from framework.book import validate
from framework.book.daemon import read_ledger
from research.alpha import Untouched, UntouchedWindowUsed
from research.features.store import panel_hash
from research.registry import Registry

REPORTS = os.path.join(HERE, "reports")
SP_START, SP_END = "2003-06-02", "2026-08-31"
HOLDOUT = validate.HOLDOUT
N_SUB = 100
UNIVERSES = ("etf", "sp100", "sp500")
# Frozen before any result: the desk estimator is the one with the lowest
# out-of-sample minimum-variance vol on the full point-in-time panel before
# the untouched window; the book's parametric VaR uses it.
RULE_UNIVERSE = "sp500"


def race_stats(res, w0):
    r = res["returns"]
    rows = {}
    for e in r.columns:
        pre = r[e][r.index < w0]
        rows[e] = {"vol_pre": risk.ann_vol(pre), "n_pre": int(pre.notna().sum()),
                   "leverage_median": float(res["leverage"][e].median()),
                   "leverage_max": float(res["leverage"][e].max())}
    return pd.DataFrame(rows).T


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    wrote = risk.ensure_etfs()
    print(f"lake: {wrote} ETF rows written")
    sp = risk.load_panel(start=SP_START, end=SP_END, universe_="sp500")
    ret_sp = risk.returns(sp)
    ret_etf = risk.etf_returns()
    factors = risk.load_factors()
    print(f"S&P panel {ret_sp.shape[0]} sessions x {ret_sp.shape[1]} names, ETFs {ret_etf.shape}, "
          f"factors through {factors.index[-1].date()}")

    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    lock.lock(ret_sp.index, HOLDOUT, panel_hash(sp["close"]))
    w0, w1 = lock.window
    print(f"untouched window {w0.date()} to {w1.date()}, opened: {lock.opened}")

    dv = (sp["raw_close"] * sp["volume"]).rolling(252, min_periods=126).median()
    pick100 = lambda names, t: dv.loc[t, names].dropna().nlargest(N_SUB).index.tolist()
    races = {"etf": risk.minvar_race(ret_etf.loc[:SP_END, risk.ETFS], factors),
             "sp100": risk.minvar_race(ret_sp, factors, pick=pick100),
             "sp500": risk.minvar_race(ret_sp, factors)}
    print(f"races done {time.time() - t0:.0f}s")
    stats = {u: race_stats(r, w0) for u, r in races.items()}
    reg = Registry(os.path.join(REPORTS, "registry"))
    for u, r in races.items():
        for e in r["returns"].columns:
            pre = r["returns"][e][r["returns"].index < w0].dropna()
            reg.record(f"GMV[{e}]", {"estimator": e, "universe": u, "window": risk.WINDOW}, [u],
                       (pre.index[0].date(), pre.index[-1].date()), pre, tags={"stage": "estimator_race"})
    winner = stats[RULE_UNIVERSE]["vol_pre"].idxmin()
    print(f"winner on {RULE_UNIVERSE} pre-window: {winner}; {reg.trials()} trials logged")
    for u in UNIVERSES:
        print(f"\n{u}\n{stats[u].round(4)}")

    ledger = read_ledger()
    last = ledger[ledger["book"] == "shadow"].iloc[-1]
    positions = json.loads(last["positions"])
    book_date = min(pd.Timestamp(last["date"]), ret_etf[list(positions)].dropna().index[-1])
    print(f"\nlive book {last['book']} as of {pd.Timestamp(last['date']).date()}, risk as of {book_date.date()}")
    ex = risk.exposures(positions, book_date, ret_etf, factors, estimator=winner)
    var_by_est = {e: risk.var(positions, book_date, ret_etf, method="parametric", estimator=e, factors=factors)
                  for e in risk.ESTIMATORS}
    v = risk.var(positions, book_date, ret_etf, method="all", estimator=winner, factors=factors)
    st = risk.stress(positions, ret_etf)
    print(f"book vol {ex['vol']:.2%} ({winner}), k={ex['k']}, FF through {ex['ff_through']}")
    print(ex["assets"].round(4))
    print(ex["by_class"].round(4))
    print(ex["ff"].round(4))
    print(ex["stat"].round(4))
    print({m: {k: round(x, 5) for k, x in d.items()} for m, d in v.items()})
    print({e: round(d["var"], 5) for e, d in var_by_est.items()})
    for k, d in st.items():
        print(f"stress {k}: {d['return']:+.2%}, worst day {d['worst_day']:+.2%} on {d['worst_date']}, "
              f"max drawdown {d['max_drawdown']:.2%}, missing {d['missing']}")

    bt = risk.var_backtest(ret_etf.loc[:SP_END], positions, estimator=winner, factors=factors)
    print(f"VaR backtest {bt.index[0].date()} to {bt.index[-1].date()}, {len(bt)} days, {time.time() - t0:.0f}s")
    cov_pre = risk.coverage(bt[bt.index < w0])
    print("coverage pre-window", {m: (round(d["rate"], 4), round(d["p_value"], 3)) for m, d in cov_pre.items()})

    def on_window(start, end):
        mv = {u: {e: {"vol": risk.ann_vol(r["returns"][e].loc[start:end]),
                      "n": int(r["returns"][e].loc[start:end].notna().sum())} for e in r["returns"].columns}
              for u, r in races.items()}
        cov = risk.coverage(bt.loc[start:end])
        holds = min(mv[RULE_UNIVERSE], key=lambda e: mv[RULE_UNIVERSE][e]["vol"]) == winner
        return {"winner": winner, "winner_holds": bool(holds), "minvar": mv, "var_coverage": cov,
                "dsr_winner": reg.dsr(races[RULE_UNIVERSE]["returns"][winner].loc[start:end])}

    try:
        window = lock.open(on_window)
        print("untouched window opened")
    except UntouchedWindowUsed:
        window = lock.result
        print("untouched window already spent; reading the stored result")
    print(json.dumps(window["minvar"], indent=1, default=float))
    print("coverage on window", {m: (round(d["rate"], 4), round(d["p_value"], 3)) for m, d in window["var_coverage"].items()})

    charts(races, stats, w0, w1, ex, bt, sp, ret_sp)
    summary = {"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "rule": RULE_UNIVERSE, "winner": winner,
               "sp_panel": {"sessions": int(ret_sp.shape[0]), "names": int(ret_sp.shape[1]), "start": SP_START,
                            "end": SP_END, "close_hash": panel_hash(sp["close"])},
               "etf_panel": {"sessions": int(ret_etf.shape[0]), "end": str(ret_etf.index[-1].date())},
               "factors_through": str(factors.index[-1].date()), "window": risk.WINDOW,
               "untouched": {"start": str(w0.date()), "end": str(w1.date())},
               "race_pre": {u: s.to_dict() for u, s in stats.items()},
               "k": {u: r["k"].describe().to_dict() for u, r in races.items() if len(r["k"])},
               "trials": reg.trials(), "window_result": window,
               "book": {"name": last["book"], "ledger_date": str(pd.Timestamp(last["date"]).date()),
                        "risk_date": str(book_date.date()), "positions": positions, "gross": float(sum(abs(x) for x in positions.values())),
                        "net": float(sum(positions.values())), "vol": ex["vol"], "ff_vol": ex["ff_vol"], "k": ex["k"],
                        "ff_through": ex["ff_through"], "assets": ex["assets"].to_dict(), "by_class": ex["by_class"].to_dict(),
                        "ff": ex["ff"].to_dict(), "stat": ex["stat"].to_dict(), "var": v,
                        "var_parametric_by_estimator": var_by_est, "stress": st},
               "var_backtest": {"start": str(bt.index[0].date()), "end": str(bt.index[-1].date()), "n": int(len(bt)),
                                "coverage_pre": cov_pre},
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.concat({u: r["returns"] for u, r in races.items()}, axis=1).to_csv(os.path.join(REPORTS, "minvar_returns.csv"))
    pd.concat({u: r["leverage"] for u, r in races.items()}, axis=1).to_csv(os.path.join(REPORTS, "minvar_leverage.csv"))
    ex["assets"].to_csv(os.path.join(REPORTS, "book_assets.csv"))
    ex["betas"].to_csv(os.path.join(REPORTS, "book_betas.csv"))
    bt.to_csv(os.path.join(REPORTS, "var_backtest.csv"))
    print(f"done in {time.time() - t0:.0f}s -> {os.path.relpath(REPORTS, HERE)}")


def charts(races, stats, w0, w1, ex, bt, sp, ret_sp):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    for ax, u in zip(axes, UNIVERSES):
        r = races[u]["returns"]
        rv = r.rolling(126).std() * np.sqrt(252)
        for e in r.columns:
            ax.plot(rv.index, rv[e], lw=1, label=f"{e} (pre {stats[u].loc[e, 'vol_pre']:.1%})")
        ax.axvspan(w0, w1, color="grey", alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(f"{u}: 126-day realised vol of each estimator's minimum-variance portfolio")
        ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "minvar_rolling_vol.png"), dpi=120)
    plt.close(fig)

    win = ret_sp.iloc[-risk.WINDOW:]
    win = win[win.columns[win.notna().all()]]
    lam = np.linalg.eigvalsh(np.corrcoef(win.to_numpy(), rowvar=False))[::-1]
    edge = risk.mp_edge(win.shape[1], len(win))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(lam[lam < 5], bins=60, color="steelblue")
    axes[0].axvline(edge, color="red", ls="--", label=f"MP edge {edge:.2f}")
    axes[0].set_title(f"S&P eigenvalues, {len(win)} sessions, N={win.shape[1]} "
                      f"(top {lam[0]:.0f} off scale, k={risk.n_factors(lam, len(win))})", fontsize=10)
    axes[0].legend()
    k = races["sp500"]["k"]
    axes[1].plot(k.index, k.to_numpy(), drawstyle="steps-post")
    axes[1].axvspan(w0, w1, color="grey", alpha=0.15)
    axes[1].set_title("factors above the edge at each monthly rebalance, full panel")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "mp_spectrum.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    a = ex["assets"].sort_values("pct")
    axes[0].barh(a.index, a["pct"], color=np.where(a["weight"] < 0, "firebrick", "steelblue"))
    axes[0].set_title(f"share of book variance by ETF (vol {ex['vol']:.1%}, red = short)")
    f = ex["ff"].drop("residual")
    axes[1].bar(f.index, f["exposure"], color="steelblue")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title(f"Fama-French exposures, betas through {ex['ff_through']}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "book_risk.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    b = bt.loc["2019-01-01":]
    ax.plot(b.index, b["return"], lw=0.5, color="grey", label="book return, weights fixed")
    for m, c in (("parametric", "steelblue"), ("historical", "darkorange"), ("fhs", "green")):
        ax.plot(b.index, -b[m], lw=1, color=c, label=f"-VaR {m}")
    hit = b[b["return"] < -b["fhs"]]
    ax.scatter(hit.index, hit["return"], color="red", s=12, zorder=5, label="FHS breach")
    ax.axvspan(w0, w1, color="grey", alpha=0.15)
    ax.set_title("1-day 99% VaR of today's weights against the next day's return")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "var_backtest.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
