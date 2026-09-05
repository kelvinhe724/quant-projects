"""Regime overlay on the live premia book: HMM states, macro nowcast, walk-forward choice, untouched window.

SPY and the FRED series come from the data lake, the book from the
framework's own loader. Three HMM sizes with and without the macro leg are
scored as a scale on the bare book's daily returns; the walk-forward picks
one; that one runs through the engine on top of both live sleeves and is
compared with the bare book on the untouched last fifth, opened once and
stored in reports/untouched.json. Every look is logged to the registry.

Run: ../.venv/bin/python3 run.py   (about 10 minutes, two daily engine runs)
"""
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data-lake"))
import lake
import regimes
from framework.book import universe, validate
from framework.book.strategies import book_config, sleeves
from framework.engine import run, sharpe
from research.alpha import Untouched, UntouchedWindowUsed
from research.models import ModelRegistry
from research.registry import Registry

REPORTS = os.path.join(HERE, "reports")
BOOK = "beta+alpha"
ALLOC = {"ETFBeta": 0.5, "EWMAC": 0.5}
FRED = {"slope": "T10Y3M", "payrolls": "PAYEMS", "vix": "VIXCLS"}
STATES = (2, 3, 4)
FIRST_YEAR = 2006
FOLDS = 4
HOLDOUT = validate.HOLDOUT
warnings.filterwarnings("ignore", message="Model is not converging")


def stats(r):
    r = r[r != 0].dropna()
    return validate.stats(r) if len(r) > 1 else {}


def load():
    bars = universe.load_bars()
    asof = bars.asof
    spy = lake.load("equities_daily", "2003-01-01", None, universe=["SPY"], as_of=asof)
    spy = pd.Series(spy["adj_close"].to_numpy(), index=pd.DatetimeIndex(spy["date"])).sort_index()
    fred = lake.load("fred", "1990-01-01", None, universe=list(FRED.values()), as_of=asof)
    fred = fred.pivot(index="date", columns="series", values="value").sort_index()
    return bars, spy, {k: fred[v].dropna() for k, v in FRED.items()}


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    bars, spy, fred = load()
    cal = bars.index
    obs = regimes.observations(spy)
    nc = regimes.nowcast(fred["slope"], fred["payrolls"], fred["vix"], obs.index)
    probs, models = {}, {}
    for k in STATES:
        probs[k], models[k] = regimes.walk_forward_probs(obs, k, FIRST_YEAR)
    scales = {}
    for k in STATES:
        for macro in (False, True):
            s = regimes.scale(probs[k][k - 1], nc if macro else None)
            scales[f"K{k}{'+macro' if macro else ''}"] = s.reindex(cal, method="ffill").fillna(1.0)
    print(f"SPY {obs.index[0].date()} to {obs.index[-1].date()}, {len(obs)} observations; "
          f"nowcast from {nc.first_valid_index().date()}; HMM refits {min(models[2])} to {max(models[2])}; "
          f"{time.time() - t0:.0f}s")

    registry = Registry(os.path.join(REPORTS, "registry"))
    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    data_hash = universe.panel_hash(bars)
    lock.lock(cal, HOLDOUT, data_hash)
    hold, end = lock.window
    pre = cal[cal < hold]
    print(f"book panel {cal[0].date()} to {cal[-1].date()}, {len(cal)} sessions, hash {data_hash}; untouched "
          f"{hold.date()} to {end.date()}" + (f", opened at {lock.read()['opened_at']}" if lock.opened else ""))

    base = run(sleeves(BOOK), bars, config=book_config(kill=False, allocations=ALLOC))
    print(f"bare {BOOK} book: Sharpe {base.metrics['sharpe']:.3f}, {len(base.trades)} trades, {time.time() - t0:.0f}s")
    instruments = list(universe.ETFS)
    R = pd.DataFrame({"none": base.returns})
    for v, s in scales.items():
        R[v] = base.returns * s.shift(1).fillna(1.0)
    R = R.loc[pre]
    for v in R.columns:
        k, macro = (None, None) if v == "none" else (int(v[1]), v.endswith("+macro"))
        registry.record(f"{BOOK} x overlay {v}", {"states": k, "macro": macro, "path": "proxy"}, instruments,
                        (pre[0].date(), pre[-1].date()), R[v], tags={"stage": "grid_proxy"})
    wf_table, wf = validate.walk_forward(R, "none", n_splits=FOLDS)
    registry.record(f"{BOOK} overlay walk-forward", {"grid": list(R.columns), "folds": FOLDS}, instruments,
                    (pre[0].date(), pre[-1].date()), _stitched(wf_table, R), tags={"stage": "walk_forward_oos"})
    picks = wf_table["picked"].value_counts()
    overlay_picks = picks.drop("none", errors="ignore")
    chosen = overlay_picks.index[0] if len(overlay_picks) else "K2"
    adopt_wf = picks.get(chosen, 0) > picks.get("none", 0)
    print("\nWalk-forward over the six scales and no overlay, pre-window sessions:\n")
    print(wf_table.round(3).to_string())
    print(f"\nselected-in-sample OOS Sharpe {wf['selected_oos_sharpe']:.3f} vs bare book {wf['v1_oos_sharpe']:.3f}; "
          f"most-picked overlay {chosen}, picked more often than none: {adopt_wf}")

    k_chosen = int(chosen[1])
    cfg = book_config(kill=False, allocations=ALLOC)
    ov = run([regimes.RegimeOverlay(s, cfg.risk, scales[chosen]) for s in sleeves(BOOK)], bars, config=cfg)
    print(f"overlay {chosen} through the engine: Sharpe {ov.metrics['sharpe']:.3f}, {len(ov.trades)} trades, "
          f"{time.time() - t0:.0f}s")
    registry.record(f"{BOOK} book", {"overlay": "none", "path": "engine"}, instruments,
                    (pre[0].date(), pre[-1].date()), base.returns.loc[pre], tags={"stage": "engine_pre"})
    registry.record(f"{BOOK} x overlay {chosen}", {"overlay": chosen, "path": "engine"}, instruments,
                    (pre[0].date(), pre[-1].date()), ov.returns.loc[pre], tags={"stage": "engine_pre"})
    extra = len(validate.trials())
    pre_dsr = {"book": registry.dsr(base.returns.loc[pre], extra), "overlay": registry.dsr(ov.returns.loc[pre], extra)}
    cut = pd.DataFrame({f"{k}_{s}": regimes.reduced_days(res.equity_by_strategy[s])
                        for k, res in (("book", base), ("overlay", ov)) for s in ALLOC})

    def final(start, stop):
        b, o = base.returns.loc[start:stop], ov.returns.loc[start:stop]
        gross = lambda res: float(res.weights.abs().sum(axis=1).loc[start:stop].mean())
        out = {"chosen": chosen, "book": stats(b), "overlay": stats(o),
               "book_dsr": registry.dsr(b, extra), "overlay_dsr": registry.dsr(o, extra),
               "book_dsr_own_trials": registry.dsr(b), "overlay_dsr_own_trials": registry.dsr(o),
               "book_trials_added": extra, "avg_gross": {"book": gross(base), "overlay": gross(ov)},
               "trades": {"book": int((base.trades["date"] >= start).sum()),
                          "overlay": int((ov.trades["date"] >= start).sum())},
               "adopt": bool(adopt_wf and sharpe(o[o != 0]) > sharpe(b[b != 0]))}
        for name, r in ((f"{BOOK} book", b), (f"{BOOK} x overlay {chosen}", o)):
            registry.record(name, {"window": "untouched", "path": "engine"}, instruments,
                            (start.date(), stop.date()), r, tags={"stage": "untouched"})
        return out

    try:
        result = lock.open(final)
        print(f"\nuntouched window opened now; result stored in {lock.path}")
    except UntouchedWindowUsed as e:
        result = lock.result
        print(f"\n{e}; reporting the stored result")
    if result["chosen"] != chosen:
        print(f"WARNING: the window was opened with {result['chosen']}, this run chose {chosen}; "
              "the stored result stands")

    last = models[k_chosen][max(models[k_chosen])]
    desc = regimes.describe(last)
    path = probs[k_chosen].dropna().to_numpy().argmax(axis=1)
    idx = probs[k_chosen].dropna().index
    emp_pre = regimes.run_lengths(path[idx < hold])
    emp_all = regimes.run_lengths(path)
    desc["empirical_duration_pre"] = emp_pre["empirical_duration"].to_numpy()
    desc["share_pre"] = emp_pre["share"].to_numpy()
    desc["empirical_duration_all"] = emp_all["empirical_duration"].to_numpy()
    desc["share_all"] = emp_all["share"].to_numpy()
    trans_last = pd.DataFrame(last.transmat_, index=desc.index, columns=desc.index)
    trans_mean = pd.DataFrame(np.mean([m.transmat_ for m in models[k_chosen].values()], axis=0),
                              index=desc.index, columns=desc.index)
    desc.to_csv(os.path.join(REPORTS, "states.csv"))
    pd.concat({"last_refit": trans_last, "mean_over_refits": trans_mean}).to_csv(os.path.join(REPORTS, "transitions.csv"))
    print(f"\nStates, K={k_chosen}, last refit {max(models[k_chosen])}:\n")
    print(desc.round(3).to_string())
    print(f"\nTransition matrix, last refit:\n{trans_last.round(4).to_string()}")
    print(f"\nTransition matrix, mean over refits:\n{trans_mean.round(4).to_string()}")

    train = obs.loc[: f"{max(models[k_chosen]) - 1}-12-31"]  # what `last` was actually fit on
    model_id = ModelRegistry(os.path.join(REPORTS, "models")).save(
        f"GaussianHMM[K={k_chosen}]", last, train,
        (train.index[0].date(), train.index[-1].date()), result["overlay"].get("sharpe"),
        meta={"states": k_chosen, "chosen": chosen, "refit_years": sorted(models[k_chosen]),
              "obs": list(obs.columns), "nowcast": FRED})

    years = pd.DataFrame({"book": base.by_year()["return"], "overlay": ov.by_year()["return"]})
    years["avg_scale"] = scales[chosen].loc[cal].groupby(cal.year).mean()
    years.to_csv(os.path.join(REPORTS, "by_year.csv"))
    daily = pd.DataFrame({f"p_high_K{k}": probs[k][k - 1] for k in STATES})
    daily["nowcast"] = nc
    for v, s in scales.items():
        daily[f"scale_{v}"] = s
    daily.to_csv(os.path.join(REPORTS, "probabilities.csv"))
    pd.DataFrame({"book": base.returns, "overlay": ov.returns, "scale": scales[chosen],
                  **{f"gross_{k}": res.weights.abs().sum(axis=1) for k, res in (("book", base), ("overlay", ov))},
                  **{f"{k}_equity_{s}": res.equity_by_strategy[s] for k, res in (("book", base), ("overlay", ov))
                     for s in ALLOC}}).to_csv(os.path.join(REPORTS, "returns.csv"))
    years["half_size_book"] = cut[[f"book_{s}" for s in ALLOC]].mean(axis=1).groupby(cal.year).mean()
    years["half_size_overlay"] = cut[[f"overlay_{s}" for s in ALLOC]].mean(axis=1).groupby(cal.year).mean()
    years.to_csv(os.path.join(REPORTS, "by_year.csv"))

    full_b, full_o = stats(base.returns), stats(ov.returns)
    pre_b, pre_o = stats(base.returns.loc[pre]), stats(ov.returns.loc[pre])
    summary = {
        "book": BOOK, "sessions": len(cal), "start": str(cal[0].date()), "end": str(cal[-1].date()),
        "panel_hash": data_hash, "spy_obs": len(obs), "fred": FRED, "states_grid": STATES,
        "first_refit_year": FIRST_YEAR, "refit_years": sorted(models[2]),
        "walk_forward": {"table": wf_table.reset_index().to_dict("records"), **wf, "picks": picks.to_dict()},
        "chosen": chosen, "adopt_walk_forward": bool(adopt_wf),
        "pre_window": {"book": pre_b, "overlay": pre_o, "overlay_dsr": pre_dsr["overlay"], "book_dsr": pre_dsr["book"],
                       "half_size_days": cut.loc[pre].mean().to_dict()},
        "untouched": {"start": str(hold.date()), "end": str(end.date()), **result,
                      "opened_at": lock.read()["opened_at"], "half_size_days": cut.loc[hold:end].mean().to_dict()},
        "full_panel": {"book": full_b, "overlay": full_o,
                       "trades": {"book": int(len(base.trades)), "overlay": int(len(ov.trades))},
                       "turnover_per_year": {"book": float(base.turnover.sum() / (len(cal) / 252)),
                                             "overlay": float(ov.turnover.sum() / (len(cal) / 252))}},
        "states": desc.to_dict("index"), "transition_last": trans_last.to_dict("index"),
        "transition_mean": trans_mean.to_dict("index"),
        "avg_scale": {v: float(s.loc[pre].mean()) for v, s in scales.items()},
        "n_trials": registry.trials(), "book_trials": extra, "model_id": model_id,
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ph = probs[k_chosen][k_chosen - 1].reindex(cal)
    axes[0].plot(spy.reindex(cal), color="black", linewidth=0.9)
    axes[0].fill_between(cal, spy.reindex(cal).min(), spy.reindex(cal).max(), where=ph > 0.5, color="firebrick",
                         alpha=0.25, label=f"P(high-vol state) > 0.5, K={k_chosen}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SPY (log)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Filtered HMM regime on SPY, refit each January, and the overlay's scale")
    axes[1].plot(nc.reindex(cal), linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("macro nowcast (z)")
    axes[2].plot(scales[chosen].loc[cal], linewidth=0.8, label=chosen)
    axes[2].plot(scales["K2"].loc[cal], linewidth=0.6, alpha=0.6, label="K2")
    axes[2].set_ylabel("gross scale")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvspan(hold, end, color="grey", alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "regimes.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(base.equity / base.equity.iloc[0], color="black", linewidth=1.2, label=f"{BOOK} book")
    axes[0].plot(ov.equity / ov.equity.iloc[0], color="firebrick", linewidth=1.2, label=f"with overlay {chosen}")
    axes[0].axvspan(hold, end, color="grey", alpha=0.15, label="untouched window")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("growth of 1 (log)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Live book with and without the regime overlay, engine cost model and overlay")
    axes[1].plot(base.weights.abs().sum(axis=1), color="black", linewidth=0.7)
    axes[1].plot(ov.weights.abs().sum(axis=1), color="firebrick", linewidth=0.7)
    axes[1].set_ylabel("gross")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "overlay.png"), dpi=140)
    plt.close(fig)

    u = result
    print(f"\nPre-window: book Sharpe {pre_b['sharpe']:.3f}, overlay {pre_o['sharpe']:.3f} "
          f"(DSR {summary['pre_window']['overlay_dsr']['dsr']:.3f} over {summary['pre_window']['overlay_dsr']['n_trials']} trials)")
    print(f"Untouched {hold.date()} to {end.date()}: book Sharpe {u['book']['sharpe']:.3f} (DSR {u['book_dsr']['dsr']:.3f}), "
          f"overlay {u['overlay']['sharpe']:.3f} (DSR {u['overlay_dsr']['dsr']:.3f}) over {u['overlay_dsr']['n_trials']} trials; "
          f"max drawdown {u['book']['max_drawdown']:.1%} vs {u['overlay']['max_drawdown']:.1%}; adopt: {u['adopt']}")
    print("Share of window days each sleeve ran at half size under the 15% drawdown cut: "
          + ", ".join(f"{k} {v:.1%}" for k, v in summary["untouched"]["half_size_days"].items()))
    print(f"Full panel: book {full_b['sharpe']:.3f}, overlay {full_o['sharpe']:.3f}; trades "
          f"{summary['full_panel']['trades']['book']} vs {summary['full_panel']['trades']['overlay']}")
    print(f"Model {model_id}; {summary['n_trials']} trials here + {extra} from the book; {summary['seconds']}s")


def _stitched(table, R):
    """Stitch each fold's picked column over its test window, the walk-forward's OOS stream."""
    parts = [R.loc[str(row.test_start):str(row.test_end), row.picked] for row in table.itertuples()]
    return pd.concat(parts)


if __name__ == "__main__":
    main()
