"""LightGBM + ridge vs 12-1 momentum on the point-in-time S&P 500, through the research platform.

Panel from the data lake, features from the store with the peek audit on,
four variants (momentum, ridge, LightGBM, ensemble) through the research
walk-forward, every fit logged as a trial, the untouched last fifth opened
once and stored in the lock, the chosen model saved to the model registry.
The lake rows are pinned to reports/snapshot/ on the first run and read from
there after, so reruns reproduce every number whatever the lake's collectors
have rewritten since; the window's result is read back from the lock.

Run: ../.venv/bin/python3 run.py   (about a minute)
"""
import json
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xs import (CAPITAL, LGBM, MEMBER_FROM, RIDGE_ALPHA, ROOT, XSModel, decile_spread, features, framework_net, ic_summary,
                importance_stability, load_raw, rank_ic, scores)
sys.path.insert(0, os.path.join(ROOT, "data-lake"))
import lake
from framework.book import validate
from framework.engine import sharpe
from research.alpha import Untouched, UntouchedWindowUsed, forward_returns, positions, walk_forward
from research.alpha.base import month_ends, window_returns
from research.features import FeatureStore
from research.features.store import long_panel, panel_hash
from research.models import ModelRegistry
from research.registry import Registry

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
SNAPSHOT = os.path.join(REPORTS, "snapshot")
START, END = "2003-06-01", "2026-08-31"
# Frozen before any result. The ensemble is the baseline; the walk-forward's
# most-picked variant is what the window is opened with, ties to the baseline.
BASELINE = {"model": "ensemble"}
GRID = [{"model": "mom"}, {"model": "ridge"}, {"model": "lgbm"}, {"model": "ensemble"}]
FOLDS, TEST_SIZE, HORIZON = 5, 504, 21
HOLDOUT = validate.HOLDOUT
BIG_CAPITAL = 10_000_000.0


def stats(r):
    r = r[r != 0].dropna()
    return validate.stats(r) if len(r) > 1 else {}


def fitted(made, params, train_end):
    return next(a for a in made if a.params == params and (a.model == "mom" or a.train_end == train_end))


def evaluate(alpha, panel, close, fwd, dollar_vol, sigma, start, stop, dates):
    """Research, gross and framework-net returns plus IC and decile spread on one window."""
    r = window_returns(alpha, panel, close, start, stop, dates)
    g = window_returns(alpha, panel, close, start, stop, dates, cost_bps=0.0)
    keep = dates[(dates >= start) & (dates <= stop)]
    before = dates[dates < start][-1:]
    pos = positions(alpha, panel, before.append(keep))
    first = before[0] if len(before) else start
    net = framework_net(pos, close.loc[first:stop], dollar_vol, sigma).loc[start:]
    big = framework_net(pos, close.loc[first:stop], dollar_vol, sigma, capital=BIG_CAPITAL).loc[start:]
    sc = scores(alpha, panel, keep)
    return {"research": r, "gross": g, "net": net, "net_10m": big,
            "ic": rank_ic(sc, fwd), "spread": decile_spread(sc, fwd)}


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    raw = load_raw(lake, START, END, snapshot=SNAPSHOT)
    spells = os.path.join(SNAPSHOT, "membership_intervals.csv")
    if not os.path.exists(spells):
        shutil.copy(lake.INTERVALS, spells)
    intervals = lake.sp500_intervals(spells)
    store = FeatureStore(features(intervals, MEMBER_FROM))
    panel = store.build(raw, audit=True)
    close = raw.frames["close"]
    cal = panel.index
    tickers = list(close.columns)
    fwd = forward_returns(close, HORIZON)
    dollar_vol, sigma = panel["dollar_vol_21"], panel["vol_21"] / np.sqrt(252)
    registry = Registry(os.path.join(REPORTS, "registry"))
    models = ModelRegistry(os.path.join(REPORTS, "models"))
    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    data_hash = panel_hash(close)
    lock.lock(cal, HOLDOUT, data_hash)
    hold, end = lock.window
    pre = cal[cal < hold]
    dates = month_ends(cal)
    members = ((panel["member"] == 1.0) & panel["ret_21"].notna()).sum(axis=1)[dates[dates >= MEMBER_FROM]]
    print(f"{len(tickers)} tickers, {cal[0].date()} to {cal[-1].date()}, {len(cal)} sessions, close hash {data_hash}; "
          f"members with a price at month end from {MEMBER_FROM.date()}: median {int(members.median())}, "
          f"min {int(members.min())}")
    print(f"untouched window {hold.date()} to {end.date()}, {len(cal) - len(pre)} sessions"
          + (f", opened at {lock.read()['opened_at']}" if lock.opened else ", not yet opened"))

    made = []

    def make(**params):
        a = XSModel(dates=dates, **params)
        made.append(a)
        return a

    wf_table, wf_oos, full = walk_forward(make, GRID, panel.loc[pre], close.loc[pre], FOLDS, TEST_SIZE,
                                          horizon=HORIZON, registry=registry, universe=tickers)
    registry.record("XS walk-forward", {"grid": GRID, "folds": FOLDS, "test_size": TEST_SIZE}, tickers,
                    (pre[0].date(), pre[-1].date()), wf_oos, tags={"stage": "walk_forward_oos"})
    picks = wf_table["picked"].value_counts()
    chosen = json.loads(picks.index[0]) if picks.iloc[0] > picks.get(json.dumps(BASELINE), 0) else BASELINE
    print("\nWalk-forward over the grid, pre-untouched sessions:\n")
    print(wf_table.round(3).to_string())
    print(f"\nselected-in-sample OOS Sharpe {sharpe(wf_oos[wf_oos != 0]):.3f}; chosen for the final window: {chosen}")

    # every variant on every fold's test window, each fit on that fold's training rows only
    oos = {json.dumps(p): [] for p in GRID}
    imp_lgbm, imp_ridge = [], []
    for k, row in wf_table.iterrows():
        tr_end, te0, te1 = pd.Timestamp(row["train_end"]), pd.Timestamp(row["test_start"]), pd.Timestamp(row["test_end"])
        for p in GRID:
            a = fitted(made, p, tr_end)
            oos[json.dumps(p)].append(evaluate(a, panel, close, fwd, dollar_vol, sigma, te0, te1, dates))
            if p["model"] == "ensemble":
                imp_lgbm.append(a.importance["lgbm"].rename(f"fold {k}"))
                imp_ridge.append(a.importance["ridge"].rename(f"fold {k}"))
    stitched = {k: {m: pd.concat([e[m] for e in v]) for m in v[0]} for k, v in oos.items()}
    imp_lgbm, imp_ridge = pd.DataFrame(imp_lgbm), pd.DataFrame(imp_ridge)
    stability = {"lgbm_gain_rank_corr": importance_stability(imp_lgbm),
                 "ridge_coef_rank_corr": importance_stability(imp_ridge),
                 "ridge_sign_agreement": float((np.sign(imp_ridge) == np.sign(imp_ridge.iloc[0])).mean().mean())}
    imp_lgbm.T.to_csv(os.path.join(REPORTS, "importance_lgbm.csv"))
    imp_ridge.T.to_csv(os.path.join(REPORTS, "importance_ridge.csv"))

    def table(res):
        return {"sharpe_research": sharpe(res["research"][res["research"] != 0]),
                "sharpe_gross": sharpe(res["gross"][res["gross"] != 0]),
                "sharpe_net": sharpe(res["net"][res["net"] != 0]),
                "sharpe_net_10m": sharpe(res["net_10m"][res["net_10m"] != 0]),
                "ic": ic_summary(res["ic"]), "spread_monthly": float(res["spread"].mean()),
                "spread_t": float(res["spread"].mean() / res["spread"].std() * np.sqrt(res["spread"].count())),
                "annual_return_net": stats(res["net"]).get("annual_return"),
                "max_drawdown_net": stats(res["net"]).get("max_drawdown")}

    wf_by_variant = {json.loads(k)["model"]: table(v) for k, v in stitched.items()}
    print("\nWalk-forward OOS by variant (stitched test windows, each fold fit on its own training rows):\n")
    print(pd.DataFrame({m: {"research": t["sharpe_research"], "gross": t["sharpe_gross"], "net": t["sharpe_net"],
                            "net $10m": t["sharpe_net_10m"], "IC": t["ic"]["mean"], "IC t": t["ic"]["t"],
                            "spread %/mo": 100 * t["spread_monthly"]} for m, t in wf_by_variant.items()}).T.round(3).to_string())
    print(f"\nimportance stability across folds: LightGBM gain rank corr {stability['lgbm_gain_rank_corr']:.2f}, "
          f"ridge coef rank corr {stability['ridge_coef_rank_corr']:.2f}, sign agreement {stability['ridge_sign_agreement']:.2f}")

    # Post-hoc ablation, walk-forward folds only, never the window: LightGBM's top gains were the three
    # macro series, which are constant within a date and so can only act as date identifiers.
    ABLATION = {"model": "lgbm", "macro": False}
    X, y = long_panel(panel.loc[pre]), long_panel(pd.concat({"y": fwd.loc[pre]}, axis=1,
                                                             names=["feature", "instrument"]))["y"]
    abl = []
    for k, row in wf_table.iterrows():
        train = pre[pre <= pd.Timestamp(row["train_end"])]
        a = make(**ABLATION).fit(X.loc[train], y.loc[train])
        abl.append(evaluate(a, panel, close, fwd, dollar_vol, sigma, pd.Timestamp(row["test_start"]),
                            pd.Timestamp(row["test_end"]), dates))
    abl = {m: pd.concat([e[m] for e in abl]) for m in abl[0]}
    registry.record("XS[lgbm, no macro] walk-forward", {**ABLATION, "folds": FOLDS, "test_size": TEST_SIZE}, tickers,
                    (pre[0].date(), pre[-1].date()), abl["research"], tags={"stage": "ablation_post_hoc"})
    wf_by_variant["lgbm_no_macro"] = table(abl)
    stitched[json.dumps(ABLATION)] = abl
    t = wf_by_variant["lgbm_no_macro"]
    print(f"post-hoc ablation, LightGBM without macro, same folds: research {t['sharpe_research']:.3f}, "
          f"gross {t['sharpe_gross']:.3f}, net {t['sharpe_net']:.3f}, IC {t['ic']['mean']:.3f} (t {t['ic']['t']:.2f}), "
          f"spread {100 * t['spread_monthly']:.2f}%/mo")

    final_alpha = fitted(made, chosen, pre[-1])
    mom = fitted(made, {"model": "mom"}, pre[-1])

    def final(start, stop):
        out = {"chosen": chosen}
        res = {"model": evaluate(final_alpha, panel, close, fwd, dollar_vol, sigma, start, stop, dates),
               "mom": evaluate(mom, panel, close, fwd, dollar_vol, sigma, start, stop, dates)}
        for name, e in res.items():
            out[name] = table(e)
            out[name]["research"] = stats(e["research"])
            out[name]["net"] = stats(e["net"])
        out["dsr"] = registry.dsr(res["model"]["research"])
        out["dsr_net"] = registry.dsr(res["model"]["net"])
        registry.record(str(final_alpha), {**chosen, "window": "untouched"}, tickers, (start.date(), stop.date()),
                        res["model"]["research"], tags={"stage": "untouched"})
        registry.record(str(mom), {"model": "mom", "window": "untouched"}, tickers, (start.date(), stop.date()),
                        res["mom"]["research"], tags={"stage": "untouched"})
        pd.DataFrame({f"{n}_{m}": e[m] for n, e in res.items() for m in ("research", "gross", "net")}).to_csv(
            os.path.join(REPORTS, "untouched_returns.csv"))
        return out

    try:
        result = lock.open(final)
        print(f"\nuntouched window opened now; result stored in {lock.path}")
    except UntouchedWindowUsed as e:
        result = lock.result
        print(f"\n{e}; reporting the stored result")
    if result["chosen"] != chosen:
        print(f"WARNING: the window was opened with {result['chosen']}, this run chose {chosen}; "
              "the stored result stands and this run's choice is in-sample")

    model_id = models.save(str(final_alpha), final_alpha,
                           close.loc[pre], (pre[0].date(), pre[-1].date()), result["model"]["sharpe_research"],
                           meta={"chosen": chosen, "grid": GRID, "picks": picks.to_dict(), "features": store.lags,
                                 "lgbm": LGBM, "ridge_alpha": RIDGE_ALPHA, "member_from": str(MEMBER_FROM.date()),
                                 "train_dates": [str(d.date()) for d in final_alpha.train_dates[[0, -1]]],
                                 "n_train_cross_sections": int(len(final_alpha.train_dates))})

    pre_stats = {json.loads(k)["model"]: stats(v) for k, v in full.items()}
    d_pre = registry.dsr(full[json.dumps(chosen)])
    summary = {
        "n_tickers": len(tickers), "sessions": len(cal), "start": str(cal[0].date()), "end": str(cal[-1].date()),
        "close_hash": data_hash, "snapshot": SNAPSHOT, "member_from": str(MEMBER_FROM.date()),
        "features": store.lags, "grid": GRID, "chosen": chosen,
        "members_at_month_end": {"median": int(members.median()), "min": int(members.min()),
                                 "max": int(members.max())},
        "walk_forward": {"table": wf_table.reset_index().to_dict("records"),
                         "selected_oos_sharpe": sharpe(wf_oos[wf_oos != 0]), "by_variant": wf_by_variant},
        "importance_stability": stability,
        "importance_lgbm_mean": imp_lgbm.mean().sort_values(ascending=False).to_dict(),
        "importance_ridge_mean": imp_ridge.mean().to_dict(),
        "in_sample_pre_window": pre_stats, "pre_window_dsr": d_pre,
        "untouched": {"start": str(hold.date()), "end": str(end.date()), **result, "opened_at": lock.read()["opened_at"]},
        "n_trials": registry.trials(), "n_runs_logged": len(registry.runs()), "model_id": model_id,
        "capital": {"book": CAPITAL, "big": BIG_CAPITAL},
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    def label(k):
        p = json.loads(k)
        return p["model"] + ("" if p.get("macro", True) else "_no_macro")

    pd.DataFrame({f"{label(k)}_{m}": v[m] for k, v in stitched.items()
                  for m in ("research", "gross", "net")}).to_csv(os.path.join(REPORTS, "walk_forward_returns.csv"))
    pd.DataFrame({f"{label(k)}_ic": v["ic"] for k, v in stitched.items()}).to_csv(
        os.path.join(REPORTS, "walk_forward_ic.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    unt = os.path.join(REPORTS, "untouched_returns.csv")
    u = pd.read_csv(unt, index_col=0, parse_dates=True) if os.path.exists(unt) else pd.DataFrame()
    for k, v in stitched.items():
        m = label(k)
        r = v["net"]
        col = f"{'model' if m == chosen['model'] else m}_net"
        if col in u:
            r = pd.concat([r, u[col]])
        axes[0].plot((1 + r).cumprod(), linewidth=1.3 if m in (chosen["model"], "mom") else 0.8,
                     label=f"{m}, framework net")
    axes[0].axvspan(hold, end, color="grey", alpha=0.15, label="untouched window")
    axes[0].set_ylabel("growth of 1")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Walk-forward OOS, long top decile / short bottom decile, PIT S&P 500, book cost model")
    for k, v in stitched.items():
        m = label(k)
        if m in (chosen["model"], "mom"):
            axes[1].plot(v["ic"].rolling(12).mean(), label=f"{m} rank IC, 12-month mean")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("rank IC")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = imp_lgbm.mean().sort_values(ascending=False).index
    imp_lgbm[order].T.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("share of LightGBM gain")
    ax.set_title("LightGBM feature importance by walk-forward fold (ensemble variant)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "importance.png"), dpi=140)
    plt.close(fig)

    c, m_ = result["model"], result["mom"]
    print(f"\nPre-window in-sample, chosen: Sharpe {pre_stats[chosen['model']]['sharpe']:.3f}, "
          f"PSR {d_pre['psr']:.3f}, DSR {d_pre['dsr']:.3f} over {d_pre['n_trials']} trials")
    print(f"Untouched window {hold.date()} to {end.date()}, {chosen['model']}: research {c['sharpe_research']:.3f}, "
          f"gross {c['sharpe_gross']:.3f}, net {c['sharpe_net']:.3f} (DSR {result['dsr']['dsr']:.3f} over "
          f"{result['dsr']['n_trials']} trials), IC {c['ic']['mean']:.3f} (t {c['ic']['t']:.2f}), "
          f"spread {100 * c['spread_monthly']:.2f}%/mo")
    print(f"Untouched window, 12-1 momentum: research {m_['sharpe_research']:.3f}, gross {m_['sharpe_gross']:.3f}, "
          f"net {m_['sharpe_net']:.3f}, IC {m_['ic']['mean']:.3f} (t {m_['ic']['t']:.2f}), "
          f"spread {100 * m_['spread_monthly']:.2f}%/mo")
    print(f"Model {model_id} saved; {summary['n_trials']} trials in the registry; {summary['seconds']}s")


if __name__ == "__main__":
    main()
