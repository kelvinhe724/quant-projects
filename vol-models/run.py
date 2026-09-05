"""Realised-vol horse race on SPY and 20 names, then the straddle timing test through the research harness.

Data through the lake, features through the store with the peek audit,
forecasts from rolling refits, the timing rule's model and threshold picked
by the harness's walk-forward on the pre-window sessions, the untouched
last fifth opened once for both the forecast scores and the trading
result, every variant logged to the registry. Forecasts are cached to
reports/forecasts.parquet; delete it to recompute (about 12 minutes).

Run: ../.venv/bin/python3 run.py
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
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data-lake"))
import lake
import vol
from framework.book import validate
from framework.engine import sharpe
from research.alpha import Untouched, UntouchedWindowUsed, backtest, positions, walk_forward
from research.alpha.base import month_ends, window_returns
from research.features import Feature, FeatureStore, Raw
from research.features.store import long_panel, panel_hash
from research.models import ModelRegistry
from research.registry import Registry

REPORTS = os.path.join(HERE, "reports")
NAMES = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ",
         "PFE", "MRK", "UNH", "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS"]
UNIVERSE = ["SPY"] + NAMES
START, END = "2003-06-02", "2026-09-03"
FRED = {"vix": "VIXCLS", "t10y2y": "T10Y2Y", "dtb3": "DTB3"}
HOLDOUT = validate.HOLDOUT
FOLDS, TEST_SIZE = 4, 504
# Frozen before any result: always-on is the baseline, the grid is four forecasters
# by three threshold quantiles, the window is the last fifth of sessions.
BASELINE = {"model": "none", "q": 0.0}
Q = (0.25, 0.5, 0.75)
GRID = [BASELINE] + [{"model": m, "q": q} for m in ("har", "garch", "gjr", "lgb") for q in Q]
TIMED = ("har", "garch", "gjr", "lgb")


def load():
    eq = lake.load("equities_daily", START, END, universe=UNIVERSE)
    piv = lambda col: eq.pivot(index="date", columns="ticker", values=col).sort_index().reindex(columns=UNIVERSE)
    frames = {"open": piv("open"), "high": piv("high"), "low": piv("low"), "close": piv("adj_close"),
              "volume": piv("volume")}
    frames = {k: v.loc[frames["close"]["SPY"].notna()] for k, v in frames.items()}
    fred = lake.load("fred", "2000-01-01", END, universe=list(FRED.values()))
    series = {}
    for k, sid in FRED.items():
        s = fred[fred["series"] == sid]
        series[k] = pd.Series(s["value"].to_numpy(float), index=pd.DatetimeIndex(s["date"])).dropna()
    return Raw(frames, series)


def cached_forecasts(panel, ret):
    path = os.path.join(REPORTS, "forecasts.parquet")
    if os.path.exists(path):
        wide = pd.read_parquet(path)
        wide.columns = pd.MultiIndex.from_tuples([(int(h), m, t) for h, m, t in wide.columns])
        out = {h: {m: wide[h][m] for m in vol.MODELS} for h in vol.HORIZONS}
        return out, {h: vol.forward_realised(ret, h) for h in vol.HORIZONS}
    t0 = time.time()
    print("fitting the forecasters (rolling refits every %d sessions), this takes a while" % vol.REFIT)
    out, targets, _ = vol.forecast_all(panel, ret)
    wide = pd.concat({(str(h), m): out[h][m] for h in out for m in out[h]}, axis=1)
    wide.to_parquet(path)
    print(f"forecasts done in {time.time() - t0:.0f}s")
    return out, targets


def race(forecasts, targets, idx, cal_pre=None):
    """Scores and DM tests over the sessions in idx."""
    scores, pooled, by_name = {}, {}, {}
    for h in vol.HORIZONS:
        rv = targets[h].loc[idx]
        fs = {m: f.loc[idx] for m, f in forecasts[h].items()}
        k = str(h)
        scores[k] = {m: vol.score(fs[m], rv) for m in fs}
        pooled[k] = {"qlike": vol.dm_table(fs, rv, h, "qlike"), "mse": vol.dm_table(fs, rv, h, "mse")}
        by_name[k] = vol.dm_by_instrument(fs, rv, h, "naive", "qlike")
    return scores, pooled, by_name


def spy_table(forecasts, targets, vix, idx):
    """SPY only, 21 days, with the VIX as a fifth forecast."""
    rv = targets[21].loc[idx, ["SPY"]]
    fs = {m: f.loc[idx, ["SPY"]] for m, f in forecasts[21].items()}
    fs["vix"] = vix.loc[idx].to_frame("SPY")
    rows = {m: vol.score(fs[m], rv) for m in fs}
    return pd.DataFrame(rows).T, vol.dm_table(fs, rv, 21, "qlike")


def stats(r):
    """Harness stats (zero days dropped) plus the full-calendar Sharpe and time in market."""
    active = r[r != 0].dropna()
    s = validate.stats(active) if len(active) > 1 else {}
    s["sharpe_full"] = sharpe(r.dropna())
    s["annual_return_full"] = float(r.mean() * 252)
    s["time_in_market"] = float((r != 0).mean())
    s["max_drawdown_full"] = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
    return s


def flat(pooled_tab):
    return {f"{a} vs {b}": (None if pd.isna(v) else round(float(v), 2))
            for a, row in pooled_tab.iterrows() for b, v in row.items() if a != b}


def clean(obj):
    """Plain JSON: string keys, Python scalars, so the lock's hash survives a round trip through the file."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    raw = load()
    store = FeatureStore(vol.features())
    panel = store.build(raw, audit=True)
    close = raw.frames["close"]
    ret = np.log(close).diff()
    cal = panel.index
    data_hash = panel_hash(close)
    registry = Registry(os.path.join(REPORTS, "registry"))
    models = ModelRegistry(os.path.join(REPORTS, "models"))
    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    lock.lock(cal, HOLDOUT, data_hash)
    hold, end = lock.window
    pre = cal[cal < hold]
    print(f"{len(UNIVERSE)} names, {cal[0].date()} to {cal[-1].date()}, {len(cal)} sessions, close hash {data_hash}")
    print(f"untouched window {hold.date()} to {end.date()}, {len(cal) - len(pre)} sessions"
          + (f", opened at {lock.read()['opened_at']}" if lock.opened else ", not yet opened"))

    forecasts, targets = cached_forecasts(panel, ret)
    vix = panel["vix"]["SPY"]
    have = pre[pre >= cal[vol.WARMUP]]
    pre_scores, pre_pooled, pre_by_name = race(forecasts, targets, have)
    spy_pre, spy_pre_dm = spy_table(forecasts, targets, vix, have)
    print(f"\nHorse race, pre-window sessions {have[0].date()} to {have[-1].date()}, 21 names pooled:")
    for h in map(str, vol.HORIZONS):
        print(f"\n{h}-day realised vol")
        print(pd.DataFrame(pre_scores[h]).T.round(4).to_string())
        print(f"pairwise DM t (QLIKE, row beats column when negative):")
        print(pre_pooled[h]["qlike"].round(2).to_string())
        beats = (pre_by_name[h] < -1.96).sum()
        print(f"names where the model beats naive at 5% (of {len(pre_by_name[h])}): {beats.to_dict()}")
    print("\nSPY alone, 21 days, with the VIX:")
    print(spy_pre.round(4).to_string())
    print(spy_pre_dm.round(2).to_string())

    pnl = vol.straddle_cycles(close["SPY"], vix * 100, month_ends(cal))
    index = (1 + pnl).cumprod().to_frame("STRADDLE")
    frames2 = {"close": index, "vix": vix.to_frame("STRADDLE")}
    for m in TIMED:
        frames2[f"f_{m}"] = forecasts[21][m]["SPY"].to_frame("STRADDLE")
    raw2 = Raw(frames2)
    store2 = FeatureStore([Feature(k, lambda r, k=k: r.frames[k], 0) for k in frames2 if k != "close"])
    panel2 = store2.build(raw2, audit=True)
    close2 = index

    wf_table, wf_oos, full = walk_forward(lambda **p: vol.Timer(**p), GRID, panel2.loc[pre], close2.loc[pre],
                                          FOLDS, TEST_SIZE, registry=registry, universe=["STRADDLE"])
    registry.record("Timer walk-forward", {"grid": GRID, "folds": FOLDS, "test_size": TEST_SIZE}, ["STRADDLE"],
                    (pre[0].date(), pre[-1].date()), wf_oos, tags={"stage": "walk_forward_oos"})
    picks = wf_table["picked"].value_counts()
    chosen = json.loads(picks.index[0]) if picks.iloc[0] > picks.get(json.dumps(BASELINE), 0) else BASELINE
    print("\nWalk-forward over the timing grid, pre-untouched sessions:\n")
    print(wf_table.round(3).to_string())
    print(f"\nselected-in-sample OOS Sharpe {sharpe(wf_oos[wf_oos != 0]):.3f} (active days), "
          f"{sharpe(wf_oos):.3f} (all days); chosen for the window: {chosen}")
    in_sample = {k: stats(v) for k, v in full.items()}
    print("\nEvery variant on the pre-window sessions (10 bps on switches, straddle costs inside the P&L):")
    print(pd.DataFrame(in_sample).T[["sharpe", "sharpe_full", "annual_return_full", "max_drawdown_full",
                                    "time_in_market"]].round(3).to_string())

    best_h21 = min(TIMED, key=lambda m: pre_scores["21"][m]["qlike"])
    X_pre = long_panel(panel2.loc[pre])
    fitted = {json.dumps(p): vol.Timer(**p).fit(X_pre, None) for p in GRID}
    reads = {"chosen": chosen, "baseline": BASELINE,
             "best_forecast": {"model": best_h21, "q": chosen["q"] if chosen["model"] != "none" else 0.5}}

    def final(start, stop):
        w = cal[(cal >= start) & (cal <= stop)]
        sc, pooled, by_name = race(forecasts, targets, w)
        spy_w, spy_w_dm = spy_table(forecasts, targets, vix, w)
        out = {"reads": reads, "scores": sc, "dm_pooled": {h: {k: flat(v) for k, v in pooled[h].items()} for h in pooled},
               "beats_naive_5pct": {h: {k: int(v) for k, v in (by_name[h] < -1.96).sum().items()} for h in by_name},
               "spy": spy_w.to_dict("index"), "spy_dm": flat(spy_w_dm), "trading": {}}
        for label, p in reads.items():
            a = fitted.get(json.dumps(p)) or vol.Timer(**p).fit(X_pre, None)
            r = window_returns(a, panel2, close2, start, stop, month_ends(cal))
            out["trading"][label] = {"config": p, "threshold": a.thr, **stats(r), "dsr": registry.dsr(r)}
            registry.record(str(a), {**p, "window": "untouched"}, ["STRADDLE"], (start.date(), stop.date()), r,
                            tags={"stage": "untouched", "read": label})
        return clean(out)

    try:
        result = lock.open(final)
        print(f"\nuntouched window opened now; result stored in {lock.path}")
    except UntouchedWindowUsed as e:
        result = lock.result
        print(f"\n{e}; reporting the stored result")
    if result["reads"]["chosen"] != chosen:
        print(f"WARNING: the window was opened with {result['reads']['chosen']}, this run chose {chosen}; "
              "the stored result stands and this run's choice is in-sample")

    print(f"\nHorse race on the untouched window {hold.date()} to {end.date()}:")
    for h in map(str, vol.HORIZONS):
        print(f"\n{h}-day realised vol")
        print(pd.DataFrame(result["scores"][h]).T.round(4).to_string())
    print("\nTiming test on the untouched window:")
    trade = pd.DataFrame({k: {**v, "dsr": v["dsr"]["dsr"], "n_trials": v["dsr"]["n_trials"]}
                          for k, v in result["trading"].items()}).T
    print(trade[["config", "sharpe", "sharpe_full", "annual_return_full", "max_drawdown_full", "time_in_market",
                 "dsr", "n_trials"]].to_string())

    X_lgb = vol.lgb_table(panel.loc[pre])
    y_lgb = np.log(targets[21].loc[pre].stack(future_stack=True)).reindex(X_lgb.index)
    ok = np.isfinite(X_lgb.to_numpy(float)).all(axis=1) & np.isfinite(y_lgb.to_numpy(float))
    ok &= X_lgb.index.get_level_values(0) <= pre[-22]
    import lightgbm
    booster = lightgbm.LGBMRegressor(**vol.LGB_PARAMS).fit(X_lgb[ok], y_lgb[ok])
    sc_w = result["scores"]["21"]
    model_id = models.save("lgb_rv21", booster, X_lgb[ok], (pre[0].date(), pre[-22].date()),
                           sc_w["lgb"]["rmse"], meta={"params": vol.LGB_PARAMS, "features": vol.LGB_FEATURES,
                                                       "target": "log RV 21d", "refit": vol.REFIT})

    pre_dsr = registry.dsr(full[json.dumps(chosen)])
    summary = {
        "universe": UNIVERSE, "sessions": len(cal), "start": str(cal[0].date()), "end": str(cal[-1].date()),
        "close_hash": data_hash, "features": store.lags, "warmup": vol.WARMUP, "refit": vol.REFIT,
        "grid": GRID, "chosen": chosen, "best_forecast_h21_pre": best_h21,
        "pre_window": {"sessions": [str(have[0].date()), str(have[-1].date())], "scores": pre_scores,
                       "dm_pooled": {h: {k: flat(v) for k, v in pre_pooled[h].items()} for h in pre_pooled},
                       "beats_naive_5pct": {h: {k: int(v) for k, v in (pre_by_name[h] < -1.96).sum().items()}
                                           for h in pre_by_name},
                       "spy": spy_pre.to_dict("index"), "spy_dm": flat(spy_pre_dm)},
        "walk_forward": {"table": wf_table.reset_index().to_dict("records"),
                         "selected_oos_sharpe_active": sharpe(wf_oos[wf_oos != 0]),
                         "selected_oos_sharpe_full": sharpe(wf_oos)},
        "in_sample_by_variant": in_sample, "pre_window_chosen_dsr": pre_dsr,
        "untouched": {"start": str(hold.date()), "end": str(end.date()), "opened_at": lock.read()["opened_at"],
                      **result},
        "straddle_cycles": len(month_ends(cal)) - 1,
        "n_trials": registry.trials(), "n_runs_logged": len(registry.runs()), "model_id": model_id,
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    for h in map(str, vol.HORIZONS):
        pd.DataFrame(pre_scores[h]).T.to_csv(os.path.join(REPORTS, f"scores_pre_h{h}.csv"))
        pre_pooled[h]["qlike"].to_csv(os.path.join(REPORTS, f"dm_pooled_qlike_pre_h{h}.csv"))
        pre_by_name[h].to_csv(os.path.join(REPORTS, f"dm_vs_naive_by_name_pre_h{h}.csv"))
    wf_table.to_csv(os.path.join(REPORTS, "walk_forward.csv"))
    pd.DataFrame({"straddle": pnl, **{k: v for k, v in full.items()}}).to_csv(os.path.join(REPORTS, "returns.csv"))
    charts(forecasts, targets, vix, cal, hold, end, pnl, full, chosen, pre_scores, result)

    r = result["trading"]
    print(f"\nPre-window chosen {chosen}: Sharpe {in_sample[json.dumps(chosen)]['sharpe_full']:.3f} all days, "
          f"DSR {pre_dsr['dsr']:.3f} over {pre_dsr['n_trials']} trials")
    print(f"Untouched window: chosen Sharpe {r['chosen']['sharpe_full']:.3f} vs always-on "
          f"{r['baseline']['sharpe_full']:.3f} (all days); DSR {r['chosen']['dsr']['dsr']:.3f} "
          f"over {r['chosen']['dsr']['n_trials']} trials")
    print(f"Model {model_id} saved; {summary['n_trials']} trials in the registry; {summary['seconds']}s")


def charts(forecasts, targets, vix, cal, hold, end, pnl, full, chosen, pre_scores, result):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(targets[21]["SPY"], color="black", lw=0.8, label="realised, next 21 days")
    for m, c in (("har", "tab:blue"), ("gjr", "tab:orange"), ("lgb", "tab:green")):
        ax.plot(forecasts[21][m]["SPY"], lw=0.7, alpha=0.8, label=m, color=c)
    ax.plot(vix, lw=0.7, alpha=0.8, color="tab:red", label="VIX")
    ax.axvspan(hold, end, color="grey", alpha=0.15, label="untouched window")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("annualised vol")
    ax.set_title("SPY: 21-day realised vol against the forecasts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "spy_forecasts.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h in zip(axes, vol.HORIZONS):
        w = result["scores"][str(h)]
        pre_q = [pre_scores[str(h)][m]["qlike"] for m in vol.MODELS]
        win_q = [w[m]["qlike"] for m in vol.MODELS]
        x = np.arange(len(vol.MODELS))
        ax.bar(x - 0.2, pre_q, 0.4, label="pre-window")
        ax.bar(x + 0.2, win_q, 0.4, label="untouched window")
        ax.set_xticks(x, vol.MODELS)
        ax.set_title(f"QLIKE, {h}-day horizon, 21 names pooled (lower is better)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "horse_race.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot((1 + pnl).cumprod(), color="black", lw=1.0, label="always on (sleeve)")
    for k, v in full.items():
        p = json.loads(k)
        if p["model"] == "none":
            continue
        lw = 1.4 if p == chosen else 0.6
        axes[0].plot((1 + v).cumprod(), lw=lw, alpha=0.9 if p == chosen else 0.5, label=f"{p['model']} q={p['q']}")
    axes[0].axvspan(hold, end, color="grey", alpha=0.15)
    axes[0].set_ylabel("growth of 1")
    axes[0].set_title("Short straddle on SPY: always on against the timed variants (pre-window fits only)")
    axes[0].legend(fontsize=7, ncol=3)
    r = full[json.dumps(chosen)]
    axes[1].plot((r != 0).astype(float), lw=0.6)
    axes[1].set_ylabel("chosen in market")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "timing.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
