"""DeepLOB-style classifier against an imbalance logistic and LightGBM on one-second BTC bars, through the research harness.

Grid of model x horizon x tau, purged walk-forward on the sessions before
the untouched last fifth, every variant's stitched out-of-sample P&L logged
as a trial, the most-picked variant refit on everything before the window,
the window opened once for all variants and the chosen one's DSR taken
against the trials logged before it was read.

Run: ../.venv/bin/python3 run.py   (about 7 minutes on Apple MPS)
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
sys.path.insert(0, os.path.dirname(HERE))
import data  # noqa: E402
import lob  # noqa: E402
from research.alpha import Untouched, UntouchedWindowUsed  # noqa: E402
from research.features.store import panel_hash  # noqa: E402
from research.models import ModelRegistry  # noqa: E402
from research.registry import Registry  # noqa: E402

REPORTS = os.path.join(HERE, "reports")
# The harness's lock records the window's start as a date, so the untouched window is whole days: the last two of seven.
HOLDOUT = 2 / 7
FOLDS, TEST = 3, 86400
VAL_FRAC = 0.1
# Frozen before any result: the baseline variant, the grid and the selection score (validation-slice daily Sharpe).
BASELINE = {"model": "deeplob", "horizon": 10, "tau": 0.1}
GRID = [{"model": m, "horizon": h, "tau": t} for m in lob.MODELS for h in lob.HORIZONS for t in lob.TAUS]
COLOURS = {"logit-imbalance": "#d97706", "lightgbm": "#2a9d8f", "deeplob": "#1f5fa8"}


def key(v):
    return f"{v['model']} h={v['horizon']} tau={v['tau']}"


def fit_all(frame, Y, rows):
    """Every model on `rows`, the last VAL_FRAC of them held back for early stopping and selection."""
    cut = len(rows) - int(len(rows) * VAL_FRAC)
    tr, va = rows[:cut], rows[cut:]
    models = {}
    for name, cls in lob.MODELS.items():
        t0 = time.time()
        models[name] = cls().fit(frame.loc[tr], Y.loc[tr], val=(frame.loc[va], Y.loc[va]))
        print(f"    {name} fit on {len(tr):,} rows in {time.time() - t0:.0f}s", flush=True)
    return models, va


def score(models, frame, Y, start, end):
    """Classification per model x horizon and (returns, positions) per grid variant on start..end."""
    cls, pnl = {}, {}
    for name, m in models.items():
        p = lob.proba_on(m, frame, start, end)
        for h in lob.HORIZONS:
            cls[(name, h)] = lob.classification(p[h], Y[h].loc[start:end])
            for tau in lob.TAUS:
                pos = lob.positions(p[h], tau)
                pnl[key({"model": name, "horizon": h, "tau": tau})] = (lob.pnl(pos, frame["mid"]), pos)
    return cls, pnl


def sharpe_daily(r):
    return lob.summarise(r, pd.Series(0.0, index=r.index))["sharpe_daily"]


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    frame, meta = data.load()
    Y = lob.label_frame(frame["mid"])
    cal = frame.index
    registry = Registry(os.path.join(REPORTS, "registry"))
    models_reg = ModelRegistry(os.path.join(REPORTS, "models"))
    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    data_hash = panel_hash(frame)
    lock.lock(cal, HOLDOUT, data_hash)
    hold, end = lock.window[0].tz_localize("UTC"), cal[-1]
    pre = cal[cal < hold]
    print(f"source: {meta['source']}; {len(cal):,} seconds {cal[0]} to {cal[-1]}, hash {data_hash}")
    if meta.get("backfilled"):
        print(f"lake L2 stream holds {meta['lake_hours']:.1f} hours of {meta['symbol']}, below {data.MIN_HOURS}: "
              f"backfilled from {meta['source']}, {meta['days'][0]} to {meta['days'][-1]}")
    print(f"untouched window {hold} to {end}, {len(cal) - len(pre):,} seconds"
          + (f", opened at {lock.read()['opened_at']}" if lock.opened else ", not yet opened"))
    print("class shares by horizon (pre-window):")
    print(Y.loc[pre].apply(lambda s: s.value_counts(normalize=True)).round(3).to_string())

    rows, cls_rows, stitched, picked = [], [], {key(v): [] for v in GRID}, []
    for k, (tr, te) in enumerate(lob.folds(pre, FOLDS, TEST)):
        train_rows, test_rows = pre[tr], pre[te]
        print(f"\nfold {k + 1}: train {train_rows[0]} to {train_rows[-1]}, test {test_rows[0]} to {test_rows[-1]}", flush=True)
        models, va = fit_all(frame, Y, train_rows)
        _, vpnl = score(models, frame, Y, va[0], va[-1])
        tcls, tpnl = score(models, frame, Y, test_rows[0], test_rows[-1])
        vs = {v: sharpe_daily(r) for v, (r, _) in vpnl.items()}
        best = max(vs, key=lambda v: vs[v] if np.isfinite(vs[v]) else -np.inf)
        picked.append(tpnl[best])
        rows.append({"fold": k + 1, "train_start": train_rows[0], "train_end": train_rows[-1],
                     "test_start": test_rows[0], "test_end": test_rows[-1], "picked": best,
                     "picked_val_sharpe": vs[best], "picked_test_sharpe": sharpe_daily(tpnl[best][0]),
                     "picked_test_bps": float(tpnl[best][0].sum() * lob.BPS),
                     "baseline_test_sharpe": sharpe_daily(tpnl[key(BASELINE)][0])})
        for (name, h), c in tcls.items():
            cls_rows.append({"fold": k + 1, "model": name, "horizon": h, **c})
        for v, rp in tpnl.items():
            stitched[v].append(rp)
        print(f"  picked {best} (val Sharpe {vs[best]:.2f}, test {rows[-1]['picked_test_sharpe']:.2f}); "
              + "; ".join(f"{n} h5 acc {tcls[(n, 5)]['accuracy']:.3f}" for n in lob.MODELS), flush=True)

    wf_table = pd.DataFrame(rows).set_index("fold")
    positions = {v: pd.concat([p for _, p in rp]) for v, rp in stitched.items()}
    stitched = {v: pd.concat([r for r, _ in rp]) for v, rp in stitched.items()}
    selected = pd.concat([r for r, _ in picked])
    sel = lob.summarise(selected, pd.concat([p for _, p in picked]))
    oos = {}
    for v in GRID:
        r = stitched[key(v)]
        s = lob.summarise(r, positions[key(v)])
        oos[key(v)] = s
        registry.record(v["model"], {"horizon": v["horizon"], "tau": v["tau"], "band_bps": lob.BAND[v["horizon"]],
                                     "fee_bps": lob.FEE_BPS, "window_s": lob.WINDOW},
                        [lob.SYMBOL], (pre[0], pre[-1]), r, metrics=s, tags={"stage": "walk_forward_oos"})
    registry.record("walk-forward selected", {"grid": GRID, "folds": FOLDS, "test_size": TEST, "select": "val daily Sharpe"},
                    [lob.SYMBOL], (pre[0], pre[-1]), selected, metrics=sel, tags={"stage": "walk_forward_oos"})
    counts = wf_table["picked"].value_counts()
    chosen_key = counts.index[0] if counts.iloc[0] > counts.get(key(BASELINE), 0) else key(BASELINE)
    chosen = next(v for v in GRID if key(v) == chosen_key)
    cls_oos = pd.DataFrame(cls_rows).groupby(["model", "horizon"])[["accuracy", "f1_macro", "majority", "n"]].mean()
    print("\nWalk-forward:\n")
    print(wf_table.drop(columns=["train_start"]).round(3).to_string())
    print(f"\nselected stitched OOS: {sel['total_bps']:+.1f} bps over {sel['hours']:.1f} h, daily Sharpe "
          f"{sel['sharpe_daily']:.2f}, {sel['trades']} trades; chosen for the final window: {chosen_key}")
    print("\nstitched OOS classification (mean over folds):\n")
    print(cls_oos.round(3).to_string())

    show = [key(BASELINE)] + [key({**BASELINE, "model": m}) for m in lob.MODELS if m != BASELINE["model"]]
    print("\nfinal fit on every pre-window second whose label ends before the window", flush=True)
    final, _ = fit_all(frame, Y, pre[pre + pd.Timedelta(seconds=max(lob.HORIZONS)) < hold])
    n_trials_before = registry.trials()

    def open_window(start, stop):
        start, stop = pd.Timestamp(start).tz_localize("UTC"), end
        cls, pnl = score(final, frame, Y, start, stop)
        r, pos = pnl[chosen_key]
        head = lob.summarise(r, pos)
        d = registry.dsr(r)
        keep = {chosen_key, *show}
        out = {"chosen": chosen, "headline": head, "dsr": d, "n_trials_before": n_trials_before,
               "classification": {f"{n} h={h}": c for (n, h), c in cls.items()},
               "pnl": {v: lob.summarise(rr, pp) for v, (rr, pp) in pnl.items()},
               "returns_1min": {v: {str(t): round(float(x), 10) for t, x in rr.resample("1min").sum().items()}
                                for v, (rr, _) in pnl.items() if v in keep}}
        for v in GRID:
            rr, pp = pnl[key(v)]
            registry.record(v["model"], {"horizon": v["horizon"], "tau": v["tau"], "window": "untouched"}, [lob.SYMBOL],
                            (start, stop), rr, metrics=lob.summarise(rr, pp), tags={"stage": "untouched"})
        return out

    try:
        result = lock.open(open_window)
        print(f"\nuntouched window opened now; result stored in {lock.path}")
    except UntouchedWindowUsed as e:
        result = lock.result
        print(f"\n{e}; reporting the stored result")
    if result["chosen"] != chosen:
        print(f"WARNING: the window was opened with {result['chosen']}, this run chose {chosen}; "
              "the stored result stands and this run's choice is in-sample")
    u_ret = {v: pd.Series({pd.Timestamp(t): x for t, x in s.items()}).sort_index()
             for v, s in result["returns_1min"].items()}

    model = final[chosen["model"]]
    model_id = models_reg.save(chosen_key, model.cpu() if hasattr(model, "cpu") else model, frame.loc[pre],
                               (pre[0], pre[-1]), result["headline"]["sharpe_daily"],
                               meta={"chosen": chosen, "grid": GRID, "picks": counts.to_dict(), "band_bps": lob.BAND,
                                     "source": meta["source"], "history": getattr(model, "history", None)})

    u = result
    summary = {
        "data": {**meta, "seconds": len(cal), "start": str(cal[0]), "end": str(cal[-1]), "hash": data_hash,
                 "features": list(lob.features(frame).columns),
                 "class_shares_pre": {h: Y[h].loc[pre].value_counts(normalize=True).to_dict() for h in lob.HORIZONS},
                 "band_bps": lob.BAND, "fee_bps": lob.FEE_BPS},
        "grid": GRID, "baseline": BASELINE, "chosen": chosen,
        "walk_forward": {"table": wf_table.reset_index().to_dict("records"), "selected": sel,
                         "variants": oos, "classification": cls_oos.reset_index().to_dict("records")},
        "untouched": {"start": str(hold), "end": str(end), "opened_at": lock.read()["opened_at"],
                      **{k: v for k, v in u.items() if k != "returns_1min"}},
        "n_trials": registry.trials(), "n_runs_logged": len(registry.runs()), "model_id": model_id,
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds_elapsed": round(time.time() - t0),
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    wf_table.to_csv(os.path.join(REPORTS, "walk_forward.csv"))
    pd.DataFrame(oos).T.to_csv(os.path.join(REPORTS, "walk_forward_variants.csv"))
    cls_unt = pd.DataFrame(u["classification"]).T
    cls_unt.index = pd.MultiIndex.from_tuples([(k.split(" h=")[0], int(k.split(" h=")[1])) for k in cls_unt.index],
                                              names=["model", "horizon"])
    pd.concat({"walk_forward_oos": cls_oos, "untouched": cls_unt}, names=["window"]).to_csv(
        os.path.join(REPORTS, "classification.csv"))
    pd.DataFrame(u["pnl"]).T.to_csv(os.path.join(REPORTS, "untouched_variants.csv"))
    minute = pd.DataFrame({v: pd.concat([stitched[v].resample("1min").sum(), u_ret[v]]) for v in show})
    minute["selected"] = pd.concat([selected.resample("1min").sum(), u_ret[key(result["chosen"])]])
    minute.round(10).to_csv(os.path.join(REPORTS, "returns_1min.csv"))

    fig, ax = plt.subplots(figsize=(11, 5))
    for v in show:
        ax.plot(minute[v].cumsum() * lob.BPS, linewidth=1.2, color=COLOURS[v.split(" ")[0]], label=v)
    ax.plot(minute["selected"].cumsum() * lob.BPS, linewidth=1.2, color="black", label="walk-forward selected")
    ax.axvspan(hold, end, color="grey", alpha=0.15, label="untouched window")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("cumulative net P&L, bps")
    ax.set_title("Stitched walk-forward test windows and the untouched window, 1 bp fee, fill one second late")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.2
    x = np.arange(len(lob.HORIZONS))
    for i, name in enumerate(lob.MODELS):
        ax.bar(x + (i - 1) * width, [u["classification"][f"{name} h={h}"]["f1_macro"] for h in lob.HORIZONS], width,
               color=COLOURS[name], label=name)
    flat = [2 * u["classification"][f"deeplob h={h}"]["majority"] / (1 + u["classification"][f"deeplob h={h}"]["majority"]) / 3
            for h in lob.HORIZONS]
    ax.plot(x, flat, "k_", markersize=40, label="always flat")
    ax.set_xticks(x, [f"{h} s" for h in lob.HORIZONS])
    ax.set_ylabel("macro F1, untouched window")
    ax.set_ylim(0.2, 0.4)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "f1.png"), dpi=140)
    plt.close(fig)

    h = u["headline"]
    print(f"\nUntouched window {hold} to {end}: {chosen_key}: {h['total_bps']:+.1f} bps over {h['hours']:.1f} h "
          f"({h['bps_per_day']:+.1f} bps/day), daily Sharpe {h['sharpe_daily']:.2f}, {h['trades']} trades at "
          f"{h['bps_per_trade']:+.3f} bps, in market {h['in_market']:.0%}; PSR {u['dsr']['psr']:.3f}, DSR "
          f"{u['dsr']['dsr']:.3f} over {u['dsr']['n_trials']} trials")
    print("untouched classification:")
    print(pd.DataFrame(u["classification"]).T.round(3).to_string())
    print(f"Model {model_id} saved; {summary['n_trials']} trials in the registry; {summary['seconds_elapsed']}s")


if __name__ == "__main__":
    main()
