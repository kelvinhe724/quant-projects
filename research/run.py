"""Worked example: a cross-sectional momentum Alpha through the whole platform on the book's 14 ETFs.

Features from the store, every grid variant logged as a trial, walk-forward
picks the lookback, the untouched last fifth is opened once and its result
stored in the lock file, the chosen alpha is saved to the model registry
with the training panel's hash, and the same Alpha runs as a framework
sleeve under the book's cost model and overlay. Reruns reproduce every
number; the untouched window's number is read back from the lock.

Run: ../.venv/bin/python3 run.py   (about 6 seconds)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.book import universe, validate
from framework.book.strategies import book_config
from framework.engine import run, sharpe
from research.alpha import Alpha, Sleeve, Untouched, UntouchedWindowUsed, backtest, positions, walk_forward
from research.alpha.base import month_ends, window_returns
from research.features import PRICE, FeatureStore, Raw
from research.features.store import panel_hash
from research.models import ModelRegistry
from research.registry import Registry

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
ETFS = list(universe.ETFS)
# Frozen before any result: the baseline is 12-1, the grid is three lookbacks, the
# untouched window is the last fifth of sessions as in framework/book/validate.py.
BASELINE = {"feature": "mom_12_1"}
GRID = [{"feature": "mom_3_1"}, {"feature": "mom_6_1"}, {"feature": "mom_12_1"}]
HOLDOUT = validate.HOLDOUT
FOLDS, TEST_SIZE = 4, 504


class XSMom(Alpha):
    """Rank the ETFs on one momentum feature; long the top third, short the bottom third, equal weight."""

    def __init__(self, feature="mom_12_1", frac=1 / 3):
        self.feature, self.frac = feature, frac
        self.name = f"XSMom[{feature}]"

    def signal(self, xs):
        s = xs[self.feature].dropna()
        k = max(int(len(s) * self.frac), 1)
        w = pd.Series(0.0, index=xs.index)
        if len(s) < 2 * k:
            return w
        rank = s.rank(method="first")
        w[rank.index] = ((rank > len(s) - k).astype(float) - (rank <= k).astype(float)) / k
        return w


def stats(r):
    r = r[r != 0].dropna()
    return validate.stats(r) if len(r) > 1 else {}


def main():
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    bars = universe.load_bars().select(ETFS)
    raw = Raw.from_bars(bars, series=[])
    store = FeatureStore(PRICE)
    panel = store.build(raw, audit=True)
    close = raw.frames["close"]
    cal = panel.index
    registry = Registry(os.path.join(REPORTS, "registry"))
    models = ModelRegistry(os.path.join(REPORTS, "models"))
    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    data_hash = panel_hash(close)
    lock.lock(cal, HOLDOUT, data_hash)
    hold, end = lock.window
    pre = cal[cal < hold]
    print(f"{len(ETFS)} ETFs, {cal[0].date()} to {cal[-1].date()}, {len(cal)} sessions, close hash {data_hash}")
    print(f"untouched window {hold.date()} to {end.date()}, {len(cal) - len(pre)} sessions"
          + (f", opened at {lock.read()['opened_at']}" if lock.opened else ", not yet opened"))

    wf_table, wf_oos, full = walk_forward(XSMom, GRID, panel.loc[pre], close.loc[pre], FOLDS, TEST_SIZE,
                                          registry=registry, universe=ETFS)
    registry.record("XSMom walk-forward", {"grid": GRID, "folds": FOLDS, "test_size": TEST_SIZE}, ETFS,
                    (pre[0].date(), pre[-1].date()), wf_oos, tags={"stage": "walk_forward_oos"})
    picks = wf_table["picked"].value_counts()
    chosen = json.loads(picks.index[0]) if picks.iloc[0] > picks.get(json.dumps(BASELINE), 0) else BASELINE
    alpha = XSMom(**chosen)
    print("\nWalk-forward over the grid, pre-untouched sessions:\n")
    print(wf_table.round(3).to_string())
    print(f"\nselected-in-sample OOS Sharpe {sharpe(wf_oos[wf_oos != 0]):.3f}; chosen for the final window: {chosen}")
    in_sample = {k: stats(v) for k, v in full.items()}

    sleeve = Sleeve(XSMom(**chosen), store, series=[])
    engine = run(sleeve, bars, config=book_config(kill=False))
    research = positions(alpha, panel, month_ends(cal))
    sent = pd.DataFrame(sleeve.targets).T.reindex(columns=ETFS)
    parity = np.allclose(sent.reindex(research.index).to_numpy(), research.to_numpy(), atol=1e-12)
    print(f"\nadapter parity at {len(sent)} month ends: {'exact' if parity else 'BROKEN'}")
    r_research = backtest(research, close)
    r_engine = engine.returns

    def final(start, stop):
        r = window_returns(alpha, panel, close, start, stop, month_ends(cal))
        e = r_engine.loc[start:stop]
        out = {"chosen": chosen, "research": stats(r), "engine": stats(e), "dsr": registry.dsr(r),
               "engine_dsr": registry.dsr(e)}
        registry.record(str(alpha), {**chosen, "window": "untouched"}, ETFS, (start.date(), stop.date()), r,
                        tags={"stage": "untouched"})
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

    model_id = models.save(str(alpha), {"alpha": "XSMom", **chosen, "features": store.names, "lags": store.lags},
                           close.loc[pre], (pre[0].date(), pre[-1].date()), result["research"]["sharpe"],
                           meta={"chosen": chosen, "grid": GRID, "picks": picks.to_dict()})

    full_r = stats(r_research)
    full_e = stats(r_engine)
    d_full = registry.dsr(r_research.loc[pre])
    summary = {
        "universe": ETFS, "sessions": len(cal), "start": str(cal[0].date()), "end": str(cal[-1].date()),
        "close_hash": data_hash, "features": store.lags, "grid": GRID, "chosen": chosen,
        "walk_forward": {"table": wf_table.reset_index().to_dict("records"),
                         "selected_oos_sharpe": sharpe(wf_oos[wf_oos != 0])},
        "in_sample_by_variant": in_sample, "pre_window_chosen": stats(r_research.loc[pre]), "pre_window_dsr": d_full,
        "untouched": {"start": str(hold.date()), "end": str(end.date()), **result,
                      "opened_at": lock.read()["opened_at"]},
        "full_panel": {"research": full_r, "engine": full_e},
        "engine": {"turnover_per_year": float(engine.turnover.sum() / (len(cal) / 252)),
                   "trades": int(len(engine.trades)), "avg_gross": engine.metrics["avg_gross_exposure"]},
        "adapter_parity": bool(parity), "n_trials": registry.trials(), "n_runs_logged": len(registry.runs()),
        "model_id": model_id, "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0),
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    pd.DataFrame({"research": r_research, "engine": r_engine}).to_csv(os.path.join(REPORTS, "returns.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, v in full.items():
        axes[0].plot((1 + v).cumprod(), linewidth=0.8, alpha=0.6, label=f"research {json.loads(k)['feature']}")
    axes[0].plot((1 + r_research).cumprod(), color="black", linewidth=1.3, label=f"research {chosen['feature']}, full")
    axes[0].plot((1 + r_engine).cumprod(), color="firebrick", linewidth=1.3, label="engine, book cost model + overlay")
    axes[0].axvspan(hold, end, color="grey", alpha=0.15, label="untouched window")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("growth of 1 (log)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("XSMom on the 14 book ETFs: research path (10 bps) vs framework sleeve")
    axes[1].plot(engine.weights.abs().sum(axis=1), linewidth=0.8)
    axes[1].set_ylabel("engine gross")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity.png"), dpi=140)
    plt.close(fig)

    u = result
    print(f"\nPre-untouched window, chosen variant: Sharpe {summary['pre_window_chosen']['sharpe']:.3f}, "
          f"PSR {d_full['psr']:.3f}, DSR {d_full['dsr']:.3f} over {d_full['n_trials']} trials")
    print(f"Untouched window {hold.date()} to {end.date()}: research Sharpe {u['research']['sharpe']:.3f} "
          f"(DSR {u['dsr']['dsr']:.3f}), engine Sharpe {u['engine']['sharpe']:.3f} (DSR {u['engine_dsr']['dsr']:.3f})")
    print(f"Full panel: research Sharpe {full_r['sharpe']:.3f}, engine Sharpe {full_e['sharpe']:.3f}, "
          f"engine turnover {summary['engine']['turnover_per_year']:.1f}x/yr, {summary['engine']['trades']} trades")
    print(f"Model {model_id} saved; {summary['n_trials']} trials in the registry; {summary['seconds']}s")


if __name__ == "__main__":
    main()
