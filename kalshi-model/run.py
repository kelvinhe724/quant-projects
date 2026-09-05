"""Walk-forward over the calibration grid, the untouched final window opened once, every look logged to the registry.

Frozen before any result: the grid is three estimators times three
post-calibrations plus the published bin curve; the baseline is
logit+platt; four expanding folds on the markets before the untouched
window, the variant with the lowest log-loss on the last fifth of each
training window (purged by the 24-hour label horizon) is held on the test
fold; the untouched window is the last fifth of markets by close time, as
in framework/book/validate.py; the chosen variant is the one picked most
often, ties to the baseline; the window is read for the chosen variant, the
baseline, the bin curve and the market mid, nothing else; the trade rule
is edge over the whole quoted spread, quarter Kelly, at the touch plus the
taker fee.

Run: ../.venv/bin/python3 run.py   (about 3 minutes; a rerun reads the window's result back from the lock)
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
from purgedcv import WalkForwardSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from framework.book import validate  # noqa: E402
from research.alpha import Untouched, UntouchedWindowUsed  # noqa: E402
from research.models import ModelRegistry  # noqa: E402
from research.models.registry import data_hash  # noqa: E402
from research.registry import Registry  # noqa: E402

import candles  # noqa: E402
import model as km  # noqa: E402

REPORTS = km.REPORTS
UNIVERSE = ["kalshi-settled-24h"]
HOLDOUT = validate.HOLDOUT
FOLDS = 4
BASELINE = "logit+platt"
GRID = [f"{m}+{c}" for m in km.MODELS for c in km.CALIBRATIONS] + ["bias"]
LEAK = "hgb+none+settle_volume"
HORIZON = pd.Timedelta(hours=candles.pm.HORIZON_HOURS)
SCENARIOS = {"mid, no cost": dict(spread=0.0, charge_fee=False), "quoted spread": dict(charge_fee=False),
             "quoted spread + fee": {}, "kelly rule, quoted spread + fee": dict(rule="kelly")}
HEADLINE = "quoted spread + fee"
T0 = time.time()


def make(name):
    if name == LEAK:
        return km.Calibrator("hgb", "none", features=km.FEATURES + ["log_settle_volume"])
    return km.make(name)


def folds(df, n_splits=FOLDS, test_size=None):
    """purgedcv WalkForwardSplit over markets in close order; a training market must close before the first test quote.

    purgedcv keeps a training row whose close equals the first test quote to
    the second, so the purge is applied again strictly.
    """
    test_size = test_size or len(df) // (n_splits + 2)
    q = df["quote_time"].dt.tz_convert(None).reset_index(drop=True)
    c = df["close_time"].dt.tz_convert(None).reset_index(drop=True)
    split = WalkForwardSplit(n_splits=n_splits, test_size=test_size, prediction_times=q, evaluation_times=c)
    for tr, te in split.split(np.zeros((len(df), 1))):
        first = q.iloc[te].min()
        tr = tr[(c.iloc[tr] < first).to_numpy()]
        yield df.index[tr], df.index[te]


def val_split(train, frac=0.8):
    """First `frac` of a training window to fit on, the rest (purged by the label horizon) to select on."""
    cut = int(len(train) * frac)
    fit, rest = train.iloc[:cut], train.iloc[cut:]
    return fit, rest[rest["quote_time"] > fit["close_time"].max()]


def walk_forward(pre, registry):
    """Every variant's test predictions on each fold, the pick per fold, the stitched selected stream."""
    names = GRID + ["market", LEAK]
    preds = {n: pd.Series(np.nan, index=pre.index) for n in names}
    selected = pd.Series(np.nan, index=pre.index)
    rows = []
    for k, (tr, te) in enumerate(folds(pre)):
        train, test = pre.loc[tr], pre.loc[te]
        fit, val = val_split(train)
        vloss = {n: km.log_loss(val["outcome"], make(n).fit(fit, fit["outcome"]).predict(val)) for n in GRID}
        best = min(vloss, key=vloss.get)
        for n in names:
            preds[n][te] = make(n).fit(train, train["outcome"]).predict(test)
        selected[te] = preds[best][te]
        rows.append({"fold": k + 1, "train_start": train["close_time"].min().date(),
                     "train_end": train["close_time"].max().date(), "test_start": test["close_time"].min().date(),
                     "test_end": test["close_time"].max().date(), "n_train": len(train), "n_test": len(test),
                     "picked": best, "picked_val_log_loss": vloss[best],
                     "picked_test_log_loss": km.log_loss(test["outcome"], preds[best][te]),
                     "baseline_test_log_loss": km.log_loss(test["outcome"], preds[BASELINE][te]),
                     "market_test_log_loss": km.log_loss(test["outcome"], preds["market"][te])})
        print(f"  fold {k + 1}: picked {best} (val log-loss {vloss[best]:.4f}), {time.time() - T0:.0f}s", flush=True)
    table = pd.DataFrame(rows).set_index("fold")
    oos = pre.loc[selected.notna()]
    window = (oos["close_time"].min().date(), oos["close_time"].max().date())
    scored = {}
    for n, p in preds.items():
        s, r = km.score(oos, p[oos.index])
        scored[n] = s
        if n != "market":
            registry.record(n, {"variant": n, "stage": "walk_forward", "rule": "edge_over_spread", "cost": HEADLINE},
                            UNIVERSE, window, r, metrics=s, tags={"stage": "walk_forward_grid"})
    s, r = km.score(oos, selected[oos.index])
    scored["selected"] = s
    registry.record("walk-forward selected", {"grid": GRID, "folds": FOLDS, "rule": "edge_over_spread",
                                              "cost": HEADLINE}, UNIVERSE, window, r, metrics=s,
                    tags={"stage": "walk_forward_oos"})
    return table, pd.DataFrame(scored).T, preds


def lake_coverage():
    """What the data lake holds on Kalshi: listing rows, tickers, snapshots, and how many carry a settlement (none yet)."""
    sys.path.insert(0, os.path.join(ROOT, "data-lake"))
    try:
        import lake
        m = lake.load("kalshi_markets")
    except Exception as e:  # no store on this machine
        return {"error": f"{type(e).__name__}: {e}"}
    if len(m) == 0:
        return {"rows": 0}
    return {"rows": int(len(m)), "tickers": int(m["ticker"].nunique()), "snapshots": int(m["ts"].nunique()),
            "first": str(m["ts"].min()), "last": str(m["ts"].max()),
            "settled": int((m["status"] != "active").sum())}


def md(table, cols):
    out = ["| variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, s in table.iterrows():
        out.append(f"| {k} | " + " | ".join(f"{s[c]:.4f}" if isinstance(s[c], float) else str(s[c]) for c in cols) + " |")
    return "\n".join(out)


def chart_calibration(test, preds, path):
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for (label, p), colour in zip(preds.items(), ("#1f77b4", "#d62728", "#2ca02c")):
        t = km.calibration.bin_calibration(p, test["outcome"].to_numpy())
        t = t[t["n"] >= 20]
        err = np.clip(np.vstack([t["freq"] - t["freq_lo"], t["freq_hi"] - t["freq"]]), 0, None)
        ax.errorbar(t["mean_price"], t["freq"], yerr=err, fmt="o-", ms=4, lw=1.2, capsize=2.5, color=colour,
                    label=label)
    ax.set_xlabel("predicted P(YES)")
    ax.set_ylabel("realised frequency")
    ax.set_title(f"Untouched window, {len(test):,} markets: reliability")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_equity(curves, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, r in curves.items():
        if len(r):
            ax.plot(r.index, (1 + r).cumprod(), lw=1.3, label=label)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_ylabel("bankroll, quarter Kelly")
    ax.set_title("Untouched window: equity under each cost scenario")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_walk_forward(scored, path):
    t = scored.drop(index=["selected"]).sort_values("log_loss")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colours = ["#d62728" if n == "market" else "#7f7f7f" if n == LEAK else "#1f77b4" for n in t.index]
    ax.bar(range(len(t)), t["log_loss"] - scored.loc["market", "log_loss"], color=colours)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t.index, rotation=40, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("OOS log-loss minus the market's")
    ax.set_title("Walk-forward test folds: every variant against the mid (grey: settlement-volume leak)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(df=None, reports=REPORTS, categories=None):
    """The whole run on `df` (default: candles.build()) into `reports`. check.py calls it on planted data."""
    global T0
    T0 = time.time()
    os.makedirs(reports, exist_ok=True)
    n_snapshot = None
    if df is None:
        df = candles.build()
        n_snapshot = len(candles.pm.load_snapshot())
        cats = candles.pm.load()[["series", "category"]].drop_duplicates("series")
        categories = dict(zip(cats["series"], cats["category"]))
    df = df.reset_index(drop=True)
    df["log_settle_volume"] = np.log1p(df["settle_volume"])
    registry = Registry(os.path.join(reports, "registry"))
    models = ModelRegistry(os.path.join(reports, "models"))
    lock = Untouched(os.path.join(reports, "untouched.json"))
    dh = data_hash(df[["ticker", "price", "outcome"]])
    lock.lock(pd.DatetimeIndex(df["close_time"]), HOLDOUT, dh)
    hold = lock.window[0].tz_localize("UTC")
    pre = df[df["close_time"] < hold]
    test = df[df["close_time"] >= hold]
    print(f"{len(df):,} markets with candles"
          + (f" of {n_snapshot:,} in the snapshot" if n_snapshot else "")
          + f", {df['close_time'].min().date()} to {df['close_time'].max().date()}, data hash {dh}")
    print(f"untouched window from {hold.date()}: {len(test):,} markets"
          + (f", opened at {lock.read()['opened_at']}" if lock.opened else ", not yet opened"))

    print("walk-forward:")
    wf_table, wf_scored, preds = walk_forward(pre, registry)
    picks = wf_table["picked"].value_counts()
    chosen = picks.index[0] if picks.iloc[0] > picks.get(BASELINE, 0) else BASELINE
    print(wf_table.to_string())
    print(f"chosen for the untouched window: {chosen}")

    # Final training set: everything whose outcome was known before the window's first quote.
    train = df[df["close_time"] < hold - HORIZON]
    alphas = {chosen: make(chosen).fit(train, train["outcome"])}
    if BASELINE not in alphas:
        alphas[BASELINE] = make(BASELINE).fit(train, train["outcome"])
    alphas["bias"] = make("bias").fit(train, train["outcome"])
    alphas["market"] = make("market")
    for a in alphas.values():
        a.categories = categories or {}

    side = {"curves": {}}

    def final(start, end):
        out = {"chosen": chosen, "n_train": len(train), "n_test": len(test), "variants": {}}
        p_out = pd.DataFrame({"ticker": test["ticker"], "close_time": test["close_time"], "price": test["price"],
                              "spread": test["spread"], "outcome": test["outcome"]})
        returns = {}
        for n, a in alphas.items():
            p = a.predict(test)
            p_out[f"p_{n}"] = p
            out["variants"][n] = {}
            for label, kw in SCENARIOS.items():
                s, r = km.score(test, p, **kw)
                out["variants"][n][label] = s
                returns[(n, label)] = r
                if n != "market":
                    registry.record(n, {"variant": n, "stage": "untouched", "scenario": label,
                                        "rule": kw.get("rule", "edge_over_spread")}, UNIVERSE,
                                    (start.date(), end.date()), r, metrics=s, tags={"stage": "untouched"})
        # Every look is in the registry now; deflate the headline rows against all of them.
        for (n, label), r in returns.items():
            if n != "market" and label == HEADLINE:
                out["variants"][n][label]["dsr"] = (registry.dsr(r) if len(r) > 2 else
                                                    {"dsr": float("nan"), "psr": float("nan"),
                                                     "n_trials": registry.trials(), "n_obs": int(len(r))})
            if label == HEADLINE or n == chosen:
                side["curves"][f"{n}, {label}"] = r
        side["predictions"] = p_out
        return out

    try:
        result = lock.open(final)
        print(f"untouched window opened now; result stored in {lock.path}")
        curves = side["curves"]
        side["predictions"].to_csv(os.path.join(reports, "untouched_predictions.csv"), index=False)
        pd.DataFrame(curves).to_csv(os.path.join(reports, "untouched_returns.csv"))
    except UntouchedWindowUsed as e:
        result = lock.result
        print(f"{e}; reporting the stored result")
        curves = pd.read_csv(os.path.join(reports, "untouched_returns.csv"), index_col=0, parse_dates=True)
        curves = {c: curves[c].dropna() for c in curves}
    if result["chosen"] != chosen:
        print(f"WARNING: the window was opened with {result['chosen']}, this run chose {chosen}; "
              "the stored result stands and this run's choice is in-sample")
    p_out = pd.read_csv(os.path.join(reports, "untouched_predictions.csv"))

    window = (train["close_time"].min().date(), train["close_time"].max().date())
    ids = {}
    for role, name in (("chosen", chosen), ("baseline", BASELINE)):
        alpha = alphas[name]
        oos = result["variants"].get(name, {}).get(HEADLINE, {}).get("log_loss")
        ids[role] = models.save(str(alpha), alpha, train[km.FEATURES], window, oos,
                                meta={"role": role, "variant": name, "grid": GRID, "picks": picks.to_dict(),
                                      "features": km.FEATURES, "oos_metric": "log_loss on the untouched window",
                                      "device": km.DEVICE})

    # the ML line is the chosen variant, or the baseline when the walk-forward chose the bin curve itself
    ml = BASELINE if result["chosen"] == "bias" else result["chosen"]
    chart_calibration(test, {"market mid": p_out["p_market"].to_numpy(), "bias curve": p_out["p_bias"].to_numpy(),
                             f"model {ml}": p_out[f"p_{ml}"].to_numpy()}, os.path.join(reports, "calibration.png"))
    chart_equity(curves, os.path.join(reports, "equity.png"))
    chart_walk_forward(wf_scored, os.path.join(reports, "walk_forward.png"))
    wf_table.to_csv(os.path.join(reports, "walk_forward.csv"))
    wf_scored.to_csv(os.path.join(reports, "walk_forward_scores.csv"))

    u = result["variants"]
    summary = {
        "n_markets": len(df), "n_snapshot": n_snapshot, "start": str(df["close_time"].min()),
        "end": str(df["close_time"].max()), "data_hash": dh, "features": km.FEATURES, "grid": GRID,
        "baseline": BASELINE, "chosen": chosen, "device": km.DEVICE, "lake": lake_coverage(),
        "walk_forward": {"table": wf_table.reset_index().to_dict("records"), "scores": wf_scored.to_dict("index")},
        "untouched": {"start": str(hold.date()), "end": str(df["close_time"].max().date()),
                      "opened_at": lock.read()["opened_at"], **result},
        "model_ids": ids, "n_trials": registry.trials(), "n_runs_logged": len(registry.runs()),
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - T0),
    }
    with open(os.path.join(reports, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    cols = ("log_loss", "brier", "reliability", "n_bets", "mean_payoff", "t", "sharpe", "max_drawdown")
    print("\nWalk-forward OOS, every variant on the stitched test folds, headline cost scenario:\n")
    print(md(wf_scored, cols))
    print(f"\nUntouched window {hold.date()} on, {result['n_test']:,} markets, trained on {result['n_train']:,}:\n")
    rows = {f"{n}, {label}": s for n, v in u.items() for label, s in v.items()}
    print(md(pd.DataFrame(rows).T, cols))
    h = u[result["chosen"]][HEADLINE]
    print(f"\nHeadline ({result['chosen']}, {HEADLINE}): log-loss {h['log_loss']:.4f} vs market "
          f"{u['market'][HEADLINE]['log_loss']:.4f} and bias curve {u['bias'][HEADLINE]['log_loss']:.4f}; "
          f"Brier {h['brier']:.4f} vs {u['market'][HEADLINE]['brier']:.4f} / {u['bias'][HEADLINE]['brier']:.4f}; "
          f"{h['n_bets']} bets, {h['mean_payoff']:+.4f} per dollar (t {h['t']:+.2f}), Sharpe {h['sharpe']:+.2f}, "
          f"DSR {h['dsr']['dsr']:.3f} over {h['dsr']['n_trials']} trials")
    print(f"lake: {summary['lake']}")
    print(f"models saved: {ids}; {summary['n_trials']} trials in the registry; {summary['seconds']}s")
    return summary


if __name__ == "__main__":
    main()
