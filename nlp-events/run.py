"""Post-announcement drift on S&P 500 earnings 8-Ks: FinBERT sentiment against the price reaction.

Filings come from the lake's edgar_8k set (backfill.py fills 2022 on), prices
from equities_daily through the point-in-time filter. FinBERT scores the
press-release exhibit; the score, the reaction-day return and the score's
residual to the reaction are ranked causally and carried forward as
features at lag 0. Three EventDrift variants per signal go through the
research walk-forward on the sessions before the untouched window; the
window opens once for the three pre-registered variants and the
walk-forward's pick, and every run lands in the registry with its DSR.

Run: ../.venv/bin/python3 run.py [--minutes N]   (scoring is cached and resumable)
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
import lake  # noqa: E402
from framework.book import validate  # noqa: E402
from framework.engine import by_year, sharpe  # noqa: E402
from research.alpha import Untouched, UntouchedWindowUsed, backtest, positions, walk_forward  # noqa: E402
from research.alpha.base import window_returns  # noqa: E402
from research.features import FeatureStore, Raw  # noqa: E402
from research.features.store import panel_hash  # noqa: E402
from research.models import ModelRegistry  # noqa: E402
from research.registry import Registry  # noqa: E402

from events import (FEATURES, SIGNALS, EventDrift, event_table, finbert_scorer, forward_returns, ic_table,  # noqa: E402
                    reaction_session)

REPORTS = os.path.join(HERE, "reports")
START, END = "2022-01-01", "2026-08-31"
WARMUP = "2021-11-01"
HOLDOUT = validate.HOLDOUT
# Frozen before any result: the baseline is the price reaction held ten
# sessions, the two sentiment variants sit next to it at the same horizon,
# the grid is three horizons per signal, the untouched window is the last
# fifth of sessions as in framework/book/validate.py.
BASELINE = {"signal": "react", "horizon": 10}
HEADLINE = [BASELINE, {"signal": "sent", "horizon": 10}, {"signal": "resid", "horizon": 10}]
GRID = [{"signal": s, "horizon": h} for s in SIGNALS for h in (5, 10, 20)]
FOLDS, TEST_SIZE = 3, 252
HORIZONS = list(range(1, 21))
SHOW = (1, 2, 5, 10, 20)


def stats(r):
    r = r[r != 0].dropna()
    return validate.stats(r) if len(r) > 1 else {}


def body(text):
    """Drop the exhibit header lines: the tag, sequence, filename and type the index page prints."""
    lines = text.split("\n")
    while lines and len(lines[0]) < 40:
        lines.pop(0)
    return "\n".join(lines)


def scores(fil, minutes):
    """FinBERT score per accession, cached in reports/sentiment.parquet and resumed on rerun."""
    path = os.path.join(REPORTS, "sentiment.parquet")
    cache = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["accession", "sent"])
    todo = fil[~fil["accession"].isin(cache["accession"])]
    if len(todo):
        t0 = time.time()
        score = finbert_scorer()
        print(f"scoring {len(todo)} filings ({len(cache)} cached)")
        for i in range(0, len(todo), 200):
            block = todo.iloc[i:i + 200]
            s = score([body(t) for t in block["text"]])
            cache = pd.concat([cache, pd.DataFrame({"accession": block["accession"].to_numpy(), "sent": s})],
                              ignore_index=True)
            cache.to_parquet(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            print(f"  {len(cache)} scored, {time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > minutes * 60 and i + 200 < len(todo):
                print(f"scoring budget of {minutes} minutes used; rerun to continue")
                sys.exit(0)
    return fil.merge(cache, on="accession", how="inner")


def load():
    px = lake.load("equities_daily", WARMUP, END, universe="sp500")
    spy = lake.load("equities_daily", WARMUP, END, universe=["SPY"]).set_index("date").sort_index()
    cal = pd.DatetimeIndex(spy.index)
    factor = px["adj_close"] / px["close"]
    px = px.assign(close=px["adj_close"], open=px["open"] * factor)
    opens = px.pivot(index="date", columns="ticker", values="open").reindex(cal)
    closes = px.pivot(index="date", columns="ticker", values="close").reindex(cal)
    fil = lake.load("edgar_8k", START, END, universe="sp500")
    fil = fil[fil["earnings"] & (fil["form"] == "8-K") & fil["accepted"].notna() & fil["ticker"].isin(closes.columns)]
    return opens, closes, spy, fil.reset_index(drop=True)


def main(minutes=9.0):
    t0 = time.time()
    os.makedirs(REPORTS, exist_ok=True)
    opens, closes, spy, fil = load()
    print(f"{closes.shape[1]} names with prices, {len(closes)} sessions, {len(fil)} earnings 8-Ks {START} to {END}")
    fil = scores(fil, minutes)
    fil["date"] = reaction_session(fil["accepted"], closes.index)
    ev = fil.dropna(subset=["date"])[["date", "ticker", "accession", "accepted", "sent"]].assign(text="")
    raw = Raw({"open": opens, "close": closes}, filings=ev)
    store = FeatureStore(FEATURES)
    panel = store.build(raw, audit=True)
    events = event_table(raw)
    cal = closes.index[:-1]
    panel, p = panel.loc[cal], opens.shift(-1).loc[cal]
    spy_open = spy["open"]
    names = list(closes.columns)

    registry = Registry(os.path.join(REPORTS, "registry"))
    models = ModelRegistry(os.path.join(REPORTS, "models"))
    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    data_hash = panel_hash(closes.loc[cal])
    lock.lock(cal, HOLDOUT, data_hash)
    hold, end = lock.window
    pre = cal[cal < hold]
    print(f"{len(events)} events on {events['ticker'].nunique()} names, close hash {data_hash}")
    print(f"untouched window {hold.date()} to {end.date()}, {len(cal) - len(pre)} sessions"
          + (f", opened at {lock.read()['opened_at']}" if lock.opened else ", not yet opened"))

    cut = cal[cal.get_loc(hold) - 21]
    ev_pre = events[events["date"] <= cut]
    ic = ic_table(ev_pre, forward_returns(ev_pre, opens, spy_open, HORIZONS))
    ic_early = ic_table(ev_pre, forward_returns(ev_pre, opens, spy_open, HORIZONS, entry=0), signals=("sent",))
    ic_day = ic_table(ev_pre, pd.DataFrame({0: ev_pre["react"] - spy["close"].pct_change().reindex(
        pd.DatetimeIndex(ev_pre["date"])).to_numpy()}), signals=("sent",))
    print(f"\nEvent study on {len(ev_pre)} pre-window events, excess over SPY, entry at the open after the "
          "reaction session:\n")
    show = ic[ic["horizon"].isin(SHOW)].pivot(index="horizon", columns="signal", values=["ic", "t"])
    print(show.round(3).to_string())
    print(f"\nreaction-day IC of sentiment: {ic_day.loc[0, 'ic']:.3f} (t {ic_day.loc[0, 't']:.1f})")

    wf_table, wf_oos, full = walk_forward(EventDrift, GRID, panel.loc[pre], p.loc[pre], FOLDS, TEST_SIZE,
                                          rebalance="daily", horizon=21, registry=registry, universe=names)
    registry.record("EventDrift walk-forward", {"grid": GRID, "folds": FOLDS, "test_size": TEST_SIZE}, names,
                    (pre[0].date(), pre[-1].date()), wf_oos, tags={"stage": "walk_forward_oos"})
    picks = wf_table["picked"].value_counts()
    chosen = json.loads(picks.index[0]) if picks.iloc[0] > picks.get(json.dumps(BASELINE), 0) else BASELINE
    print("\nWalk-forward over the grid, pre-window sessions:\n")
    print(wf_table.round(3).to_string())
    print(f"\nselected-in-sample OOS Sharpe {sharpe(wf_oos[wf_oos != 0]):.3f}; chosen for the final window: {chosen}")
    in_sample = {k: stats(v) for k, v in full.items()}
    print("\nPre-window, every grid variant, 10 bps:\n")
    print(pd.DataFrame(in_sample).T[["sharpe", "annual_return", "annual_vol", "max_drawdown"]].round(3).to_string())

    final_set = HEADLINE + ([chosen] if chosen not in HEADLINE else [])

    def final(start, stop):
        out = {"variants": {}, "chosen": chosen}
        rets = {}
        for params in final_set:
            a = EventDrift(**params)
            r = window_returns(a, panel, p, start, stop, cal)
            g = window_returns(a, panel, p, start, stop, cal, cost_bps=0.0)
            rets[str(a)] = r
            out["variants"][str(a)] = {"params": params, **stats(r), "sharpe_gross": sharpe(g[g != 0])}
            registry.record(str(a), {**params, "window": "untouched"}, names, (start.date(), stop.date()), r,
                            tags={"stage": "untouched"})
        for k, r in rets.items():
            out["variants"][k]["dsr"] = registry.dsr(r)
        ev_w = events[(events["date"] >= start) & (events["date"] <= stop)]
        icw = ic_table(ev_w, forward_returns(ev_w, opens, spy_open, HORIZONS))
        out["ic"] = icw[icw["horizon"].isin(SHOW)].to_dict("records")
        out["n_events"] = int(len(ev_w))
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

    model_id = models.save(f"EventDrift[{chosen['signal']},{chosen['horizon']}]",
                           {"alpha": "EventDrift", **chosen, "features": store.names, "lags": store.lags,
                            "model": "ProsusAI/finbert", "chunks": "first 4 x 510 tokens"},
                           closes.loc[pre], (pre[0].date(), pre[-1].date()),
                           result["variants"][f"EventDrift[{chosen['signal']},{chosen['horizon']}]"]["sharpe"],
                           meta={"chosen": chosen, "grid": GRID, "picks": picks.to_dict()})

    full_r = {}
    for params in final_set:
        a = EventDrift(**params)
        full_r[str(a)] = backtest(positions(a, panel, cal), p)
    full_stats = {k: {**stats(r), "by_year": by_year(r).round(3).to_dict()} for k, r in full_r.items()}
    fwd_all = forward_returns(events, opens, spy_open, HORIZONS)
    dec = pd.DataFrame({"decile": pd.cut(events["sent_pct"], np.linspace(0, 1, 11), labels=False, include_lowest=True),
                        "reaction": events["react"], "drift_10": fwd_all[10]}).dropna().groupby("decile").mean()

    summary = {
        "universe_size": len(names), "sessions": len(cal), "start": str(cal[0].date()), "end": str(cal[-1].date()),
        "close_hash": data_hash, "n_filings_scored": int(len(fil)), "n_events": int(len(events)),
        "n_events_pre": int(len(ev_pre)), "accepted_before_0900_et": float(
            (pd.DatetimeIndex(events["accepted"]).tz_convert("America/New_York").strftime("%H:%M") < "09:00").mean()),
        "features": store.lags, "grid": GRID, "headline": HEADLINE, "chosen": chosen,
        "ic_pre": ic.to_dict("records"), "ic_early_entry_pre": ic_early.to_dict("records"),
        "ic_reaction_day": ic_day.to_dict("records")[0],
        "walk_forward": {"table": wf_table.reset_index().to_dict("records"),
                         "selected_oos_sharpe": sharpe(wf_oos[wf_oos != 0])},
        "pre_window_by_variant": in_sample,
        "pre_window_dsr": {k: registry.dsr(v) for k, v in full.items()},
        "untouched": {"start": str(hold.date()), "end": str(end.date()), **result,
                      "opened_at": lock.read()["opened_at"]},
        "full_panel": full_stats, "sentiment_deciles": dec.to_dict(),
        "n_trials": registry.trials(), "n_runs_logged": len(registry.runs()), "model_id": model_id,
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    ic.to_csv(os.path.join(REPORTS, "ic_by_horizon.csv"), index=False)
    pd.DataFrame(full_r).to_csv(os.path.join(REPORTS, "returns.csv"))
    events.drop(columns=["text"]).to_csv(os.path.join(REPORTS, "events.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for sig in SIGNALS:
        d = ic[ic["signal"] == sig]
        ax.plot(d["horizon"], d["ic"], marker="o", markersize=3, label=sig)
    ax.plot(ic_early["horizon"], ic_early["ic"], linestyle="--", color="grey", label="sent, entry at open(R)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("sessions from the entry open")
    ax.set_ylabel("Spearman IC with excess return")
    ax.set_title(f"IC by horizon, {len(ev_pre)} pre-window earnings 8-Ks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "ic_by_horizon.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for k, r in full_r.items():
        ax.plot((1 + r).cumprod(), linewidth=1.1, label=k)
    ax.axvspan(hold, end, color="grey", alpha=0.15, label="untouched window")
    ax.set_ylabel("growth of 1, 10 bps a side")
    ax.set_title("EventDrift, research path, long top fifth / short bottom fifth, daily")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(dec.index, dec["reaction"] * 100)
    axes[0].set_title("reaction-day return by sentiment decile (%)")
    axes[1].bar(dec.index, dec["drift_10"] * 100)
    axes[1].set_title("10-session excess drift from the next open by sentiment decile (%)")
    for ax in axes:
        ax.set_xlabel("FinBERT score decile (causal rank)")
        ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "sentiment_deciles.png"), dpi=140)
    plt.close(fig)

    u = result["variants"]
    print(f"\nUntouched window {hold.date()} to {end.date()}, {result['n_events']} events:")
    for k, v in u.items():
        print(f"  {k:22s} Sharpe net {v['sharpe']:.3f} gross {v['sharpe_gross']:.3f}  DSR {v['dsr']['dsr']:.3f} "
              f"over {v['dsr']['n_trials']} trials  return {v['annual_return']:.1%}  drawdown {v['max_drawdown']:.1%}")
    print("  IC on the window: " + "; ".join(f"{r['signal']} h{r['horizon']} {r['ic']:.3f} (t {r['t']:.1f})"
                                             for r in result["ic"] if r["horizon"] in (1, 10, 20)))
    print(f"Model {model_id} saved; {summary['n_trials']} trials in the registry; {summary['seconds']}s")


if __name__ == "__main__":
    m = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 9.0
    main(m)
