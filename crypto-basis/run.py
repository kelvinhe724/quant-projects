"""Basis distribution, the walk-forward on the mean-reversion rule, the untouched window, the venue monitor.

Run: ../.venv/bin/python run.py     (about 15 minutes; the data must already be in the lake, see data.py)
"""
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from research.alpha.base import Untouched, backtest, positions, walk_forward, window_returns  # noqa: E402
from research.features.store import FeatureStore, Raw, panel_hash  # noqa: E402
from research.models.registry import ModelRegistry  # noqa: E402
from research.registry.experiments import Registry  # noqa: E402

import data  # noqa: E402
from basis import (BAR, BARS_PER_DAY, COST_BPS, FEES, BasisMR, basis, daily, daily_sharpe, dislocation,  # noqa: E402
                   exceedance, gap_runs, reversion, tr_index, venue_gap, zscore_feature)

REPORTS = os.path.join(HERE, "reports")
START, END = "2024-01-01", "2026-07-31 23:59:59"
# frozen before any result: windows in 5-minute bars (1h, 4h, 1d), entry z, baseline first
GRID = [{"window": w, "entry": z} for w in (288, 48, 12) for z in (2.0, 3.0)]
BASELINE = GRID[0]
THRESHOLDS = {"taker": 2 * FEES["taker"], "maker": 2 * FEES["maker"], "vip": 2 * FEES["vip"]}
# cross-venue: two taker legs on the same product, 1 bp slippage each; inventory already on both venues
CROSS = {"spot": 2 * (10 + 1), "perp": 2 * (5 + 1)}
t0 = time.time()


def say(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    spot, perp, funding = data.bars(START, END, BAR)
    # the harness's lock and walk-forward speak naive timestamps; everything here is UTC wall time
    for f in (*spot.values(), *perp.values(), funding):
        f.index = f.index.tz_convert(None)
    P, b = tr_index(spot["close"], perp["close"], funding), basis(spot["close"], perp["close"])
    cal = P.index
    say(f"{len(cal)} bars, {list(P.columns)}, {cal[0]} to {cal[-1]}")

    raw = Raw({"close": P, "basis": b, "open": spot["open"], "high": spot["high"], "low": spot["low"],
               "volume": spot["volume"]})
    store = FeatureStore([zscore_feature(g["window"]) for g in GRID[::2]])
    panel = store.build(raw, audit=True)
    say("features built, peek audit passed")

    lock = Untouched(os.path.join(REPORTS, "untouched.json"))
    lock.lock(cal, 0.2, panel_hash(P))
    w0, w1 = lock.window
    pre = cal[cal < w0]
    say(f"untouched window {w0} to {w1}, {int((cal >= w0).sum())} bars, opened={lock.opened}")

    # basis distribution and how often a dislocation clears a round trip
    bps = b * 1e4
    dist = pd.DataFrame({"mean": bps.mean(), "median": bps.median(), "std": bps.std(),
                         "p1": bps.quantile(0.01), "p99": bps.quantile(0.99), "min": bps.min(), "max": bps.max(),
                         "share_negative": (b < 0).mean(), "autocorr_1h": [b[c].autocorr(12) for c in b],
                         "autocorr_1d": [b[c].autocorr(BARS_PER_DAY) for c in b]}).T
    dist.to_csv(os.path.join(REPORTS, "basis_distribution.csv"))
    by_year = pd.concat({"mean": bps.groupby(bps.index.year).mean(), "std": bps.groupby(bps.index.year).std()}, axis=1)
    by_year.to_csv(os.path.join(REPORTS, "basis_by_year.csv"))
    d = dislocation(b)
    exc = exceedance(d, sorted({1, 2, 5, 10, 20, 50, *THRESHOLDS.values()}))
    exc.to_csv(os.path.join(REPORTS, "exceedance.csv"))
    rev = reversion(d, 10, [1, 12, 48, BARS_PER_DAY])
    rev.to_csv(os.path.join(REPORTS, "reversion.csv"))
    fund_bps = funding * 1e4
    say("basis tables written")

    # walk-forward on the pre-window bars, six variants, four folds; folds scored in the registry unit (daily Sharpe)
    reg = Registry(os.path.join(REPORTS, "registry"))
    close_pre, panel_pre = P.loc[pre], panel.loc[pre]
    table, oos, full = walk_forward(lambda **k: BasisMR(**k), GRID, panel_pre, close_pre, n_splits=4,
                                    test_size=len(pre) // 5, rebalance="daily", horizon=12, cost_bps=COST_BPS,
                                    score=daily_sharpe)
    table.to_csv(os.path.join(REPORTS, "walk_forward.csv"))
    window = (pre[0].date(), pre[-1].date())
    entries = {}
    for params in GRID:
        r = daily(full[json.dumps(params)])
        entries[json.dumps(params)] = reg.record(str(BasisMR(**params)), params, list(P.columns), window, r,
                                                 tags={"stage": "walk_forward_grid", "cost_bps": COST_BPS, "bar": BAR})
    oos_d = daily(oos)
    reg.record("BasisMR[walk_forward_oos]", {"grid": GRID, "n_splits": 4}, list(P.columns), window, oos_d,
               tags={"stage": "walk_forward_oos", "cost_bps": COST_BPS, "bar": BAR})
    say(f"walk-forward done, {reg.trials()} trials logged")
    picks = table["picked"].value_counts()
    chosen = json.loads(picks.index[0]) if picks.iloc[0] > picks.get(json.dumps(BASELINE), 0) else BASELINE
    pre_daily = daily(full[json.dumps(chosen)])
    pre_dsr = reg.dsr(pre_daily)
    pos_pre = positions(BasisMR(**chosen), panel_pre, pre)
    trades_pre = pos_pre.diff().abs().sum().sum() / 2
    gross_pre = reg.dsr(daily(backtest(pos_pre, close_pre, cost_bps=0.0)))
    # what one round trip captured before fees: total gross P&L per unit over the number of trades
    gross_per_trade_bps = float((1 + backtest(pos_pre, close_pre, cost_bps=0.0)).prod() - 1) / trades_pre * 1e4

    # the one look at the untouched window: chosen variant, three fee tiers in the same call
    def read(start, end):
        out = {}
        alpha = BasisMR(**chosen)
        pos = positions(alpha, panel, cal[(cal >= start) & (cal <= end)])
        for tier, fee in FEES.items():
            r = daily(window_returns(BasisMR(**chosen), panel, P, start, end, cal, cost_bps=fee))
            out[tier] = {**reg.dsr(r), "annual_return_365": float((1 + r).prod() ** (365 / len(r)) - 1),
                         "max_drawdown": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())}
            if tier == "taker":
                r.to_csv(os.path.join(REPORTS, "untouched_returns.csv"))
        out["trades"] = float(pos.diff().abs().sum().sum() / 2)
        out["bars_in_position"] = float((pos != 0).mean().mean())
        out["chosen"] = chosen
        return out
    if lock.opened:
        untouched = lock.result
        say("untouched window already spent, reporting the stored result")
    else:
        untouched = lock.open(read)
        r_u = pd.read_csv(os.path.join(REPORTS, "untouched_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        reg.record(str(BasisMR(**chosen)) + "[untouched]", chosen, list(P.columns), (w0.date(), w1.date()), r_u,
                   tags={"stage": "untouched", "cost_bps": COST_BPS, "bar": BAR})
        say("untouched window opened once")

    models = ModelRegistry(os.path.join(REPORTS, "models"))
    mid = models.save("BasisMR", BasisMR(**chosen), panel_pre, window, untouched["taker"]["sharpe"],
                      meta={**chosen, "exit": 0.5, "lags": store.lags, "cost_bps": COST_BPS, "bar": BAR})

    # cross-venue monitor on July 2026 minutes
    okx = data.okx_bars("2026-07-01", "2026-07-31 23:59:59")
    bs = data.wide(data.lake.load("crypto_klines_1m", "2026-07-01", "2026-07-31 23:59:59"), "close", "1min")
    bp = data.wide(data.lake.load("crypto_perp_klines_1m", "2026-07-01", "2026-07-31 23:59:59"), "close", "1min")
    monitor, gaps = {}, {}
    for f in (*okx.values(), bs, bp):
        f.index = f.index.tz_convert(None)
    for coin in ("BTC", "ETH"):
        pairs = {f"{coin} spot: OKX/Binance": (okx["last"][f"{coin}-USDT"], bs[f"{coin}USDT"], CROSS["spot"]),
                 f"{coin} perp: OKX/Binance": (okx["last"][f"{coin}-USDT-SWAP"], bp[f"{coin}USDT"], CROSS["perp"]),
                 f"{coin} OKX basis": (okx["last"][f"{coin}-USDT-SWAP"], okx["last"][f"{coin}-USDT"], THRESHOLDS["taker"]),
                 f"{coin} Binance basis": (bp[f"{coin}USDT"], bs[f"{coin}USDT"], THRESHOLDS["taker"])}
        for name, (a, c, thr) in pairs.items():
            g = venue_gap(a, c)
            gaps[name] = g
            gb = g * 1e4
            runs = gap_runs(g, thr)
            monitor[name] = {"minutes": int(len(g)), "mean_bps": float(gb.mean()), "std_bps": float(gb.std()),
                             "p1_bps": float(gb.quantile(0.01)), "p99_bps": float(gb.quantile(0.99)),
                             "abs_p99_bps": float(gb.abs().quantile(0.99)), "autocorr_1m": float(g.autocorr(1)),
                             "round_trip_bps": thr, "share_over_cost": float((gb.abs() > thr).mean()),
                             "runs_over_cost": int(len(runs)),
                             "median_run_min": float(np.median(runs)) if len(runs) else 0.0,
                             "share_over_10bps": float((gb.abs() > 10).mean())}
    mon = pd.DataFrame(monitor).T
    mon.to_csv(os.path.join(REPORTS, "venue_monitor.csv"))
    say("venue monitor done")

    # charts
    fig, ax = plt.subplots(3, 1, figsize=(12, 11))
    daily_b = bps.resample("1D").mean()
    for c in daily_b:
        ax[0].plot(daily_b.index, daily_b[c], lw=0.8, label=c)
    ax[0].axvspan(w0, w1, color="grey", alpha=0.15, label="untouched")
    ax[0].set_title("Binance perp-spot basis, daily mean of 5-minute bars (bps)")
    ax[0].legend()
    for k, r in full.items():
        e = (1 + daily(r)).cumprod()
        ax[1].plot(e.index, e, lw=0.8, label=k)
    eu = (1 + pd.read_csv(os.path.join(REPORTS, "untouched_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]).cumprod()
    ax[1].plot(eu.index, eu, lw=1.4, color="k", label="chosen, untouched window")
    ax[1].axvspan(w0, w1, color="grey", alpha=0.15)
    ax[1].set_title(f"Equity of one unit, {COST_BPS} bps a side, pre-window variants and the chosen one on the window")
    ax[1].legend(fontsize=7)
    for name in ("BTC spot: OKX/Binance", "BTC perp: OKX/Binance", "BTC OKX basis", "BTC Binance basis"):
        ax[2].plot(gaps[name].index, gaps[name] * 1e4, lw=0.5, label=name)
    ax[2].set_title("July 2026, 1-minute venue gaps (bps)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "basis.png"), dpi=120)

    summary = {
        "bars": int(len(cal)), "bar": BAR, "start": str(cal[0]), "end": str(cal[-1]), "close_hash": panel_hash(P),
        "basis_distribution_bps": dist.to_dict(), "basis_by_year_bps": {str(k): v for k, v in by_year.to_dict().items()},
        "funding_8h_bps": {"mean": fund_bps.mean().to_dict(), "share_negative": (funding < 0).mean().to_dict()},
        "exceedance": exc.to_dict(), "reversion_10bps": rev.to_dict(), "thresholds_bps": {**THRESHOLDS, "cross_venue": CROSS},
        "walk_forward": table.reset_index().to_dict("records"),
        "walk_forward_oos_sharpe": float(reg.runs().set_index("name").loc["BasisMR[walk_forward_oos]", "metrics"]["sharpe"]),
        "grid_pre_window": {k: v["metrics"] for k, v in entries.items()},
        "chosen": chosen, "pre_window": {**pre_dsr, "trades": float(trades_pre)},
        "pre_window_gross": {**gross_pre, "per_trade_bps": gross_per_trade_bps},
        "untouched": untouched, "untouched_window": [str(w0), str(w1)],
        "registry_entry": reg.runs().set_index("name").loc[str(BasisMR(**chosen)), "id"],
        "n_trials": int(reg.trials()), "model_id": mid, "venue_monitor": monitor,
    }
    json.dump(summary, open(os.path.join(REPORTS, "summary.json"), "w"), indent=2, default=str)
    say("done")
    print(json.dumps({k: summary[k] for k in ("chosen", "pre_window", "pre_window_gross", "untouched", "n_trials", "registry_entry",
                                              "walk_forward_oos_sharpe")}, indent=1, default=str))
    print(exc.round(4))
    print(rev.round(3))
    print(mon[["mean_bps", "std_bps", "abs_p99_bps", "round_trip_bps", "share_over_cost", "runs_over_cost",
               "median_run_min"]].round(4))


if __name__ == "__main__":
    main()
