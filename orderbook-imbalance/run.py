"""Full pipeline on the collected OKX snapshots: regressions, binned plot, threshold strategy.

Run: python3 run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import data
from imbalance import (BPS, DEPTHS, HORIZONS_S, THRESHOLDS, backtest, binned_expectation,
                       forward_move_bps, half_spread_bps, hit_rate, imbalance, mid_price,
                       regress, summarise)

REPORTS = os.path.join(data.HERE, "reports")
FLOW_WINDOW_S = 5
TAKER_FEE_BPS = 10.0
# Written down before looking at any result: the no-search strategy.
FIXED = {"depth": 5, "threshold": 0.5, "horizon_s": 5}
STRATEGY_HORIZONS = (1, 5, 10)


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def trade_flow(trades, ts_ms, window_s):
    """Signed trade volume in the trailing window before each snapshot, scaled to [-1, 1]."""
    t = trades["ts_ms"].to_numpy()
    signed = np.where(trades["side"].to_numpy() == "buy", 1.0, -1.0) * trades["sz"].to_numpy()
    cum_signed = np.concatenate([[0.0], np.cumsum(signed)])
    cum_abs = np.concatenate([[0.0], np.cumsum(np.abs(signed))])
    hi = np.searchsorted(t, ts_ms, side="right")
    lo = np.searchsorted(t, ts_ms - window_s * 1000, side="right")
    net, gross = cum_signed[hi] - cum_signed[lo], cum_abs[hi] - cum_abs[lo]
    return np.where(gross > 0, net / np.where(gross > 0, gross, 1), 0.0)


def last_trade_price(trades, ts_ms):
    """Price of the most recent trade at or before each snapshot."""
    t, px = trades["ts_ms"].to_numpy(), trades["px"].to_numpy(dtype=float)
    i = np.searchsorted(t, ts_ms, side="right") - 1
    out = np.full(len(ts_ms), np.nan)
    out[i >= 0] = px[i[i >= 0]]
    return out


def regression_table(imbs, mid, ts, mask, interval_s):
    """Slope, t-stat, R2 and hit rate for every depth x horizon on the masked rows."""
    rows = []
    for depth, x in imbs.items():
        for h in HORIZONS_S:
            # forward moves are built inside the window, so a formation row near the
            # split never reads an evaluation price
            xm, ym = x[mask], forward_move_bps(mid[mask], ts[mask], h)
            fit = regress(xm, ym, h, interval_s)
            rows.append({"depth": depth, "horizon_s": h, "slope_bps": fit["slope"],
                         "t_stat": fit["t_stat"], "r2": fit["r2"],
                         "hit_rate": hit_rate(xm, ym), "n": fit["n"]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    book, trades = data.load()
    bid_px, bid_sz, ask_px, ask_sz = data.book_arrays(book)
    ts = book["ts_local_ms"].to_numpy()
    mid = mid_price(bid_px, ask_px)
    half = half_spread_bps(bid_px, ask_px)
    gaps = np.diff(ts) / 1000
    interval = float(np.median(gaps))
    minutes = (ts[-1] - ts[0]) / 60000
    start = pd.Timestamp(ts[0], unit="ms", tz="UTC")
    end = pd.Timestamp(ts[-1], unit="ms", tz="UTC")

    print(f"exchange {data.EXCHANGE} {data.INST}, {len(book)} snapshots over {minutes:.1f} minutes "
          f"({start:%Y-%m-%d %H:%M} to {end:%H:%M} UTC)")
    print(f"snapshot interval: median {interval:.2f}s, max gap {gaps.max():.1f}s, "
          f"gaps over 3s: {(gaps > 3).sum()}")
    print(f"trades: {len(trades)} ({trades['side'].eq('buy').mean():.1%} buyer-initiated)")
    print(f"mid: {mid[0]:.1f} -> {mid[-1]:.1f}, range {mid.min():.1f} to {mid.max():.1f}")
    print(f"half-spread: median {np.median(half):.3f} bps, mean {half.mean():.3f} bps, "
          f"tick = {0.1 / mid.mean() * BPS:.3f} bps")
    changed = np.mean(np.diff(mid) != 0)
    print(f"mid changes between consecutive snapshots {changed:.1%} of the time")

    imbs = {d: imbalance(bid_sz, ask_sz, d) for d in DEPTHS}
    for d, x in imbs.items():
        print(f"imbalance depth {d}: mean {x.mean():+.3f}, std {x.std():.3f}, "
              f"|x|>0.5 in {np.mean(np.abs(x) > 0.5):.1%} of snapshots, "
              f"lag-1 autocorr {np.corrcoef(x[:-1], x[1:])[0, 1]:.2f}")

    half_t = ts[0] + (ts[-1] - ts[0]) / 2
    formation, evaluation = ts < half_t, ts >= half_t
    print(f"\nformation: first {formation.sum()} snapshots, evaluation: last {evaluation.sum()}")

    full = regression_table(imbs, mid, ts, np.ones(len(ts), bool), interval)
    full.to_csv(os.path.join(REPORTS, "regressions.csv"), index=False)
    show("Forward mid move (bps) on imbalance, full window",
         full.assign(slope_bps=full.slope_bps.round(3), t_stat=full.t_stat.round(2),
                     r2=full.r2.round(4), hit_rate=full.hit_rate.round(3)))

    split = pd.concat([regression_table(imbs, mid, ts, formation, interval).assign(window="formation"),
                       regression_table(imbs, mid, ts, evaluation, interval).assign(window="evaluation")])
    split.to_csv(os.path.join(REPORTS, "regressions_split.csv"), index=False)
    piv = split[split.depth == 5].pivot(index="horizon_s", columns="window",
                                        values=["slope_bps", "t_stat", "r2"]).round(4)
    show("Depth-5 imbalance, formation half vs evaluation half", piv)

    # Robustness: the same regressions against the last traded price instead of the mid,
    # and with trailing signed trade flow as a competing regressor.
    last_px = last_trade_price(trades, ts)
    flow = trade_flow(trades, ts, FLOW_WINDOW_S)
    rows = []
    for h in (1, 5, 10):
        y_mid = forward_move_bps(mid, ts, h)
        y_trd = forward_move_bps(last_px, ts, h)
        a = regress(imbs[5], y_trd, h, interval)
        ok = ~np.isnan(y_mid)
        X = sm.add_constant(np.column_stack([imbs[5][ok], flow[ok]]))
        both = sm.OLS(y_mid[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, int(h / interval))})
        rows.append({"horizon_s": h, "slope_vs_trade_px": a["slope"], "t_vs_trade_px": a["t_stat"],
                     "slope_imb_with_flow": both.params[1], "t_imb_with_flow": both.tvalues[1],
                     "slope_flow": both.params[2], "t_flow": both.tvalues[2], "r2_both": both.rsquared})
    robust = pd.DataFrame(rows)
    robust.to_csv(os.path.join(REPORTS, "robustness.csv"), index=False)
    show("Robustness, depth 5: target = last trade price; and imbalance alongside "
         f"{FLOW_WINDOW_S}s trade flow", robust.round(4))

    binned = {}
    for d in DEPTHS:
        for h in (1, 5, 30):
            binned[(d, h)] = binned_expectation(imbs[d], forward_move_bps(mid, ts, h))
    pd.concat(binned, names=["depth", "horizon_s"]).to_csv(os.path.join(REPORTS, "binned.csv"))
    show("Binned conditional expectation, depth 5, 5s horizon", binned[(5, 5)].round(4))

    print("\nStrategy: trade when |imbalance| > threshold at t, fill at t+1 crossing the spread, "
          "unwind after the horizon")
    grid = []
    for d in DEPTHS:
        for thr in THRESHOLDS:
            for h in STRATEGY_HORIZONS:
                tr = backtest(imbs[d][formation], bid_px[formation], ask_px[formation],
                              ts[formation], h, thr, TAKER_FEE_BPS)
                grid.append({"depth": d, "threshold": thr, "horizon_s": h, **summarise(tr)})
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(REPORTS, "strategy_formation_grid.csv"), index=False)
    show("Formation-half grid (per-trade bps)",
         grid[["depth", "threshold", "horizon_s", "n_trades", "gross_bps", "net_half_bps",
               "net_full_bps", "hit_rate", "t_stat"]].round(3))
    live = grid[grid.n_trades >= 20]
    chosen = live.loc[live.total_net_half_bps.idxmax()]
    chosen = {"depth": int(chosen.depth), "threshold": float(chosen.threshold),
              "horizon_s": int(chosen.horizon_s)}
    print(f"\nchosen on formation by total net-of-half-spread P&L: {chosen}")
    print(f"fixed in advance: {FIXED}")

    results = {}
    for name, cfg in (("fixed", FIXED), ("chosen", chosen)):
        for wname, mask in (("formation", formation), ("evaluation", evaluation)):
            tr = backtest(imbs[cfg["depth"]][mask], bid_px[mask], ask_px[mask], ts[mask],
                          cfg["horizon_s"], cfg["threshold"], TAKER_FEE_BPS)
            results[(name, wname)] = summarise(tr)
            if (name, wname) == ("chosen", "evaluation"):
                eval_trades = tr
    results = pd.DataFrame(results).T
    results.to_csv(os.path.join(REPORTS, "strategy.csv"))
    show("Strategy results (per-trade bps; net_full includes the "
         f"{TAKER_FEE_BPS:.0f} bps OKX taker fee each way)", results.round(3))

    eval_half = half[evaluation].mean()
    ev = results.loc[("chosen", "evaluation")]
    print(f"\nevaluation half: mean half-spread {eval_half:.3f} bps, expected gross move per trade "
          f"{ev['gross_bps']:+.3f} bps, ratio {ev['gross_bps'] / eval_half:+.2f}x")

    charts(binned, full, eval_trades, chosen, mid, imbs[5], ts)
    print(f"\ncharts and tables in {REPORTS}")


def charts(binned, full, eval_trades, chosen, mid, imb5, ts):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, h in zip(axes, (1, 5, 30)):
        for d in DEPTHS:
            b = binned[(d, h)]
            ax.errorbar(b["imbalance"], b["move_bps"], yerr=b["se_bps"], marker="o",
                        capsize=2, label=f"depth {d}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"next {h}s mid move vs imbalance")
        ax.set_xlabel("imbalance (equal-count bins)")
        ax.set_ylabel("mean forward mid move, bps")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "binned.png"), dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for d in DEPTHS:
        sub = full[full.depth == d]
        axes[0].plot(sub.horizon_s, sub.slope_bps, marker="o", label=f"depth {d}")
        axes[1].plot(sub.horizon_s, sub.r2, marker="o", label=f"depth {d}")
    for ax, title in zip(axes, ("slope, bps per unit imbalance", "R2")):
        ax.set_xscale("log")
        ax.set_xlabel("horizon, seconds")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "slope_by_horizon.png"), dpi=120)

    if not eval_trades.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(1, len(eval_trades) + 1)
        for col, lab in (("gross_bps", "gross (mid to mid)"),
                         ("net_half_bps", "net of half-spread"),
                         ("net_full_bps", "net of full spread and taker fees")):
            ax.plot(x, eval_trades[col].cumsum(), label=lab)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("trade number, evaluation half")
        ax.set_ylabel("cumulative bps")
        ax.set_title(f"depth {chosen['depth']}, |imbalance| > {chosen['threshold']}, "
                     f"{chosen['horizon_s']}s hold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(REPORTS, "strategy_pnl.png"), dpi=120)

    n = min(600, len(ts))
    t = (ts[:n] - ts[0]) / 1000
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, mid[:n], color="k", lw=0.8)
    ax1.set_ylabel("mid")
    ax2 = ax1.twinx()
    ax2.fill_between(t, imb5[:n], 0, alpha=0.3, color="tab:blue")
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("depth-5 imbalance")
    ax1.set_xlabel("seconds from start")
    ax1.set_title("first ten minutes: mid and imbalance")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "mid_and_imbalance.png"), dpi=120)


if __name__ == "__main__":
    main()
