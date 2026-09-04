"""Full pipeline on the sampled quotes: panel, gaps, persistence, fee breakeven, simulations.

Run: python3 run.py        (needs quotes.csv from data.py and the cached OHLC history)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from arb import (PRO_BPS, TAKER_BPS, TRANSFER, align, best_gap, close_gap_bps, gap_persistence,
                 half_life, mid_spread_bps, simulate)

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS, exist_ok=True)
LOG = open(os.path.join(REPORTS, "run_log.txt"), "w")
THRESHOLDS = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]
MIN_EDGE_GRID = [0.0, 1.0, 2.0, 5.0]
DT = data.INTERVAL_S


def log(*parts):
    s = " ".join(str(p) for p in parts)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def show(title, frame):
    log(f"\n{title}")
    log("-" * len(title))
    log(frame.to_string())


def describe_gap(bps):
    return pd.Series({"median": bps.median(), "p95": bps.quantile(0.95), "p99": bps.quantile(0.99),
                      "max": bps.max(), "share>0": (bps > 0).mean(),
                      "share>pro_rt(20)": (bps > 20).mean(),
                      "share>retail_rt(min 50)": (bps > 50).mean()})


quotes = data.load_quotes()
span = quotes["t_local"].max() - quotes["t_local"].min()
log(f"window: {quotes['t_local'].min():%Y-%m-%d %H:%M:%S}Z to {quotes['t_local'].max():%H:%M:%S}Z "
    f"({span.total_seconds() / 60:.1f} min), {quotes['tick'].nunique()} ticks, {len(quotes)} quotes")
log(f"venues: {sorted(quotes['exchange'].unique())}  (binance.com HTTP 451 and bybit 403 from the US)")
expected = quotes["tick"].nunique() * 2
show("quotes received per venue (expected %d) and request latency" % expected,
     quotes.groupby("exchange").agg(quotes=("bid", "size"), latency_med_ms=("latency_ms", "median"),
                                    latency_p95_ms=("latency_ms", lambda s: s.quantile(.95))).assign(
         missing=lambda d: expected - d["quotes"]))
skew = quotes.groupby("tick")["t_local"].agg(lambda s: (s.max() - s.min()).total_seconds() * 1000)
log(f"within-tick timestamp skew across venues: median {skew.median():.0f}ms, p95 {skew.quantile(.95):.0f}ms, "
    f"max {skew.max():.0f}ms")
usdt = quotes["usdt_usd"].dropna()
log(f"USDT/USD used for OKX conversion: {usdt.min():.5f} to {usdt.max():.5f}")

panels = align(quotes, freq=f"{int(DT)}s", tolerance=f"{int(2 * DT)}s")
summary_rows = []
sim_rows = []
for sym, panel in panels.items():
    log(f"\n==== {sym}: {len(panel)} aligned samples, {panel.columns.levels[0].size} venues")
    ms = mid_spread_bps(panel)
    show(f"{sym} mid-to-mid spread by pair, bps",
         pd.DataFrame({"mean": ms.mean(), "median_abs": ms.abs().median(), "p95_abs": ms.abs().quantile(.95),
                       "share_abs>5": (ms.abs() > 5).mean(), "share_abs>10": (ms.abs() > 10).mean()}).round(2))
    bg = best_gap(panel)
    stats = describe_gap(bg["bps"])
    show(f"{sym} best executable gap per sample (sell bid / buy ask), bps", stats.round(3).to_frame("value"))
    show(f"{sym} which pair carries the best gap", bg["pair"].value_counts().head(6))
    pers = gap_persistence(bg["bps"], THRESHOLDS, DT)
    show(f"{sym} persistence of the best executable gap", pers.round(3))
    pers.assign(symbol=sym).to_csv(os.path.join(REPORTS, f"persistence_{sym}.csv"), index=False)
    bg.to_csv(os.path.join(REPORTS, f"best_gap_{sym}.csv"))
    panel.to_csv(os.path.join(REPORTS, f"panel_{sym}.csv"))

    fees = np.r_[0.0, 0.5, 1.0, 1.5, 2.0, np.arange(2.5, 61, 2.5)]
    breakeven = pd.DataFrame({"fee_per_leg_bps": fees,
                              "share_of_samples_profitable": [(bg["bps"] > 2 * f).mean() for f in fees]})
    breakeven.to_csv(os.path.join(REPORTS, f"breakeven_{sym}.csv"), index=False)
    show(f"{sym} share of samples where the best gap beats a flat per-leg fee",
         breakeven[breakeven["fee_per_leg_bps"].isin([0, 0.5, 1, 1.5, 2, 2.5, 5, 10, 20, 40, 60])].round(4))

    # Formation / evaluation split: pick min_edge on the first half, trade the second.
    half = len(panel) // 2
    form, evalp = panel.iloc[:half], panel.iloc[half:]
    grid = pd.DataFrame([dict(min_edge_bps=e, **{k: v for k, v in
                              simulate(form, PRO_BPS, min_edge_bps=e)[1].items() if k != "mode"})
                         for e in MIN_EDGE_GRID])
    show(f"{sym} formation half: min_edge grid at pro fees (10bps/leg), prepositioned", grid.round(3))
    best_edge = grid.sort_values(["net_usd_total", "min_edge_bps"], ascending=[False, True])["min_edge_bps"].iloc[0]
    log(f"{sym} chosen min_edge = {best_edge} bps (best formation net); now applied once to the second half")

    scenarios = {
        "eval_pro_prepositioned": dict(panel=evalp, fees=PRO_BPS, min_edge_bps=best_edge),
        "eval_retail_prepositioned": dict(panel=evalp, fees=TAKER_BPS, min_edge_bps=best_edge),
        "eval_pro_transfer": dict(panel=evalp, fees=PRO_BPS, min_edge_bps=best_edge, mode="transfer",
                                  transfer_s=TRANSFER[sym]["seconds"], withdraw_fee=TRANSFER[sym]["withdraw_fee"]),
        "full_zero_fee_prepositioned": dict(panel=panel, fees={k: 0.0 for k in PRO_BPS}, min_edge_bps=0.0),
        "full_pro_prepositioned": dict(panel=panel, fees=PRO_BPS, min_edge_bps=0.0),
        "full_retail_prepositioned": dict(panel=panel, fees=TAKER_BPS, min_edge_bps=0.0),
    }
    for name, kw in scenarios.items():
        p = kw.pop("panel")
        f = kw.pop("fees")
        tlog, s = simulate(p, f, **kw)
        s.update(symbol=sym, scenario=name, samples=len(p))
        sim_rows.append(s)
        if len(tlog):
            tlog.to_csv(os.path.join(REPORTS, f"trades_{sym}_{name}.csv"), index=False)
    summary_rows.append(dict(symbol=sym, samples=len(panel), **stats.to_dict(),
                             episodes_gt20=int(pers.loc[pers["threshold_bps"] == 20, "episodes"].iloc[0]),
                             median_s_gt20=float(pers.loc[pers["threshold_bps"] == 20, "median_s"].iloc[0])))

    # Charts: mids relative to the cross-venue average, the best gap with fee lines, and its histogram.
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    mids = panel.xs("mid", axis=1, level=1)
    rel = (mids.sub(mids.mean(axis=1), axis=0)).div(mids.mean(axis=1), axis=0) * 1e4
    rel.plot(ax=ax[0], lw=0.8)
    ax[0].set_title(f"{sym}: each venue's mid vs the cross-venue average, bps")
    ax[0].axhline(0, color="k", lw=0.5)
    bg["bps"].plot(ax=ax[1], lw=0.8, color="C3")
    ax[1].axhline(20, color="C2", ls="--", lw=0.8, label="pro round trip 20bps")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title(f"{sym}: best executable gap (sell bid minus buy ask), bps")
    ax[1].legend(loc="upper right")
    ax[2].hist(bg["bps"], bins=80, color="C0")
    ax[2].axvline(20, color="C2", ls="--", lw=0.8)
    ax[2].set_title(f"{sym}: distribution of the best executable gap, bps")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, f"gaps_{sym}.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(breakeven["fee_per_leg_bps"], breakeven["share_of_samples_profitable"] * 100)
    ax.set_xlabel("taker fee per leg, bps")
    ax.set_ylabel("% of samples with a gap above the round trip")
    ax.set_title(f"{sym}: how much of the time an arb exists, by fee level")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, f"breakeven_{sym}.png"), dpi=110)
    plt.close(fig)

summary = pd.DataFrame(summary_rows).set_index("symbol")
summary.to_csv(os.path.join(REPORTS, "gap_summary.csv"))
sims = pd.DataFrame(sim_rows).set_index(["symbol", "scenario"])
sims.to_csv(os.path.join(REPORTS, "simulations.csv"))
show("simulation summary ($10k lots, 5 lots per side, signal t / fill t+1)", sims.round(3))

# Longer horizon: Coinbase vs Kraken candle closes, hourly for 30 days and daily for 2 years.
log("\n==== long horizon: Coinbase minus Kraken close, bps")
hist_rows = []
fig, ax = plt.subplots(2, 1, figsize=(11, 6))
for i, freq in enumerate(["hourly", "daily"]):
    for sym in ["BTC", "ETH"]:
        g = close_gap_bps(data.load_history(sym, freq))
        hist_rows.append(dict(freq=freq, symbol=sym, n=len(g), start=g.index.min().date(),
                              end=g.index.max().date(), mean=g.mean(), median_abs=g.abs().median(),
                              p95_abs=g.abs().quantile(.95), max_abs=g.abs().max(),
                              share_abs_gt20=(g.abs() > 20).mean(), ar1=g.autocorr(1),
                              half_life_periods=half_life(g)))
        g.to_csv(os.path.join(REPORTS, f"close_gap_{sym}_{freq}.csv"))
        if freq == "hourly":
            ax[i].plot(g.index, g, lw=0.6, label=sym)
        else:
            ax[i].plot(g.index, g, lw=0.6, label=sym)
    ax[i].axhline(0, color="k", lw=0.5)
    ax[i].set_title(f"{freq} close gap, Coinbase minus Kraken, bps")
    ax[i].legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "close_gap_history.png"), dpi=110)
plt.close(fig)
hist = pd.DataFrame(hist_rows)
hist.to_csv(os.path.join(REPORTS, "close_gap_summary.csv"), index=False)
show("candle-close gap statistics (closes are not synchronous across venues, see README)", hist.round(3))
LOG.close()
