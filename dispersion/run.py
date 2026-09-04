"""Full pipeline on real data: today's implied correlation from the chains, the
historical implied-minus-realised gap, and the simulated dispersion book.

Run: python3 run.py
Everything printed is also written to reports/run_log.txt.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from dispersion import (HEDGE_COST, MONTH, SPREAD_INDEX, SPREAD_SINGLE, atm_vol,
                        average_pairwise_correlation, implied_correlation, metrics,
                        realised_correlation, realised_vol, simulate, vol_at_horizon)

REPORTS = data.REPORTS
FORMATION = (data.START, data.FORMATION_END)
TEST = (data.TEST_START, data.END)


class Tee:
    def __init__(self, path):
        self.file, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.file.write(s)
        self.out.write(s)

    def flush(self):
        self.file.flush()
        self.out.flush()


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def fmt(m):
    return pd.Series({
        "months": m["months"],
        "mean (vol pts/month)": f"{m['mean']:+.2f}",
        "std": f"{m['std']:.2f}",
        "Sharpe": f"{m['sharpe']:.2f}",
        "total": f"{m['total']:+.1f}",
        "hit rate": f"{m['hit_rate']:.0%}",
        "skew": f"{m['skew']:+.2f}",
        "5th pct month": f"{m['p5']:+.2f}",
        "1st pct month": f"{m['p1']:+.2f}",
        "worst month": f"{m['worst_month']:+.2f} ({m['worst_month_date']:%Y-%m-%d})",
        "max drawdown": f"{m['max_drawdown']:+.2f}",
        "drawdown peak": f"{m['dd_peak']:%Y-%m-%d}",
        "drawdown trough": f"{m['dd_trough']:%Y-%m-%d}",
    })


def month_ends(index, start, end):
    days = pd.Series(index, index=index)
    out = pd.DatetimeIndex(sorted(days.groupby([index.year, index.month]).max()))
    return out[(out >= start) & (out <= end)]


def main():
    px = data.get_daily()
    weights = data.get_weights()
    rets = data.log_returns(px)
    single, index = rets[data.NAMES], rets[data.INDEX]
    vix = px[data.VIX].reindex(rets.index).ffill()
    vix_last = px[data.VIX].dropna().iloc[-1]
    last = rets.index[-1]
    print(f"prices {px.index[0].date()} to {px.index[-1].date()}, {len(rets)} return days")
    print(f"basket: {', '.join(data.NAMES)}")
    print(f"weights (cap within basket): "
          + ", ".join(f"{t} {w:.1%}" for t, w in weights.items()))

    print("\n" + "=" * 78)
    print("TODAY: IMPLIED CORRELATION FROM THE OPTION CHAINS")
    print("=" * 78)

    chains = data.get_chains()
    if chains is None:
        print("no option chains available; the current implied section is skipped")
        implied30, premium, wedge = None, pd.Series(1.0, index=data.NAMES), 0.0
    else:
        implied30, premium, wedge = current(chains, single, index, weights, vix_last, last)

    print("\n" + "=" * 78)
    print("HISTORY: IMPLIED MINUS REALISED CORRELATION, 2015 ONWARD")
    print("=" * 78)
    gap = history(single, index, weights, vix, premium, wedge)

    print("\n" + "=" * 78)
    print("SIMULATED DISPERSION BOOK")
    print("=" * 78)
    books = simulation(single, index, weights, vix, premium, wedge, gap)

    charts(gap, books, implied30)
    print(f"\ncharts and tables written to {REPORTS}/")


def current(chains, single, index, weights, vix_last, last):
    """Back out ATM vols per name, interpolate to 30 and 60 days, solve for rho."""
    snapshot = chains["snapshot"].iloc[0]
    print(f"chain snapshot {snapshot:%Y-%m-%d %H:%M} UTC, risk-free {data.RISK_FREE:.0%}")
    have = list(chains["ticker"].unique())
    missing = [t for t in [data.INDEX] + data.NAMES if t not in have]
    if missing:
        print(f"no chain for {', '.join(missing)}; implied correlation uses the names present")
    names = [t for t in data.NAMES if t in have]

    per_expiry, at30, at60 = {}, {}, {}
    for t in [data.INDEX] + names:
        atm = atm_vol(chains[chains["ticker"] == t], data.RISK_FREE)
        per_expiry[t] = atm["atm_iv"]
        at30[t] = vol_at_horizon(atm, 30)
        at60[t] = vol_at_horizon(atm, 60)
    surface = pd.DataFrame(per_expiry)
    surface.index = surface.index.strftime("%Y-%m-%d")
    show("ATM implied vol by expiry (call/put average at the strike nearest the forward)",
         surface.map(lambda v: f"{v:.1%}" if pd.notna(v) else ""))
    surface.to_csv(os.path.join(REPORTS, "atm_by_expiry.csv"))

    rv21 = realised_vol(single, 21).loc[last]
    rv63 = realised_vol(single, 63).loc[last]
    table = pd.DataFrame({
        "weight": weights,
        "implied 30d": pd.Series(at30),
        "implied 60d": pd.Series(at60),
        "realised 21d": rv21,
        "realised 63d": rv63,
    }).loc[names]
    table.loc[data.INDEX] = [np.nan, at30[data.INDEX], at60[data.INDEX],
                             realised_vol(index, 21).loc[last], realised_vol(index, 63).loc[last]]
    table["implied/realised 21d"] = table["implied 30d"] / table["realised 21d"]
    show(f"30-day implied vs trailing realised vol as of {last.date()}",
         table.assign(**{c: table[c].map("{:.1%}".format) for c in table.columns[:5]},
                      **{"implied/realised 21d": table["implied/realised 21d"].map("{:.2f}".format)}))
    table.to_csv(os.path.join(REPORTS, "current_vols.csv"))

    w = weights.loc[names] / weights.loc[names].sum()
    rho30 = implied_correlation(at30[data.INDEX], table.loc[names, "implied 30d"], w)
    rho60 = implied_correlation(at60[data.INDEX], table.loc[names, "implied 60d"], w)
    rc = {win: realised_correlation(single[names], index, w, win).loc[last]
          for win in (21, 63, 252)}
    apc = {win: average_pairwise_correlation(single[names], win).loc[last]
           for win in (21, 63, 252)}

    print(f"\nimplied correlation, 30 day: {rho30:.3f}")
    print(f"implied correlation, 60 day: {rho60:.3f}")
    wedge = vix_last - at30[data.INDEX] * 100
    print(f"VIX close on {vix_last:.2f} (SPY chain 30d ATM: {at30[data.INDEX]:.1%}, "
          f"wedge {wedge:.1f} vol pts: VIX is a variance-swap strip including the skew, "
          f"an ATM straddle is not)")
    for win in (21, 63, 252):
        print(f"realised correlation, trailing {win:>3}d: weighted {rc[win]:.3f}, "
              f"average pairwise {apc[win]:.3f}")
    print(f"implied minus realised (30d vs 21d): {rho30 - rc[21]:+.3f}; "
          f"(30d vs 63d): {rho30 - rc[63]:+.3f}")

    cboe = data.get_cboe_correlation()
    if cboe is None or cboe.empty:
        print("Cboe COR1M/COR3M: not available from Yahoo")
    else:
        row = cboe.iloc[-1]
        print(f"Cboe implied correlation on {cboe.index[-1].date()}: "
              f"COR1M {row.get('^COR1M', np.nan):.1f}, COR3M {row.get('^COR3M', np.nan):.1f} "
              f"(S&P 500 top-50 basket, Cboe methodology; mine is a 10-name basket)")

    pd.DataFrame({"horizon": ["30d", "60d"], "implied_corr": [rho30, rho60],
                  "realised_21d": rc[21], "realised_63d": rc[63], "realised_252d": rc[252],
                  "pairwise_21d": apc[21], "pairwise_63d": apc[63]}).to_csv(
        os.path.join(REPORTS, "implied_corr_now.csv"), index=False)

    # One pooled premium ratio rather than ten: a single day's ratio per name is
    # dominated by whichever names just had earnings, which the table above shows.
    k = float((w * table.loc[names, "implied/realised 21d"]).sum())
    print(f"pooled single-name premium ratio (cap-weighted implied 30d / realised 21d): {k:.2f}")
    return table, pd.Series(k, index=data.NAMES), wedge


def history(single, index, weights, vix, premium, wedge):
    """Build the daily implied-correlation proxy and the gap to trailing realised."""
    print("""
Single-name option prices are not available historically, so the daily implied
correlation below is a proxy: the index side is VIX, the single-name side is
each name's trailing 21-day realised vol scaled by a constant premium ratio.
Two versions are shown. `k=1` uses raw VIX and puts no premium on single names,
so every bit of the VIX premium over realised is attributed to correlation,
which is an upper bound. `k=snapshot` subtracts today's VIX-to-ATM wedge from
VIX and applies today's pooled single-name implied/realised ratio backward,
which is the best fixed estimate available but assumes both constants held for
eleven years. Neither is the number a desk would have seen.""")
    rv21 = realised_vol(single, 21)
    out = pd.DataFrame(index=single.index)
    out["vix"] = vix
    for label, k, c in (("k=1", pd.Series(1.0, index=data.NAMES), 0.0),
                        ("k=snapshot", premium, wedge)):
        vols = rv21.mul(k, axis=1)
        ws = vols * weights.values
        own = (ws ** 2).sum(axis=1)
        cross = ws.sum(axis=1) ** 2 - own
        out[f"implied {label}"] = (((vix - c) / 100) ** 2 - own) / cross
    for win in (21, 63):
        out[f"realised {win}d"] = realised_correlation(single, index, weights, win)
        out[f"pairwise {win}d"] = average_pairwise_correlation(single, win)
    out["gap k=1 vs 21d"] = out["implied k=1"] - out["realised 21d"]
    out["gap k=snapshot vs 21d"] = out["implied k=snapshot"] - out["realised 21d"]
    out["gap k=snapshot vs 63d"] = out["implied k=snapshot"] - out["realised 63d"]
    out = out.dropna()
    out.to_csv(os.path.join(REPORTS, "gap_history.csv"))

    print(f"\nk=snapshot constants: premium ratio {premium.iloc[0]:.2f} on every name, "
          f"wedge {wedge:.1f} vol pts off VIX")

    cols = ["implied k=1", "implied k=snapshot", "realised 21d", "realised 63d", "pairwise 21d",
            "gap k=1 vs 21d", "gap k=snapshot vs 21d", "gap k=snapshot vs 63d"]
    rows = {}
    for label, span in (("full", (data.START, data.END)), ("formation", FORMATION),
                        ("test", TEST)):
        sub = out.loc[span[0]:span[1], cols]
        rows[label] = pd.concat([sub.mean().rename("mean"), sub.median().rename("median"),
                                 sub.std().rename("std"), (sub > 0).mean().rename("share > 0")],
                                axis=1)
    stats = pd.concat(rows, axis=0)
    show("Implied correlation proxy, realised correlation and the gap: daily statistics",
         stats.round(3))
    stats.to_csv(os.path.join(REPORTS, "gap_stats.csv"))

    yearly = out.groupby(out.index.year)[["implied k=snapshot", "realised 21d",
                                          "gap k=snapshot vs 21d"]].mean()
    show("By year", yearly.round(3))

    worst = out["gap k=snapshot vs 21d"].nsmallest(5)
    show("Five most negative gap days (realised correlation far above implied)",
         out.loc[worst.index, ["vix", "implied k=snapshot", "realised 21d",
                                "gap k=snapshot vs 21d"]].round(3))
    return out


def simulation(single, index, weights, vix, premium, wedge, gap):
    """Run the monthly book always-on, conditioned, and under alternative weights."""
    decisions = month_ends(single.index, data.START, data.END)
    print(f"\nmonthly cycle: decide at the month-end close, fill at the next close, hold "
          f"{MONTH} trading days. {len(decisions)} decision dates.")
    print(f"costs per vega unit: index half-spread {SPREAD_INDEX} vol pts, single-name "
          f"{SPREAD_SINGLE}, hedging allowance {HEDGE_COST} per leg per month")
    print("P&L is in vol points per one vega unit of index straddle. Read 1 vol point "
          "as 1% of capital if the book is sized so that the index vega is 1% of "
          "capital per vol point.")

    print(f"index implied in the book is VIX minus the snapshot wedge of {wedge:.1f} vol pts; "
          f"single-name implied is {premium.iloc[0]:.2f} times trailing 21d realised")
    ones = pd.Series(1.0, index=data.NAMES)
    always = simulate(single, index, vix, weights, premium, decisions, wedge=wedge)
    equal = pd.Series(1 / len(data.NAMES), index=data.NAMES)
    always_ew = simulate(single, index, vix, equal, premium, decisions, wedge=wedge)
    always_k1 = simulate(single, index, vix, weights, ones, decisions)
    always_63 = simulate(single, index, vix, weights, premium, decisions, lookback=63,
                         wedge=wedge)
    corr_w = simulate(single, index, vix, weights, premium, decisions, wedge=wedge,
                      corr_weighted=True)
    print(f"correlation-weighted book: single-name vega scaled by the parallel vega, "
          f"mean {corr_w['scale'].mean():.2f}, range {corr_w['scale'].min():.2f} to "
          f"{corr_w['scale'].max():.2f}")

    # Conditional book: trade only when the decision-date gap clears the formation
    # median. The threshold is read off the formation window once and then frozen.
    signal = gap["gap k=snapshot vs 63d"].reindex(always["decision"]).values
    threshold = gap.loc[FORMATION[0]:FORMATION[1], "gap k=snapshot vs 63d"].median()
    on = signal > threshold
    conditional = always.copy()
    conditional[["gross", "single_leg", "index_leg", "cost", "net"]] = \
        conditional[["gross", "single_leg", "index_leg", "cost", "net"]].mul(on, axis=0)
    print(f"conditional book: on when the decision-date gap (k=snapshot vs 63d) exceeds "
          f"the formation median {threshold:+.3f}; on in {on.mean():.0%} of months "
          f"({on[always['decision'] >= TEST[0]].mean():.0%} in the test window)")

    books = {"always on, cap weights": always, "always on, equal weights": always_ew,
             "always on, k=1, raw VIX": always_k1, "always on, 63d proxy": always_63,
             "correlation-weighted": corr_w, "conditional on gap": conditional}
    always.to_csv(os.path.join(REPORTS, "book.csv"))

    perf = {}
    for name, b in books.items():
        for label, span in (("formation", FORMATION), ("test", TEST)):
            sub = b.loc[span[0]:span[1]]
            for leg in ("gross", "net"):
                perf[(name, label, leg)] = fmt(metrics(sub[leg]))
    perf = pd.DataFrame(perf)
    perf.columns = [f"{a} | {b} | {c}" for a, b, c in perf.columns]
    show("Formation (2015-2020) and test (2021-2026) windows, gross and net",
         perf[[c for c in perf.columns if c.startswith("always on, cap")]])
    show("Alternative books, net", perf[[c for c in perf.columns
                                         if "net" in c and not c.startswith("always on, cap")]])
    perf.T.to_csv(os.path.join(REPORTS, "performance.csv"))

    full = metrics(always["net"])
    print(f"\nfull sample, always on, net: Sharpe {full['sharpe']:.2f}, "
          f"max drawdown {full['max_drawdown']:+.2f} vol pts "
          f"({full['dd_peak']:%Y-%m-%d} to {full['dd_trough']:%Y-%m-%d}), "
          f"worst month {full['worst_month']:+.2f} ({full['worst_month_date']:%Y-%m-%d})")

    tail = always.nsmallest(6, "net")[["decision", "implied_corr", "realised_corr",
                                       "single_leg", "index_leg", "gross", "net"]]
    show("Six worst months, always on", tail.round(2))
    legs = always[["single_leg", "index_leg", "gross"]]
    print(f"\nleg contribution, full sample: single-name leg {legs['single_leg'].sum():+.1f}, "
          f"index leg {legs['index_leg'].sum():+.1f}, gross {legs['gross'].sum():+.1f} vol pts")
    # Net P&L is linear in the premium ratio, so the ratio at which the book
    # breaks even is one division away and says how much of the result is the
    # assumption rather than the data.
    trailing = realised_vol(single, MONTH).loc[always.index].mul(weights, axis=1).sum(axis=1) * 100
    k = premium.iloc[0]
    slope = trailing.mean()
    print(f"break-even single-name premium ratio: net mean is {always['net'].mean():+.2f} at "
          f"k={k:.2f} and falls {slope:.1f} vol pts per unit of k, so the book breaks even at "
          f"k={k + always['net'].mean() / slope:.3f} (gross: k={k + always['gross'].mean() / slope:.3f})")
    print(f"correlation of monthly gross P&L with realised minus implied correlation: "
          f"{always['gross'].corr(always['realised_corr'] - always['implied_corr']):+.2f}")

    on_off = always.assign(on=on).groupby("on")[["single_leg", "index_leg", "gross"]].mean()
    show("Conditional filter: mean monthly leg P&L in months it is on vs off", on_off.round(2))

    rows = {}
    for kk, c in ((1.0, 0.0), (1.0, wedge), (float(premium.iloc[0]), 0.0),
                  (float(premium.iloc[0]), wedge)):
        b = simulate(single, index, vix, weights, ones * kk, decisions, wedge=c)
        rows[f"k={kk:.2f}, wedge={c:.1f}"] = {
            "net mean": round(b["net"].mean(), 2),
            "single leg mean": round(b["single_leg"].mean(), 2),
            "index leg mean": round(b["index_leg"].mean(), 2),
            "net Sharpe": round(metrics(b["net"])["sharpe"], 2),
            "max drawdown": round(metrics(b["net"])["max_drawdown"], 1),
        }
    for kk, c in ((1.0, 0.0), (float(premium.iloc[0]), wedge)):
        b = simulate(single, index, vix, weights, ones * kk, decisions, wedge=c,
                     corr_weighted=True)
        rows[f"k={kk:.2f}, wedge={c:.1f}, correlation-weighted"] = {
            "net mean": round(b["net"].mean(), 2),
            "single leg mean": round(b["single_leg"].mean(), 2),
            "index leg mean": round(b["index_leg"].mean(), 2),
            "net Sharpe": round(metrics(b["net"])["sharpe"], 2),
            "max drawdown": round(metrics(b["net"])["max_drawdown"], 1),
        }
    cal = pd.DataFrame(rows).T
    show("Calibration sensitivity, full sample net: the two unobservable constants", cal)
    cal.to_csv(os.path.join(REPORTS, "calibration_sensitivity.csv"))

    rows = {}
    for label, si, ss, h in (("frictionless", 0, 0, 0), ("half", 0.125, 0.375, 0.05),
                             ("base", SPREAD_INDEX, SPREAD_SINGLE, HEDGE_COST),
                             ("double", 0.5, 1.5, 0.2), ("triple", 0.75, 2.25, 0.3)):
        b = simulate(single, index, vix, weights, premium, decisions, wedge=wedge,
                     spread_index=si, spread_single=ss, hedge=h)
        rows[f"{label} ({si}/{ss}/{h})"] = {
            "cost per month": round(b["cost"].iloc[0], 2),
            "formation Sharpe": round(metrics(b.loc[FORMATION[0]:FORMATION[1], "net"])["sharpe"], 2),
            "test Sharpe": round(metrics(b.loc[TEST[0]:TEST[1], "net"])["sharpe"], 2),
            "full Sharpe": round(metrics(b["net"])["sharpe"], 2),
        }
    cs = pd.DataFrame(rows).T
    show("Net Sharpe by cost scenario (index spread / single spread / hedge, vol pts)", cs)
    cs.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))
    return books


def charts(gap, books, implied30):
    split = pd.Timestamp(data.TEST_START)
    always = books["always on, cap weights"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(gap["implied k=snapshot"], linewidth=0.9, label="implied correlation proxy (k=snapshot)")
    ax.plot(gap["implied k=1"], linewidth=0.7, alpha=0.6, label="implied proxy, k=1 (upper bound)")
    ax.plot(gap["realised 63d"], linewidth=0.9, label="realised, trailing 63d")
    ax.plot(gap["realised 21d"], linewidth=0.6, alpha=0.6, label="realised, trailing 21d")
    ax.set_ylim(-0.2, 1.3)
    ax.axvline(split, color="black", linewidth=0.8)
    ax.set_title("Implied vs realised correlation of the 10-name basket "
                 "(implied side is a VIX-based proxy, see README)")
    ax.legend(fontsize=8, ncol=2)
    ax = axes[1]
    g = gap["gap k=snapshot vs 63d"]
    ax.fill_between(g.index, g, 0, where=g > 0, color="seagreen", alpha=0.6)
    ax.fill_between(g.index, g, 0, where=g < 0, color="firebrick", alpha=0.6)
    ax.axvline(split, color="black", linewidth=0.8)
    ax.set_title("Implied minus realised (63d) correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "gap_history.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, b in books.items():
        ax.plot(b["net"].cumsum(), linewidth=1.1, label=f"{name}, net")
    ax.plot(always["gross"].cumsum(), "--", linewidth=1.0, color="grey",
            label="always on, cap weights, gross")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(split, color="black", linewidth=0.8)
    ax.text(split, ax.get_ylim()[1], " test", va="top", fontsize=8)
    ax.set_ylabel("cumulative P&L, vol points per vega unit")
    ax.set_title("Dispersion book: short index straddle, long weighted single-name straddles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "equity_curve.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(always["net"], bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Monthly net P&L, vol points")
    axes[1].scatter(always["realised_corr"] - always["implied_corr"], always["gross"], s=12)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set_xlabel("realised minus implied correlation over the month")
    axes[1].set_ylabel("gross P&L, vol points")
    axes[1].set_title("Short correlation: P&L against the correlation surprise")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "pnl_distribution.png"), dpi=140)
    plt.close(fig)

    if implied30 is not None:
        t = implied30.drop(index=data.INDEX)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["implied 30d"], 0.4, label="implied 30d")
        ax.bar(x + 0.2, t["realised 21d"], 0.4, label="realised 21d")
        ax.set_xticks(x, t.index)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title("Today's single-name implied vs trailing realised vol")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORTS, "current_vols.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(REPORTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(REPORTS, "run_log.txt"))
    main()
