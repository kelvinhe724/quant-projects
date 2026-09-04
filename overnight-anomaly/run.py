"""Full pipeline on real prices: decomposition, significance, stability, and the cost reality.

Run: python3 run.py
Writes tables and charts to reports/ and a copy of the console output to reports/run_log.txt.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from decompose import (annualise, breakeven_cost_bps, by_year, legs, nw_tstat, strategy,
                       summarise)

REPORTS = data.REPORTS
COSTS_BPS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
PERIODS = {
    "full 2000-2026": (None, None),
    "pre-publication 2000-2015": (None, data.IS_END),
    "post-publication 2016-2026": (data.TEST_START, None),
    "post-2010": ("2010-01-01", None),
}


class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def show(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string())


def slice_period(frame, period):
    start, end = PERIODS[period]
    return frame.loc[start:end]


def leg_table(all_legs, period):
    """One row per asset: annual return and NW t-stat of each leg over the period."""
    rows = {}
    for t, L in all_legs.items():
        Lp = slice_period(L, period)
        if len(Lp) < 250:
            continue
        s = summarise(Lp)
        rows[t] = {
            "group": data.GROUP[t],
            "overnight_ann": s.loc["overnight", "annual_return"],
            "intraday_ann": s.loc["intraday", "annual_return"],
            "total_ann": s.loc["close_to_close", "annual_return"],
            "overnight_t": s.loc["overnight", "nw_tstat"],
            "intraday_t": s.loc["intraday", "nw_tstat"],
            "overnight_bps": s.loc["overnight", "mean_daily_bps"],
            "intraday_bps": s.loc["intraday", "mean_daily_bps"],
            "days": int(s.loc["overnight", "days"]),
        }
    return pd.DataFrame(rows).T.infer_objects()


def cumulative_chart(L, ticker, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in [("close_to_close", "buy and hold"), ("overnight", "overnight only"),
                       ("intraday", "intraday only")]:
        ax.plot((1 + L[col]).cumprod(), label=label)
    ax.set_yscale("log")
    ax.set_title(f"{ticker}: growth of $1, no costs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def bar_chart(frame, cols, labels, title, path, rotate=0):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(frame))
    w = 0.8 / len(cols)
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(x + i * w - 0.4 + w / 2, frame[c] * 100, width=w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(frame.index, rotation=rotate)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("annual return, %")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(REPORTS, "run_log.txt"))

    panel = data.get_panel()
    all_legs = {}
    hygiene = {}
    for t, f in panel.items():
        kept = data.drop_stale_years(f)
        hygiene[t] = {"start": f.index[0].date(), "end": f.index[-1].date(),
                      "stale_open_pct": data.stale_open(f).mean() * 100,
                      "dropped_days": len(f) - len(kept)}
        all_legs[t] = legs(kept)
    hygiene = pd.DataFrame(hygiene).T
    show("DATA", hygiene)
    hygiene.to_csv(os.path.join(REPORTS, "data_hygiene.csv"))

    print("\nDecomposition identity on real data:")
    worst = max(((1 + L["overnight"]) * (1 + L["intraday"]) - 1 - L["close_to_close"]).abs().max()
                for L in all_legs.values())
    print(f"  max |(1+on)(1+id) - (1+c2c)| across all assets = {worst:.2e}")

    # Full-sample table, one row per asset
    full = leg_table(all_legs, "full 2000-2026")
    show("BY ASSET, FULL SAMPLE (annual return; NW t-stat on daily mean, 5 lags)",
         full.round(3))
    full.to_csv(os.path.join(REPORTS, "by_asset.csv"))
    bar_chart(full, ["overnight_ann", "intraday_ann", "total_ann"],
              ["overnight", "intraday", "close to close"],
              "Annualised return by leg, 2000-2026, no costs",
              os.path.join(REPORTS, "by_asset.png"), rotate=45)

    print("\nCount of assets where the overnight leg out-earns the intraday leg: "
          f"{(full['overnight_ann'] > full['intraday_ann']).sum()} of {len(full)}")
    print(f"Assets with overnight t > 2: {(full['overnight_t'] > 2).sum()}; "
          f"intraday t > 2: {(full['intraday_t'] > 2).sum()}; "
          f"intraday t < -2: {(full['intraday_t'] < -2).sum()}")

    # Sub-periods, for the index ETFs and for each group's equal-weight average
    rows = []
    for period in PERIODS:
        tab = leg_table(all_legs, period)
        for t in data.INDEX_ETFS:
            if t in tab.index:
                r = tab.loc[t]
                rows.append({"period": period, "asset": t, **r.drop("group").to_dict()})
        for g in ("sector", "stock"):
            members = [t for t in tab.index if data.GROUP[t] == g]
            ew = pd.DataFrame({
                leg: pd.concat([slice_period(all_legs[t], period)[leg] for t in members],
                               axis=1).mean(axis=1)
                for leg in ("overnight", "intraday", "close_to_close")}).dropna()
            s = summarise(ew)
            rows.append({"period": period, "asset": f"EW {g}s ({len(members)})",
                         "overnight_ann": s.loc["overnight", "annual_return"],
                         "intraday_ann": s.loc["intraday", "annual_return"],
                         "total_ann": s.loc["close_to_close", "annual_return"],
                         "overnight_t": s.loc["overnight", "nw_tstat"],
                         "intraday_t": s.loc["intraday", "nw_tstat"],
                         "overnight_bps": s.loc["overnight", "mean_daily_bps"],
                         "intraday_bps": s.loc["intraday", "mean_daily_bps"],
                         "days": int(s.loc["overnight", "days"])})
    periods = pd.DataFrame(rows).set_index(["period", "asset"])
    show("BY PERIOD", periods.round(3))
    periods.to_csv(os.path.join(REPORTS, "by_period.csv"))

    # SPY in depth
    spy = all_legs["SPY"]
    cumulative_chart(spy, "SPY", os.path.join(REPORTS, "cumulative_spy.png"))
    cumulative_chart(all_legs["QQQ"], "QQQ", os.path.join(REPORTS, "cumulative_qqq.png"))
    cumulative_chart(all_legs["IWM"], "IWM", os.path.join(REPORTS, "cumulative_iwm.png"))

    years = by_year(spy)
    years["overnight_t"] = [nw_tstat(g["overnight"])[1] for _, g in spy.groupby(spy.index.year)]
    years["intraday_t"] = [nw_tstat(g["intraday"])[1] for _, g in spy.groupby(spy.index.year)]
    show("SPY BY YEAR", years.round(3))
    years.to_csv(os.path.join(REPORTS, "spy_by_year.csv"))
    bar_chart(years, ["overnight", "intraday"], ["overnight", "intraday"],
              "SPY: overnight vs intraday return by calendar year",
              os.path.join(REPORTS, "spy_by_year.png"), rotate=90)
    print(f"\nYears overnight > intraday: {(years['overnight'] > years['intraday']).sum()} "
          f"of {len(years)}; years overnight > 0: {(years['overnight'] > 0).sum()}; "
          f"years intraday > 0: {(years['intraday'] > 0).sum()}")

    # Rolling 3-year t-stat on SPY legs
    roll = pd.DataFrame({
        leg: spy[leg].rolling(756).apply(lambda x: nw_tstat(x)[1], raw=False)
        for leg in ("overnight", "intraday")})
    fig, ax = plt.subplots(figsize=(10, 4))
    roll.plot(ax=ax)
    ax.axhline(2, color="grey", ls="--", lw=0.8)
    ax.axhline(-2, color="grey", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("SPY: rolling 3-year Newey-West t-stat of the daily mean")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "spy_rolling_t.png"), dpi=120)
    plt.close(fig)

    # The cost reality
    cost_rows = {}
    for t, L in all_legs.items():
        row = {"breakeven_overnight_bps": breakeven_cost_bps(L["overnight"]),
               "breakeven_intraday_bps": breakeven_cost_bps(L["intraday"]),
               "buy_hold_ann": annualise(L["close_to_close"])}
        for c in COSTS_BPS:
            row[f"overnight_net_{c:g}bps"] = annualise(strategy(L, "overnight", c))
        for c in COSTS_BPS:
            row[f"intraday_net_{c:g}bps"] = annualise(strategy(L, "intraday", c))
        cost_rows[t] = row
    costs = pd.DataFrame(cost_rows).T
    show("COST SENSITIVITY (net annual return of holding one leg only, one-way cost in bps)",
         costs.round(4))
    costs.to_csv(os.path.join(REPORTS, "cost_sensitivity.csv"))

    beats = {c: (costs[f"overnight_net_{c:g}bps"] > costs["buy_hold_ann"]).sum()
             for c in COSTS_BPS}
    print("\nAssets where overnight-only beats buy and hold, by one-way cost:")
    for c, n in beats.items():
        print(f"  {c:>5g} bps: {n} of {len(costs)}")

    grid = np.linspace(0, 6, 61)
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in data.INDEX_ETFS:
        L = all_legs[t]
        ax.plot(grid, [annualise(strategy(L, "overnight", c)) * 100 for c in grid],
                label=f"{t} overnight only")
        ax.axhline(annualise(L["close_to_close"]) * 100, ls=":", lw=0.9,
                   color=ax.lines[-1].get_color())
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("one-way cost per trade, bps")
    ax.set_ylabel("net annual return, %")
    ax.set_title("Overnight-only strategy vs cost. Dotted = buy and hold of the same ETF")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "cost_curve.png"), dpi=120)
    plt.close(fig)

    # Dividends land in the overnight leg (ex-date gap is close to open), so part of the
    # overnight premium on the ETFs is just the yield.
    print("\nDIVIDENDS: average annual yield vs overnight leg, index ETFs")
    for t in data.INDEX_ETFS:
        y = data.dividend_yield(t).mean()
        on = full.loc[t, "overnight_ann"]
        print(f"  {t}: yield {y:.2%}/yr, overnight leg {on:.2%}/yr, "
              f"ex-dividend share {y / on:.0%}")

    # Post-publication test for the trading version, at a realistic cost
    print("\nPOST-PUBLICATION (2016-2026) overnight-only vs buy and hold, 1bp one-way:")
    for t in data.INDEX_ETFS:
        L = slice_period(all_legs[t], "post-publication 2016-2026")
        print(f"  {t}: overnight gross {annualise(L['overnight']):+.2%}, "
              f"net {annualise(strategy(L, 'overnight', 1.0)):+.2%}, "
              f"buy and hold {annualise(L['close_to_close']):+.2%}, "
              f"overnight t {nw_tstat(L['overnight'])[1]:.2f}, "
              f"intraday t {nw_tstat(L['intraday'])[1]:.2f}")

    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
