"""Build the point-in-time panel, report its coverage, and measure the survivorship
bias in the momentum and pairs projects by rerunning their rules on both universes.

Run: python3 run.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import survivorship as sv

REPORTS = data.REPORTS


class Tee:
    """Write stdout to the terminal and to reports/run_log.txt at the same time."""

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


def pct(x, signed=True):
    if pd.isna(x):
        return "n/a"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def fmt_table(t):
    """Percent-format the return rows, leave ratios and counts alone."""
    rows = {}
    for row in t.index:
        is_pct = any(k in row for k in ("return", "leg", "drawdown", "vol", "share", "hit rate"))
        signed = any(k in row for k in ("return", "leg", "drawdown"))
        cells = []
        for col, v in t.loc[row].items():
            if is_pct:
                cells.append(pct(v, signed or col.startswith("difference")))
            else:
                cells.append(f"{v:+.2f}" if col.startswith("difference") else
                             (f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"))
        rows[row] = cells
    return pd.DataFrame(rows, index=t.columns).T


def panel_stats(panel, px, changes, snaps, corrections):
    """Per-year size, price coverage and change counts of the panel."""
    priced = panel & px.reindex(columns=panel.columns).notna()
    flips = panel.astype(int).diff()
    years = panel.index.year
    out = pd.DataFrame({
        "members (median)": panel.sum(axis=1).groupby(years).median(),
        "priced members (median)": priced.sum(axis=1).groupby(years).median(),
        "additions": (flips == 1).sum(axis=1).groupby(years).sum(),
        "removals": (flips == -1).sum(axis=1).groupby(years).sum(),
        "change rows": changes.groupby(changes["date"].dt.year).size(),
        "snapshots": pd.Series([d.year for d in snaps]).value_counts(),
        "anchor corrections": corrections.groupby(corrections["anchor"].dt.year).size(),
    }).fillna(0).astype(int)
    out["priced share"] = (priced.sum(axis=1) / panel.sum(axis=1)).groupby(years).mean().round(3)
    return out.loc[out.index >= 2003]


def main():
    os.makedirs(REPORTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(REPORTS, "run_log.txt"))

    d = data.load()
    panel, px, vol, cov = d["panel"], d["px"], d["vol"], d["coverage"]
    today, snaps, chg = d["today"], d["snapshots"], d["changes"]
    bench = px[data.BENCHMARK].pct_change()
    px, vol = px.reindex(columns=panel.columns), vol.reindex(columns=panel.columns)

    print(f"panel: {panel.shape[1]} distinct names, {panel.index[0].date()} to "
          f"{panel.index[-1].date()}, {len(panel)} trading days")
    print(f"sources: {len(snaps)} list snapshots ({min(snaps).date()} to {max(snaps).date()}), "
          f"{len(chg)} change rows ({chg['date'].min().date()} to {chg['date'].max().date()})")
    print(f"symbol renames applied: {len(d['renames'])} ({len(data.RENAMES)} by hand, "
          f"{len(d['renames']) - len(data.RENAMES)} matched on security name)")
    print(f"unresolved snapshot differences: {len(d['unresolved'])} "
          f"(changes the table lacks; the panel snaps to the snapshot for these)")
    print(f"today's list: {len(today)} names")

    stats = panel_stats(panel, px, chg, snaps, d["corrections"])
    show("Membership panel by year", stats)
    stats.to_csv(os.path.join(REPORTS, "panel_by_year.csv"))
    data.intervals(panel).to_csv(os.path.join(REPORTS, "membership_intervals.csv"), index=False)
    cov.to_csv(os.path.join(REPORTS, "coverage.csv"))
    d["unresolved"].to_csv(os.path.join(REPORTS, "unresolved.csv"), index=False)
    d["corrections"].to_csv(os.path.join(REPORTS, "anchor_corrections.csv"), index=False)
    pd.DataFrame([(k, v[0], v[1]) for k, v in d["renames"].items()],
                 columns=["old", "new", "effective"]).to_csv(os.path.join(REPORTS, "renames.csv"), index=False)

    removed = cov[~cov["in_today"]]
    any_px = (removed["price_share"] > 0).sum()
    full_px = (removed["price_share"] >= 0.9).sum()
    print(f"\nnames removed from the index at some point: {len(removed)}")
    print(f"  with any price history on Yahoo: {any_px} ({any_px / len(removed):.0%})")
    print(f"  with prices on at least 90% of their member days: {full_px} ({full_px / len(removed):.0%})")
    print(f"  with no price history at all: {(removed['price_share'] == 0).sum()}")
    stub = cov[cov["in_today"] & (cov["price_share"] < 0.9)]
    print(f"names in today's list whose history Yahoo has already cut to a stub: "
          f"{len(stub)} ({', '.join(stub.index)})")
    by_year = removed.assign(year=removed["last_member"].dt.year, hit=removed["price_share"] >= 0.9)
    hit = by_year.groupby("year")["hit"].agg(["sum", "count"])
    hit.columns = ["recovered", "removed"]
    show("Removed names recovered from Yahoo, by year of removal", hit.T)

    print("\n" + "=" * 78)
    print("LEVEL OF THE BIAS: EQUAL-WEIGHT BASKETS")
    print("=" * 78)
    today_mask = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    today_mask[[c for c in panel.columns if c in today]] = True
    dated = today_mask & panel
    ew = {}
    for name, win in sv.MOM_WINDOWS.items():
        rows = {}
        for label, mask in (("today's members, full history", today_mask),
                            ("today's members, only while members", dated),
                            ("point-in-time members", panel)):
            r = sv.equal_weight(px, mask).loc[win[0]:win[1]]
            rows[label] = sv.annualised(r)
        rows["SPY"] = sv.annualised(bench.loc[win[0]:win[1]])
        ew[name] = pd.Series(rows)
    ew = pd.DataFrame(ew)
    show("Equal-weight annual return by universe", ew.map(pct))
    ew.to_csv(os.path.join(REPORTS, "equal_weight_levels.csv"))
    print("""
The first row is what every project in this repo was doing. The second keeps
today's names but only from the day each one joined, so the gap between rows
one and two is the "future winners held early" half of the bias, and the gap
between rows two and three is the "losers deleted" half. Row three still lacks
every removed name Yahoo no longer serves, so it understates the true index.""")

    print("\n" + "=" * 78)
    print("MOMENTUM: 12-1 DECILE LONG-SHORT, SAME RULES, TWO UNIVERSES")
    print("=" * 78)
    tables, books = sv.momentum_comparison(px, vol, panel, today)
    for name, t in tables.items():
        show(f"12-1 momentum, {name} ({sv.MOM_WINDOWS[name][0]} to {sv.MOM_WINDOWS[name][1]})",
             fmt_table(t))
        t.to_csv(os.path.join(REPORTS, f"momentum_{name.replace(' ', '_')}.csv"))
    print(f"\nstock-days held past a name's last print (point-in-time book): "
          f"{books['pit'].attrs['held_returns_missing']}")

    print("\n" + "=" * 78)
    print("PAIRS: WITHIN-SECTOR ENGLE-GRANGER SCREEN, SAME RULES, TWO UNIVERSES")
    print("=" * 78)
    sectors_today = snaps[max(snaps)].set_index("ticker")["sector"]
    sectors_pit = data.sectors_at(snaps, sv.PAIRS_FORMATION[1])
    print(f"screening date {sv.PAIRS_FORMATION[1]}: sectors for the point-in-time universe come "
          f"from the list snapshot of {max(dd for dd in snaps if dd <= pd.Timestamp(sv.PAIRS_FORMATION[1])).date()}")
    ptables, chosen = sv.pairs_comparison(px, panel, today, sectors_today, sectors_pit)
    for name, t in ptables.items():
        show(f"pairs, {name}", fmt_table(t))
        t.to_csv(os.path.join(REPORTS, f"pairs_{name.replace(' ', '_')}.csv"))
    for k, c in chosen.items():
        c.to_csv(os.path.join(REPORTS, f"pairs_selected_{k}.csv"), index=False)
    a, b = set(map(tuple, chosen["today"][["a", "b"]].values)), set(map(tuple, chosen["pit"][["a", "b"]].values))
    print(f"\npairs chosen on both universes: {len(a & b)} of {len(a)} / {len(b)}")
    gone = [t for t in set(chosen["pit"]["a"]) | set(chosen["pit"]["b"]) if t not in today]
    print(f"names in the point-in-time selection that are no longer in the index: "
          f"{len(gone)} ({', '.join(sorted(gone))})")

    charts(panel, px, books, cov, today)
    print(f"\ntables and charts written to {REPORTS}/")


def charts(panel, px, books, cov, today):
    have = px.reindex(columns=panel.columns).notna()
    n_members = panel.sum(axis=1)
    n_priced = (panel & have).sum(axis=1)
    n_today = (panel & pd.DataFrame({c: c in today for c in panel.columns}, index=panel.index)).sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(n_members, label="names in the index (point-in-time panel)", linewidth=1.3)
    ax.plot(n_priced, label="of which Yahoo still serves a price", linewidth=1.3)
    ax.plot(n_today, label="of which are in today's list", linewidth=1.0, linestyle="--")
    ax.set_ylim(0, 560)
    ax.set_ylabel("names")
    ax.set_title("Index size vs what a yfinance backtest can actually see")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "index_size_vs_recoverable.png"), dpi=140)
    plt.close(fig)

    span = slice(sv.MOM_WINDOWS["full"][0], sv.MOM_WINDOWS["full"][1])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, book, style in (("today's members, gross", books["today"], "--"),
                               ("today's members, net", books["today"], "-"),
                               ("point-in-time, gross", books["pit"], "--"),
                               ("point-in-time, net", books["pit"], "-")):
        col = "gross" if "gross" in label else "net"
        ax.plot((1 + book.loc[span, col].fillna(0)).cumprod(), style, linewidth=1.1, label=label)
    ax.axvline(pd.Timestamp(sv.MOM_WINDOWS["final test"][0]), color="black", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title("12-1 momentum on the two universes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "momentum_two_universes.png"), dpi=140)
    plt.close(fig)

    removed = cov[~cov["in_today"]]
    by_year = removed.assign(year=removed["last_member"].dt.year, hit=removed["price_share"] >= 0.9)
    hit = by_year.groupby("year")["hit"].agg(["sum", "count"])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(hit.index, hit["count"], color="lightgrey", label="removed from the index")
    ax.bar(hit.index, hit["sum"], color="steelblue", label="still priced on Yahoo")
    ax.set_title("Removed names by year of removal, and how many a yfinance backtest can recover")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "removed_recovered.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
