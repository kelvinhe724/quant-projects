"""Describe what the store holds: row counts, equity coverage against the PIT panel, L2 spread, curve.

Writes reports/inventory.csv, reports/coverage_by_year.csv, reports/coverage.png,
reports/crypto_spread.png and reports/summary.md.
Run: ../.venv/bin/python3 run.py   (offline, reads the store only)
"""
import os

import matplotlib
import pandas as pd
import pyarrow.parquet as pq

import lake

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def inventory():
    rows = []
    for name, (tcol, kcol, kind, cadence) in lake.DATASETS.items():
        if kind == "external":
            df = lake.load("options")
            rows.append({"dataset": name, "parts": df["snapshot"].dt.normalize().nunique() if len(df) else 0,
                         "rows": len(df), "keys": df[kcol].nunique() if len(df) else 0,
                         "first": df[tcol].min() if len(df) else None, "last": df[tcol].max() if len(df) else None,
                         "mb": None, "cadence": cadence})
            continue
        names = lake.parts(name)
        n = sum(pq.ParquetFile(lake.part_path(name, p)).metadata.num_rows for p in names)
        mb = sum(os.path.getsize(lake.part_path(name, p)) for p in names) / 1e6
        keys = set()
        first = last = None
        for p in (names if kind != "key" else names):
            sample = pd.read_parquet(lake.part_path(name, p), columns=[tcol, kcol]) if kind != "key" \
                else pd.read_parquet(lake.part_path(name, p), columns=[tcol])
            if kind != "key":
                keys |= set(sample[kcol])
            else:
                keys.add(p)
            if len(sample):
                t0, t1 = sample[tcol].min(), sample[tcol].max()
                first = t0 if first is None or t0 < first else first
                last = t1 if last is None or t1 > last else last
        rows.append({"dataset": name, "parts": len(names), "rows": n, "keys": len(keys),
                     "first": first, "last": last, "mb": round(mb, 1), "cadence": cadence})
    return pd.DataFrame(rows).set_index("dataset")


def coverage_by_year():
    """Per year: names in the PIT panel, names with a price row in the store, share."""
    iv = lake.sp500_intervals()
    out = []
    for y in range(2004, 2027):
        d = pd.Timestamp(f"{y}-06-30")
        members = set(iv[(iv.start <= d) & (iv.end >= d)].ticker)
        px = lake.load("equities_daily", f"{y}-06-24", f"{y}-06-30", universe=sorted(members))
        priced = set(px.ticker)
        out.append({"year": y, "members": len(members), "priced": len(priced), "share": len(priced) / len(members)})
    return pd.DataFrame(out).set_index("year")


def main():
    os.makedirs(lake.REPORTS, exist_ok=True)
    inv = inventory()
    inv.to_csv(os.path.join(lake.REPORTS, "inventory.csv"))
    print(inv.to_string())

    cov = coverage_by_year()
    cov.to_csv(os.path.join(lake.REPORTS, "coverage_by_year.csv"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cov.index, cov.members, label="PIT S&P 500 members (pit-universe)")
    ax.plot(cov.index, cov.priced, label="with a price row in the store")
    ax.set_title("Equity coverage at each June 30")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(lake.REPORTS, "coverage.png"), dpi=120)

    l2 = lake.load("crypto_l2")
    spread = None
    if len(l2):
        l2 = l2.assign(spread_bps=(l2.ask_px_1 - l2.bid_px_1) / l2.bid_px_1 * 1e4)
        spread = l2.groupby(["venue", "symbol"]).spread_bps.describe()[["count", "mean", "50%", "max"]]
        print(spread.to_string())
        fig, ax = plt.subplots(figsize=(8, 4))
        for (v, s), g in l2.groupby(["venue", "symbol"]):
            ax.plot(g.ts, g.spread_bps, lw=0.6, label=f"{v} {s}")
        ax.set_ylabel("top-of-book spread, bps")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(lake.REPORTS, "crypto_spread.png"), dpi=120)

    curve = lake.load("fred", universe=["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"])
    latest = curve[curve.date == curve.date.max()].set_index("series").value if len(curve) else None

    with open(os.path.join(lake.REPORTS, "summary.md"), "w") as f:
        f.write(f"# Store inventory, {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC\n\n")
        f.write(inv.to_markdown() + "\n\n")
        f.write("## Equity coverage by year\n\n" + cov.round(3).to_markdown() + "\n\n")
        if spread is not None:
            f.write("## Crypto L2 top-of-book spread (bps)\n\n" + spread.round(3).to_markdown() + "\n\n")
        if latest is not None:
            f.write(f"## Treasury curve on {curve.date.max().date()}\n\n" + latest.to_frame().T.to_markdown() + "\n")
    print(cov.to_string())


if __name__ == "__main__":
    main()
