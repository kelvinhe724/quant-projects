"""Daily data-quality report: staleness, gaps, duplicates per dataset, written to reports/status.json.

Run: ../.venv/bin/python3 quality.py   (seconds; also runs at the end of every launchd job)
"""
import json
import os

import pandas as pd
import pyarrow.parquet as pq

import lake

# how old the newest row may be before the feed counts as stale
STALE_AFTER = {"equities_daily": "4D", "fred": "4D", "kalshi_markets": "2h", "kalshi_books": "2h",
               "edgar_8k": "4D", "crypto_l2": "3min", "crypto_klines_1m": "3D", "crypto_bookdepth": "3D",
               "options": "4D"}
# (parts to scan for gaps and duplicates, expected cadence)
GAP_SCAN = {"kalshi_markets": 7, "kalshi_books": 7, "crypto_l2": 24, "crypto_klines_1m": 1,
            "crypto_bookdepth": 1, "equities_daily": ["SPY"], "fred": ["DGS10"]}


def newest(dataset, names):
    """Latest time across the given parts, from the row data of the last few parts."""
    tcol = lake.DATASETS[dataset][0]
    best = None
    for n in names:
        df = pd.read_parquet(lake.part_path(dataset, n), columns=[tcol])
        if len(df):
            t = pd.to_datetime(df[tcol]).max()
            best = t if best is None or t > best else best
    return best


def to_utc(t):
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def scan(dataset, names):
    """Gaps and duplicate keys over the named parts."""
    tcol, kcol, kind, cadence = lake.DATASETS[dataset]
    frames = [pd.read_parquet(lake.part_path(dataset, n)) for n in names]
    if not frames:
        return [], 0
    df = pd.concat(frames, ignore_index=True)
    if kind == "key":
        df = df[pd.to_datetime(df[tcol]) >= pd.to_datetime(df[tcol]).max() - pd.Timedelta(days=120)]
    keys = [tcol, kcol] + (["side", "price"] if dataset == "kalshi_books" else []) \
        + (["percentage"] if dataset == "crypto_bookdepth" else [])
    dupes = int(df.duplicated([k for k in keys if k in df.columns]).sum())
    if dataset == "kalshi_books":
        df = df.groupby(tcol).size().reset_index()
    if kind == "key":
        g = lake.gaps(df[tcol], cadence)
    else:
        g = pd.concat([lake.gaps(s[tcol], cadence) for _, s in df.groupby(kcol)], ignore_index=True) \
            if kind == "hour" or dataset.startswith("crypto") else lake.gaps(df[tcol].drop_duplicates(), cadence)
    rows = [{"after": str(r.after), "before": str(r.before), "missing": int(r.missing)}
            for r in g.sort_values("missing", ascending=False).head(10).itertuples()] if len(g) else []
    return rows, dupes


def report(now=None):
    now = to_utc(now or pd.Timestamp.now("UTC"))
    out = {"generated_at": now.isoformat(), "datasets": {}}
    for name, (tcol, kcol, kind, cadence) in lake.DATASETS.items():
        d = {"parts": 0, "rows": 0, "last": None, "stale": True, "gaps": [], "dupes": 0}
        if kind == "external":
            try:
                df = lake.load("options", start=now - pd.Timedelta(days=10))
                d["rows"] = int(len(df))
                d["parts"] = int(df["snapshot"].dt.normalize().nunique()) if len(df) else 0
                last = df["snapshot"].max() if len(df) else None
            except Exception as e:
                d["note"] = f"{type(e).__name__}: {e}"
                last = None
        else:
            names = lake.parts(name)
            d["parts"] = len(names)
            d["rows"] = sum(pq.ParquetFile(lake.part_path(name, n)).metadata.num_rows for n in names)
            scan_names = GAP_SCAN.get(name)
            if isinstance(scan_names, int):
                scan_names = names[-scan_names:]
            elif scan_names:
                scan_names = [n for n in scan_names if n in names]
            recent = names[-3:] if kind != "key" else (scan_names or names[-1:])
            last = newest(name, recent) if names else None
            if scan_names:
                d["gaps"], d["dupes"] = scan(name, scan_names)
        if last is not None:
            d["last"] = str(last)
            d["stale"] = bool(now - to_utc(last) > pd.Timedelta(STALE_AFTER[name]))
        out["datasets"][name] = d
    stale = [n for n, d in out["datasets"].items() if d["stale"]]
    dup = [n for n, d in out["datasets"].items() if d["dupes"]]
    out["ok"] = not stale and not dup
    out["summary"] = ("all feeds fresh" if not stale else "stale: " + ", ".join(stale)) \
        + ("" if not dup else "; duplicates in " + ", ".join(dup))
    return out


def main():
    r = report()
    os.makedirs(lake.REPORTS, exist_ok=True)
    json.dump(r, open(os.path.join(lake.REPORTS, "status.json"), "w"), indent=1, default=str)
    print(f"quality: {r['summary']}")
    for n, d in r["datasets"].items():
        print(f"  {n:18s} parts {d['parts']:6d} rows {d['rows']:10d} last {d['last']}  "
              f"{'STALE' if d['stale'] else 'fresh'}  gaps {len(d['gaps'])} dupes {d['dupes']}")
    return r


if __name__ == "__main__":
    main()
