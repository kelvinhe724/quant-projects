"""Daily data-quality report: staleness, gaps, duplicates per dataset, written to reports/status.json.

Run: ../.venv/bin/python3 quality.py   (seconds; also runs at the end of every launchd job)
"""
import json
import os
import shutil

import pandas as pd
import pyarrow.parquet as pq

import lake

# how old the newest row may be before the feed counts as stale
STALE_AFTER = {"equities_daily": "4D", "fred": "4D", "kalshi_markets": "2h", "kalshi_books": "2h",
               "kalshi_books_5m": "15min", "edgar_8k": "4D", "crypto_l2": "3min", "crypto_klines_1m": "3D",
               "crypto_bookdepth": "3D", "options": "4D"}
# (parts to scan for gaps and duplicates, expected cadence)
GAP_SCAN = {"kalshi_markets": 7, "kalshi_books": 7, "kalshi_books_5m": 24, "crypto_l2": 24, "crypto_klines_1m": 1,
            "crypto_bookdepth": 1, "equities_daily": ["SPY"], "fred": ["DGS10"]}
# book snapshots: one key per (ts, ticker, side, price); gaps are judged on the snapshot clock, not per ticker
BOOKS = ("kalshi_books", "kalshi_books_5m")
GROWTH_DAYS = 7        # storage guard: growth is measured over the trailing week
RUNWAY_MIN_DAYS = 30   # ... and the lake goes not-ok when the disk would fill sooner than this at that rate
HISTORY_EVERY = pd.Timedelta(hours=23)   # one size entry a day is enough for a week-long window


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
    keys = [tcol, kcol] + (["side", "price"] if dataset in BOOKS else []) \
        + (["percentage"] if dataset == "crypto_bookdepth" else [])
    dupes = int(df.duplicated([k for k in keys if k in df.columns]).sum())
    if dataset in BOOKS:
        df = df.groupby(tcol).size().reset_index()
    if kind == "key":
        g = lake.gaps(df[tcol], cadence)
    else:
        g = pd.concat([lake.gaps(s[tcol], cadence) for _, s in df.groupby(kcol)], ignore_index=True) \
            if (kind == "hour" and dataset not in BOOKS) or dataset.startswith("crypto") \
            else lake.gaps(df[tcol].drop_duplicates(), cadence)
    rows = [{"after": str(r.after), "before": str(r.before), "missing": int(r.missing)}
            for r in g.sort_values("missing", ascending=False).head(10).itertuples()] if len(g) else []
    return rows, dupes


def du(path):
    """Bytes of every file under path; a file that vanishes mid-walk (lake.write renames) counts as nothing."""
    n = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return n


def sizes():
    """Bytes on disk per store directory, registered or not, with options under its raw folder."""
    on_disk = {n for n in os.listdir(lake.STORE) if os.path.isdir(os.path.join(lake.STORE, n))} \
        if os.path.isdir(lake.STORE) else set()
    out = {n: du(os.path.join(lake.STORE, n)) for n in on_disk | set(lake.DATASETS)}
    out["options"] = du(lake._options().ROOT)
    return out


def free_bytes():
    return shutil.disk_usage(lake.STORE if os.path.isdir(lake.STORE) else lake.HERE).free


def storage(now):
    """Size per dataset, growth a day over the trailing GROWTH_DAYS, and days until the disk is full.

    Growth needs the sizes of a week ago, so the previous status.json carries one entry a day
    in disk.history. With under half a day of history, growth and runway are None.
    """
    cur, free = sizes(), free_bytes()
    try:
        hist = lake.status()["disk"]["history"]
    except (TypeError, KeyError, ValueError, OSError):   # no status.json yet, or one from before this field
        hist = []
    hist = [h for h in hist if now - to_utc(h["at"]) <= pd.Timedelta(days=GROWTH_DAYS + 1)]
    ref = next((h for h in hist if now - to_utc(h["at"]) <= pd.Timedelta(days=GROWTH_DAYS)), None)
    days = (now - to_utc(ref["at"])).total_seconds() / 86400 if ref else 0
    known = days >= 0.5
    per = {}
    for n, b in cur.items():
        g = (b - ref["bytes"].get(n, 0)) / days if known else None
        per[n] = {"bytes": b, "growth_per_day": g, "runway_days": free / g if g and g > 0 else None}
    g = sum(v["growth_per_day"] for v in per.values()) if known else None
    runway = free / g if g and g > 0 else None
    if not hist or now - to_utc(hist[-1]["at"]) >= HISTORY_EVERY:
        hist.append({"at": now.isoformat(), "bytes": cur})
    return {"bytes": sum(cur.values()), "free_bytes": free, "window_days": round(days, 2), "growth_per_day": g,
            "runway_days": runway, "low": runway is not None and runway < RUNWAY_MIN_DAYS,
            "datasets": per, "history": hist}


def disk_line(d):
    """Lake size, growth and runway as one clause of the summary, which is what the desk report prints."""
    s = f"{'DISK LOW: ' if d['low'] else ''}lake {d['bytes'] / 1e9:.1f} GB"
    if d["runway_days"] is not None:
        s += f", +{d['growth_per_day'] / 1e6:.0f} MB/day, disk full in " \
             + ("over 10 years" if d["runway_days"] > 3650 else f"{d['runway_days']:,.0f} days")
    return s


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
                opt = lake._options()   # latest day per snapshot session; only close is judged stale
                d["sessions"] = {k: (days[-1] if (days := opt.list_days(k)) else None) for k in opt.SESSIONS}
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
    disk = out["disk"] = storage(now)
    out["ok"] = not stale and not dup and not disk["low"]
    out["summary"] = ("all feeds fresh" if not stale else "stale: " + ", ".join(stale)) \
        + ("" if not dup else "; duplicates in " + ", ".join(dup)) \
        + "; " + disk_line(disk)
    return out


def main():
    r = report()
    os.makedirs(lake.REPORTS, exist_ok=True)
    path = os.path.join(lake.REPORTS, "status.json")
    with open(path + ".tmp", "w") as fh:   # status.json carries the disk history now, so never leave it half-written
        json.dump(r, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)
    print(f"quality: {r['summary']}")
    for n, d in r["datasets"].items():
        print(f"  {n:18s} parts {d['parts']:6d} rows {d['rows']:10d} last {d['last']}  "
              f"{'STALE' if d['stale'] else 'fresh'}  gaps {len(d['gaps'])} dupes {d['dupes']}")
    return r


if __name__ == "__main__":
    main()
