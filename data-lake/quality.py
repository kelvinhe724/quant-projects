"""Daily data-quality report: staleness, gaps, duplicates per dataset, written to reports/status.json.

Run: ../.venv/bin/python3 quality.py   (seconds; also runs at the end of every launchd job)
"""
import json
import os
import shutil
import sys

import pandas as pd
import pyarrow.parquet as pq

import lake
from collect import VENUES

sys.path.append(lake.ROOT)
from framework.book.universe import sessions  # noqa: E402  the book's NYSE calendar

# how old the newest row may be before the feed counts as stale
# wall-clock backstop only: daily feeds are judged by sessions in DAILY_BARS (due 06:00 CT the next day), and these
# widths cover a three-day weekend so a Monday or Tuesday morning report does not alarm on Friday's bar
STALE_AFTER = {"equities_daily": "7D", "fred": "7D", "kalshi_markets": "2h", "kalshi_books": "2h",
               "kalshi_books_5m": "15min", "edgar_8k": "5D", "crypto_l2": "3min", "crypto_klines_1m": "5D",
               "crypto_bookdepth": "5D", "options": "4D"}
# (parts to scan for gaps and duplicates, expected cadence)
GAP_SCAN = {"kalshi_markets": 7, "kalshi_books": 7, "kalshi_books_5m": 24, "crypto_l2": 24, "crypto_klines_1m": 1,
            "crypto_bookdepth": 1, "equities_daily": ["SPY"], "fred": ["DGS10"]}
# book snapshots: one key per (ts, ticker, side, price); gaps are judged on the snapshot clock, not per ticker
BOOKS = ("kalshi_books", "kalshi_books_5m")
GROWTH_DAYS = 7        # storage guard: growth is measured over the trailing week
RUNWAY_MIN_DAYS = 30   # ... and the lake goes not-ok when the disk would fill sooner than this at that rate
HISTORY_EVERY = pd.Timedelta(hours=23)   # one size entry a day is enough for a week-long window
# daily bars due by 06:00 CT the day after a session, and how many sessions each may trail:
# FRED posts a day's rates the next afternoon, after the nightly run, so fred is a session behind
DAILY_BARS = {"equities_daily": 0, "fred": 2}  # 2: the bond market closes on Columbus and Veterans Day while NYSE is open
BOOKS_5M_MIN = 0.6     # share of the 288 five-minute snapshots kalshi_books_5m must hold on the last full UTC day
CT = "America/Chicago"
SPLICE_TOL = 1e-5      # adj_close/close may move this much between adjacent rows before a fetch boundary counts as a splice
JUMP_TOL = 0.5         # a raw close that moves this much across a fetch boundary is a different series (the CTVA type)


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


def splices(tol=SPLICE_TOL, jump=JUMP_TOL):
    """Fetch-boundary breaks in equities_daily, flag only: {"adj": {ticker: [dates]}, "jump": {...}, counts}.

    Yahoo's adj_close is a snapshot from the fetch date, so r = adj_close/close steps where an older fetch meets a
    newer one; a true ex-date step sits inside one fetch (one collected_at) and is not flagged. Dates are the later row.
    """
    out = {"adj": {}, "jump": {}}
    for t in lake.parts("equities_daily"):
        have = set(pq.ParquetFile(lake.part_path("equities_daily", t)).schema_arrow.names)
        if not {"date", "close", "adj_close", "collected_at"} <= have:
            continue
        df = pd.read_parquet(lake.part_path("equities_daily", t), columns=["date", "close", "adj_close", "collected_at"])
        df = df.sort_values("date")
        fetch = df["collected_at"] != df["collected_at"].shift()   # True where a new fetch starts
        fetch.iloc[0] = False
        r, c = df["adj_close"] / df["close"], df["close"]
        for key, step, lim in (("adj", r / r.shift() - 1, tol), ("jump", c / c.shift() - 1, jump)):
            hit = df["date"][fetch & (step.abs() > lim)]
            if len(hit):
                out[key][t] = [str(d.date()) for d in hit]
    out["adj_count"], out["jump_count"] = (sum(map(len, out[k].values())) for k in ("adj", "jump"))
    return out


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


def last_runs():
    """reports/last_run.json from collect.bundle(): {command: {started, finished, steps}}, {} if never written."""
    try:
        return json.load(open(os.path.join(lake.REPORTS, "last_run.json")))
    except (OSError, ValueError):
        return {}


def since(dataset, start, now, cols):
    """Rows of an hour-partitioned dataset in [start, now], reading only `cols` of the hour parts that can hold them."""
    tcol = lake.DATASETS[dataset][0]
    names = [n for n in lake.parts(dataset) if pd.Timestamp(f"{n[:10]} {n[11:13]}:00", tz="UTC") >= start.floor("h")]
    frames = [pd.read_parquet(lake.part_path(dataset, n), columns=cols) for n in names]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    t = pd.to_datetime(df[tcol], utc=True)
    return df[(t >= start) & (t <= now)]


def feed_problems(now, datasets):
    """What freshness by newest row cannot see: a failed nightly step, a missing daily bar, a dead venue, a thin day."""
    out = []
    run = last_runs().get("daily")
    if run:
        out += [f"daily run {run['started'][:16]}Z: {s['step']} failed: {s['error'][:120]}"
                for s in run["steps"] if s["status"] == "failed"]
    local = now.tz_convert(CT).tz_localize(None)
    days = sessions(local - pd.Timedelta(days=14), local)
    due = days[(days + pd.Timedelta(hours=30)).tz_localize(CT) <= now]   # 06:00 CT the next calendar day
    for name, lag in DAILY_BARS.items():
        want, last = due[-1 - lag], datasets[name]["last"]
        if last is None or pd.Timestamp(last).normalize() < want:
            out.append(f"{name}: no {want.date()} bar (last {str(last)[:10]})")
    day = now.floor("D") - pd.Timedelta(days=1)
    l2 = since("crypto_l2", now - pd.Timedelta(hours=24), now, ["ts", "venue"])
    out += [f"crypto_l2 {v}: no rows in 24h" for v in VENUES if not (l2["venue"] == v).any()]
    b = since("kalshi_books_5m", day, now, ["ts"])
    t = pd.to_datetime(b["ts"], utc=True)
    if not (t > now - pd.Timedelta(hours=24)).any():
        out.append("kalshi_books_5m: no rows in 24h")
    n = t[t < day + pd.Timedelta(days=1)].dt.floor("5min").nunique()   # a doubled launch in one slot counts once
    if n < BOOKS_5M_MIN * 288:
        out.append(f"kalshi_books_5m: {n} of 288 snapshots on {day.date()}, under {BOOKS_5M_MIN:.0%}")
    return out


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
    feeds = (["stale: " + ", ".join(stale)] if stale else []) + (["duplicates in " + ", ".join(dup)] if dup else []) \
        + feed_problems(now, out["datasets"])
    out["problems"] = feeds + ([disk_line(disk)] if disk["low"] else [])   # every reason ok is false, one line each
    out["ok"] = not out["problems"]
    out["splices"] = splices()   # flag only: not a problem, does not touch ok
    out["summary"] = ("; ".join(feeds) or "all feeds fresh") + "; " + disk_line(disk)
    return out


def main():
    r = report()
    os.makedirs(lake.REPORTS, exist_ok=True)
    path = os.path.join(lake.REPORTS, "status.json")
    with open(path + ".tmp", "w") as fh:   # status.json carries the disk history now, so never leave it half-written
        json.dump(r, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)
    print(f"quality: {r['summary']}")
    print(f"  splices: {r['splices']['adj_count']} adj_close fetch boundaries, {r['splices']['jump_count']} close jumps over {JUMP_TOL:.0%}")
    for n, d in r["datasets"].items():
        print(f"  {n:18s} parts {d['parts']:6d} rows {d['rows']:10d} last {d['last']}  "
              f"{'STALE' if d['stale'] else 'fresh'}  gaps {len(d['gaps'])} dupes {d['dupes']}")
    return r


if __name__ == "__main__":
    main()
