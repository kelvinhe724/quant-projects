"""Snapshot the option chains once per trading day and never overwrite a day.

Run: python3 collect.py run --daily            today's New York date
     python3 collect.py run --date 2026-09-04  a specific date folder

Idempotent per date: a ticker whose file already exists in the day folder is
skipped, so a rerun only fills gaps. The manifest is rewritten on every run
from what is on disk.
"""
import argparse
import json
import sys

import pandas as pd

import data


def collect_day(date, now=None, tickers=data.UNIVERSE, fetch=None, rate=None, caps=None):
    """Collect every ticker for `date`. Fetch hooks exist so check.py can run offline."""
    now = pd.Timestamp.now("UTC") if now is None else now
    fetch = fetch or data.fetch_chain
    rate = rate or data.bill_rate
    caps = caps or data.market_caps
    d = data.day_dir(date)
    d.mkdir(parents=True, exist_ok=True)

    rates_path = d / "rates.csv"
    if not rates_path.exists():
        r, obs, series = rate(date)
        if r is None:
            print(f"{date}: FRED and Yahoo both unreachable, rate recorded as missing",
                  file=sys.stderr)
        pd.DataFrame([{"date": date, "series": series, "rate": r, "observation_date": obs}]) \
            .to_csv(rates_path, index=False)
    r = pd.read_csv(rates_path)["rate"].iloc[0]
    r = None if pd.isna(r) else float(r)

    written, skipped, missing, closes = [], [], [], []
    for t in tickers:
        path = d / f"{t}.csv"
        if path.exists():
            skipped.append(t)
            continue
        try:
            q, spot, hist_date = fetch(t, now, r)
        except Exception as e:
            print(f"{date}: {t} failed: {e}", file=sys.stderr)
            missing.append(t)
            continue
        # mode x refuses to clobber a file that appeared since the check above
        with open(path, "x") as f:
            q.to_csv(f, index=False)
        written.append(t)
        closes.append({"date": date, "ticker": t, "close": spot, "close_date": hist_date})

    under_path = d / "underlying.csv"
    if closes:
        old = pd.read_csv(under_path) if under_path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(closes)], ignore_index=True).to_csv(under_path, index=False)

    weights_path = d / "weights.csv"
    if not weights_path.exists():
        caps_series = caps(data.NAMES)
        if len(caps_series):
            caps_series.to_frame().to_csv(weights_path)

    collected = sorted(t for t in tickers if (d / f"{t}.csv").exists())
    rows = sum(len(pd.read_csv(d / f"{t}.csv", usecols=["strike"])) for t in collected)
    manifest = {
        "date": date,
        "snapshot_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session": data.session(now),
        "rate": r,
        "collected": collected,
        "written_this_run": written,
        "skipped_existing": skipped,
        "missing": missing,
        "rows": rows,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="collect one day")
    run.add_argument("--daily", action="store_true", help="use today's New York date")
    run.add_argument("--date", help="YYYY-MM-DD folder to collect into")
    a = p.parse_args(argv)

    now = pd.Timestamp.now("UTC")
    date = a.date or data.trading_date(now)
    if date != data.trading_date(now):
        # the quotes are always live; a folder named for another day holds today's chain
        print(f"warning: quotes fetched now, {data.trading_date(now)}, are being filed under "
              f"{date}; the manifest's snapshot_utc says when they were really taken",
              file=sys.stderr)
    if pd.Timestamp(date).weekday() >= 5 and not a.date:
        print(f"{date} is a weekend, nothing to collect")
        return 0
    m = collect_day(date, now)
    print(f"{date} [{m['session']}] wrote {len(m['written_this_run'])}, "
          f"skipped {len(m['skipped_existing'])}, missing {len(m['missing'])}, "
          f"{m['rows']} rows, r={m['rate']}")
    if m["session"] == "intraday":
        print("snapshot taken before the close; quotes are intraday, not closing")
    return 1 if m["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
