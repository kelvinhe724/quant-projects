"""Snapshot the option chains up to three times per trading day and never overwrite one.

Run: python3 collect.py run --daily                    today's New York date, close session
     python3 collect.py run --daily --session midday   open | midday | close (default close)
     python3 collect.py run --date 2026-09-04          a specific date folder

Idempotent per date and session: a ticker whose file already exists in that
session's folder is skipped, so a rerun only fills gaps. The manifest is
rewritten on every run from what is on disk. close is the day folder itself;
open and midday are <date>/open/ and <date>/midday/.
"""
import argparse
import json
import sys

import pandas as pd

import data


def collect_day(date, now=None, tickers=data.UNIVERSE, fetch=None, rate=None, caps=None, session="close"):
    """Collect every ticker for `date` into one session's folder. Fetch hooks exist so check.py can run offline."""
    now = pd.Timestamp.now("UTC") if now is None else now
    fetch = fetch or data.fetch_chain
    rate = rate or data.bill_rate
    caps = caps or data.market_caps
    d = data.session_dir(date, session)
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
        # close keeps the clock label (a close run before 4pm is still "intraday"); open and midday are their own
        "session": data.session(now) if session == "close" else session,
        "rate": r,
        "collected": collected,
        "written_this_run": written,
        "skipped_existing": skipped,
        "missing": missing,
        "rows": rows,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None, now=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="collect one day")
    run.add_argument("--daily", action="store_true", help="use today's New York date")
    run.add_argument("--date", help="YYYY-MM-DD folder to collect into")
    run.add_argument("--session", choices=data.SESSIONS, default="close",
                     help="which snapshot of the day; close is the one the summaries are built from")
    a = p.parse_args(argv)

    now = pd.Timestamp.now("UTC") if now is None else now
    date = a.date or data.trading_date(now)
    if date != data.trading_date(now):
        # the quotes are always live; a folder named for another day holds today's chain
        print(f"warning: quotes fetched now, {data.trading_date(now)}, are being filed under "
              f"{date}; the manifest's snapshot_utc says when they were really taken",
              file=sys.stderr)
    if pd.Timestamp(date).weekday() >= 5 and not a.date:
        print(f"{date} is a weekend, nothing to collect")
        return 0
    if a.session != "close" and data.session(now) == "close":
        # launchd runs a missed job when the Mac wakes; a 5pm chain must not be filed as the open
        print(f"the cash session is over, not filing a {a.session} snapshot taken now", file=sys.stderr)
        return 1
    m = collect_day(date, now, session=a.session)
    print(f"{date} [{m['session']}] wrote {len(m['written_this_run'])}, "
          f"skipped {len(m['skipped_existing'])}, missing {len(m['missing'])}, "
          f"{m['rows']} rows, r={m['rate']}")
    if m["session"] == "intraday":
        print("snapshot taken before the close; quotes are intraday, not closing")
    # A few names the vendor would not serve is a bad day, not a failed run: the
    # folder is written, the misses are on stderr, and a rerun fills the gaps.
    # Only a majority miss means the vendor is down and the day is worthless.
    if len(m["missing"]) > data.MISSING_ABORT_FRACTION * len(data.UNIVERSE):
        print(f"{len(m['missing'])} of {len(data.UNIVERSE)} tickers missing "
              f"({', '.join(m['missing'])}); the vendor looks down", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
