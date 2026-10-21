"""Offline checks on the loader, PIT filter, gap detector, CIK map, book deltas, 8-K parsing,
the 5-minute Kalshi books, options sessions and the storage guard.

Runs against a temporary store with planted data; the real store is never touched.
Run: ../.venv/bin/python3 check.py   (exits 1 on any failure)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import collect
import lake
import quality

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILS.append(name)


def synthetic_store():
    """Three equities over 2019-2020, one FRED series, one day of L2 with a planted gap."""
    days = pd.bdate_range("2019-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    for t in ("AAA", "BBB", "CCC"):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        df = pd.DataFrame({"date": days, "ticker": t, "open": px, "high": px * 1.01, "low": px * 0.99,
                           "close": px, "adj_close": px, "volume": 1000.0,
                           "collected_at": pd.Timestamp("2021-01-01")})
        lake.write("equities_daily", t, df)
    fred = pd.DataFrame({"date": days, "series": "DGS10", "value": 2.0, "collected_at": pd.Timestamp("2021-01-01")})
    lake.write("fred", "DGS10", fred)
    ts = pd.date_range("2026-01-01 10:00:00", periods=600, freq="1s", tz="UTC")
    ts = ts.delete(range(100, 130))
    l2 = pd.DataFrame({"ts": ts, "symbol": "BTCUSDT", "venue": "test", "bid_px_1": 1.0, "ask_px_1": 1.1})
    lake.write("crypto_l2", "2026-01-01/10", l2)


T0 = pd.Timestamp("2026-10-21 12:00:00", tz="UTC")


def mkt(ticker, bid, ask, hours=48, oi=0, vol=0):
    """One open-market row as the Kalshi API sends it: dollar fields are strings."""
    return {"ticker": ticker, "event_ticker": ticker.split("-")[0], "market_type": "binary", "status": "active",
            "yes_bid_dollars": bid, "yes_ask_dollars": ask, "no_bid_dollars": None, "no_ask_dollars": None,
            "last_price_dollars": None, "volume_fp": "0", "volume_24h_fp": str(vol), "open_interest_fp": str(oi),
            "liquidity_dollars": None, "open_time": None, "expiration_time": None,
            "close_time": None if hours is None else (T0 + pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")}


class FakeKalshi:
    """Stands in for kalshi.Client: a paged listing and an order book per ticker, every call recorded."""

    def __init__(self, markets, page=1000, books=None, fail_listing=False, fail_books=()):
        self.all, self.page, self.books = markets, page, books or {}
        self.fail_listing, self.fail_books = fail_listing, set(fail_books)
        self.pages, self.asked = [], []

    def get(self, path, auth=False, **params):
        assert path == "/markets" and not auth, "the 5-minute feed makes public listing calls only"
        if self.fail_listing:
            raise RuntimeError("listing down")
        i = int(params.get("cursor") or 0)
        self.pages.append(i)
        return {"markets": self.all[i:i + self.page], "cursor": str(i + self.page) if i + self.page < len(self.all) else ""}

    def markets(self, **params):          # the hourly feed's call
        return list(self.all)

    def orderbook(self, ticker, depth=10):
        self.asked.append(ticker)
        if ticker in self.fail_books:
            raise RuntimeError("404")
        return self.books.get(ticker, {"yes_dollars": [["0.30", "10"], ["0.31", "5"]], "no_dollars": [["0.40", "7"]]})


def plant_options(opt, date, session, spot):
    """A one-row chain in the layout the options collector writes: the day folder for close, a subfolder otherwise."""
    d = opt.session_dir(date, session)
    d.mkdir(parents=True, exist_ok=True)
    snap = pd.Timestamp(f"{date} 20:35:00")
    pd.DataFrame({"snapshot": [snap], "expiry": [snap.normalize() + pd.Timedelta(days=30)], "lastTradeDate": [snap],
                  "strike": [100.0], "spot": [spot]}).to_csv(d / "AAA.csv", index=False)
    (d / "manifest.json").write_text(json.dumps({"collected": ["AAA"], "session": session}))


def reset_books():
    shutil.rmtree(os.path.join(lake.STORE, "kalshi_books_5m"), ignore_errors=True)


def books_5m_checks():
    """The wide-book filter and cap, the page limit, the dataset round trip, and failed snapshots through bundle()."""
    print("kalshi books, 5 minutes")
    sleeps, qcalls = [], []
    real = (collect.kalshi_client, collect.now, collect.time.sleep, collect.quality)
    collect.now, collect.time.sleep, collect.quality = (lambda: T0), sleeps.append, (lambda: qcalls.append(1))
    try:
        # the filter: 0.35 - 0.25 is 0.0999... in floats (55 of the 89 ten-cent quotes are) and still counts as 10c
        ms = [mkt("EDGE-10C", "0.2500", "0.3500", oi=5), mkt("NARROW-9C", "0.4000", "0.4900", oi=99),
              mkt("SOON-23H", "0.2000", "0.6000", hours=23, oi=99), mkt("EXACT-24H", "0.2000", "0.6000", hours=24, oi=1),
              mkt("NOBID", "0.0000", "0.5000", oi=99), mkt("NOASK", "0.4000", "1.0000", oi=99),
              mkt("NOQUOTE", None, None, oi=99), mkt("NOCLOSE", "0.2000", "0.6000", hours=None, oi=99),
              mkt("BIGOI", "0.1000", "0.9000", oi=500), mkt("ZZZ-VOL", "0.2000", "0.6000", vol=100),
              mkt("AAA-NOVOL", "0.2000", "0.6000"), mkt("BBB-NOVOL", "0.2000", "0.6000")]
        wide = ["BIGOI", "EDGE-10C", "EXACT-24H", "ZZZ-VOL", "AAA-NOVOL", "BBB-NOVOL"]
        got = collect.wide_books(ms, T0)
        check("filter keeps two-sided, >= 10c (float edge included), >= 24h (exactly 24h included), and nothing else",
              set(got) == set(wide), got)
        check("ranked by open interest, then 24h volume, then ticker", got == wide, got)
        check("an empty listing selects nothing", collect.wide_books([], T0) == [])
        many = [mkt(f"M{i:03d}", "0.3000", "0.5000", oi=i) for i in range(400)]
        check("the cap is the 300 highest open interest, best first",
              collect.wide_books(many, T0) == [f"M{i:03d}" for i in range(399, 99, -1)])

        # the page limit and the pause between pages
        fake = FakeKalshi(many * 17, page=1000)
        listed = collect.kalshi_listing(fake)
        check("the listing stops at KALSHI_MAX_PAGES pages",
              fake.pages == [0, 1000, 2000, 3000, 4000] and len(listed) == 5000, fake.pages)
        check("it pauses between pages and not after the last", sleeps == [collect.PAGE_SLEEP] * 4, sleeps)
        sleeps.clear()
        fake = FakeKalshi(many[:250], page=100)
        check("a short listing ends on the empty cursor", len(collect.kalshi_listing(fake)) == 250 and fake.pages == [0, 100, 200])
        sleeps.clear()

        # failed snapshots exit nonzero through bundle(), and skip quality
        reset_books()
        for what, fake in (("a listing that fails", FakeKalshi([], fail_listing=True)),
                           ("a listing with no wide market", FakeKalshi([mkt("NARROW", "0.4000", "0.4900")])),
                           ("every book failing", FakeKalshi(ms, fail_books=[m["ticker"] for m in ms]))):
            collect.kalshi_client = lambda fake=fake: fake
            rc = collect.bundle(collect.kalshi_books, with_quality=False)
            check(f"{what} exits 1 through bundle(), writes nothing, skips quality",
                  rc == 1 and lake.parts("kalshi_books_5m") == [] and qcalls == [], rc)
        collect.kalshi_client = lambda: FakeKalshi(ms)
        check("a snapshot that works exits 0 through the same bundle()",
              collect.bundle(collect.kalshi_books, with_quality=False) == 0)
        check("bundle() still runs quality by default", collect.bundle(lambda: None) == 0 and qcalls == [1])
        qcalls.clear()
        reset_books()
        collect.kalshi_client = lambda: FakeKalshi(ms, fail_books=wide[:2])
        check("losing a third of the books is a degraded snapshot, not a failure",
              collect.bundle(collect.kalshi_books, with_quality=False) == 0 and lake.load("kalshi_books_5m").ticker.nunique() == 4)
        reset_books()
        collect.kalshi_client = lambda: FakeKalshi(ms, fail_books=wide[:4])
        check("losing more than half fails the run but keeps the books it got",
              collect.bundle(collect.kalshi_books, with_quality=False) == 1 and lake.load("kalshi_books_5m").ticker.nunique() == 2)
        reset_books()
        real_budget, collect.BOOKS_BUDGET = collect.BOOKS_BUDGET, -1
        fake = FakeKalshi(ms)
        collect.kalshi_client = lambda: fake
        check("a spent time budget stops fetching books and fails the run",
              collect.bundle(collect.kalshi_books, with_quality=False) == 1 and fake.asked == [])
        collect.BOOKS_BUDGET = real_budget

        # the round trip, the cap, the rate pause, the schema
        reset_books()
        sleeps.clear()
        fake = FakeKalshi(many)
        collect.kalshi_client = lambda: fake
        collect.kalshi_books()
        df = lake.load("kalshi_books_5m")
        check("400 wide markets: 300 books asked for, best open interest first",
              fake.asked == [f"M{i:03d}" for i in range(399, 99, -1)], len(fake.asked))
        check("each book is three levels, so 900 rows, one reason, one snapshot time",
              len(df) == 900 and set(df.reason) == {"wide_book"} and df.ts.nunique() == 1 and df.ticker.nunique() == 300)
        check("a pause before every book", sleeps == [collect.BOOK_SLEEP] * 300)
        check("hourly part named <date>/<HH>", lake.parts("kalshi_books_5m") == ["2026-10-21/12"], lake.parts("kalshi_books_5m"))
        top = df[df.ticker == "M399"].sort_values(["side", "price"])[["side", "price", "size"]].values.tolist()
        check("levels are stored as the book gave them", top == [["no", 0.40, 7.0], ["yes", 0.30, 10.0], ["yes", 0.31, 5.0]], top)
        collect.now = lambda: T0 + pd.Timedelta(minutes=5)
        collect.kalshi_books()
        collect.kalshi_books()
        df = lake.load("kalshi_books_5m")
        check("a second snapshot adds rows to the hour part, and a rerun of the same snapshot adds none",
              len(df) == 1800 and df.ts.nunique() == 2 and lake.parts("kalshi_books_5m") == ["2026-10-21/12"])
        collect.now = lambda: T0
        sleeps.clear()
        reset_books()
        collect.kalshi_client = lambda: FakeKalshi(ms)
        collect.kalshi()
        collect.kalshi_books()
        hourly, fast = lake.load("kalshi_books"), lake.load("kalshi_books_5m")
        check("same schema as the hourly books plus reason (names and dtypes)",
              list(fast.columns) == list(hourly.columns) + ["reason"] and (fast.dtypes[:5] == hourly.dtypes).all(),
              list(zip(fast.columns, fast.dtypes)))

        # quality sees the new dataset: gaps on the snapshot clock, freshness in minutes
        reset_books()
        snaps = [T0 + pd.Timedelta(minutes=5 * k) for k in range(12) if k not in (3, 4, 5)]
        rows = [(t, tk, "yes", 0.3, 1.0, "wide_book") for t in snaps
                for tk in (["A"] + (["B"] if t >= T0 + pd.Timedelta(minutes=35) else []))]
        lake.write("kalshi_books_5m", "2026-10-21/12",
                   pd.DataFrame(rows, columns=["ts", "ticker", "side", "price", "size", "reason"]),
                   keys=["ts", "ticker", "side", "price"])
        gaps, dupes = quality.scan("kalshi_books_5m", ["2026-10-21/12"])
        check("one planted 15-minute hole is one gap of 3 snapshots, and a ticker joining late is not a gap",
              [g["missing"] for g in gaps] == [3] and dupes == 0, gaps)
        d = quality.report(now=T0 + pd.Timedelta(minutes=59))["datasets"]["kalshi_books_5m"]
        check("fresh 4 minutes after the last snapshot", not d["stale"], d["last"])
        d = quality.report(now=T0 + pd.Timedelta(minutes=80))["datasets"]["kalshi_books_5m"]
        check("stale 25 minutes after it", d["stale"], d["last"])
    finally:
        collect.kalshi_client, collect.now, collect.time.sleep, collect.quality = real
        reset_books()


def options_session_checks(tmp):
    """lake.load('options') reads close by default and the others on request; the old layout still loads."""
    print("options sessions")
    opt = lake._options()
    real_root, opt.ROOT = opt.ROOT, Path(tmp) / "raw"
    try:
        (opt.ROOT / "2026-10-19").mkdir(parents=True)
        plant_options(opt, "2026-10-19", "close", 1.0)       # written before sessions existed: root files, no subfolders
        for s, spot in (("open", 2.0), ("midday", 3.0), ("close", 4.0)):
            plant_options(opt, "2026-10-20", s, spot)
        plant_options(opt, "2026-10-21", "open", 5.0)         # today: the close run has not started
        spots = lambda **kw: sorted(lake.load("options", **kw).spot)
        check("default is close, old single-file layout included", spots() == [1.0, 4.0], spots())
        check("session='open' and 'midday' read the others", spots(session="open") == [2.0, 5.0] and spots(session="midday") == [3.0])
        check("date windows still apply per session",
              spots(start="2026-10-20") == [4.0] and spots(session="open", end="2026-10-20") == [2.0])
        try:
            lake.load("equities_daily", session="open")
            raised = False
        except ValueError:
            raised = True
        check("session= on any other dataset is refused, not ignored", raised)
        d = quality.report(now=pd.Timestamp("2026-10-21 21:00", tz="UTC"))["datasets"]["options"]
        check("quality names the latest day of each session",
              d.get("sessions") == {"open": "2026-10-21", "midday": "2026-10-20", "close": "2026-10-20"}, d.get("sessions", d))
        check("options freshness follows close only: today's open does not move it",
              d["last"] == "2026-10-20 20:35:00" and not d["stale"], d["last"])
    finally:
        opt.ROOT = real_root


def storage_checks(tmp):
    """Size per dataset, growth over the trailing week from the previous status.json, runway against free space."""
    print("storage guard")
    opt = lake._options()
    real_root, real_free, opt.ROOT = opt.ROOT, quality.free_bytes, Path(tmp) / "raw"
    try:
        stray = os.path.join(lake.STORE, "crypto_l2 2")
        os.makedirs(stray)
        with open(os.path.join(stray, "copy.bin"), "wb") as fh:
            fh.write(b"x" * 1000)
        cur = quality.sizes()
        planted = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fs in os.walk(os.path.join(lake.STORE, "equities_daily")) for f in fs)
        check("equities_daily bytes are the sum of its part files", cur["equities_daily"] == planted > 0, cur["equities_daily"])
        check("a store directory the lake does not know is still counted", cur["crypto_l2 2"] == 1000)
        check("every registered dataset reports, empty ones as zero", set(lake.DATASETS) <= set(cur) and cur["kalshi_books_5m"] == 0)

        now = pd.Timestamp("2026-10-21 12:00", tz="UTC")
        os.makedirs(lake.REPORTS, exist_ok=True)
        status = os.path.join(lake.REPORTS, "status.json")

        def previous(*entries):
            hist = [{"at": (now - pd.Timedelta(hours=h)).isoformat(), "bytes": b} for h, b in entries]
            with open(status, "w") as fh:
                json.dump({"disk": {"history": hist}}, fh)

        if os.path.exists(status):
            os.remove(status)
        quality.free_bytes = lambda: 10 ** 12
        first = quality.storage(now)
        check("no history yet: growth and runway are unknown, one entry recorded",
              first["growth_per_day"] is None and first["runway_days"] is None and len(first["history"]) == 1)

        zeros = {n: 0 for n in cur}
        total = sum(cur.values())
        previous((7 * 24, zeros))
        d = quality.storage(now)
        check("growth is bytes now minus bytes a week ago, over seven days",
              abs(d["growth_per_day"] - total / 7) < 1e-6
              and abs(d["datasets"]["equities_daily"]["growth_per_day"] - cur["equities_daily"] / 7) < 1e-6
              and d["window_days"] == 7.0, d["growth_per_day"])
        quality.free_bytes = lambda: int(total / 7 * 100)
        d = quality.storage(now)
        check("runway is free space over that growth (100 days) and is not low", abs(d["runway_days"] - 100) < 0.01 and not d["low"], d["runway_days"])
        check("per dataset too", abs(d["datasets"]["equities_daily"]["runway_days"]
                                     - quality.free_bytes() / (cur["equities_daily"] / 7)) < 0.01)
        quality.free_bytes = lambda: int(total / 7 * 10)
        d = quality.storage(now)
        check("ten days of runway is low", d["low"] and abs(d["runway_days"] - 10) < 0.01, d["runway_days"])
        r = quality.report(now)
        check("and the lake summary says so, so the desk report shows it",
              "DISK LOW" in r["summary"] and not r["ok"] and r["disk"]["low"], r["summary"])
        quality.free_bytes = lambda: 10 ** 12
        summary = quality.report(now)["summary"]
        check("a healthy runway is in the summary without the alarm", "DISK LOW" not in summary and "disk full in" in summary, summary)
        previous((3, zeros))
        check("under half a day of history says nothing yet", quality.storage(now)["growth_per_day"] is None)
        previous((2, zeros))
        check("an entry under 23 hours old is not followed by another", len(quality.storage(now)["history"]) == 1)
        previous((30, zeros), (9 * 24, zeros))
        h = quality.storage(now)["history"]
        check("a day later one is added, and anything older than eight days is dropped",
              len(h) == 2 and h[-1]["at"] == now.isoformat(), [e["at"] for e in h])
        with open(status, "w") as fh:
            fh.write("{not json")
        check("an unreadable previous status.json restarts the history instead of crashing", len(quality.storage(now)["history"]) == 1)

        os.remove(status)
        quality.main()
        quality.main()
        with open(status) as fh:
            st = json.load(fh)
        check("main() writes the disk block and the second run keeps one daily entry",
              set(lake.DATASETS) <= set(st["disk"]["datasets"]) and len(st["disk"]["history"]) == 1
              and not os.path.exists(status + ".tmp"))
    finally:
        opt.ROOT, quality.free_bytes = real_root, real_free


def main():
    tmp = tempfile.mkdtemp()
    real_store, real_intervals, real_reports = lake.STORE, lake.INTERVALS, lake.REPORTS
    lake.STORE = os.path.join(tmp, "store")
    lake.REPORTS = os.path.join(tmp, "reports")
    lake.INTERVALS = os.path.join(tmp, "intervals.csv")
    real_opt_root = lake._options().ROOT
    lake._options().ROOT = Path(tmp) / "raw"      # quality reads options through the collector: keep it off the real raw folder
    pd.DataFrame([("AAA", "2019-01-01", "2020-12-31"), ("BBB", "2020-01-01", "2020-06-30"),
                  ("CCC", "2019-06-01", "2019-12-31")], columns=["ticker", "start", "end"]).to_csv(lake.INTERVALS, index=False)
    try:
        synthetic_store()

        print("round trip")
        full = lake.load("equities_daily")
        check("all rows back", len(full) == 3 * 523, f"{len(full)}")
        w = lake.load("equities_daily", "2019-03-01", "2019-03-31", universe=["AAA"])
        check("window and universe", len(w) == 21 and set(w.ticker) == {"AAA"} and w.date.min() == pd.Timestamp("2019-03-01"))
        check("sorted by date", w.date.is_monotonic_increasing)
        again = lake.write("equities_daily", "AAA", full[full.ticker == "AAA"].head(10))
        check("rewrite dedupes on keys", again == 523, f"{again}")
        n = lake.write("equities_daily", "AAA", full[full.ticker == "AAA"].head(1).assign(close=999.0))
        check("rewrite replaces the value", lake.load("equities_daily", universe=["AAA"]).close.iloc[0] == 999.0 and n == 523)

        print("point in time")
        for d in ("2019-06-14", "2020-03-31", "2020-12-30"):
            asof = lake.load("equities_daily", as_of=d)
            check(f"as_of {d} has no later rows", asof.date.max() <= pd.Timestamp(d), f"max {asof.date.max().date()}")
        late = lake.load("equities_daily", end="2020-12-31", as_of="2020-01-15")
        check("as_of beats end", late.date.max() == pd.Timestamp("2020-01-15"))
        u = lake.load("equities_daily", "2019-01-01", "2019-12-31", universe="sp500")
        check("sp500 excludes BBB before it joined", "BBB" not in set(u.ticker))
        check("sp500 keeps CCC only inside its spell", u[u.ticker == "CCC"].date.min() >= pd.Timestamp("2019-06-01")
              and u[u.ticker == "CCC"].date.max() <= pd.Timestamp("2019-12-31"))
        u2 = lake.load("equities_daily", "2020-07-01", "2020-12-31", universe="sp500")
        check("sp500 drops BBB after it left", set(u2.ticker) == {"AAA"})
        f = lake.load("fred", as_of="2019-02-01", universe=["DGS10"])
        check("fred as_of", f.date.max() <= pd.Timestamp("2019-02-01") and len(f) == 24)
        tz = lake.load("crypto_l2", start="2026-01-01 10:01:00", end="2026-01-01 10:02:00")
        check("tz-aware window on naive bounds", len(tz) == 40, f"{len(tz)}")
        tz = lake.load("crypto_l2", start=pd.Timestamp("2026-01-01 10:01:00", tz="UTC"), end=pd.Timestamp("2026-01-01 10:02:00", tz="UTC"))
        check("tz-aware bounds do not raise", len(tz) == 40, f"{len(tz)}")
        cap = lake.load("crypto_l2", end="2026-01-01", as_of="2026-01-01 10:05:00")
        check("as_of is an instant on intraday rows", cap.ts.max() <= pd.Timestamp("2026-01-01 10:05:00", tz="UTC") and len(cap) == 271, f"{len(cap)}")
        check("as_of at midnight is not widened to the day", len(lake.load("crypto_l2", as_of="2026-01-01")) == 0)

        print("gaps")
        g = lake.gaps(lake.load("crypto_l2").ts, "1s")
        check("one planted 30s gap", len(g) == 1 and int(g.missing.iloc[0]) == 30, g.to_dict("records"))
        check("no gap in a clean second series", len(lake.gaps(pd.date_range("2026-01-01", periods=100, freq="1s"), "1s")) == 0)
        hours = pd.date_range("2026-01-01", periods=48, freq="1h").delete([10, 11, 12, 30])
        g = lake.gaps(hours, "1h")
        check("hourly gaps found with counts", list(g.missing) == [3, 1], list(g.missing))
        b = pd.bdate_range("2026-01-05", "2026-01-30").delete([3, 4, 5, 15])
        g = lake.gaps(b, "1B")
        check("business-day gaps ignore weekends", list(g.missing) == [3, 1] and str(g.after.iloc[0].date()) == "2026-01-07", g.to_dict("records"))
        check("weekend alone is not a gap", len(lake.gaps(pd.bdate_range("2026-01-05", "2026-01-16"), "1B")) == 0)
        check("empty input", len(lake.gaps([], "1s")) == 0)

        print("cik map")
        table = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                 "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
                 "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
                 "3": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."}}
        m = collect.cik_map(["AAPL", "BRK-B", "GOOG", "GOOGL", "ZZZZ"], table)
        check("ten-digit zero padded", m["AAPL"] == "0000320193", m.get("AAPL"))
        check("dash class ticker maps", m["BRK-B"] == "0001067983")
        check("unknown ticker dropped", "ZZZZ" not in m)
        check("two tickers one cik", m["GOOG"] == m["GOOGL"])

        print("edgar parsing")
        sub = ("<SEC-HEADER>x</SEC-HEADER>\n<DOCUMENT>\n<TYPE>8-K\n<TEXT>\n<html><body><p>Item 2.02 Results of Operations"
               " and Financial Condition.</p><p>Item 9.01 Financial Statements &amp; Exhibits</p></body></html>\n</TEXT>\n</DOCUMENT>\n"
               "<DOCUMENT>\n<TYPE>EX-99.1\n<TEXT>\n<html><body><div>Revenue was $1.0&nbsp;billion.</div><script>x=1</script></body></html>\n</TEXT>\n</DOCUMENT>\n"
               "<DOCUMENT>\n<TYPE>GRAPHIC\n<TEXT>\nbegin 644 img.jpg\nAAAA\n</TEXT>\n</DOCUMENT>\n")
        types, text, items = collect.parse_submission(sub)
        check("keeps 8-K and EX-99, drops graphic", types == ["8-K", "EX-99.1"], types)
        check("items parsed", items == ["2.02", "9.01"], items)
        check("html stripped and entities decoded", "Revenue was $1.0 billion." in text and "<" not in text and "x=1" not in text)
        idx = ("Form Type   Company Name   CIK   Date Filed   File Name\n"
               "8-K         APPLE INC                   320193      20260903    edgar/data/320193/0000320193-26-000001.txt\n"
               "8-K/A       ACME CORP                   12345       20260903    edgar/data/12345/0000012345-26-000002.txt\n"
               "10-Q        OTHER CO                    99          20260903    edgar/data/99/0000000099-26-000003.txt\n")
        rows = [l for l in idx.splitlines() if l.startswith(("8-K ", "8-K/A "))]
        check("index filter keeps both 8-K forms", len(rows) == 2)

        print("book deltas")
        book = {"bid": {}, "ask": {}}
        snap = json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "snapshot",
                           "data": {"s": "BTCUSDT", "b": [["100", "1"], ["99", "2"], ["98", "3"]], "a": [["101", "1"], ["102", "2"]]}})
        delta = json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "delta",
                            "data": {"s": "BTCUSDT", "b": [["100", "0"], ["99.5", "5"]], "a": [["101", "0.5"]]}})
        books = {}
        collect.parse_msg("bybit", snap, books)
        collect.parse_msg("bybit", delta, books)
        t = collect.top(books["BTCUSDT"], 3)
        check("bybit delete and insert", t["bid_px_1"] == 99.5 and t["bid_sz_1"] == 5 and t["bid_px_2"] == 99 and t["bid_px_3"] == 98, t)
        check("bybit size update", t["ask_px_1"] == 101 and t["ask_sz_1"] == 0.5)
        okx = json.dumps({"arg": {"channel": "books", "instId": "ETH-USDT"}, "action": "snapshot",
                          "data": [{"bids": [["10", "1", "0", "1"]], "asks": [["11", "1", "0", "1"], ["12", "2", "0", "1"]]}]})
        okx2 = json.dumps({"arg": {"channel": "books", "instId": "ETH-USDT"}, "action": "update",
                           "data": [{"bids": [["10", "0", "0", "0"], ["9", "4", "0", "1"]], "asks": [["12", "0", "0", "0"]]}]})
        collect.parse_msg("okx", okx, books)
        collect.parse_msg("okx", okx2, books)
        t = collect.top(books["ETHUSDT"], 2)
        check("okx delete and insert", t["bid_px_1"] == 9 and t["ask_px_1"] == 11 and t["ask_px_2"] is None, t)
        check("pong ignored", collect.parse_msg("okx", "pong", books) is None)
        check("top pads to depth", len(collect.top(books["ETHUSDT"], 20)) == 80)

        print("quality")
        r = quality.report(now=pd.Timestamp("2026-01-01 10:12:00", tz="UTC"))
        d = r["datasets"]["crypto_l2"]
        check("l2 fresh inside 3 minutes of the last row", not d["stale"] and d["gaps"][0]["missing"] == 30, d)
        r = quality.report(now=pd.Timestamp("2026-01-01 12:00:00", tz="UTC"))
        check("l2 stale two hours later", r["datasets"]["crypto_l2"]["stale"])
        check("equities stale in 2026 against a 2020 store", r["datasets"]["equities_daily"]["stale"])
        check("summary names stale feeds", "crypto_l2" in r["summary"] and not r["ok"])
        r = quality.report(now=pd.Timestamp("2021-01-01", tz="UTC"))
        check("equities fresh the day after the last bar", not r["datasets"]["equities_daily"]["stale"])
        check("no duplicates counted", r["datasets"]["equities_daily"]["dupes"] == 0)

        books_5m_checks()
        options_session_checks(tmp)
        storage_checks(tmp)
    finally:
        lake.STORE, lake.INTERVALS, lake.REPORTS = real_store, real_intervals, real_reports
        lake._options().ROOT = real_opt_root
        shutil.rmtree(tmp, ignore_errors=True)
    _q = collect.quality
    collect.quality = lambda: None
    try:
        ok_rc = collect.bundle(lambda: None)
        def boom():
            raise RuntimeError("vendor down")
        bad_rc = collect.bundle(lambda: None, boom, lambda: None)
    finally:
        collect.quality = _q
    if not (ok_rc == 0 and bad_rc == 1):
        FAILS.append(f"bundle exit codes: ok {ok_rc}, failing step {bad_rc}")
    else:
        print("  ok  a bundle exits 0 when every step ran and 1 when one failed, after quality()")
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
