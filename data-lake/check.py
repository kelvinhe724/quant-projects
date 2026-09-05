"""Offline checks on the loader, PIT filter, gap detector, CIK map, book deltas and 8-K parsing.

Runs against a temporary store with planted data; the real store is never touched.
Run: ../.venv/bin/python3 check.py   (exits 1 on any failure)
"""
import json
import os
import shutil
import sys
import tempfile

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


def main():
    tmp = tempfile.mkdtemp()
    real_store, real_intervals, real_reports = lake.STORE, lake.INTERVALS, lake.REPORTS
    lake.STORE = os.path.join(tmp, "store")
    lake.REPORTS = os.path.join(tmp, "reports")
    lake.INTERVALS = os.path.join(tmp, "intervals.csv")
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
    finally:
        lake.STORE, lake.INTERVALS, lake.REPORTS = real_store, real_intervals, real_reports
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
