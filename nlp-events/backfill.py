"""Backfill earnings 8-Ks (item 2.02) into the lake's edgar_8k dataset from the SEC submissions API.

The lake's own collector walks the daily form index and reaches back three
days. This fills START onward for every name that was in the point-in-time
S&P 500 over the span, keeping the first EX-99 exhibit (the press release)
of each filing and the acceptance timestamp, which the collector's daily
index does not carry. Rows go through lake.write with the collector's
columns plus `accepted`. Idempotent and resumable: an accession already in
the store is skipped, and --minutes stops cleanly so the run can continue.

Run: ../.venv/bin/python3 backfill.py [--minutes 9]
"""
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data-lake"))
import collect  # noqa: E402
import lake  # noqa: E402

START, END = "2022-01-01", "2026-08-31"
UA = collect.UA
WORKERS = 8
GROUP = 4  # tickers whose filings share one pool pass
SLEEP = 0.7  # per worker between requests: 8 workers stay under the SEC's 10 a second


def get(url, tries=3):
    for k in range(tries):
        time.sleep(SLEEP)
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if k == tries - 1:
                collect.log(f"{url}: {e}")
                return None
            time.sleep(2)
    return None


def earnings_filings(cik, start=START, end=END):
    """8-K rows with item 2.02 for one CIK between start and end, paging into the older submission files."""
    r = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if r is None:
        return pd.DataFrame()
    j = r.json()
    frames = [pd.DataFrame(j["filings"]["recent"])]
    for f in j["filings"].get("files", []):
        if frames[-1]["filingDate"].min() <= start:
            break
        r = get("https://data.sec.gov/submissions/" + f["name"])
        if r is None:
            break
        frames.append(pd.DataFrame(r.json()))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(["8-K", "8-K/A"]) & df["items"].str.contains("2.02", regex=False)]
    df = df[(df["filingDate"] >= start) & (df["filingDate"] <= end)]
    return df.assign(cik=cik)


def exhibit(cik, accession):
    """(docs, text, url) of the filing's first EX-99 exhibit, or its primary document if it has none."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
    r = get(base + accession + "-index.html")
    if r is None:
        return None
    pick = None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        href = re.findall(r'href="([^"]+)"', row)
        if len(cells) < 4 or not href:
            continue
        typ = cells[3]
        if typ.startswith("EX-99") and not typ.startswith("EX-99.SUP"):
            pick = (typ, href[0])
            break
        if typ.startswith("8-K") and pick is None:
            pick = (typ, href[0])
    if pick is None:
        return None
    typ, href = pick
    href = href.replace("/ix?doc=", "")
    r = get("https://www.sec.gov" + href)
    if r is None:
        return None
    return typ, f"[{typ}]\n" + collect.strip_html(r.text), "https://www.sec.gov" + href


def fetch(row):
    out = exhibit(row["cik"], row["accessionNumber"])
    if out is None:
        return None
    docs, text, url = out
    return {"filed": pd.Timestamp(row["filingDate"]), "ticker": row["ticker"], "cik": row["cik"],
            "accession": row["accessionNumber"], "form": row["form"], "items": row["items"],
            "earnings": True, "docs": docs, "text": text, "url": url,
            "accepted": pd.Timestamp(row["acceptanceDateTime"]).tz_convert("UTC"),
            "collected_at": collect.now().tz_localize(None)}


def main(minutes=9.0):
    t0 = time.time()
    iv = lake.sp500_intervals()
    tickers = sorted(set(iv.loc[iv["end"] >= START, "ticker"]))
    ciks = collect.cik_map(tickers)
    have = lake.load("edgar_8k", START, END)
    have = set(have["accession"]) if len(have) else set()
    collect.log(f"{len(tickers)} tickers, {len(ciks)} with a CIK, {len(have)} filings already stored")
    done_file = os.path.join(HERE, "reports", "backfill_done.txt")
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
    todo = [(t, c) for t, c in ciks.items() if t not in done]
    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(todo), GROUP):
            if time.time() - t0 > minutes * 60:
                collect.log(f"stopping after {i} tickers; {len(todo) - i} left, rerun to continue")
                return
            group = todo[i:i + GROUP]
            frames = [earnings_filings(c).assign(ticker=t) for t, c in group if c]
            f = pd.concat([x for x in frames if len(x)], ignore_index=True) if any(len(x) for x in frames) \
                else pd.DataFrame(columns=["accessionNumber"])
            f = f[~f["accessionNumber"].isin(have)]
            rows = [r for r in pool.map(fetch, [r for _, r in f.iterrows()]) if r is not None]
            n = 0
            if rows:
                df = pd.DataFrame(rows)
                for day, part in df.groupby(df["filed"].dt.strftime("%Y-%m-%d")):
                    lake.write("edgar_8k", day, part, keys=["accession"])
                    n += len(part)
                have |= set(df["accession"])
            with open(done_file, "a") as fh:
                fh.write("".join(t + "\n" for t, _ in group))
            collect.log(f"{', '.join(t for t, _ in group)}: {len(f)} new earnings 8-Ks, {n} written "
                        f"({i + len(group)}/{len(todo)})")
    collect.log(f"backfill complete, {len(have)} filings in store, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    m = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 9.0
    main(m)
