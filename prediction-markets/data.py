"""Pull settled binary markets from the public Kalshi API and cache them as CSV."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "source-material",
    "kalshi",
    "settled_markets.csv",
)
PAGES_PER_SERIES = 4
PAGE_SIZE = 1000
THREADS = 16


def list_series():
    """Return every series ticker with its category."""
    r = requests.get(API + "/series", timeout=120)
    r.raise_for_status()
    return [(s["ticker"], s.get("category", "")) for s in r.json()["series"]]


def _fetch_series(args):
    ticker, category = args
    session = requests.Session()
    rows = []
    cursor = None
    for _ in range(PAGES_PER_SERIES):
        params = {"limit": PAGE_SIZE, "status": "settled", "series_ticker": ticker}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(4):
            try:
                r = session.get(API + "/markets", params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                payload = r.json()
                break
            except requests.RequestException:
                if attempt == 3:
                    return rows
                time.sleep(1.5 * (attempt + 1))
        else:
            return rows
        markets = payload.get("markets", [])
        for m in markets:
            row = _parse(m, ticker, category)
            if row is not None:
                rows.append(row)
        cursor = payload.get("cursor")
        if not cursor or not markets:
            break
    return rows


def _parse(m, series_ticker, category):
    """Turn one API market object into a row, or None if it is not usable."""
    if m.get("market_type") != "binary":
        return None
    result = m.get("result")
    if result not in ("yes", "no"):
        return None
    try:
        price = float(m.get("last_price_dollars"))
        volume = float(m.get("volume_fp", 0))
    except (TypeError, ValueError):
        return None
    if volume <= 0 or not 0.0 < price < 1.0:
        return None
    ticker = m.get("ticker", "")
    if ticker.startswith("KXMVE"):
        return None
    return {
        "ticker": ticker,
        "series": series_ticker,
        "category": category,
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "price": price,
        "volume": volume,
        "open_interest": float(m.get("open_interest_fp", 0) or 0),
        "outcome": 1 if result == "yes" else 0,
    }


def download():
    series = list_series()
    print(f"{len(series)} series, pulling settled markets with {THREADS} threads")
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as pool:
        for i, chunk in enumerate(pool.map(_fetch_series, series)):
            rows.extend(chunk)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(series)} series, {len(rows)} markets, "
                      f"{time.time() - t0:.0f}s")
    df = pd.DataFrame(rows).drop_duplicates(subset="ticker")
    df = enrich(df)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"wrote {len(df)} markets to {CACHE}")
    return df


def enrich(df):
    """Add parsed times and market lifetime in hours."""
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], format="ISO8601", utc=True)
    df["duration_hours"] = (
        df["close_time"] - df["open_time"]
    ).dt.total_seconds() / 3600.0
    return df.sort_values("close_time").reset_index(drop=True)


def load(refresh=False):
    if refresh or not os.path.exists(CACHE):
        return download()
    df = pd.read_csv(CACHE)
    return enrich(df)


# The market list only carries last_price_dollars, which is the last trade at an
# unknown time. On a thin market that trade can be days stale, so the quoted book
# at a fixed horizon before close is used instead.

SNAPSHOT = CACHE.replace("settled_markets.csv", "snapshot_24h.csv")
HORIZON_HOURS = 24
LOOKBACK_HOURS = 12
MIN_LIFETIME_HOURS = 36
SAMPLE_SIZE = 24_000
SNAPSHOT_THREADS = 48
SNAPSHOT_CHUNK = 2_000


def _quote(row):
    """Return the last valid bid/ask quoted at or before the horizon."""
    target = int(row.close_time.timestamp()) - HORIZON_HOURS * 3600
    url = (f"{API}/series/{row.series}/markets/{row.ticker}/candlesticks")
    params = {
        "start_ts": target - LOOKBACK_HOURS * 3600,
        "end_ts": target,
        "period_interval": 60,
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            bars = r.json().get("candlesticks", [])
            break
        except requests.RequestException:
            if attempt == 2:
                return None
            time.sleep(1.0 * (attempt + 1))
    else:
        return None

    for bar in reversed(bars):
        try:
            bid = float(bar["yes_bid"]["close_dollars"])
            ask = float(bar["yes_ask"]["close_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < bid < ask < 1:
            return {
                "ticker": row.ticker,
                "series": row.series,
                "category": row.category,
                "close_time": row.close_time,
                "bid": bid,
                "ask": ask,
                "price": (bid + ask) / 2,
                "spread": ask - bid,
                "last_price": row.price,
                "volume": row.volume,
                "duration_hours": row.duration_hours,
                "outcome": row.outcome,
            }
    return None


def download_snapshot(seed=0):
    """Fetch a T-24h quote per sampled market, appending each chunk to disk.

    Resumable: tickers already present in the cache, and those recorded as tried
    but unquotable, are skipped on a rerun.
    """
    df = load()
    eligible = df[df["duration_hours"] >= MIN_LIFETIME_HOURS]
    n = min(SAMPLE_SIZE, len(eligible))
    sample = eligible.sample(n, random_state=seed)

    tried_path = SNAPSHOT.replace(".csv", "_tried.txt")
    tried = set()
    if os.path.exists(tried_path):
        with open(tried_path) as fh:
            tried = {line.strip() for line in fh if line.strip()}
    todo = sample[~sample["ticker"].isin(tried)]
    print(f"{len(eligible):,} eligible markets, sample {n:,}, "
          f"{len(tried):,} already tried, {len(todo):,} to go, "
          f"{SNAPSHOT_THREADS} threads")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(SNAPSHOT_THREADS) as pool:
        for start in range(0, len(todo), SNAPSHOT_CHUNK):
            chunk = todo.iloc[start:start + SNAPSHOT_CHUNK]
            rows = [r for r in pool.map(_quote, chunk.itertuples(index=False))
                    if r is not None]
            if rows:
                frame = pd.DataFrame(rows)
                header = not os.path.exists(SNAPSHOT)
                frame.to_csv(SNAPSHOT, mode="a", header=header, index=False)
            with open(tried_path, "a") as fh:
                fh.write("".join(t + "\n" for t in chunk["ticker"]))
            done += len(chunk)
            print(f"  {done:,}/{len(todo):,} tried, +{len(rows):,} quoted, "
                  f"{time.time() - t0:.0f}s", flush=True)

    snap = load_snapshot()
    print(f"wrote {len(snap):,} quoted markets to {SNAPSHOT}")
    return snap


def load_snapshot(refresh=False):
    if refresh or not os.path.exists(SNAPSHOT):
        return download_snapshot()
    snap = pd.read_csv(SNAPSHOT).drop_duplicates(subset="ticker")
    snap["close_time"] = pd.to_datetime(snap["close_time"], format="ISO8601", utc=True)
    return snap.sort_values("close_time").reset_index(drop=True)


if __name__ == "__main__":
    load(refresh=not os.path.exists(CACHE))
    s = download_snapshot()
    print(s.groupby("category").size().sort_values(ascending=False))
