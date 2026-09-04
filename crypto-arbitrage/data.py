"""Public REST quotes from four exchanges, sampled together every few seconds, plus OHLC history.

Run: python3 data.py [minutes]    samples top-of-book for that long and writes quotes.csv
Binance.com (HTTP 451) and Bybit (CloudFront 403) are geo-blocked from the US, so the
panel is Coinbase, Kraken, Binance.US and OKX. OKX's USD book is thin, so its USDT
book is sampled and converted at Kraken's USDT/USD mid, which is stored alongside.
"""
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "source-material", "crypto-arbitrage")
QUOTES = os.path.join(CACHE, "quotes.csv")
SYMBOLS = ["BTC", "ETH"]
INTERVAL_S = 3.0
TIMEOUT_S = 2.5

session = requests.Session()
session.headers["User-Agent"] = "crypto-arbitrage-study/0.1"


def get(url, params=None):
    r = session.get(url, params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def coinbase(sym):
    d = get(f"https://api.exchange.coinbase.com/products/{sym}-USD/ticker")
    return float(d["bid"]), float(d["ask"])


def binance_us(sym):
    d = get("https://api.binance.us/api/v3/ticker/bookTicker", {"symbol": f"{sym}USD"})
    return float(d["bidPrice"]), float(d["askPrice"])


KRAKEN_KEYS = {"BTC": "XXBTZUSD", "ETH": "XETHZUSD", "USDT": "USDTZUSD"}


def kraken_all():
    """Fetch BTC, ETH and USDT against USD in one call; Kraken's public limit is about 1/s."""
    d = get("https://api.kraken.com/0/public/Ticker", {"pair": "XBTUSD,ETHUSD,USDTUSD"})
    if d["error"]:
        raise RuntimeError(d["error"])
    return {s: (float(d["result"][k]["b"][0]), float(d["result"][k]["a"][0]))
            for s, k in KRAKEN_KEYS.items()}


def okx(sym):
    d = get("https://www.okx.com/api/v5/market/ticker", {"instId": f"{sym}-USDT"})
    row = d["data"][0]
    return float(row["bidPx"]), float(row["askPx"])


FETCHERS = {"coinbase": coinbase, "binance_us": binance_us, "okx": okx}
FIELDS = ["tick", "t_local", "exchange", "symbol", "bid", "ask", "latency_ms",
          "raw_bid", "raw_ask", "usdt_usd"]


def timed(fn, *args):
    t0 = time.time()
    try:
        out = fn(*args)
    except Exception as e:
        return None, t0, time.time(), repr(e)[:80]
    return out, t0, time.time(), ""


def sample_tick(tick, pool):
    """Fire every request for one tick concurrently and return one row per exchange-symbol."""
    jobs = {(ex, s): pool.submit(timed, fn, s) for ex, fn in FETCHERS.items() for s in SYMBOLS}
    jobs[("kraken", "all")] = pool.submit(timed, kraken_all)
    results = {k: f.result() for k, f in jobs.items()}

    rows, errors = [], []
    kr, k0, k1, kerr = results.pop(("kraken", "all"))
    usdt = None
    if kr:
        usdt = (kr["USDT"][0] + kr["USDT"][1]) / 2
        for s in SYMBOLS:
            rows.append(dict(tick=tick, t_local=(k0 + k1) / 2, exchange="kraken", symbol=s,
                             bid=kr[s][0], ask=kr[s][1], latency_ms=round((k1 - k0) * 1000),
                             raw_bid=kr[s][0], raw_ask=kr[s][1], usdt_usd=usdt))
    else:
        errors.append(f"kraken {kerr}")

    for (ex, s), (q, t0, t1, err) in results.items():
        if q is None:
            errors.append(f"{ex}/{s} {err}")
            continue
        bid, ask = q
        row = dict(tick=tick, t_local=(t0 + t1) / 2, exchange=ex, symbol=s, bid=bid, ask=ask,
                   latency_ms=round((t1 - t0) * 1000), raw_bid=bid, raw_ask=ask, usdt_usd=usdt)
        if ex == "okx":
            if usdt is None:
                errors.append("okx skipped, no USDT rate this tick")
                continue
            row["bid"], row["ask"] = bid * usdt, ask * usdt
        rows.append(row)
    return rows, errors


def sample(minutes, path=QUOTES, interval=INTERVAL_S):
    """Poll all venues every `interval` seconds for `minutes`, appending to CSV as it goes."""
    os.makedirs(CACHE, exist_ok=True)
    fresh = not os.path.exists(path)
    end = time.time() + minutes * 60
    n_rows = n_err = 0
    with open(path, "a", newline="") as f, ThreadPoolExecutor(max_workers=8) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        tick = 0
        try:
            while time.time() < end:
                t_next = time.time() + interval
                rows, errors = sample_tick(tick, pool)
                w.writerows(rows)
                f.flush()
                n_rows += len(rows)
                n_err += len(errors)
                if errors:
                    print(f"tick {tick}: " + "; ".join(errors), file=sys.stderr)
                if tick % 100 == 0:
                    print(f"tick {tick}  rows {n_rows}  errors {n_err}  "
                          f"{datetime.now(timezone.utc):%H:%M:%S}Z", flush=True)
                tick += 1
                time.sleep(max(0.0, t_next - time.time()))
        except KeyboardInterrupt:
            print("stopped by user", file=sys.stderr)
    print(f"done: {tick} ticks, {n_rows} rows, {n_err} failed requests -> {path}")
    return path


def load_quotes(path=QUOTES):
    q = pd.read_csv(path)
    q["t_local"] = pd.to_datetime(q["t_local"], unit="s", utc=True)
    return q


def coinbase_candles(sym, granularity, days):
    """Page Coinbase candles backwards; the API caps each request at 300 rows."""
    step = timedelta(seconds=granularity * 300)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    out = []
    while end > start:
        lo = max(start, end - step)
        d = get(f"https://api.exchange.coinbase.com/products/{sym}-USD/candles",
                {"granularity": granularity, "start": lo.isoformat(), "end": end.isoformat()})
        out += d
        end = lo
        time.sleep(0.2)
    df = pd.DataFrame(out, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.drop_duplicates("time").set_index("time").sort_index()[["open", "high", "low", "close", "volume"]]


def kraken_candles(sym, interval_min):
    """Kraken returns the most recent 720 candles for the interval, whatever `since` says."""
    d = get("https://api.kraken.com/0/public/OHLC", {"pair": f"{'XBT' if sym == 'BTC' else sym}USD",
                                                     "interval": interval_min})
    rows = d["result"][KRAKEN_KEYS[sym]]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    df = df.set_index("time").astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def load_history(sym, freq):
    """Return {exchange: OHLC frame} for 'hourly' (30 days) or 'daily' (720 days), cached."""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for ex in ("coinbase", "kraken"):
        path = os.path.join(CACHE, f"{ex}_{sym}_{freq}.csv")
        if os.path.exists(path):
            out[ex] = pd.read_csv(path, index_col=0, parse_dates=True)
            continue
        if ex == "coinbase":
            df = coinbase_candles(sym, 3600 if freq == "hourly" else 86400, 30 if freq == "hourly" else 720)
        else:
            df = kraken_candles(sym, 60 if freq == "hourly" else 1440)
        df.to_csv(path)
        out[ex] = df
    return out


if __name__ == "__main__":
    sample(float(sys.argv[1]) if len(sys.argv) > 1 else 55)
