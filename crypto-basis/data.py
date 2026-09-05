"""Binance and OKX archives into the data lake, and the bar panels this project reads back out of it.

Binance's REST API is geo-blocked from the US; the monthly zips at
data.binance.vision are not. OKX publishes daily trade dumps on its CDN and
no candles, so its 1-minute bars are built here from the trades. Every file
is cached under source-material/crypto-basis/ and written into the lake once;
loads go through lake.load.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data-lake"))
import lake  # noqa: E402

CACHE = os.path.join(ROOT, "source-material", "crypto-basis")
VISION = "https://data.binance.vision/data"
OKX = "https://static.okx.com/cdn/okex/traderecords/trades/daily"

SYMBOLS = ("BTCUSDT", "ETHUSDT")
START, END = "2024-01", "2026-07"
OKX_DAYS = ("2026-07-01", "2026-07-31")

# ponytail: the lake's DATASETS dict is its registry; these three are added at import so
# lake.write/load accept them without editing data-lake/lake.py.
for name in ("crypto_perp_klines_1m", "okx_trades_1m"):
    lake.DATASETS.setdefault(name, ("ts", "symbol", "month", "1min"))
lake.DATASETS.setdefault("crypto_funding", ("ts", "symbol", "month", "8h"))

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def fetch(url, name):
    """Download once to the cache; return the first CSV in the zip as bytes, or None on 404."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        r = requests.get(url, timeout=180)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        open(path + ".tmp", "wb").write(r.content)
        os.replace(path + ".tmp", path)
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0])


def to_utc(t):
    """Binance stamps are milliseconds until 2025 and microseconds after, in the same column."""
    t = pd.to_numeric(t)
    return pd.to_datetime(np.where(t > 1e15, t // 1000, t), unit="ms", utc=True)


def read_csv(raw, cols):
    df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
    if df.iloc[0, 0] == cols[0]:
        df = df.iloc[1:]
    df.columns = cols
    return df


def klines(raw, symbol):
    df = read_csv(raw, KLINE_COLS)
    out = df.drop(columns=["open_time", "ignore"]).apply(pd.to_numeric)
    out.insert(0, "ts", to_utc(df["open_time"]))
    out.insert(1, "symbol", symbol)
    return out


def months(start=START, end=END):
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def collect_binance(symbols=SYMBOLS, start=START, end=END):
    """Spot 1m, perp 1m and funding for each month into the lake. Skips months already stored."""
    have = {d: set(lake.parts(d)) for d in ("crypto_klines_1m", "crypto_perp_klines_1m", "crypto_funding")}
    for m in months(start, end):
        for s in symbols:
            if m not in have["crypto_klines_1m"] or not _stored("crypto_klines_1m", m, s):
                raw = fetch(f"{VISION}/spot/monthly/klines/{s}/1m/{s}-1m-{m}.zip", f"{s}-spot-1m-{m}.zip")
                if raw is not None:
                    lake.write("crypto_klines_1m", m, klines(raw, s).assign(collected_at=pd.Timestamp.now("UTC")))
            if not _stored("crypto_perp_klines_1m", m, s):
                raw = fetch(f"{VISION}/futures/um/monthly/klines/{s}/1m/{s}-1m-{m}.zip", f"{s}-perp-1m-{m}.zip")
                if raw is not None:
                    lake.write("crypto_perp_klines_1m", m, klines(raw, s).assign(collected_at=pd.Timestamp.now("UTC")))
            if not _stored("crypto_funding", m, s):
                raw = fetch(f"{VISION}/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip",
                            f"{s}-funding-{m}.zip")
                if raw is not None:
                    df = read_csv(raw, FUNDING_COLS)
                    out = pd.DataFrame({"ts": to_utc(df["calc_time"]), "symbol": s,
                                        "rate": pd.to_numeric(df["last_funding_rate"]),
                                        "collected_at": pd.Timestamp.now("UTC")})
                    lake.write("crypto_funding", m, out)
        print(f"binance {m} done", flush=True)


def _stored(dataset, month, symbol):
    p = lake.part_path(dataset, month)
    if not os.path.exists(p):
        return False
    df = pd.read_parquet(p, columns=["ts", "symbol"])
    df = df[df["symbol"] == symbol]
    # a whole month of 1m bars is at least 28 days; funding is 3 a day
    need = 28 * 3 if dataset == "crypto_funding" else 28 * 1440
    return len(df) >= need


def okx_minutes(raw, inst):
    """1-minute last, vwap, volume and trade count from one OKX daily trades file."""
    df = pd.read_csv(io.BytesIO(raw))
    df["ts"] = pd.to_datetime(df["created_time"], unit="ms", utc=True).dt.floor("min")
    df["notional"] = df["price"] * df["size"]
    g = df.groupby("ts")
    out = pd.DataFrame({"last": g["price"].last(), "vwap": g["notional"].sum() / g["size"].sum(),
                        "volume": g["size"].sum(), "trades": g.size()})
    out = out.reset_index()
    out.insert(1, "symbol", inst)
    return out


def collect_okx(insts=("BTC-USDT", "BTC-USDT-SWAP", "ETH-USDT", "ETH-USDT-SWAP"), days=OKX_DAYS):
    for d in pd.date_range(*days):
        ds = d.strftime("%Y-%m-%d")
        for inst in insts:
            raw = fetch(f"{OKX}/{d.strftime('%Y%m%d')}/{inst}-trades-{ds}.zip", f"{inst}-trades-{ds}.zip")
            if raw is None:
                print(f"okx {inst} {ds}: not published", flush=True)
                continue
            lake.write("okx_trades_1m", ds[:7], okx_minutes(raw, inst).assign(collected_at=pd.Timestamp.now("UTC")))
        print(f"okx {ds} done", flush=True)


def wide(df, field, freq):
    """dates x symbols of one field, resampled from 1m bars by last (or first for open, sum for volume)."""
    how = {"open": "first", "high": "max", "low": "min", "volume": "sum", "quote_volume": "sum"}.get(field, "last")
    w = df.pivot(index="ts", columns="symbol", values=field).sort_index()
    return w if freq == "1min" else w.resample(freq, label="right", closed="right").agg(how)


def bars(start, end, freq="5min", symbols=SYMBOLS):
    """spot and perp OHLCV frames on one grid, keyed by field, plus the funding series per symbol."""
    spot = lake.load("crypto_klines_1m", start, end, universe=list(symbols))
    perp = lake.load("crypto_perp_klines_1m", start, end, universe=list(symbols))
    fund = lake.load("crypto_funding", start, end, universe=list(symbols))
    fields = ("open", "high", "low", "close", "volume")
    s = {f: wide(spot, f, freq) for f in fields}
    p = {f: wide(perp, f, freq) for f in fields}
    idx = s["close"].index.intersection(p["close"].index)
    s = {f: v.reindex(idx) for f, v in s.items()}
    p = {f: v.reindex(idx) for f, v in p.items()}
    funding = fund.pivot(index="ts", columns="symbol", values="rate").sort_index()
    return s, p, funding


def okx_bars(start, end):
    df = lake.load("okx_trades_1m", start, end)
    return {f: df.pivot(index="ts", columns="symbol", values=f).sort_index() for f in ("last", "vwap", "volume")}


if __name__ == "__main__":
    collect_binance()
    collect_okx()
    for d in ("crypto_klines_1m", "crypto_perp_klines_1m", "crypto_funding", "okx_trades_1m"):
        print(d, lake.parts(d))
