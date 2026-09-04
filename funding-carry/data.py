"""Binance funding rates and 8h bars from the public data dumps, cached to CSV.

The REST endpoints at fapi.binance.com are geo-blocked from the US. The monthly
archives at data.binance.vision are the same exchange data and are not blocked,
so everything here reads those instead.
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

VISION = "https://data.binance.vision/data"
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                     "source-material", "crypto"))

SYMBOLS = ("BTCUSDT", "ETHUSDT")
START = "2020-01"
END = "2026-07"
FUNDING_HOURS = 8
PERIODS_PER_YEAR = 365 * 24 / FUNDING_HOURS

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def _months(start=START, end=END):
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def _fetch_zip(path):
    """Download one monthly archive and return its single CSV as raw bytes."""
    r = requests.get(f"{VISION}/{path}", timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return z.read(z.namelist()[0])


def _to_utc(ms):
    # Binance switched kline timestamps from milliseconds to microseconds during
    # 2025, so the same column has two units depending on the month.
    ms = pd.to_numeric(ms)
    return pd.to_datetime(np.where(ms > 1e15, ms // 1000, ms), unit="ms", utc=True)


def _cached(name, build):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    frame = build()
    os.makedirs(CACHE, exist_ok=True)
    frame.to_csv(path)
    return frame


def funding(symbol):
    """Realised 8h funding rates for one perpetual, indexed by settlement time."""
    def build():
        parts = []
        for m in _months():
            raw = _fetch_zip(f"futures/um/monthly/fundingRate/{symbol}/"
                             f"{symbol}-fundingRate-{m}.zip")
            if raw is None:
                continue
            parts.append(pd.read_csv(io.BytesIO(raw)))
        df = pd.concat(parts, ignore_index=True)
        df["time"] = _to_utc(df["calc_time"]).round("1h")
        df = df.rename(columns={"last_funding_rate": "rate",
                                "funding_interval_hours": "interval_hours"})
        return df.set_index("time")[["rate", "interval_hours"]].sort_index()
    return _cached(f"{symbol}_funding.csv", build)


def bars(symbol, market, interval="8h"):
    """OHLCV bars for `market` in {"spot", "um"}, indexed by bar open time."""
    root = "spot" if market == "spot" else "futures/um"

    def build():
        parts = []
        for m in _months():
            raw = _fetch_zip(f"{root}/monthly/klines/{symbol}/{interval}/"
                             f"{symbol}-{interval}-{m}.zip")
            if raw is None:
                continue
            part = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
            parts.append(part[pd.to_numeric(part["open"], errors="coerce").notna()])
        df = pd.concat(parts, ignore_index=True)
        df["time"] = _to_utc(df["open_time"])
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)
        return df.set_index("time")[cols].sort_index()
    return _cached(f"{symbol}_{market}_{interval}.csv", build)


def panel(symbol):
    """Join funding to the spot and perp bar that ends at each settlement time.

    Funding at time T settles the bar spanning [T-8h, T), so that bar's close is
    the price at T and its high is the worst point for a short over the period.
    """
    fr = funding(symbol)
    spot = bars(symbol, "spot")
    perp = bars(symbol, "um")
    bar_open = fr.index - pd.Timedelta(hours=FUNDING_HOURS)

    cols = {"rate": fr["rate"], "interval_hours": fr["interval_hours"]}
    for name, src in (("spot", spot), ("perp", perp)):
        cols[name] = src["close"].reindex(bar_open)
        for field in ("open", "high", "low"):
            cols[f"{name}_{field}"] = src[field].reindex(bar_open)
    df = pd.DataFrame({k: np.asarray(v) for k, v in cols.items()}, index=fr.index)
    return df.dropna()


def stablecoin():
    """USDC/USDT 8h bars, a direct read on how far USDT drifted from a dollar."""
    return bars("USDCUSDT", "spot")


def tbill():
    """13-week T-bill discount rate, the cash return the carry has to beat."""
    def build():
        import yfinance as yf
        px = yf.download("^IRX", start="2019-12-01", end="2026-08-01",
                         progress=False, auto_adjust=False)["Close"]
        return px.rename(columns={"^IRX": "rate_pct"}) / 100.0
    return _cached("tbill_13w.csv", build)["rate_pct"]


def load_all():
    """Return {symbol: panel} for the study universe."""
    return {s: panel(s) for s in SYMBOLS}
