"""SPY and VIX history plus a recent SPY option chain, cached to CSV."""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPORTS = Path(__file__).resolve().parent / "reports"
DAILY_CACHE = REPORTS / "daily.csv"

START = "2010-01-01"
END = "2026-09-02"


def get_daily(start=START, end=END, cache=DAILY_CACHE):
    """Return a frame of SPY closes, SPY log returns and VIX, cached to CSV."""
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    px = yf.download(["SPY", "^VIX"], start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    df = pd.DataFrame({"spy": px["SPY"], "vix": px["^VIX"]}).dropna()
    df["ret"] = np.log(df["spy"]).diff()
    df = df.dropna()
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(cache)
    return df


def fetch_chain(ticker="SPY", min_days=20, max_days=45):
    """Download listed expiries in the min_days..max_days window with a mid quote."""
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    now = pd.Timestamp.utcnow().tz_localize(None)

    frames = []
    for s in tk.options:
        days = (pd.Timestamp(s) + pd.Timedelta(hours=20) - now).days
        if not min_days <= days <= max_days:
            continue
        ch = tk.option_chain(s)
        for cp, df in (("C", ch.calls), ("P", ch.puts)):
            d = df.copy()
            d["cp"] = cp
            d["expiry"] = pd.Timestamp(s)
            frames.append(d)
    if not frames:
        return None, spot, now

    q = pd.concat(frames, ignore_index=True)
    q["snapshot"] = now
    q["spot"] = spot
    q["T"] = (q["expiry"] + pd.Timedelta(hours=20) - now).dt.total_seconds() / (365 * 86400)
    q["mid"] = (q["bid"] + q["ask"]) / 2
    cols = ["snapshot", "expiry", "T", "cp", "strike", "bid", "ask", "mid",
            "volume", "openInterest", "impliedVolatility", "spot"]
    return q[cols], spot, now


def get_chain(ticker="SPY", **kw):
    """Use the most recent cached chain if there is one, otherwise download and cache.

    The cached snapshot is what the README quotes; delete reports/chain_*.csv to
    take a fresh one.
    """
    REPORTS.mkdir(exist_ok=True)
    hit = sorted(REPORTS.glob(f"chain_{ticker}_*.csv"))
    if hit:
        q = pd.read_csv(hit[-1], parse_dates=["snapshot", "expiry"])
        return q, float(q["spot"].iloc[0]), q["snapshot"].iloc[0]
    q, spot, now = fetch_chain(ticker, **kw)
    if q is not None:
        q.to_csv(REPORTS / f"chain_{ticker}_{now:%Y-%m-%d}.csv", index=False)
    return q, spot, now
