"""Daily SPY closes from yfinance, cached to source-material/regime-hmm/ so reruns are offline."""
import os

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
CACHE = os.path.join(os.path.dirname(ROOT), "source-material", "regime-hmm", "spy.csv")

TICKER = "SPY"
DATA_START = "2000-01-01"
DATA_END = "2026-09-01"
FORMATION_END = "2009-12-31"   # first out-of-sample fit uses 2000-2009 only


def download():
    """Return the cached SPY close series, downloading it first if needed."""
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)["Close"]
    raw = yf.download(TICKER, start=DATA_START, end=DATA_END, auto_adjust=True,
                      progress=False)
    px = raw["Close"].squeeze().dropna().sort_index()
    px.name = "Close"
    if len(px) < 5000:
        raise RuntimeError(f"only {len(px)} rows of SPY from yfinance, refusing to cache")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    px.to_frame().to_csv(CACHE)
    return px


def log_returns(px):
    """Daily log returns, first day dropped."""
    return np.log(px).diff().dropna()
