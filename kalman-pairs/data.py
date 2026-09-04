"""Prices and pair list shared with ../pairs-trading so the two projects are comparable.

The pairs, formation window and out-of-sample window are taken verbatim from the
pairs project. Prices are cached under source-material/kalman-pairs/; on first
run the cache is seeded from the pairs project's own price file when it exists,
otherwise downloaded from Yahoo.
"""
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_PROJECT = os.path.join(os.path.dirname(HERE), "pairs-trading")
PAIRS_FILE = os.path.join(PAIRS_PROJECT, "reports", "selected_pairs.csv")
PAIRS_PRICES = os.path.join(PAIRS_PROJECT, "reports", "prices.csv")
CACHE = os.path.join(os.path.dirname(HERE), "source-material", "kalman-pairs", "prices.csv")
REPORTS = os.path.join(HERE, "reports")

# Must match pairs-trading/data.py.
FORMATION_START = "2015-01-01"
FORMATION_END = "2019-12-31"
OOS_START = "2020-01-01"
OOS_END = "2026-08-31"


def selected_pairs():
    """Load the 35 pairs the pairs project chose, with its static beta and intercept."""
    if not os.path.exists(PAIRS_FILE):
        raise FileNotFoundError(f"{PAIRS_FILE} missing; run ../pairs-trading/run.py first")
    return pd.read_csv(PAIRS_FILE)


def download(tickers, cache=CACHE):
    """Return adjusted closes for the tickers, seeding the cache from the pairs project."""
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(PAIRS_PRICES):
        shutil.copy(PAIRS_PRICES, cache)
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(tickers, start=FORMATION_START, end=OOS_END,
                     auto_adjust=True, progress=False)["Close"].sort_index()
    px.to_csv(cache)
    return px


def get_prices():
    """Return (pairs, prices, log prices) restricted to the names the pairs use."""
    pairs = selected_pairs()
    tickers = sorted(set(pairs.a) | set(pairs.b))
    px = download(tickers)[tickers].ffill().dropna()
    return pairs, px, np.log(px)


def split(frame):
    """Cut a time-indexed frame into the formation and out-of-sample halves."""
    return frame.loc[FORMATION_START:FORMATION_END], frame.loc[OOS_START:OOS_END]
