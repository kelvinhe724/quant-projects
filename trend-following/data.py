"""Daily closes for a cross-asset futures universe and an ETF cross-check universe.

Downloads once from yfinance and caches to source-material/trend-following/, so
reruns are offline. Yahoo's "=F" tickers are front-month splices with no roll
adjustment, which is the main data caveat the README discusses.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
CACHE = os.path.join(os.path.dirname(ROOT), "source-material", "trend-following", "prices.csv")

FUTURES = {
    "ES=F": "equities", "NQ=F": "equities", "EFA": "equities", "EEM": "equities",
    "ZN=F": "bonds", "ZB=F": "bonds", "TLT": "bonds",
    "CL=F": "commodities", "GC=F": "commodities", "HG=F": "commodities",
    "ZC=F": "commodities", "NG=F": "commodities",
    "6E=F": "fx", "6J=F": "fx", "6B=F": "fx",
}

# Same four classes built only from ETFs. Every ETF here is a total-return series
# with no roll splice, so it is the check on whether the futures result is an
# artefact of Yahoo's contract stitching. Several list after 2005 and enter as
# they list.
ETFS = {
    "SPY": "equities", "QQQ": "equities", "EFA": "equities", "EEM": "equities",
    "IEF": "bonds", "TLT": "bonds",
    "USO": "commodities", "GLD": "commodities", "SLV": "commodities",
    "DBA": "commodities", "DBC": "commodities",
    "FXE": "fx", "FXY": "fx", "FXB": "fx",
}

BENCHMARK = "SPY"
DATA_START = "2003-06-01"
DATA_END = "2026-09-01"
START = "2005-01-31"
IS_END = "2015-12-31"
TEST_START = "2016-01-01"  # Jan-2016 month-end is the 29th; a 01-31 bound silently dropped that month


def download():
    """Return the cached daily close panel, downloading it first if needed."""
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    tickers = sorted(set(FUTURES) | set(ETFS) | {BENCHMARK})
    raw = yf.download(tickers, start=DATA_START, end=DATA_END, auto_adjust=True,
                      progress=False, group_by="column")
    px = raw["Close"].sort_index()
    if px.isna().all().any():
        raise RuntimeError(f"no data for {list(px.columns[px.isna().all()])}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    px.to_csv(CACHE)
    return px


def clean(px):
    """Fill gaps inside a live series, never before its first print, and drop non-positive prints.

    The only non-positive print in the sample is CL=F on 2020-04-20, the day the
    expiring May contract settled at -37.63. A percentage return off a negative
    price is meaningless, so the day is treated as missing.
    """
    return px.where(px > 0).ffill().where(px.notna().cummax())


def get_panel(universe=FUTURES):
    """Return daily prices, daily returns, asset classes and benchmark daily returns."""
    px = clean(download())
    names = [t for t in universe if t in px.columns]
    classes = pd.Series({t: universe[t] for t in names}, name="class")
    return px[names], px[names].pct_change(), classes, px[BENCHMARK].pct_change()


def month_ends(index):
    """Return the last trading day of each month in `index`."""
    days = pd.Series(index, index=index)
    return pd.DatetimeIndex(sorted(days.groupby([index.year, index.month]).max()))


def monthly_returns(px):
    """Compound daily prices to month-end-to-month-end simple returns."""
    return px.loc[month_ends(px.index)].pct_change()


if __name__ == "__main__":
    px, rets, classes, bench = get_panel()
    print(f"{len(px.columns)} assets, {px.index[0].date()} to {px.index[-1].date()}")
    print(px.apply(lambda s: s.first_valid_index().date()).to_string())
    print(f"daily moves over 10%:\n{(rets.abs() > 0.10).sum().to_string()}")
