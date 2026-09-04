"""Daily OHLC for the overnight study, cached to source-material/overnight-anomaly/ so reruns are offline.

Prices come from yfinance with auto_adjust=True, which scales Open and Close by
the same factor on each day, so Open/prev Close and Close/Open are unaffected by
the adjustment. Dividends land in the overnight leg because the ex-date drop is
between one close and the next open. The README discusses what that does to the
result.
"""
import os

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
CACHE_DIR = os.path.join(os.path.dirname(ROOT), "source-material", "overnight-anomaly")

INDEX_ETFS = ["SPY", "QQQ", "IWM"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
LARGE_CAPS = ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ", "PG", "WMT", "GE", "INTC", "KO", "PFE"]
UNIVERSE = INDEX_ETFS + SECTOR_ETFS + LARGE_CAPS
GROUP = {**{t: "index" for t in INDEX_ETFS}, **{t: "sector" for t in SECTOR_ETFS},
         **{t: "stock" for t in LARGE_CAPS}}

DATA_START = "2000-01-01"
DATA_END = "2026-09-01"

# The papers this tests were written on data ending around 2014 (Kelly and Clifton)
# and 2017 (Lou, Polk and Skouras). Everything from 2016 on is post-publication.
IS_END = "2015-12-31"
TEST_START = "2016-01-01"


def download(ticker):
    """Return daily OHLC for one ticker, from cache if present, else from yfinance."""
    path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = yf.download(ticker, start=DATA_START, end=DATA_END, auto_adjust=True,
                      progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned nothing for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ohlc = raw[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    ohlc.to_csv(path)
    return ohlc


def dividend_yield(ticker):
    """Return annual dividend yield by year (cash paid / unadjusted close on ex-date), cached."""
    path = os.path.join(CACHE_DIR, f"{ticker}_dividends.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)["yield"]
    tk = yf.Ticker(ticker)
    div = tk.dividends
    px = tk.history(start=DATA_START, end=DATA_END, auto_adjust=False)["Close"]
    div.index, px.index = div.index.tz_localize(None), px.index.tz_localize(None)
    div = div[div.index >= DATA_START]
    y = (div / px.reindex(div.index, method="ffill")).groupby(div.index.year).sum()
    y.name = "yield"
    os.makedirs(CACHE_DIR, exist_ok=True)
    y.to_csv(path)
    return y


def get_panel(tickers=UNIVERSE):
    """Return {ticker: OHLC frame} with non-positive prints and duplicate dates removed."""
    out = {}
    for t in tickers:
        f = download(t)
        f = f[~f.index.duplicated()]
        f = f[(f["Open"] > 0) & (f["Close"] > 0)]
        out[t] = f
    return out


STALE_YEAR_LIMIT = 0.25


def stale_open(ohlc):
    """Flag days whose open equals the prior close exactly.

    Yahoo has stretches, mostly 2000-2001 in the older stocks, where the open field
    is the previous close copied forward. Those days put the whole move into the
    intraday leg. A few percent of exact matches is normal for a quiet large cap;
    60% in a year is a data problem.
    """
    return (ohlc["Open"] / ohlc["Close"].shift(1) - 1).abs() < 1e-6


def drop_stale_years(ohlc):
    """Remove any calendar year where more than STALE_YEAR_LIMIT of opens are stale."""
    stale = stale_open(ohlc)
    bad = stale.groupby(ohlc.index.year).mean() > STALE_YEAR_LIMIT
    return ohlc[~ohlc.index.year.isin(bad[bad].index)]


if __name__ == "__main__":
    panel = get_panel()
    for t, f in panel.items():
        kept = drop_stale_years(f)
        print(f"{t:5s} {f.index[0].date()} to {f.index[-1].date()}  {len(f):5d} days  "
              f"stale opens {stale_open(f).mean():.1%}  dropped {len(f) - len(kept)} days")
