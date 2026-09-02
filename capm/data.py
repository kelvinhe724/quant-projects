"""Price and T-bill downloads, turned into daily excess returns."""
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


def get_prices(tickers, start="2023-01-01", end="2026-01-01"):
    px = yf.download(tickers + ["^GSPC"], start=start, end=end, auto_adjust=True)["Close"]
    return px.dropna()


def get_risk_free(start="2023-01-01", end="2026-01-01"):
    # DTB3 arrives as an annualised percent (4.2 = 4.2%/yr); /100/252 makes it daily
    rf = pdr.DataReader("DTB3", "fred", start, end)["DTB3"] / 100 / 252
    return rf.ffill()


def build_excess_returns(tickers, start="2023-01-01", end="2026-01-01"):
    """Returns (excess returns per stock, market excess return)."""
    px = get_prices(tickers, start, end)
    rets = px.pct_change().dropna()
    rf = get_risk_free(start, end).reindex(rets.index).ffill()
    excess = rets.sub(rf, axis=0)
    market = excess.pop("^GSPC")
    return excess, market
