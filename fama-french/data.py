"""Fama-French factors and test assets, monthly, in percent."""
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

STOCKS = ["AAPL", "MSFT", "JNJ", "XOM", "KO", "GE", "WMT", "JPM", "PG", "CAT"]

FACTORS_3 = ["Mkt-RF", "SMB", "HML"]
FACTORS_5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def get_factors(start="1963-07-01"):
    """Download the 5 factors plus momentum. Monthly percent, PeriodIndex."""
    ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start)[0]
    mom = pdr.DataReader("F-F_Momentum_Factor", "famafrench", start=start)[0]
    mom.columns = [c.strip() for c in mom.columns]
    return ff.join(mom, how="inner")


def get_test_portfolios(start="1963-07-01"):
    """The 25 size/book-to-market portfolios, value weighted."""
    return pdr.DataReader("25_Portfolios_5x5", "famafrench", start=start)[0]


def get_stock_returns(tickers=STOCKS, start="1990-01-01"):
    """Monthly total returns in percent, PeriodIndex to match the French files."""
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    monthly = px.resample("ME").last().pct_change() * 100
    monthly.index = monthly.index.to_period("M")
    return monthly.dropna(how="all")


def excess(returns, rf):
    """Subtract the risk-free rate over the overlapping months."""
    idx = returns.index.intersection(rf.index)
    return returns.loc[idx].sub(rf.loc[idx], axis=0).dropna(how="all")
