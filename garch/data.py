"""Index prices from Yahoo, converted to daily log returns x100."""
import numpy as np
import yfinance as yf

INDICES = {
    "^GSPC": "S&P 500",
    "^STOXX50E": "EURO STOXX 50",
    "^N225": "Nikkei 225",
    "^FTSE": "FTSE 100",
}


def get_prices(start="2015-01-01", end="2026-01-01"):
    px = yf.download(list(INDICES), start=start, end=end, auto_adjust=True)["Close"]
    # indices trade on different calendars; keep each series' own days rather
    # than forcing a common one
    return px.dropna(how="all")


def log_returns(px):
    # x100 keeps the optimizer on numbers near 1 instead of 1e-4
    return (np.log(px) - np.log(px.shift(1))) * 100
