"""Monthly total returns for 14 ETFs plus a T-bill rate, cached under source-material."""
import os

import pandas as pd
import requests
import yfinance as yf

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "source-material", "portfolio-construction")
PRICE_CACHE = os.path.join(CACHE, "prices_daily.csv")
RF_CACHE = os.path.join(CACHE, "TB3MS.csv")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# Nine SPDR sectors, two Treasury tenors, gold, developed and emerging ex-US.
# All fourteen launched before 2005, and all fourteen are still trading, which is
# the survivorship caveat the README discusses.
TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB",
           "TLT", "IEF", "GLD", "EFA", "EEM"]
START = "2004-06-01"
END = "2026-09-01"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"


def download():
    """Fetch daily adjusted closes and the FRED 3-month T-bill series, using the cache if present."""
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(PRICE_CACHE):
        raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True,
                          progress=False, group_by="column")
        raw["Close"].sort_index().to_csv(PRICE_CACHE)
    if not os.path.exists(RF_CACHE):
        r = requests.get(FRED_URL, timeout=30)
        r.raise_for_status()
        with open(RF_CACHE, "w") as f:
            f.write(r.text)
    px = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)[TICKERS]
    rf = pd.read_csv(RF_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    return px, rf


def monthly_panel():
    """Return month-end simple returns, the matching monthly risk-free rate, and excess returns.

    Adjusted closes already include dividends, so a close-to-close change is a
    total return. TB3MS is the monthly average of the 3-month bill's discount
    yield, annualised, dated by FRED to the first of its month. It is converted
    to a monthly decimal and lagged one month, so the rate subtracted from month
    t's return is the one published before t began.
    """
    px, rf = download()
    month_end = px.resample("ME").last()
    rets = month_end.pct_change().dropna(how="any")
    rf_monthly = ((1 + rf / 100) ** (1 / 12) - 1).shift(1)
    rf_monthly.index = rf_monthly.index + pd.offsets.MonthEnd(0)
    rf_monthly = rf_monthly.reindex(rets.index).ffill()
    return rets, rf_monthly, rets.sub(rf_monthly, axis=0)


if __name__ == "__main__":
    rets, rf, excess = monthly_panel()
    print(f"{rets.shape[1]} assets, {len(rets)} months, {rets.index[0].date()} to {rets.index[-1].date()}")
    print(f"annualised mean excess return by asset:\n{(excess.mean() * 12).round(4).to_string()}")
    print(f"risk-free: mean {rf.mean() * 12:.2%}/yr, last {rf.iloc[-1] * 12:.2%}/yr")
