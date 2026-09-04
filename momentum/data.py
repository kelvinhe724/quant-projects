"""Daily prices and volumes for the momentum universe, cached to CSV so reruns are offline."""
import os

import numpy as np
import pandas as pd
import yfinance as yf

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
PRICE_CACHE = os.path.join(REPORTS, "prices.csv")
VOLUME_CACHE = os.path.join(REPORTS, "volumes.csv")

# A liquid slice of the S&P 500, roughly 14-20 names per GICS sector. This is the
# current membership list, which is the survivorship problem the README quantifies:
# yfinance has no point-in-time constituent history, so names deleted from the index
# between 2005 and today are simply absent.
UNIVERSE = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "ORCL": "Information Technology", "CRM": "Information Technology",
    "ADBE": "Information Technology", "AMD": "Information Technology",
    "INTC": "Information Technology", "CSCO": "Information Technology",
    "ACN": "Information Technology", "TXN": "Information Technology",
    "QCOM": "Information Technology", "IBM": "Information Technology",
    "MU": "Information Technology", "AMAT": "Information Technology",
    "LRCX": "Information Technology", "KLAC": "Information Technology",
    "ADI": "Information Technology", "NOW": "Information Technology",

    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
    "USB": "Financials", "PNC": "Financials", "TFC": "Financials",
    "BK": "Financials", "CB": "Financials", "MMC": "Financials",
    "AIG": "Financials", "MET": "Financials", "PRU": "Financials",
    "TRV": "Financials", "ALL": "Financials",

    "JNJ": "Health Care", "UNH": "Health Care", "LLY": "Health Care",
    "PFE": "Health Care", "ABBV": "Health Care", "MRK": "Health Care",
    "TMO": "Health Care", "ABT": "Health Care", "DHR": "Health Care",
    "BMY": "Health Care", "AMGN": "Health Care", "GILD": "Health Care",
    "CVS": "Health Care", "CI": "Health Care", "ELV": "Health Care",
    "ISRG": "Health Care", "SYK": "Health Care", "BDX": "Health Care",
    "ZTS": "Health Care", "MDT": "Health Care",

    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "YUM": "Consumer Discretionary", "MAR": "Consumer Discretionary",
    "HLT": "Consumer Discretionary", "DHI": "Consumer Discretionary",
    "LEN": "Consumer Discretionary", "CMG": "Consumer Discretionary",

    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "GIS": "Consumer Staples", "STZ": "Consumer Staples",
    "SYY": "Consumer Staples", "KR": "Consumer Staples", "HSY": "Consumer Staples",
    "CHD": "Consumer Staples", "TSN": "Consumer Staples", "EL": "Consumer Staples",

    "CAT": "Industrials", "DE": "Industrials", "HON": "Industrials",
    "UNP": "Industrials", "UPS": "Industrials", "BA": "Industrials",
    "GE": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "MMM": "Industrials", "CSX": "Industrials", "NSC": "Industrials",
    "ETN": "Industrials", "EMR": "Industrials", "ITW": "Industrials",
    "PH": "Industrials", "GD": "Industrials", "NOC": "Industrials",
    "WM": "Industrials", "FDX": "Industrials",

    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PSX": "Energy", "MPC": "Energy", "VLO": "Energy",
    "OXY": "Energy", "WMB": "Energy", "KMI": "Energy", "HAL": "Energy",
    "DVN": "Energy", "HES": "Energy", "BKR": "Energy", "OKE": "Energy",

    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AEP": "Utilities", "EXC": "Utilities", "SRE": "Utilities", "XEL": "Utilities",
    "ED": "Utilities", "PEG": "Utilities", "WEC": "Utilities", "ES": "Utilities",
    "AEE": "Utilities", "DTE": "Utilities", "PPL": "Utilities", "CMS": "Utilities",

    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials", "DOW": "Materials", "DD": "Materials",
    "PPG": "Materials", "NUE": "Materials", "VMC": "Materials", "MLM": "Materials",
    "IFF": "Materials", "ALB": "Materials",

    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "SPG": "Real Estate",
    "O": "Real Estate", "WELL": "Real Estate", "AVB": "Real Estate",
    "EQR": "Real Estate", "DLR": "Real Estate", "VTR": "Real Estate",
    "ESS": "Real Estate", "MAA": "Real Estate",

    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CMCSA": "Communication Services", "VZ": "Communication Services",
    "T": "Communication Services", "TMUS": "Communication Services",
    "CHTR": "Communication Services", "EA": "Communication Services",
    "OMC": "Communication Services", "IPG": "Communication Services",
}

BENCHMARK = "SPY"

# Warm-up starts a year before scoring so the 252-day momentum window and the
# 200-day moving average are both complete on the first scored day.
DATA_START = "2003-06-01"
IS_START = "2005-01-03"
IS_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2026-08-31"

MIN_DOLLAR_VOLUME = 10e6
ADV_WINDOW = 63
MIN_HISTORY = 252


def download(start=DATA_START, end=TEST_END):
    """Fetch adjusted closes and volumes for the universe, using the CSV cache if present."""
    if os.path.exists(PRICE_CACHE) and os.path.exists(VOLUME_CACHE):
        read = dict(index_col=0, parse_dates=True)
        return (pd.read_csv(PRICE_CACHE, **read), pd.read_csv(VOLUME_CACHE, **read))
    tickers = sorted(UNIVERSE) + [BENCHMARK]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column")
    px, vol = raw["Close"].sort_index(), raw["Volume"].sort_index()
    os.makedirs(REPORTS, exist_ok=True)
    px.to_csv(PRICE_CACHE)
    vol.to_csv(VOLUME_CACHE)
    return px, vol


def eligibility(px, vol):
    """Flag each stock-date the strategy is allowed to hold, using only past data.

    A name qualifies once it has MIN_HISTORY closes behind it and its trailing
    median dollar volume clears MIN_DOLLAR_VOLUME. The dollar-volume window ends
    at t-1 so today's tape is never part of today's eligibility test.

    There is deliberately no minimum-price filter. The cached closes are
    split-adjusted, so a $5 floor on them would judge 2005 eligibility by splits
    that happened in 2024 and throw out NVDA, AAPL, AMZN and NFLX in exactly the
    years they were the biggest winners. Dollar volume is split-invariant.
    """
    traded = px.notna()
    history = traded.cumsum()
    dollar_volume = (px * vol).shift(1).rolling(ADV_WINDOW, min_periods=ADV_WINDOW).median()
    return (traded
            & (history >= MIN_HISTORY)
            & (dollar_volume >= MIN_DOLLAR_VOLUME))


def get_panel():
    """Return prices, simple returns, the eligibility mask, sectors and the benchmark."""
    px, vol = download()
    bench = px[BENCHMARK]
    names = [t for t in sorted(UNIVERSE) if t in px.columns]
    px, vol = px[names], vol[names]

    # A gap inside a live series is a halt or a bad print, not a delisting: fill it
    # so the moving averages stay warm, but never fill before a name's first print.
    px = px.where(px > 0).ffill().where(px.notna().cummax())

    returns = px.pct_change()
    sectors = pd.Series({t: UNIVERSE[t] for t in names}, name="sector")
    return px, returns, eligibility(px, vol), sectors, bench.pct_change()


def rebalance_dates(index, start=None, end=None):
    """Return the last trading day of each month in `index`."""
    days = pd.Series(index, index=index)
    ends = days.groupby([index.year, index.month]).max()
    out = pd.DatetimeIndex(sorted(ends.values))
    return out[(out >= (start or out[0])) & (out <= (end or out[-1]))]


def window(frame, start, end):
    """Slice a time-indexed frame or series to [start, end]."""
    return frame.loc[start:end]


if __name__ == "__main__":
    px, rets, elig, sectors, bench = get_panel()
    print(f"{len(px.columns)} names, {px.index[0].date()} to {px.index[-1].date()}")
    print(f"eligible names per date: min {elig.sum(axis=1).min()}, "
          f"median {int(elig.sum(axis=1).median())}, max {elig.sum(axis=1).max()}")
    print(f"first fully eligible date: {elig.sum(axis=1).gt(50).idxmax().date()}")
    print(f"benchmark {BENCHMARK}: {bench.notna().sum()} return days")
