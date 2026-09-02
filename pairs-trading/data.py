"""Price downloads for the pairs universe, cached to CSV so reruns are offline."""
import os

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = os.path.join(os.path.dirname(__file__), "reports", "prices.csv")

# A liquid subset of the S&P 500, roughly 16-20 names per GICS sector. Not the
# full 500: 192 names keeps the within-sector pair count near 1,600 instead of
# 125,000, which is the whole point of the sector restriction.
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

FORMATION_START = "2015-01-01"
FORMATION_END = "2019-12-31"
OOS_START = "2020-01-01"
OOS_END = "2026-08-31"


def download(start=FORMATION_START, end=OOS_END, cache=CACHE):
    """Fetch adjusted closes for the universe, using the CSV cache if present."""
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    px = yf.download(list(UNIVERSE), start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    px = px.sort_index()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    px.to_csv(cache)
    return px


def clean(px, min_coverage=0.99):
    """Drop names with gappy or short history, then forward-fill the rest."""
    keep = px.columns[px.notna().mean() >= min_coverage]
    px = px[keep].ffill().dropna()
    return px


def get_prices(start=FORMATION_START, end=OOS_END):
    """Return cleaned prices, log prices and the sector label per surviving name."""
    px = clean(download(start, end))
    sectors = pd.Series({t: UNIVERSE[t] for t in px.columns}, name="sector")
    return px, np.log(px), sectors


def split(frame):
    """Cut a time-indexed frame into the formation and out-of-sample halves."""
    return frame.loc[FORMATION_START:FORMATION_END], frame.loc[OOS_START:OOS_END]
