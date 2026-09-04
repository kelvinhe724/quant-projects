"""FRED spot rates and OECD 3-month interbank rates for the G10, cached under
source-material/fx-carry/ so every rerun is offline.

Every spot series is converted to USD per one unit of foreign currency, so a
rising number always means the foreign currency strengthened.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "source-material", "fx-carry")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# H.10 daily noon buying rates in New York. The second field says whether FRED
# quotes the series as USD per foreign unit or foreign units per USD.
SPOT = {
    "EUR": ("DEXUSEU", "usd_per_fx"),
    "JPY": ("DEXJPUS", "fx_per_usd"),
    "GBP": ("DEXUSUK", "usd_per_fx"),
    "CAD": ("DEXCAUS", "fx_per_usd"),
    "AUD": ("DEXUSAL", "usd_per_fx"),
    "NZD": ("DEXUSNZ", "usd_per_fx"),
    "CHF": ("DEXSZUS", "fx_per_usd"),
    "NOK": ("DEXNOUS", "fx_per_usd"),
    "SEK": ("DEXSDUS", "fx_per_usd"),
}

# OECD Main Economic Indicators, 3-month interbank rate, monthly average, percent
# per annum. One series per currency, USD included as the funding leg.
RATE = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N",
    "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CHF": "IR3TIB01CHM156N",
    "NOK": "IR3TIB01NOM156N",
    "SEK": "IR3TIB01SEM156N",
}

CURRENCIES = list(RATE)
FOREIGN = list(SPOT)

# The Japanese rate series starts 2002-04, which sets the start of the sample.
IS_START = "2002-04-30"
IS_END = "2015-12-31"
TEST_START = "2016-01-01"

# OECD publishes with a lag and the euro and sterling series currently end a few
# months before spot. A rate is carried forward at most this many months; beyond
# that the currency drops out of the sort rather than trade on a stale number.
MAX_STALE_MONTHS = 8


def fetch(series_id):
    """Return one FRED series as a float Series, downloading only if uncached."""
    path = os.path.join(CACHE_DIR, f"{series_id}.csv")
    if not os.path.exists(path):
        import certifi
        import requests

        response = requests.get(FRED.format(series_id), verify=certifi.where(), timeout=60)
        response.raise_for_status()
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(response.content)
    raw = pd.read_csv(path, na_values=".", index_col=0, parse_dates=True)
    return raw.iloc[:, 0].astype(float).rename(series_id)


def spot_daily():
    """Daily USD per unit of each foreign currency, holidays forward-filled."""
    cols = {}
    for ccy, (sid, quote) in SPOT.items():
        s = fetch(sid)
        cols[ccy] = 1.0 / s if quote == "fx_per_usd" else s
    px = pd.DataFrame(cols)
    px = px.dropna(how="all").ffill()
    return px.dropna()


def rates_monthly():
    """Month-end indexed 3-month rates in percent, one column per currency."""
    r = pd.DataFrame({ccy: fetch(sid) for ccy, sid in RATE.items()})
    r.index = r.index.to_period("M")
    r = r.ffill(limit=MAX_STALE_MONTHS)
    return r.loc[pd.Period(IS_START, "M"):]


def get_panel():
    """Return daily spot, daily rates (month's rate carried through its days) and month ends."""
    px = spot_daily().loc[IS_START:]
    rates = rates_monthly()
    px = px.loc[:rates.dropna().index[-1].to_timestamp("M")]
    month_ends = pd.DatetimeIndex(
        px.groupby(px.index.to_period("M")).apply(lambda d: d.index[-1]).values)
    # the rate for month M is used from M's last trading day until the next month end
    daily_rates = rates.reindex(month_ends.to_period("M"))
    daily_rates.index = month_ends
    daily_rates = daily_rates.reindex(px.index).ffill()
    return px, daily_rates, month_ends


if __name__ == "__main__":
    px, rates, ends = get_panel()
    print(f"spot: {len(px)} days, {px.index[0].date()} to {px.index[-1].date()}")
    print(f"rates: {rates.dropna().index[0].date()} to {rates.dropna().index[-1].date()}")
    print(f"month ends: {len(ends)}")
    print(rates.loc[ends].tail(3).round(2).to_string())
