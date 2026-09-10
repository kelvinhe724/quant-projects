"""Instrument lists and the one Bars panel the book runs on.

ETFs come from yfinance through engine.load_yfinance (cached under
source-material/framework/yf/), BTC and ETH from Alpaca's crypto data API,
which needs no key (cached under source-material/framework/alpaca/), and the
OECD 3-month rates from the fx-carry cache. Everything is aligned to the NYSE
calendar; a crypto bar on a NYSE date is that UTC day's bar, and weekend moves
land in Monday's open.
"""
import hashlib
import os

import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday, sunday_to_monday)

from framework.engine import Bars, load_fred, load_yfinance
from framework.engine.data import (CACHE, CACHE_MAX_AGE_DAYS, DOWNLOAD_TRIES, DOWNLOAD_WAIT,
                                   FIELDS, SUBSTITUTED, _cached_bars, clean_prices)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_START = "2003-06-01"
DATA_END = "2026-09-01"

# Same 14 ETFs and four classes as trend-following/data.py::ETFS.
ETFS = {
    "SPY": "equities", "QQQ": "equities", "EFA": "equities", "EEM": "equities",
    "IEF": "bonds", "TLT": "bonds",
    "USO": "commodities", "GLD": "commodities", "SLV": "commodities",
    "DBA": "commodities", "DBC": "commodities",
    "FXE": "fx", "FXY": "fx", "FXB": "fx",
}

# Currency ETF -> (currency, FRED id of the OECD 3-month interbank rate).
FX_ETFS = {
    "FXE": ("EUR", "IR3TIB01EZM156N"),
    "FXY": ("JPY", "IR3TIB01JPM156N"),
    "FXB": ("GBP", "IR3TIB01GBM156N"),
    "FXA": ("AUD", "IR3TIB01AUM156N"),
    "FXC": ("CAD", "IR3TIB01CAM156N"),
    "FXF": ("CHF", "IR3TIB01CHM156N"),
}
USD_RATE = "IR3TIB01USM156N"
# ponytail: no per-currency splice. OECD publishes these with a lag that differs
# by currency, so load_rates truncates every column to the last month they all
# have. A uniform lag is a ranking the carry sleeve can trust; a mixed vintage
# (current EUR/GBP against three-month-old everything else) is not. Splicing in
# each currency's own overnight rate (ESTR, SONIA, TONAR, RBA cash, CORRA,
# SARON, EFFR) is the upgrade if this sleeve ever carries real weight.
CRYPTO = ["BTC/USD", "ETH/USD"]
BENCHMARK = "SPY"

CLASSES = pd.Series({**ETFS, **{t: "fx" for t in FX_ETFS}, **{c: "crypto" for c in CRYPTO}})
SLEEVES = {"TrendETF": list(ETFS), "FXCarryETF": list(FX_ETFS), "CryptoTrend": list(CRYPTO),
           "EWMAC": list(ETFS), "ETFBeta": list(ETFS)}
TICKERS = sorted(set(ETFS) | set(FX_ETFS) | {BENCHMARK})
CALENDAR_AHEAD = 60


class NYSE(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def sessions(start, end):
    """NYSE trading days between start and end inclusive; unscheduled closures are not known."""
    days = pd.bdate_range(start, end)
    return days.difference(NYSE().holidays(days[0], days[-1]))


def load_rates(cache=os.path.join(ROOT, "source-material", "fx-carry"), refresh=False):
    """Monthly OECD 3-month rates in percent, one column per currency, USD included.

    FRED dates month M's average at the first of M. It is only knowable after
    M ends, so each value is dated the first day of M+1 here. The research
    path reads the frozen fx-carry cache; refresh=True (the daemon) downloads
    every series again into the engine's FRED cache.

    Every column is cut back to the last month all currencies have, so one
    cross-sectional rank is never a mix of vintages.
    """
    ids = {c: sid for c, sid in FX_ETFS.values()}
    ids["USD"] = USD_RATE
    out = {}
    for ccy, sid in ids.items():
        if refresh:
            s = load_fred(sid, refresh=True)
        else:
            raw = pd.read_csv(os.path.join(cache, f"{sid}.csv"), na_values=".", index_col=0, parse_dates=True)
            s = raw.iloc[:, 0].astype(float)
        s = s.dropna()
        s.index = s.index + pd.DateOffset(months=1)
        out[ccy] = s
    df = pd.DataFrame(out)
    return df.loc[:min(df[c].last_valid_index() for c in df)]


def _alpaca_bars(sym, start, end, tries=DOWNLOAD_TRIES, wait=DOWNLOAD_WAIT):
    """One pair's daily bars from Alpaca, retrying a transient failure. None when it never comes back."""
    import time

    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                            start=pd.Timestamp(start), end=pd.Timestamp(end))
    for i in range(tries):
        try:
            df = CryptoHistoricalDataClient().get_crypto_bars(req).df
        except Exception as e:
            print(f"alpaca error for {sym} (try {i + 1}/{tries}): {type(e).__name__}: {e}")
            df = None
        if df is not None:
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)
            if not df.empty:
                df = df[list(FIELDS)]
                df.index = df.index.tz_convert(None).normalize()
                df.index.name = "date"
                return df
        if i + 1 < tries:
            time.sleep(wait)
    return None


def load_crypto(start=DATA_START, end=DATA_END, cache=os.path.join(CACHE, "alpaca"),
                max_age_days=CACHE_MAX_AGE_DAYS):
    """Daily OHLCV per crypto pair from Alpaca, downloaded once into `cache`.

    Same rule as load_yfinance: a pair Alpaca will not serve falls back to its
    newest cached bars if those are within `max_age_days`, and says so loudly.
    Raises when a missing pair has no usable cache, or when every pair is
    missing, which means the vendor is down rather than one symbol being flaky.
    """
    os.makedirs(cache, exist_ok=True)
    per, substituted, unusable = {}, [], []
    for sym in CRYPTO:
        stem = sym.replace("/", "-")
        path = os.path.join(cache, f"{stem}_daily_{end}.csv")
        if not os.path.exists(path):
            df = _alpaca_bars(sym, start, end)
            if df is None:
                cached, age = _cached_bars(stem, start, end, cache, max_age_days)
                if cached is None:
                    unusable.append(sym)
                    continue
                print(f"{SUBSTITUTED}: {sym} unavailable from alpaca, using cached bars through "
                      f"{cached.index[-1].date()} ({age} days old)")
                substituted.append(sym)
                per[sym] = cached
                continue
            df.to_csv(path)
        per[sym] = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = substituted + unusable
    # ponytail: two pairs, so "majority" is "all of them"; revisit if CRYPTO grows.
    if len(missing) == len(CRYPTO):
        raise RuntimeError(f"every crypto pair unavailable ({', '.join(sorted(missing))}); "
                           f"alpaca looks down, not one flaky symbol")
    if unusable:
        raise RuntimeError(f"no crypto bars and no usable cache for {', '.join(unusable)}")
    return {f: pd.DataFrame({s: d[f] for s, d in per.items()}).sort_index() for f in FIELDS}


def load_bars(start=DATA_START, end=DATA_END, rates=None):
    """Return one Bars over every instrument on the NYSE calendar, with the rate series attached.

    `end` is exclusive, as in yfinance. The daemon passes a moving window and
    freshly downloaded `rates`; the research scripts use the frozen defaults
    so their numbers stay reproducible.
    """
    etf = load_yfinance(TICKERS, start, end)
    crypto = load_crypto(start, end)
    frames = {}
    for f in FIELDS:
        panel = pd.concat([etf.field(f), crypto[f].reindex(etf.calendar)], axis=1)
        frames[f] = panel if f == "volume" else clean_prices(panel)
    # The calendar runs past the last bar so month ends are known on the day,
    # not one bar late: a view whose last bar is the newest print must not
    # mistake it for the end of the month.
    last = etf.calendar[-1]
    ahead = sessions(last + pd.Timedelta(days=1), last + pd.Timedelta(days=CALENDAR_AHEAD))
    idx = etf.calendar.union(ahead)
    rates = load_rates() if rates is None else rates
    return Bars({f: df.reindex(idx) for f, df in frames.items()}, {c: rates[c] for c in rates.columns},
                end=len(etf.calendar))


def panel_hash(bars, start=None, end=None):
    """Short fingerprint of the close panel between start and end, logged so a silent re-pull is visible.

    The daemon hashes a fixed window so the value only moves when history is
    rewritten (a dividend under auto_adjust), not because a day was added.
    """
    close = bars.close.loc[start:end]
    rows = pd.util.hash_pandas_object(close, index=True).to_numpy().tobytes()
    return hashlib.sha256(",".join(bars.instruments).encode() + rows).hexdigest()[:16]


if __name__ == "__main__":
    bars = load_bars()
    px = bars.close
    print(f"{len(px.columns)} instruments, {px.index[0].date()} to {px.index[-1].date()}, "
          f"{len(px)} bars, hash {panel_hash(bars)}")
    print(pd.DataFrame({"first": px.apply(lambda s: s.first_valid_index().date()),
                        "last": px.apply(lambda s: s.last_valid_index().date()),
                        "class": CLASSES}).to_string())
    empty = [c for c in px.columns if px[c].isna().all()]
    print(f"all-NaN columns: {empty or 'none'}")
    print(f"rate series: {bars.series_names}, last month {bars.series('USD').dropna().index[-1].date()}")
