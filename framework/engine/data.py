"""Daily OHLCV panels with an as-of view that cannot see past its date.

A Bars object holds one frame per field (open, high, low, close, volume), all on
the same trading calendar, plus any number of named daily series (FRED rates).
bars.upto(date) returns a view whose every frame ends at `date`; that view is the
only thing a strategy is ever handed, so look-ahead is impossible by
construction rather than by discipline.
"""
import glob
import io
import os
import time

import numpy as np
import pandas as pd
import requests

FIELDS = ("open", "high", "low", "close", "volume")
# A vendor that returns nothing for one symbol must not stop a session. One
# flaky ticker falls back to its own cached bars; a vendor that is actually
# down (most of the universe missing) still raises.
DOWNLOAD_TRIES = 3
DOWNLOAD_WAIT = 2.0
MISSING_ABORT_FRACTION = 0.25
CACHE_MAX_AGE_DAYS = 5  # a long weekend plus a holiday; a week-old close is not a price
SUBSTITUTED = "SUBSTITUTED-CACHED-BARS"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(ROOT), "source-material", "framework")


class LookAheadError(Exception):
    """Raised when code asks an as-of view for data after its date."""


class Bars:
    def __init__(self, frames, series=None, end=None):
        idx = frames["close"].index
        self._frames = {f: frames[f].reindex(idx) for f in FIELDS if f in frames}
        # Last value at or before each session: a series dated on a weekend
        # or holiday (FRED monthly rates land on the 1st) must not be lost.
        self._series = {k: v.dropna().sort_index().reindex(idx, method="ffill")
                        for k, v in (series or {}).items()}
        self.calendar = idx
        # Number of rows visible. The full calendar stays visible on purpose:
        # exchange holidays are published years ahead and knowing them is not
        # knowing prices.
        self._end = len(idx) if end is None else end

    @property
    def asof(self):
        return self.calendar[self._end - 1] if self._end else None

    @property
    def index(self):
        return self.calendar[:self._end]

    @property
    def instruments(self):
        return list(self._frames["close"].columns)

    def field(self, name):
        return self._frames[name].iloc[:self._end]

    open = property(lambda self: self.field("open"))
    high = property(lambda self: self.field("high"))
    low = property(lambda self: self.field("low"))
    close = property(lambda self: self.field("close"))
    volume = property(lambda self: self.field("volume"))

    def series(self, name):
        return self._series[name].iloc[:self._end]

    @property
    def series_names(self):
        return list(self._series)

    def upto(self, date):
        """Return a view of everything through `date` inclusive."""
        date = pd.Timestamp(date)
        if self.asof is not None and date > self.asof:
            raise LookAheadError(f"asked for {date.date()} from a view ending {self.asof.date()}")
        end = int(self.calendar.searchsorted(date, side="right"))
        return Bars(self._frames, self._series, end)

    def select(self, instruments):
        """Restrict to a subset of instruments."""
        frames = {f: df[list(instruments)] for f, df in self._frames.items()}
        return Bars(frames, self._series, self._end)

    def is_month_end(self, date):
        """True when `date` is the last calendar day of its month."""
        i = self.calendar.get_loc(pd.Timestamp(date))
        return i + 1 == len(self.calendar) or self.calendar[i + 1].month != self.calendar[i].month

    def __len__(self):
        return self._end


def clean_prices(px):
    """Drop non-positive prints and fill gaps inside a live series, never before its first print."""
    return px.where(px > 0).ffill().where(px.notna().cummax())


def _download(ticker, start, end, tries=DOWNLOAD_TRIES, wait=DOWNLOAD_WAIT):
    """Download one ticker, retrying a transient empty result. None when it never comes back."""
    import yfinance as yf

    for i in range(tries):
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except Exception as e:
            print(f"yfinance error for {ticker} (try {i + 1}/{tries}): {type(e).__name__}: {e}")
            raw = None
        if raw is not None and not raw.empty:
            raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
            raw.index.name = "date"
            return raw
        if i + 1 < tries:
            time.sleep(wait)
    return None


def _cached_bars(ticker, start, end, cache_dir, max_age_days=CACHE_MAX_AGE_DAYS):
    """Newest cached frame for `ticker`, trimmed to [start, end). None when absent or too stale."""
    best, best_last = None, None
    for path in glob.glob(os.path.join(cache_dir, f"{ticker}_*.csv")):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            continue
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
        if df.empty or not set(FIELDS) <= set(df.columns):
            continue
        if best_last is None or df.index[-1] > best_last:
            best, best_last = df, df.index[-1]
    if best is None:
        return None, None
    age = (pd.Timestamp(end) - pd.Timedelta(days=1) - best_last).days
    if age > max_age_days:
        print(f"cache for {ticker} ends {best_last.date()}, {age} days old, past the "
              f"{max_age_days}-day limit; refusing to trade on it")
        return None, age
    return best, age


def load_yfinance(tickers, start, end, cache=CACHE, max_age_days=CACHE_MAX_AGE_DAYS,
                  abort_fraction=MISSING_ABORT_FRACTION):
    """Return Bars of adjusted daily OHLCV, downloading each ticker once into `cache`.

    A ticker the vendor will not serve falls back to the newest cached bars it
    already has, as long as those are within `max_age_days`. Raises when a
    missing ticker has no usable cache, or when more than `abort_fraction` of
    the universe is missing, which means the vendor is down rather than one
    symbol being flaky.
    """
    cache_dir = os.path.join(cache, "yf")
    os.makedirs(cache_dir, exist_ok=True)
    per, substituted, unusable = {}, [], []
    for t in tickers:
        path = os.path.join(cache_dir, f"{t}_{start}_{end}.csv")
        if not os.path.exists(path):
            raw = _download(t, start, end)
            if raw is None:
                cached, age = _cached_bars(t, start, end, cache_dir, max_age_days)
                if cached is None:
                    unusable.append(t)
                    continue
                print(f"{SUBSTITUTED}: {t} unavailable from yfinance, using cached bars through "
                      f"{cached.index[-1].date()} ({age} days old)")
                substituted.append(t)
                per[t] = cached
                continue
            raw.to_csv(path)
        per[t] = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = substituted + unusable
    if len(missing) > abort_fraction * len(tickers):
        raise RuntimeError(f"{len(missing)} of {len(tickers)} tickers unavailable "
                           f"({', '.join(sorted(missing))}); the vendor looks down, not one flaky symbol")
    if unusable:
        raise RuntimeError(f"no data and no usable cache for {', '.join(unusable)}")
    frames = {}
    for f in FIELDS:
        panel = pd.DataFrame({t: df[f] for t, df in per.items()}).sort_index()
        frames[f] = panel if f == "volume" else clean_prices(panel)
    # Yahoo publishes today's row with a volume but NaN prices until the bar is
    # final; left in, clean_prices would forward-fill yesterday's close onto it.
    last = frames["close"].notna().any(axis=1)[::-1].idxmax()
    return Bars({f: df.loc[:last] for f, df in frames.items()})


def load_fred(series_id, cache=CACHE, refresh=False):
    """Return one FRED series as a daily Series in percent, cached to `cache`; refresh re-downloads."""
    os.makedirs(os.path.join(cache, "fred"), exist_ok=True)
    path = os.path.join(cache, "fred", f"{series_id}.csv")
    if refresh or not os.path.exists(path):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "w") as fh:
            fh.write(r.text)
    df = pd.read_csv(path, index_col=0, parse_dates=True, na_values=".")
    return df.iloc[:, 0].astype(float).rename(series_id)


def synthetic(n_days=252 * 8, instruments=("A", "B", "C", "D"), vol=0.2, drift=0.0,
              start="2010-01-01", seed=0):
    """Random-walk OHLCV panel for offline checks."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    k = len(instruments)
    vol = np.broadcast_to(np.asarray(vol, float), (k,))
    drift = np.broadcast_to(np.asarray(drift, float), (k,))
    rets = rng.normal(drift / 252, vol / np.sqrt(252), (n_days, k))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=list(instruments))
    gap = rng.normal(0, vol / np.sqrt(252) / 3, (n_days, k))
    open_ = close.shift(1).fillna(100.0) * np.exp(gap)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, (n_days, k))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, (n_days, k))))
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (n_days, k)), index=idx,
                          columns=list(instruments)).astype(float)
    return Bars({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
