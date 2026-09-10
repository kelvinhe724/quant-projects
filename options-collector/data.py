"""Universe, storage layout, and the three fetches: option chains, closes, the bill rate.

Raw snapshots live under source-material/options-collector/<date>/, one CSV per
ticker plus underlying.csv, rates.csv, weights.csv and manifest.json. Derived
summaries live under source-material/options-collector/summary/. Both are
gitignored by the repo-level source-material/ rule.
"""
import json
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent / "source-material" / "options-collector"
SUMMARY = ROOT / "summary"
REPORTS = Path(__file__).resolve().parent / "reports"

INDEX = "SPY"
ETFS = ["SPY", "QQQ", "IWM"]
# the ten largest SPY weights when the dispersion project was built, kept fixed
# so the implied-correlation series is comparable day to day
NAMES = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "XOM", "UNH"]
UNIVERSE = ETFS + NAMES

MAX_DAYS = 400
# Same rule as the framework engine: a transient vendor failure is retried before
# the ticker is called missing, and one flaky name must not fail the whole run.
FETCH_TRIES = 3
FETCH_WAIT = 5.0
MISSING_ABORT_FRACTION = 0.25
MARKET_TZ = "America/New_York"
CLOSE_HOUR = 16

CHAIN_COLS = ["snapshot", "expiry", "T", "cp", "strike", "bid", "ask", "lastPrice",
              "volume", "openInterest", "impliedVolatility", "lastTradeDate",
              "spot", "r"]


def trading_date(now=None):
    """Return the New York calendar date of `now` (UTC) as YYYY-MM-DD."""
    now = pd.Timestamp.now("UTC") if now is None else now
    return now.tz_convert(MARKET_TZ).strftime("%Y-%m-%d")


def session(now):
    """Label a UTC timestamp 'close' if after the New York cash close, else 'intraday'."""
    local = now.tz_convert(MARKET_TZ)
    after = local.hour >= CLOSE_HOUR or local.weekday() >= 5
    return "close" if after else "intraday"


def day_dir(date):
    return ROOT / date


def bill_rate(date):
    """Latest 3-month bill rate on or before `date`: (rate_decimal, observation_date, series).

    FRED DTB3 first (discount basis, percent, published with a one-day lag).
    Yahoo ^IRX as the fallback when FRED is unreachable. (None, None, None) if
    both fail; the day is still collected and the rate column left empty.
    """
    start = (pd.Timestamp(date) - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3&cosd={start}&coed={date}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        s = pd.read_csv(StringIO(r.text), na_values=".").dropna()
        if not s.empty:
            return float(s.iloc[-1, 1]) / 100, str(s.iloc[-1, 0]), "DTB3"
    except requests.RequestException:
        pass
    try:
        h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) / 100, h.index[-1].strftime("%Y-%m-%d"), "IRX"
    except Exception:
        return None, None, None


def _fetch_chain_once(ticker, now, r, max_days=MAX_DAYS):
    """Download every listed expiry up to max_days out; one row per contract.

    Calls and puts are stacked with a `cp` flag. T is in years to the 4pm New
    York expiry. Raises on network failure so the caller can record the miss.
    """
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")["Close"].dropna()
    spot = float(hist.iloc[-1])
    naive_now = now.tz_convert("UTC").tz_localize(None)
    frames = []
    for s in tk.options:
        expiry = pd.Timestamp(s) + pd.Timedelta(hours=20)
        days = (expiry - naive_now).days
        if not 0 <= days <= max_days:
            continue
        ch = tk.option_chain(s)
        for cp, df in (("C", ch.calls), ("P", ch.puts)):
            d = df.copy()
            d["cp"] = cp
            d["expiry"] = pd.Timestamp(s)
            d["T"] = (expiry - naive_now).total_seconds() / (365 * 86400)
            frames.append(d)
    if not frames:
        raise RuntimeError(f"{ticker}: no expiries listed inside {max_days} days")
    q = pd.concat(frames, ignore_index=True)
    q["snapshot"] = naive_now.floor("s")
    q["spot"] = spot
    q["r"] = r
    return q[CHAIN_COLS], spot, hist.index[-1].strftime("%Y-%m-%d")


def fetch_chain(ticker, now, r, max_days=MAX_DAYS, tries=FETCH_TRIES, wait=FETCH_WAIT):
    """_fetch_chain_once with retries. Raises the last error when it never comes back."""
    for i in range(tries):
        try:
            return _fetch_chain_once(ticker, now, r, max_days)
        except Exception as e:
            if i + 1 == tries:
                raise
            print(f"{ticker} fetch failed (try {i + 1}/{tries}): {type(e).__name__}: {e}",
                  file=sys.stderr)
            time.sleep(wait)


def market_caps(tickers):
    """Market cap per ticker from yfinance fast_info; names that fail are omitted."""
    out = {}
    for t in tickers:
        try:
            out[t] = float(yf.Ticker(t).fast_info["marketCap"])
        except Exception:
            pass
    return pd.Series(out, name="market_cap")


def list_days():
    """Dates with a raw snapshot folder, oldest first."""
    if not ROOT.exists():
        return []
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name[:2] == "20")


def load_day(date):
    """Return (chains by ticker, manifest dict) for one collected date."""
    d = day_dir(date)
    manifest = json.loads((d / "manifest.json").read_text())
    chains = {}
    for t in manifest["collected"]:
        chains[t] = pd.read_csv(d / f"{t}.csv", parse_dates=["snapshot", "expiry",
                                                             "lastTradeDate"])
    return chains, manifest


def load_weights(date):
    """Cap weights of the single names for `date`, normalised to one.

    Falls back to the most recent earlier day that has them, then to the
    dispersion project's frozen weights file.
    """
    for day in [date] + [d for d in reversed(list_days()) if d < date]:
        p = day_dir(day) / "weights.csv"
        if p.exists():
            w = pd.read_csv(p, index_col=0)["market_cap"].reindex(NAMES).dropna()
            if len(w) == len(NAMES):
                return w / w.sum(), day
    p = ROOT.parent / "dispersion" / "weights.csv"
    if p.exists():
        w = pd.read_csv(p, index_col=0)["weight"].reindex(NAMES)
        return w / w.sum(), "dispersion/weights.csv"
    return None, None


if __name__ == "__main__":
    print(f"storage root: {ROOT}")
    print(f"days collected: {len(list_days())}")
    for d in list_days()[-5:]:
        m = json.loads((day_dir(d) / "manifest.json").read_text())
        print(f"  {d}  {m['session']:9s} {len(m['collected'])} tickers  "
              f"{m['rows']} rows  missing: {m['missing'] or 'none'}")
