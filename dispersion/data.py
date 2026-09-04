"""Daily prices for SPY, its ten largest constituents and VIX since 2015, plus a
snapshot of each name's option chain, cached under source-material/dispersion/
so reruns are offline."""
import os

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "source-material", "dispersion")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

INDEX = "SPY"
VIX = "^VIX"
NAMES = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "XOM", "UNH"]

START = "2015-01-01"
END = "2026-09-04"
FORMATION_END = "2020-12-31"
TEST_START = "2021-01-01"

# Only expiries in this window are pulled. Shorter ones are dominated by pin risk
# and the weekly cycle, longer ones are thin on most single names.
MIN_DAYS, MAX_DAYS = 14, 75
RISK_FREE = 0.04


def _read(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_daily():
    """Return adjusted closes for the names, SPY and VIX as one frame."""
    path = os.path.join(CACHE, "daily.csv")
    if os.path.exists(path):
        return _read(path)
    px = yf.download(NAMES + [INDEX, VIX], start=START, end=END, auto_adjust=True,
                     progress=False)["Close"]
    px = px[NAMES + [INDEX, VIX]].dropna(how="all")
    os.makedirs(CACHE, exist_ok=True)
    px.to_csv(path)
    return px


def get_weights():
    """Return the basket weights, market cap of each name normalised to sum to one.

    SPY is float-adjusted cap weighted, so within the basket cap weights are the
    right proportions up to the float adjustment. The basket itself is roughly a
    third of SPY, which the README discusses.
    """
    path = os.path.join(CACHE, "weights.csv")
    if os.path.exists(path):
        w = _read(path)["weight"]
    else:
        caps = {t: float(yf.Ticker(t).fast_info["marketCap"]) for t in NAMES}
        w = pd.Series(caps, name="weight")
        w = w / w.sum()
        os.makedirs(CACHE, exist_ok=True)
        w.to_frame().to_csv(path)
    return w.reindex(NAMES)


def fetch_chain(ticker):
    """Download every listed expiry inside the day window with bid, ask and spot."""
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    now = pd.Timestamp.now("UTC").tz_localize(None)
    frames = []
    for s in tk.options:
        expiry = pd.Timestamp(s) + pd.Timedelta(hours=16)
        days = (expiry - now).days
        if not MIN_DAYS <= days <= MAX_DAYS:
            continue
        ch = tk.option_chain(s)
        for cp, df in (("C", ch.calls), ("P", ch.puts)):
            d = df[["strike", "bid", "ask", "volume", "openInterest"]].copy()
            d["cp"] = cp
            d["expiry"] = pd.Timestamp(s)
            d["T"] = (expiry - now).total_seconds() / (365 * 86400)
            frames.append(d)
    if not frames:
        return None
    q = pd.concat(frames, ignore_index=True)
    q["ticker"] = ticker
    q["spot"] = spot
    q["snapshot"] = now.floor("min")
    return q


def get_chains():
    """Return one frame of quotes for SPY and every name, from cache if present.

    Delete source-material/dispersion/chains_*.csv to take a fresh snapshot.
    Names whose download fails are reported, not filled in.
    """
    hit = sorted(f for f in os.listdir(CACHE) if f.startswith("chains_")) \
        if os.path.isdir(CACHE) else []
    if hit:
        return pd.read_csv(os.path.join(CACHE, hit[-1]), parse_dates=["expiry", "snapshot"])
    frames, missing = [], []
    for t in [INDEX] + NAMES:
        try:
            q = fetch_chain(t)
        except Exception as e:
            q = None
            print(f"chain download failed for {t}: {e}")
        if q is None:
            missing.append(t)
        else:
            frames.append(q)
    if missing:
        print(f"no chain for: {', '.join(missing)}")
    if not frames:
        return None
    q = pd.concat(frames, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    q.to_csv(os.path.join(CACHE, f"chains_{q['snapshot'].iloc[0]:%Y-%m-%d}.csv"), index=False)
    return q


def get_cboe_correlation():
    """Return today's Cboe 1-month and 3-month implied correlation prints, or None.

    Yahoo carries only the latest value of ^COR1M and ^COR3M, not history, so
    these serve as a same-day cross-check on the chain-derived number and
    nothing more.
    """
    path = os.path.join(CACHE, "cboe_cor.csv")
    if os.path.exists(path):
        return _read(path)
    try:
        h = yf.download(["^COR1M", "^COR3M"], period="5d", progress=False,
                        auto_adjust=True)["Close"].dropna(how="all")
    except Exception as e:
        print(f"Cboe correlation index download failed: {e}")
        return None
    if h.empty:
        return None
    os.makedirs(CACHE, exist_ok=True)
    h.to_csv(path)
    return h


def log_returns(px):
    # Drop incomplete rows before differencing. A row that is NaN for every
    # name (Yahoo posts VIX on days it has no equity closes) would otherwise
    # take the next day's return down with it.
    return np.log(px[NAMES + [INDEX]].dropna()).diff().dropna()


if __name__ == "__main__":
    px = get_daily()
    print(f"{px.index[0].date()} to {px.index[-1].date()}, {len(px)} days")
    print(get_weights().round(4).to_string())
    q = get_chains()
    if q is not None:
        print(q.groupby("ticker")["expiry"].nunique().to_string())
