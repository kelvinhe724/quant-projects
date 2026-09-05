"""Hourly candlestick history behind every market in the prediction-markets snapshot, and the feature table built from it.

The snapshot (../prediction-markets/data.py::load_snapshot) is one quoted
book per settled market, read 24 hours before close. This module pulls the
96 hours of hourly candles that end at that quote, so the model can see
the recent price path, the volume traded before the quote and the open
interest at the quote. Nothing dated after the quote is read: the API
filters candles on end_period_ts, and features() drops any bar past the
target again.

The listing's `volume` and `open_interest` are settlement-time values and
never enter the feature table; volume after the quote is a leak.

Run: ../.venv/bin/python3 candles.py   (resumable, about 50 minutes at Kalshi's rate limit)
"""
import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def research_module(name):
    """Import ../prediction-markets/<name>.py by path, so its data.py never collides with anything on sys.path."""
    spec = importlib.util.spec_from_file_location(f"pm_{name}", os.path.join(ROOT, "prediction-markets", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pm = research_module("data")

API = pm.API
CACHE = os.path.join(ROOT, "source-material", "kalshi-model", "candles.csv")
TRIED = CACHE.replace(".csv", "_tried.txt")
WINDOW_HOURS = 96
THREADS = 24
CHUNK = 1000
CATEGORIES = ("Sports", "Climate and Weather", "Commodities", "Mentions", "Financials", "Entertainment",
              "Economics", "Politics", "Elections")
FEATURES = ("logit_mid", "spread", "log_spread_rel", "quote_age_h", "hours_to_close", "log_duration",
            "log_vol_96", "log_vol_24", "log_oi", "n_quoted", "n_traded", "mid_chg_24", "mid_chg_96",
            "mid_vol", "mid_range", "last_gap", "no_trade") + tuple(f"cat_{c}" for c in CATEGORIES)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def target_ts(close_time):
    return int(pd.Timestamp(close_time).timestamp()) - pm.HORIZON_HOURS * 3600


def fetch_candles(series, ticker, end_ts, hours=WINDOW_HOURS, session=None):
    """Hourly bars for one market ending at or before end_ts; None on a failed request."""
    session = session or requests
    url = f"{API}/series/{series}/markets/{ticker}/candlesticks"
    params = {"start_ts": end_ts - hours * 3600, "end_ts": end_ts, "period_interval": 60}
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=25)
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.0 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        return parse_bars(ticker, r.json().get("candlesticks", []), end_ts)
    return None


def parse_bars(ticker, bars, end_ts):
    rows = []
    for b in bars:
        end = int(b["end_period_ts"])
        if end > end_ts:
            continue
        rows.append({"ticker": ticker, "end_ts": end,
                     "bid": _f((b.get("yes_bid") or {}).get("close_dollars")),
                     "ask": _f((b.get("yes_ask") or {}).get("close_dollars")),
                     "last": _f((b.get("price") or {}).get("close_dollars")),
                     "volume": _f(b.get("volume_fp")), "oi": _f(b.get("open_interest_fp"))})
    return rows


def _pull(row):
    session = requests.Session()
    out = fetch_candles(row.series, row.ticker, target_ts(row.close_time), session=session)
    if out is None:
        return []
    return out or [{"ticker": row.ticker, "end_ts": np.nan, "bid": np.nan, "ask": np.nan, "last": np.nan,
                    "volume": np.nan, "oi": np.nan}]


def download(snap=None, budget_s=None):
    """Pull candles for every snapshot market not yet tried; append to the cache as it goes.

    budget_s stops after the chunk that crosses it, so the pull can be run in
    slices; every slice is a full write of its chunk.
    """
    snap = pm.load_snapshot() if snap is None else snap
    tried = set()
    if os.path.exists(TRIED):
        tried = {line.strip() for line in open(TRIED) if line.strip()}
    todo = snap[~snap["ticker"].isin(tried)]
    print(f"{len(snap):,} snapshot markets, {len(tried):,} tried, {len(todo):,} to go, {THREADS} threads")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as pool:
        for start in range(0, len(todo), CHUNK):
            chunk = todo.iloc[start:start + CHUNK]
            rows = [r for rs in pool.map(_pull, chunk.itertuples(index=False)) for r in rs]
            if rows:
                pd.DataFrame(rows).to_csv(CACHE, mode="a", header=not os.path.exists(CACHE), index=False)
            with open(TRIED, "a") as fh:
                fh.write("".join(t + "\n" for t in chunk["ticker"]))
            got = len({r["ticker"] for r in rows})
            print(f"  {start + len(chunk):,}/{len(todo):,} tried, {got:,} answered, {time.time() - t0:.0f}s",
                  flush=True)
            if budget_s and time.time() - t0 > budget_s:
                print("  budget reached, rerun to continue")
                break


def load_candles():
    c = pd.read_csv(CACHE)
    return c.dropna(subset=["end_ts"]).drop_duplicates(["ticker", "end_ts"]).sort_values(["ticker", "end_ts"])


def path_features(bars, end_ts):
    """Recent-path, volume and open-interest features from one market's bars up to end_ts.

    bars: frame with end_ts, bid, ask, last, volume, oi. Works for a settled
    market (end_ts the T-24h target) and a live one (end_ts now).
    """
    b = bars[bars["end_ts"] <= end_ts]
    q = b[(b["bid"] > 0) & (b["ask"] < 1) & (b["bid"] < b["ask"])]
    out = {"n_quoted": len(q), "n_traded": int(b["last"].notna().sum()),
           "log_vol_96": np.log1p(b["volume"].fillna(0).sum()),
           "log_vol_24": np.log1p(b.loc[b["end_ts"] > end_ts - 24 * 3600, "volume"].fillna(0).sum()),
           "log_oi": np.log1p(b["oi"].dropna().iloc[-1]) if b["oi"].notna().any() else 0.0}
    if len(q) == 0:
        out.update(quote_age_h=np.nan, mid_chg_24=0.0, mid_chg_96=0.0, mid_vol=0.0, mid_range=0.0,
                   last_gap=0.0, no_trade=1.0)
        return out
    mid = (q["bid"] + q["ask"]) / 2
    last_end = int(q["end_ts"].iloc[-1])
    out["quote_age_h"] = (end_ts - last_end) / 3600
    back = mid[q["end_ts"] <= last_end - 24 * 3600]
    out["mid_chg_24"] = float(mid.iloc[-1] - (back.iloc[-1] if len(back) else mid.iloc[0]))
    out["mid_chg_96"] = float(mid.iloc[-1] - mid.iloc[0])
    out["mid_vol"] = float(mid.diff().std(ddof=0)) if len(mid) > 1 else 0.0
    out["mid_range"] = float(mid.max() - mid.min())
    trades = b["last"].dropna()
    out["last_gap"] = float(trades.iloc[-1] - mid.iloc[-1]) if len(trades) else 0.0
    out["no_trade"] = float(len(trades) == 0)
    return out


def market_features(mid, spread, hours_to_close, duration_hours, category, path):
    """One feature row from the quote, the market's clock, its category and path_features()."""
    m = float(np.clip(mid, 1e-3, 1 - 1e-3))
    row = {"logit_mid": float(np.log(m / (1 - m))), "spread": float(spread),
           "log_spread_rel": float(np.log(spread / min(m, 1 - m))), "quote_age_h": path["quote_age_h"],
           "hours_to_close": float(hours_to_close), "log_duration": float(np.log(max(duration_hours, 1.0)))}
    row.update({k: v for k, v in path.items() if k != "quote_age_h"})
    for c in CATEGORIES:
        row[f"cat_{c}"] = float(category == c)
    return row


def build(snap=None, candles=None):
    """Feature table: one row per snapshot market with candles, plus the columns kelly.py reads."""
    snap = pm.load_snapshot() if snap is None else snap
    candles = load_candles() if candles is None else candles
    groups = {t: g for t, g in candles.groupby("ticker")}
    rows = []
    for r in snap.itertuples(index=False):
        g = groups.get(r.ticker)
        if g is None:
            continue
        end = target_ts(r.close_time)
        p = path_features(g, end)
        age = p["quote_age_h"] if np.isfinite(p["quote_age_h"]) else 0.0
        p["quote_age_h"] = age
        f = market_features(r.price, r.spread, pm.HORIZON_HOURS + age, r.duration_hours, r.category, p)
        f.update(ticker=r.ticker, series=r.series, category=r.category, close_time=r.close_time,
                 bid=r.bid, ask=r.ask, price=r.price, outcome=int(r.outcome), settle_volume=r.volume,
                 quote_time=pd.Timestamp(end, unit="s", tz="UTC"))
        rows.append(f)
    df = pd.DataFrame(rows).sort_values(["close_time", "ticker"]).reset_index(drop=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


if __name__ == "__main__":
    download(budget_s=float(sys.argv[1]) if len(sys.argv) > 1 else None)
