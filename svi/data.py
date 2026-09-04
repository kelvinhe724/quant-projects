"""SPY option chains from Yahoo: download, cache, and clean to a mid-quote panel."""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPORTS = Path(__file__).resolve().parent / "reports"


def fetch_chain(ticker="SPY", min_days=7, max_days=400, max_expiries=8):
    """Download every listed expiry between min_days and max_days out.

    Returns (quotes, spot, snapshot_time). One row per contract, calls and
    puts stacked with a `cp` flag.
    """
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    r = bill_rate()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    expiries = []
    for s in tk.options:
        days = (pd.Timestamp(s) + pd.Timedelta(hours=20) - now).days
        if min_days <= days <= max_days:
            expiries.append(s)
    # thin the ladder so the slices are spread over the year instead of
    # bunched in the first month
    if len(expiries) > max_expiries:
        idx = np.linspace(0, len(expiries) - 1, max_expiries).round().astype(int)
        expiries = [expiries[i] for i in dict.fromkeys(idx)]

    frames = []
    for s in expiries:
        ch = tk.option_chain(s)
        for cp, df in (("C", ch.calls), ("P", ch.puts)):
            d = df.copy()
            d["cp"] = cp
            d["expiry"] = pd.Timestamp(s)
            frames.append(d)

    q = pd.concat(frames, ignore_index=True)
    q["snapshot"] = now
    q["spot"] = spot
    q["r"] = r
    q["T"] = (q["expiry"] + pd.Timedelta(hours=20) - now).dt.total_seconds() / (365 * 86400)
    cols = ["snapshot", "expiry", "T", "cp", "strike", "bid", "ask", "lastPrice",
            "volume", "openInterest", "impliedVolatility", "lastTradeDate",
            "spot", "r"]
    return q[cols], spot, now


def cache_path(ticker, snapshot):
    return REPORTS / f"chain_{ticker}_{snapshot:%Y-%m-%d_%H%M}.csv"


def load_or_fetch(ticker="SPY", refresh=False, **kw):
    """Use the newest cached chain unless refresh is set, else download and cache one.

    Pinning to the cache is what makes a rerun reproduce the reported numbers;
    a fresh snapshot is a different market and gives different fits.
    """
    REPORTS.mkdir(exist_ok=True)
    cached = sorted(REPORTS.glob(f"chain_{ticker}_*.csv"))
    if cached and not refresh:
        q = pd.read_csv(cached[-1], parse_dates=["snapshot", "expiry", "lastTradeDate"])
        return q, float(q["spot"].iloc[0]), q["snapshot"].iloc[0]
    q, spot, now = fetch_chain(ticker, **kw)
    q.to_csv(cache_path(ticker, now), index=False)
    return q, spot, now


def bill_rate():
    """13-week T-bill discount rate from ^IRX, as a decimal."""
    h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
    return float(h.iloc[-1]) / 100


def forward_and_discount(q, r):
    """Forward per expiry from put-call parity: F = K + (C - P) / D.

    D is exp(-rT) off a flat bill rate. The dividend yield never appears
    because the parity forward already contains it, which is the point of
    doing it this way rather than assuming a yield.
    """
    out = []
    for (exp, T), g in q.groupby(["expiry", "T"]):
        piv = g.pivot_table(index="strike", columns="cp", values="mid")
        if not {"C", "P"}.issubset(piv.columns):
            continue
        piv = piv.dropna()
        spot = g["spot"].iloc[0]
        near = piv[(piv.index > 0.95 * spot) & (piv.index < 1.05 * spot)]
        if len(near) < 3:
            continue
        D = np.exp(-r * T)
        K = near.index.to_numpy(float)
        F = np.median(K + (near["C"] - near["P"]).to_numpy(float) / D)
        out.append({"expiry": exp, "T": T, "F": F, "D": D,
                    "basis": F / spot - 1, "n_parity": len(near)})
    return pd.DataFrame(out).sort_values("T").reset_index(drop=True)


def clean(q, r=None, max_rel_spread=0.25, max_abs_spread=0.10,
          max_abs_k=0.5, stale_days=1, min_quotes=12):
    """Drop unusable quotes and keep the out-of-the-money wing of each expiry.

    Filters: positive bid, positive spread, traded within stale_days, non-zero
    volume, spread inside either the relative or the absolute cap, and
    log-moneyness inside max_abs_k. Returns (clean_quotes, forwards, counts).
    """
    counts = {"raw": len(q)}
    q = q.copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2

    q = q[q["bid"] > 0]
    counts["after zero bid"] = len(q)
    q = q[q["ask"] > q["bid"]]
    counts["after crossed/locked"] = len(q)

    # the forward is fitted before the liquidity filters, on every strike that
    # still has both legs, because parity needs a wide strike range
    r = float(q["r"].iloc[0]) if r is None else r
    fwd = forward_and_discount(q, r)

    last = pd.to_datetime(q["lastTradeDate"], utc=True).dt.tz_localize(None)
    snap = pd.to_datetime(q["snapshot"]).dt.tz_localize(None)
    q = q[(snap - last) <= pd.Timedelta(days=stale_days)]
    counts["after stale quotes"] = len(q)

    q = q[q["volume"].fillna(0) > 0]
    counts["after no volume"] = len(q)

    spread = q["ask"] - q["bid"]
    q = q[(spread / q["mid"] <= max_rel_spread) | (spread <= max_abs_spread)]
    counts["after wide spread"] = len(q)

    q = q.merge(fwd[["expiry", "F", "D"]], on="expiry", how="inner")
    counts["after parity forward"] = len(q)

    q["k"] = np.log(q["strike"] / q["F"])
    q = q[q["k"].abs() <= max_abs_k]
    counts["after moneyness band"] = len(q)

    # keep only the out-of-the-money side: puts below the forward, calls above.
    # In-the-money quotes carry the same information through parity but trade
    # wider, and mixing both would double-count every strike.
    q = q[((q["cp"] == "P") & (q["k"] < 0)) | ((q["cp"] == "C") & (q["k"] >= 0))]
    counts["after OTM only"] = len(q)

    # five free parameters need more than a handful of strikes to pin down
    q = q.groupby("expiry").filter(lambda g: len(g) >= min_quotes)
    counts["after thin slices"] = len(q)

    return q.reset_index(drop=True), fwd, counts
