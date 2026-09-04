"""Scan open Kalshi markets, score each against the fitted longshot-bias curve, rank."""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.join(HERE, "..", "prediction-markets")
sys.path.insert(0, RESEARCH)

from calibration import bias_curve, bin_calibration  # noqa: E402
from kelly import kelly_fraction  # noqa: E402

import kalshi  # noqa: E402

CALIBRATION_CSV = os.path.join(RESEARCH, "reports", "calibration_quoted_mid.csv")
CACHE_DIR = os.path.join(HERE, "..", "source-material", "kalshi-desk")

# The backtest priced every market 24h before close. Keep the forward test in
# the same neighbourhood rather than trading things that settle in ten minutes.
MIN_HOURS = 6
MAX_HOURS = 72
MAX_SPREAD = 0.10
MIN_VOLUME = 10
MAX_PAGES = 5
TOP_BOOKS = 40
BANKROLL_FALLBACK = 100.0


def load_curve():
    """Price -> estimated true probability, from the research calibration table."""
    if os.path.exists(CALIBRATION_CSV):
        table = pd.read_csv(CALIBRATION_CSV)
    else:
        import data  # the research downloader; only if the table is missing
        snap = data.load_snapshot()
        table = bin_calibration(snap["price"], snap["outcome"])
    return bias_curve(table)


def hours_to_close(m, now=None):
    now = now or time.time()
    t = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()
    return (t - now) / 3600


def candidates(client, max_pages=MAX_PAGES):
    """Open binary markets closing in the window with a tight two-sided quote."""
    now = int(time.time())
    listed = client.markets(max_pages=max_pages,
                            min_close_ts=now + MIN_HOURS * 3600,
                            max_close_ts=now + MAX_HOURS * 3600)
    rows = []
    for m in listed:
        if m.get("market_type") != "binary" or m.get("status") != "active":
            continue
        q = kalshi.quote(m)
        if q is None or q[1] - q[0] > MAX_SPREAD:
            continue
        if float(m.get("volume_fp") or 0) < MIN_VOLUME:
            continue
        rows.append({
            "ticker": m["ticker"],
            "title": m.get("title", ""),
            "close_time": m["close_time"],
            "hours": hours_to_close(m, now),
            "volume": float(m["volume_fp"]),
            "bid": q[0],
            "ask": q[1],
            "bid_size": float(m.get("yes_bid_size_fp") or 0),
            "ask_size": float(m.get("yes_ask_size_fp") or 0),
        })
    return len(listed), rows


def score(rows, curve, k=0.25, bankroll=BANKROLL_FALLBACK, max_order=5.0,
          max_open=50.0, min_edge=0.0):
    """Side, edge after the spread, Kelly stake and contract count per market.

    Buying YES costs the ask; buying NO costs one minus the bid. That is the
    "quoted spread" scenario of the backtest, the one that lost 1.2 cents.
    """
    out = []
    for r in rows:
        bid, ask = r["bid"], r["ask"]
        mid = (bid + ask) / 2
        p = float(curve(mid))
        f_yes = float(kelly_fraction(p, ask))
        f_no = float(kelly_fraction(1 - p, 1 - bid))
        if f_yes >= f_no:
            side, f, cost, p_side, mid_cost = "yes", f_yes, ask, p, mid
        else:
            side, f, cost, p_side, mid_cost = "no", f_no, 1 - bid, 1 - p, 1 - mid
        if f <= min_edge:
            continue
        stake = min(k * f * bankroll, max_order)
        contracts = math.floor(stake / cost)
        out.append({
            **r,
            "mid": mid,
            "spread": ask - bid,
            "p_model": p,
            "side": side,
            "cost": cost,
            "edge_mid": p_side - mid_cost,
            "edge": p_side - cost,
            "f": f,
            "stake": stake,
            "contracts": contracts,
        })
    out.sort(key=lambda x: x["edge"], reverse=True)
    return out


def enrich_with_books(client, ranked, top=TOP_BOOKS):
    """Replace listing quotes with the live orderbook touch for the top rows."""
    kept = []
    for r in ranked[:top]:
        try:
            t = kalshi.touch(client.orderbook(r["ticker"], depth=5))
        except Exception as e:  # one dead book must not kill the scan
            print(f"  orderbook {r['ticker']}: {e}", file=sys.stderr)
            continue
        if t is None or t["ask"] - t["bid"] > MAX_SPREAD:
            continue
        kept.append({**r, **t})
    return kept


def cache(payload):
    os.makedirs(CACHE_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(CACHE_DIR, f"scan_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    return path


def latest_scan():
    if not os.path.isdir(CACHE_DIR):
        return None
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith("scan_"))
    if not files:
        return None
    with open(os.path.join(CACHE_DIR, files[-1])) as fh:
        return json.load(fh)


def run(client=None, settings=None, bankroll=None, quiet=False):
    settings = settings or kalshi.load_env()
    client = client or kalshi.Client("prod", settings)
    k = float(settings["KELLY_FRACTION"])
    max_order = float(settings["MAX_ORDER_DOLLARS"])
    max_open = float(settings["MAX_OPEN_DOLLARS"])
    bankroll = bankroll or BANKROLL_FALLBACK
    curve = load_curve()

    n_listed, rows = candidates(client)
    prelim = score(rows, curve, k, bankroll, max_order, max_open)
    booked = enrich_with_books(client, prelim)
    ranked = score(booked, curve, k, bankroll, max_order, max_open)
    path = cache({
        "ts": datetime.now(timezone.utc).isoformat(),
        "env": client.env,
        "n_listed": n_listed,
        "n_candidates": len(rows),
        "bankroll": bankroll,
        "k": k,
        "ranked": ranked,
    })
    if not quiet:
        print(f"{n_listed} markets listed, {len(rows)} tight two-sided candidates, "
              f"{len(ranked)} scored with live books. Cached to {path}")
        print(table(ranked))
    return ranked


def table(ranked, n=25):
    if not ranked:
        return "no candidates"
    df = pd.DataFrame(ranked[:n])[
        ["ticker", "side", "bid", "ask", "mid", "p_model", "edge_mid", "edge",
         "f", "stake", "contracts", "hours", "volume"]
    ]
    return df.to_string(index=False, formatters={
        "bid": "{:.2f}".format, "ask": "{:.2f}".format, "mid": "{:.3f}".format,
        "p_model": "{:.3f}".format, "edge_mid": "{:+.3f}".format,
        "edge": "{:+.3f}".format, "f": "{:.3f}".format, "stake": "{:.2f}".format,
        "hours": "{:.0f}".format, "volume": "{:.0f}".format,
    })


if __name__ == "__main__":
    run()
