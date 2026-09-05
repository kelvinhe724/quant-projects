"""Simulated crypto fills from the data lake's L2 stream (top 20 levels, once a second).

Market orders cross the spread and walk the opposite side of the book level by
level. Limit orders join the queue at their price and are filled by a
queue-position model over the following snapshots. Both take snapshot rows in
the lake's `crypto_l2` schema (bid_px_i / bid_sz_i / ask_px_i / ask_sz_i), so
the model is tested on planted books and only the two loaders touch the lake.
"""
import os
import sys

import numpy as np
import pandas as pd

LAKE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data-lake")
LEVELS = 20
SYMBOL = {"BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT"}
SNAPSHOT_WAIT = pd.Timedelta(seconds=5)
LIMIT_HORIZON = pd.Timedelta(minutes=60)


def snapshots(name, start, end):
    """Rows of the lake's L2 stream for one book instrument between two UTC instants."""
    if LAKE not in sys.path:
        sys.path.insert(0, LAKE)
    import lake

    df = lake.load("crypto_l2", pd.Timestamp(start), pd.Timestamp(end), universe=[SYMBOL[name]])
    return df.sort_values("ts").reset_index(drop=True)


def levels(row, side):
    """(prices, sizes) of one side of a snapshot, empty levels dropped, best first."""
    px = np.array([row.get(f"{side}_px_{i}", np.nan) for i in range(1, LEVELS + 1)], dtype=float)
    sz = np.array([row.get(f"{side}_sz_{i}", np.nan) for i in range(1, LEVELS + 1)], dtype=float)
    m = np.isfinite(px) & np.isfinite(sz) & (sz > 0)
    return px[m], sz[m]


def walk(row, side, qty=None, notional=None):
    """Cross the spread: a buy eats asks from the touch up, a sell eats bids from the touch down.

    Returns {"price": average, "filled": qty, "exhausted": bool}. If the
    visible book runs out, the remainder is priced at the last level and
    flagged, which is optimistic for anything bigger than the top 20 levels.
    """
    px, sz = levels(row, "ask" if side == "buy" else "bid")
    if not len(px):
        return {"price": np.nan, "filled": 0.0, "exhausted": True}
    if qty is None:
        qty = float(notional) / px[0]
    qty = float(qty)
    ahead = np.concatenate([[0.0], np.cumsum(sz)[:-1]])
    take = np.clip(qty - ahead, 0.0, sz)
    filled = take.sum()
    cost = (take * px).sum()
    exhausted = filled < qty - 1e-12
    if exhausted:
        cost += (qty - filled) * px[-1]
        filled = qty
    return {"price": float(cost / filled), "filled": float(filled), "exhausted": bool(exhausted)}


def queue(rows, side, qty, price):
    """Queue-position fill of a resting limit over successive snapshots.

    Placement (first row): a price that crosses the touch is marketable and
    walks the book up to the limit. Otherwise the order joins behind whatever
    rests at its price. Each later row: if the far touch reaches the price, the
    level was traded through and the rest fills at the limit; else any drop in
    the resting size at the price is counted as executions ahead, and once
    those pass the queue the order fills by the same amount. Size that joins
    the level is behind and ignored.

    ponytail: a drop in resting size is read as trades, never as cancels, so
    fills come early; tape data (trades) would split the two.
    Returns {"price", "filled", "remaining", "ts"} where ts is the last row seen.
    """
    rows = list(rows)
    if not rows:
        return {"price": np.nan, "filled": 0.0, "remaining": float(qty), "ts": None}
    own, far = ("bid", "ask") if side == "buy" else ("ask", "bid")
    crosses = (lambda touch: touch <= price) if side == "buy" else (lambda touch: touch >= price)
    remaining, filled, cost = float(qty), 0.0, 0.0

    def at_price(row):
        px, sz = levels(row, own)
        hit = np.isclose(px, price)
        return float(sz[hit][0]) if hit.any() else 0.0

    first = rows[0]
    far_px, _ = levels(first, far)
    if len(far_px) and crosses(far_px[0]):
        px, sz = levels(first, far)
        ok = px <= price if side == "buy" else px >= price
        px, sz = px[ok], sz[ok]
        take = np.clip(remaining - np.concatenate([[0.0], np.cumsum(sz)[:-1]]), 0.0, sz)
        filled += take.sum()
        cost += (take * px).sum()
        remaining -= take.sum()
        if remaining <= 1e-12:
            return {"price": float(cost / filled), "filled": float(filled), "remaining": 0.0, "ts": first.get("ts")}
    ahead = at_price(first)
    prev = ahead
    ts = first.get("ts")
    for row in rows[1:]:
        ts = row.get("ts")
        far_px, _ = levels(row, far)
        if len(far_px) and crosses(far_px[0]):
            filled += remaining
            cost += remaining * price
            remaining = 0.0
            break
        now = at_price(row)
        if now < prev:
            done = prev - now
            past = max(done - ahead, 0.0)
            ahead = max(ahead - done, 0.0)
            got = min(past, remaining)
            if got > 0:
                filled += got
                cost += got * price
                remaining -= got
                if remaining <= 1e-12:
                    break
        prev = now
    return {"price": float(cost / filled) if filled else np.nan, "filled": float(filled), "remaining": float(remaining), "ts": ts}


def market_fill(name, side, ts, qty=None, notional=None):
    """Market order at instant `ts` against the first snapshot at or after it (within 5 s)."""
    ts = pd.Timestamp(ts).tz_convert("UTC")
    df = snapshots(name, ts, ts + SNAPSHOT_WAIT)
    if not len(df):
        raise ValueError(f"no L2 snapshot for {name} within {SNAPSHOT_WAIT} of {ts}")
    return walk(df.iloc[0], side, qty=qty, notional=notional)


def limit_fill(name, side, qty, price, ts, horizon=LIMIT_HORIZON):
    """Resting limit placed at `ts`, worked through the stream for `horizon`."""
    ts = pd.Timestamp(ts).tz_convert("UTC")
    df = snapshots(name, ts, ts + horizon)
    if not len(df):
        raise ValueError(f"no L2 snapshots for {name} after {ts}")
    return queue((r for _, r in df.iterrows()), side, qty, price)
