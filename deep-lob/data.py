"""One-second bars for the DeepLOB project: the lake's L2 stream when it holds enough, Binance aggTrades otherwise.

Both sources come out as one frame indexed by UTC second with a `mid` column
and a block of feature columns. The lake stream gives twenty levels a side and
no trades. The archive gives no book at all, so its touch is reconstructed
from the last taker-buy and taker-sell prints of each second and the rest of
the block is trade flow. `load()` picks the lake once it holds MIN_HOURS of
the symbol and says which one it used.
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "source-material", "deep-lob")
sys.path.insert(0, os.path.join(ROOT, "data-lake"))
sys.path.insert(0, os.path.join(ROOT, "orderbook-imbalance"))
import lake  # noqa: E402
from imbalance import imbalance  # noqa: E402

SYMBOL = "BTCUSDT"
LEVELS = 20
MIN_HOURS = 6
DAYS = tuple(str(d.date()) for d in pd.date_range("2026-08-28", "2026-09-03"))
URL = "https://data.binance.vision/data/spot/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
COLS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker", "best_match"]
BPS = 1e4


def fetch(day, symbol=SYMBOL):
    """Path of one day's aggTrades archive, downloaded into RAW if it is not there yet."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{symbol}-aggTrades-{day}.zip")
    if not os.path.exists(path):
        r = requests.get(URL.format(s=symbol, d=day), timeout=120)
        r.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(r.content)
    return path


def read_trades(day, symbol=SYMBOL):
    """One day's aggregated trades: ts (UTC), price, qty, is_buyer_maker."""
    with zipfile.ZipFile(fetch(day, symbol)) as z, z.open(z.namelist()[0]) as fh:
        df = pd.read_csv(fh, header=None, names=COLS, usecols=["price", "qty", "ts", "is_buyer_maker"])
    if df["price"].dtype == object:
        df = df.iloc[1:]
        df = df.astype({"price": float, "qty": float, "ts": np.int64})
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower() == "true"
    unit = "us" if df["ts"].iloc[0] > 1e14 else "ms"
    df["ts"] = pd.to_datetime(df["ts"], unit=unit, utc=True)
    return df.reset_index(drop=True)


def seconds_from_trades(trades):
    """One row per second: last price as `mid`, taker flow, and the touch reconstructed from the prints.

    A taker buy prints at the ask and a taker sell at the bid, so the last of
    each inside the second is the best quote as of that print. Seconds with
    no trade carry the previous price and zero flow.
    """
    t = trades.sort_values("ts")
    sec = t["ts"].dt.floor("s")
    taker_buy = ~t["is_buyer_maker"].to_numpy()
    g = t.groupby(sec)
    out = pd.DataFrame({
        "mid": g["price"].last(),
        "n_trades": g.size().astype(float),
        "buy_vol": t["qty"].where(taker_buy, 0.0).groupby(sec).sum(),
        "sell_vol": t["qty"].where(~taker_buy, 0.0).groupby(sec).sum(),
        "notional": (t["price"] * t["qty"]).groupby(sec).sum(),
        "ask": t["price"].where(taker_buy).groupby(sec).last(),
        "bid": t["price"].where(~taker_buy).groupby(sec).last(),
    })
    out = out.reindex(pd.date_range(out.index[0], out.index[-1], freq="s"))
    out[["mid", "ask", "bid"]] = out[["mid", "ask", "bid"]].ffill()
    out[["n_trades", "buy_vol", "sell_vol", "notional"]] = out[["n_trades", "buy_vol", "sell_vol", "notional"]].fillna(0.0)
    vol = out["buy_vol"] + out["sell_vol"]
    vwap = (out["notional"] / vol.where(vol > 0)).fillna(out["mid"])
    mid = out["mid"]
    return pd.DataFrame({
        "mid": mid,
        "ret_bps": np.log(mid).diff().fillna(0.0) * BPS,
        "buy_vol": out["buy_vol"],
        "sell_vol": out["sell_vol"],
        "n_trades": out["n_trades"],
        "vwap_bps": (vwap / mid - 1) * BPS,
        "bid_bps": (out["bid"] / mid - 1) * BPS,
        "ask_bps": (out["ask"] / mid - 1) * BPS,
        "flow_imb": imbalance(out[["buy_vol"]].to_numpy(), out[["sell_vol"]].to_numpy(), 1),
    })


def seconds_from_lake(rows, symbol=SYMBOL, levels=LEVELS):
    """The lake's L2 rows for one symbol on a one-second grid: mid, then each level's price offset in bps and size."""
    r = rows[rows["symbol"] == symbol].copy()
    r["ts"] = pd.to_datetime(r["ts"], utc=True).dt.floor("s")
    r = r.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    mid = (r["bid_px_1"] + r["ask_px_1"]) / 2
    cols = {"mid": mid}
    for i in range(1, levels + 1):
        for side in ("bid", "ask"):
            cols[f"{side}_px_{i}"] = (r[f"{side}_px_{i}"] / mid - 1) * BPS
            cols[f"{side}_sz_{i}"] = r[f"{side}_sz_{i}"]
    out = pd.DataFrame(cols)
    return out.reindex(pd.date_range(out.index[0], out.index[-1], freq="s")).ffill()


def lake_rows(symbol=SYMBOL):
    """Every L2 row the lake holds for the symbol."""
    return lake.load("crypto_l2", universe=[symbol])


def load(source="auto", days=DAYS, symbol=SYMBOL):
    """(frame, meta). source: "auto" takes the lake once it holds MIN_HOURS of the symbol, else the archive."""
    meta = {"symbol": symbol}
    if source in ("auto", "lake"):
        rows = lake_rows(symbol)
        meta["lake_rows"] = int(len(rows))
        meta["lake_hours"] = float(len(rows) / 3600)
        if source == "lake" or meta["lake_hours"] >= MIN_HOURS:
            frame = seconds_from_lake(rows, symbol)
            meta.update(source="lake crypto_l2", venue=sorted(rows["venue"].unique().tolist()), levels=LEVELS)
            return frame, meta
    frames = [seconds_from_trades(read_trades(d, symbol)) for d in days]
    frame = pd.concat(frames)
    frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="s"))
    frame["mid"] = frame["mid"].ffill()
    frame = frame.fillna(0.0)
    meta.update(source="data.binance.vision spot aggTrades", days=list(days), levels=0,
                backfilled=True, reason=f"lake stream below {MIN_HOURS} hours")
    return frame, meta


def features(frame):
    """Every column but the price."""
    return frame.drop(columns=["mid"])
