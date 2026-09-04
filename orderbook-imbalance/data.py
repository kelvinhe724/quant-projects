"""Collect and load BTC-USDT order-book snapshots and trades from OKX's public REST API.

Binance's depth endpoint returns HTTP 451 (geo-blocked) from this machine, so the
collector uses OKX, which needs no key. Snapshots are polled at a fixed interval
for a bounded window and written to CSV as they arrive, so a crash or Ctrl-C keeps
whatever was collected.

Collect:  python3 data.py 45        (minutes; default 45)
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "source-material", "orderbook")
BOOK_CSV = os.path.join(RAW, "okx_btcusdt_depth.csv")
TRADES_CSV = os.path.join(RAW, "okx_btcusdt_trades.csv")

EXCHANGE = "OKX"
INST = "BTC-USDT"
LEVELS = 20
BOOK_URL = f"https://www.okx.com/api/v5/market/books?instId={INST}&sz={LEVELS}"
# 500 is the endpoint's cap. At 100 the 2s poll dropped 19% of trades in the
# collected sample (BTC-USDT prints more than 100 trades in a busy 2s window).
TRADES_URL = f"https://www.okx.com/api/v5/market/trades?instId={INST}&limit=500"
INTERVAL_S = 1.0
TRADES_EVERY = 2

BOOK_COLS = (["ts_local_ms", "ts_exchange_ms"]
             + [f"bid_px_{i}" for i in range(1, LEVELS + 1)]
             + [f"bid_sz_{i}" for i in range(1, LEVELS + 1)]
             + [f"ask_px_{i}" for i in range(1, LEVELS + 1)]
             + [f"ask_sz_{i}" for i in range(1, LEVELS + 1)])
TRADE_COLS = ["trade_id", "ts_ms", "px", "sz", "side"]


def flatten_book(payload, ts_local_ms):
    """Turn one OKX depth payload into a flat row, bids and asks best-first."""
    book = payload["data"][0]
    bids, asks = book["bids"][:LEVELS], book["asks"][:LEVELS]
    if len(bids) < LEVELS or len(asks) < LEVELS:
        return None
    row = [ts_local_ms, int(book["ts"])]
    row += [float(b[0]) for b in bids] + [float(b[1]) for b in bids]
    row += [float(a[0]) for a in asks] + [float(a[1]) for a in asks]
    return row


def collect(minutes=45):
    """Poll the book every second and trades every other second for `minutes`."""
    os.makedirs(RAW, exist_ok=True)
    session = requests.Session()
    seen_trades = set()
    deadline = time.time() + minutes * 60
    n_book = n_trade = n_err = 0
    tick = 0
    with open(BOOK_CSV, "w", newline="") as fb, open(TRADES_CSV, "w", newline="") as ft:
        wb, wt = csv.writer(fb), csv.writer(ft)
        wb.writerow(BOOK_COLS)
        wt.writerow(TRADE_COLS)
        while time.time() < deadline:
            start = time.time()
            try:
                r = session.get(BOOK_URL, timeout=5)
                row = flatten_book(r.json(), int(start * 1000))
                if row:
                    wb.writerow(row)
                    n_book += 1
                if tick % TRADES_EVERY == 0:
                    for t in session.get(TRADES_URL, timeout=5).json()["data"]:
                        if t["tradeId"] not in seen_trades:
                            seen_trades.add(t["tradeId"])
                            wt.writerow([t["tradeId"], int(t["ts"]), t["px"], t["sz"], t["side"]])
                            n_trade += 1
            except (requests.RequestException, ValueError, KeyError, IndexError) as e:
                n_err += 1
                if n_err % 20 == 1:
                    print(f"error #{n_err}: {e}", flush=True)
            if tick % 60 == 0:
                fb.flush()
                ft.flush()
                print(f"{time.strftime('%H:%M:%S')} books {n_book} trades {n_trade} "
                      f"errors {n_err}", flush=True)
            tick += 1
            time.sleep(max(0.0, INTERVAL_S - (time.time() - start)))
    print(f"done: {n_book} snapshots, {n_trade} trades, {n_err} errors")


def load():
    """Read the cached CSVs. Raise if the collector has not been run."""
    if not os.path.exists(BOOK_CSV):
        raise FileNotFoundError(f"no snapshots at {BOOK_CSV}; run: python3 data.py")
    book = pd.read_csv(BOOK_CSV)
    book = book.drop_duplicates("ts_exchange_ms").sort_values("ts_local_ms")
    book = book.reset_index(drop=True)
    trades = pd.read_csv(TRADES_CSV).sort_values("ts_ms").reset_index(drop=True)
    return book, trades


def book_arrays(book):
    """Split a flat snapshot frame into (bid_px, bid_sz, ask_px, ask_sz), each n x LEVELS."""
    pick = lambda side, what: book[[f"{side}_{what}_{i}" for i in range(1, LEVELS + 1)]].to_numpy()
    return pick("bid", "px"), pick("bid", "sz"), pick("ask", "px"), pick("ask", "sz")


if __name__ == "__main__":
    collect(float(sys.argv[1]) if len(sys.argv) > 1 else 45)
