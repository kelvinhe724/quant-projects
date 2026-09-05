"""Collectors. Each subcommand pulls one feed into the store and prints one summary line.

    collect.py equities            daily OHLCV, point-in-time S&P universe, incremental
    collect.py fred                the macro set, full history each run
    collect.py kalshi              market listing plus depth on the most traded books
    collect.py edgar [--days N]    8-K full text for universe CIKs, last N filing days
    collect.py archive [--days N]  data.binance.vision klines and futures book depth
    collect.py crypto [--seconds N]  live L2 websocket, 1s top-20 snapshots (runs forever by default)
    collect.py daily / hourly      what the launchd jobs run
"""
import asyncio
import gzip
import html
import io
import json
import logging
import os
import re
import ssl
import sys
import time
import zipfile

import certifi
import pandas as pd
import requests
import websockets
import yfinance as yf

import lake

CACHE = os.path.join(lake.ROOT, "source-material", "data-lake")
UA = {"User-Agent": "Kelvin He kelvinhe724@uchicago.edu data-lake"}
PANEL_START = "2003-06-01"

FRED_SERIES = ["DFF", "SOFR", "DTB3", "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10",
               "DGS30", "T10Y2Y", "T10Y3M", "CPIAUCSL", "CPILFESL", "PAYEMS", "UNRATE", "VIXCLS",
               "BAMLH0A0HYM2", "DFEDTARU"]

KALSHI_DEPTH_TOP = 200
EDGAR_RATE = 0.12
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEPTH = 20
# Binance.com answers 451 from here (see crypto-arbitrage/); Bybit's REST is
# behind a CloudFront block but its websocket is not, and OKX is open both ways.
VENUES = ["bybit", "okx"]


def now():
    return pd.Timestamp.now("UTC").floor("s")


def log(msg):
    print(f"{now().isoformat()} {msg}", flush=True)


# equities

def universe():
    """Every name that was ever in the point-in-time panel, plus SPY."""
    iv = lake.sp500_intervals()
    return sorted(set(iv["ticker"]) | {"SPY"})


def equities(full=False):
    """Pull OHLCV for the universe from the last stored date; the first run backfills from 2003."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    tickers = universe()
    have = lake.parts("equities_daily")
    if full or not have:
        # backfill only names with no part yet, so an interrupted first run resumes
        start = PANEL_START
        tickers = [t for t in tickers if t not in set(have)]
    else:
        last = max(pd.read_parquet(lake.part_path("equities_daily", t), columns=["date"])["date"].max()
                   for t in have if not t.startswith("_"))
        start = (pd.Timestamp(last) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    rows, got = 0, 0
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        # threads=True gives every ticker its own curl handle whose sockets and pipes
        # only go away on garbage collection, ~2 fds a ticker; under launchd's 256 fd
        # cap the run died mid-universe. Sequential is ~100s for the whole panel.
        raw = yf.download(chunk, start=start, auto_adjust=False, progress=False,
                          group_by="column", threads=False)
        if raw.empty:
            continue
        for t in chunk:
            if t not in raw["Close"].columns:
                continue
            df = pd.DataFrame({
                "date": raw.index, "ticker": t,
                "open": raw["Open"][t].to_numpy(), "high": raw["High"][t].to_numpy(),
                "low": raw["Low"][t].to_numpy(), "close": raw["Close"][t].to_numpy(),
                "adj_close": raw["Adj Close"][t].to_numpy(), "volume": raw["Volume"][t].to_numpy(),
            }).dropna(subset=["close"])
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df["collected_at"] = now().tz_localize(None)
            rows += lake.write("equities_daily", t, df)
            got += 1
        log(f"equities {min(i + 100, len(tickers))}/{len(tickers)}")
    log(f"equities: {got} tickers with rows since {start}, {rows} rows in touched parts")
    return got


# fred

def fred():
    """Full history of each series from fredgraph.csv; a revised value replaces the stored one."""
    n = 0
    for s in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), na_values=".")
        df.columns = ["date", "value"]
        df = df.dropna()
        df["date"] = pd.to_datetime(df["date"])
        df["series"] = s
        df["collected_at"] = now().tz_localize(None)
        n += lake.write("fred", s, df[["date", "series", "value", "collected_at"]])
        time.sleep(0.2)
    log(f"fred: {len(FRED_SERIES)} series, {n} rows")
    return n


# kalshi

def kalshi_client():
    sys.path.insert(0, os.path.join(lake.ROOT, "kalshi-desk"))
    import kalshi  # noqa: E402
    return kalshi.Client(env="prod")


def kalshi():
    """Listing of every open market plus full depth on the most traded ones."""
    c = kalshi_client()
    ts = now()
    markets = c.markets(status="open")
    keep = ["ticker", "event_ticker", "market_type", "status", "yes_bid_dollars", "yes_ask_dollars",
            "no_bid_dollars", "no_ask_dollars", "last_price_dollars", "volume_fp", "volume_24h_fp",
            "open_interest_fp", "liquidity_dollars", "open_time", "close_time", "expiration_time"]
    df = pd.DataFrame([{k: m.get(k) for k in keep} for m in markets])
    for k in keep[4:13]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df.insert(0, "ts", ts)
    day = ts.strftime("%Y-%m-%d")
    n = lake.write("kalshi_markets", day, df)
    top = df.sort_values("volume_24h_fp", ascending=False)["ticker"].head(KALSHI_DEPTH_TOP)
    rows = []
    for t in top:
        try:
            book = c.orderbook(t, depth=50)
        except Exception as e:
            log(f"kalshi book {t}: {e}")
            continue
        for side in ("yes", "no"):
            for px, sz in book.get(f"{side}_dollars") or []:
                rows.append((ts, t, side, float(px), float(sz)))
    books = pd.DataFrame(rows, columns=["ts", "ticker", "side", "price", "size"])
    m = lake.write("kalshi_books", day, books, keys=["ts", "ticker", "side", "price"]) if len(books) else 0
    log(f"kalshi: {len(df)} markets, {len(books)} depth rows on {top.size} books; day parts {n}/{m} rows")
    return len(df)


# edgar

def cik_map(tickers, table=None):
    """Ticker -> 10-digit CIK string from SEC's company_tickers.json (fetched if table is None)."""
    if table is None:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60)
        r.raise_for_status()
        table = r.json()
    by = {row["ticker"].upper().replace(".", "-"): f"{int(row['cik_str']):010d}" for row in table.values()}
    return {t: by[t] for t in tickers if t in by}


def form_index(day):
    """8-K rows from the daily form index for `day`, or an empty frame on a non-filing day."""
    d = pd.Timestamp(day)
    q = (d.month - 1) // 3 + 1
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx"
    r = requests.get(url, headers=UA, timeout=60)
    # 404 on a non-filing day; 403 while the day's index has not been published yet
    if r.status_code in (403, 404):
        return pd.DataFrame(columns=["form", "name", "cik", "filed", "path"])
    r.raise_for_status()
    rows = []
    for line in r.text.splitlines():
        if not line.startswith(("8-K ", "8-K/A ")):
            continue
        form, rest = line.split(None, 1)
        parts_ = rest.rsplit(None, 3)
        if len(parts_) != 4:
            continue
        name, cik, filed, path = parts_
        rows.append((form, name.strip(), f"{int(cik):010d}", filed, path))
    return pd.DataFrame(rows, columns=["form", "name", "cik", "filed", "path"])


def strip_html(s):
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"[ \t\xa0]+", " ", re.sub(r"\n\s*\n+", "\n", s)).strip()


def parse_submission(txt):
    """Split a complete submission .txt into (types, text, items) keeping the 8-K body and EX-99 exhibits."""
    types, texts = [], []
    for m in re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", txt, re.S):
        doc = m.group(1)
        typ = re.search(r"<TYPE>([^\n<]+)", doc)
        typ = typ.group(1).strip() if typ else ""
        if not (typ.startswith("8-K") or typ.startswith("EX-99")):
            continue
        body = re.search(r"<TEXT>(.*)", doc, re.S)
        if not body:
            continue
        types.append(typ)
        texts.append(f"[{typ}]\n" + strip_html(body.group(1)))
    text = "\n\n".join(texts)
    items = sorted(set(re.findall(r"(?i)item\s+(\d\.\d\d)", text[:20000])))
    return types, text, items


def edgar(days=3, ciks=None):
    """Full text of every 8-K filed by a universe company over the last `days` filing days."""
    ciks = ciks or cik_map(universe())
    by_cik = {}
    for t, c in ciks.items():
        by_cik.setdefault(c, t)
    total = 0
    for k in range(days):
        day = (pd.Timestamp.now("America/New_York").normalize() - pd.Timedelta(days=k)).tz_localize(None)
        if day.weekday() >= 5:
            continue
        idx = form_index(day)
        idx = idx[idx["cik"].isin(by_cik)]
        rows = []
        for _, f in idx.iterrows():
            url = "https://www.sec.gov/Archives/" + f["path"]
            time.sleep(EDGAR_RATE)
            try:
                r = requests.get(url, headers=UA, timeout=60)
                r.raise_for_status()
            except requests.RequestException as e:
                log(f"edgar {url}: {e}")
                continue
            types, text, items = parse_submission(r.text)
            rows.append({"filed": pd.Timestamp(f["filed"]), "ticker": by_cik[f["cik"]], "cik": f["cik"],
                         "accession": os.path.basename(f["path"])[:-4], "form": f["form"],
                         "items": ",".join(items), "earnings": "2.02" in items,
                         "docs": ",".join(types), "text": text, "url": url,
                         "collected_at": now().tz_localize(None)})
        if rows:
            n = lake.write("edgar_8k", day.strftime("%Y-%m-%d"), pd.DataFrame(rows), keys=["accession"])
            total += len(rows)
            log(f"edgar {day.date()}: {len(rows)} filings fetched, {n} in part")
        else:
            log(f"edgar {day.date()}: no universe 8-Ks in the index ({len(idx)} matched)")
    return total


# binance archives

ARCHIVES = {
    "crypto_klines_1m": ("data/spot/daily/klines/{s}/1m/{s}-1m-{d}.zip",
                         ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
                          "trades", "taker_buy_base", "taker_buy_quote", "ignore"], ["ts", "symbol"]),
    "crypto_bookdepth": ("data/futures/um/daily/bookDepth/{s}/{s}-bookDepth-{d}.zip",
                         ["timestamp", "percentage", "depth", "notional"], ["ts", "symbol", "percentage"]),
}


def archive(days=30):
    """Backfill the last `days` daily zips from data.binance.vision into monthly parts."""
    os.makedirs(CACHE, exist_ok=True)
    total = 0
    for name, (pattern, cols, keys) in ARCHIVES.items():
        for s in CRYPTO_SYMBOLS:
            frames = {}
            for k in range(1, days + 1):
                d = (pd.Timestamp.now("UTC") - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
                rel = pattern.format(s=s, d=d)
                path = os.path.join(CACHE, os.path.basename(rel))
                if not os.path.exists(path):
                    r = requests.get("https://data.binance.vision/" + rel, timeout=120)
                    if r.status_code == 404:
                        continue
                    r.raise_for_status()
                    open(path, "wb").write(r.content)
                with zipfile.ZipFile(path) as z:
                    raw = z.read(z.namelist()[0])
                df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
                if df.iloc[0, 0] == cols[0]:
                    df = df.iloc[1:]
                df.columns = cols
                tcol = cols[0]
                t = pd.to_numeric(df[tcol], errors="coerce")
                if t.notna().all():
                    df["ts"] = pd.to_datetime(t, unit="us" if t.iloc[0] > 1e14 else "ms", utc=True)
                else:
                    df["ts"] = pd.to_datetime(df[tcol], utc=True)
                df["symbol"] = s
                df = df.drop(columns=[tcol, "ignore"], errors="ignore")
                for c in df.columns:
                    if c not in ("ts", "symbol"):
                        df[c] = pd.to_numeric(df[c])
                frames.setdefault(d[:7], []).append(df)
            for month, fs in frames.items():
                total += lake.write(name, month, pd.concat(fs, ignore_index=True), keys=keys)
            log(f"archive {name} {s}: {sum(len(f) for fs in frames.values() for f in fs)} rows over {len(frames)} months")
    return total


# crypto websocket

def apply_delta(book, side, levels):
    """Update one side of a dict book: size zero deletes a price, anything else sets it."""
    for px, sz in levels:
        px, sz = float(px), float(sz)
        if sz == 0:
            book[side].pop(px, None)
        else:
            book[side][px] = sz


def top(book, n=DEPTH):
    """Best n bids (descending) and asks (ascending) as flat columns."""
    bids = sorted(book["bid"].items(), reverse=True)[:n]
    asks = sorted(book["ask"].items())[:n]
    out = {}
    for i in range(n):
        out[f"bid_px_{i + 1}"], out[f"bid_sz_{i + 1}"] = bids[i] if i < len(bids) else (None, None)
        out[f"ask_px_{i + 1}"], out[f"ask_sz_{i + 1}"] = asks[i] if i < len(asks) else (None, None)
    return out


VENUE = {
    "bybit": {"url": "wss://stream.bybit.com/v5/public/spot",
              "sub": lambda syms: {"op": "subscribe", "args": [f"orderbook.50.{s}" for s in syms]},
              "ping": lambda: {"op": "ping"}},
    "okx": {"url": "wss://ws.okx.com:8443/ws/v5/public",
            "sub": lambda syms: {"op": "subscribe", "args": [{"channel": "books", "instId": f"{s[:-4]}-USDT"} for s in syms]},
            "ping": lambda: "ping"},
}


def parse_msg(venue, raw, books):
    """Apply one websocket message to `books`; return the symbol touched or None."""
    if raw == "pong":
        return None
    m = json.loads(raw)
    if venue == "bybit":
        if "topic" not in m:
            return None
        sym = m["data"]["s"]
        if m["type"] == "snapshot":
            books[sym] = {"bid": {}, "ask": {}}
        apply_delta(books[sym], "bid", m["data"]["b"])
        apply_delta(books[sym], "ask", m["data"]["a"])
        return sym
    if "data" not in m:
        return None
    sym = m["arg"]["instId"].replace("-", "")
    for d in m["data"]:
        if m.get("action") == "snapshot":
            books[sym] = {"bid": {}, "ask": {}}
        books.setdefault(sym, {"bid": {}, "ask": {}})
        apply_delta(books[sym], "bid", [(p, s) for p, s, *_ in d["bids"]])
        apply_delta(books[sym], "ask", [(p, s) for p, s, *_ in d["asks"]])
    return sym


def flush(buf):
    """Write buffered snapshot rows into their hour parts."""
    if not buf:
        return 0
    df = pd.DataFrame(buf)
    n = 0
    for (day, hour), g in df.groupby([df["ts"].dt.strftime("%Y-%m-%d"), df["ts"].dt.strftime("%H")]):
        n += lake.write("crypto_l2", f"{day}/{hour}", g)
    buf.clear()
    return n


async def stream(venue, seconds, buf, state):
    """Hold one connection, sample every symbol's top-20 once a second until `seconds` elapse."""
    spec = VENUE[venue]
    ctx = ssl.create_default_context(cafile=certifi.where())
    books = {}
    async with websockets.connect(spec["url"], ssl=ctx, open_timeout=20, ping_interval=None) as ws:
        await ws.send(json.dumps(spec["sub"](CRYPTO_SYMBOLS)))
        state["venue"] = venue
        last_sample = last_flush = last_ping = time.time()
        while seconds is None or time.time() - state["t0"] < seconds:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                parse_msg(venue, raw, books)
            except asyncio.TimeoutError:
                raise ConnectionError("no message for 5s")
            t = time.time()
            if t - last_ping > 20:
                await ws.send(json.dumps(spec["ping"]()) if venue == "bybit" else spec["ping"]())
                last_ping = t
            if int(t) > int(last_sample):
                ts = pd.Timestamp(int(t), unit="s", tz="UTC")
                for sym, b in books.items():
                    if b["bid"] and b["ask"]:
                        buf.append({"ts": ts, "symbol": sym, "venue": venue, **top(b)})
                last_sample = t
            if t - last_flush > 60:
                state["rows"] += len(buf)
                flush(buf)
                last_flush = t


def crypto(seconds=None):
    """Run the L2 stream, reconnecting on any error and falling through the venue list."""
    buf, state = [], {"t0": time.time(), "rows": 0, "venue": None}
    fails = 0
    while seconds is None or time.time() - state["t0"] < seconds:
        venue = VENUES[min(fails // 3, len(VENUES) - 1)]
        try:
            log(f"crypto: connecting {venue}")
            asyncio.run(stream(venue, seconds, buf, state))
            fails = 0
        except Exception as e:
            fails += 1
            log(f"crypto: {venue} dropped ({type(e).__name__}: {str(e)[:120]}); fail {fails}")
            state["rows"] += len(buf)
            flush(buf)
            time.sleep(min(2 ** min(fails, 5), 30))
    state["rows"] += len(buf)
    flush(buf)
    log(f"crypto: {state['rows']} snapshot rows written, last venue {state['venue']}")
    return state


def quality():
    import quality as q
    return q.main()


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "daily"
    opt = {a.lstrip("-"): int(args[i + 1]) for i, a in enumerate(args)
           if a.startswith("--") and i + 1 < len(args) and args[i + 1].isdigit()}
    if cmd == "equities":
        equities(full="--full" in args)
    elif cmd == "fred":
        fred()
    elif cmd == "kalshi":
        kalshi()
    elif cmd == "edgar":
        edgar(days=opt.get("days", 3))
    elif cmd == "archive":
        archive(days=opt.get("days", 30))
    elif cmd == "crypto":
        crypto(seconds=opt.get("seconds"))
    elif cmd == "daily":
        for step in (equities, fred, edgar, archive):
            try:
                step()
            except Exception as e:
                log(f"{step.__name__} failed: {type(e).__name__}: {e}")
        quality()
    elif cmd == "hourly":
        try:
            kalshi()
        except Exception as e:
            log(f"kalshi failed: {type(e).__name__}: {e}")
        quality()
    else:
        print(__doc__)
        sys.exit(2)
