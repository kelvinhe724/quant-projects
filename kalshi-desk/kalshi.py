"""Thin Kalshi trade API v2 client: RSA-PSS signing, retry, pagination, env switch."""

import base64
import json
import os
import time
import uuid
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")

BASE = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
}
SETTINGS_URL = {
    "demo": "https://demo.kalshi.co/account/profile",
    "prod": "https://kalshi.com/account/profile",
}
RETRIES = 5
TIMEOUT = 30


class NoKey(RuntimeError):
    """Raised on any authenticated call when the key id or key file is missing."""


def load_env(path=ENV_FILE):
    """Read KEY=value lines from .env into a dict, falling back to os.environ."""
    env = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        if k.startswith("KALSHI_") or k in ("LIVE_TRADING", "MAX_ORDER_DOLLARS",
                                            "MAX_OPEN_DOLLARS", "KELLY_FRACTION",
                                            "SHADOW_BANKROLL"):
            env.setdefault(k, v)
    env.setdefault("KALSHI_ENV", "demo")
    env.setdefault("LIVE_TRADING", "false")
    env.setdefault("MAX_ORDER_DOLLARS", "5")
    env.setdefault("MAX_OPEN_DOLLARS", "50")
    env.setdefault("KELLY_FRACTION", "0.25")
    env.setdefault("SHADOW_BANKROLL", "100")
    return env


def sign(private_key, timestamp_ms, method, path):
    """Return the base64 RSA-PSS-SHA256 signature of timestamp + METHOD + path.

    `path` is the full path from the host root with no query string, e.g.
    /trade-api/v2/portfolio/balance. The docs are explicit that query
    parameters are never part of the signed message.
    """
    path = path.split("?", 1)[0]
    msg = f"{timestamp_ms}{method.upper()}{path}".encode()
    sig = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def load_private_key(path):
    with open(path, "rb") as fh:
        return serialization.load_pem_private_key(fh.read(), password=None)


def setup_hint(env):
    """One line saying exactly what to create and where to put it."""
    tag = env.upper()
    return (f"No {env} API key. Create one at {SETTINGS_URL[env]} > API Keys > "
            f"Create New API Key, save the downloaded .txt as keys/kalshi-{env}.key, "
            f"then set KALSHI_{tag}_KEY_ID and KALSHI_{tag}_KEY_PATH in .env")


class Client:
    """One environment, one key. Public endpoints never touch the key."""

    def __init__(self, env=None, settings=None):
        self.settings = settings or load_env()
        self.env = env or self.settings["KALSHI_ENV"]
        if self.env not in BASE:
            raise ValueError(f"KALSHI_ENV must be demo or prod, got {self.env!r}")
        self.base = BASE[self.env]
        self.session = requests.Session()
        self._key = None
        self._key_id = None

    def _load_key(self):
        if self._key is not None:
            return
        tag = self.env.upper()
        key_id = self.settings.get(f"KALSHI_{tag}_KEY_ID", "")
        key_path = self.settings.get(f"KALSHI_{tag}_KEY_PATH", "")
        if not key_id or not key_path:
            raise NoKey(setup_hint(self.env))
        if not os.path.isabs(key_path):
            key_path = os.path.join(HERE, key_path)
        if not os.path.exists(key_path):
            raise NoKey(f"Key file {key_path} not found. " + setup_hint(self.env))
        self._key = load_private_key(key_path)
        self._key_id = key_id

    def has_key(self):
        try:
            self._load_key()
            return True
        except NoKey:
            return False

    def _headers(self, method, url):
        self._load_key()
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sign(self._key, ts, method, urlparse(url).path),
            "Content-Type": "application/json",
        }

    def request(self, method, path, params=None, body=None, auth=False):
        """Send one request. Retries 429 and 5xx with exponential backoff."""
        url = self.base + path
        for attempt in range(RETRIES):
            headers = self._headers(method, url) if auth else {}
            try:
                r = self.session.request(method, url, params=params,
                                         data=json.dumps(body) if body is not None else None,
                                         headers=headers, timeout=TIMEOUT)
            except requests.RequestException as e:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(0.5 * 2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == RETRIES - 1:
                    r.raise_for_status()
                time.sleep(0.5 * 2 ** attempt)
                continue
            if r.status_code == 401:
                raise NoKey(f"Kalshi {self.env} rejected the key (401): {r.text[:200]}. "
                            + setup_hint(self.env))
            if not r.ok:
                raise requests.HTTPError(f"{r.status_code} {method} {path}: {r.text[:300]}")
            return r.json() if r.content else {}

    def get(self, path, auth=False, **params):
        return self.request("GET", path, params=params or None, auth=auth)

    def pages(self, path, key, auth=False, max_pages=10, **params):
        """Yield items across cursor pages."""
        cursor = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            page = self.get(path, auth=auth, **params)
            items = page.get(key, [])
            yield from items
            cursor = page.get("cursor")
            if not cursor or not items:
                break

    # public
    def markets(self, max_pages=5, **params):
        params.setdefault("limit", 1000)
        params.setdefault("mve_filter", "exclude")
        return list(self.pages("/markets", "markets", max_pages=max_pages, **params))

    def market(self, ticker):
        return self.get(f"/markets/{ticker}")["market"]

    def orderbook(self, ticker, depth=10):
        return self.get(f"/markets/{ticker}/orderbook", depth=depth)["orderbook_fp"]

    # authenticated
    def balance(self):
        return self.get("/portfolio/balance", auth=True)

    def positions(self):
        return list(self.pages("/portfolio/positions", "market_positions", auth=True,
                               limit=200, count_filter="position"))

    def fills(self, **params):
        params.setdefault("limit", 200)
        return list(self.pages("/portfolio/fills", "fills", auth=True, **params))

    def orders(self, **params):
        params.setdefault("limit", 200)
        return list(self.pages("/portfolio/orders", "orders", auth=True, **params))

    def place_order(self, ticker, side, price, count, client_order_id=None,
                    expiration_ts=None):
        """Place a GTC limit order. side is bid (buy YES) or ask (buy NO).

        Prices are always YES-leg dollars: buying NO at 0.92 is side=ask at 0.08.
        """
        body = {
            "ticker": ticker,
            "side": side,
            "price": f"{price:.4f}",
            "count": f"{count:.2f}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "client_order_id": client_order_id or str(uuid.uuid4()),
        }
        if expiration_ts:
            body["expiration_time"] = int(expiration_ts)
        return self.request("POST", "/portfolio/events/orders", body=body, auth=True)

    def cancel_order(self, order_id, ticker):
        return self.request("DELETE", f"/portfolio/events/orders/{order_id}",
                            params={"market_ticker": ticker}, auth=True)


def quote(m):
    """Best YES bid/ask from a market listing row, or None if not two-sided."""
    try:
        bid = float(m["yes_bid_dollars"])
        ask = float(m["yes_ask_dollars"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 < bid < ask < 1:
        return None
    return bid, ask


def touch(book):
    """Best YES bid/ask and the size at each from an orderbook_fp payload.

    The book lists bids only. A NO bid at y is a YES ask at 1 - y.
    """
    yes = book.get("yes_dollars") or []
    no = book.get("no_dollars") or []
    if not yes or not no:
        return None
    bid, bid_size = float(yes[-1][0]), float(yes[-1][1])
    ask, ask_size = 1 - float(no[-1][0]), float(no[-1][1])
    if not 0 < bid < ask < 1:
        return None
    return {"bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size}
