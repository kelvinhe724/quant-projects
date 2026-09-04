"""Offline checks. No network, no keys. Exits nonzero on the first failure."""

import base64
import os
import sqlite3
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import kalshi
import live
import paper
import scanner
import shadow
from calibration import bias_curve, bin_calibration


def _verify(public_key, sig_b64, message):
    public_key.verify(
        base64.b64decode(sig_b64), message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_signature_matches_docs():
    """The signed message is timestamp + METHOD + path with no query string."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ts = "1700000000000"
    sig = kalshi.sign(key, ts, "get", "/trade-api/v2/portfolio/orders?limit=5")
    _verify(key.public_key(), sig, b"1700000000000GET/trade-api/v2/portfolio/orders")
    try:
        _verify(key.public_key(), sig, b"1700000000000GET/trade-api/v2/portfolio/orders?limit=5")
    except InvalidSignature:
        pass
    else:
        raise AssertionError("query string leaked into the signed message")

    pem = key.private_bytes(serialization.Encoding.PEM,
                            serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    with tempfile.NamedTemporaryFile("wb", suffix=".key", delete=False) as fh:
        fh.write(pem)
    try:
        c = kalshi.Client("demo", {"KALSHI_ENV": "demo",
                                   "KALSHI_DEMO_KEY_ID": "kid-123",
                                   "KALSHI_DEMO_KEY_PATH": fh.name})
        h = c._headers("POST", c.base + "/portfolio/events/orders")
        assert h["KALSHI-ACCESS-KEY"] == "kid-123"
        assert h["KALSHI-ACCESS-TIMESTAMP"].isdigit()
        _verify(key.public_key(), h["KALSHI-ACCESS-SIGNATURE"],
                (h["KALSHI-ACCESS-TIMESTAMP"] + "POST/trade-api/v2/portfolio/events/orders").encode())
        assert "BEGIN" not in repr(h) and "kid-123" in repr(h)
    finally:
        os.unlink(fh.name)


def test_missing_key_is_one_clear_line():
    c = kalshi.Client("demo", {"KALSHI_ENV": "demo"})
    assert not c.has_key()
    try:
        c.balance()
    except kalshi.NoKey as e:
        msg = str(e)
        assert "API Keys" in msg and "KALSHI_DEMO_KEY_ID" in msg and "\n" not in msg, msg
    else:
        raise AssertionError("authenticated call without a key did not raise NoKey")
    c = kalshi.Client("prod", {"KALSHI_PROD_KEY_ID": "x", "KALSHI_PROD_KEY_PATH": "nope.key"})
    try:
        c.balance()
    except kalshi.NoKey as e:
        assert "nope.key" in str(e) and "KALSHI_PROD_KEY_ID" in str(e)
    else:
        raise AssertionError


def test_orderbook_touch():
    book = {"yes_dollars": [["0.0100", "200.00"], ["0.4200", "13.00"]],
            "no_dollars": [["0.0100", "100.00"], ["0.5600", "17.00"]]}
    t = kalshi.touch(book)
    assert abs(t["bid"] - 0.42) < 1e-9 and abs(t["ask"] - 0.44) < 1e-9, t
    assert t["bid_size"] == 13 and t["ask_size"] == 17
    assert kalshi.touch({"yes_dollars": [], "no_dollars": [["0.5", "1"]]}) is None
    assert kalshi.quote({"yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}) is None
    assert kalshi.quote({"yes_bid_dollars": "0.08", "yes_ask_dollars": "0.12"}) == (0.08, 0.12)


def planted_curve():
    """Fit the research bias curve on simulated markets with a planted distortion."""
    rng = np.random.default_rng(1)
    price = rng.uniform(0.01, 0.99, 200_000)
    g = 1.3
    true_p = price ** g / (price ** g + (1 - price) ** g)
    outcome = (rng.uniform(size=len(price)) < true_p).astype(int)
    return bias_curve(bin_calibration(price, outcome))


def test_scanner_scores_planted_bias():
    curve = planted_curve()
    rows = [
        {"ticker": "CHEAP", "bid": 0.09, "ask": 0.11},
        {"ticker": "CHEAP-WIDE", "bid": 0.05, "ask": 0.15},
        {"ticker": "FAV", "bid": 0.89, "ask": 0.91},
        {"ticker": "COIN", "bid": 0.49, "ask": 0.51},
    ]
    out = {r["ticker"]: r for r in scanner.score(rows, curve, k=0.25, bankroll=100,
                                                   max_order=5, min_edge=-1)}
    assert out["CHEAP"]["side"] == "no", out["CHEAP"]
    assert out["FAV"]["side"] == "yes", out["FAV"]
    for r in out.values():
        assert r["edge"] < r["edge_mid"], r
        assert abs((r["edge_mid"] - r["edge"]) - r["spread"] / 2) < 1e-9, r
    assert out["CHEAP"]["edge"] > 0 and out["FAV"]["edge"] > 0
    assert out["CHEAP-WIDE"]["edge"] < out["CHEAP"]["edge"]
    assert abs(out["COIN"]["edge_mid"]) < 0.02
    ranked = scanner.score(rows, curve, min_edge=-1)
    edges = [r["edge"] for r in ranked]
    assert edges == sorted(edges, reverse=True)
    assert all(r["stake"] <= 5 for r in ranked)
    assert all(r["contracts"] * r["cost"] <= r["stake"] + 1e-9 for r in ranked)


def test_caps_enforced():
    ranked = [{"ticker": f"T{i}", "side": "no", "bid": 0.4, "ask": 0.42, "mid": 0.41,
               "p_model": 0.35, "edge": 0.05 - i * 0.001, "f": 0.1, "stake": 40.0,
               "cost": 0.6} for i in range(10)]
    picks = paper.size_under_caps(ranked, open_dollars=38.0, max_order=5, max_open=50, n=10)
    assert picks, "nothing sized"
    assert all(p["dollars"] <= 5 + 1e-9 for p in picks), picks
    assert sum(p["dollars"] for p in picks) <= 50 - 38 + 1e-9
    assert len(picks) == 3, [p["dollars"] for p in picks]
    assert paper.size_under_caps(ranked, 50.0, 5, 50) == []
    assert len(paper.size_under_caps(ranked, 0.0, 5, 50, n=2)) == 2
    assert paper.size_under_caps([{**ranked[0], "cost": 0.9, "stake": 0.5}], 0, 5, 50) == []
    assert paper.order_args({"side": "yes", "bid": 0.4, "ask": 0.42}) == ("bid", 0.42)
    assert paper.order_args({"side": "no", "bid": 0.4, "ask": 0.42}) == ("ask", 0.4)


ORDER_INS = "insert into orders values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
FILL_INS = "insert into fills values (?,?,?,?,?,?,?,?,?)"


def test_ledger_pnl():
    db = paper.open_ledger(":memory:")
    ins = ORDER_INS
    db.execute(ins, ("o1", "c1", "demo", "A", "no", 0.45, 4, 0, 0.55, 0.455, 0.40, 0.04, 0.1, "executed", "t0", "", "demo"))
    db.execute(ins, ("o2", "c2", "demo", "B", "yes", 0.90, 2, 0, 0.90, 0.895, 0.93, 0.03, 0.1, "executed", "t0", "", "demo"))
    db.execute(ins, ("o3", "c3", "demo", "C", "no", 0.40, 1, 0, 0.60, 0.405, 0.35, 0.05, 0.1, "executed", "t0", "", "demo"))
    db.execute(ins, ("o4", "c4", "demo", "D", "no", 0.30, 3, 2, 0.70, 0.305, 0.25, 0.05, 0.1, "resting", "t0", "", "demo"))
    fills = [("f1", "o1", "A", "no", 0.55, 4, 0.05, "t1", "demo"),
             ("f2", "o2", "B", "yes", 0.90, 2, 0.02, "t1", "demo"),
             ("f3", "o3", "C", "no", 0.60, 1, 0.01, "t2", "demo"),
             ("f4", "o4", "D", "no", 0.70, 1, 0.01, "t2", "demo")]
    db.executemany(FILL_INS, fills)
    assert abs(paper.exposure(db) - (0.55 * 4 + 0.9 * 2 + 0.6 * 1 + 0.7 * 1 + 0.7 * 2)) < 1e-9
    db.executemany("insert into settlements values (?,?,?)",
                   [("A", "no", "s1"), ("B", "no", "s2"), ("C", "no", "s3")])
    assert abs(paper.exposure(db) - (0.7 * 1 + 0.7 * 2)) < 1e-9

    bets = paper.settled_bets(db)
    assert len(bets) == 3
    expect = {"A": 0.45 / 0.55, "B": -1.0, "C": 0.40 / 0.60}
    for _, r in bets.iterrows():
        assert abs(r["payoff"] - expect[r["ticker"]]) < 1e-9, r
    s = paper.stats(db)
    assert s["n"] == 3
    mean = np.mean(list(expect.values()))
    assert abs(s["mean_payoff"] - mean) < 1e-9
    se = np.std(list(expect.values()), ddof=1) / np.sqrt(3)
    assert abs(s["t"] - mean / se) < 1e-9
    assert abs(s["pnl"] - (0.45 * 4 - 1.8 + 0.40)) < 1e-9
    assert abs(s["pnl_net"] - (s["pnl"] - 0.08)) < 1e-9
    assert abs(s["staked"] - (2.2 + 1.8 + 0.6)) < 1e-9
    assert 0 <= s["brier"] <= 1
    eq = paper.equity_curve(db)
    assert [round(p["equity"], 6) for p in eq] == [round(x, 6) for x in
                                                   np.cumsum([1.8 - 0.05, -1.8 - 0.02, 0.4 - 0.01])]


BOOK = {"yes_dollars": [["0.0100", "200.00"], ["0.4200", "13.00"]],
        "no_dollars": [["0.0100", "100.00"], ["0.5600", "17.00"]]}  # touch 0.42 / 0.44


def test_shadow_fill_logic():
    t = kalshi.touch(BOOK)
    assert abs(shadow.fill("bid", 0.44, 17, t) - 0.44) < 1e-9  # buy YES at the ask
    assert abs(shadow.fill("bid", 0.45, 5, t) - 0.44) < 1e-9   # limit above the ask fills at the ask
    assert shadow.fill("bid", 0.43, 1, t) is None          # below the ask: rests, no fill
    assert shadow.fill("bid", 0.44, 18, t) is None         # more than the ask size
    assert abs(shadow.fill("ask", 0.42, 13, t) - 0.58) < 1e-9  # buy NO at 1 - bid
    assert abs(shadow.fill("ask", 0.41, 1, t) - 0.58) < 1e-9   # selling YES below the bid: hits it
    assert shadow.fill("ask", 0.43, 1, t) is None          # above the bid: rests
    assert shadow.fill("ask", 0.42, 14, t) is None         # more than the bid size
    assert shadow.fill("bid", 0.44, 1, None) is None


class FakeBook(shadow.Book):
    """Book with the network cut: a fixed orderbook, and no HTTP session."""
    def __init__(self, db, book=BOOK):
        super().__init__(db, {"SHADOW_BANKROLL": "100"})
        self.book = book

    def orderbook(self, ticker, depth=10):
        return self.book


def fake_ranked(*a, **k):
    return [{"ticker": "T-YES", "title": "", "bid": 0.42, "ask": 0.44, "bid_size": 13.0,
             "ask_size": 17.0, "mid": 0.43, "p_model": 0.50, "side": "yes", "cost": 0.44,
             "edge": 0.06, "f": 0.1, "stake": 2.0},
            {"ticker": "T-NO", "title": "", "bid": 0.42, "ask": 0.44, "bid_size": 13.0,
             "ask_size": 17.0, "mid": 0.43, "p_model": 0.36, "side": "no", "cost": 0.58,
             "edge": 0.06, "f": 0.1, "stake": 2.0},
            {"ticker": "T-MOVED", "title": "", "bid": 0.41, "ask": 0.43, "bid_size": 13.0,
             "ask_size": 17.0, "mid": 0.42, "p_model": 0.50, "side": "yes", "cost": 0.43,
             "edge": 0.05, "f": 0.1, "stake": 2.0}]  # scan saw 0.43; the book is now 0.44


def test_shadow_place_settle_no_double_fill():
    real = scanner.run
    scanner.run = fake_ranked
    try:
        db = paper.open_ledger(":memory:")
        settings = {"MAX_ORDER_DOLLARS": "5", "MAX_OPEN_DOLLARS": "50",
                    "KELLY_FRACTION": "0.25", "SHADOW_BANKROLL": "100"}
        logs = []
        placed = paper.place(FakeBook(db), db, 5, settings, log=logs.append, book="shadow")
        assert len(placed) == 3, placed
        fills = {r["ticker"]: dict(r) for r in db.execute("select * from fills")}
        assert set(fills) == {"T-YES", "T-NO"}, fills
        assert abs(fills["T-YES"]["cost"] - 0.44) < 1e-9 and fills["T-YES"]["count"] == 4
        assert abs(fills["T-NO"]["cost"] - 0.58) < 1e-9 and fills["T-NO"]["count"] == 3
        assert all(f["book"] == "shadow" for f in fills.values())
        orders = {r["ticker"]: dict(r) for r in db.execute("select * from orders")}
        assert orders["T-YES"]["status"] == "executed" and orders["T-YES"]["book"] == "shadow"
        assert orders["T-MOVED"]["status"] == "canceled" and "not marketable" in orders["T-MOVED"]["note"]
        assert "0.42x13 / 0.44x17" in orders["T-YES"]["note"], orders["T-YES"]["note"]
        assert abs(paper.exposure(db, "shadow") - (0.44 * 4 + 0.58 * 3)) < 1e-9

        # Re-run: filled tickers are held, the missed one is retried, still one fill each.
        paper.place(FakeBook(db), db, 5, settings, log=logs.append, book="shadow")
        assert db.execute("select count(*) from fills").fetchone()[0] == 2
        assert db.execute("select count(*) from orders where ticker = 'T-YES'").fetchone()[0] == 1
        assert db.execute("select count(*) from orders where ticker = 'T-MOVED'").fetchone()[0] == 2
        # Even a bypassed held-filter cannot double-count: the fill id is per ticker.
        FakeBook(db).place_order("T-YES", "bid", 0.44, 4)
        assert db.execute("select count(*) from fills").fetchone()[0] == 2

        # Settle on the public result, twice; the second pass is a no-op.
        class Pub:
            env = "prod"
            def market(self, ticker):
                return {"result": "yes"} if ticker == "T-YES" else {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42"}
        assert paper.settle(Pub(), db, "shadow") == (1, 1)
        assert paper.settle(Pub(), db, "shadow") == (0, 1)
        assert db.execute("select count(*) from settlements").fetchone()[0] == 1
        s = paper.stats(db, "shadow")
        assert s["n"] == 1 and abs(s["mean_payoff"] - 0.56 / 0.44) < 1e-9, s
        assert abs(s["pnl"] - 0.56 * 4) < 1e-9 and abs(s["staked"] - 0.44 * 4) < 1e-9
        assert abs(paper.exposure(db, "shadow") - 0.58 * 3) < 1e-9
    finally:
        scanner.run = real


def test_book_isolation():
    db = paper.open_ledger(":memory:")
    db.execute(ORDER_INS, ("d1", "c", "demo", "A", "yes", 0.44, 2, 0, 0.44, 0.43, 0.5, 0.06, 0.1, "executed", "t0", "", "demo"))
    db.execute(ORDER_INS, ("s1", None, "prod", "A", "yes", 0.44, 2, 0, 0.44, 0.43, 0.5, 0.06, 0.1, "executed", "t0", "", "shadow"))
    db.execute(ORDER_INS, ("d2", "c", "demo", "B", "no", 0.30, 5, 5, 0.70, 0.31, 0.25, 0.05, 0.1, "resting", "t0", "", "demo"))
    db.executemany(FILL_INS, [("df", "d1", "A", "yes", 0.46, 2, 0.02, "t1", "demo"),
                              ("sf", "s1", "A", "yes", 0.44, 2, 0.0, "t1", "shadow")])
    db.execute("insert into settlements values ('A', 'no', 's1')")
    assert paper.stats(db, "shadow")["n"] == 1 and paper.stats(db, "demo")["n"] == 1
    assert paper.stats(db)["n"] == 2
    assert abs(paper.stats(db, "shadow")["pnl"] + 0.88) < 1e-9
    assert abs(paper.stats(db, "demo")["pnl"] + 0.92) < 1e-9
    assert paper.stats(db, "live")["n"] == 0 and paper.equity_curve(db, "live") == []
    assert paper.exposure(db, "shadow") == 0 and abs(paper.exposure(db, "demo") - 3.5) < 1e-9
    assert len(paper.equity_curve(db, "shadow")) == 1
    # A ledger created before the book column existed gets it on open, rows as demo.
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as fh:
        path = fh.name
    try:
        old = sqlite3.connect(path)
        old.executescript(paper.SCHEMA.replace(", book text default 'demo'", ""))
        old.execute("insert into fills values ('f', 'o', 'A', 'yes', 0.5, 1, 0, 't')")
        old.commit()
        old.close()
        db = paper.open_ledger(path)
        assert [r["book"] for r in db.execute("select book from fills")] == ["demo"]
        db.close()
    finally:
        os.unlink(path)


def test_live_refuses_by_default():
    def boom(_):
        raise AssertionError("live.py prompted while LIVE_TRADING was false")
    assert live.armed({"LIVE_TRADING": "false"}, prompt=boom) is False
    assert live.armed({}, prompt=boom) is False
    assert live.armed({"LIVE_TRADING": "True "}, prompt=lambda _: "no") is False
    assert live.armed({"LIVE_TRADING": "true"}, prompt=lambda _: live.PHRASE) is True

    def eof(_):
        raise EOFError
    assert live.armed({"LIVE_TRADING": "true"}, prompt=eof) is False
    # The shadow book cannot sign even when a prod key is configured.
    b = shadow.Book(paper.open_ledger(":memory:"),
                    {"KALSHI_PROD_KEY_ID": "x", "KALSHI_PROD_KEY_PATH": "nope.key",
                     "SHADOW_BANKROLL": "100"})
    try:
        b._headers("POST", b.base + "/portfolio/events/orders")
    except kalshi.NoKey:
        raise AssertionError("shadow book reached the key loader")
    except RuntimeError:
        pass
    else:
        raise AssertionError("shadow book signed a request")
    assert shadow.Book.place_order is not kalshi.Client.place_order
    env = kalshi.load_env(os.path.join(kalshi.HERE, ".env.example"))
    assert env["LIVE_TRADING"] == "false", "shipped .env.example must default LIVE off"


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)  # one-bet t-stats divide by zero
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as e:
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                sys.exit(1)
            print(f"ok  {name}")
    print("all checks passed")
