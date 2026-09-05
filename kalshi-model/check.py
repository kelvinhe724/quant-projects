"""Offline checks on planted markets. No network. Exits 1 on the first failure.

The markets are simulated with a Prelec distortion between the quoted mid
and the true probability, so the answer is known: a calibrator has to
recover the distortion, a calibrated null must not turn into an edge, a
bar dated after the quote must not move a feature, and the whole run.py
pipeline, untouched window included, has to go through on the planted
table into a temporary directory before it is ever pointed at real data.

Run: ../.venv/bin/python3 check.py   (about three minutes; the pipeline check trains the whole grid twice on 2,000 markets)
"""
import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import candles  # noqa: E402
import model as km  # noqa: E402
import run  # noqa: E402

warnings.filterwarnings("ignore")
PASSED = 0


def ok(cond, msg):
    global PASSED
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASSED += 1
    print(f"ok   {msg}")


def prelec(p, a=0.65):
    """Prelec weighting: what a market quotes for a true probability p. Overweights longshots."""
    return np.exp(-(-np.log(p)) ** a)


def prelec_inv(w, a=0.65):
    return np.exp(-(-np.log(w)) ** (1 / a))


def synthetic(n, seed=0, distort=True):
    """A feature table shaped like candles.build(): mids, spreads, clocks, path features and known outcomes."""
    rng = np.random.default_rng(seed)
    mid = np.clip(rng.beta(0.8, 1.4, n), 0.02, 0.98)
    spread = rng.choice([0.01, 0.02, 0.03, 0.05, 0.10, 0.30], n, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.1])
    bid = np.clip(mid - spread / 2, 0.01, 0.98)
    ask = np.clip(mid + spread / 2, bid + 0.01, 0.99)
    p_true = prelec_inv(mid) if distort else mid
    outcome = (rng.random(n) < p_true).astype(int)
    hours = np.sort(rng.uniform(0, 60 * 24, n))
    close = pd.Timestamp("2026-06-01", tz="UTC") + pd.to_timedelta(hours, unit="h")
    age = rng.uniform(0, 2, n)
    cats = rng.choice(candles.CATEGORIES, n)
    df = pd.DataFrame({
        "logit_mid": np.log(mid / (1 - mid)), "spread": spread, "log_spread_rel": np.log(spread / np.minimum(mid, 1 - mid)),
        "quote_age_h": age, "hours_to_close": 24 + age, "log_duration": rng.normal(4, 1, n),
        "log_vol_96": rng.normal(5, 2, n), "log_vol_24": rng.normal(3, 2, n), "log_oi": rng.normal(6, 2, n),
        "n_quoted": rng.integers(1, 96, n), "n_traded": rng.integers(0, 50, n),
        "mid_chg_24": rng.normal(0, 0.05, n), "mid_chg_96": rng.normal(0, 0.08, n),
        "mid_vol": np.abs(rng.normal(0, 0.02, n)), "mid_range": np.abs(rng.normal(0, 0.1, n)),
        "last_gap": rng.normal(0, 0.02, n), "no_trade": (rng.random(n) < 0.1).astype(float),
    })
    for c in candles.CATEGORIES:
        df[f"cat_{c}"] = (cats == c).astype(float)
    df["ticker"] = [f"SYN-{i}" for i in range(n)]
    df["series"] = "SYN"
    df["category"] = cats
    df["close_time"] = close
    df["quote_time"] = close - pd.Timedelta(hours=24)
    df["bid"], df["ask"], df["price"], df["outcome"] = bid, ask, mid, outcome
    df["settle_volume"] = np.exp(rng.normal(5, 2, n)) + 50 * outcome
    df["p_true"] = p_true
    return df


def check_scoring():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    expect = -np.mean(np.log([0.8, 0.8, 0.6, 0.6]))
    ok(abs(km.log_loss(y, p) - expect) < 1e-12, "log-loss matches the hand value")
    ok(abs(km.fee(0.5) - 0.0175) < 1e-12 and abs(km.fee(0.1) - 0.0063) < 1e-12, "taker fee is 0.07 x c x (1 - c)")


def check_trade_rule():
    df = pd.DataFrame({"ticker": ["a", "b", "c"], "price": [0.30, 0.30, 0.30], "spread": [0.03, 0.03, 0.03],
                       "outcome": [1, 1, 0], "close_time": pd.to_datetime(["2026-07-01"] * 3, utc=True)})
    p = np.array([0.36, 0.32, 0.20])  # edges at the mid: +0.06 (yes), +0.02 (yes), +0.10 (no)
    b = km.bets(df, p, charge_fee=False)
    ok(list(b["ticker"]) == ["a", "c"], "edge-over-spread keeps only edges wider than the quoted spread")
    ok(b.loc[b["ticker"] == "a", "side"].item() == "yes" and b.loc[b["ticker"] == "c", "side"].item() == "no",
       "the side follows the sign of the edge")
    a = b[b["ticker"] == "a"].iloc[0]
    ok(abs(a["cost"] - 0.315) < 1e-12 and abs(a["payoff"] - (1 - 0.315) / 0.315) < 1e-12,
       "YES is bought at the ask and a win pays (1 - cost) / cost")
    c = b[b["ticker"] == "c"].iloc[0]
    ok(abs(c["cost"] - (1 - 0.285)) < 1e-12 and c["payoff"] == (1 - c["cost"]) / c["cost"],
       "NO is bought at one minus the bid")
    bf = km.bets(df, p)
    cost = 0.315 + km.fee(0.315)
    ok(abs(bf[bf["ticker"] == "a"]["cost"].item() - cost) < 1e-12, "the fee is added to the cost when charged")
    k = km.bets(df, p, charge_fee=False, rule="kelly")
    ok(len(k) == 3, "the research project's Kelly rule trades every positive-f market")
    m = km.bets(df, p, spread=0.0, charge_fee=False)
    ok(abs(m[m["ticker"] == "a"]["cost"].item() - 0.30) < 1e-12, "spread=0 trades at the mid")


def check_leak_guard():
    bars = pd.DataFrame({"ticker": "x", "end_ts": [3600 * i for i in range(1, 6)],
                         "bid": [0.3, 0.31, 0.32, 0.33, 0.34], "ask": [0.32, 0.33, 0.34, 0.35, 0.36],
                         "last": [0.31, np.nan, 0.33, np.nan, 0.35], "volume": [10, 0, 20, 0, 30],
                         "oi": [100, 100, 120, 120, 150]})
    end = 3600 * 3
    a = candles.path_features(bars, end)
    future = pd.concat([bars, pd.DataFrame([{"ticker": "x", "end_ts": end + 1, "bid": 0.9, "ask": 0.95,
                                              "last": 0.9, "volume": 1e6, "oi": 1e6}])])
    b = candles.path_features(future, end)
    ok(a == b, "a bar dated after the quote changes no feature")
    ok(a["n_quoted"] == 3 and abs(a["log_vol_96"] - np.log1p(30)) < 1e-12 and abs(a["log_oi"] - np.log1p(120)) < 1e-12,
       "volume and open interest are read only through the quote")
    ok(a["quote_age_h"] == 0.0 and abs(a["mid_chg_96"] - 0.02) < 1e-12, "quote age and path change are right")
    e = candles.path_features(bars.iloc[:0], end)
    ok(np.isnan(e["quote_age_h"]) and e["no_trade"] == 1.0 and e["n_quoted"] == 0, "no bars gives the empty row")
    row = candles.market_features(0.25, 0.03, 24.5, 100.0, "Sports", {**a, "quote_age_h": 0.5})
    ok(set(row) == set(candles.FEATURES), "market_features produces exactly the declared feature set")
    ok(row["cat_Sports"] == 1.0 and sum(row[f"cat_{c}"] for c in candles.CATEGORIES) == 1.0, "category one-hot")


def check_folds(df):
    n = 0
    for tr, te in run.folds(df):
        train, test = df.loc[tr], df.loc[te]
        ok(train["close_time"].max() < test["quote_time"].min(),
           f"fold {n + 1}: every training market closed before the first test quote")
        ok(len(train) > 0 and len(test) > 0 and test.index.min() > train.index.max(), f"fold {n + 1}: train precedes test")
        n += 1
    ok(n == run.FOLDS, f"{run.FOLDS} folds")
    # a market closing at the exact second of the first test quote must be purged
    d = df.copy()
    tr, te = next(iter(run.folds(d)))
    first_q = d.loc[te, "quote_time"].min()
    d.loc[tr[-1], "close_time"] = first_q
    tr2, _ = next(iter(run.folds(d)))
    ok(tr[-1] not in tr2, "a training market closing at the first test quote's second is purged")
    fit, val = run.val_split(df.iloc[:2000])
    ok(val["quote_time"].min() > fit["close_time"].max() and len(val) > 0, "val_split is purged by the horizon")


def check_recovers_distortion():
    tr, te = synthetic(8000, seed=1), synthetic(4000, seed=2)
    market = km.log_loss(te["outcome"], te["price"])
    truth = km.log_loss(te["outcome"], te["p_true"])
    for name in ("logit+platt", "hgb+isotonic", "mlp+none", "bias"):
        a = km.make(name).fit(tr, tr["outcome"])
        p = a.predict(te)
        ll = km.log_loss(te["outcome"], p)
        mae = float(np.mean(np.abs(p - te["p_true"])))
        ok(ll < market and ll - truth < 0.6 * (market - truth),
           f"{name}: recovers most of the planted distortion (log-loss {ll:.4f}, market {market:.4f}, truth {truth:.4f})")
        ok(mae < 0.04, f"{name}: mean abs error to the true probability {mae:.3f}")
    a = km.make("logit+none").fit(tr, tr["outcome"])
    w = a.signal(te.iloc[:50])
    b = km.bets(te.iloc[:50], a.predict(te.iloc[:50]))
    ok((w != 0).sum() == len(b) and all((w[b.index] > 0) == (b["side"] == "yes")), "signal() is the signed Kelly weight")


def check_null():
    """Calibrated markets: no variant may manufacture an edge that survives the spread."""
    ts = []
    for seed in range(4):
        tr, te = synthetic(6000, seed=10 + seed, distort=False), synthetic(4000, seed=20 + seed, distort=False)
        a = km.make("logit+platt").fit(tr, tr["outcome"])
        s, _ = km.score(te, a.predict(te))
        ts.append(s["t"] if np.isfinite(s["t"]) else 0.0)
        ok(s["n_bets"] < 0.5 * len(te), f"null seed {seed}: the rule trades {s['n_bets']} of {len(te)} calibrated markets")
    ok(max(ts) < 3.0 and np.mean(ts) < 2.0, f"null: no edge after the spread (t-stats {np.round(ts, 2).tolist()})")


def check_pipeline():
    df = synthetic(2000, seed=5)
    tmp = tempfile.mkdtemp()
    try:
        s = run.main(df.drop(columns="p_true"), reports=tmp, categories={"SYN": "Sports"})
        ok(os.path.exists(os.path.join(tmp, "summary.json")), "pipeline writes summary.json")
        u = s["untouched"]
        ok(u["opened_at"] is not None and u["n_test"] + u["n_train"] < len(df), "the untouched window was opened once")
        ok(s["chosen"] in run.GRID and set(u["variants"]) >= {s["chosen"], "bias", "market", run.BASELINE},
           "the window was read for the chosen variant, the baseline, the bin curve and the mid")
        h = u["variants"][s["chosen"]][run.HEADLINE]
        ok("dsr" in h and h["dsr"]["n_trials"] == s["n_trials"], "the headline DSR uses every trial logged in the run")
        ok(u["variants"][s["chosen"]][run.HEADLINE]["log_loss"] < u["variants"]["market"][run.HEADLINE]["log_loss"],
           "on planted distortion the chosen model beats the mid on the untouched window")
        ok(s["walk_forward"]["scores"][run.LEAK]["log_loss"] < s["walk_forward"]["scores"][s["chosen"]]["log_loss"],
           "the planted settlement-volume leak scores better than any honest variant, so a leak would be visible")
        s2 = run.main(df.drop(columns="p_true"), reports=tmp, categories={"SYN": "Sports"})
        same = json.dumps(s2["untouched"]["variants"], sort_keys=True) == json.dumps(u["variants"], sort_keys=True)
        ok(same and s2["n_trials"] == s["n_trials"],
           "a rerun reads the window back from the lock and logs no new trial")
        m, e = km.registered("chosen", tmp)
        ok(e["meta"]["variant"] == s["chosen"] and len(m.predict(df.iloc[:5])) == 5, "the chosen model loads from the registry")
        mb, eb = km.registered("baseline", tmp)
        ok(eb["meta"]["variant"] == run.BASELINE, "the baseline model loads from the registry")
        for f in ("calibration.png", "equity.png", "walk_forward.png", "untouched_predictions.csv"):
            ok(os.path.exists(os.path.join(tmp, f)), f"{f} written")
    finally:
        shutil.rmtree(tmp)


def check_desk_hook():
    df = synthetic(500, seed=7)
    a = km.make("logit+none").fit(df, df["outcome"])
    a.categories = {"SYN": "Sports"}
    bars = [{"ticker": "SYN-1", "end_ts": 3600 * i, "bid": 0.3, "ask": 0.33, "last": 0.31, "volume": 5.0, "oi": 50.0}
            for i in range(1, 10)]
    orig = candles.fetch_candles
    candles.fetch_candles = lambda *a, **k: bars
    try:
        row = {"ticker": "SYN-1-X", "bid": 0.30, "ask": 0.33, "hours": 30.0, "close_time": "2026-07-02T00:00:00Z",
               "open_time": "2026-06-01T00:00:00Z"}
        f = km.live_features(row, a, now=3600 * 10)
        ok(set(f) == set(candles.FEATURES) | {"price", "spread"}, "live_features builds the training feature row")
        ok(f["cat_Sports"] == 1.0 and f["hours_to_close"] == 30.0 and abs(f["price"] - 0.315) < 1e-12,
           "live row carries the listing's clock, category and mid")
        p = a.predict(pd.DataFrame([f]))[0]
        ok(0 < p < 1, "the registered model scores a live row")
    finally:
        candles.fetch_candles = orig
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kalshi-desk"))
    os.environ.pop("SCANNER_MODEL", None)
    import scanner
    c = scanner.load_curve()
    ok(not getattr(c, "needs_row", False) and 0 < float(c(0.3)) < 1, "the desk's default curve is still the bin curve")


if __name__ == "__main__":
    check_scoring()
    check_trade_rule()
    check_leak_guard()
    check_folds(synthetic(6000, seed=3))
    check_recovers_distortion()
    check_null()
    check_desk_hook()
    check_pipeline()
    print(f"{PASSED} checks passed")
