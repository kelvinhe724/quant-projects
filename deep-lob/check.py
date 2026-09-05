"""Offline checks on planted data: bar building, labels, the fill lag, the fee, the purge, and three models on a planted signal.

Run: ../.venv/bin/python3 check.py   (about a minute, CPU)
"""
import numpy as np
import pandas as pd

import data
import lob

rng = np.random.default_rng(3)
checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def planted_frame(n, slope_bps=3.0, noise_bps=0.3, seed=3):
    """n one-second bars in the archive schema where the next second's drift is slope x flow imbalance."""
    r = np.random.default_rng(seed)
    imb = np.zeros(n)
    for t in range(1, n):
        imb[t] = 0.7 * imb[t - 1] + r.normal(0, 0.4)
    imb = np.clip(imb, -0.95, 0.95)
    ret = np.zeros(n)
    ret[1:] = slope_bps * imb[:-1] / 1e4 + r.normal(0, noise_bps / 1e4, n - 1)
    mid = 80_000 * np.exp(np.cumsum(ret))
    vol = r.exponential(0.2, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC")
    return pd.DataFrame({"mid": mid, "ret_bps": np.r_[0, np.diff(np.log(mid))] * 1e4,
                         "buy_vol": (1 + imb) * vol / 2, "sell_vol": (1 - imb) * vol / 2,
                         "n_trades": r.poisson(10, n).astype(float), "vwap_bps": r.normal(0, 0.1, n),
                         "bid_bps": -0.01 + r.normal(0, 0.01, n), "ask_bps": 0.01 + r.normal(0, 0.01, n),
                         "flow_imb": imb}, index=idx)


print("BARS FROM TRADES\n")

ts = pd.Timestamp("2026-09-03 00:00:00", tz="UTC")
trades = pd.DataFrame({
    "ts": [ts + pd.Timedelta(milliseconds=ms) for ms in (100, 600, 900, 2200, 2300)],
    "price": [100.0, 100.2, 100.1, 100.5, 100.4], "qty": [1.0, 2.0, 1.0, 0.5, 0.5],
    "is_buyer_maker": [True, False, True, False, True]})
bars = data.seconds_from_trades(trades)
check("one row per second across the span, including the empty second", len(bars) == 3)
check("mid is the last print of the second", np.allclose(bars["mid"], [100.1, 100.1, 100.4]))
check("the empty second carries the previous price and zero flow",
      bars["mid"].iloc[1] == 100.1 and bars["buy_vol"].iloc[1] == 0 and bars["n_trades"].iloc[1] == 0)
check("taker buys and sells are split by is_buyer_maker",
      bars["buy_vol"].iloc[0] == 2.0 and bars["sell_vol"].iloc[0] == 2.0 and bars["buy_vol"].iloc[2] == 0.5)
check("the ask proxy is the last taker-buy print and the bid proxy the last taker-sell print",
      np.isclose(bars["ask_bps"].iloc[0], (100.2 / 100.1 - 1) * 1e4)
      and np.isclose(bars["bid_bps"].iloc[2], 0.0) and np.isclose(bars["ask_bps"].iloc[2], (100.5 / 100.4 - 1) * 1e4))
check("flow imbalance is the orderbook project's formula on buy vs sell volume",
      np.isclose(bars["flow_imb"].iloc[0], 0.0) and np.isclose(bars["flow_imb"].iloc[2], 0.0)
      and np.isclose(bars["vwap_bps"].iloc[0], ((100 + 200.4 + 100.1) / 4 / 100.1 - 1) * 1e4))

lvl = {f"{s}_{w}_{i}": [v] for i in range(1, 21) for s, w, v in
       (("bid", "px", 100 - 0.1 * i), ("bid", "sz", float(i)), ("ask", "px", 100 + 0.1 * i), ("ask", "sz", 2.0 * i))}
l2 = pd.DataFrame({"ts": [ts + pd.Timedelta(milliseconds=400)], "symbol": ["BTCUSDT"], "venue": ["bybit"], **lvl})
l2 = pd.concat([l2, l2.assign(ts=ts + pd.Timedelta(seconds=2, milliseconds=100), bid_px_1=99.95)], ignore_index=True)
lk = data.seconds_from_lake(l2)
check("lake rows floor to the second, fill the gap and keep 80 level columns plus mid",
      len(lk) == 3 and lk.shape[1] == 81 and lk["mid"].iloc[1] == lk["mid"].iloc[0])
check("lake mid is the touch midpoint and level prices are offsets from it in bps",
      np.isclose(lk["mid"].iloc[0], 100.0) and np.isclose(lk["bid_px_3"].iloc[0], -30.0)
      and np.isclose(lk["ask_px_1"].iloc[0], 10.0) and lk["ask_sz_4"].iloc[0] == 8.0)
check("book imbalance over five levels matches the orderbook project's function",
      np.isclose(lob.imbalance_feature(lk).iloc[0], (15 - 30) / 45))

print("\nLABELS AND FEATURES\n")

mid = pd.Series([100.0, 100.0, 100.02, 99.97, 100.0, 100.0], index=pd.date_range("2026-01-01", periods=6, freq="s"))
y, move = lob.labels(mid, 2, band=1.0)
check("label is +1 above the band, -1 below, 0 inside, nan past the end",
      list(y.iloc[:4]) == [1.0, -1.0, -1.0, 1.0] and y.iloc[4:].isna().all())
check("the band scales with the square root of the horizon, fixed in advance",
      np.isclose(lob.BAND[5], 0.5 * np.sqrt(5)) and np.isclose(lob.BAND[30], 0.5 * np.sqrt(30)))
X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 2}, index=pd.date_range("2026-01-01", periods=10, freq="s"))
ls = lob.lag_stack(X, (0, 3))
check("lag stack holds each feature at each lag under a flat column name",
      list(ls.columns) == ["a_l0", "b_l0", "a_l3", "b_l3"] and ls["a_l3"].iloc[5] == 2.0 and np.isnan(ls["b_l3"].iloc[2]))
fi = lob.imbalance_feature(pd.DataFrame({"buy_vol": [1.0, 3.0, 0.0], "sell_vol": [1.0, 1.0, 0.0]}), depth=2)
check("flow imbalance is the last-`depth`-seconds sum through the shared formula",
      np.allclose(fi, [0.0, 1 / 3, 0.5]))

print("\nEXECUTION AND FEES\n")

idx = pd.date_range("2026-01-01", periods=8, freq="s", tz="UTC")
mid = pd.Series([100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0], index=idx)
pos = pd.Series([1.0, 0, 0, 0, 0, 0, 0, 0], index=idx)
r = lob.pnl(pos, mid)
check("a position decided at t fills at bar t+1 and earns the move from t+1 to t+2",
      np.isclose(r.iloc[2], 0.01 - 1e-4) and np.isclose(r.iloc[3], -1e-4) and np.isclose(r.drop(r.index[[2, 3]]).sum(), 0))
jump = pd.Series([100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0], index=idx)
check("a jump between bar t and t+1 is not earned by a position decided at t",
      np.isclose(lob.pnl(pos, jump).sum(), -2e-4))
check("the fee is 1 bp per unit traded, charged twice on a round trip",
      np.isclose(lob.pnl(pos, jump, fee_bps=2.0).sum(), -4e-4))
p = pd.DataFrame({-1: [0.1, 0.6, 0.3, 0.2], 0: [0.4, 0.3, 0.4, 0.6], 1: [0.5, 0.1, 0.3, 0.2]}, index=idx[:4])
check("positions follow P(up) - P(down) against tau, flat inside it",
      list(lob.positions(p, 0.3)) == [1.0, -1.0, 0.0, 0.0] and list(lob.positions(p, 0.5)) == [0.0, 0.0, 0.0, 0.0])
s = lob.summarise(r, pos)
check("summary counts one trade per position change and reports total bps",
      s["trades"] == 2 and np.isclose(s["total_bps"], (0.01 - 2e-4) * 1e4) and np.isclose(s["in_market"], 1 / 8))
cl = lob.classification(p, pd.Series([1.0, -1.0, 0.0, 1.0], index=idx[:4]))
check("classification accuracy, macro F1 and the majority share are on the same rows",
      np.isclose(cl["accuracy"], 0.75) and cl["n"] == 4 and np.isclose(cl["majority"], 0.5))

print("\nWALK-FORWARD PURGE\n")

cal = pd.date_range("2026-01-01", periods=3000, freq="s", tz="UTC")
fl = lob.folds(cal, 3, 500)
starts = [cal[te[0]] for _, te in fl]
check("three disjoint test windows of 500 seconds in time order",
      all(len(te) == 500 for _, te in fl) and starts == sorted(starts)
      and all(cal[fl[k][1][-1]] < cal[fl[k + 1][1][0]] for k in range(2)))
check("no training row's 30-second label reaches a test-window price",
      all(cal[tr[-1]] + pd.Timedelta(seconds=30) < cal[te[0]] for tr, te in fl))
check("training rows all precede the test window and the window expands",
      all(cal[tr[-1]] < cal[te[0]] for tr, te in fl) and len(fl[0][0]) < len(fl[1][0]) < len(fl[2][0]))

print("\nPLANTED SIGNAL\n")

frame = planted_frame(9000)
Y = lob.label_frame(frame["mid"])
n_tr = 6000
tr, va, te = frame.iloc[:n_tr - 600], frame.iloc[n_tr - 600:n_tr], frame.index[n_tr:]
Ytr, Yva = Y.iloc[:n_tr - 600], Y.iloc[n_tr - 600:n_tr]
cpu = "cpu"
models = {"logit-imbalance": lob.ImbalanceLogit(), "lightgbm": lob.LightGBM(n_estimators=100),
          "deeplob": lob.DeepLOB(window=20, epochs=3, batch=256, dev=cpu)}
acc = {}
for name, m in models.items():
    m.fit(tr, Ytr, val=(va, Yva))
    p = lob.proba_on(m, frame, te[0], te[-1])
    acc[name] = {h: lob.classification(p[h], Y[h].loc[te]) for h in lob.HORIZONS}
    check(f"{name} scores exactly the test rows, none before or after",
          all(v.index[0] == te[0] and v.index[-1] == te[-1] and len(v) == len(te) for v in p.values()))
for name in models:
    a = acc[name][5]
    check(f"{name} beats the majority class on the planted 5-second signal ({a['accuracy']:.2f} vs {a['majority']:.2f})",
          a["accuracy"] > a["majority"] + 0.10)
check("deeplob's validation loss improved from its first epoch",
      models["deeplob"].history[-1]["val_loss"] < models["deeplob"].history[0]["val_loss"] + 1e-9
      or len(models["deeplob"].history) == 1)
p5 = lob.proba_on(models["logit-imbalance"], frame, te[0], te[-1])[5]
pos = lob.positions(p5, 0.1)
r = lob.pnl(pos, frame["mid"].loc[te])
check(f"the planted edge survives the 1 bp fee at a 1-second fill ({lob.summarise(r, pos)['bps_per_trade']:.2f} bps a trade)",
      r.sum() > 0)
shuffled = Y.copy()
shuffled.loc[tr.index] = rng.permutation(Ytr.to_numpy())
null = lob.ImbalanceLogit().fit(tr, shuffled.loc[tr.index])
pn = lob.proba_on(null, frame, te[0], te[-1])[5]
a = lob.classification(pn, Y[5].loc[te])
check(f"shuffled labels leave the logistic at the majority class ({a['accuracy']:.2f} vs {a['majority']:.2f})",
      abs(a["accuracy"] - a["majority"]) < 0.05)
d = lob.DeepLOB(window=20, epochs=1, dev=cpu).fit(tr, Ytr)
first = d.proba(frame.iloc[:50])[5]
check("deeplob has no prediction before its window is full and one after",
      np.isnan(first[:19]).all() and np.isfinite(first[19:]).all() and np.allclose(first[19:].sum(axis=1), 1.0))
pk = d.cpu()
check("the CPU copy predicts the same probabilities", np.allclose(pk.proba(frame.iloc[:50])[5][19:], first[19:], atol=1e-5))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
