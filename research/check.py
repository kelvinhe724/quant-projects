"""Offline checks of the research platform on synthetic data with a planted momentum effect.

Twelve fake instruments whose drift wanders slowly, so trailing momentum
predicts the next month. The feature store must refuse a peeking feature,
the untouched window must open once, the registry must count a rerun once,
and an Alpha wrapped as a sleeve must send the engine the weights the
research path computed. Exits 1 on any failure.

Run: ../.venv/bin/python3 check.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.engine import Bars, Config, CostModel, run, sharpe
from research.alpha import (Alpha, LockError, Sleeve, Untouched, UntouchedWindowUsed, backtest, positions,
                            walk_forward)
from research.alpha.base import month_ends
from research.features import PRICE, Feature, FeatureStore, PeekError, Raw, cross_section, edgar_counts, macro
from research.models import ModelRegistry, data_hash
from research.registry import Registry

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


rng = np.random.default_rng(11)
N, T = 12, 1500
IDX = pd.bdate_range("2012-01-02", periods=T)
NAMES = [f"S{i:02d}" for i in range(N)]


def planted(persistence=0.995, drift_vol=0.00008, noise=0.011):
    drift = np.zeros((T, N))
    drift[0] = rng.normal(0, 0.0006, N)
    for t in range(1, T):
        drift[t] = persistence * drift[t - 1] + rng.normal(0, drift_vol, N)
    rets = drift + rng.normal(0, noise, (T, N))
    close = pd.DataFrame(50 * np.exp(np.cumsum(rets, axis=0)), index=IDX, columns=NAMES)
    open_ = close.shift(1).fillna(50.0) * np.exp(rng.normal(0, 0.002, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, (T, N))))
    vol = pd.DataFrame(rng.integers(500_000, 3_000_000, (T, N)).astype(float), index=IDX, columns=NAMES)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


frames = planted()
cpi = pd.Series(np.cumsum(rng.normal(0.2, 0.1, 60)), index=pd.date_range("2011-12-01", periods=60, freq="MS"))
fil_dates = rng.choice(IDX[IDX.weekday < 5], 80)
filings = pd.DataFrame({"date": fil_dates, "ticker": rng.choice(NAMES, 80),
                        "text": [" ".join(rng.choice(["risk", "growth", "loss", "uncertain", "cash", "going",
                                                      "concern", "going concern"], 200)) for _ in range(80)]})
raw = Raw(frames, {"CPI": cpi}, filings)
bars = Bars(frames)


class XSMom(Alpha):
    """Long the top third by a momentum feature, short the bottom third, equal weight."""

    def __init__(self, feature="mom_12_1", frac=1 / 3):
        self.feature, self.frac = feature, frac
        self.name = f"XSMom[{feature}]"

    def signal(self, xs):
        s = xs[self.feature].dropna()
        k = max(int(len(s) * self.frac), 1)
        w = pd.Series(0.0, index=xs.index)
        if len(s) < 2 * k:
            return w
        rank = s.rank(method="first")
        w[rank.index] = ((rank > len(s) - k).astype(float) - (rank <= k).astype(float)) / k
        return w


print("FEATURE STORE: LAG ENFORCEMENT\n")
store = FeatureStore(PRICE + macro({"CPI": 30}) + edgar_counts())
panel = store.build(raw, audit=True)
check("starter feature set builds and passes the peek audit",
      panel.shape == (T, len(store.names) * N), f"{panel.shape}")
check("every feature carries an explicit lag", all(isinstance(v, int) and v >= 0 for v in store.lags.values()))

pos = 1000
bumped = raw.perturbed(IDX[pos])
f1 = Feature("ret_1", lambda r: r.frames["close"].pct_change(), 1)
a, b = f1.compute(raw), f1.compute(bumped)
check("lag=1 feature: row t does not see day t", a.iloc[:pos + 1].equals(b.iloc[:pos + 1]))
check("lag=1 feature: row t+1 does see day t", not a.iloc[pos + 1].equals(b.iloc[pos + 1]))
f0 = Feature("ret_0", lambda r: r.frames["close"].pct_change(), 0)
a0, b0 = f0.compute(raw), f0.compute(bumped)
check("lag=0 feature: row t sees day t and row t-1 does not",
      not a0.iloc[pos].equals(b0.iloc[pos]) and a0.iloc[:pos].equals(b0.iloc[:pos]))
f30 = Feature("CPI", lambda r: r.series["CPI"], 30)
a30, b30 = f30.compute(raw), f30.compute(bumped)
first_move = (a30.ne(b30) & ~(a30.isna() & b30.isna())).any(axis=1).idxmax()
check("macro lag=30: a bumped series value first shows 30 sessions later",
      IDX.get_loc(first_move) == pos + 30, f"{IDX.get_loc(first_move) - pos}")

leak = Feature("leak", lambda r: r.frames["close"].pct_change().shift(-1), 1)
try:
    FeatureStore([leak]).audit(raw, IDX[pos])
    check("a feature that reads tomorrow is refused by the audit", False)
except PeekError as e:
    check("a feature that reads tomorrow is refused by the audit", "leak" in str(e), str(e)[:60])
centred = Feature("centred", lambda r: r.frames["close"].rolling(5, center=True).mean(), 1)
try:
    FeatureStore([centred]).audit(raw, IDX[pos])
    check("a centred rolling window is refused by the audit", False)
except PeekError:
    check("a centred rolling window is refused by the audit", True)
try:
    Feature("neg", lambda r: r.frames["close"], -1)
    check("negative lag is rejected", False)
except ValueError:
    check("negative lag is rejected", True)
try:
    Feature("nolag", lambda r: r.frames["close"])
    check("a feature without a declared lag is rejected", False)
except TypeError:
    check("a feature without a declared lag is rejected", True)

print("\nFEATURE STORE: STARTER SETS\n")
xs = cross_section(panel, IDX[-1])
mom = frames["close"].shift(21) / frames["close"].shift(252) - 1
check("mom_12_1 at t equals (close[t-22] / close[t-253] - 1)",
      np.allclose(xs["mom_12_1"].to_numpy(), mom.iloc[-2].to_numpy()))
check("macro feature is broadcast to every instrument", xs["CPI"].nunique() == 1 and xs["CPI"].notna().all())
one = filings.sort_values("date").iloc[0]
d = pd.Timestamp(one["date"])
n_risk = one["text"].split().count("risk") + 0
at = panel.loc[IDX[IDX > d][0], ("edgar_risk", one["ticker"])]
same_day = panel.loc[d, ("edgar_risk", one["ticker"])]
check("edgar word count lands the session after the filing, not on it",
      np.isnan(same_day) and at == n_risk, f"{same_day} then {at} vs {n_risk}")
gc = one["text"].count("going concern")
later = filings[(filings["ticker"] != one["ticker"]) & (filings["date"] > d)]["date"].min()
check("a count is carried past another ticker's later filing",
      panel.loc[IDX[IDX > later][0], ("edgar_risk", one["ticker"])] == n_risk)
check("multi-word term counted as a phrase",
      panel.loc[IDX[IDX > d][0], ("edgar_going_concern", one["ticker"])] == gc)
check("edgar_age is the age as of t - lag, so 0 on the first row that carries the filing",
      panel.loc[IDX[IDX > d][0], ("edgar_age", one["ticker"])] == 0
      and panel.loc[IDX[IDX > d][5], ("edgar_age", one["ticker"])] == 5)


def fake_load(dataset, start, end):
    if dataset == "equities/daily":
        long = pd.concat({f: frames[f].stack() for f in frames}, axis=1).reset_index()
        long.columns = ["date", "ticker"] + list(frames)
        return long
    if dataset == "fred/CPI":
        return pd.DataFrame({"date": cpi.index, "value": cpi.to_numpy()})
    if dataset == "edgar/filings":
        return filings
    raise KeyError(dataset)


lake = Raw.from_lake(fake_load, NAMES, IDX[0], IDX[-1], fred=["CPI"], filings=True)
lake_panel = FeatureStore(PRICE + macro({"CPI": 30}) + edgar_counts()).build(lake)
check("Raw.from_lake on the stubbed loader equals Raw from frames", lake_panel.equals(panel))
fb = Raw.from_bars(bars.upto(IDX[900]))
check("Raw.from_bars cuts at the view's as-of date", fb.calendar[-1] == IDX[900] and len(fb.calendar) == 901)

print("\nUNTOUCHED WINDOW LOCK\n")
tmp = tempfile.mkdtemp()
lock = Untouched(os.path.join(tmp, "untouched.json"))
body = lock.lock(IDX, 0.2, data_hash="abc")
check("lock covers the last fifth of sessions", body["n_sessions"] == 300 and body["end"] == str(IDX[-1].date()))
check("relocking the same window is a no-op", lock.lock(IDX, 0.2, "abc")["hash"] == body["hash"])
try:
    lock.lock(IDX, 0.3, "abc")
    check("locking a different window is refused", False)
except LockError:
    check("locking a different window is refused", True)
try:
    lock.lock(IDX, 0.2, "other-data")
    check("locking the same dates on different data is refused", False)
except LockError:
    check("locking the same dates on different data is refused", True)
calls = []
res = lock.open(lambda s, e: calls.append((s, e)) or {"sharpe": 1.0})
check("first open runs the function once and stores its result", calls == [lock.window] and lock.result == res)
try:
    lock.open(lambda s, e: calls.append("again"))
    check("second open is refused", False)
except UntouchedWindowUsed:
    check("second open is refused", len(calls) == 1)
with open(lock.path) as fh:
    tampered = json.load(fh)
tampered["opened_at"] = None
with open(lock.path, "w") as fh:
    json.dump(tampered, fh)
try:
    lock.open(lambda s, e: calls.append("again"))
    check("nulling opened_at by hand breaks the hash, so the window stays spent", False)
except LockError:
    check("nulling opened_at by hand breaks the hash, so the window stays spent", len(calls) == 1)
tampered["start"] = str(IDX[100].date())
with open(lock.path, "w") as fh:
    json.dump(tampered, fh)
try:
    lock.window
    check("editing the lock file by hand breaks its hash", False)
except LockError:
    check("editing the lock file by hand breaks its hash", True)

print("\nEXPERIMENT REGISTRY\n")
reg = Registry(os.path.join(tmp, "registry"))
alpha = XSMom("mom_12_1")
dates = month_ends(IDX)
pos_a = positions(alpha, panel, dates)
r_a = backtest(pos_a, frames["close"])
e1 = reg.record("XSMom", {"feature": "mom_12_1"}, NAMES, (IDX[0].date(), IDX[-1].date()), r_a)
e2 = reg.record("XSMom", {"feature": "mom_12_1"}, NAMES, (IDX[0].date(), IDX[-1].date()), r_a)
check("same name, config, universe and window twice is one trial", e1["id"] == e2["id"] and e2["n_trials"] == 1)
check("runs() keeps the latest row per key", len(reg.runs()) == 1 and sum(1 for _ in open(reg.file)) == 2)
e3 = reg.record("XSMom", {"feature": "mom_6_1"}, NAMES, (IDX[0].date(), IDX[-1].date()),
                backtest(positions(XSMom("mom_6_1"), panel, dates), frames["close"]))
check("a different config is a new trial", e3["n_trials"] == 2)
e4 = reg.record("XSMom-renamed", {"feature": "mom_12_1"}, NAMES, (IDX[0].date(), IDX[-1].date()), r_a)
check("an identical return stream under another name counts once", e4["n_trials"] == 2 and len(reg.runs()) == 3)
d = reg.dsr(r_a)
check("dsr comes from framework.book.validate's purgedcv call and is a probability",
      0 <= d["dsr"] <= 1 and 0 <= d["psr"] <= 1 and d["n_trials"] == 2, f"psr {d['psr']:.3f} dsr {d['dsr']:.3f}")
check("deflating for trials never raises the probability", d["dsr"] <= d["psr"] + 1e-12)
check("planted momentum: the research path finds it", sharpe(r_a[r_a != 0]) > 0.5, f"{sharpe(r_a[r_a != 0]):.2f}")


class Placebo(Alpha):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def signal(self, xs):
        w = self.rng.choice((-1.0, 0.0, 1.0), len(xs))
        return pd.Series(w / max(np.abs(w).sum(), 1) * 2, index=xs.index)


r_p = backtest(positions(Placebo(), panel, dates), frames["close"])
check("placebo ranks do not", abs(sharpe(r_p[r_p != 0])) < 0.5, f"{sharpe(r_p[r_p != 0]):.2f}")

print("\nWALK-FORWARD\n")
grid = [{"feature": "mom_3_1"}, {"feature": "mom_6_1"}, {"feature": "mom_12_1"}]
table, oos, full = walk_forward(XSMom, grid, panel, frames["close"], n_splits=3, test_size=200, registry=reg,
                                universe=NAMES)
check("three folds, tests tile the end of the panel", len(table) == 3 and table["test_end"].iloc[-1] == IDX[-1].date())
gaps = [(IDX.get_loc(pd.Timestamp(r.test_start)) - IDX.get_loc(pd.Timestamp(r.train_end))) for r in table.itertuples()]
check("train ends at least the label horizon before test starts", min(gaps) >= 21, f"gaps {gaps}")
check("stitched OOS covers every test session once", len(oos) == 600 and oos.index.is_unique)
check("every grid variant is logged; the two whose streams were already logged count once",
      reg.trials() == 3 and len(reg.runs()) == 6, f"{reg.trials()} trials, {len(reg.runs())} keys")
check("walk-forward OOS on planted momentum is positive", sharpe(oos[oos != 0]) > 0, f"{sharpe(oos[oos != 0]):.2f}")

print("\nADAPTER PARITY\n")
price_store = FeatureStore(PRICE)
alpha = XSMom("mom_12_1")
sleeve = Sleeve(alpha, price_store)
res = run(sleeve, bars, config=Config(costs=CostModel()))
research = positions(alpha, price_store.build(raw), dates)
sent = pd.DataFrame(sleeve.targets).T.reindex(columns=NAMES)
common = sent.index.intersection(research.index)
check("sleeve rebalances on every month end the research path does", len(common) == len(dates) == len(sent))
check("Alpha positions == sleeve positions at every rebalance",
      np.allclose(sent.loc[common].to_numpy(), research.loc[common].to_numpy(), atol=1e-12))
held = res.weights.reindex(research.index)
check("engine holds the sent weights the next session, not the same one",
      not np.allclose(held.loc[common].to_numpy(), research.loc[common].to_numpy())
      and np.allclose(res.weights.shift(-1).reindex(common[:-1]).to_numpy(), research.loc[common[:-1]].to_numpy(),
                      atol=0.02))
peek_store = FeatureStore([Feature("mom_12_1", lambda r: r.frames["close"].pct_change(21).shift(-21), 1)])
peek = Sleeve(XSMom("mom_12_1"), peek_store)
run(peek, bars.upto(IDX[400]), config=Config(costs=CostModel()))
peek_research = positions(XSMom("mom_12_1"), peek_store.build(raw), month_ends(IDX[:401]))
peek_sent = pd.DataFrame(peek.targets).T.reindex(columns=NAMES)
check("a peeking feature breaks parity, so the check has teeth",
      not np.allclose(peek_sent.to_numpy(), peek_research.loc[peek_sent.index].to_numpy()))

print("\nMODEL REGISTRY\n")
models = ModelRegistry(os.path.join(tmp, "models"))
train = frames["close"].iloc[:1200]
m1 = models.save("XSMom", {"feature": "mom_12_1"}, train, (IDX[0].date(), IDX[1199].date()), 0.42,
                 meta={"feature": "mom_12_1"})
m2 = models.save("XSMom", {"feature": "mom_12_1"}, train, (IDX[0].date(), IDX[1199].date()), 0.42,
                 meta={"feature": "mom_12_1"})
check("same model on the same data is one entry", m1 == m2 and len(models.entries()) == 1)
bent = train.copy()
bent.iloc[500, 3] += 0.01
m3 = models.save("XSMom", {"feature": "mom_12_1"}, bent, (IDX[0].date(), IDX[1199].date()), 0.42,
                 meta={"feature": "mom_12_1"})
check("one changed training cell changes the data hash and the id", m3 != m1 and data_hash(bent) != data_hash(train))
obj, meta = models.load(m1)
check("load returns the pickled object and its metadata",
      obj == {"feature": "mom_12_1"} and meta["oos_score"] == 0.42 and meta["data_hash"] == data_hash(train))

shutil.rmtree(tmp)
n_fail = checks.count(False)
print(f"\n{len(checks) - n_fail} of {len(checks)} checks passed")
sys.exit(1 if n_fail else 0)
