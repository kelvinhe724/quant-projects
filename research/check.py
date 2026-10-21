"""Offline checks of the research platform on synthetic data with a planted momentum effect.

Twelve fake instruments whose drift wanders slowly, so trailing momentum
predicts the next month. The feature store must refuse a peeking feature,
the untouched window must open once, the registry must count a rerun once,
and an Alpha wrapped as a sleeve must send the engine the weights the
research path computed. Every harness evaluation must log itself (or
refuse to run), an edited rule must be a new key, and the pre-registration
checker must fail a planted short the broker refuses. Exits 1 on any failure.

Run: ../.venv/bin/python3 check.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.engine import Bars, Config, CostModel, run, sharpe
from research import prereg_check
from research.alpha import (Alpha, LockError, Sleeve, UnloggedEvaluation, Untouched, UntouchedWindowUsed, backtest,
                            positions, walk_forward, window_returns)
from research.alpha.base import month_ends
from research.features import PRICE, Feature, FeatureStore, PeekError, Raw, cross_section, edgar_counts, macro
from research.models import ModelRegistry, data_hash
from research.registry import Registry, code_hash, use

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
try:
    Untouched(os.path.join(tmp, "unpinned.json")).lock(IDX, 0.2, data_hash="abc")
    check("a new lock with no registry attached is refused", False)
except UnloggedEvaluation:
    check("a new lock with no registry attached is refused", not os.path.exists(os.path.join(tmp, "unpinned.json")))
lock_reg = Registry(os.path.join(tmp, "lock-registry"))
body = lock.lock(IDX, 0.2, data_hash="abc", registry=lock_reg)
check("the lock records the registry's real path and a hash of its runs.jsonl (none yet)",
      body["registry"] == os.path.realpath(lock_reg.path) and body["registry_sha256_at_lock"] is None
      and body["registry_bytes_at_lock"] == 0)
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
try:
    lock.open(lambda s, e: {"sharpe": 1.0})
    check("an open with no registry attached is refused and leaves the window shut", False)
except UnloggedEvaluation:
    check("an open with no registry attached is refused and leaves the window shut", not lock.opened)
try:
    lock.open(lambda s, e: {"sharpe": 1.0}, registry=lock_reg)
    check("an open on an empty registry is refused (it would freeze n_trials 0)", False)
except LockError:
    check("an open on an empty registry is refused (it would freeze n_trials 0)", not lock.opened)
PRE = IDX[IDX < lock.window[0]]
for i in (1, 2):
    lock_reg.record(f"trial{i}", {}, NAMES, (PRE[0], PRE[-1]), frames["close"].iloc[:, i].pct_change().loc[PRE])
other = Registry(os.path.join(tmp, "throwaway"))
other.record("trial1", {}, NAMES, (PRE[0], PRE[-1]), frames["close"].iloc[:, 1].pct_change().loc[PRE])
try:
    lock.open(lambda s, e: {"sharpe": 1.0}, registry=other)
    check("an open with a registry other than the one the lock names is refused", False)
except LockError:
    check("an open with a registry other than the one the lock names is refused", not lock.opened)
use(other.path)
try:
    lock.open(lambda s, e: {"sharpe": 1.0})
    check("... and so is one attached with use()", False)
except LockError:
    check("... and so is one attached with use()", not lock.opened)
use(None)
peek = Registry(os.path.join(tmp, "peek"))
plock = Untouched(os.path.join(tmp, "peek.json"))
plock.lock(IDX, 0.2, data_hash="abc", registry=peek)
peek.record("in sample", {}, NAMES, (PRE[0], PRE[-1]), frames["close"].iloc[:, 1].pct_change().loc[PRE])
peek.record("full panel", {}, NAMES, (IDX[0], IDX[-1]), frames["close"].iloc[:, 2].pct_change())
try:
    plock.open(lambda s, e: {"sharpe": 1.0}, registry=peek)
    check("an open is refused when a registry row already covers part of the window", False)
except LockError as e:
    check("an open is refused when a registry row already covers part of the window",
          not plock.opened and "full panel" in str(e), str(e)[-70:])
rw = Registry(os.path.join(tmp, "rewritten"))
rw.record("in sample", {}, NAMES, (PRE[0], PRE[-1]), frames["close"].iloc[:, 1].pct_change().loc[PRE])
rlock = Untouched(os.path.join(tmp, "rewritten.json"))
rlock.lock(IDX, 0.2, data_hash="abc", registry=rw)
with open(rw.file) as fh:
    kept = fh.read()
with open(rw.file, "w") as fh:
    fh.write(kept.replace('"in sample"', '"renamed"'))
try:
    rlock.open(lambda s, e: {"sharpe": 1.0}, registry=rw)
    check("an open is refused when runs.jsonl no longer starts with what it held at the lock", False)
except LockError:
    check("an open is refused when runs.jsonl no longer starts with what it held at the lock", not rlock.opened)
legacy = Untouched(os.path.join(tmp, "legacy.json"))
lb = {"start": body["start"], "end": body["end"], "n_sessions": body["n_sessions"], "data_hash": "abc",
      "opened_at": None, "result": None}
legacy.write(lb)
try:
    legacy.open(lambda s, e: {"sharpe": 1.0}, registry=Registry(os.path.join(tmp, "legacy-empty")))
    check("a lock written before registries were recorded still refuses an empty registry", False)
except LockError:
    check("a lock written before registries were recorded still refuses an empty registry", not legacy.opened)
legacy.open(lambda s, e: {"sharpe": 1.0}, registry=lock_reg)
check("... and records the registry it is opened with, the count and a hash of runs.jsonl at the open",
      legacy.read()["registry"] == os.path.realpath(lock_reg.path) and legacy.read()["n_trials"] == 2
      and len(legacy.read()["registry_sha256_at_open"]) == 64)
calls = []


def the_read(s, e):
    calls.append((s, e))
    lock_reg.record("the read", {}, NAMES, (s, e), frames["close"].iloc[:, 3].pct_change().loc[s:e])
    return {"sharpe": 1.0}


res = lock.open(the_read, registry=lock_reg)
check("first open runs the function once and stores its result", calls == [lock.window] and lock.result == res)
check("the open freezes the trial count before the read (2 of the 3 logged) inside the lock and its hash",
      lock.read()["n_trials"] == 2 and lock_reg.trials() == 3, f"{lock.read()['n_trials']} at the open")
d_frozen = lock_reg.dsr(frames["close"].iloc[:, 3].pct_change().loc[lock.window[0]:], n_trials=lock.read()["n_trials"])
check("dsr(n_trials=...) deflates by the count the lock froze, not the count after the read logged itself",
      d_frozen["n_trials"] == lock.read()["n_trials"] == 2)
with open(lock.path) as fh:
    edited = json.load(fh)
edited["n_trials"] = 1
with open(lock.path, "w") as fh:
    json.dump(edited, fh)
try:
    lock.read()
    check("lowering the stored trial count by hand breaks the hash", False)
except LockError:
    check("lowering the stored trial count by hand breaks the hash", True)
edited["n_trials"] = 2
with open(lock.path, "w") as fh:
    json.dump(edited, fh)
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
r_a = backtest(pos_a, frames["close"], log=False)  # recorded by hand below: the registry's own rules are under test
e1 = reg.record("XSMom", {"feature": "mom_12_1"}, NAMES, (IDX[0].date(), IDX[-1].date()), r_a)
e2 = reg.record("XSMom", {"feature": "mom_12_1"}, NAMES, (IDX[0].date(), IDX[-1].date()), r_a)
check("same name, config, universe and window twice is one trial", e1["id"] == e2["id"] and e2["n_trials"] == 1)
check("runs() keeps the latest row per key", len(reg.runs()) == 1 and sum(1 for _ in open(reg.file)) == 2)
e3 = reg.record("XSMom", {"feature": "mom_6_1"}, NAMES, (IDX[0].date(), IDX[-1].date()),
                backtest(positions(XSMom("mom_6_1"), panel, dates), frames["close"], log=False))
check("a different config is a new trial", e3["n_trials"] == 2)
e4 = reg.record("XSMom-renamed", {"feature": "mom_12_1"}, NAMES, (IDX[0].date(), IDX[-1].date()), r_a)
check("an identical return stream under another name counts once", e4["n_trials"] == 2 and len(reg.runs()) == 3)
d = reg.dsr(r_a)
check("dsr comes from framework.book.validate's purgedcv call and is a probability",
      0 <= d["dsr"] <= 1 and 0 <= d["psr"] <= 1 and d["n_trials"] == 2, f"psr {d['psr']:.3f} dsr {d['dsr']:.3f}")
check("deflating for trials never raises the probability", d["dsr"] <= d["psr"] + 1e-12)
flat = pd.Series(0.001, index=IDX)
e5 = reg.record("cash", {}, NAMES, (IDX[0].date(), IDX[-1].date()), flat)
d5 = reg.dsr(r_a)
check("a constant stream (float std about 1e-19) gets a NaN Sharpe: counted as a trial, left out of the variance",
      np.isnan(e5["sharpe_daily"]) and d5["n_trials"] == 3 and d5["n_nan_sharpe"] == 1
      and abs(d5["var_sharpe"] - d["var_sharpe"]) < 1e-15, f"var {d5['var_sharpe']:.2e}")
check("planted momentum: the research path finds it", sharpe(r_a[r_a != 0]) > 0.5, f"{sharpe(r_a[r_a != 0]):.2f}")


class Placebo(Alpha):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def signal(self, xs):
        w = self.rng.choice((-1.0, 0.0, 1.0), len(xs))
        return pd.Series(w / max(np.abs(w).sum(), 1) * 2, index=xs.index)


r_p = backtest(positions(Placebo(), panel, dates), frames["close"], log=False)
check("placebo ranks do not", abs(sharpe(r_p[r_p != 0])) < 0.5, f"{sharpe(r_p[r_p != 0]):.2f}")

print("\nWALK-FORWARD\n")
grid = [{"feature": "mom_3_1"}, {"feature": "mom_6_1"}, {"feature": "mom_12_1"}]
table, oos, full = walk_forward(XSMom, grid, panel, frames["close"], n_splits=3, test_size=200, registry=reg,
                                universe=NAMES)
check("three folds, tests tile the end of the panel", len(table) == 3 and table["test_end"].iloc[-1] == IDX[-1].date())
gaps = [(IDX.get_loc(pd.Timestamp(r.test_start)) - IDX.get_loc(pd.Timestamp(r.train_end))) for r in table.itertuples()]
check("train ends at least the label horizon before test starts", min(gaps) >= 21, f"gaps {gaps}")
check("stitched OOS covers every test session once", len(oos) == 600 and oos.index.is_unique)
stages = reg.runs()["tags"].map(lambda t: t.get("stage")).value_counts()
check("every evaluation is logged (3 variants x 3 training folds, 3 test windows, 3 whole-window variants); "
      "the two whole-window streams already logged count once",
      (stages.get("walk_forward_train"), stages.get("walk_forward_test"), stages.get("walk_forward_grid")) == (9, 3, 3)
      and len(reg.runs()) == 4 + 15, f"{len(reg.runs())} keys")
check("fold rows (walk_forward_train, walk_forward_test) stay on disk but are not counted: 2 + 1 new whole-window "
      "variant + the cash row", reg.trials() == 4 and reg.dsr(r_a)["n_trials"] == 4, f"{reg.trials()} trials")
check("an empty window with a registry attached returns an empty stream and logs nothing",
      window_returns(alpha, panel, frames["close"], IDX[-1] + pd.Timedelta(days=3), IDX[-1] + pd.Timedelta(days=30),
                     dates, registry=reg).empty and len(reg.runs()) == 19)
check("walk-forward OOS on planted momentum is positive", sharpe(oos[oos != 0]) > 0, f"{sharpe(oos[oos != 0]):.2f}")

print("\nTHE HARNESS LOGS EVERY EVALUATION\n")
refused = []
for attempt in (lambda: backtest(pos_a, frames["close"]),
                lambda: window_returns(alpha, panel, frames["close"], IDX[1200], IDX[-1], dates),
                lambda: walk_forward(XSMom, grid, panel, frames["close"], n_splits=3, test_size=200)):
    try:
        attempt()
        refused.append(False)
    except UnloggedEvaluation:
        refused.append(True)
check("with no registry attached, backtest, window_returns and walk_forward refuse to run", refused == [True] * 3)
loose = []
for flag in (None, 0, ""):
    try:
        backtest(pos_a, frames["close"], log=flag)
        loose.append(False)
    except UnloggedEvaluation:
        loose.append(True)
check("only the literal log=False runs unlogged (None, 0 and '' still raise), and it computes the same stream",
      loose == [True] * 3 and backtest(pos_a, frames["close"], log=False).equals(r_a))
default = use(os.path.join(tmp, "default"))
backtest(pos_a, frames["close"], alpha=alpha, config={"feature": "mom_12_1"})
backtest(pos_a, frames["close"], log=False)
use(None)
check("research.registry.use(path) attaches a default that every evaluation logs to, log=False included",
      sorted(default.runs()["name"]) == ["XSMom[mom_12_1]", "backtest"])
RULE = '''import pandas as pd

from research.alpha import Alpha


class Rule(Alpha):
    name = "Rule"

    def signal(self, xs):
        return pd.Series(SIGN1.0 / len(xs), index=xs.index)
'''


def load_rule(tag, source):
    path = os.path.join(tmp, f"rule_{tag}.py")
    with open(path, "w") as fh:
        fh.write(source)
    spec = importlib.util.spec_from_file_location(f"rule_{tag}", path)
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # inspect finds a class through its module
    spec.loader.exec_module(mod)
    return mod


rules = [load_rule(t, RULE.replace("SIGN", s)).Rule() for t, s in (("a", ""), ("b", ""), ("c", "-"))]
win = (IDX[0], IDX[-1])
keys = [Registry.key("Rule", {}, NAMES, win, code_hash(x)) for x in rules]
dyn_mod = load_rule("d", "x = 1\n")
Dyn = type("Dyn", (Alpha,), {"__module__": "rule_d", "signal": lambda self, xs: pd.Series(0.0, index=xs.index)})
before_edit = code_hash(Dyn)
with open(dyn_mod.__file__, "a") as fh:
    fh.write("x = 2\n")
check("an edited rule is a new key, the same source in another file is the same key, and a class inspect cannot "
      "read is keyed on its module file's bytes", keys[0] == keys[1] != keys[2] and code_hash(Dyn) != before_edit,
      f"{keys[0]} {keys[1]} {keys[2]}")
hand = Registry(os.path.join(tmp, "hand"))
ids = {hand.record("Rule", {"lookback": "frozen"}, NAMES, win, frames["close"].iloc[:, i].pct_change(), code="same")["id"]
       for i in (1, 2, 1)}
check("a hand-recorded row whose stream changed (the rule edited in a module no hash sees) is a new row, "
      "the same stream again is the same row", len(ids) == 2 and len(hand.runs()) == 2 and hand.trials() == 2)
stored = reg.stream(e1["id"])
runs = reg.runs()
grid_ids = runs.loc[runs["tags"].map(lambda t: t.get("stage")) == "walk_forward_grid", "id"]
check("each logged return stream round-trips from returns/<id>.parquet, so the DSR can be recomputed from disk",
      stored.equals(r_a) and reg.dsr(stored) == reg.dsr(r_a)
      and all(reg.stream(i).equals(full[json.dumps(g)]) for i, g in zip(grid_ids, grid)))

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

print("\nPRE-REGISTRATION CHECKER\n")
PLANTED = """# Pre-registration: FXE-SHORT (planted)

## 2. Universe and instrument
- Instruments actually traded: FXE (Invesco CurrencyShares Euro Trust) shares, on Alpaca.
- The sleeve goes short FXE when the euro carry turns negative.

## 2a. Window provenance
- ACK

## 2b. Power
- Forward track 252 sessions at a target net Sharpe of 1.0; a no-edge strategy passes 15.9% of the time.
  If it passes anyway, Kelvin's judgment decides, named now.

## 5. The test
- **Prior layer:** backtest 2010-01-04 to 2022-12-30; untouched window 2023-01-03 to 2024-12-31, locked first.
"""
FXE = {"name": "Invesco CurrencyShares Euro Trust", "asset_class": "us_equity", "tradable": True, "shortable": False,
       "marginable": True, "easy_to_borrow": False}
real_asset, prereg_check.asset = prereg_check.asset, lambda sym: FXE if sym == "FXE" else None  # offline
try:
    ok, lines = prereg_check.tradability(PLANTED)
finally:
    prereg_check.asset = real_asset
check("a planted pre-registration that shorts FXE fails tradability on Alpaca's shortable=False",
      not ok and any("FXE" in s and "will not short" in s for s in lines), lines[0])
repo = os.path.join(tmp, "repo")
os.makedirs(os.path.join(repo, "fx"))
with open(os.path.join(repo, "fx", "README.md"), "w") as fh:
    fh.write("# FX carry\n\nDaily FXE closes, 2015-01-02 to 2024-12-31: Sharpe 0.41.\n")
acked = prereg_check.provenance(PLANTED.replace("ACK", "fx/README.md published 2015-01-02 to 2024-12-31 on FXE; "
                                                       "this window is a second look"), repo)
silent = prereg_check.provenance(PLANTED.replace("ACK", "none"), repo)
check("the same window hit passes provenance when section 2a names it and fails when it does not",
      acked[0] and not silent[0] and any("fx/README.md" in s for s in acked[1]), acked[1][-1])
ok, lines = prereg_check.power(PLANTED)
check("power: no edge clears Sharpe 1.0 over 252 sessions 15.9% of the time and over 126 sessions 24.0% "
      "(1 - N(bar x sqrt(N/252))); a 2b with a stated rate and a decision rule passes",
      ok and abs(prereg_check.no_edge_pass_rate(252, 1.0) - 0.158655) < 1e-6
      and abs(prereg_check.no_edge_pass_rate(126, 1.0) - 0.239750) < 1e-6, lines[1][:70])

shutil.rmtree(tmp)
n_fail = checks.count(False)
print(f"\n{len(checks) - n_fail} of {len(checks)} checks passed")
sys.exit(1 if n_fail else 0)
