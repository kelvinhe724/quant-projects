"""Offline checks on simulated paths with planted volatility dynamics. Exits 1 on any failure.

Six fake names follow a GJR-GARCH with known parameters (three of them
symmetric), so GARCH must recover alpha and beta, GJR must find the
planted gamma and not find one where there is none, and every forecaster
must beat the naive trailing vol in QLIKE while never reading past its
own date. The straddle cycles, the scoring, the DM test and the timing
rule are checked on planted inputs.

Run: ../.venv/bin/python3 check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vol
from research.features import FeatureStore, Raw
from research.features.store import long_panel

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


OMEGA, ALPHA, BETA, GAMMA = 0.02, 0.05, 0.90, 0.08
rng = np.random.default_rng(5)
T, NAMES = 2600, ["S0", "S1", "S2", "A0", "A1", "A2"]
IDX = pd.bdate_range("2012-01-02", periods=T)


def simulate(gamma):
    z = rng.standard_t(6, T) * np.sqrt(4 / 6)
    eps, sig2 = np.zeros(T), np.zeros(T)
    sig2[0] = OMEGA / (1 - ALPHA - BETA - gamma / 2)
    eps[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, T):
        sig2[t] = OMEGA + (ALPHA + gamma * (eps[t - 1] < 0)) * eps[t - 1] ** 2 + BETA * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
    return eps / 100, np.sqrt(sig2) / 100


paths = {n: simulate(0.0 if n.startswith("S") else GAMMA) for n in NAMES}
ret = pd.DataFrame({n: paths[n][0] for n in NAMES}, index=IDX)
close = 100 * np.exp(ret.cumsum())
high = close * np.exp(np.abs(rng.normal(0, 0.004, close.shape)))
low = close * np.exp(-np.abs(rng.normal(0, 0.004, close.shape)))
series = {"vix": pd.Series(20.0 + rng.normal(0, 1, T), index=IDX),
          "t10y2y": pd.Series(1.0, index=IDX), "dtb3": pd.Series(2.0, index=IDX)}
raw = Raw({"open": close, "high": high, "low": low, "close": close, "volume": close * 0 + 1e6}, series)
store = FeatureStore(vol.features())
panel = store.build(raw, audit=True)
check("feature set passes the peek audit at lag 0", True)
lr = np.log(close).diff()
check("rv_21 at t equals the trailing 21-day vol through t",
      np.isclose(panel["rv_21"]["S0"].iloc[300], np.sqrt(252 * (lr["S0"].iloc[280:301] ** 2).mean())))
check("t10y2y declared at lag 1 is one session stale",
      np.isnan(panel["t10y2y"]["S0"].iloc[0]) and panel["t10y2y"]["S0"].iloc[1] == 1.0)

WARM, REFIT = 500, 250
fc, targets, _ = vol.forecast_all(panel, lr, warmup=WARM, refit=REFIT)
rv = targets[5]
check("forward RV over 5 days is dated the session before the window",
      np.isclose(rv["S0"].iloc[100], np.sqrt(252 * (lr["S0"].iloc[101:106] ** 2).mean())))

from arch import arch_model
fit = arch_model(ret["S0"] * 100, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
check(f"GARCH recovers alpha {ALPHA} (got {fit.params['alpha[1]']:.3f})", abs(fit.params["alpha[1]"] - ALPHA) < 0.03)
check(f"GARCH recovers beta {BETA} (got {fit.params['beta[1]']:.3f})", abs(fit.params["beta[1]"] - BETA) < 0.04)
g_asym = arch_model(ret["A0"] * 100, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off").params["gamma[1]"]
g_sym = arch_model(ret["S0"] * 100, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off").params["gamma[1]"]
check(f"GJR finds the planted gamma {GAMMA} (got {g_asym:.3f})", abs(g_asym - GAMMA) < 0.05)
check(f"GJR finds no gamma on the symmetric path (got {g_sym:.3f})", abs(g_sym) < 0.04)

true_vol = pd.DataFrame({n: paths[n][1] for n in NAMES}, index=IDX)
idx = IDX[WARM:]
for m in ("har", "garch", "gjr", "lgb"):
    a = vol.score(fc[5][m].loc[idx], rv.loc[idx])["qlike"]
    b = vol.score(fc[5]["naive"].loc[idx], rv.loc[idx])["qlike"]
    check(f"{m} beats naive trailing vol in QLIKE at 5 days ({a:.3f} vs {b:.3f})", a < b)
tab = vol.dm_table({m: fc[5][m].loc[idx] for m in vol.MODELS}, rv.loc[idx], 5)
check("DM: garch beats naive on the pooled series at 5%", tab.loc["garch", "naive"] < -1.96, f"t {tab.loc['garch', 'naive']:.2f}")
check("DM table is antisymmetric", np.isclose(tab.loc["garch", "naive"], -tab.loc["naive", "garch"]))
same = vol.dm(pd.Series(np.ones(200)), pd.Series(np.ones(200)), 5)
check("DM on identical losses is nan, not a division error", np.isnan(same["t"]))
true5 = np.sqrt(252 * (true_vol ** 2).rolling(5).mean().shift(-5))
sq = (fc[5]["gjr"].loc[idx] - true5.loc[idx]).abs().mean().mean()
sn = (fc[5]["naive"].loc[idx] - true5.loc[idx]).abs().mean().mean()
check("GJR forecast is closer to the planted 5-day conditional vol than trailing vol", sq < sn, f"{sq:.4f} vs {sn:.4f}")

# Causality: change everything after session p and require every forecast dated before p unchanged.
p = WARM + 2 * REFIT + 1  # right after a refit, so a training label that was not purged would change a forecast dated before p
bumped = raw.perturbed(IDX[p])
for t in range(p + 1, T):
    for k in ("open", "high", "low", "close"):
        bumped.frames[k].iloc[t] = bumped.frames[k].iloc[t] * (1 + 0.01 * np.sin(t))
panel_b = store.build(bumped)
lr_b = np.log(bumped.frames["close"]).diff()
fc_b, _, _ = vol.forecast_all(panel_b, lr_b, warmup=WARM, refit=REFIT)
for m in vol.MODELS:
    a, b = fc[5][m].iloc[:p], fc_b[5][m].iloc[:p]
    check(f"{m}: forecasts before the perturbed session are unchanged", a.equals(b))
check("the perturbation did move forecasts after it", not fc[5]["har"].iloc[p + 5:].equals(fc_b[5]["har"].iloc[p + 5:]))

s = vol.score(rv.loc[idx], rv.loc[idx])
check("a perfect forecast scores zero RMSE and zero QLIKE", s["rmse"] == 0 and abs(s["qlike"]) < 1e-12)
check("score bias has the sign of forecast minus realised", vol.score(rv.loc[idx] * 1.1, rv.loc[idx])["bias"] > 0)

spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, T))), index=IDX)
vix = pd.Series(16.0, index=IDX)
starts = IDX[::21]
pnl = vol.straddle_cycles(spy, vix, starts)
first = pnl.loc[starts[0]:starts[1]]
one = vol.vrp.simulate(pd.DataFrame({"spy": spy, "vix": vix}).loc[starts[0]:starts[1]], horizon=21)["pnl"]
check("straddle cycles: a cycle is one vrp.simulate call with the entry moved off the start session",
      first.iloc[0] == 0 and np.isclose(first.sum(), one.sum()) and np.isclose(first.iloc[1], one.iloc[0] + one.iloc[1]))
check("straddle cycles: nothing before the first start", (pnl.loc[:starts[0]] == 0).all())
flat = vol.straddle_cycles(spy, vix, starts[:1])
check("one start date gives no cycle", (flat == 0).all())
rich = vol.straddle_cycles(spy, vix + 10, starts).sum()
check("a richer implied vol earns more", rich > pnl.sum(), f"{rich:.3f} vs {pnl.sum():.3f}")

X = pd.DataFrame({"vix": [0.20, 0.20, 0.20, 0.20], "f_har": [0.10, 0.15, 0.18, 0.25]},
                 index=pd.MultiIndex.from_product([IDX[:4], ["STRADDLE"]], names=["date", "instrument"]))
timer = vol.Timer("har", 0.5).fit(X, None)
check("Timer threshold is the training-gap quantile", np.isclose(timer.thr, np.quantile([0.10, 0.05, 0.02, -0.05], 0.5)))
on = timer.signal(pd.DataFrame({"vix": [0.20], "f_har": [0.05]}, index=["STRADDLE"]))
off = timer.signal(pd.DataFrame({"vix": [0.20], "f_har": [0.19]}, index=["STRADDLE"]))
nan = timer.signal(pd.DataFrame({"vix": [0.20], "f_har": [np.nan]}, index=["STRADDLE"]))
check("Timer sells when the gap clears the threshold and stands aside otherwise",
      on["STRADDLE"] == 1.0 and off["STRADDLE"] == 0.0 and nan["STRADDLE"] == 0.0)
always = vol.Timer("none").fit(X, None).signal(pd.DataFrame({"vix": [0.20], "f_har": [np.nan]}, index=["STRADDLE"]))
check("Timer 'none' is always on", always["STRADDLE"] == 1.0)

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
raise SystemExit(0 if checks and all(checks) else 1)
