"""Offline checks: planted regimes recovered, the filter and the refits are causal, the overlay reaches the engine.

Run: ../.venv/bin/python3 check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.engine import Config, RiskConfig, Strategy, run, synthetic
import regimes

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


# Planted truth in the model's own observation space: a calm state and a
# violent one, each a 2-d Gaussian over (return in percent, log vol).
TRUE_MU = np.array([[0.05, 2.3], [-0.10, 3.4]])
TRUE_SD = np.array([[0.7, 0.15], [2.0, 0.2]])
TRUE_A = np.array([[0.98, 0.02], [0.05, 0.95]])
rng = np.random.default_rng(3)
N = 5000
IDX = pd.bdate_range("2000-01-03", periods=N)


def simulate(n, gen):
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = gen.choice(2, p=TRUE_A[states[t - 1]])
    x = gen.normal(TRUE_MU[states], TRUE_SD[states])
    return pd.DataFrame(x, index=IDX[:n], columns=["ret", "logvol"]), states


obs, true_states = simulate(N, rng)
px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1, 100) / 100)), index=IDX[:100])
o = regimes.observations(px)
check("observations: return in percent and log of the annualised 21-day vol, first 21 rows dropped",
      len(o) == 79 and np.isclose(o["ret"].iloc[0], 100 * np.log(px.iloc[21] / px.iloc[20]))
      and np.isclose(o["logvol"].iloc[-1], np.log((np.log(px).diff() * 100).iloc[-21:].std() * np.sqrt(252))))
model = regimes.fit(obs, 2)
mu, sd, A = model.means_, np.sqrt(np.array([np.diag(c) for c in model.covars_])), model.transmat_
check(f"planted means recovered ({mu[:, 0].round(3)} vs {TRUE_MU[:, 0]})", np.allclose(mu, TRUE_MU, atol=0.05))
check(f"planted vols recovered ({sd[:, 0].round(3)} vs {TRUE_SD[:, 0]})", np.allclose(sd, TRUE_SD, rtol=0.10))
check(f"planted persistence recovered ({np.diag(A).round(3)} vs {np.diag(TRUE_A)})",
      np.allclose(np.diag(A), np.diag(TRUE_A), atol=0.02))
check("states sorted by return variance for two seeds",
      regimes.fit(obs, 2, seed=1).covars_[0, 0, 0] < regimes.fit(obs, 2, seed=1).covars_[1, 0, 0]
      and model.covars_[0, 0, 0] < model.covars_[1, 0, 0])

p = regimes.filtered(model, obs)
hit = (p.argmax(axis=1) == true_states).mean()
check(f"filtered argmax matches the planted path {hit:.1%} of days", hit > 0.85)
x = obs.to_numpy().copy()
x[3000:] += 5.0
check("filter is causal: rows before a perturbation are unchanged",
      np.array_equal(regimes.filtered(model, x)[:3000], p[:3000]))
check("filter rows sum to one", np.allclose(p.sum(axis=1), 1.0))

probs, models = regimes.walk_forward_probs(obs, 2, first_year=2004)
check("no probability before the first refit", probs.loc[:"2003-12-31"].isna().all().all()
      and probs.loc["2004-01-01":].notna().all().all())
check(f"one model per year from 2004 ({len(models)})", sorted(models) == list(range(2004, IDX[N - 1].year + 1)))
bumped = obs.copy()
bumped.loc["2012-01-01":] += 3.0
probs2, _ = regimes.walk_forward_probs(bumped, 2, first_year=2004)
check("refits are causal: probabilities before 2012 unchanged when 2012+ data changes",
      np.allclose(probs.loc[:"2011-12-31"].fillna(-1), probs2.loc[:"2011-12-31"].fillna(-1), atol=1e-9))

d = regimes.describe(model)
check("model-implied durations are 1/(1-p_stay)", np.allclose(d["duration"], 1 / (1 - np.diag(A))))
rl = regimes.run_lengths(np.array([0, 0, 0, 1, 1, 0, 0, 1]))
check("run lengths: state 0 mean 2.5 sessions, state 1 mean 1.5, share 5/8",
      np.isclose(rl.loc[0, "empirical_duration"], 2.5) and np.isclose(rl.loc[1, "empirical_duration"], 1.5)
      and np.isclose(rl.loc[0, "share"], 5 / 8))

cal = IDX
months = pd.date_range("1999-01-01", "2019-12-01", freq="MS")
payrolls = pd.Series(np.cumsum(rng.normal(150, 50, len(months))), index=months)
slope = pd.Series(rng.normal(1, 0.5, N), index=cal)
vix = pd.Series(rng.normal(20, 5, N), index=cal)
nc = regimes.nowcast(slope, payrolls, vix, cal)
pay2 = payrolls.copy()
pay2.loc["2015-03-01"] += 1e6
nc2 = regimes.nowcast(slope, pay2, vix, cal)
first_seen = nc.index[(nc.fillna(0) != nc2.fillna(0)).to_numpy()][0]
check(f"a payroll print dated 2015-03-01 is first visible {first_seen.date()}, {regimes.PAYROLL_LAG_DAYS} days later",
      first_seen == cal[cal >= pd.Timestamp("2015-03-01") + pd.Timedelta(days=regimes.PAYROLL_LAG_DAYS)][0])
check("nowcast is NaN until the z-score window has enough history", nc.iloc[:regimes.MIN_TRAIN - 1].isna().all()
      and nc.iloc[regimes.MIN_TRAIN:].notna().all())
vix2 = vix.copy()
vix2.iloc[4000:] += 30
check("nowcast before a VIX shock is unchanged by it", nc.iloc[:4000].equals(regimes.nowcast(slope, payrolls, vix2, cal).iloc[:4000]))

ph = pd.Series([0.0, 1.0, 0.5, np.nan], index=cal[:4])
s = regimes.scale(ph)
check(f"scale is 1 - p_high floored at {regimes.MIN_SCALE} and 1 where undefined",
      np.allclose(s, [1.0, regimes.MIN_SCALE, 0.5, 1.0]))
nz = pd.Series([1.0, -1.0, 0.0, 0.0], index=cal[:4])
s = regimes.scale(pd.Series(0.0, index=cal[:4]), nz)
check("a positive nowcast never adds, a negative one shrinks by exp(z)", np.allclose(s, [1.0, np.exp(-1), 1.0, 1.0]))


class Long(Strategy):
    def on_bar(self, asof, bars):
        return {n: 1 / len(bars.instruments) for n in bars.instruments}


bars = synthetic(n_days=600, seed=1)
risk = dict(target_vol=0.10, max_gross=3.0, dd_threshold=0.15, buffer=0.10)
base = run(Long(), bars, config=Config(risk=RiskConfig(**risk)))
ones = pd.Series(1.0, index=bars.calendar)
cfg = Config(risk=RiskConfig(**risk))
same = run(regimes.RegimeOverlay(Long(), cfg.risk, ones), bars, config=cfg)
check("overlay at scale 1 reproduces the bare sleeve exactly", same.returns.equals(base.returns))
half = ones.copy()
half.loc[bars.calendar[300]:] = 0.5
cfg = Config(risk=RiskConfig(**risk))
res = run(regimes.RegimeOverlay(Long(), cfg.risk, half), bars, config=cfg)
g_before = res.weights.abs().sum(axis=1).iloc[100:300].mean() / base.weights.abs().sum(axis=1).iloc[100:300].mean()
g_after = res.weights.abs().sum(axis=1).iloc[320:].mean() / base.weights.abs().sum(axis=1).iloc[320:].mean()
check(f"scale 0.5 from day 300 halves the engine's gross after it ({g_after:.3f}) and not before ({g_before:.3f})",
      abs(g_before - 1) < 0.02 and abs(g_after - 0.5) < 0.05)
check("the overlay traded on the day the scale changed", (res.trades["date"] == bars.calendar[301]).any())
fresh = RiskConfig(**risk)
check("a scale series that does not cover a date leaves the target alone",
      regimes.RegimeOverlay(Long(), fresh, ones.iloc[:10]).on_bar(bars.calendar[50], bars.upto(bars.calendar[50]))
      is not None and fresh.target_vol == risk["target_vol"])

eq = pd.Series([100, 90, 84, 88, 92, 93, 80, 95], index=cal[:8], dtype=float)
rd = regimes.reduced_days(eq)
check("drawdown cut replay: arms at -16%, stays through -8%, clears at -7%, re-arms at -20%",
      rd.tolist() == [False, False, True, True, True, False, True, False])

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
