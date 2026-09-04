"""Offline checks on synthetic data: planted regimes recovered, filter is causal, labels stable.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

import regime
from backtest import hmm_exposure_oos, run, vol_target_exposure

rng = np.random.default_rng(3)
N = 6000
IDX = pd.bdate_range("2000-01-03", periods=N)

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


# Planted truth, in percent per day: a calm bull and a violent bear.
TRUE_MU = np.array([0.06, -0.12])
TRUE_SD = np.array([0.8, 2.2])
TRUE_A = np.array([[0.985, 0.015],
                   [0.040, 0.960]])


def simulate(n, gen):
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = gen.choice(2, p=TRUE_A[states[t - 1]])
    x = gen.normal(TRUE_MU[states], TRUE_SD[states])
    return pd.Series(x, index=IDX[:n]), states


ret_pct, true_states = simulate(N, rng)
model = regime.fit(ret_pct, 2, seed=0)
desc = regime.describe(model)
mu, sd, A = model.means_.ravel(), np.sqrt(model.covars_.ravel()), model.transmat_

check(f"planted means recovered (got {mu.round(3)}, want {TRUE_MU})",
      np.allclose(mu, TRUE_MU, atol=0.04))
check(f"planted vols recovered (got {sd.round(3)}, want {TRUE_SD})",
      np.allclose(sd, TRUE_SD, rtol=0.08))
check(f"planted transition diagonal recovered (got {np.diag(A).round(3)})",
      np.allclose(np.diag(A), np.diag(TRUE_A), atol=0.015))

runs = pd.Series(true_states).groupby((pd.Series(true_states).diff() != 0).cumsum()).agg(["first", "size"])
empirical_dur = runs.groupby("first")["size"].mean().to_numpy()
check(f"expected duration 1/(1-p_stay) matches the simulated run lengths "
      f"(got {desc['expected_duration'].round(1).tolist()}, simulated {empirical_dur.round(1)})",
      np.allclose(desc["expected_duration"], empirical_dur, rtol=0.25))
check("stationary distribution sums to one and is positive",
      np.isclose(desc["stationary_prob"].sum(), 1) and (desc["stationary_prob"] > 0).all())

decoded = model.predict(ret_pct.to_numpy().reshape(-1, 1))
check(f"Viterbi path agrees with the planted states ({(decoded == true_states).mean():.1%})",
      (decoded == true_states).mean() > 0.90)

# Label switching: a different seed must land on the same state 0 = low vol.
other = regime.fit(ret_pct, 2, seed=7)
check("state 0 is the low-vol state under every seed",
      model.covars_.ravel()[0] < model.covars_.ravel()[1]
      and other.covars_.ravel()[0] < other.covars_.ravel()[1])
check("two seeds give the same parameters after sorting",
      np.allclose(model.means_, other.means_, atol=0.01)
      and np.allclose(model.transmat_, other.transmat_, atol=0.005))

# Mutation test: garbage after the cut must not touch filtered rows before it,
# and must touch smoothed rows, otherwise the test would pass vacuously.
CUT = 4000
mutated = ret_pct.copy()
mutated.iloc[CUT:] = rng.normal(0, 5, N - CUT)
f_clean, f_mut = regime.filtered(model, ret_pct), regime.filtered(model, mutated)
s_clean, s_mut = regime.smoothed(model, ret_pct), regime.smoothed(model, mutated)
check("filtered probabilities before the cut ignore data after it",
      np.allclose(f_clean[:CUT], f_mut[:CUT]))
check("filtered probabilities after the cut do react",
      not np.allclose(f_clean[CUT:], f_mut[CUT:]))
check("smoothed probabilities before the cut DO change (they read the future)",
      not np.allclose(s_clean[CUT - 50:CUT], s_mut[CUT - 50:CUT]))
check("filtered rows are proper distributions",
      np.allclose(f_clean.sum(axis=1), 1) and (f_clean >= 0).all())
check(f"filtered bear-probability tracks the truth "
      f"({np.corrcoef(f_clean[:, 1], true_states)[0, 1]:.2f} correlation)",
      np.corrcoef(f_clean[:, 1], true_states)[0, 1] > 0.7)

# Rolling refit: the exposure for year y must not depend on data from year y+1.
exp_full, fits = hmm_exposure_oos(ret_pct, 2, first_test_year=2015)
exp_short, _ = hmm_exposure_oos(ret_pct[ret_pct.index.year <= 2017], 2, first_test_year=2015)
common = exp_short.index
check("out-of-sample exposure through 2017 is identical whether or not 2018+ exists",
      np.allclose(exp_full[common], exp_short))
check("one parameter table per refit year", sorted(fits["fit_year"].unique()) ==
      sorted(y for y in ret_pct.index.year.unique() if y >= 2015))
# The truncation test above cannot see a fit that includes the current year (both
# runs would include it), so check the refit parameters directly against a fit
# on strictly prior years.
manual = regime.describe(regime.fit(ret_pct[ret_pct.index.year < 2016], 2))
check("the 2016 refit is fit on returns before 2016 only",
      np.allclose(fits[fits["fit_year"] == 2016].drop(columns="fit_year"), manual))

# Engine.
ret = ret_pct / 100
exp = exp_full
free = run(ret, exp, cost_bps=0.0)
costly = run(ret, exp, cost_bps=20.0)
check("position at t is the exposure decided at t-1",
      np.allclose(free["position"].iloc[1:].to_numpy(), exp.reindex(ret.index).iloc[:-1].fillna(0).to_numpy()))
check("zero-cost net equals gross", np.allclose(free["net"], free["gross"]))
check("costs reduce net", costly["net"].sum() < free["net"].sum())
check("gross does not depend on costs", np.allclose(free["gross"], costly["gross"]))
check("exposure stays in [0, 1]", (exp >= 0).all() and (exp <= 1).all())

vt = vol_target_exposure(ret)
check("vol-target exposure is in [0, 1] and drops when vol spikes",
      (vt.dropna() <= 1).all() and (vt.dropna() >= 0).all()
      and vt[true_states == 1].mean() < vt[true_states == 0].mean())

# With regimes this strong, staying out of the planted bear must beat holding it.
window = free.index.year >= 2015
bh = run(ret, pd.Series(1.0, index=ret.index), cost_bps=0.0)
sr = lambda r: r.mean() / r.std() * np.sqrt(252)
check(f"on planted regimes the HMM rule beats buy-and-hold "
      f"({sr(free['gross'][window]):.2f} vs {sr(bh['gross'][window]):.2f} Sharpe)",
      sr(free["gross"][window]) > sr(bh["gross"][window]))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
