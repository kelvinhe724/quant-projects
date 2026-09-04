"""Offline checks on synthetic pairs: filter tracking, convergence, no look-ahead, split.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from kalman import kalman_filter, ols
from strategy import (RULES, build_spread, evaluate, pair_backtest, static_spread,
                      tune_noise)
from backtest import pair_backtest as pairs_project_backtest  # noqa: E402

rng = np.random.default_rng(3)
N = 1500
IDX = pd.bdate_range("2015-01-02", periods=N)
FORMATION = ("2015-01-01", "2018-12-31")
OOS = ("2019-01-01", "2030-12-31")

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def ar1(n, phi, sigma):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sigma)
    return x


x = np.cumsum(rng.normal(0, 0.02, N)) + 4.0
noise = ar1(N, 0.9, 0.01)

# Hedge ratio wanders as a slow random walk, about 0.12 a year.
drift_beta = 0.85 + np.cumsum(rng.normal(0, 0.003, N))
y_drift = drift_beta * x + 0.5 + noise
y_const = 0.85 * x + 0.5 + noise

b_ols, a_ols, r_var = ols(y_drift, x)
kf = kalman_filter(y_drift, x, noise_ratio=1e-3, obs_var=r_var)
rmse_kf = np.sqrt(np.mean((kf.beta.to_numpy()[200:] - drift_beta[200:]) ** 2))
rmse_ols = np.sqrt(np.mean((b_ols - drift_beta[200:]) ** 2))
check(f"drifting hedge ratio: filter RMSE {rmse_kf:.3f} < static OLS RMSE {rmse_ols:.3f}",
      rmse_kf < 0.5 * rmse_ols)

b_c, a_c, r_c = ols(y_const, x)
rls = kalman_filter(y_const, x, noise_ratio=0.0, obs_var=r_c)
check(f"constant hedge ratio, zero state noise: filter ends at OLS "
      f"({rls.beta.iloc[-1]:.5f} vs {b_c:.5f})",
      abs(rls.beta.iloc[-1] - b_c) < 1e-4 and abs(rls.alpha.iloc[-1] - a_c) < 1e-4)
check("with state noise the filter still lands near the true constant beta",
      abs(kalman_filter(y_const, x, 1e-5, r_c).beta.iloc[-500:].mean() - 0.85) < 0.05)

bumped = y_drift.copy()
bumped[700] += 1.0
kf_b = kalman_filter(bumped, x, 1e-3, r_var)
check("mutation at t=700 leaves every filter output before 700 untouched",
      np.allclose(kf.iloc[:700], kf_b.iloc[:700]))
check("and does change the innovation and state at t=700",
      not np.isclose(kf.innovation.iloc[700], kf_b.innovation.iloc[700])
      and not np.isclose(kf.beta.iloc[700], kf_b.beta.iloc[700]))
check("the innovation at t uses the state from t-1, not the updated one",
      np.isclose(kf.innovation.iloc[700],
                 y_drift[700] - kf.beta.iloc[699] * x[700] - kf.alpha.iloc[699]))

log_px = pd.DataFrame({"A": y_drift, "B": x}, index=IDX)
beta_f, int_f, _ = ols(log_px.loc[:FORMATION[1], "A"], log_px.loc[:FORMATION[1], "B"])
pair = pd.DataFrame([{"a": "A", "b": "B", "beta": beta_f, "intercept": int_f}])
row = next(pair.itertuples())

sp, bpath = static_spread(log_px["A"], log_px["B"], beta_f, int_f)
mine = pair_backtest(log_px, "A", "B", sp, bpath, 10.0, **RULES)
theirs = pairs_project_backtest(log_px, row, cost_bps=10.0, **RULES)
check("static path reproduces the pairs project's backtest to the cent",
      np.allclose(mine["gross"], theirs["gross"]) and np.allclose(mine["net"], theirs["net"])
      and np.allclose(mine["turnover"], theirs["turnover"]))

from pairs import positions, zscore  # noqa: E402
target = positions(zscore(sp, RULES["lookback"]), RULES["entry_z"], RULES["exit_z"],
                   RULES["stop_z"], RULES["max_hold"])
check("position at t is the signal from t-1",
      mine["position"].iloc[0] == 0
      and np.allclose(mine["position"].iloc[1:], target.iloc[:-1])
      and (mine["position"] != 0).sum() > 0)

ksp, kbeta = build_spread(log_px, row, FORMATION[1], noise_ratio=1e-4)
kleg = pair_backtest(log_px, "A", "B", ksp, kbeta, 10.0, **RULES)
in_trade = (kleg["position"] != 0) & (kleg["position"].diff() == 0)
check("hedge ratio applied on day t is the estimate from t-1",
      np.allclose(kleg["beta"].iloc[1:], kbeta.iloc[:-1]))
check("a drifting hedge ratio pays turnover while a position is held",
      (kleg.loc[in_trade, "turnover"] > 0).all() and in_trade.sum() > 0)
check("a frozen hedge ratio pays nothing while a position is held",
      np.allclose(mine.loc[(mine["position"] != 0) & (mine["position"].diff() == 0),
                           "turnover"], 0.0))
free = pair_backtest(log_px, "A", "B", ksp, kbeta, 0.0, **RULES)
check("costs reduce net return and never touch gross",
      free["net"].sum() > kleg["net"].sum() and np.allclose(free["gross"], kleg["gross"]))

q_star, table = tune_noise(log_px, pair, FORMATION)
garbage = log_px.copy()
garbage.loc[OOS[0]:] = garbage.loc[OOS[0]:] + rng.normal(0, 0.5, (len(garbage.loc[OOS[0]:]), 2))
q_garbage, table_g = tune_noise(garbage, pair, FORMATION)
check(f"state-noise tuning ignores everything after formation (picked {q_star:g})",
      q_star == q_garbage and np.allclose(table.formation_sharpe, table_g.formation_sharpe))

m, port, legs = evaluate(log_px, pair, OOS, FORMATION[1], q_star)
check("out-of-sample scoring window starts after formation ends",
      port.index.min() > pd.Timestamp(FORMATION[1])
      and all(v.index.min() > pd.Timestamp(FORMATION[1]) for v in legs.values()))
_, port_form, _ = evaluate(log_px, pair, FORMATION, FORMATION[1], q_star)
check("formation and out-of-sample scores come from disjoint dates",
      len(port_form.index.intersection(port.index)) == 0
      and len(port_form) + len(port) == len(log_px))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
