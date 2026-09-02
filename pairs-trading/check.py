"""Offline checks on synthetic data: planted cointegration, no look-ahead, costs.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from backtest import pair_backtest, trades
from pairs import (bonferroni, hedge_ratio, positions, screen, select, spread,
                   test_pair, zscore)

rng = np.random.default_rng(11)
N = 1200
IDX = pd.bdate_range("2015-01-02", periods=N)

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def ar1(n, phi, sigma, gen):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + gen.normal(0, sigma)
    return x


TRUE_BETA = 0.85
common = np.cumsum(rng.normal(0, 0.012, N)) + 4.0
coint_a = TRUE_BETA * common + ar1(N, 0.90, 0.010, rng) + 0.5
indep_b = np.cumsum(rng.normal(0, 0.012, N)) + 4.0

log_px = pd.DataFrame({"COINT_A": coint_a, "SHARED_B": common, "INDEP_C": indep_b},
                      index=IDX)
sectors = pd.Series({"COINT_A": "Test", "SHARED_B": "Test", "INDEP_C": "Test"})

screened = screen(log_px, sectors)
p = screened.set_index(screened["a"] + "/" + screened["b"])["pvalue"]

check(f"planted pair is cointegrated (p={p['COINT_A/SHARED_B']:.4f})",
      p["COINT_A/SHARED_B"] < 0.01)
check(f"two independent walks are not cointegrated (p={p['INDEP_C/SHARED_B']:.3f})",
      p["INDEP_C/SHARED_B"] > 0.05)

# The single-draw check above is not enough: with this seed COINT_A vs INDEP_C
# comes back at p=0.008 purely by chance, which is the multiple-testing problem
# the screener has to survive. Test the false-positive rate instead.
false_p = np.array([
    test_pair(np.cumsum(rng.normal(0, 0.012, 750)),
              np.cumsum(rng.normal(0, 0.012, 750)))[0]
    for _ in range(200)
])
rate = (false_p < 0.05).mean()
check(f"false-positive rate on independent walks is near 5% (got {rate:.1%})",
      0.01 < rate < 0.12)
check(f"Bonferroni kills almost all of them ({(false_p < bonferroni(200)).sum()} left)",
      (false_p < bonferroni(200)).sum() <= 1)

beta, intercept = hedge_ratio(log_px["COINT_A"], log_px["SHARED_B"])
check(f"hedge ratio recovers the planted {TRUE_BETA} (got {beta:.3f})",
      abs(beta - TRUE_BETA) < 0.05)

kept = select(screened, pvalue_max=0.01)
check("select keeps only the planted pair", len(kept) == 1
      and {kept.a[0], kept.b[0]} == {"COINT_A", "SHARED_B"})
check("select drops a name already used", len(select(screened, pvalue_max=1.0)) == 1)
check("bonferroni threshold shrinks with test count",
      bonferroni(1600) < bonferroni(10) < 0.05)

sp = spread(log_px["COINT_A"], log_px["SHARED_B"], beta, intercept)
z = zscore(sp, 60)

spiked = sp.copy()
spiked.iloc[900] += 5.0
z_spiked = zscore(spiked, 60)
check("z-score is trailing: a spike at t=900 leaves t<900 untouched",
      np.allclose(z.iloc[:900].dropna(), z_spiked.iloc[:900].dropna()))
check("the spike does move the z-score from t=900 onward",
      not np.isclose(z.iloc[900], z_spiked.iloc[900]))
check("z-score has no value before the lookback fills",
      z.iloc[:59].isna().all() and not np.isnan(z.iloc[59]))

pos = positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0, max_hold=60)
entries = z[(pos != 0) & (pos.shift(1).fillna(0) == 0)]
check("every entry happens beyond the entry threshold", (entries.abs() > 2.0).all())
check("position is short the spread when z is high",
      (pos[entries.index][entries > 0] == -1).all())
check("holding period never exceeds max_hold",
      trades(pd.DataFrame({"position": pos, "net": 0.0}))["days"].max() <= 60)

pair = pd.DataFrame([{"a": "COINT_A", "b": "SHARED_B", "beta": beta,
                      "intercept": intercept}]).itertuples().__next__()
rules = dict(lookback=60, entry_z=2.0, exit_z=0.5, stop_z=4.0, max_hold=60)
free = pair_backtest(log_px, pair, cost_bps=0.0, **rules)
costly = pair_backtest(log_px, pair, cost_bps=20.0, **rules)

check("position at t is the signal from t-1",
      (free["position"] != 0).sum() > 0 and free["position"].iloc[0] == 0)
check("gross return is identical whatever the cost assumption",
      np.allclose(free["gross"], costly["gross"]))
check(f"costs reduce the net return ({free['net'].sum():.4f} -> {costly['net'].sum():.4f})",
      costly["net"].sum() < free["net"].sum())
check("zero-cost net equals gross", np.allclose(free["net"], free["gross"]))
check("turnover only moves on position changes",
      np.allclose(costly["turnover"], costly["position"].diff().abs().fillna(
          abs(costly["position"].iloc[0]))))
check("the planted mean-reverting spread is profitable before costs",
      free["gross"].sum() > 0)

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
