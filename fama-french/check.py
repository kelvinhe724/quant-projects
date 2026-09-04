"""Offline checks: synthetic assets built from known loadings and a known alpha."""
import numpy as np
import pandas as pd

from factors import blend, fit_panel, grs, performance, premia, regress

rng = np.random.default_rng(7)
T = 600
idx = pd.period_range("1975-01", periods=T, freq="M")

factors = pd.DataFrame({
    "Mkt-RF": rng.normal(0.6, 4.5, T),
    "SMB": rng.normal(0.2, 3.0, T),
    "HML": rng.normal(0.3, 2.8, T),
}, index=idx)

LOADINGS = {"A": (1.20, 0.80, -0.30), "B": (0.90, -0.20, 0.60), "C": (1.05, 0.10, 0.05)}
TRUE_ALPHA = 0.25

assets = pd.DataFrame({
    name: TRUE_ALPHA + factors @ np.array(b) + rng.normal(0, 1.5, T)
    for name, b in LOADINGS.items()
}, index=idx)

checks = []


def check(name, ok):
    checks.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)


r = regress(assets["A"], factors)
check("regress returns the expected keys",
      all(k in r for k in ("alpha", "t_alpha", "r2", "Mkt-RF", "t_SMB")))
for f, true_b in zip(factors.columns, LOADINGS["A"]):
    check(f"asset A loading on {f} recovered ({r[f]:.3f} vs {true_b})",
          abs(r[f] - true_b) < 0.05)
check(f"planted alpha recovered ({r['alpha']:.3f} vs {TRUE_ALPHA})",
      abs(r["alpha"] - TRUE_ALPHA) < 0.1)
check(f"R-squared high with only 1.5 of noise (got {r['r2']:.3f})", r["r2"] > 0.9)

table, resid = fit_panel(assets, factors)
check("fit_panel: one row per asset", list(table.index) == list(LOADINGS))
max_err = max(abs(table.loc[n, f] - b)
              for n, bs in LOADINGS.items() for f, b in zip(factors.columns, bs))
check(f"fit_panel loadings all within 0.05 (max error {max_err:.3f})", max_err < 0.05)
check("residuals are orthogonal to the factors",
      abs(np.corrcoef(resid["B"], factors["HML"])[0, 1]) < 1e-8)

g = grs(table["alpha"], resid, factors)
check(f"GRS rejects when a real alpha is planted (p={g['p_value']:.2e})", g["p_value"] < 0.01)

# A portfolio that is a fixed mix of the factors themselves has no alpha by
# construction: it lives inside the span of the regressors.
combo = factors @ np.array([0.5, 0.3, 0.2])
z = regress(combo, factors)
check(f"pure factor combination has ~zero alpha (got {z['alpha']:.2e})",
      abs(z["alpha"]) < 1e-8)
check(f"and R-squared of 1 (got {z['r2']:.6f})", abs(z["r2"] - 1) < 1e-8)

zero_alpha = pd.DataFrame({n: factors @ np.array(b) + rng.normal(0, 1.5, T)
                           for n, b in LOADINGS.items()}, index=idx)
t0, e0 = fit_panel(zero_alpha, factors)
g0 = grs(t0["alpha"], e0, factors)
check(f"GRS does not reject when alphas are truly zero (p={g0['p_value']:.3f})",
      g0["p_value"] > 0.05)

# Under the null the statistic is F(N, T-N-K), whose mean is (T-N-K)/(T-N-K-2).
# A wrong prefactor (the textbook T/(T-K-1) paired with MLE covariances) shifts
# that mean by more than 0.1 in this small-sample design, so the check catches it.
nrng = np.random.default_rng(2)
Tn, Nn, Kn, sims = 30, 12, 2, 1000
nidx = pd.period_range("2000-01", periods=Tn, freq="M")
stats = []
for _ in range(sims):
    Fn = nrng.normal(0.5, 4, (Tn, Kn))
    Xn = np.column_stack([np.ones(Tn), Fn])
    Yn = Fn @ nrng.normal(1, 0.5, (Kn, Nn)) + nrng.normal(0, 2, (Tn, Nn))
    Bn = np.linalg.lstsq(Xn, Yn, rcond=None)[0]
    stats.append(grs(Bn[0], pd.DataFrame(Yn - Xn @ Bn, index=nidx),
                     pd.DataFrame(Fn, index=nidx))["grs"])
null_mean, f_mean = np.mean(stats), (Tn - Nn - Kn) / (Tn - Nn - Kn - 2)
check(f"GRS null mean matches F(N, T-N-K) ({null_mean:.3f} vs {f_mean:.3f}, {sims} sims)",
      abs(null_mean - f_mean) < 0.08)

p = premia(factors, split=idx[T // 2])
check("premia reports full and both subsamples", set(p["sample"]) == {
    "full", f"pre-{idx[T // 2]}", f"{idx[T // 2]}+"})
check("premia mean matches pandas", abs(p.loc["SMB"].iloc[0]["mean"] - factors["SMB"].mean()) < 1e-12)

b = blend(factors, ["Mkt-RF", "SMB", "HML"], lookback=60)
check("blend drops the lookback window", len(b) == T - 60)
check("blend vol sits below the loudest factor",
      b.std() < factors["Mkt-RF"].std())

flat = pd.Series(np.zeros(120), index=pd.period_range("2000-01", periods=120, freq="M"))
perf = performance(flat)
check("flat series: zero drawdown, zero return",
      abs(perf["max_drawdown"]) < 1e-9 and abs(perf["ann_return"]) < 1e-9)
down = pd.Series([-10.0] * 6 + [0.0] * 6, index=pd.period_range("2000-01", periods=12, freq="M"))
check(f"max drawdown of six -10% months (got {performance(down)['max_drawdown']:.2f}%)",
      abs(performance(down)["max_drawdown"] - (0.9 ** 6 - 1) * 100) < 1e-9)

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
raise SystemExit(0 if checks and all(checks) else 1)
