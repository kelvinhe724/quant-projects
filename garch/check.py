"""Offline checks for garch.py: a simulated GARCH(1,1) with known parameters."""
import numpy as np
import pandas as pd

from garch import fit_garch, fit_all, var_backtest

TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA = 0.02, 0.08, 0.90

rng = np.random.default_rng(3)
n = 5000
z = rng.standard_normal(n)
eps = np.zeros(n)
sig2 = np.zeros(n)
sig2[0] = TRUE_OMEGA / (1 - TRUE_ALPHA - TRUE_BETA)  # start at long-run variance
eps[0] = np.sqrt(sig2[0]) * z[0]
for t in range(1, n):
    sig2[t] = TRUE_OMEGA + TRUE_ALPHA * eps[t - 1] ** 2 + TRUE_BETA * sig2[t - 1]
    eps[t] = np.sqrt(sig2[t]) * z[t]

idx = pd.bdate_range("2010-01-04", periods=n)
returns = pd.Series(eps, index=idx, name="FAKE")

checks = []

def check(name, ok):
    checks.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)

r = fit_garch(returns)
check("fit_garch returns the expected keys",
      all(k in r for k in ("omega", "alpha", "beta", "persistence", "aic", "bic")))
check(f"alpha close to the true {TRUE_ALPHA} (got {r['alpha']:.3f})",
      abs(r["alpha"] - TRUE_ALPHA) < 0.03)
check(f"beta close to the true {TRUE_BETA} (got {r['beta']:.3f})",
      abs(r["beta"] - TRUE_BETA) < 0.04)
check(f"persistence below 1 (got {r['persistence']:.3f})", r["persistence"] < 1)
check("conditional vol tracks the planted vol",
      np.corrcoef(r["res"].conditional_volatility, np.sqrt(sig2))[0, 1] > 0.9)

table, fits = fit_all(pd.DataFrame({"A": returns, "B": returns * 0.5}))
check("fit_all: one row per index", len(table) == 2)
check("fit_all: keeps the fitted results", set(fits) == {"A", "B"})

vb = var_backtest(returns, r["res"], level=0.05)
check(f"5% VaR hit rate near 5% (got {vb['observed']:.3f})",
      abs(vb["observed"] - 0.05) < 0.01)

t = fit_garch(returns, dist="t")
check("student-t fit also recovers alpha", abs(t["alpha"] - TRUE_ALPHA) < 0.03)

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
raise SystemExit(0 if checks and all(checks) else 1)
