"""Offline checks for capm.py: simulated data with a known beta. Run: python3 check.py"""
import numpy as np
import pandas as pd

from capm import fit_capm, fit_all, rolling_beta

rng = np.random.default_rng(7)
n = 1000
idx = pd.bdate_range("2022-01-03", periods=n)
market = pd.Series(rng.normal(0.0003, 0.011, n), index=idx)

TRUE_BETA = 1.4
stock = TRUE_BETA * market + pd.Series(rng.normal(0, 0.012, n), index=idx)

checks = []

def check(name, ok):
    checks.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)

try:
    r = fit_capm(stock, market)
    check("fit_capm returns the four keys",
          all(k in r for k in ("alpha_annual", "beta", "r2", "alpha_pvalue")))
    check(f"beta close to the true {TRUE_BETA} (got {r['beta']:.3f})",
          abs(r["beta"] - TRUE_BETA) < 0.1)
    check(f"alpha near zero (got {r['alpha_annual']:.4f})",
          abs(r["alpha_annual"]) < 0.05)
    check(f"r2 sensible (got {r['r2']:.2f})", 0.3 < r["r2"] < 0.9)
    check(f"alpha p-value says 'luck' (got {r['alpha_pvalue']:.2f})",
          r["alpha_pvalue"] > 0.05)
except NotImplementedError:
    print("fit_capm not written yet")

try:
    # three tickers with known betas 0.5 < 1.4 < 2.0, so the sort is testable
    df = fit_all(pd.DataFrame({"FAKE1": stock, "FAKE2": 0.5 * market,
                               "FAKE3": 2.0 * market}), market)
    check("fit_all: one row per ticker", len(df) == 3)
    check("fit_all: sorted ascending by beta",
          list(df["ticker"]) == ["FAKE2", "FAKE1", "FAKE3"])
except NotImplementedError:
    print("fit_all not written yet")

try:
    rb = rolling_beta(stock, market, window=60)
    rb = rb.dropna()
    check(f"rolling beta hovers near {TRUE_BETA} (mean {rb.mean():.2f})",
          abs(rb.mean() - TRUE_BETA) < 0.15)
    check("rolling beta actually moves around", rb.std() > 0.01)
except NotImplementedError:
    print("rolling_beta not written yet")

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
raise SystemExit(0 if checks and all(checks) else 1)
