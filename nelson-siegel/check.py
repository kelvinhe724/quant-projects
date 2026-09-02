"""Offline checks for ns.py and data.py: a planted NS curve plus noise."""
import numpy as np

from data import contract_table, synthetic_panel, daily_slices, last_trading_date
from ns import (ns_curve, fit_static, fit_all_static, fit_sequential,
                change_scale, path_roughness, default_start)

rng = np.random.default_rng(3)
checks = []

def check(name, ok):
    checks.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)

check("ns_curve at long tenor -> beta0",
      abs(ns_curve(1000.0, 4.3, -0.1, 0.05, 6.0) - 4.3) < 1e-3)
check("ns_curve near tau=0 -> beta0+beta1",
      abs(ns_curve(1e-6, 4.3, -0.1, 0.05, 6.0) - 4.2) < 1e-3)

# lam has to decay inside the 1-12 month window or it is not identified:
# a longer lam has equal-RMSE twins
TRUE = dict(beta0=4.30, beta1=-0.15, beta2=0.10, lam=2.0)
tau = np.arange(1, 13, dtype=float)          # 12 monthly tenors
y = ns_curve(tau, **TRUE) + rng.normal(0, 0.0005, len(tau))
r = fit_static(tau, y)
check("static fit converges", r["success"])
check(f"beta0 recovered (got {r['beta0']:.3f})", abs(r["beta0"] - TRUE["beta0"]) < 0.02)
check(f"beta1 recovered (got {r['beta1']:.3f})", abs(r["beta1"] - TRUE["beta1"]) < 0.03)
check(f"beta2 recovered (got {r['beta2']:.3f})", abs(r["beta2"] - TRUE["beta2"]) < 0.05)
check(f"lambda recovered (got {r['lam']:.2f})", abs(r["lam"] - TRUE["lam"]) < 0.5)
check(f"rmse tiny (got {r['rmse']:.5f})", r["rmse"] < 0.002)

# planted constant curve plus noise; the ridge path should come out smoother
# than independent random-start fits
slices = []
import pandas as pd
for i, d in enumerate(pd.bdate_range("2026-01-02", periods=60)):
    yy = ns_curve(tau, **TRUE) + rng.normal(0, 0.002, len(tau))
    slices.append((d, tau, yy, [], False))

bench = fit_all_static(slices, mode="random")
check("change_scale gives 4 positive scales", (change_scale(bench) > 0).all())
scale = np.array([0.02, 0.02, 0.05, 0.1])
ridge = fit_sequential(slices, "ridge", strength=1e-4, scale=scale, burn_in=10)
l1 = fit_sequential(slices, "l1", strength=1e-4, scale=scale, burn_in=10)

r_bench = path_roughness(bench, skip=10)
r_ridge = path_roughness(ridge, skip=10)
r_l1 = path_roughness(l1, skip=10)
check(f"ridge path smoother than random-start benchmark ({r_ridge:.4f} < {r_bench:.4f})",
      r_ridge < r_bench)
check(f"l1 path smoother than benchmark ({r_l1:.4f} < {r_bench:.4f})", r_l1 < r_bench)
fit_ridge = np.mean([x["rmse"] for x in ridge[10:]])
check(f"ridge still fits (mean rmse {fit_ridge:.4f} < 2x noise)", fit_ridge < 0.004)
b0_err = abs(np.mean([x["beta0"] for x in ridge[10:]]) - TRUE["beta0"])
check(f"ridge beta0 path centred on truth (err {b0_err:.4f})", b0_err < 0.02)

# level jumps mid-sample; the ridge path has to follow it
slices2 = []
for i, d in enumerate(pd.bdate_range("2026-01-02", periods=40)):
    b0 = 4.30 if i < 20 else 4.40
    yy = ns_curve(tau, b0, TRUE["beta1"], TRUE["beta2"], TRUE["lam"]) \
        + rng.normal(0, 0.002, len(tau))
    slices2.append((d, tau, yy, [], False))
ridge2 = fit_sequential(slices2, "ridge", strength=1e-4, scale=scale, burn_in=5)
check(f"ridge follows a real level shift (end beta0 {ridge2[-1]['beta0']:.3f})",
      abs(ridge2[-1]["beta0"] - 4.40) < 0.02)

check("COH6 last trade = end Jan 2026 per ICE rule",
      str(last_trading_date("COH6").date()) == "2026-01-30")
panel = synthetic_panel()
check("no post-expiry prices in panel",
      (panel["date"] <= panel["last_trade"]).all())
sl = daily_slices(panel)
check("slices built for every trading day", len(sl) > 90)
check("first slice has 12 contracts", len(sl[0][1]) == 12)
check("tenors strictly increasing in a slice",
      bool(np.all(np.diff(sl[0][1]) > 0)))
late = [s for s in sl if s[0] > pd.Timestamp("2026-05-15")]
check("late days flagged incomplete under 13-contract scope (see README)",
      all(s[4] for s in late))

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
