"""Offline checks: planted SVI parameters, a planted arbitrage, and a known price."""
import numpy as np
import pandas as pd

from data import forward_and_discount
from svi import (add_implied_vols, bs_price, butterfly_check, calendar_check,
                 durrleman_g, fit_slice, fit_surface, implied_vol,
                 market_calendar_check, params_table, raw_svi, svi_derivs)

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


TRUE = np.array([0.010, 0.130, -0.62, 0.045, 0.140])
T = 0.5
F, D = 100.0, 0.98
rng = np.random.default_rng(11)

price = float(bs_price(100.0, 95.0, 0.75, 0.24, 0.97, "C"))
back = implied_vol(price, 100.0, 95.0, 0.75, 0.97, "C")
check(f"Black-76 inverts a known call price (0.24 -> {back:.6f})", abs(back - 0.24) < 1e-6)

put = float(bs_price(100.0, 105.0, 0.75, 0.31, 0.97, "P"))
check(f"and a known put price (0.31 -> {implied_vol(put, 100.0, 105.0, 0.75, 0.97, 'P'):.6f})",
      abs(implied_vol(put, 100.0, 105.0, 0.75, 0.97, "P") - 0.31) < 1e-6)

check("put-call parity holds on the synthetic prices",
      abs((float(bs_price(100.0, 105.0, 0.75, 0.31, 0.97, "C")) - put) - 0.97 * (100 - 105)) < 1e-10)

check("an option priced below intrinsic returns NaN rather than a vol",
      np.isnan(implied_vol(0.5 * D * (F - 90), F, 90.0, T, D, "C")))

# pin the formula itself, not just its self-consistency: a=.04 b=.1 rho=-.5 m=.02
# sigma=.1 at k=.12 gives .04 + .1*(-.05 + sqrt(.02)) by hand
check("raw SVI matches a hand-computed value",
      abs(raw_svi(0.12, [0.04, 0.1, -0.5, 0.02, 0.1]) - (0.04 + 0.1 * (-0.05 + np.sqrt(0.02)))) < 1e-12)
h = 1e-5
kk = np.array([-0.3, 0.0, 0.25])
w0, w1, w2 = svi_derivs(kk, TRUE)
fd1 = (raw_svi(kk + h, TRUE) - raw_svi(kk - h, TRUE)) / (2 * h)
fd2 = (raw_svi(kk + h, TRUE) - 2 * w0 + raw_svi(kk - h, TRUE)) / h**2
g_fd = (1 - kk * fd1 / (2 * w0))**2 - fd1**2 / 4 * (1 / w0 + 0.25) + fd2 / 2
check("closed-form w', w'' and Durrleman g agree with finite differences",
      np.allclose(w1, fd1, atol=1e-7) and np.allclose(w2, fd2, atol=1e-4)
      and np.allclose(durrleman_g(kk, TRUE), g_fd, atol=1e-4))

NOISE = 0.002
k = np.linspace(-0.35, 0.30, 40)
w_true = raw_svi(k, TRUE)
iv_true = np.sqrt(w_true / T)
iv_noisy = iv_true + rng.normal(0, NOISE, len(k))
w_noisy = iv_noisy**2 * T

p = fit_slice(k, w_noisy, T)
dev = np.abs(np.sqrt(raw_svi(k, p) / T) - iv_true)
print("      true", np.round(TRUE, 4), "\n      fit ", np.round(p, 4))
check(f"recovered vol curve inside the {NOISE*1e4:.0f} bp noise level "
      f"(RMSE {np.sqrt((dev**2).mean())*1e4:.2f} bp, max {dev.max()*1e4:.2f})",
      np.sqrt((dev**2).mean()) < NOISE and dev.max() < 2 * NOISE)
check(f"recovered the ATM level (a+b*sigma*sqrt(1-rho^2)) within 1e-3",
      abs((p[0] + p[1] * p[4] * np.sqrt(1 - p[2] ** 2)) -
          (TRUE[0] + TRUE[1] * TRUE[4] * np.sqrt(1 - TRUE[2] ** 2))) < 1e-3)
check(f"recovered the skew slope b*rho within 0.02 ({p[1]*p[2]:.4f} vs {TRUE[1]*TRUE[2]:.4f})",
      abs(p[1] * p[2] - TRUE[1] * TRUE[2]) < 0.02)
check("the planted slice itself is butterfly-free", not butterfly_check(TRUE)["violates"])
check("and so is the recovered one", not butterfly_check(p)["violates"])

# b large with rho near the boundary drives Durrleman negative in the left wing
BAD = np.array([0.005, 0.85, -0.92, 0.02, 0.03])
bad = butterfly_check(BAD)
check(f"a planted arbitrageable slice is flagged (min g = {bad['min_g']:.4f} at k={bad['k_at_min']:.2f})",
      bad["violates"])
check("its Durrleman function really does go negative somewhere",
      durrleman_g(np.linspace(-1, 1, 501), BAD).min() < 0)

SHORT = {"T": 0.25, "params": np.array([0.020, 0.10, -0.5, 0.0, 0.12])}
LONG_BAD = {"T": 0.50, "params": np.array([0.012, 0.10, -0.5, 0.0, 0.12])}
LONG_OK = {"T": 0.50, "params": np.array([0.040, 0.12, -0.5, 0.0, 0.12])}
check("a long slice below a short one is flagged as calendar arbitrage",
      calendar_check([SHORT, LONG_BAD])[0]["violates"])
check("a properly ordered pair is not",
      not calendar_check([SHORT, LONG_OK])[0]["violates"])

# a surface whose middle slice sits too low, fitted with and without enforcement
levels = [0.030, 0.028, 0.075]
mats = [0.25, 0.50, 1.00]
rows = []
for T_i, lvl in zip(mats, levels):
    pi = np.array([lvl - 0.13 * 0.12 * np.sqrt(1 - 0.6**2), 0.13, -0.6, 0.03, 0.12])
    for kk in np.linspace(-0.30, 0.25, 25):
        w = raw_svi(kk, pi) + rng.normal(0, 1e-5)
        iv = np.sqrt(w / T_i)
        rows.append({"T": T_i, "expiry": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(365 * T_i)),
                     "k": kk, "iv_mid": iv, "w_mid": w,
                     "w_bid": w * 0.97, "w_ask": w * 1.03})
synth = pd.DataFrame(rows)

check("the planted crossing is visible in the market total variances",
      any(c["violates"] for c in market_calendar_check(synth)))
free = fit_surface(synth, enforce_calendar=False, enforce_butterfly=False)
check("an unconstrained fit inherits the calendar arbitrage",
      any(c["violates"] for c in calendar_check(free)))
forced = fit_surface(synth, enforce_calendar=True, enforce_butterfly=True)
check("enforcing calendar removes it", not any(c["violates"] for c in calendar_check(forced)))
check("and no butterfly arbitrage is introduced",
      not any(butterfly_check(f["params"])["violates"] for f in forced))
# lifting the middle slice above the short one costs at least 0.83 vol points
# by construction, so this bounds the damage rather than demanding a free lunch
worst = params_table(forced).rmse_vol.max()
check(f"the price of enforcement stays under 1.5 vol points (worst slice {worst*100:.2f})",
      worst < 0.015)

# parity forward recovery: build a chain from a known forward and read it back
K = np.arange(80.0, 121.0, 2.0)
chain = []
for cp in ("C", "P"):
    px = bs_price(103.5, K, 0.5, 0.20, 0.98, cp)
    chain += [{"expiry": pd.Timestamp("2026-07-01"), "T": 0.5, "cp": cp,
               "strike": k_, "mid": float(p_), "spot": 100.0} for k_, p_ in zip(K, px)]
fwd = forward_and_discount(pd.DataFrame(chain), r=-np.log(0.98) / 0.5)
check(f"put-call parity recovers a planted forward of 103.5 (got {fwd['F'].iloc[0]:.4f})",
      abs(fwd["F"].iloc[0] - 103.5) < 1e-6)

# implied vols come back out of a synthetic chain unchanged
panel = pd.DataFrame({"bid": [4.9], "mid": [5.0], "ask": [5.1], "F": [100.0],
                      "strike": [100.0], "T": [0.5], "D": [0.98], "cp": ["C"],
                      "impliedVolatility": [0.18]})
panel = add_implied_vols(panel)
check("add_implied_vols orders bid <= mid <= ask vols",
      panel["iv_bid"].iloc[0] < panel["iv_mid"].iloc[0] < panel["iv_ask"].iloc[0])
check("and total variance is iv^2 * T",
      abs(panel["w_mid"].iloc[0] - panel["iv_mid"].iloc[0] ** 2 * 0.5) < 1e-12)

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
raise SystemExit(0 if checks and all(checks) else 1)
