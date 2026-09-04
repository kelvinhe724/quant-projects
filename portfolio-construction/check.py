"""Offline checks on simulated returns with known means and covariance.

The optimiser must recover the analytic tangency weights when handed the truth,
the rules must satisfy their defining identities, and the engine must trade on
the right clock and charge the right turnover.

Run: python3 check.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import backtest
from portfolio import (equal_weight, erc, estimate, inverse_vol, james_stein, ledoit_wolf,
                       long_only_tangency, min_variance, risk_contributions, tangency)

rng = np.random.default_rng(3)
checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def planted_moments(n=6):
    """A covariance with a factor structure and means that reward a real spread of positions."""
    beta = rng.uniform(0.5, 1.5, n)
    idio = rng.uniform(0.02, 0.05, n)
    sigma = np.outer(beta, beta) * 0.04 ** 2 + np.diag(idio ** 2)
    mu = 0.004 + 0.004 * beta + rng.normal(0, 0.002, n)
    return mu, sigma


def sharpe(w, mu, sigma):
    return (w @ mu) / np.sqrt(w @ sigma @ w)


MU, SIGMA = planted_moments()
N = len(MU)

print("PORTFOLIO RULES ON KNOWN MOMENTS\n")

w_tan = tangency(MU, SIGMA)
analytic = np.linalg.inv(SIGMA) @ MU
analytic /= analytic.sum()
check("tangency matches the closed form inv(Sigma) mu / 1' inv(Sigma) mu",
      np.allclose(w_tan, analytic, atol=1e-12))
perturbed = [sharpe(w_tan + rng.normal(0, 0.05, N), MU, SIGMA) for _ in range(2000)]
check("no random perturbation of the tangency weights has a higher Sharpe",
      max(perturbed) < sharpe(w_tan, MU, SIGMA) + 1e-12)

big = rng.multivariate_normal(MU, SIGMA, size=400_000)
mu_hat, sigma_hat = estimate(big)
check("tangency on 400k simulated months recovers the true weights within 2%",
      np.abs(tangency(mu_hat, sigma_hat) - w_tan).max() < 0.02)

check("1/N weights sum to one and are all equal",
      np.isclose(equal_weight(N).sum(), 1.0) and np.ptp(equal_weight(N)) == 0)
check("every rule returns weights summing to one",
      all(np.isclose(f.sum(), 1.0) for f in
          (w_tan, min_variance(SIGMA), inverse_vol(SIGMA), erc(SIGMA),
           long_only_tangency(MU, SIGMA))))

w_mv = min_variance(SIGMA)
trials = [w_mv + d - d.mean() for d in rng.normal(0, 0.05, (2000, N))]
check("minimum variance beats every fully invested perturbation on variance",
      min(t @ SIGMA @ t for t in trials) > w_mv @ SIGMA @ w_mv)

w_erc = erc(SIGMA)
rc = risk_contributions(w_erc, SIGMA)
check("ERC gives equal risk contributions (max gap under 1e-6)",
      np.ptp(rc) < 1e-6 and np.isclose(rc.sum(), 1.0))
check("ERC is long only", (w_erc > 0).all())
check("inverse vol equals ERC when assets are uncorrelated",
      np.allclose(erc(np.diag(np.diag(SIGMA))), inverse_vol(SIGMA), atol=1e-6))

all_positive_mu = np.full(N, 0.01)
diag_sigma = np.diag(rng.uniform(0.02, 0.06, N) ** 2)
check("long-only tangency matches unconstrained tangency when the latter is already long only",
      np.allclose(long_only_tangency(all_positive_mu, diag_sigma),
                  tangency(all_positive_mu, diag_sigma), atol=1e-5))
w_lo = long_only_tangency(MU, SIGMA)
check("long-only tangency has no short and no Sharpe above the unconstrained optimum",
      (w_lo >= 0).all() and sharpe(w_lo, MU, SIGMA) <= sharpe(w_tan, MU, SIGMA) + 1e-9)

print("\nSHRINKAGE\n")

small = rng.multivariate_normal(MU, SIGMA, size=36)
s_sample = np.cov(small, rowvar=False)
s_lw, delta = ledoit_wolf(small)
target = np.eye(N) * np.trace(s_sample) / N
dist = lambda a: np.linalg.norm(a - target)
check("Ledoit-Wolf intensity lies in (0, 1]", 0 < delta <= 1)
check("Ledoit-Wolf covariance sits closer to the scaled identity than the sample covariance",
      dist(s_lw) < dist(s_sample))
check("Ledoit-Wolf is the convex combination (1-d) S + d target",
      np.allclose(s_lw, (1 - delta) * np.cov(small, rowvar=False, ddof=0) + delta
                  * np.eye(N) * np.trace(np.cov(small, rowvar=False, ddof=0)) / N, atol=1e-10))
check("Ledoit-Wolf covariance is closer to the truth than the sample covariance on 36 months",
      np.linalg.norm(s_lw - SIGMA) < np.linalg.norm(s_sample - SIGMA))

mu_js, lam = james_stein(small, s_sample)
grand = small.mean(axis=0).mean()
check("James-Stein intensity lies in (0, 1]", 0 < lam <= 1)
check("James-Stein means sit strictly between the sample means and the grand mean",
      np.all(np.abs(mu_js - grand) <= np.abs(small.mean(axis=0) - grand) + 1e-15)
      and np.isclose(mu_js.mean(), grand))
check("James-Stein shrinks harder on 36 months than on 3600",
      lam > james_stein(rng.multivariate_normal(MU, SIGMA, size=3600), SIGMA)[1])

print("\nENGINE\n")

idx = pd.date_range("2020-01-31", periods=6, freq="ME")
two = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [0.10, -0.10], [0.0, 0.0], [0.20, 0.0], [0.0, 0.0]],
                   index=idx, columns=["A", "B"])
book, path = backtest.run("1/N", two, window=2, cost_bps=100.0, total=two)
# Hand case: start from cash so the first trade is the full book (turnover 1).
# 50/50 drifts to 55/45 after +10%/-10%, rebalancing back costs 0.10.
# After a flat month nothing moves. After +20%/0% the book is 60/50 -> 0.545/0.455, turnover 0.0909.
hand = [1.0, 0.10, 0.0, 0.10 / 1.1]
check("turnover matches the hand-computed drift case",
      np.allclose(book["turnover"].to_numpy(), hand, atol=1e-12))
check("cost is turnover times the bps charge",
      np.allclose(book["gross"] - book["net"], book["turnover"] * 0.01))
check("weights formed at t earn the return of t+1",
      np.isclose(book["gross"].iloc[2], 0.5 * 0.20) and book["gross"].iloc[1] == 0.0)

n_months, n_assets, m = 200, N, 60
sim = pd.DataFrame(rng.multivariate_normal(MU, SIGMA, size=n_months),
                   index=pd.date_range("2000-01-31", periods=n_months, freq="ME"))
_, path = backtest.run("tangency", sim, m)
spiked = sim.copy()
spiked.iloc[150:] += 0.5
_, path_spiked = backtest.run("tangency", spiked, m)
check("changing returns after t leaves every weight set at or before t unchanged",
      np.allclose(path.loc[:sim.index[149]], path_spiked.loc[:sim.index[149]]))
check("weights change once the estimation window reaches the altered data",
      not np.allclose(path.loc[sim.index[150]:], path_spiked.loc[sim.index[150]:]))
check("each weight vector uses exactly the trailing window",
      np.allclose(path.iloc[0], tangency(*estimate(sim.iloc[:m]))))

print("\nMETRICS\n")

flat = pd.DataFrame({"net": np.full(48, 0.01), "turnover": 0.0, "gross_exposure": 1.0, "max_weight": 0.5},
                    index=pd.date_range("2000-01-31", periods=48, freq="ME"))
flat.loc[flat.index[::2], "net"] = 0.03
m = backtest.metrics(flat)
sd = 0.01 * np.sqrt(48 / 47)  # alternating 0.01 / 0.03 has population sd 0.01; pandas uses ddof=1
check("annualised mean is 12x the monthly mean and vol is sqrt(12)x the monthly sd",
      np.isclose(m["mean_excess"], 0.24) and np.isclose(m["vol"], sd * np.sqrt(12)))
check("Sharpe is the monthly ratio scaled by sqrt(12)", np.isclose(m["sharpe"], 0.02 / sd * np.sqrt(12)))
check("max drawdown finds a planted -50% leg",
      np.isclose(backtest.max_drawdown(pd.Series([0.1, -0.5, 0.2, 0.5])), -0.5))

a, b = rng.normal(0.2, 1, 120), rng.normal(0, 1, 120)
sa, sb, rho, t = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1), np.corrcoef(a, b)[0, 1], 120
memmel = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / t
check("Jobson-Korkie p-value matches Memmel's variance formula worked by hand",
      np.isclose(backtest.sharpe_diff_pvalue(a, b), 2 * (1 - norm.cdf(abs(sa - sb) / np.sqrt(memmel)))))
check("a series compared with a near-copy of itself gets p-value 1",
      np.isclose(backtest.sharpe_diff_pvalue(a, a + 1e-9 * rng.normal(size=120)), 1.0, atol=1e-3))

print("\nESTIMATION ERROR STORY\n")

true_tan = sharpe(w_tan, MU, SIGMA)
true_ew = sharpe(equal_weight(N), MU, SIGMA)
check("planted moments give the tangency portfolio a real edge over 1/N", true_tan > true_ew)


def expected_oos_sharpe(window, draws=300):
    """Average true Sharpe of plug-in tangency weights estimated on `window` simulated months."""
    out = []
    for _ in range(draws):
        w = tangency(*estimate(rng.multivariate_normal(MU, SIGMA, size=window)))
        out.append(sharpe(w, MU, SIGMA))
    return np.mean(out)


short, long = expected_oos_sharpe(24), expected_oos_sharpe(2400)
check("with 24 months of history the plug-in tangency portfolio loses to 1/N", short < true_ew)
check("with 2400 months it recovers most of the true edge",
      long > true_ew and long > true_tan - 0.3 * (true_tan - true_ew))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
