"""Offline checks on simulated data with a known hedge ratio.

Builds a two-asset system where the target's P&L is a fixed multiple of the hedge's
plus independent basis noise, so the right answer is known before the estimator runs.
Also runs the spurious-regression demonstration that motivates working in returns.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from data import SPECS, contract_pnl, notional
from hedging import (block_bootstrap, contracts_from_return_beta, effectiveness,
                     hedged_pnl, integer_hedge, min_var_ratio, ols_hac, r2_matrix,
                     rolling_ratio, screen, spurious_demo, walk_forward)

rng = np.random.default_rng(11)
N_DAYS = 1500
IDX = pd.bdate_range("2015-01-05", periods=N_DAYS)

TRUE_H = 1.75
BASIS_SD = 40.0
HEDGE_SD = 300.0

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def planted(h=TRUE_H, basis_sd=BASIS_SD, n=N_DAYS):
    """Two P&L series where dV_target = h * dV_hedge + independent noise."""
    hedge = rng.normal(0, HEDGE_SD, n)
    target = h * hedge + rng.normal(0, basis_sd, n)
    return (pd.Series(target, index=IDX[:n], name="target"),
            pd.Series(hedge, index=IDX[:n], name="hedge"))


target, hedge = planted()

print("RECOVERING A PLANTED HEDGE RATIO\n")

h_hat = min_var_ratio(target, hedge)
check(f"the minimum-variance ratio recovers the planted {TRUE_H} (got {h_hat:.4f})",
      abs(h_hat - TRUE_H) < 0.01)

fit = ols_hac(target, hedge)
check("the OLS slope and the covariance ratio are the same number",
      np.isclose(fit["beta"], h_hat))
check(f"the planted ratio sits inside two HAC standard errors "
      f"({fit['beta']:.4f} +/- {2 * fit['beta_se']:.4f})",
      abs(fit["beta"] - TRUE_H) < 2 * fit["beta_se"])

implied_r2 = (TRUE_H * HEDGE_SD) ** 2 / ((TRUE_H * HEDGE_SD) ** 2 + BASIS_SD ** 2)
check(f"R-squared matches the signal-to-noise ratio it was built from "
      f"({fit['r2']:.4f} vs {implied_r2:.4f})",
      abs(fit["r2"] - implied_r2) < 0.01)

# A hedge ratio should not care which series carries the bigger multiplier, but the
# R-squared should be identical in both directions while the slopes are reciprocal-ish.
reverse = ols_hac(hedge, target)
check("R-squared is symmetric across the regression direction",
      np.isclose(fit["r2"], reverse["r2"], atol=1e-10))
check("the two slopes are not equal, so slope is a sizing statistic and not a quality score",
      abs(fit["beta"] - reverse["beta"]) > 0.5)

print("\nVARIANCE REDUCTION\n")

eff = effectiveness(target, hedge, h_hat)
check(f"hedging at the estimated ratio reduces variance ({eff:.4f} of it)", eff > 0.9)
check("the hedged series really does have lower variance than the unhedged one",
      hedged_pnl(target, hedge, h_hat).var() < target.var())
check("effectiveness equals the regression R-squared when the ratio is the OLS slope",
      np.isclose(eff, fit["r2"], atol=1e-6))

check("hedging at zero contracts changes nothing", np.isclose(effectiveness(target, hedge, 0.0), 0.0))
check("over-hedging by a factor of three increases variance",
      effectiveness(target, hedge, 3 * h_hat) < 0)
check("hedging with the wrong sign increases variance",
      effectiveness(target, hedge, -h_hat) < 0)

# The minimum-variance ratio is a minimum, so nothing else can beat it in sample.
grid = np.linspace(TRUE_H - 1, TRUE_H + 1, 201)
best = grid[np.argmax([effectiveness(target, hedge, g) for g in grid])]
check(f"no ratio on a grid beats the estimated one in sample (best grid point {best:.3f})",
      abs(best - h_hat) < 0.02)

independent = pd.Series(rng.normal(0, HEDGE_SD, N_DAYS), index=IDX)
check("hedging with an unrelated contract achieves roughly nothing",
      abs(effectiveness(target, independent, min_var_ratio(target, independent))) < 0.02)

print("\nSPURIOUS REGRESSION ON PRICE LEVELS\n")

demo = spurious_demo(n=2000, n_pairs=200, seed=1)
lev_r2, dif_r2 = demo["r2_levels"].mean(), demo["r2_diffs"].mean()
check(f"independent random walks regressed in LEVELS produce a high mean R-squared "
      f"({lev_r2:.3f})", lev_r2 > 0.15)
check(f"the same series in DIFFERENCES produce essentially none ({dif_r2:.5f})",
      dif_r2 < 0.01)
check(f"the level R-squared is at least 20x the difference R-squared "
      f"({lev_r2 / dif_r2:.0f}x)", lev_r2 / dif_r2 > 20)

lev_sig = (demo["t_levels"] > 1.96).mean()
dif_sig = (demo["t_diffs"] > 1.96).mean()
check(f"the level regression rejects a zero slope {lev_sig:.0%} of the time on pure noise",
      lev_sig > 0.5)
check(f"the difference regression rejects at roughly the nominal 5% ({dif_sig:.1%})",
      dif_sig < 0.12)
check(f"level residuals are strongly autocorrelated, Durbin-Watson "
      f"{demo['dw_levels'].mean():.3f} against a null of 2",
      demo["dw_levels"].mean() < 0.5)

# The same trap with the project's own machinery: random walks priced as futures and
# pushed through contract_pnl. One draw is not enough, because the spurious R-squared
# is itself a random variable with most of its mass spread over [0, 0.8].
level_r2s, return_r2s, oos_effs = [], [], []
for _ in range(50):
    px = pd.DataFrame({"CL=F": np.cumsum(rng.normal(0, 0.8, N_DAYS)) + 70,
                       "BZ=F": np.cumsum(rng.normal(0, 0.8, N_DAYS)) + 70}, index=IDX)
    level_r2s.append(ols_hac(px["CL=F"], px["BZ=F"])["r2"])
    return_r2s.append(r2_matrix(px.pct_change().dropna()).loc["CL=F", "BZ=F"])
    pnl = contract_pnl(px).dropna()
    oos_effs.append(effectiveness(pnl["CL=F"].iloc[750:], pnl["BZ=F"].iloc[750:],
                                  min_var_ratio(pnl["CL=F"].iloc[:750], pnl["BZ=F"].iloc[:750])))
level_r2, return_r2 = np.median(level_r2s), np.median(return_r2s)
check(f"fake crude series look related in levels (median R2 {level_r2:.3f}) "
      f"and unrelated in returns (median R2 {return_r2:.5f})",
      level_r2 > 0.15 and return_r2 < 0.005)
check(f"a level R-squared above 0.5 turns up on pure noise "
      f"{(np.array(level_r2s) > 0.5).mean():.0%} of the time",
      (np.array(level_r2s) > 0.5).mean() > 0.1)
check(f"a hedge fitted to those fake series delivers no out-of-sample variance "
      f"reduction (median {np.median(oos_effs):+.4f})",
      np.median(oos_effs) < 0.02 and np.mean(np.array(oos_effs) < 0.05) > 0.9)

print("\nINTEGER CONTRACTS\n")

# Rounding a continuous ratio to whole contracts can only give back some of the
# variance reduction, and the amount it gives back is bounded by the rounding error
# times the hedge's variance. This asserts that bound rather than a hand-picked number.
var_target = float(np.var(target, ddof=1))
var_hedge = float(np.var(hedge, ddof=1))
worst_gap = 0.0
bound_holds = []
for n_target in (1, 2, 3, 5, 10, 25, 100):
    continuous = h_hat * n_target
    eff_cont = effectiveness(target, hedge, continuous, n_target=n_target)
    for candidate in integer_hedge(h_hat, n_target):
        eff_int = effectiveness(target, hedge, candidate, n_target=n_target)
        gap = eff_cont - eff_int
        bound = (candidate - continuous) ** 2 * var_hedge / (n_target ** 2 * var_target)
        ok = eff_int <= eff_cont + 1e-12 and gap <= bound + 1e-9
        if not ok:
            print(f"      n={n_target} candidate={candidate} gap={gap:.6f} bound={bound:.6f}")
        worst_gap = max(worst_gap, gap)
        bound_holds.append(ok)
check(f"rounding never beats the continuous ratio and never costs more than the "
      f"analytic bound at {len(bound_holds)} candidates (worst gap {worst_gap:.4f})",
      all(bound_holds))

nearest_gaps = []
for n_target in (1, 2, 5, 10, 25, 100):
    continuous = h_hat * n_target
    eff_cont = effectiveness(target, hedge, continuous, n_target=n_target)
    eff_round = effectiveness(target, hedge, round(continuous), n_target=n_target)
    nearest_gaps.append(eff_cont - eff_round)
check(f"the rounding cost shrinks as the position scales up "
      f"({nearest_gaps[0]:.4f} at 1 contract, {nearest_gaps[-1]:.6f} at 100)",
      nearest_gaps[-1] < nearest_gaps[0] / 10)
check("a whole-contract hedge still beats no hedge at every size tested",
      all(effectiveness(target, hedge, round(h_hat * n), n_target=n) > 0.5
          for n in (1, 2, 5, 10, 25, 100)))

print("\nUNIT CONVERSION\n")

check("the multiplier is the lot size times the quote conversion, for every contract",
      all(SPECS[t]["multiplier"] == SPECS[t]["lot"] * SPECS[t]["quote_conversion"]
          for t in SPECS))
check("a $1/bbl move on one WTI contract is $1,000", SPECS["CL=F"]["multiplier"] == 1000)
check("a 1-cent/lb move on one Coffee C contract is $375", SPECS["KC=F"]["multiplier"] == 375)
check("a 1-cent/bushel move on one corn contract is $50", SPECS["ZC=F"]["multiplier"] == 50)
check("a $0.01/gallon move on one ULSD contract is $420",
      np.isclose(SPECS["HO=F"]["multiplier"] * 0.01, 420))

prices = pd.Series({"CL=F": 70.0, "BZ=F": 74.0, "HO=F": 2.30, "KC=F": 200.0,
                    "ZC=F": 450.0, "GC=F": 2000.0, "SI=F": 25.0, "NG=F": 3.0,
                    "RB=F": 2.10, "HG=F": 4.00, "PL=F": 950.0, "ZW=F": 600.0,
                    "ZS=F": 1300.0})
nots = notional(prices)
check("WTI at $70 is a $70,000 contract", np.isclose(nots["CL=F"], 70_000))
check("ULSD at $2.30/gal is a $96,600 contract", np.isclose(nots["HO=F"], 96_600))
check("coffee at 200 cents/lb is a $75,000 contract, not $7.5m",
      np.isclose(nots["KC=F"], 75_000))

# The worked example in section 15.1 of the brief.
ratio = contracts_from_return_beta(0.92, 70.0, 74.0, "CL=F", "BZ=F")
check(f"the brief's WTI-on-Brent example reproduces at 0.87 contracts ({ratio:.4f})",
      abs(ratio - 0.87) < 0.005)
ratio_ho = contracts_from_return_beta(0.75, 70.0, 2.30, "CL=F", "HO=F")
check(f"the brief's WTI-on-ULSD example reproduces at 0.54 contracts ({ratio_ho:.4f})",
      abs(ratio_ho - 0.54) < 0.005)

# Return beta and P&L ratio must agree when notionals are constant, because the two
# differ only by the price levels used to convert.
flat_px = pd.DataFrame({"CL=F": 70.0, "BZ=F": 74.0}, index=IDX)
shocks = pd.DataFrame({"BZ=F": rng.normal(0, 1.0, N_DAYS)}, index=IDX)
shocks["CL=F"] = 0.9 * shocks["BZ=F"] * (70 / 74) + rng.normal(0, 0.15, N_DAYS)
sim_px = flat_px + shocks.cumsum()
sim_rets = sim_px.pct_change().dropna()
sim_pnl = contract_pnl(sim_px).dropna()
beta_ret = ols_hac(sim_rets["CL=F"], sim_rets["BZ=F"])["beta"]
h_pnl = min_var_ratio(sim_pnl["CL=F"], sim_pnl["BZ=F"])
converted = contracts_from_return_beta(beta_ret, sim_px["CL=F"].mean(),
                                       sim_px["BZ=F"].mean(), "CL=F", "BZ=F")
check(f"the return-beta route and the P&L route agree to within 5% "
      f"({converted:.4f} vs {h_pnl:.4f})",
      abs(converted - h_pnl) / h_pnl < 0.05)

print("\nSTABILITY AND OUT-OF-SAMPLE MACHINERY\n")

roll = rolling_ratio(target, hedge, window=60)
check("the rolling ratio needs a full window before it produces anything",
      len(roll) == N_DAYS - 59)
check(f"the rolling ratio hovers around the planted value "
      f"(median {roll['h'].median():.3f}, IQR {roll['h'].quantile(.75) - roll['h'].quantile(.25):.3f})",
      abs(roll["h"].median() - TRUE_H) < 0.05)

spiked_target = target.copy()
spiked_target.iloc[1000] += 50_000
spiked_roll = rolling_ratio(spiked_target, hedge, window=60)
check("a shock at t=1000 leaves every rolling estimate before it untouched",
      np.allclose(roll["h"].iloc[:941], spiked_roll["h"].iloc[:941]))

boot = block_bootstrap(target, hedge, block=10, n_reps=300, seed=3)
lo, hi = boot["h"].quantile([0.025, 0.975])
check(f"the block bootstrap interval covers the planted ratio ([{lo:.3f}, {hi:.3f}])",
      lo < TRUE_H < hi)
check("the bootstrap interval is narrow when the relationship is strong",
      (hi - lo) < 0.2)

# An unstable system: the ratio flips halfway through. The bootstrap should widen,
# and the interval should straddle a region containing neither true value alone.
unstable = pd.concat([hedge.iloc[:750] * 2.5, hedge.iloc[750:] * -1.0]) \
    + pd.Series(rng.normal(0, BASIS_SD, N_DAYS), index=IDX)
boot_unstable = block_bootstrap(unstable, hedge, block=10, n_reps=300, seed=3)
u_lo, u_hi = boot_unstable["h"].quantile([0.025, 0.975])
check(f"a ratio that flips sign mid-sample gives a much wider interval "
      f"([{u_lo:.2f}, {u_hi:.2f}])", (u_hi - u_lo) > 5 * (hi - lo))

wf = walk_forward(target, hedge, window=60)
check("the walk-forward ratio applied on day t was estimated strictly before t",
      np.allclose(wf["h"].to_numpy(), roll["h"].to_numpy()[:len(wf)])
      and wf.index[0] > roll.index[0])
check(f"walk-forward hedging reduces variance on the planted system "
      f"({1 - wf['hedged'].var() / wf['unhedged'].var():.4f})",
      wf["hedged"].var() < wf["unhedged"].var())

# Estimate on the first half, apply to the second. On a genuinely stable relationship
# the out-of-sample number should land close to the in-sample one.
h_train = min_var_ratio(target.iloc[:750], hedge.iloc[:750])
eff_is = effectiveness(target.iloc[:750], hedge.iloc[:750], h_train)
eff_oos = effectiveness(target.iloc[750:], hedge.iloc[750:], h_train)
check(f"a frozen training ratio still works out of sample "
      f"({eff_is:.4f} in sample, {eff_oos:.4f} out)", eff_oos > eff_is - 0.02)
check("the out-of-sample number is not mechanically the in-sample optimum",
      eff_oos <= effectiveness(target.iloc[750:], hedge.iloc[750:],
                               min_var_ratio(target.iloc[750:], hedge.iloc[750:])) + 1e-12)

print("\nSCREENING\n")

panel = pd.DataFrame({"a": target, "b": hedge,
                      "c": rng.normal(0, 100, N_DAYS),
                      "d": 0.5 * hedge + rng.normal(0, 500, N_DAYS)})
ranked = screen(panel)
check("the screen returns every unordered pair once",
      len(ranked) == 6 and len(set(map(frozenset, zip(ranked["a"], ranked["b"])))) == 6)
check("the planted pair ranks first",
      set(ranked.iloc[0][["a", "b"]]) == {"a", "b"})
check("the R-squared matrix agrees with the pairwise screen",
      np.isclose(r2_matrix(panel).loc["a", "b"], ranked.iloc[0]["r2"]))
check("R-squared discards the sign but the screen keeps it",
      (ranked["r2"] >= 0).all() and "corr" in ranked.columns)

flipped = panel.copy()
flipped["b"] = -flipped["b"]
check("flipping the sign of a series leaves R-squared alone and flips the correlation",
      np.isclose(screen(flipped).iloc[0]["r2"], ranked.iloc[0]["r2"])
      and screen(flipped).iloc[0]["corr"] == -ranked.iloc[0]["corr"])

scaled = panel.copy()
scaled["a"] = scaled["a"] * 1000
check("rescaling a series leaves R-squared alone but changes the slope",
      np.isclose(screen(scaled).iloc[0]["r2"], ranked.iloc[0]["r2"])
      and not np.isclose(ols_hac(scaled["a"], scaled["b"])["beta"], fit["beta"]))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
