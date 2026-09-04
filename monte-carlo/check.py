"""Offline checks for mc.py. No network: every truth here is either a closed form
or an identity the simulated prices must satisfy exactly."""
import numpy as np

from mc import (STEPS_PER_YEAR, bs_greeks, bs_price, geometric_asian_price, greeks_fd,
                greeks_lr, greeks_pathwise, implied_vol, n_steps_for, price_option, price_qmc)

S0, K, DAYS, R, SIGMA = 100.0, 100.0, 63, 0.04, 0.25
T = DAYS / STEPS_PER_YEAR
N = 200_000

checks = []


def check(name, ok):
    checks.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)


check("step rule: half-days under 30 days to expiry", n_steps_for(20) == 40)
check("step rule: whole days at 30 and above", n_steps_for(63) == 63)

bs_call = bs_price(S0, K, T, R, SIGMA, "call")
bs_put = bs_price(S0, K, T, R, SIGMA, "put")
check(f"closed-form put-call parity (call {bs_call:.4f}, put {bs_put:.4f})",
      abs(bs_call - bs_put - (S0 - K * np.exp(-R * T))) < 1e-10)
check("implied vol round-trips the price it was backed out of",
      abs(implied_vol(bs_call, S0, K, T, R) - SIGMA) < 1e-6)

call = price_option(S0, K, DAYS, R, SIGMA, "call", n_paths=N, seed=1)
put = price_option(S0, K, DAYS, R, SIGMA, "put", n_paths=N, seed=1)
check(f"MC call within 3 SE of Black-Scholes ({call['price']:.4f} vs {bs_call:.4f}, "
      f"SE {call['se']:.4f})", abs(call["price"] - bs_call) < 3 * call["se"])
check(f"MC put within 3 SE of Black-Scholes ({put['price']:.4f} vs {bs_put:.4f}, "
      f"SE {put['se']:.4f})", abs(put["price"] - bs_put) < 3 * put["se"])
check("Black-Scholes lands inside the MC 95% interval",
      call["ci"][0] < bs_call < call["ci"][1])

parity_gap = (call["price"] - put["price"]) - (S0 - K * np.exp(-R * T))
check(f"put-call parity in the simulated prices (gap {parity_gap:+.5f})",
      abs(parity_gap) < 3 * np.sqrt(call["se"] ** 2 + put["se"] ** 2))

m = n_steps_for(DAYS)
geo_closed = geometric_asian_price(S0, K, T, R, SIGMA, m, "call")
geo_mc = price_option(S0, K, DAYS, R, SIGMA, option_style="asian",
                      asian_average="geometric", n_paths=N, seed=2)
check(f"geometric-Asian closed form matches its own MC ({geo_closed:.4f} vs "
      f"{geo_mc['price']:.4f}, SE {geo_mc['se']:.4f})",
      abs(geo_mc["price"] - geo_closed) < 3 * geo_mc["se"])
check(f"geometric Asian sits below the European call ({geo_closed:.4f} < {bs_call:.4f})",
      geo_closed < bs_call)

arith = price_option(S0, K, DAYS, R, SIGMA, option_style="asian", n_paths=N, seed=2)
check(f"arithmetic Asian above geometric ({arith['price']:.4f} > {geo_mc['price']:.4f})",
      arith["price"] > geo_mc["price"])

anti = price_option(S0, K, DAYS, R, SIGMA, "call", n_paths=N, method="antithetic", seed=3)
vrf_anti = (call["se"] / anti["se"]) ** 2
check(f"antithetic cuts European variance ({vrf_anti:.1f}x)", vrf_anti > 1.5)
check(f"antithetic price still agrees with Black-Scholes ({anti['price']:.4f})",
      abs(anti["price"] - bs_call) < 3 * anti["se"])

arith_cv = price_option(S0, K, DAYS, R, SIGMA, option_style="asian",
                        n_paths=N, control=True, seed=2)
vrf_cv = (arith["se"] / arith_cv["se"]) ** 2
check(f"geometric-Asian control cuts arithmetic-Asian variance ({vrf_cv:.0f}x)", vrf_cv > 10)
check(f"control-variate price agrees with the plain one "
      f"({arith_cv['price']:.4f} vs {arith['price']:.4f})",
      abs(arith_cv["price"] - arith["price"]) < 3 * arith["se"])

sob = price_qmc(S0, K, DAYS, R, SIGMA, n_paths=131_072, n_scrambles=16, seed=4)
check(f"scrambled Sobol cuts European variance ({(call['se'] / sob['se']) ** 2:.0f}x at "
      f"{sob['n_paths']:,} paths)", sob["se"] < call["se"])
check(f"Sobol price agrees with Black-Scholes ({sob['price']:.4f})",
      abs(sob["price"] - bs_call) < 3 * sob["se"])

B = 120.0
ko = price_option(S0, K, DAYS, R, SIGMA, "call", "barrier", n_paths=N,
                  barrier=B, barrier_type="up-and-out", seed=5)
ki = price_option(S0, K, DAYS, R, SIGMA, "call", "barrier", n_paths=N,
                  barrier=B, barrier_type="up-and-in", seed=5)
vanilla = price_option(S0, K, DAYS, R, SIGMA, "call", n_paths=N, seed=5)
check(f"in-out parity on shared paths (KO {ko['price']:.4f} + KI {ki['price']:.4f} "
      f"= {vanilla['price']:.4f})", abs(ko["price"] + ki["price"] - vanilla["price"]) < 1e-10)
check("knock-out is cheaper than the vanilla it is carved from", ko["price"] < vanilla["price"])
check("a barrier far above spot leaves the knock-out nearly whole",
      abs(price_option(S0, K, DAYS, R, SIGMA, "call", "barrier", n_paths=N, barrier=1e6,
                       barrier_type="up-and-out", seed=5)["price"] - vanilla["price"]) < 1e-10)

exact = bs_greeks(S0, K, T, R, SIGMA, "call")
pw = greeks_pathwise(S0, K, DAYS, R, SIGMA, n_paths=N, seed=6)
lr = greeks_lr(S0, K, DAYS, R, SIGMA, n_paths=N, seed=6)
fd = greeks_fd(S0, K, DAYS, R, SIGMA, n_paths=N, seed=6)
check(f"pathwise delta within 3 SE of closed form ({pw['delta']:.4f} vs {exact['delta']:.4f})",
      abs(pw["delta"] - exact["delta"]) < 3 * pw["delta_se"])
check(f"pathwise vega within 3 SE of closed form ({pw['vega']:.4f} vs {exact['vega']:.4f})",
      abs(pw["vega"] - exact["vega"]) < 3 * pw["vega_se"])
check(f"likelihood-ratio delta within 3 SE of closed form ({lr['delta']:.4f})",
      abs(lr["delta"] - exact["delta"]) < 3 * lr["delta_se"])
check(f"pathwise delta is the quieter estimator (SE {pw['delta_se']:.5f} vs LR "
      f"{lr['delta_se']:.5f})", pw["delta_se"] < lr["delta_se"])
check(f"common-random-number finite difference lands on delta ({fd['delta']:.4f})",
      abs(fd["delta"] - exact["delta"]) < 0.01)
check(f"common-random-number finite difference lands on vega ({fd['vega']:.4f})",
      abs(fd["vega"] - exact["vega"]) < 0.01)

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
raise SystemExit(0 if checks and all(checks) else 1)
