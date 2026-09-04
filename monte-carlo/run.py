"""Full pipeline: convergence to Black-Scholes, variance reduction, Greeks, and a live SPY quote."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mc import (STEPS_PER_YEAR, bs_greeks, bs_price, geometric_asian_price, greeks_fd,
                greeks_lr, greeks_pathwise, implied_vol, n_steps_for, path_stats,
                payoff, price_option, price_qmc)

S0, K, DAYS, R, SIGMA = 100.0, 100.0, 63, 0.04, 0.25
T = DAYS / STEPS_PER_YEAR
M = n_steps_for(DAYS)
GRID = [4096, 16384, 65536, 262144, 1_048_576]
TARGET_SE = 0.005

print(f"S0={S0} K={K} expiry={DAYS} trading days (T={T:.4f}y) r={R} sigma={SIGMA}")
print(f"discretisation: {M} steps of {DAYS / M} day\n")

bs_call = bs_price(S0, K, T, R, SIGMA, "call")
bs_put = bs_price(S0, K, T, R, SIGMA, "put")
geo_closed = geometric_asian_price(S0, K, T, R, SIGMA, M, "call")

print("European call, plain MC against the closed form:")
print(f"{'paths':>10} {'MC price':>10} {'SE':>8} {'95% CI':>20} {'error':>9} {'err/SE':>7}")
for n in GRID:
    out = price_option(S0, K, DAYS, R, SIGMA, "call", n_paths=n, seed=11)
    err = out["price"] - bs_call
    print(f"{n:>10,} {out['price']:>10.4f} {out['se']:>8.4f} "
          f"[{out['ci'][0]:8.4f},{out['ci'][1]:8.4f}] {err:>+9.4f} {err / out['se']:>7.2f}")
print(f"{'Black-Scholes':>10} {bs_call:>10.4f}\n")


def sweep(pricer, label, **kw):
    """Run one method across the path grid and return prices, standard errors and timings."""
    rows = []
    for n in GRID:
        t0 = time.perf_counter()
        out = pricer(n, **kw)
        rows.append((n, out["price"], out["se"], time.perf_counter() - t0))
    print(f"  {label:<24} done")
    return np.array(rows)


def euro(n, **kw):
    return price_option(S0, K, DAYS, R, SIGMA, "call", n_paths=n, seed=11, **kw)


def euro_qmc(n):
    return price_qmc(S0, K, DAYS, R, SIGMA, n_paths=n, n_scrambles=16, seed=11)


def asian(n, **kw):
    return price_option(S0, K, DAYS, R, SIGMA, option_style="asian", n_paths=n, seed=12, **kw)


def asian_qmc(n):
    return price_qmc(S0, K, DAYS, R, SIGMA, option_style="asian",
                     n_paths=n, n_scrambles=16, seed=12)


print("Running the variance-reduction sweep:")
euro_runs = {
    "plain": sweep(euro, "European plain"),
    "antithetic": sweep(euro, "European antithetic", method="antithetic"),
    "control (S_T)": sweep(euro, "European control", control=True),
    "sobol (scrambled)": sweep(euro_qmc, "European Sobol"),
}
asian_runs = {
    "plain": sweep(asian, "Asian plain"),
    "antithetic": sweep(asian, "Asian antithetic", method="antithetic"),
    "control (geo Asian)": sweep(asian, "Asian control", control=True),
    "sobol (scrambled)": sweep(asian_qmc, "Asian Sobol"),
}


def report(runs, title, reference=None):
    """Print the variance-reduction table at the largest path count on the grid."""
    base = runs["plain"]
    n, _, se_base, t_base = base[-1]
    print(f"\n{title} at {int(n):,} paths (plain SE {se_base:.5f}, {t_base:.2f}s)")
    print(f"{'method':<22} {'price':>9} {'SE':>9} {'var cut':>9} {'time':>7} "
          f"{'speedup':>8} {'rate':>6} {f'paths @ SE {TARGET_SE}':>20}")
    for name, rows in runs.items():
        _, price, se, secs = rows[-1]
        vrf = (se_base / se) ** 2
        speedup = vrf * (t_base / secs)
        rate = -np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 2]), 1)[0]
        need = n * (se / TARGET_SE) ** (1 / rate)
        print(f"{name:<22} {price:>9.4f} {se:>9.5f} {vrf:>8.1f}x {secs:>6.2f}s "
              f"{speedup:>7.1f}x {rate:>6.2f} {need:>20,.0f}")
    if reference is not None:
        print(f"{'closed form':<22} {reference:>9.4f}")


report(euro_runs, "European call variance reduction", bs_call)
report(asian_runs, "Arithmetic Asian call variance reduction")
print("\n`rate` is the fitted slope of log(SE) against log(paths); 0.5 is the plain "
      "Monte Carlo law.\nThe paths column extrapolates with each method's own fitted rate.")

print("\nPath-dependent prices (1,000,000 paths, antithetic):")
big = dict(n_paths=1_000_000, method="antithetic", seed=13)
arith = price_option(S0, K, DAYS, R, SIGMA, option_style="asian", **big)
arith_cv = price_option(S0, K, DAYS, R, SIGMA, option_style="asian", control=True,
                        n_paths=1_000_000, seed=13)
geo_mc = price_option(S0, K, DAYS, R, SIGMA, option_style="asian",
                      asian_average="geometric", **big)
print(f"  European call                {bs_call:8.4f}  (closed form)")
print(f"  arithmetic Asian call        {arith['price']:8.4f} +/- {arith['se']:.4f}")
print(f"  arithmetic Asian, controlled {arith_cv['price']:8.4f} +/- {arith_cv['se']:.4f}")
print(f"  geometric Asian call         {geo_mc['price']:8.4f} +/- {geo_mc['se']:.4f}  "
      f"(closed form {geo_closed:.4f})")

print("\nUp-and-out call, barrier monitored at every simulated step:")
vanilla = price_option(S0, K, DAYS, R, SIGMA, "call", **big)
for B in (105, 110, 115, 120, 130, 150):
    ko = price_option(S0, K, DAYS, R, SIGMA, "call", "barrier", barrier=B,
                      barrier_type="up-and-out", **big)
    ki = price_option(S0, K, DAYS, R, SIGMA, "call", "barrier", barrier=B,
                      barrier_type="up-and-in", **big)
    print(f"  B={B:>3}  knock-out {ko['price']:7.4f} +/- {ko['se']:.4f}   "
          f"knock-in {ki['price']:7.4f}   sum {ko['price'] + ki['price']:7.4f} "
          f"vs vanilla {vanilla['price']:.4f}")

print("\nStep-rule sensitivity on a 20-day option (half-day vs forced whole-day steps):")
short_days = 20
for label, steps in (("half-day (rule)", n_steps_for(short_days)), ("whole-day", short_days)):
    T_s = short_days / STEPS_PER_YEAR
    st = path_stats(S0, T_s, R, SIGMA, steps, 1_000_000, "antithetic", seed=14)
    ko = np.exp(-R * T_s) * payoff(st, K, "call", "barrier", 110, "up-and-out")
    ko = ko.reshape(-1, 2).mean(axis=1)
    print(f"  {label:<16} {steps:>3} steps   up-and-out B=110: {ko.mean():.4f} "
          f"+/- {ko.std(ddof=1) / np.sqrt(len(ko)):.4f}")

print("\nGreeks for the European call (200,000 paths each):")
exact = bs_greeks(S0, K, T, R, SIGMA, "call")
rows = []
for name, fn in (("pathwise", greeks_pathwise), ("likelihood ratio", greeks_lr)):
    g = fn(S0, K, DAYS, R, SIGMA, n_paths=200_000, seed=21)
    rows.append((name, g["delta"], g["delta_se"], g["vega"], g["vega_se"]))
fd = greeks_fd(S0, K, DAYS, R, SIGMA, n_paths=200_000, seed=21)
fd_indep = {"delta": (price_option(S0 * 1.01, K, DAYS, R, SIGMA, n_paths=200_000, seed=31)["price"]
                      - price_option(S0 * 0.99, K, DAYS, R, SIGMA, n_paths=200_000, seed=32)["price"])
                     / (2 * 0.01 * S0)}
print(f"{'estimator':<22} {'delta':>9} {'SE':>9} {'vega':>9} {'SE':>9}")
for name, d, dse, v, vse in rows:
    print(f"{name:<22} {d:>9.4f} {dse:>9.5f} {v:>9.4f} {vse:>9.5f}")
print(f"{'finite diff (CRN)':<22} {fd['delta']:>9.4f} {'-':>9} {fd['vega']:>9.4f} {'-':>9}")
print(f"{'finite diff (indep)':<22} {fd_indep['delta']:>9.4f} {'-':>9} {'-':>9} {'-':>9}")
print(f"{'Black-Scholes':<22} {exact['delta']:>9.4f} {'-':>9} {exact['vega']:>9.4f} {'-':>9}")

seeds = range(40, 60)
spread = {
    "pathwise": [greeks_pathwise(S0, K, DAYS, R, SIGMA, n_paths=50_000, seed=s)["delta"] for s in seeds],
    "likelihood ratio": [greeks_lr(S0, K, DAYS, R, SIGMA, n_paths=50_000, seed=s)["delta"] for s in seeds],
    "finite diff (CRN)": [greeks_fd(S0, K, DAYS, R, SIGMA, n_paths=50_000, seed=s)["delta"] for s in seeds],
}
print("\nDelta across 20 independent seeds at 50,000 paths (spread = estimator noise):")
for name, vals in spread.items():
    print(f"  {name:<20} mean {np.mean(vals):.4f}  sd {np.std(vals, ddof=1):.5f}  "
          f"bias {np.mean(vals) - exact['delta']:+.5f}")

print("\nLive market check")
try:
    from data import atm_call, realised_vol, risk_free_rate
    quote = atm_call("SPY")
    r_mkt = risk_free_rate()
    rv, rv_days = realised_vol("SPY")
    market = True
except Exception as exc:
    print(f"market data unavailable ({type(exc).__name__}: {exc}); skipping this section")
    market = False

if market:
    T_m = quote["trading_days"] / STEPS_PER_YEAR
    args = (quote["spot"], quote["strike"], T_m, r_mkt)
    iv = {side: implied_vol(quote[side], *args) for side in ("bid", "mid", "ask")}
    print(f"{quote['ticker']} spot {quote['spot']:.2f}, {quote['expiry']} call "
          f"K={quote['strike']:.0f}, {quote['calendar_days']} calendar / "
          f"{quote['trading_days']} trading days")
    print(f"  market  bid {quote['bid']:.2f}  mid {quote['mid']:.2f}  ask {quote['ask']:.2f}  "
          f"(vol {quote['volume']:.0f}, OI {quote['open_interest']:.0f})")
    print(f"  r = {r_mkt:.4f} from ^IRX")
    print(f"  implied vol   bid {iv['bid']:.4f}  mid {iv['mid']:.4f}  ask {iv['ask']:.4f}  "
          f"(Yahoo reports {quote['yahoo_iv']:.4f})")
    print(f"  realised vol over the last {rv_days} days: {rv:.4f}")

    mc_iv = price_option(quote["spot"], quote["strike"], quote["trading_days"], r_mkt, iv["mid"],
                         n_paths=2_000_000, method="antithetic", control=True, seed=99)
    mc_rv = price_option(quote["spot"], quote["strike"], quote["trading_days"], r_mkt, rv,
                         n_paths=2_000_000, method="antithetic", control=True, seed=99)
    print(f"  MC at the implied vol   {mc_iv['price']:.4f} +/- {mc_iv['se']:.4f}   "
          f"market mid {quote['mid']:.2f}   gap {mc_iv['price'] - quote['mid']:+.4f}")
    print(f"  MC at the realised vol  {mc_rv['price']:.4f} +/- {mc_rv['se']:.4f}   "
          f"gap {mc_rv['price'] - quote['mid']:+.4f}  "
          f"(variance risk premium: implied {iv['mid']:.4f} vs realised {rv:.4f})")
    print(f"  inside the bid-ask: {quote['bid'] <= mc_iv['price'] <= quote['ask']}")

    B_mkt = round(quote["spot"] * 1.03, 0)
    ko_mkt = price_option(quote["spot"], quote["strike"], quote["trading_days"], r_mkt, iv["mid"],
                          "call", "barrier", barrier=B_mkt, barrier_type="up-and-out",
                          n_paths=2_000_000, method="antithetic", seed=99)
    asian_mkt = price_option(quote["spot"], quote["strike"], quote["trading_days"], r_mkt,
                             iv["mid"], option_style="asian", n_paths=2_000_000,
                             control=True, seed=99)
    print(f"  same vol, no closed form: up-and-out B={B_mkt:.0f} {ko_mkt['price']:.4f} "
          f"+/- {ko_mkt['se']:.4f};  arithmetic Asian {asian_mkt['price']:.4f} "
          f"+/- {asian_mkt['se']:.6f}")

    q = 0.011
    print(f"  where the gap to Yahoo's {quote['yahoo_iv']:.4f} could come from: "
          f"a {q:.1%} dividend yield moves the mid implied vol to "
          f"{implied_vol(quote['mid'], quote['spot'] * np.exp(-q * T_m), quote['strike'], T_m, r_mkt):.4f}, "
          f"calendar-day time to "
          f"{implied_vol(quote['mid'], quote['spot'], quote['strike'], quote['calendar_days'] / 365, r_mkt):.4f}. "
          f"Neither closes it: Yahoo's number implies "
          f"{bs_price(quote['spot'], quote['strike'], T_m, r_mkt, quote['yahoo_iv']):.2f}, "
          f"above the {quote['ask']:.2f} ask.")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
panels = [(euro_runs, "European call", bs_call, "Black-Scholes"),
          (asian_runs, "Arithmetic Asian call", arith_cv["price"], "control variate, 1M paths")]
for ax, (runs, title, ref, ref_label) in zip(axes, panels):
    for name, rows in runs.items():
        ax.plot(rows[:, 0], rows[:, 1], marker="o", ms=3, lw=1, label=name)
        ax.fill_between(rows[:, 0], rows[:, 1] - 1.96 * rows[:, 2],
                        rows[:, 1] + 1.96 * rows[:, 2], alpha=0.15)
    ax.axhline(ref, color="k", ls="--", lw=1, label=ref_label)
    ax.set_xscale("log")
    ax.set_xlabel("paths")
    ax.set_ylabel("price")
    ax.set_title(f"{title}: convergence with 95% bands")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("reports/convergence.png", dpi=120)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (runs, title) in zip(axes, [(euro_runs, "European call"),
                                    (asian_runs, "Arithmetic Asian call")]):
    for name, rows in runs.items():
        ax.plot(rows[:, 0], rows[:, 2], marker="o", ms=3, lw=1, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("paths")
    ax.set_ylabel("standard error")
    ax.set_title(f"{title}: standard error vs work")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("reports/standard_error.png", dpi=120)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
paths = np.exp(np.log(S0) + np.cumsum((R - 0.5 * SIGMA ** 2) * (T / M) + SIGMA
               * np.sqrt(T / M) * np.random.default_rng(7).standard_normal((60, M)), axis=1))
axes[0].plot(np.arange(1, M + 1), paths.T, lw=0.5, alpha=0.6)
axes[0].axhline(K, color="k", ls="--", lw=1, label="strike")
axes[0].axhline(120, color="r", ls=":", lw=1, label="barrier 120")
axes[0].set_xlabel("trading day")
axes[0].set_ylabel("price")
axes[0].set_title("60 simulated GBM paths")
axes[0].legend(fontsize=8)

big_st = path_stats(S0, T, R, SIGMA, M, 200_000, "plain", seed=8)
axes[1].hist(np.exp(-R * T) * payoff(big_st, K, "call"), bins=80, alpha=0.6,
             label="European call", log=True)
axes[1].hist(np.exp(-R * T) * payoff(big_st, K, "call", "asian"), bins=80, alpha=0.6,
             label="arithmetic Asian", log=True)
axes[1].set_xlabel("discounted payoff")
axes[1].set_ylabel("paths (log scale)")
axes[1].set_title("Payoff distributions: most paths pay nothing")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("reports/paths_and_payoffs.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.boxplot(list(spread.values()), tick_labels=list(spread))
ax.axhline(exact["delta"], color="r", ls="--", lw=1, label="Black-Scholes delta")
ax.set_ylabel("delta")
ax.set_title("Delta estimator noise, 20 seeds at 50,000 paths")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("reports/greeks.png", dpi=120)

print("\nCharts saved to reports/: convergence.png, standard_error.png, "
      "paths_and_payoffs.png, greeks.png")
