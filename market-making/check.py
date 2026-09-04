"""Offline checks on a simulated market where the truth is planted.

Run: python3 check.py
"""
import math

import numpy as np

import sim
from maker import AvellanedaStoikov, Naive, Skewed

STEPS = 250
LAG = 50
SESSIONS = 250
SEED = 7

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def market(**kw):
    return sim.Market(n_steps=STEPS, informed_lag=LAG, position_limit=10 ** 6, **kw)


def run(maker, m, n=SESSIONS, seed=SEED):
    return {k: np.array([r[k] for r in sim.run_sessions(maker, m, n, seed=seed)])
            for k in sim.summarise(sim.simulate(maker, m, sim.draw_session(m, seed)))}


def tstat(x):
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


# 1. With no informed flow the naive maker is profitable in expectation.
clean = run(Naive(), market(informed_frac=0.0))
check(f"no informed flow: naive P&L is positive (mean {clean['pnl'].mean():.2f}, "
      f"t={tstat(clean['pnl']):.1f})",
      clean["pnl"].mean() > 0 and tstat(clean["pnl"]) > 5)
check(f"no informed flow: spread capture is the whole edge "
      f"({clean['spread'].mean():.2f} of {clean['pnl'].mean():.2f})",
      clean["spread"].mean() > 0.95 * clean["pnl"].mean())
check(f"no informed flow: adverse selection is statistically zero "
      f"(mean {clean['adverse_selection'].mean():.2f}, t={tstat(clean['adverse_selection']):.1f})",
      abs(tstat(clean["adverse_selection"])) < 3)
check(f"no informed flow: every fill earns the half-spread "
      f"({clean['spread'].sum() / clean['fills'].sum():.3f} vs 0.850)",
      abs(clean["spread"].sum() / clean["fills"].sum() - Naive().half_spread) < 1e-9)

# 2. Naive P&L falls monotonically as more of the flow is informed.
grid = [0.0, 0.25, 0.5, 0.75, 1.0]
sweep = [run(Naive(), market(informed_frac=phi)) for phi in grid]
curve = [r["pnl"].mean() for r in sweep]
adverse = [r["adverse_selection"].mean() for r in sweep]
print("     naive P&L by informed fraction: "
      + ", ".join(f"{phi:.2f}->{p:.2f}" for phi, p in zip(grid, curve)))
check("naive P&L is strictly decreasing in the informed fraction",
      all(a > b for a, b in zip(curve, curve[1:])))
check("adverse selection cost is strictly increasing in the informed fraction",
      all(a < b for a, b in zip(adverse, adverse[1:])))
check(f"all-informed flow loses money on every trade at the mark horizon "
      f"(spread {curve[-1]:.2f} is negative)", curve[-1] < 0)
all_informed = {mk.name: run(mk, market(informed_frac=1.0))["pnl"]
                for mk in (Skewed(), AvellanedaStoikov())}
check("every maker loses money against all-informed flow, not just naive ("
      + ", ".join(f"{k} {v.mean():.2f}" for k, v in all_informed.items()) + ")",
      all(v.mean() < 0 and tstat(v) < -3 for v in all_informed.values()))

# 3. The Avellaneda-Stoikov reservation price leans against the position.
avs = AvellanedaStoikov(gamma=0.1)
m = market()
flat = avs.quote(100.0, 0, 1.0, m)
long_q = avs.quote(100.0, 5, 1.0, m)
short_q = avs.quote(100.0, -5, 1.0, m)
mid = lambda q: (q[0] + q[1]) / 2
check(f"flat inventory quotes symmetrically around the price ({mid(flat):.4f})",
      abs(mid(flat) - 100.0) < 1e-9)
check(f"long inventory pushes the reservation price down ({mid(long_q):.3f} < 100)",
      mid(long_q) < 100.0)
check(f"short inventory pushes it up ({mid(short_q):.3f} > 100)", mid(short_q) > 100.0)
check("the shift is symmetric in the inventory sign",
      abs((mid(long_q) - 100.0) + (mid(short_q) - 100.0)) < 1e-9)
check("the shift matches q * gamma * sigma^2 * (T - t)",
      abs((100.0 - mid(long_q)) - 5 * 0.1 * m.sigma ** 2 * 1.0) < 1e-9)
check("the spread matches gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa)",
      abs((flat[1] - flat[0])
          - (0.1 * m.sigma ** 2 * 1.0 + (2 / 0.1) * math.log(1 + 0.1 / m.decay))) < 1e-9)
late = avs.quote(100.0, 5, 0.0, m)
check("both the skew and the inventory term vanish at the close",
      abs(mid(late) - 100.0) < 1e-9 and (late[1] - late[0]) < (flat[1] - flat[0]))

# 4. Inventory stays bounded under skewed quoting and wanders under naive.
m = market(informed_frac=0.3)
naive_r = run(Naive(), m)
as_r = run(avs, m)
skew_r = run(Skewed(), m)
check(f"AS holds less inventory than naive (sd {as_r['inventory_std'].mean():.2f} "
      f"vs {naive_r['inventory_std'].mean():.2f})",
      as_r["inventory_std"].mean() < 0.6 * naive_r["inventory_std"].mean())
check(f"AS peak inventory is smaller (worst |q| {as_r['max_abs_inventory'].max():.0f} "
      f"vs {naive_r['max_abs_inventory'].max():.0f})",
      as_r["max_abs_inventory"].max() < naive_r["max_abs_inventory"].max())
check(f"AS ends flatter than naive (sd of closing inventory "
      f"{as_r['end_inventory'].std():.2f} vs {naive_r['end_inventory'].std():.2f})",
      as_r["end_inventory"].std() < naive_r["end_inventory"].std())
check("skewing alone gets most of the inventory control",
      skew_r["inventory_std"].mean() < 0.6 * naive_r["inventory_std"].mean())

# Naive inventory is a random walk: its variance grows with the session length.
short_m = sim.Market(n_steps=STEPS, informed_lag=LAG, position_limit=10 ** 6,
                     informed_frac=0.3, horizon=0.25)
short_naive = run(Naive(), short_m)
check(f"naive inventory keeps growing with session length "
      f"({short_naive['max_abs_inventory'].mean():.1f} -> "
      f"{naive_r['max_abs_inventory'].mean():.1f} over 4x the horizon)",
      naive_r["max_abs_inventory"].mean() > 1.5 * short_naive["max_abs_inventory"].mean())
short_as = run(avs, short_m)
check(f"AS inventory does not ({short_as['max_abs_inventory'].mean():.1f} -> "
      f"{as_r['max_abs_inventory'].mean():.1f})",
      as_r["max_abs_inventory"].mean() < 1.5 * short_as["max_abs_inventory"].mean())

# 5. The decomposition is an identity, not an approximation.
for maker in (Naive(), Skewed(), avs):
    worst = 0.0
    for k in range(40):
        session = sim.simulate(maker, m, sim.draw_session(m, SEED + k))
        p = sim.decompose(session)
        worst = max(worst, abs(p["total"] - session.total_pnl))
        cash_mark = session.cash[-1] + session.inventory[-1] * session.price[-1]
        worst = max(worst, abs(cash_mark - session.total_pnl))
    check(f"{maker.name}: spread - adverse selection + inventory = total P&L "
          f"(max error {worst:.2e})", worst < 1e-9)

# Mechanics.
m = market(informed_frac=0.3)
check("the same seed gives the same prices and the same flow",
      np.array_equal(sim.draw_session(m, 3).price, sim.draw_session(m, 3).price)
      and not np.array_equal(sim.draw_session(m, 3).price, sim.draw_session(m, 4).price))
# Fill rates are measured with no informed flow, because informed traders decline
# fills and would contaminate the demand curve; the target is the exact per-step
# fill probability ratio, which sits a little under exp(kappa * 0.8) at this dt.
clean_m = market(informed_frac=0.0)
rates = [run(Naive(h), clean_m, n=150)["fills"].mean() for h in (0.4, 0.8, 1.2, 1.6)]
print("     fills by half-spread 0.4/0.8/1.2/1.6: " + ", ".join(f"{r:.1f}" for r in rates))
check("fill count falls as the quote moves away from the price",
      all(a > b for a, b in zip(rates, rates[1:])))
ratio = rates[0] / rates[2]
fill_prob = lambda h: -math.expm1(-sim._intensity(clean_m, h) * clean_m.dt)
target = fill_prob(0.4) / fill_prob(1.2)
check(f"the decay is exponential in distance at rate kappa "
      f"(0.4 vs 1.2 gives {ratio:.2f}, exact {target:.2f}, exp(1.5*0.8) = {math.exp(1.2):.2f})",
      abs(ratio - target) < 0.15)

capped = sim.Market(n_steps=STEPS, informed_lag=LAG, informed_frac=0.3, position_limit=3)
inv = np.concatenate([sim.simulate(Naive(), capped, sim.draw_session(capped, SEED + k)).inventory
                      for k in range(40)])
check(f"the position limit is never breached (max |q| {np.abs(inv).max():.0f} of 3)",
      np.abs(inv).max() <= 3)

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
