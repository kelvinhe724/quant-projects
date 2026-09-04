"""Offline checks on hand-built curves.

Carry and rolldown arithmetic against numbers worked by hand, zero rolldown
on a flat curve, the position rule on a planted upward slope, the realised
return reconciling to the expectation on an unchanged curve, the ETF map
carrying the tenor's duration, the regression recovering a planted slope,
and the engine strategy trading only at month ends from the as-of view.

Run: ../.venv/bin/python3 check.py
"""
import numpy as np
import pandas as pd

import carry
import data
from framework.engine import Bars, run, synthetic

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def curve_from(fn, tenors=data.TENORS):
    """A fitted curve from exact NS-shaped points, so fit and truth agree."""
    return carry.fit_curve(tenors, fn(tenors))


print("BOND ARITHMETIC\n")

check("a par bond prices at 1 at its own yield",
      np.isclose(carry.bond_price(5.0, 5.0, 10), 1.0) and np.isclose(carry.bond_price(0.0, 0.0, 7), 1.0))
check("modified duration of a 10y par bond at 5% is (1 - 1.05^-10) / 0.05 = 7.72",
      np.isclose(carry.par_duration(5.0, 10), (1 - 1.05 ** -10) / 0.05)
      and np.isclose(carry.par_duration(5.0, 10), 7.7217, atol=1e-3))
check("duration of a par bond at zero yield is its maturity", np.isclose(carry.par_duration(0.0, 20), 20.0))
check("a 1% yield rise on a 10y par bond at 5% costs about duration percent",
      np.isclose(carry.bond_price(5.0, 6.0, 10) - 1, -0.0736, atol=5e-4))

print("\nCARRY AND ROLLDOWN\n")

# planted NS curve: level 5, slope -3 (3m near 2, long end near 5), curvature 1, lambda 2
planted = lambda t: carry.ns.ns_curve(t, 5.0, -3.0, 1.0, 2.0)
c = curve_from(planted)
check(f"the fit recovers the planted curve (rmse {c.rmse * 100:.3f} bp)", c.rmse < 1e-3)

funding = float(planted(0.25))
tab = carry.expected_excess(c, funding, [2, 5, 10, 30])
check("carry is the tenor yield minus the funding rate",
      np.allclose(tab["carry"], planted(np.array([2, 5, 10, 30])) - funding, atol=1e-3))
by_hand = planted(10.0) - planted(10.0 - 1 / 12)
check(f"10y rolldown is y(10) - y(10 - 1/12) = {by_hand * 100:.2f} bp",
      np.isclose(tab.loc[10, "rolldown_bp"], by_hand * 100, atol=0.1))
check("rolldown in return terms is duration x yield rolled x 12",
      np.isclose(tab.loc[10, "rolldown"], carry.par_duration(planted(10.0), 10) * by_hand * 12, atol=1e-3))
check("expected excess return is carry plus rolldown",
      np.allclose(tab["expected"], tab["carry"] + tab["rolldown"]))
check("rolldown is positive everywhere on an upward-sloping curve", (tab["rolldown"] > 0).all())

flat_fn = lambda t: np.full_like(np.asarray(t, float), 4.0)
check("rolldown is exactly zero on a flat curve",
      all(carry.rolldown_yield(flat_fn, T) == 0.0 for T in (2, 5, 10, 30)))
flat = curve_from(flat_fn)
flat_tab = carry.expected_excess(flat, 4.0, [2, 5, 10, 30])
check(f"and within a hundredth of a basis point on the fit to one ({flat_tab['rolldown_bp'].abs().max():.4f} bp)",
      np.allclose(flat_tab["rolldown_bp"], 0.0, atol=1e-2))
check("carry is zero on a flat curve funded at the same rate", np.allclose(flat_tab["carry"], 0.0, atol=1e-9))

inverted = curve_from(lambda t: carry.ns.ns_curve(t, 3.0, 2.5, 0.0, 2.0))
inv_tab = carry.expected_excess(inverted, float(inverted(0.25)), [2, 5, 10, 30])
check("rolldown is negative on an inverted curve", (inv_tab["rolldown"] < 0).all())

print("\nREALISED RETURN\n")

for T in (2, 10, 30):
    got = carry.realised_excess(c, c, funding, T, 1 / 12)
    check(f"on an unchanged curve the realised {T}y excess return is the expected one "
          f"({got:.3f} vs {tab.loc[T, 'expected']:.3f}, convexity apart)",
          abs(got - tab.loc[T, "expected"]) < 0.05 * max(1.0, abs(tab.loc[T, "expected"])))

up = curve_from(lambda t: planted(t) + 1.0)
check("a parallel 1% rise loses money on every tenor, more the longer the bond",
      (lambda r: (r < 0).all() and np.all(np.diff(r) < 0))(
          np.array([carry.realised_excess(c, up, funding, T, 1 / 12) for T in (2, 10, 30)])))
check("the ffill-proof interpolation returns the observed yield when nothing is rolled",
      np.isclose(carry.interpolate(c, 10.0, 0.0), c.observed[10.0]))

print("\nPOSITION RULE\n")

w = carry.weights(tab["expected"], tab["duration"], n_leg=2, target=7.0)
check("on the planted upward slope the rule goes long the two longest tenors",
      set(w[w > 0].index) == {10.0, 30.0} and (w[[2.0, 5.0]] == 0).all())
check("the long book carries exactly the target duration",
      np.isclose(float((w * tab["duration"]).sum()), 7.0))
check("the rule holds nothing short", (w >= 0).all())

w_inv = carry.weights(inv_tab["expected"], inv_tab["duration"], n_leg=2, target=7.0)
check("with every expected excess return negative the rule is flat", (w_inv == 0).all())

ls = carry.weights(tab["expected"], tab["duration"], n_leg=1, target=7.0, long_short=True)
check("the long/short variant is duration neutral",
      np.isclose(float((ls * tab["duration"]).sum()), 0.0, atol=1e-9) and ls[30.0] > 0 > ls[2.0])

partial = tab["expected"].copy()
partial[30.0] = np.nan
w_part = carry.weights(partial, tab["duration"], n_leg=2, target=7.0)
check("a tenor with no quote is never held", w_part[30.0] == 0 and set(w_part[w_part > 0].index) == {5.0, 10.0})

print("\nETF MAP\n")

etf_w = carry.etf_weights(w, data.ETF_TENOR, data.ETF_DURATION, tab["duration"])
for etf, T in data.ETF_TENOR.items():
    if w.get(T, 0) > 0:
        held = etf_w[etf] * data.ETF_DURATION[etf]
        want = w[T] * tab.loc[T, "duration"]
        check(f"{etf} carries the {T:g}y position's duration ({held:.2f} vs {want:.2f} years)",
              np.isclose(held, want))
check("funds whose tenor is not held get zero", etf_w["SHY"] == 0 and etf_w["IEI"] == 0)

TOL = 0.25
for etf, T in data.ETF_TENOR.items():
    par = float(carry.par_duration(4.0, T))
    gap = abs(data.ETF_DURATION[etf] - par) / par
    check(f"{etf} duration {data.ETF_DURATION[etf]} is within {TOL:.0%} of a {T:g}y par bond's {par:.1f} "
          f"(gap {gap:.0%})", gap <= TOL)

rng = np.random.default_rng(3)
days = pd.bdate_range("2015-01-01", periods=1500)
y_path = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.03, len(days))), index=days)
fund_ret = -7.5 * y_path.diff() / 100 + rng.normal(0, 0.001, len(days))
D_hat, n = carry.empirical_duration(fund_ret, y_path)
check(f"empirical duration recovers a planted 7.5 from fund returns ({D_hat:.2f})", abs(D_hat - 7.5) < 0.3)

print("\nPREDICTIVE REGRESSION\n")

months = pd.date_range("1970-01-31", periods=600, freq="ME")
expected = pd.DataFrame(rng.normal(1.5, 1.5, (600, 3)), index=months, columns=[2.0, 10.0, 30.0])
realised = 0.5 + 1.0 * expected + rng.normal(0, 4, (600, 3))
reg = carry.predictive_regression(expected, realised)
check(f"the regression recovers a planted slope of 1 on every tenor "
      f"({', '.join(f'{v:.2f}' for v in reg.loc[['2y', '10y', '30y'], 'slope'])})",
      (abs(reg.loc[["2y", "10y", "30y"], "slope"] - 1.0) < 0.35).all())
check(f"and rejects zero (pooled t {reg.loc['pooled', 't']:.1f})", reg.loc["pooled", "t"] > 3)
shuffled = realised.sample(frac=1, random_state=1).set_index(months)
reg0 = carry.predictive_regression(expected, shuffled)
check(f"shuffled realisations give a slope near zero (pooled {reg0.loc['pooled', 'slope']:.2f}, "
      f"t {reg0.loc['pooled', 't']:.2f})", abs(reg0.loc["pooled", "t"]) < 2.5)

print("\nMONTHLY PANEL\n")

idx = pd.bdate_range("2000-01-03", periods=520)
wide = pd.DataFrame({t: planted(t) + 0.2 * np.sin(np.arange(len(idx)) / 40) for t in data.TENORS}, index=idx)
panel = carry.monthly_panel(wide, pd.Series(wide[0.25].to_numpy(), index=idx), [2, 10, 30])
check("the panel has one row per month end", len(panel["expected"]) == len(data.month_ends(idx)))
check("a sample ending on the 1st does not get a one-day month",
      data.month_ends(pd.bdate_range("2000-01-03", "2000-03-01"))[-1] == pd.Timestamp("2000-02-29"))
check("the last month has an expectation but no realised return",
      panel["realised"].iloc[-1].isna().all() and panel["expected"].iloc[-1].notna().all())
check("realised returns line up with the month they were formed in, not the month after",
      np.isclose(panel["realised"].iloc[0, 1],
                 carry.realised_excess(carry.fit_curve(data.TENORS, wide.loc[panel["expected"].index[0]].to_numpy()),
                                       carry.fit_curve(data.TENORS, wide.loc[panel["expected"].index[1]].to_numpy()),
                                       float(wide.loc[panel["expected"].index[0], 0.25]), 10,
                                       panel["dt"].iloc[0])))
r, held = carry.sleeve_returns(panel)
check("sleeve returns are the held weights times realised returns, scaled to the month",
      np.isclose(r.iloc[5], float((held.iloc[5] * panel["realised"].iloc[5]).sum()) / 100 * panel["dt"].iloc[5]))

print("\nENGINE STRATEGY\n")

bars0 = synthetic(n_days=len(idx), instruments=tuple(data.ETF_TENOR), vol=0.08, start=idx[0], seed=1)
assert bars0.calendar.equals(idx)
series = {f"y{t:g}": wide[t].reindex(bars0.calendar) for t in data.TENORS}
series[data.FUNDING] = wide[0.25].reindex(bars0.calendar)
bars = Bars({f: bars0.field(f) for f in ("open", "high", "low", "close", "volume")}, series)
strat = carry.RatesCarryETF()
sent = {d: strat.on_bar(d, bars.upto(d)) for d in bars.calendar}
month_ends = set(data.month_ends(bars.calendar))
check("targets are sent at month ends and never in between",
      all((v is not None) == (d in month_ends) for d, v in sent.items()))
last = sent[max(month_ends)]
fund_of = {T: etf for etf, T in data.ETF_TENOR.items()}
by_hand = carry.expected_excess(carry.fit_curve(data.TENORS, wide.loc[max(month_ends)].to_numpy()),
                                float(wide.loc[max(month_ends), 0.25]), carry.SLEEVE_TENORS)["expected"]
top2 = {fund_of[T] for T in by_hand.nlargest(2).index}
check(f"a month-end target is long the two funds whose tenors pay the most ({', '.join(sorted(top2))})",
      set(k for k, v in last.items() if v > 0) == top2 and all(v >= 0 for v in last.values()))

gap = {k: v.copy() for k, v in series.items()}
gap["y30"].iloc[:] = np.inf
strat_gap = carry.RatesCarryETF()
sent_gap = strat_gap.on_bar(max(month_ends), Bars(bars._frames, gap).upto(max(month_ends)))
top2_gap = {fund_of[T] for T in by_hand.drop(30.0).nlargest(2).index}
check(f"a tenor flagged as unpublished (inf) is not held and the next tenor takes its place "
      f"({', '.join(sorted(top2_gap))})",
      sent_gap.get("TLT", 0) == 0 and set(k for k, v in sent_gap.items() if v > 0) == top2_gap
      and all(np.isfinite(v) for v in sent_gap.values()))

res = run(carry.RatesCarryETF(), bars)
trade_days = set(res.trades["date"])
check("the engine fills each month-end target on the following session",
      all(bars.calendar[bars.calendar.get_loc(d) - 1] in month_ends for d in trade_days))
check("nothing is held before the first fill", res.weights.iloc[0].abs().sum() == 0)

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
