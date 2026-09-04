"""Offline checks on simulated books: conversion, de-vigging, a null market, the vig.

Run: python3 check.py
"""
import numpy as np

from odds import (booksum, brier, calibration, calibration_regression,
                  devig_power, devig_proportional, devig_shin, implied,
                  log_loss, overround, shin_quotes, wilson)
from strategy import (arbitrage, best_across_books, flat_bet, kelly_fraction,
                      kelly_growth, profit)

rng = np.random.default_rng(7)
checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def fair_book(n, margin, gen, alpha=(3.0, 2.5, 2.5)):
    """Draw true probabilities, simulate results, quote them with a flat margin."""
    p = gen.dirichlet(alpha, size=n)
    draw = gen.random((n, 1))
    y = (np.cumsum(p, axis=1) > draw)
    y = (y & ~np.pad(y, ((0, 0), (1, 0)), constant_values=False)[:, :-1]).astype(float)
    return p, y, 1.0 / (p * (1.0 + margin))


# odds to probability
check("evens pays 2.0 and implies 0.5", implied(2.0) == 0.5)
check("a 4.0 shot implies 0.25", implied(4.0) == 0.25)
check("implied is the exact reciprocal on a vector",
      np.allclose(implied([1.5, 3.2, 11.0]), [1 / 1.5, 1 / 3.2, 1 / 11.0]))

flat = np.array([[2.0, 4.0, 4.0]])
check(f"a book summing to exactly 1 has zero overround ({overround(flat)[0]:.1e})",
      abs(overround(flat)[0]) < 1e-12)
check("a 5% margin book shows a 5% overround",
      abs(overround(1.0 / (np.array([[0.5, 0.25, 0.25]]) * 1.05))[0] - 0.05) < 1e-12)

# de-vigging
p_true, y_true, quoted = fair_book(4000, 0.06, rng)
for name, fn in (("proportional", devig_proportional), ("power", devig_power),
                 ("shin", devig_shin)):
    s = fn(quoted).sum(axis=1)
    check(f"{name} de-vig sums to 1 (max error {np.abs(s - 1).max():.2e})",
          np.allclose(s, 1.0, atol=1e-9))

check("proportional de-vig recovers a proportionally-vigged book exactly",
      np.allclose(devig_proportional(quoted), p_true, atol=1e-12))

Z_TRUE = 0.035
shin_quoted = shin_quotes(p_true, Z_TRUE)
p_shin, z_hat = devig_shin(shin_quoted, return_z=True)
check(f"shin recovers a planted insider share z={Z_TRUE} "
      f"(median {np.median(z_hat):.4f})", abs(np.median(z_hat) - Z_TRUE) < 1e-3)
check("shin recovers the planted probabilities of a Shin book",
      np.allclose(p_shin, p_true, atol=1e-6))
check("the Shin forward map produces a positive overround",
      (overround(shin_quoted) > 0).all())

one = np.array([[1.80, 3.60, 4.50]])
fav = {n: float(fn(one)[0, 0]) for n, fn in
       (("prop", devig_proportional), ("power", devig_power), ("shin", devig_shin))}
check(f"shin shades the favourite up vs proportional "
      f"({fav['prop']:.4f} -> {fav['shin']:.4f})", fav["shin"] > fav["prop"])
check(f"power shades the favourite up vs proportional "
      f"({fav['prop']:.4f} -> {fav['power']:.4f})", fav["power"] > fav["prop"])
gaps = [abs(devig_shin(1.0 / (p_true[:200] * m))
            - devig_proportional(1.0 / (p_true[:200] * m))).max()
        for m in (1.0005, 1.005, 1.05)]
check(f"the methods converge as the overround shrinks "
      f"({gaps[2]:.4f} -> {gaps[1]:.4f} -> {gaps[0]:.6f})",
      gaps[0] < gaps[1] < gaps[2] and gaps[0] < 1e-3)
check("the gap scales roughly linearly with the overround",
      abs(gaps[1] / gaps[0] - 10.0) < 1.0)

# null test: a perfectly calibrated book shows no bias
p_null, y_null, quoted_null = fair_book(60000, 0.05, rng)
recovered = devig_proportional(quoted_null)
fit = calibration_regression(recovered, y_null)
naive = calibration_regression(recovered, y_null, cluster=False)
check(f"null book: calibration slope is 1 (got {fit['slope']:.3f} +- "
      f"{fit['slope_se']:.3f}, p={fit['slope_p']:.2f})",
      abs(fit["slope"] - 1.0) < 0.05 and fit["slope_p"] > 0.01)
check(f"null book: intercept is 0 (got {fit['intercept']:.3f})",
      abs(fit["intercept"]) < 0.05)
check(f"clustering by match widens the standard error as it should "
      f"({naive['slope_se']:.4f} -> {fit['slope_se']:.4f})",
      fit["slope_se"] > naive["slope_se"] * 1.1)

cal = calibration(recovered, y_null)
check(f"null book: every bin is within 3 sigma (max |z| = {cal['z'].abs().max():.2f})",
      cal["z"].abs().max() < 3.0)
check("null book: realised frequency sits inside the Wilson interval everywhere",
      ((cal["mean_p"] >= cal["lo"]) & (cal["mean_p"] <= cal["hi"])).all())

# A book that really is biased has to be caught, or the null test above proves
# nothing. Tilt the quotes so longshots are priced short and favourites long.
tilt = p_null ** 0.85
tilt = tilt / tilt.sum(axis=1, keepdims=True)
bias_fit = calibration_regression(tilt, y_null)
check(f"a deliberately tilted book is flagged (slope {bias_fit['slope']:.3f}, "
      f"p={bias_fit['slope_p']:.1e})",
      bias_fit["slope"] > 1.05 and bias_fit["slope_p"] < 1e-6)

# scoring
check("Brier is 0 for a perfect forecast", brier(y_null, y_null).mean() == 0.0)
check("Brier is 2 for a confidently wrong forecast",
      brier(np.array([[1.0, 0, 0]]), np.array([[0.0, 0, 1]]))[0] == 2.0)
check("log loss of a 1/3-1/3-1/3 book is log 3",
      abs(log_loss(np.full((1, 3), 1 / 3), np.array([[1.0, 0, 0]]))[0]
          - np.log(3)) < 1e-12)
check(f"the true probabilities score better than a uniform book "
      f"({brier(p_null, y_null).mean():.4f} < "
      f"{brier(np.full_like(p_null, 1 / 3), y_null).mean():.4f})",
      brier(p_null, y_null).mean() < brier(np.full_like(p_null, 1 / 3), y_null).mean())

# Closing lines should beat opening lines when they are simply less noisy.
noisy = np.abs(p_null + rng.normal(0, 0.08, p_null.shape))
noisy /= noisy.sum(axis=1, keepdims=True)
sharper = np.abs(p_null + rng.normal(0, 0.03, p_null.shape))
sharper /= sharper.sum(axis=1, keepdims=True)
check(f"a less noisy forecast wins on Brier ({brier(sharper, y_null).mean():.4f} < "
      f"{brier(noisy, y_null).mean():.4f})",
      brier(sharper, y_null).mean() < brier(noisy, y_null).mean())
check("and on log loss",
      log_loss(sharper, y_null).mean() < log_loss(noisy, y_null).mean())

lo, hi = wilson(50, 100)
check(f"Wilson interval on 50/100 brackets 0.5 ({lo:.3f}, {hi:.3f})",
      lo < 0.5 < hi and 0.09 < hi - lo < 0.21)
check("Wilson interval narrows with sample size",
      np.diff(wilson(5000, 10000))[0] < np.diff(wilson(50, 100))[0])

# the vig costs exactly what it should
MARGIN = 0.045
n_flips = 400000
o = 2.0 / (1.0 + MARGIN)
won = rng.random(n_flips) < 0.5
pnl = profit(o, won)
expected = 1.0 / (1.0 + MARGIN) - 1.0
check(f"a fair coin at a {MARGIN:.1%} book loses {expected:.4%} per unit "
      f"(simulated {pnl.mean():.4%})", abs(pnl.mean() - expected) < 0.003)
check("that loss equals minus the margin per unit of book sum",
      abs(expected - (1.0 / booksum([[o, o]]) - 1.0)) < 1e-12)

for rule_odds, label in ((1.25, "heavy favourite"), (6.0, "longshot")):
    p_win = 1.0 / (rule_odds * (1.0 + MARGIN))
    hits = rng.random(200000) < p_win
    got = profit(rule_odds, hits).mean()
    check(f"flat betting the {label} at a fair {MARGIN:.1%} book returns "
          f"{expected:.3%} (got {got:+.3%})", abs(got - expected) < 0.006)

_, y_flat, quoted_flat = fair_book(200000, MARGIN, rng)
for rule, cols in (("favourite", quoted_flat.argmin(axis=1)),
                   ("longshot", quoted_flat.argmax(axis=1))):
    _, _, r = flat_bet(quoted_flat, y_flat, cols)
    check(f"with no bias, backing every {rule} still returns the margin "
          f"({r.mean():+.3%} vs {expected:.3%})", abs(r.mean() - expected) < 0.008)

# Kelly
check("Kelly stakes nothing when the odds are exactly fair",
      kelly_fraction(0.5, 2.0) == 0.0)
check("Kelly stakes nothing on a negative edge", kelly_fraction(0.4, 2.0) == 0.0)
check("Kelly on a 60% shot at evens is 20% of bankroll",
      abs(kelly_fraction(0.6, 2.0) - 0.2) < 1e-12)
check("Kelly grows with the edge",
      kelly_fraction(0.7, 2.0) > kelly_fraction(0.6, 2.0) > kelly_fraction(0.55, 2.0))
edge_win = rng.random(100000) < 0.55
f_star = kelly_fraction(0.55, 2.0)
check(f"full Kelly at a real 5% edge grows the bankroll "
      f"(mean log growth {kelly_growth(f_star, 2.0, edge_win).mean():+.5f})",
      kelly_growth(f_star, 2.0, edge_win).mean() > 0)
check("over-betting past Kelly grows slower",
      kelly_growth(3 * f_star, 2.0, edge_win).mean()
      < kelly_growth(f_star, 2.0, edge_win).mean())
no_edge = rng.random(200000) < 1.0 / (2.0 * (1 + MARGIN))
check("Kelly sized off a de-vigged line still bleeds when the line is right",
      kelly_growth(0.05, 2.0, no_edge).mean() < 0)

# arbitrage
book_a = np.array([[2.10, 3.40, 3.90], [1.90, 3.50, 4.20]])
book_b = np.array([[1.95, 3.30, 4.60], [1.85, 3.60, 4.00]])
best = best_across_books([book_a, book_b])
check("best-of-books takes the longest price on each outcome",
      np.allclose(best, [[2.10, 3.40, 4.60], [1.90, 3.60, 4.20]]))
s, ret = arbitrage(best)
check(f"the planted arb is detected (book sum {s[0]:.4f}, return {ret[0]:.3%})",
      s[0] < 1.0 and ret[0] > 0)
check("the non-arb match returns nothing", ret[1] == 0.0)
check("arb return matches the split-stake payoff",
      abs(ret[0] - (1.0 / s[0] - 1.0)) < 1e-12)
check("no arb is found inside a single margined book",
      (arbitrage(quoted_null)[1] == 0).all())

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
