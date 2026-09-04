"""Offline checks on synthetic prices with a planted trend.

Twelve fake assets in four classes. One panel has slowly wandering drift, so
each asset's own past year predicts its next month; the other is a pure random
walk. The strategy must profit on the first, do nothing on the second, hit its
volatility target, and never see the future.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from backtest import MONTHS, by_year, max_drawdown, metrics, run
from data import month_ends, monthly_returns
from trend import (TRADING_DAYS, build, ex_ante_vol, portfolio_weights, positions,
                   trend_signal)

rng = np.random.default_rng(11)
N_DAYS = 252 * 20
IDX = pd.bdate_range("2004-01-01", periods=N_DAYS)
CLASSES = pd.Series({f"A{i}": cls for i, cls in enumerate(
    ["equities"] * 4 + ["bonds"] * 3 + ["commodities"] * 3 + ["fx"] * 2)})
NAMES = list(CLASSES.index)
TRUE_VOL = np.array([0.18, 0.20, 0.22, 0.25, 0.06, 0.08, 0.12, 0.30, 0.20, 0.40, 0.09, 0.10])

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def simulate(persistence, drift_vol):
    """Simulate log prices whose daily drift follows an AR(1); persistence 0 is a random walk."""
    daily_vol = TRUE_VOL / np.sqrt(TRADING_DAYS)
    drift = np.zeros((N_DAYS, len(NAMES)))
    for t in range(1, N_DAYS):
        drift[t] = persistence * drift[t - 1] + rng.normal(0, drift_vol * daily_vol)
    rets = drift + rng.normal(0, daily_vol, (N_DAYS, len(NAMES)))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=IDX, columns=NAMES)


trending = simulate(persistence=0.998, drift_vol=0.005)
walk = simulate(persistence=0.0, drift_vol=0.0)
ends = month_ends(IDX)

print("SIGNAL\n")

sig = trend_signal(trending.loc[ends])
check("signal is undefined until 12 month-ends are behind it",
      sig.iloc[:12].isna().all().all() and sig.iloc[12].notna().all())
check("signal is only ever +1 or -1", set(np.unique(sig.dropna().to_numpy())) <= {-1.0, 1.0})
t = 100
manual = np.sign(trending.loc[ends[t - 1]] / trending.loc[ends[t - 12]] - 1)
check("signal at t is the sign of P(t-1 month) / P(t-12 months) - 1", (sig.iloc[t] == manual).all())
check("skip=0 uses the most recent month, skip=1 does not",
      not sig.equals(trend_signal(trending.loc[ends], skip=0)))
try:
    trend_signal(trending.loc[ends], lookback=1, skip=1)
    check("a lookback no longer than the skip is refused", False)
except ValueError:
    check("a lookback no longer than the skip is refused", True)

print("\nNO LOOK-AHEAD (mutation test)\n")

daily = trending.pct_change()
w = build(trending, daily, CLASSES)
cut = ends[150]
mutated = trending.copy()
after = mutated.index > cut
mutated.loc[after] *= np.exp(rng.normal(0, 0.05, (after.sum(), len(NAMES))).cumsum(axis=0))
w_mut = build(mutated, mutated.pct_change(), CLASSES)
check("rewriting every price after t leaves every weight up to and including t byte-identical",
      w.loc[:cut].equals(w_mut.loc[:cut]))
check("the mutation is real: weights after t differ",
      not np.allclose(w.loc[cut:].iloc[1:], w_mut.loc[cut:].iloc[1:]))
vol = ex_ante_vol(daily)
vol_mut = ex_ante_vol(mutated.pct_change())
check("ex-ante vol at t is untouched by prices after t", vol.loc[:cut].equals(vol_mut.loc[:cut]))

mrets = monthly_returns(trending)
book = run(w, mrets, cost_bps=0.0)
check("the weight earning month t's return is the weight set at month-end t-1",
      np.allclose(book["gross"].iloc[1:], (w.shift(1) * mrets).sum(axis=1).iloc[1:]))
check("nothing is held in the first month", book["leverage"].iloc[0] == 0)

leaked = np.sign(mrets.shift(-1)).where(mrets.shift(-1).notna())
leak_book = run(portfolio_weights(positions(leaked, vol), CLASSES), mrets, cost_bps=0.0)
leak_sharpe, real_sharpe = metrics(leak_book["gross"])["sharpe"], metrics(book["gross"])["sharpe"]
check(f"a signal that peeks at next month scores absurdly (Sharpe {leak_sharpe:.1f})",
      leak_sharpe > 5)
check(f"the real pipeline does not (Sharpe {real_sharpe:.2f})", real_sharpe < 3)

print("\nPLANTED TREND VS RANDOM WALK\n")

check(f"trending series: positive gross P&L (Sharpe {real_sharpe:.2f})",
      book["gross"].sum() > 0 and real_sharpe > 0.8)
check("shorting the trend loses", run(-w, mrets, cost_bps=0.0)["gross"].sum() < 0)

walk_book = run(build(walk, walk.pct_change(), CLASSES), monthly_returns(walk), cost_bps=0.0)
walk_sharpe = metrics(walk_book["gross"])["sharpe"]
check(f"random walk: Sharpe near zero ({walk_sharpe:+.2f})", abs(walk_sharpe) < 0.4)
check("the planted trend is worth several times the random-walk result",
      real_sharpe > 3 * abs(walk_sharpe))

for lb in (1, 3, 6):
    lb_book = run(build(trending, daily, CLASSES, lookback=lb, skip=0), mrets, cost_bps=0.0)
    check(f"a {lb}-month lookback also finds the planted trend "
          f"(Sharpe {metrics(lb_book['gross'])['sharpe']:.2f})", lb_book["gross"].sum() > 0)

print("\nVOLATILITY SCALING\n")

walk_daily = walk.pct_change()
walk_m = monthly_returns(walk)
walk_vol = ex_ante_vol(walk_daily)
pos = positions(trend_signal(walk.loc[ends]), walk_vol, target=0.40)
realised = (pos.shift(1) * walk_m).std() * np.sqrt(MONTHS)
check(f"every asset lands near the 40% target regardless of its own vol "
      f"(range {realised.min():.0%} to {realised.max():.0%})",
      ((realised > 0.30) & (realised < 0.52)).all())
check("the lowest-vol asset gets the largest position",
      pos.abs().mean().idxmax() == NAMES[int(TRUE_VOL.argmin())])
check("position size is inversely proportional to ex-ante vol",
      np.allclose((pos.abs() * walk_vol.reindex(ends)).dropna(), 0.40))
check("a 20% target halves every position relative to 40%",
      np.allclose(positions(trend_signal(walk.loc[ends]), walk_vol, 0.20),
                  pos / 2, equal_nan=True))

print("\nPORTFOLIO WEIGHTS\n")

unit = pd.DataFrame(1.0, index=ends[-5:], columns=NAMES)
cw = portfolio_weights(unit, CLASSES)
by_class = cw.T.groupby(CLASSES).sum().T
check("class scheme gives each of the four classes exactly a quarter of the book",
      np.allclose(by_class, 0.25))
check("within a class every live asset gets the same share",
      np.allclose(cw["A0"], cw["A3"]) and np.allclose(cw["A4"], cw["A6"]))
check("equal scheme gives every asset 1/N",
      np.allclose(portfolio_weights(unit, CLASSES, "equal"), 1 / len(NAMES)))
partial = unit.copy()
partial["A11"] = np.nan
pw = portfolio_weights(partial, CLASSES)
check("a missing asset is dropped and its class share goes to its siblings",
      pw["A11"].eq(0).all() and np.isclose(pw["A10"].iloc[0], 0.25))
gone = unit.copy()
gone[["A10", "A11"]] = np.nan
check("a class with no live assets is dropped and the others split the book",
      np.allclose(portfolio_weights(gone, CLASSES).T.groupby(CLASSES).sum().T.drop(columns="fx"),
                  1 / 3))

print("\nCOSTS AND METRICS\n")

costly = run(w, mrets, cost_bps=10.0)
check("gross does not depend on the cost assumption", np.allclose(book["gross"], costly["gross"]))
check("zero cost makes net equal gross", np.allclose(book["net"], book["gross"]))
check("costs reduce net", costly["net"].sum() < book["net"].sum())
check("cost equals traded notional times bps",
      np.allclose(costly["gross"] - costly["net"], costly["traded"] * 10 / 1e4))
check("turnover is booked only when held weights move",
      np.allclose(costly["traded"].iloc[1:], w.shift(1).diff().abs().sum(axis=1).iloc[1:]))
check("a flat book returns exactly zero",
      np.allclose(run(w * 0, mrets)["net"], 0.0))

steady = pd.Series(0.01, index=ends[:48])
m = metrics(steady)
check("a constant +1%/month compounds to the right annual return",
      np.isclose(m["annual_return"], 1.01 ** 12 - 1))
check("a series that never falls has zero drawdown and a 100% hit rate",
      m["max_drawdown"] == 0 and m["hit_rate"] == 1.0)
path = pd.Series([0.10, -0.50, 0.20, 0.30, 0.40], index=ends[:5])
dd = max_drawdown(path)
check(f"max drawdown finds the planted -50% leg ({dd['max_drawdown']:.0%})",
      np.isclose(dd["max_drawdown"], -0.5) and dd["dd_peak"] < dd["dd_trough"])
years = by_year(costly)
check("per-year net returns compound back to the full-sample net return",
      np.isclose((1 + years["net"]).prod(), (1 + costly["net"]).prod()))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
