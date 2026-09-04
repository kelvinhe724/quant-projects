"""Offline checks on synthetic currencies with planted rate differentials.

Builds ten fake currencies with known interest rates and simulated spot paths.
The sort must recover the planted ranking, a world with no differential must
earn no carry, costs must bite, and the UIP regression must return a slope of
one when spot moves exactly as UIP says.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from backtest import TRADING_DAYS, by_year, max_drawdown, metrics, run
from carry import carry_weights, differential, excess_returns, uip_regression

rng = np.random.default_rng(11)
CCY = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "NZD", "CHF", "NOK", "SEK"]
FOREIGN = CCY[1:]
IDX = pd.bdate_range("2010-01-01", periods=TRADING_DAYS * 12)
ENDS = pd.DatetimeIndex(pd.Series(IDX, index=IDX).groupby(IDX.to_period("M")).max().values)

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def planted_rates(levels):
    """Constant rates in percent, one per currency, on the daily index."""
    return pd.DataFrame({c: float(v) for c, v in zip(CCY, levels)}, index=IDX)


def random_walk_spot(vol=0.006, drift=None):
    """Spot paths with no expected change, or with a daily drift per currency."""
    shocks = rng.normal(0, vol, (len(IDX), len(FOREIGN)))
    if drift is not None:
        shocks = shocks + drift
    return pd.DataFrame(np.exp(np.cumsum(shocks, axis=0)), index=IDX, columns=FOREIGN)


LEVELS = [2.0, 1.0, 0.0, 3.0, 2.5, 6.0, 7.0, -0.5, 5.0, 1.5]
rates = planted_rates(LEVELS)
spot = random_walk_spot()

print("SIGNAL AND SORT\n")

diff = differential(rates)
check("the base currency's own differential is zero", (diff["USD"] == 0).all())
check("differential is rate minus the base rate",
      np.allclose(diff.iloc[0].values, np.array(LEVELS) - 2.0))

w = carry_weights(rates, ENDS)
longs = set(w.columns[(w.iloc[0] > 0).values])
shorts = set(w.columns[(w.iloc[0] < 0).values])
check("planted differentials put the three highest-rate currencies in the long leg",
      longs == {"NZD", "AUD", "NOK"})
check("and the three lowest in the short leg", shorts == {"CHF", "JPY", "EUR"})
check("each leg is equal weighted at 1/3", np.allclose(sorted(set(np.abs(w.values[w.values != 0]))), [1 / 3]))
check("the book is 200% gross and 0% net on every date",
      np.allclose(w.abs().sum(axis=1), 2.0) and np.allclose(w.sum(axis=1), 0.0))

usd_high = planted_rates([9.0] + LEVELS[1:])
w_usd = carry_weights(usd_high, ENDS)
check("the base currency can be ranked and held long when its rate is high",
      (w_usd["USD"] > 0).all())
check("holding the base long leaves the book net short foreign currency",
      np.allclose(w_usd.drop(columns="USD").sum(axis=1), -1 / 3))
w_dn = carry_weights(usd_high, ENDS, dollar_neutral=True)
check("the dollar-neutral sort never holds the base",
      (w_dn["USD"] == 0).all())
check("and its foreign notionals cancel exactly",
      np.allclose(w_dn.sum(axis=1), 0.0) and np.allclose(w_dn.abs().sum(axis=1), 2.0))

flipping = rates.copy()
flipping.loc[ENDS[10]:, "NZD"] = -3.0
w_flip = carry_weights(flipping, ENDS)
check("a currency whose rate collapses moves from the long leg to the short leg",
      w_flip.loc[ENDS[9], "NZD"] > 0 and w_flip.loc[ENDS[10], "NZD"] < 0)
check("weights on other dates do not react to a later rate change",
      np.allclose(w_flip.loc[:ENDS[9]], w.loc[:ENDS[9]]))

thin = rates.copy()
thin.loc[:, ["AUD", "NZD", "NOK", "SEK", "CHF"]] = np.nan
check("a cross-section with fewer than six rated currencies trades nothing",
      np.allclose(carry_weights(thin, ENDS), 0.0))

print("\nEXCESS RETURNS\n")

rets = excess_returns(spot, rates)
check("the base currency's excess return is identically zero", (rets["USD"] == 0).all())

flat_spot = spot.copy()
flat_spot.loc[:] = 1.0
carry_only = excess_returns(flat_spot, rates)
years = (IDX[-1] - IDX[1]).days / 365.0
check("with spot pinned, a currency earns exactly its differential",
      np.isclose(carry_only["NZD"].sum(), 0.05 * years, rtol=1e-3)
      and np.isclose(carry_only["JPY"].sum(), -0.02 * years, rtol=1e-3))

stepped = rates.copy()
stepped.loc[IDX[500]:, "NZD"] = 20.0
step_ret = excess_returns(flat_spot, stepped)
check("the accrual on day t uses the rate known at t-1, not t's own rate",
      np.isclose(step_ret.loc[IDX[500], "NZD"], carry_only.loc[IDX[500], "NZD"])
      and step_ret.loc[IDX[501], "NZD"] > 3 * carry_only.loc[IDX[501], "NZD"])

zero_diff = planted_rates([2.0] * 10)
check("a zero-differential world earns no carry at all",
      np.allclose(excess_returns(flat_spot, zero_diff), 0.0))
check("and its excess returns are pure spot changes",
      np.allclose(excess_returns(spot, zero_diff)[FOREIGN].iloc[1:],
                  spot.pct_change().iloc[1:]))

book_zero = run(carry_weights(zero_diff, ENDS), excess_returns(flat_spot, zero_diff), cost_bps=0.0)
check("carry on a zero-differential world with pinned spot returns exactly zero",
      np.allclose(book_zero["net"], 0.0))

print("\nEXECUTION LAG AND COSTS\n")

book = run(w, rets, cost_bps=0.0)
daily_w = w.reindex(rets.index).ffill().fillna(0.0)
check("the weight earning t's return is the weight set at t-1",
      np.allclose(book["gross"].iloc[1:], (daily_w.shift(1) * rets).sum(axis=1).iloc[1:]))
check("nothing is held before the first rebalance", np.allclose(book.loc[:ENDS[0], "gross"], 0.0))

spiked = spot.copy()
spiked.iloc[800] *= 1.3
r_spiked = excess_returns(spiked, rates)
check("a spot spike at t=800 leaves every earlier excess return untouched",
      np.allclose(rets.iloc[:800], r_spiked.iloc[:800]) and not np.allclose(rets.iloc[800], r_spiked.iloc[800]))

costly = run(w, rets, cost_bps=5.0)
check("gross return does not depend on the cost assumption", np.allclose(book["gross"], costly["gross"]))
check(f"costs reduce the net return ({book['net'].sum():.4f} -> {costly['net'].sum():.4f})",
      costly["net"].sum() < book["net"].sum())
check("cost is charged only on the first trade when the ranking never changes",
      (costly["traded"] > 0).sum() == 1
      and np.isclose(costly["cost"].sum(), 2.0 * 5.0 / 1e4))

usd_book = run(w_usd, rets, cost_bps=5.0)
check("a position in the base currency is never charged",
      np.isclose(usd_book["cost"].sum(), w_usd.drop(columns="USD").iloc[0].abs().sum() * 5.0 / 1e4))

churn = carry_weights(flipping, ENDS)
churn_book = run(churn, rets, cost_bps=5.0)
check("a rebalance that swaps names books turnover on that day only",
      (churn_book["traded"] > 0).sum() == 2
      and np.isclose(churn_book.loc[churn_book.index > ENDS[10], "traded"].iloc[0],
                     (churn.loc[ENDS[10]] - churn.loc[ENDS[9]]).abs().sum()))

print("\nTHE PLANTED PREMIUM\n")

# spot is a driftless random walk, so the differential is earned in full
premium = run(w, rets, cost_bps=0.0)
sharpe = metrics(premium, "gross")["sharpe"]
check(f"carry is profitable when spot does not offset the differential (Sharpe {sharpe:.2f})",
      sharpe > 1.0)
expected = (0.05 + 0.04 + 0.03 - (-0.025 - 0.02 - 0.01)) / 3
pinned = run(w, carry_only, cost_bps=0.0)
realised = pinned["gross"].loc[ENDS[0]:].sum() / ((IDX[-1] - ENDS[0]).days / 365.0)
check(f"with spot pinned the gross annual return is the planted spread ({realised:.3%} vs {expected:.3%})",
      abs(realised - expected) < 1e-3)


def drifting_spot(rates, sign, vol):
    """Spot whose expected change each day is sign times the forward premium.

    sign=+1 is the UIP world (high-rate currencies depreciate by their
    differential); sign=-1 is the Fama world, where they appreciate instead.
    """
    dt = (IDX.to_series().diff().dt.days.fillna(0) / 365.0).values[:, None]
    drift = -sign * differential(rates).drop(columns="USD").shift(1).fillna(0.0).values / 100.0 * dt
    shocks = rng.normal(0, vol, drift.shape) + drift
    return pd.DataFrame(np.exp(np.cumsum(shocks, axis=0)), index=IDX, columns=FOREIGN)


uip_book = run(w, excess_returns(drifting_spot(rates, +1, 0.006), rates), cost_bps=0.0)
uip_sharpe = metrics(uip_book, "gross")["sharpe"]
check(f"in a UIP-consistent world the same book earns about nothing (Sharpe {uip_sharpe:.2f})",
      abs(uip_sharpe) < 0.5)

reversed_book = run(-w, rets, cost_bps=0.0)
check("shorting high rates and buying low rates loses money", reversed_book["gross"].sum() < 0)

shuffled = w.copy()
for d in shuffled.index:
    shuffled.loc[d] = rng.permutation(shuffled.loc[d].to_numpy())
placebo = metrics(run(shuffled, rets, cost_bps=0.0), "gross")["sharpe"]
check(f"shuffling the ranks destroys the edge (Sharpe {placebo:.2f})", abs(placebo) < sharpe / 2)

print("\nUIP REGRESSION\n")

# rates that wander month to month, so each currency's differential has a
# time series and the per-currency slope is identified
steps = rng.normal(0, 0.4, (len(ENDS), len(CCY))).cumsum(axis=0)
wander = pd.DataFrame(np.array(LEVELS) + steps, index=ENDS, columns=CCY).reindex(IDX).bfill().ffill()

table = uip_regression(drifting_spot(wander, +1, 0.002), wander, ENDS)
check(f"UIP-consistent data gives a pooled slope near 1 ({table.loc['pooled', 'slope']:.2f})",
      abs(table.loc["pooled", "slope"] - 1) < 0.15)
check("and the median per-currency slope is nearer 1 than 0",
      abs(table["slope"].drop("pooled").median() - 1) < 0.5)

fama = uip_regression(drifting_spot(wander, -1, 0.002), wander, ENDS)
check(f"data where high-rate currencies appreciate gives a slope near -1 ({fama.loc['pooled', 'slope']:.2f})",
      abs(fama.loc["pooled", "slope"] + 1) < 0.15)

rw = uip_regression(random_walk_spot(vol=0.002), wander, ENDS)
check(f"a driftless random walk gives a slope near zero ({rw.loc['pooled', 'slope']:.2f})",
      abs(rw.loc["pooled", "slope"]) < 0.15)
check("the regression covers every foreign currency plus a pooled row",
      list(table.index) == FOREIGN + ["pooled"])

print("\nMETRICS\n")

path = pd.Series([0.10, -0.50, 0.20, 0.30, 0.40], index=pd.bdate_range("2020-01-01", periods=5))
dd = max_drawdown(path)
check(f"max drawdown finds the planted -50% leg (got {dd['max_drawdown']:.1%})",
      np.isclose(dd["max_drawdown"], -0.5))
check("drawdown peak precedes trough, and recovery follows",
      dd["dd_peak"] < dd["dd_trough"] <= dd["dd_recovery"])

crash = costly.copy()
crash.loc[IDX[1000], "net"] = -0.15
check("a single planted crash day makes daily skew negative",
      metrics(crash, "net")["skew_daily"] < -1)

years_table = by_year(costly)
check("per-year net returns compound back to the full-sample net return",
      np.isclose((1 + years_table["net"]).prod() - 1, (1 + costly["net"]).cumprod().iloc[-1] - 1))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
