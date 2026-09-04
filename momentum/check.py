"""Offline checks on synthetic prices with a planted momentum effect.

Builds 60 fake stocks whose drift is slowly varying, so trailing 12-1 momentum
genuinely predicts the next month. The strategy must find it, the placebo must
not, and the engine must refuse to see the future.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from backtest import BORROW_BPS, TRADING_DAYS, by_year, max_drawdown, metrics, run
from momentum import (SKIP, cross_sectional_weights, crossover_weights,
                      information_coefficient, leg_weights, ma_signal, momentum_score,
                      quantile_returns)

rng = np.random.default_rng(7)
N_STOCKS = 60
N_DAYS = 2600
IDX = pd.bdate_range("2010-01-04", periods=N_DAYS)
NAMES = [f"S{i:02d}" for i in range(N_STOCKS)]

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def planted_panel(drift_persistence=0.995, drift_vol=0.00007, noise=0.011):
    """Simulate prices whose drift wanders slowly, making past returns predictive."""
    drift = np.zeros((N_DAYS, N_STOCKS))
    drift[0] = rng.normal(0, 0.0006, N_STOCKS)
    for t in range(1, N_DAYS):
        drift[t] = drift_persistence * drift[t - 1] + rng.normal(0, drift_vol, N_STOCKS)
    rets = drift + rng.normal(0, noise, (N_DAYS, N_STOCKS))
    px = pd.DataFrame(50 * np.exp(np.cumsum(rets, axis=0)), index=IDX, columns=NAMES)
    return px, pd.DataFrame(rets, index=IDX, columns=NAMES)


px, true_rets = planted_panel()
returns = px.pct_change()
eligible = pd.DataFrame(True, index=IDX, columns=NAMES)
eligible.iloc[:252] = False
dates = pd.DatetimeIndex(pd.Series(IDX, index=IDX).groupby([IDX.year, IDX.month]).max())
dates = dates[dates >= IDX[300]]

score = momentum_score(px)
signal = ma_signal(px)

print("SIGNAL CONSTRUCTION\n")

check("momentum score is undefined until the full lookback fills",
      score.iloc[:251].isna().all().all() and score.iloc[252].notna().all())

spiked = px.copy()
spiked.iloc[1500] *= 1.5
score_spiked = momentum_score(spiked)
check("a price spike at t=1500 leaves every momentum score before t=1521 untouched",
      np.allclose(score.iloc[:1521].fillna(0), score_spiked.iloc[:1521].fillna(0))
      and not np.allclose(score.iloc[1521], score_spiked.iloc[1521]))

manual = px.iloc[1500 - 21] / px.iloc[1500 - 252] - 1
check("momentum score matches the hand-computed P(t-21)/P(t-252)-1",
      np.allclose(score.iloc[1500], manual))

ma_spiked = ma_signal(spiked)
check("a price spike at t=1500 leaves every crossover signal before it untouched",
      np.allclose(signal.iloc[:1500].fillna(0), ma_spiked.iloc[:1500].fillna(0)))
check("crossover signal is undefined until the 200-day window fills",
      signal.iloc[:199].isna().all().all() and signal.iloc[200].notna().all())
check("crossover signal is only ever +1 or -1",
      set(np.unique(signal.dropna().to_numpy())) <= {-1.0, 1.0})

print("\nPORTFOLIO CONSTRUCTION\n")

# the real universe has ~180 eligible names, so its decile clears the default
# basket floor; 60 synthetic names give a 6-name decile, so the floor drops here
w = cross_sectional_weights(score, eligible, dates, tail=0.10, min_names=5)
gross, net = w.abs().sum(axis=1), w.sum(axis=1)
check("gross exposure is 200% on every rebalance date", np.allclose(gross, 2.0))
check("net exposure is 0% on every rebalance date", np.allclose(net, 0.0, atol=1e-12))
check("the long leg holds the top decile: 6 of 60 names",
      ((w > 0).sum(axis=1) == 6).all() and ((w < 0).sum(axis=1) == 6).all())
held_weights = w.to_numpy()[w.to_numpy() != 0]
check("every long weight is identical, as is every short weight",
      np.allclose(np.unique(held_weights.round(12)), [-1 / 6, 1 / 6]))

last = dates[-1]
ranked = score.loc[last].sort_values()
check("the names held long are exactly the highest-scoring eligible names",
      set(w.loc[last][w.loc[last] > 0].index) == set(ranked.index[-6:]))
check("the names held short are exactly the lowest-scoring eligible names",
      set(w.loc[last][w.loc[last] < 0].index) == set(ranked.index[:6]))

half_eligible = eligible.copy()
half_eligible[NAMES[:30]] = False
w_half = cross_sectional_weights(score, half_eligible, dates, tail=0.10, min_names=3)
check("an ineligible name is never held",
      (w_half[NAMES[:30]] == 0).all().all())
check("ineligible names are excluded before ranking, not after",
      ((w_half > 0).sum(axis=1) == 3).all())

thin = pd.DataFrame(True, index=IDX, columns=NAMES)
thin[NAMES[10:]] = False
check("a cross-section too thin for the minimum basket size trades nothing",
      np.allclose(cross_sectional_weights(score, thin, dates, tail=0.10, min_names=5), 0.0))
check("leg_weights drops both sides when only one side is thin",
      np.allclose(leg_weights(pd.DataFrame([[True] * 20 + [False] * 40], columns=NAMES),
                              pd.DataFrame([[False] * 59 + [True]], columns=NAMES)), 0.0))

print("\nEXECUTION LAG AND COSTS\n")

book = run(w, returns, cost_bps=0.0, borrow_bps=0.0)
daily_w = w.reindex(returns.index).ffill().fillna(0.0)
check("the weight earning t's return is the weight set at t-1",
      np.allclose(book["gross"].iloc[1:],
                  (daily_w.shift(1) * returns).sum(axis=1).iloc[1:]))
check("nothing is held on the first day of the sample", book["gross_exposure"].iloc[0] == 0)
check("gross exposure sits at 200% once positions are on",
      np.allclose(book["gross_exposure"].loc[dates[1]:], 2.0))

# Same weights, executed a further day late. If the engine were peeking, an extra
# lag would barely change anything; on a real one-day-lagged signal it costs a little.
lagged = run(w.shift(1, freq="B"), returns, cost_bps=0.0, borrow_bps=0.0)
check("an extra day of execution delay changes the result",
      not np.allclose(book["gross"].sum(), lagged["gross"].sum()))

# Rank on the return the strategy is about to earn, then push it through the same
# weighting and execution code. If the lag were missing anywhere, honest results
# would look like this.
leaked_score = px.shift(-SKIP) / px - 1
leak = run(cross_sectional_weights(leaked_score, eligible, dates, tail=0.10, min_names=5),
           returns, cost_bps=0.0, borrow_bps=0.0)
leak_sharpe = metrics(leak, "gross")["sharpe"]
real_sharpe = metrics(book, "gross")["sharpe"]
check(f"a deliberately leaked signal scores absurdly (Sharpe {leak_sharpe:.1f})",
      leak_sharpe > 5)
check(f"the real pipeline does not (Sharpe {real_sharpe:.2f})", real_sharpe < 3)

costly = run(w, returns, cost_bps=10.0, borrow_bps=BORROW_BPS)
check("gross return does not depend on the cost assumption",
      np.allclose(book["gross"], costly["gross"]))
check("zero-cost net return equals gross return", np.allclose(book["net"], book["gross"]))
check(f"costs reduce the net return ({book['net'].sum():.3f} -> {costly['net'].sum():.3f})",
      costly["net"].sum() < book["net"].sum())

expected_cost = costly["traded"] * 10.0 / 1e4 + 1.0 * BORROW_BPS / 1e4 / TRADING_DAYS
check("cost is charged on traded notional plus a daily accrual on short value",
      np.allclose(costly["cost"].loc[dates[1]:], expected_cost.loc[dates[1]:]))
quiet = costly.loc[dates[1]:]
check("on a day with no trade the only charge is the borrow accrual",
      np.allclose(quiet.loc[quiet["traded"] == 0, "cost"],
                  1.0 * BORROW_BPS / 1e4 / TRADING_DAYS))

no_borrow = run(w, returns, cost_bps=0.0, borrow_bps=100.0)
check("borrow cost tracks short market value, not gross exposure",
      np.isclose((book["net"] - no_borrow["net"]).loc[dates[1]:].iloc[0],
                 1.0 * 100.0 / 1e4 / TRADING_DAYS))

check("turnover is only booked when the held weights actually move",
      np.allclose(costly["traded"],
                  daily_w.shift(1).fillna(0.0).diff().abs().sum(axis=1).fillna(0.0)))
trade_days = (costly["traded"] > 1e-9).sum()
check(f"a monthly book never trades more often than monthly "
      f"({trade_days} trade days, {len(dates)} rebalances)",
      trade_days <= len(dates)
      and trade_days / (len(costly) / TRADING_DAYS) < 13)

print("\nTHE PLANTED EFFECT\n")

fwd = pd.DataFrame({c: (1 + returns[c]).rolling(21).apply(np.prod, raw=True).shift(-21) - 1
                    for c in returns.columns}).loc[dates]
ic = information_coefficient(score.loc[dates], fwd, eligible.loc[dates])
check(f"the planted signal has a positive information coefficient ({ic.mean():.3f})",
      ic.mean() > 0.05)

# 60 names split five ways is 12 per bucket, too thin for the middle ranks to
# order cleanly; three buckets of 20 is the sharper test on a sample this size.
thirds = quantile_returns(score.loc[dates], fwd, eligible.loc[dates], n_buckets=3)
check(f"forward returns rise monotonically across signal terciles "
      f"({', '.join(f'{v:+.2%}' for v in thirds)})",
      thirds.is_monotonic_increasing)

fifths = quantile_returns(score.loc[dates], fwd, eligible.loc[dates], n_buckets=5)
check(f"the top quintile beats the bottom by a wide margin "
      f"({fifths.iloc[-1] - fifths.iloc[0]:+.2%} per month)",
      fifths.iloc[-1] - fifths.iloc[0] > 0.005)

check(f"the strategy is profitable gross on planted data "
      f"(Sharpe {real_sharpe:.2f})", real_sharpe > 0.5)

reversed_book = run(-w, returns, cost_bps=0.0, borrow_bps=0.0)
check("buying the losers and shorting the winners loses money",
      reversed_book["gross"].sum() < 0)

shuffled = w.copy()
for date in shuffled.index:
    shuffled.loc[date] = rng.permutation(shuffled.loc[date].to_numpy())
placebo = run(shuffled, returns, cost_bps=0.0, borrow_bps=0.0)
placebo_sharpe = metrics(placebo, "gross")["sharpe"]
check(f"shuffling the ranks within each date destroys the edge "
      f"(Sharpe {placebo_sharpe:.2f})", abs(placebo_sharpe) < real_sharpe / 2)

flat = pd.DataFrame(0.0, index=dates, columns=NAMES)
check("a flat book returns exactly zero",
      np.allclose(run(flat, returns)["net"], 0.0))

print("\nMETRICS\n")

steady = pd.DataFrame({"net": 0.001, "gross": 0.001, "traded": 0.0, "cost": 0.0,
                       "gross_exposure": 2.0, "net_exposure": 0.0},
                      index=pd.bdate_range("2015-01-01", periods=TRADING_DAYS * 4))
m = metrics(steady)
check(f"a constant +10bps/day compounds to the right annual return ({m['annual_return']:.4f})",
      np.isclose(m["annual_return"], 1.001 ** TRADING_DAYS - 1, atol=1e-6))
check("a series that never falls has zero drawdown and a 100% hit rate",
      m["max_drawdown"] == 0 and m["hit_rate_monthly"] == 1.0)

path = pd.Series([0.10, -0.50, 0.20, 0.30, 0.40],
                 index=pd.bdate_range("2020-01-01", periods=5))
dd = max_drawdown(path)
check(f"max drawdown finds the planted -50% leg (got {dd['max_drawdown']:.1%})",
      np.isclose(dd["max_drawdown"], -0.5))
check("drawdown peak precedes trough, and recovery follows it",
      dd["dd_peak"] < dd["dd_trough"] <= dd["dd_recovery"])

years = by_year(costly)
check("the per-year table covers every calendar year in the sample",
      list(years.index) == sorted(set(costly.index.year)))
check("per-year net returns compound back to the full-sample net return",
      np.isclose((1 + years["net"]).prod() - 1,
                 (1 + costly["net"]).cumprod().iloc[-1] - 1))
traded_years = years[years["turnover"] > 0]
check("gross beats net in every year the book actually traded",
      (traded_years["gross"] > traded_years["net"]).all() and len(traded_years) > 5)

print("\nCROSSOVER STRATEGY\n")

ma_w = crossover_weights(signal, eligible)
ma_book = run(ma_w, returns, cost_bps=10.0, borrow_bps=BORROW_BPS)
live = ma_w.abs().sum(axis=1) > 0
check("crossover book is dollar neutral whenever it is live",
      np.allclose(ma_w[live].sum(axis=1), 0.0, atol=1e-12))
check("crossover book runs at 200% gross whenever it is live",
      np.allclose(ma_w[live].abs().sum(axis=1), 2.0))
check("the crossover book can trade any day, the monthly book only at rebalances",
      (ma_book["traded"] > 1e-9).sum() > 5 * (costly["traded"] > 1e-9).sum())
check("no position is taken before the 200-day window fills",
      ma_w.iloc[:199].abs().sum().sum() == 0)

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
