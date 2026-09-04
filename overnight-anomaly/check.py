"""Offline checks on synthetic OHLC with a planted overnight drift.

Run: python3 check.py
Exits 1 if anything fails.
"""
import numpy as np
import pandas as pd

from data import STALE_YEAR_LIMIT, drop_stale_years, stale_open
from decompose import (TRADING_DAYS, annualise, breakeven_cost_bps, by_year, legs, nw_tstat,
                       strategy, summarise)

rng = np.random.default_rng(3)
N_DAYS = 5000
IDX = pd.bdate_range("2005-01-03", periods=N_DAYS)

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def synthetic_ohlc(overnight_drift, intraday_drift, vol_overnight=0.006, vol_intraday=0.009):
    """Build OHLC where each day's open is the prior close times a random overnight move."""
    on = overnight_drift + rng.normal(0, vol_overnight, N_DAYS)
    day = intraday_drift + rng.normal(0, vol_intraday, N_DAYS)
    close = np.empty(N_DAYS)
    open_ = np.empty(N_DAYS)
    prev_close = 100.0
    for i in range(N_DAYS):
        open_[i] = prev_close * (1 + on[i])
        close[i] = open_[i] * (1 + day[i])
        prev_close = close[i]
    f = pd.DataFrame({"Open": open_, "Close": close}, index=IDX)
    f["High"] = f[["Open", "Close"]].max(axis=1)
    f["Low"] = f[["Open", "Close"]].min(axis=1)
    f["Volume"] = 1e6
    return f, pd.Series(on, IDX), pd.Series(day, IDX)


print("DECOMPOSITION\n")

ohlc, true_on, true_day = synthetic_ohlc(overnight_drift=0.0005, intraday_drift=0.0)
L = legs(ohlc)

check("the first day has no overnight leg and is dropped", L.index[0] == IDX[1])
check("(1 + overnight)(1 + intraday) equals 1 + close_to_close to machine precision",
      np.allclose((1 + L["overnight"]) * (1 + L["intraday"]), 1 + L["close_to_close"],
                  rtol=0, atol=1e-12))
check("the overnight leg reproduces the simulated overnight moves exactly",
      np.allclose(L["overnight"], true_on.iloc[1:], atol=1e-12))
check("the intraday leg reproduces the simulated intraday moves exactly",
      np.allclose(L["intraday"], true_day.iloc[1:], atol=1e-12))
check("compounding both legs reproduces the close-to-close price path",
      np.isclose((1 + L["close_to_close"]).prod(), ohlc["Close"].iloc[-1] / ohlc["Close"].iloc[0]))

print("\nTHE PLANTED EFFECT\n")

S = summarise(L)
check(f"a planted +5bps/day overnight drift is recovered "
      f"({S.loc['overnight', 'mean_daily_bps']:.1f}bps, t={S.loc['overnight', 'nw_tstat']:.1f})",
      abs(S.loc["overnight", "mean_daily_bps"] - 5) < 1.5 and S.loc["overnight", "nw_tstat"] > 3)
check(f"the zero-drift intraday leg is not significant (t={S.loc['intraday', 'nw_tstat']:.2f})",
      abs(S.loc["intraday", "nw_tstat"]) < 2)
check("nearly all of the total return is in the overnight leg",
      S.loc["overnight", "annual_return"] > 0.8 * S.loc["close_to_close", "annual_return"] > 0)

ohlc2, _, _ = synthetic_ohlc(overnight_drift=0.0, intraday_drift=0.0005)
S2 = summarise(legs(ohlc2))
check(f"the mirror image, drift planted intraday, is recovered on that leg instead "
      f"(overnight t={S2.loc['overnight', 'nw_tstat']:.1f}, intraday t={S2.loc['intraday', 'nw_tstat']:.1f})",
      abs(S2.loc["overnight", "nw_tstat"]) < 2 and S2.loc["intraday", "nw_tstat"] > 3)

ohlc0, _, _ = synthetic_ohlc(overnight_drift=0.0, intraday_drift=0.0)
S0 = summarise(legs(ohlc0))
check("with no drift anywhere, neither leg is significant",
      (S0[["nw_tstat"]].abs() < 2.5).all().all())

print("\nSTATISTICS\n")

white = pd.Series(rng.normal(0, 0.01, N_DAYS), IDX)
m, t = nw_tstat(white)
naive_t = white.mean() / white.std() * np.sqrt(N_DAYS)
check(f"Newey-West t on white noise is close to the naive t ({t:.2f} vs {naive_t:.2f})",
      abs(t - naive_t) < 0.3)

ar = np.zeros(N_DAYS)
for i in range(1, N_DAYS):
    ar[i] = 0.5 * ar[i - 1] + rng.normal(0, 0.01)
ar = pd.Series(ar + 0.001, IDX)
_, t_ar = nw_tstat(ar)
naive_ar = ar.mean() / ar.std() * np.sqrt(N_DAYS)
check(f"Newey-West shrinks the t-stat of a positively autocorrelated series "
      f"({t_ar:.1f} vs naive {naive_ar:.1f})", t_ar < 0.8 * naive_ar)

const = pd.Series(0.001, index=IDX)
check("a constant +10bps/day annualises to the right number",
      np.isclose(annualise(const), 1.001 ** TRADING_DAYS - 1))

years = by_year(L)
check("per-year returns compound back to the full-sample return for every leg",
      np.allclose((1 + years).prod(), (1 + L).prod()))

print("\nCOSTS\n")

# Overnight edge is +5bps/day. One trade in, one trade out. Below 2.5bps one-way
# the strategy should keep some of it; above, it should lose.
edge_bps = L["overnight"].mean() * 1e4
be = breakeven_cost_bps(L["overnight"])
check(f"breakeven one-way cost is half the daily edge ({be:.2f}bps vs edge {edge_bps:.2f}bps)",
      abs(be - edge_bps / 2) < 0.02)
net_at_be = strategy(L, "overnight", be)
check("at the breakeven cost the net mean daily return is zero",
      abs(net_at_be.mean()) < 1e-7)

cheap = strategy(L, "overnight", cost_bps=1.0)
dear = strategy(L, "overnight", cost_bps=4.0)
check(f"edge > 2x cost: overnight-only still profitable at 1bp one-way "
      f"({annualise(cheap):+.1%}/yr)", annualise(cheap) > 0)
check(f"edge < 2x cost: overnight-only loses money at 4bps one-way "
      f"({annualise(dear):+.1%}/yr)", annualise(dear) < 0)
check("zero cost reproduces the raw leg", np.allclose(strategy(L, "overnight", 0.0), L["overnight"]))
check("buy and hold pays nothing", np.allclose(strategy(L, "close_to_close", 50.0), L["close_to_close"]))
check("a negative-edge leg has zero breakeven",
      breakeven_cost_bps(pd.Series([-0.001] * 10)) == 0.0)

daily_drag = 1 - (1 - 4 / 1e4) ** 2
check("the daily cost drag of two trades at 4bps is about 8bps",
      abs(daily_drag * 1e4 - 8) < 0.01)

print("\nDATA HYGIENE\n")

stale = ohlc.copy()
year_2007 = stale.index.year == 2007
stale.loc[year_2007, "Open"] = stale["Close"].shift(1)[year_2007]
flag = stale_open(stale)
check("stale opens are detected on exactly the planted year",
      flag[year_2007].mean() > 0.99 and flag[~year_2007].mean() < 0.01)
kept = drop_stale_years(stale)
check(f"a year with more than {STALE_YEAR_LIMIT:.0%} stale opens is dropped, others kept",
      2007 not in kept.index.year and len(kept) == len(stale) - year_2007.sum())
check("stale opens push the whole move into the intraday leg on those days",
      np.allclose(legs(stale).loc["2007", "overnight"], 0.0))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
