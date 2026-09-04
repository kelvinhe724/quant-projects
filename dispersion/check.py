"""Offline checks with planted truths: known option prices, a basket with a known
correlation, and a dispersion book whose P&L sign is known in advance.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

from dispersion import (MONTH, atm_vol, average_pairwise_correlation, basket_vol,
                        book_pnl, bs_price, forward_realised_vol, implied_correlation,
                        implied_vol, metrics, parallel_vega, realised_correlation,
                        realised_vol, simulate, vol_at_horizon)

rng = np.random.default_rng(11)
checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


print("BLACK-SCHOLES\n")

S, r = 100.0, 0.03
K = np.array([70, 85, 95, 100, 105, 115, 140], float)
for T in (0.05, 0.25, 1.0):
    for sigma in (0.08, 0.25, 0.80):
        for cp in (1, -1):
            price = bs_price(S, K, T, r, sigma, cp)
            back = implied_vol(price, S, K, T, r, cp)
            # a quote with no time value left carries no vol information
            live = price - np.maximum(cp * (S - K * np.exp(-r * T)), 0) > 1e-6
            good = np.allclose(back[live], sigma, atol=1e-6)
            if not good:
                check(f"solver recovers sigma={sigma} T={T} cp={cp}", False)
check("solver recovers the planted vol across strikes, tenors, calls and puts",
      all(checks))

call = bs_price(S, K, 0.5, r, 0.3, 1)
put = bs_price(S, K, 0.5, r, 0.3, -1)
check("put-call parity holds", np.allclose(call - put, S - K * np.exp(-r * 0.5)))
check("a call is worth at least its discounted intrinsic value",
      (call >= np.maximum(S - K * np.exp(-r * 0.5), 0) - 1e-12).all())
check("a price below intrinsic returns nan rather than a vol",
      np.isnan(implied_vol(2.0, S, 90.0, 0.5, r, 1)))
check("a price above the spot returns nan rather than a vol",
      np.isnan(implied_vol(101.0, S, 90.0, 0.5, r, 1)))
check("implied vol is monotone in price",
      np.all(np.diff(implied_vol(np.linspace(6, 20, 8), S, 100.0, 0.5, r, 1)) > 0))

print("\nATM EXTRACTION\n")

planted = {0.08: 0.22, 0.16: 0.25, 0.33: 0.28}
rows = []
for T, sig in planted.items():
    for k in np.arange(80, 121, 2.5):
        for cp, sign in (("C", 1), ("P", -1)):
            p = float(bs_price(S, k, T, r, sig, sign))
            rows.append({"expiry": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(T * 365)),
                         "T": T, "strike": k, "bid": p - 0.02, "ask": p + 0.02,
                         "cp": cp, "spot": S})
chain = pd.DataFrame(rows)
atm = atm_vol(chain, r)
check("ATM vol recovers the planted vol at every expiry",
      np.allclose(atm["atm_iv"].values, list(planted.values()), atol=1e-3))
check("ATM strike is the one nearest the forward",
      (atm["strike"] == 100.0).all() or (atm["strike"] == 102.5).all())
check("call and put ATM vols agree at the same strike",
      np.allclose(atm["call_iv"], atm["put_iv"], atol=1e-3))

flat = atm.copy()
flat["atm_iv"] = 0.3
check("interpolating a flat surface returns the flat vol",
      np.isclose(vol_at_horizon(flat, 30), 0.3))
v30 = vol_at_horizon(atm, 30)
check(f"the 30-day vol sits between the bracketing expiries ({v30:.4f})",
      0.22 < v30 < 0.25)
check("beyond the last expiry the last vol is used",
      np.isclose(vol_at_horizon(atm, 400), 0.28, atol=1e-3))

print("\nIMPLIED CORRELATION\n")

n = 10
weights = pd.Series(rng.uniform(0.5, 2.0, n))
weights /= weights.sum()
vols = rng.uniform(0.18, 0.45, n)
for rho in (0.1, 0.35, 0.7, 0.95):
    idx = basket_vol(vols, weights, rho)
    check(f"the formula recovers a planted correlation of {rho} from the basket vol",
          np.isclose(implied_correlation(idx, vols, weights), rho))

check("correlation one gives the weighted average vol",
      np.isclose(basket_vol(vols, weights, 1.0), np.sum(weights * vols)))
check("higher index vol with the same single-name vols means higher correlation",
      implied_correlation(0.25, vols, weights) > implied_correlation(0.20, vols, weights))

print("\nREALISED CORRELATION\n")

N_DAYS = 3000
idx_dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
rho_true = 0.4
corr = np.full((n, n), rho_true)
np.fill_diagonal(corr, 1.0)
z = rng.standard_normal((N_DAYS, n)) @ np.linalg.cholesky(corr).T
daily_vol = vols / np.sqrt(252)
rets = pd.DataFrame(z * daily_vol, index=idx_dates, columns=[f"N{i}" for i in range(n)])
weights.index = rets.columns
index_rets = (rets * weights).sum(axis=1)

rc = realised_correlation(rets, index_rets, weights, 252)
check(f"weighted realised correlation over 252 days recovers 0.4 ({rc.mean():.3f})",
      abs(rc.mean() - rho_true) < 0.03)
apc = average_pairwise_correlation(rets, 252)
check(f"average pairwise correlation recovers 0.4 ({apc.mean():.3f})",
      abs(apc.mean() - rho_true) < 0.03)
check("both estimators are undefined until the window fills",
      rc.iloc[:251].isna().all() and apc.iloc[:251].isna().all()
      and rc.iloc[251:].notna().all())

indep = pd.DataFrame(rng.standard_normal((N_DAYS, n)) * daily_vol, index=idx_dates,
                     columns=rets.columns)
apc0 = average_pairwise_correlation(indep, 252)
check(f"independent names show correlation near zero ({apc0.mean():+.3f})",
      abs(apc0.mean()) < 0.03)

rv = realised_vol(rets, 252)
check(f"realised vol recovers the planted single-name vols "
      f"(max error {np.abs(rv.mean() - vols).max():.3f})",
      np.abs(rv.mean().values - vols).max() < 0.02)
fv = forward_realised_vol(rets, MONTH)
manual = np.sqrt((rets.iloc[101:122] ** 2).mean() * 252)
check("forward realised vol at t covers exactly the 21 returns after t",
      np.allclose(fv.iloc[100], manual))
check("forward realised vol is undefined at the end of the sample",
      fv.iloc[-MONTH:].isna().all().all())

print("\nDISPERSION BOOK\n")

w_ser = pd.Series(weights.values, index=rets.columns)
one = pd.Index([0])
single = pd.DataFrame([vols * 100], columns=rets.columns, index=one)
fair_index = pd.Series([basket_vol(vols, weights, 0.4) * 100], index=one)

flat_book = book_pnl(fair_index, fair_index, single, single, w_ser)
check("equal implied and realised vols and correlation gives zero gross P&L",
      np.isclose(flat_book["gross"].iloc[0], 0.0))
check("zero costs make net equal gross",
      np.isclose(book_pnl(fair_index, fair_index, single, single, w_ser, spread_index=0,
                          spread_single=0, hedge=0)["net"].iloc[0], 0.0))
check("with costs the flat book loses exactly the cost",
      np.isclose(flat_book["net"].iloc[0], -flat_book["cost"].iloc[0])
      and flat_book["cost"].iloc[0] > 0)

spike = pd.Series([basket_vol(vols, weights, 0.9) * 100], index=one)
spiked = book_pnl(fair_index, spike, single, single, w_ser)
check(f"a correlation spike from 0.4 to 0.9 loses money ({spiked['gross'].iloc[0]:+.2f} vol pts)",
      spiked["gross"].iloc[0] < 0)
check("the loss is entirely on the index leg",
      np.isclose(spiked["single_leg"].iloc[0], 0.0) and spiked["index_leg"].iloc[0] < 0)

collapse = pd.Series([basket_vol(vols, weights, 0.1) * 100], index=one)
check("a correlation collapse from 0.4 to 0.1 makes money",
      book_pnl(fair_index, collapse, single, single, w_ser)["gross"].iloc[0] > 0)

# Single names realise above implied with the index unchanged: the trade also
# pays when the names move more than priced, which is the same thing as lower
# correlation given a fixed index vol.
hot = single * 1.2
check("single names realising above implied with the index unchanged makes money",
      book_pnl(fair_index, fair_index, single, hot, w_ser)["gross"].iloc[0] > 0)

rich = pd.Series([basket_vol(vols, weights, 0.6) * 100], index=one)
check("implied correlation above realised is the profitable side of the trade",
      book_pnl(rich, fair_index, single, single, w_ser)["gross"].iloc[0] > 0)

shift = 0.5
up_single = single + shift
up_index = pd.Series([basket_vol(vols + shift / 100, weights, 0.4) * 100], index=one)
lam = parallel_vega(fair_index.iloc[0], single.iloc[0], weights)
eq_vol = basket_vol(np.ones(n), np.full(n, 1 / n), 0.4)
eq_lam = parallel_vega(eq_vol, np.ones(n), np.full(n, 1 / n))
check(f"parallel vega on an equal unit-vol basket equals the basket vol, which tends to "
      f"sqrt(rho) with many names ({eq_lam:.3f}, sqrt(rho) {np.sqrt(0.4):.3f})",
      np.isclose(eq_lam, eq_vol) and abs(eq_lam - np.sqrt(0.4)) < 0.1)
check("a parallel vol shift moves the index vol by parallel_vega times the shift",
      np.isclose(up_index.iloc[0] - fair_index.iloc[0], lam * shift, rtol=0.02))
flat_shift = book_pnl(fair_index, up_index, single, up_single, w_ser, 1.0, 0, 0, 0)
corr_shift = book_pnl(fair_index, up_index, single, up_single, w_ser, lam, 0, 0, 0)
check(f"the vega-weighted book is long a parallel vol shift ({flat_shift['gross'].iloc[0]:+.3f})",
      flat_shift["gross"].iloc[0] > 0.1)
check(f"the correlation-weighted book is flat to it ({corr_shift['gross'].iloc[0]:+.4f})",
      abs(corr_shift["gross"].iloc[0]) < 0.01)
corr_spike = book_pnl(fair_index, spike, single, single, w_ser, lam, 0, 0, 0)
check("the correlation-weighted book still loses on a correlation spike",
      corr_spike["gross"].iloc[0] < 0)

print("\nSIMULATION CLOCK\n")

vix = pd.Series(basket_vol(vols, weights, 0.5) * 100, index=idx_dates)
premium = pd.Series(1.0, index=rets.columns)
decisions = pd.DatetimeIndex(pd.Series(idx_dates, index=idx_dates)
                             .groupby([idx_dates.year, idx_dates.month]).max())[:-2]
book = simulate(rets, index_rets, vix, w_ser, premium, decisions)
check("every fill is the trading day after its decision date",
      all(rets.index.get_loc(f) == rets.index.get_loc(d) + 1
          for f, d in zip(book.index, book["decision"])))
check(f"implied correlation in the book reads the planted VIX correlation "
      f"({book['implied_corr'].mean():.3f})",
      abs(book["implied_corr"].mean() - 0.5) < 0.05)
check(f"realised correlation in the book reads the planted 0.4 "
      f"({book['realised_corr'].mean():.3f})",
      abs(book["realised_corr"].mean() - 0.4) < 0.05)
check("implied above realised correlation earns a positive gross mean",
      book["gross"].mean() > 0)

f0 = book.index[10]
i = rets.index.get_loc(f0)
after = rets.copy()
after.iloc[i + MONTH + 1] *= 5
book_after = simulate(after, (after * weights).sum(axis=1), vix, w_ser, premium, decisions)
check("a shock after the holding window leaves that month's P&L unchanged",
      np.isclose(book.loc[f0, "gross"], book_after.loc[f0, "gross"]))
inside = rets.copy()
inside.iloc[i + 1] *= 5
book_inside = simulate(inside, (inside * weights).sum(axis=1), vix, w_ser, premium, decisions)
check("a shock on the first day after the fill changes it",
      not np.isclose(book.loc[f0, "gross"], book_inside.loc[f0, "gross"]))

vix_spiked = vix.copy()
vix_spiked.loc[book.loc[f0, "decision"]] = 80.0
book_vix = simulate(rets, index_rets, vix_spiked, w_ser, premium, decisions)
check("VIX on the decision date is not the fill price",
      np.isclose(book.loc[f0, "gross"], book_vix.loc[f0, "gross"]))
vix_spiked = vix.copy()
vix_spiked.loc[f0] = 80.0
book_vix = simulate(rets, index_rets, vix_spiked, w_ser, premium, decisions)
check("VIX on the fill date is the price traded",
      not np.isclose(book.loc[f0, "gross"], book_vix.loc[f0, "gross"]))

cw = simulate(rets, index_rets, vix, w_ser, premium, decisions, corr_weighted=True)
check("correlation weighting scales the single-name leg by the per-fill parallel vega",
      np.allclose(cw["single_leg"], book["single_leg"] * cw["scale"])
      and (cw["scale"] < 1).all() and (cw["scale"] > 0).all())
wedged = simulate(rets, index_rets, vix, w_ser, premium, decisions, wedge=2.0)
check("the wedge lowers the index implied, and so the gross P&L, by exactly its size",
      np.allclose(wedged["gross"], book["gross"] - 2.0))

# Plant a one-month correlation spike in the middle of the sample and read the
# book: that month must be the worst one.
z2 = rng.standard_normal((N_DAYS, n))
crash = np.full((n, n), 0.95)
np.fill_diagonal(crash, 1.0)
f1 = book.index[40]
j = rets.index.get_loc(f1)
mixed = rets.copy()
mixed.iloc[j + 1:j + 1 + MONTH] = (z2[j + 1:j + 1 + MONTH] @ np.linalg.cholesky(crash).T
                                   * daily_vol * 1.5)
book_crash = simulate(mixed, (mixed * weights).sum(axis=1), vix, w_ser, premium, decisions)
check(f"a planted correlation spike is the worst month in the book "
      f"({book_crash['gross'].min():+.2f} vol pts on {book_crash['gross'].idxmin().date()})",
      book_crash["gross"].idxmin() == f1 and book_crash.loc[f1, "gross"] < 0)

print("\nMETRICS\n")

path = pd.Series([1.0, 2.0, -3.0, 1.0, 1.0, 1.0],
                 index=pd.date_range("2020-01-31", periods=6, freq="ME"))
m = metrics(path)
check("max drawdown finds the planted -3 leg", np.isclose(m["max_drawdown"], -3.0))
check("drawdown peak precedes trough", m["dd_peak"] < m["dd_trough"])
check("worst month is dated", m["worst_month_date"] == path.index[2])
with np.errstate(divide="ignore", invalid="ignore"):
    constant = metrics(pd.Series([1.0] * 12, index=pd.date_range("2020-01-31", periods=12,
                                                                 freq="ME")))
check("Sharpe of a constant stream is infinite or nan, not a number we would report",
      not np.isfinite(constant["sharpe"]))

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
