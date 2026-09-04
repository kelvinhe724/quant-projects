"""Offline checks on synthetic spreads with a planted probit relationship.

Run: python3 check.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from data import HORIZON, recession_ahead, sahm
from model import (auc, brier, calibration, expanding_forecasts, fit_probit,
                   inversion_episodes, predict, roc_curve)

rng = np.random.default_rng(3)
checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def planted_panel(n=900, alpha=-0.8, beta=-0.9):
    """Monthly panel where P(label) = Phi(alpha + beta * spread) exactly, spread an AR(1)."""
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.97 * spread[t - 1] + rng.normal(0, 0.35)
    p = norm.cdf(alpha + beta * spread)
    label = (rng.uniform(size=n) < p).astype(float)
    idx = pd.date_range("1950-01-01", periods=n, freq="MS")
    return pd.DataFrame({"spread": spread, "recession_ahead": label, "p_true": p}, index=idx)


print("LABEL CONSTRUCTION\n")

usrec = pd.Series(0.0, index=pd.date_range("2000-01-01", periods=60, freq="MS"))
usrec.iloc[30:36] = 1.0
label = recession_ahead(usrec)
check("label is 1 for the 12 months before a recession starts and 0 before that",
      (label.iloc[18:30] == 1).all() and (label.iloc[:18] == 0).all())
check("label stays 1 through the recession up to its last month",
      (label.iloc[30:35] == 1).all() and label.iloc[35] == 0)
check("label is undefined for the final 12 months where the window is not observed",
      label.iloc[-HORIZON:].isna().all() and label.iloc[-HORIZON - 1] == 0)

un = pd.Series(4.0, index=usrec.index)
un.iloc[40:] = [4.2, 4.4, 4.6, 4.8, 5.0] + [5.0] * 15
gap = sahm(un)
check("Sahm gap is zero while unemployment is flat", np.isclose(gap.iloc[39], 0.0))
check("Sahm gap reaches 0.5 once the 3-month average is half a point above its 12-month low",
      gap.iloc[42] < 0.5 <= gap.iloc[43])

print("\nPROBIT RECOVERY\n")

panel = planted_panel()
fit = fit_probit(panel, ["spread"])
a_hat, b_hat = fit.params["const"], fit.params["spread"]
check(f"planted intercept -0.8 recovered ({a_hat:+.2f})", abs(a_hat + 0.8) < 0.15)
check(f"planted slope -0.9 recovered ({b_hat:+.2f})", abs(b_hat + 0.9) < 0.15)
check("slope is significant at any conventional level", fit.tvalues["spread"] < -5)
p_hat = predict(fit, panel, ["spread"])
check("fitted probabilities track the true ones",
      np.corrcoef(p_hat, panel["p_true"])[0, 1] > 0.98)

noise = panel.copy()
noise["noise"] = rng.normal(size=len(noise))
fit_noise = fit_probit(noise, ["spread", "noise"])
check(f"a pure-noise regressor gets a coefficient near zero ({fit_noise.params['noise']:+.2f})",
      abs(fit_noise.params["noise"]) < 0.15)

print("\nEXPANDING WINDOW\n")

oos = expanding_forecasts(panel, ["spread"], start="1975-01-01")
check("no forecast before the start date", oos.loc[:"1974-12-01"].isna().all())
check("forecasts exist for every month from the start date", oos.loc["1975-01-01":].notna().all())
check(f"out-of-sample AUC on planted data is high ({auc(panel['recession_ahead'].loc[oos.notna()], oos.dropna()):.3f})",
      auc(panel["recession_ahead"].loc[oos.notna()], oos.dropna()) > 0.8)

# Mutation test. Corrupt everything after a cut date; every forecast made at or
# before it must be bit-identical, and the ones after must move.
cut = pd.Timestamp("1995-06-01")
mutated = panel.copy()
after = mutated.index > cut
mutated.loc[after, "spread"] = rng.normal(0, 3, after.sum())
mutated.loc[after, "recession_ahead"] = rng.integers(0, 2, after.sum()).astype(float)
oos_mut = expanding_forecasts(mutated, ["spread"], start="1975-01-01")
check("mutating the future leaves every forecast up to the cut date unchanged",
      np.array_equal(oos.loc[:cut].to_numpy(), oos_mut.loc[:cut].to_numpy(), equal_nan=True))
check("mutating the future changes forecasts after the cut date",
      not np.allclose(oos.loc[cut:].iloc[HORIZON + 2:], oos_mut.loc[cut:].iloc[HORIZON + 2:]))

# Labels are not knowable until 12 months after the feature date, so the label at
# the cut date must not enter the forecast until 12 months later.
leaky = panel.copy()
leaky.loc[cut, "recession_ahead"] = 1.0 - leaky.loc[cut, "recession_ahead"]
oos_leak = expanding_forecasts(leaky, ["spread"], start="1975-01-01")
first_diff = (oos_leak != oos).loc[oos.notna()].idxmax()
check(f"flipping the label at the cut date first moves the forecast {HORIZON} months later",
      first_diff == cut + pd.DateOffset(months=HORIZON))

oos_lag = expanding_forecasts(panel, ["spread"], start="1975-01-01", announce_lag=6)
first_diff_lag = (expanding_forecasts(leaky, ["spread"], start="1975-01-01", announce_lag=6)
                  != oos_lag).loc[oos_lag.notna()].idxmax()
check("an announcement lag delays the label's entry by exactly that many months",
      first_diff_lag == cut + pd.DateOffset(months=HORIZON + 6))

print("\nMETRICS\n")

y = rng.integers(0, 2, 5000).astype(float)
random_auc = auc(y, rng.uniform(size=5000))
check(f"AUC of a random predictor is about 0.5 ({random_auc:.3f})", abs(random_auc - 0.5) < 0.03)
check("AUC of a perfect predictor is 1", auc(y, y) == 1.0)
check("AUC of an inverted perfect predictor is 0", auc(y, 1 - y) == 0.0)
check("AUC of a constant predictor is 0.5", auc(y, np.full(5000, 0.3)) == 0.5)
fp, tp = roc_curve(y, y)
check("ROC curve runs from (0,0) to (1,1)", fp[0] == 0 and tp[0] == 0 and fp[-1] == 1 and tp[-1] == 1)

p_cal = rng.uniform(0.02, 0.98, 20000)
y_cal = (rng.uniform(size=20000) < p_cal).astype(float)
table = calibration(y_cal, p_cal, bins=10)
check("a perfectly calibrated forecast sits on the diagonal in every decile",
      (np.abs(table["forecast"] - table["observed"]) < 0.04).all())
check("calibration deciles have equal counts", table["n"].nunique() == 1)
half = calibration(y_cal, p_cal / 2, bins=10)
check("halving the probabilities shows as underconfidence in every decile",
      (half["observed"] > half["forecast"]).all())

check("Brier of a perfect forecast is 0", brier(y, y) == 0.0)
check("Brier of a constant 0.5 forecast is 0.25", brier(y, np.full(5000, 0.5)) == 0.25)
check("Brier of the true probability beats a constant on calibrated data",
      brier(y_cal, p_cal) < brier(y_cal, np.full(20000, y_cal.mean())))

print("\nINVERSIONS\n")

s = pd.Series([1, 1, -1, -1, -1, 1, -1, 1, 1], index=pd.date_range("2000-01-01", periods=9, freq="MS"))
ep = inversion_episodes(s)
check("two inversion episodes found with the right lengths",
      len(ep) == 2 and list(ep["months"]) == [3, 1])
check("a still-open inversion at the end of the sample is closed at the last date",
      inversion_episodes(pd.Series([1, -1, -1], index=s.index[:3])).iloc[0]["end"] == s.index[2])

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
