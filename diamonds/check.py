"""Offline checks on synthetic data with planted coefficients.

Everything here is generated, so the right answer is known before the code runs.
The estimators must recover it, and the machinery around them must behave the way
the report claims it does.

Run: python3 check.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge

from model import (FeatureBuilder, benjamini_hochberg, build_features, dollar_metrics,
                   fit_ols, holm, information_criteria, robust_inference, select_backward,
                   smearing, stability_summary, vif)

rng = np.random.default_rng(11)
N = 4000

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def planted(n=N, k=5, noise=0.5, seed=11):
    """Design matrix with known coefficients and one column that does nothing."""
    local = np.random.default_rng(seed)
    X = pd.DataFrame(local.normal(size=(n, k)), columns=[f"x{i}" for i in range(k)])
    beta = np.array([2.0, -1.5, 0.8, 0.0, 0.3][:k])
    y = 1.0 + X.values @ beta + local.normal(0, noise, n)
    return X, pd.Series(y, name="y"), beta


print("OLS RECOVERY\n")

X, y, beta = planted()
fit = fit_ols(X, y)
check("OLS recovers the planted intercept of 1.0 within 0.05",
      abs(fit.params[0] - 1.0) < 0.05)
check("OLS recovers every planted slope within 0.05",
      np.max(np.abs(fit.params[1:] - beta)) < 0.05)
check("the planted zero coefficient is not significant at 5%",
      fit.pvalues[4] > 0.05)
check("every non-zero planted coefficient is significant at 1%",
      np.all(fit.pvalues[[1, 2, 3, 5]] < 0.01))

X_big, y_big, beta_big = planted(n=200_000, noise=0.5, seed=12)
check("estimation error shrinks when the sample grows 50x",
      np.max(np.abs(fit_ols(X_big, y_big).params[1:] - beta_big))
      < np.max(np.abs(fit.params[1:] - beta)))

correlated = X.copy()
correlated["x5"] = correlated["x0"] + rng.normal(0, 0.05, len(correlated))
check("VIF flags a near-duplicate column and leaves independent columns near 1",
      vif(correlated)["x5"] > 50 and vif(correlated)["x1"] < 1.5)


print("\nRIDGE SHRINKAGE\n")

standardised = (X - X.mean()) / X.std(ddof=0)
norms = [np.abs(Ridge(alpha=a).fit(standardised, y).coef_).sum()
         for a in [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6]]
check("ridge L1 norm falls monotonically as alpha rises",
      all(a > b for a, b in zip(norms, norms[1:])))
check("ridge at alpha=1e6 has shrunk the coefficients below 1% of the OLS norm",
      norms[-1] < 0.01 * norms[0])
check("ridge at alpha=0 reproduces the OLS slopes",
      np.max(np.abs(Ridge(alpha=1e-12).fit(standardised, y).coef_
                    - fit_ols(standardised, y).params[1:])) < 1e-6)
check("ridge is biased toward zero: every fitted slope is smaller in magnitude",
      np.all(np.abs(Ridge(alpha=50).fit(standardised, y).coef_)
             <= np.abs(fit_ols(standardised, y).params[1:]) + 1e-9))


print("\nLASSO SELECTION\n")

lasso_coef = pd.Series(Lasso(alpha=0.05).fit(standardised, y).coef_, index=X.columns)
check("lasso sets the genuinely irrelevant feature x3 exactly to zero",
      lasso_coef["x3"] == 0.0)
check("lasso keeps every genuinely relevant feature non-zero",
      np.all(lasso_coef.drop("x3") != 0.0))

path = [(a, int((Lasso(alpha=a).fit(standardised, y).coef_ != 0).sum()))
        for a in [0.001, 0.01, 0.1, 0.5, 2.0]]
check("the number of surviving lasso coefficients falls as alpha rises",
      all(a >= b for (_, a), (_, b) in zip(path, path[1:])) and path[-1][1] == 0)

X_corr, y_corr, _ = planted(seed=13)
X_corr["x0_twin"] = X_corr["x0"] + rng.normal(0, 0.02, len(X_corr))
twin_standardised = (X_corr - X_corr.mean()) / X_corr.std(ddof=0)
twin_coef = Lasso(alpha=0.02).fit(twin_standardised, y_corr).coef_
check("with two near-identical columns lasso splits or drops one, so the pair "
      "cannot be read as two separate effects",
      min(abs(twin_coef[0]), abs(twin_coef[-1])) < 0.5 * max(abs(twin_coef[0]),
                                                             abs(twin_coef[-1])))


print("\nHC3 ROBUST STANDARD ERRORS\n")

x = rng.normal(size=N)
homo = 1.0 + 2.0 * x + rng.normal(0, 1.0, N)
hetero = 1.0 + 2.0 * x + rng.normal(0, 1.0, N) * np.abs(x) * 2

design = sm.add_constant(x)
plain_homo = sm.OLS(homo, design).fit()
robust_homo = sm.OLS(homo, design).fit(cov_type="HC3")
plain_hetero = sm.OLS(hetero, design).fit()
robust_hetero = sm.OLS(hetero, design).fit(cov_type="HC3")

check("under homoskedastic noise HC3 and OLS standard errors agree within 10%",
      abs(robust_homo.bse[1] / plain_homo.bse[1] - 1) < 0.10)
check("under deliberately heteroskedastic noise HC3 differs from OLS by over 25%",
      abs(robust_hetero.bse[1] / plain_hetero.bse[1] - 1) > 0.25)
check("the point estimates are untouched by the covariance choice",
      np.allclose(plain_hetero.params, robust_hetero.params))
white_design = sm.add_constant(np.column_stack([x, x ** 2]))
check("White's test rejects homoskedasticity on the heteroskedastic sample only",
      sm.stats.diagnostic.het_breuschpagan(plain_hetero.resid, white_design)[1] < 0.01
      and sm.stats.diagnostic.het_breuschpagan(plain_homo.resid, white_design)[1] > 0.01)
check("plain Breusch-Pagan misses it, because the variance here is symmetric in x",
      sm.stats.diagnostic.het_breuschpagan(plain_hetero.resid, design)[1] > 0.01)

frame = pd.DataFrame({"a": x, "b": rng.normal(size=N)})
_, table, joint = robust_inference(frame, 1.0 + 2.0 * x + rng.normal(0, 1, N),
                                   groups={"real": ["a"], "noise": ["b"]})
check("the joint F-test rejects for the group that carries signal",
      joint.loc["real", "p"] < 1e-6)
check("the joint F-test does not reject for the group that is pure noise",
      joint.loc["noise", "p"] > 0.05)
check("HC3 confidence intervals bracket the planted slope of 2.0",
      table.loc["a", "ci_low"] < 2.0 < table.loc["a", "ci_high"])
_, hetero_table, _ = robust_inference(pd.DataFrame({"x": x}), hetero, groups={})
check("robust_inference really uses HC3: on heteroskedastic data its SE differs "
      "from the classical one by over 25%",
      abs(hetero_table.loc["x", "hc3_se"] / plain_hetero.bse[1] - 1) > 0.25)


print("\nAIC AND BIC SELECTION\n")

X_true, y_true, _ = planted(n=2000, k=3, seed=21)
X_over = X_true.copy()
for j in range(8):
    X_over[f"junk{j}"] = rng.normal(size=len(X_over))

true_ic = information_criteria(X_true, y_true)
over_ic = information_criteria(X_over, y_true)
check("AIC prefers the true model over the same model plus 8 noise columns",
      true_ic["aic"] < over_ic["aic"])
check("BIC prefers the true model, and by a wider margin than AIC",
      true_ic["bic"] < over_ic["bic"]
      and (over_ic["bic"] - true_ic["bic"]) > (over_ic["aic"] - true_ic["aic"]))
check("the overfitted model still has the higher in-sample R2, which is why R2 "
      "cannot be used for selection",
      over_ic["r2"] > true_ic["r2"])

selected_aic, _ = select_backward(X_over, y_true, "aic", parents={})
selected_bic, _ = select_backward(X_over, y_true, "bic", parents={})
check("backward AIC keeps all three true columns",
      set(X_true.columns).issubset(selected_aic))
check("backward AIC discards at least 6 of the 8 noise columns",
      sum(c.startswith("junk") for c in selected_aic) <= 2)
check("backward BIC is at least as parsimonious as backward AIC",
      len(selected_bic) <= len(selected_aic))
check("backward BIC recovers the true model exactly",
      set(selected_bic) == set(X_true.columns))

hierarchy = X_true.copy()
hierarchy["x0_sq"] = hierarchy["x0"] ** 2
kept, _ = select_backward(hierarchy, y_true, "aic",
                          parents={"x0_sq": ["x0"]})
check("hierarchy holds: the square never survives without its main effect",
      "x0" in kept or "x0_sq" not in kept)


print("\nBACK-TRANSFORMATION AND METRICS\n")

log_price = rng.normal(7.5, 0.9, 20_000)
price = np.exp(log_price)
residual = rng.normal(0, 0.4, 20_000)
naive = np.exp(log_price)
corrected = np.exp(log_price) * smearing(residual)
check("exp() of a log-scale prediction underestimates the mean price, and "
      "smearing corrects the level upward",
      corrected.mean() > naive.mean()
      and abs(corrected.mean() / (price * np.exp(0.08)).mean() - 1) < 0.05)
check("smearing on zero residuals is exactly 1", smearing(np.zeros(10)) == 1.0)

truth = np.array([100.0, 200.0, 300.0])
metrics = dollar_metrics(truth, truth + np.array([10.0, -10.0, 20.0]))
check("dollar metrics match hand computation",
      abs(metrics["mae"] - 40 / 3) < 1e-9
      and abs(metrics["rmse"] - np.sqrt(600 / 3)) < 1e-9
      and metrics["medae"] == 10.0)
check("R2 of a perfect prediction is 1", dollar_metrics(truth, truth)["r2"] == 1.0)


print("\nMULTIPLE-TESTING ADJUSTMENT\n")

raw = np.array([0.001, 0.02, 0.04, 0.3, 0.8])
check("Holm adjusted p-values are never below raw", np.all(holm(raw) >= raw))
check("Holm is at least as conservative as Benjamini-Hochberg",
      np.all(holm(raw) >= benjamini_hochberg(raw) - 1e-12))
check("Holm matches the hand-computed step-down values",
      np.allclose(holm(raw), [0.005, 0.08, 0.12, 0.6, 0.8]))
check("Holm enforces monotonicity: 3 x 0.010 = 0.030 carries over 2 x 0.011 = 0.022",
      np.allclose(holm(np.array([0.010, 0.011, 0.5])), [0.03, 0.03, 0.5]))
check("both adjustments preserve the ordering of the raw p-values",
      np.all(np.argsort(holm(raw)) == np.argsort(raw))
      and np.all(np.argsort(benjamini_hochberg(raw)) == np.argsort(raw)))

null_p = pd.Series(rng.uniform(size=200))
check("under a global null Holm almost never produces a rejection",
      (holm(null_p.values) < 0.05).sum() <= 1)


print("\nFEATURE PIPELINE\n")

fake = pd.DataFrame({
    "carat": rng.uniform(0.3, 2.5, 500),
    "cut": rng.choice(["Fair", "Good", "Very Good", "Premium", "Ideal"], 500),
    "color": rng.choice(list("DEFGHIJ"), 500),
    "clarity": rng.choice(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], 500),
    "depth": rng.normal(61.7, 1.4, 500),
    "table": rng.normal(57.5, 2.2, 500),
    "x": rng.uniform(4, 9, 500), "y": rng.uniform(4, 9, 500), "z": rng.uniform(2, 6, 500),
})
fake["price"] = np.exp(8 + 1.7 * np.log(fake["carat"]) + rng.normal(0, 0.15, 500))

builder = FeatureBuilder().fit(fake.iloc[:300])
train_features = builder.transform(fake.iloc[:300])
test_features = builder.transform(fake.iloc[300:])
check("depth is centred on the training mean, so the training column has mean zero",
      abs(train_features["depth_c"].mean()) < 1e-9)
check("the held-out block is centred with the training constant, not its own",
      abs(test_features["depth_c"].mean()) > 1e-9)
check("refitting the builder on the held-out block gives a different constant",
      builder.centers_["depth"] != FeatureBuilder().fit(fake.iloc[300:]).centers_["depth"])

features, _ = build_features(fake)
check("log_carat_sq is exactly the square of log_carat",
      np.allclose(features["log_carat_sq"], features["log_carat"] ** 2))
check("the interaction terms are exactly the product of their parents",
      np.allclose(features["carat_x_cut"],
                  features["log_carat"] * features["cut_score"]))
check("the candidate specification has 18 terms", features.shape[1] == 18)

recovered = fit_ols(features[["log_carat"]], np.log(fake["price"]))
check("OLS on the planted log-log relation recovers the 1.7 elasticity",
      abs(recovered.params[1] - 1.7) < 0.05)


print("\nSTABILITY SUMMARY\n")

stable = pd.DataFrame({"steady": rng.normal(1.0, 0.02, 300),
                       "flipping": rng.normal(0.0, 1.0, 300),
                       "dropped": np.where(rng.uniform(size=300) < 0.4, 0.0, 0.5)})
summary = stability_summary(stable)
check("a steady coefficient scores sign consistency near 1",
      summary.loc["steady", "sign_consistency"] > 0.99)
check("a coefficient centred on zero scores sign consistency near 0.5",
      abs(summary.loc["flipping", "sign_consistency"] - 0.5) < 0.1)
check("selection frequency recovers the planted 60% inclusion rate",
      abs(summary.loc["dropped", "selection_frequency"] - 0.6) < 0.06)
check("relative dispersion separates the steady term from the unstable one",
      summary.loc["steady", "relative_dispersion"]
      < summary.loc["flipping", "relative_dispersion"])


print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
