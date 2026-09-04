"""Run the full diamond-price modelling pipeline and write tables and figures to reports/."""
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import (CLARITY_ORDER, COLOR_ORDER, CUT_ORDER, REPORTS, SEED, audit,  # noqa: E402
                  clean, load_raw, split)
from model import (CANDIDATE_TERMS, GROUPS, bootstrap_coefficients, build_features,  # noqa: E402
                   cv_predict_dollars, dollar_metrics, elastic_grid, fit_ols,
                   information_criteria, log_pipeline, nested_cv, one_hot_features,
                   penalised, repeated_cv, repeated_folds, ridge_grid, robust_inference,
                   select_backward, smearing, stability_summary, tune, vif)

START = time.time()
os.makedirs(REPORTS, exist_ok=True)
pd.set_option("display.width", 160)


def save(frame, name):
    """Write a table to reports/ and return it."""
    frame.to_csv(os.path.join(REPORTS, name))
    return frame


def figure(name):
    """Save and close the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, name), dpi=140)
    plt.close()


def heading(text):
    print(f"\n{text}\n" + "-" * len(text))


heading("1. DATA AUDIT")

raw = load_raw()
flags = save(audit(raw), "audit.csv")
print(f"raw file: {raw.shape[0]} rows x {raw.shape[1]} columns")
print(flags.to_string(index=False))

df = clean(raw)
role = split(df, SEED)
selection = df[role == "selection"].reset_index(drop=True)
inference = df[role == "inference"].reset_index(drop=True)
test = df[role == "test"].reset_index(drop=True)
print(f"\nsample flow: {len(raw)} raw -> {len(df)} modelling "
      f"({len(raw) - len(df)} removed: impossible dimensions and exact duplicates)")
print(f"still flagged inside the modelling sample: {int(df['flagged'].sum())}")
print(f"selection {len(selection)} | inference {len(inference)} | test {len(test)}")

price = df["price"]
print(f"price ${price.min()} to ${price.max()}, median ${price.median():.0f}, "
      f"skew {price.skew():.2f}; log price skew {np.log(price).skew():.2f}")


heading("2. EDA AND THE CONFOUNDING CHECK")

quality = []
for column, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)]:
    grouped = selection.groupby(column, observed=True)
    for grade in order:
        block = grouped.get_group(grade)
        quality.append({"variable": column, "grade": grade, "n": len(block),
                        "mean_price": block["price"].mean(),
                        "mean_carat": block["carat"].mean()})
quality = pd.DataFrame(quality)

bands = pd.qcut(selection["carat"], 5, labels=[f"Q{i}" for i in range(1, 6)])
within = []
for column, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)]:
    table = (selection.assign(band=bands, log_price=np.log(selection["price"]))
             .groupby(["band", column], observed=True)["log_price"].mean().unstack())
    within.append(table.reindex(columns=order).assign(variable=column))
within = pd.concat(within)
save(quality, "quality_raw_means.csv")
save(within, "quality_within_carat_band.csv")

print("raw mean price by grade, worst to best (the naive read):")
print(quality.pivot(index="grade", columns="variable",
                   values="mean_price").reindex(
    sorted(set(CUT_ORDER + COLOR_ORDER + CLARITY_ORDER))).round(0).to_string())
print("\nmean carat by grade, which is where the confounding lives:")
for column, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)]:
    row = quality[quality["variable"] == column].set_index("grade").loc[order, "mean_carat"]
    print(f"  {column:>8}: " + "  ".join(f"{g}={v:.2f}" for g, v in row.items()))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].hist(price, bins=80, color="#4a6fa5")
axes[0, 0].set_title(f"price (skew {price.skew():.2f})")
axes[0, 1].hist(np.log(price), bins=80, color="#4a6fa5")
axes[0, 1].set_title(f"log price (skew {np.log(price).skew():.2f})")
sample = selection.sample(6000, random_state=1)
axes[1, 0].scatter(sample["carat"], sample["price"], s=3, alpha=0.2, color="#4a6fa5")
axes[1, 0].set_xlabel("carat")
axes[1, 0].set_ylabel("price")
axes[1, 0].set_title("levels: curved and fanning")
axes[1, 1].scatter(np.log(sample["carat"]), np.log(sample["price"]), s=3, alpha=0.2,
                   color="#4a6fa5")
axes[1, 1].set_xlabel("log carat")
axes[1, 1].set_ylabel("log price")
axes[1, 1].set_title("logs: straight and even")
figure("eda_target.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (column, order) in zip(axes, [("cut", CUT_ORDER), ("color", COLOR_ORDER),
                                      ("clarity", CLARITY_ORDER)]):
    raw_means = quality[quality["variable"] == column].set_index("grade").loc[order, "mean_price"]
    ax.plot(order, raw_means.values, "o-", label="raw mean price", color="#c0392b")
    ax.set_ylabel("mean price ($)")
    twin = ax.twinx()
    band_means = within[within["variable"] == column].drop(columns="variable")
    twin.plot(order, band_means.loc["Q3", order].values, "s--", color="#2c3e50",
              label="mean log price, middle carat quintile")
    twin.set_ylabel("mean log price")
    ax.set_title(f"{column}: worst grade on the left")
    ax.tick_params(axis="x", rotation=45)
axes[0].legend(loc="upper left", fontsize=8)
figure("quality_confounding.png")


heading("3. BASELINE LADDER AND AIC/BIC SELECTION")

X_selection, centers = build_features(selection)
y_log = np.log(selection["price"].values)
folds = repeated_folds(len(selection), 5, 1, seed=SEED)[0]

LADDERS = {
    "carat only": ["log_carat"],
    "carat + curvature": ["log_carat", "log_carat_sq"],
    "carat + quality": ["log_carat", "log_carat_sq", "cut_score", "color_score",
                        "clarity_score"],
    "full 18-term": CANDIDATE_TERMS,
}

aic_terms, aic_value = select_backward(X_selection, y_log, "aic")
bic_terms, bic_value = select_backward(X_selection, y_log, "bic")
LADDERS["AIC selected"] = aic_terms
LADDERS["BIC selected"] = bic_terms
print(f"AIC keeps {len(aic_terms)} terms, dropping "
      f"{sorted(set(CANDIDATE_TERMS) - set(aic_terms)) or 'nothing'}")
print(f"BIC keeps {len(bic_terms)} terms, dropping "
      f"{sorted(set(CANDIDATE_TERMS) - set(bic_terms)) or 'nothing'}")

rows = []
mean_prediction = np.full(len(selection), np.nan)
for train, held in folds:
    mean_prediction[held] = selection["price"].values[train].mean()
rows.append({"model": "mean price", "terms": 0,
             **dollar_metrics(selection["price"].values, mean_prediction)})

level_prediction = np.full(len(selection), np.nan)
for train, held in folds:
    fit = fit_ols(selection[["carat"]].iloc[train], selection["price"].values[train])
    level_prediction[held] = fit.predict(
        sm.add_constant(selection[["carat"]].iloc[held].values, has_constant="add"))
rows.append({"model": "carat only, price in dollars", "terms": 1,
             **dollar_metrics(selection["price"].values, level_prediction)})

for name, terms in LADDERS.items():
    pipeline = log_pipeline(penalised("ridge", alpha=1e-9), terms)
    predicted, train_rmse = cv_predict_dollars(pipeline, selection, folds)
    criteria = information_criteria(X_selection[terms], y_log)
    rows.append({"model": name, **criteria,
                 **dollar_metrics(selection["price"].values, predicted),
                 "train_rmse": train_rmse})

one_hot_X, _ = one_hot_features(selection)
one_hot_pipeline = log_pipeline(penalised("ridge", alpha=1e-9), encoding="one_hot")
predicted, train_rmse = cv_predict_dollars(one_hot_pipeline, selection, folds)
rows.append({"model": "one-hot quality benchmark", **information_criteria(one_hot_X, y_log),
             **dollar_metrics(selection["price"].values, predicted), "train_rmse": train_rmse})

ladder = pd.DataFrame(rows).set_index("model")
save(ladder, "baseline_ladder.csv")
print("\n" + ladder.round(3).to_string())


heading("4. REGULARISATION")

GRIDS = {"ridge": (penalised("ridge"), ridge_grid()),
         "lasso": (penalised("lasso", max_iter=20000), ridge_grid(-6, 1, 25)),
         "elastic_net": (penalised("elastic_net", max_iter=20000), elastic_grid(-6, 1))}

tuned, paths = {}, {}
for name, (estimator, grid) in GRIDS.items():
    search, choice, table = tune(estimator, grid, selection, CANDIDATE_TERMS, seed=SEED)
    tuned[name] = (search, choice)
    paths[name] = table
    print(f"{name:>12}: best {search.best_params_} | one-SE choice {choice} "
          f"| CV log-RMSE {-search.best_score_:.5f}")
print("the one-SE choice is what the stability and final-test sections use")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name in zip(axes, GRIDS):
    table = paths[name]
    if "param_model__l1_ratio" in table:
        for ratio, block in table.groupby("param_model__l1_ratio"):
            ax.plot(block["param_model__alpha"].astype(float),
                    -block["mean_test_score"], label=f"l1={ratio}")
        ax.legend(fontsize=7)
    else:
        ax.plot(table["param_model__alpha"].astype(float), -table["mean_test_score"])
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("CV RMSE, log price")
    ax.set_title(name)
figure("regularisation_curves.png")

alphas = np.logspace(-6, 0, 40)
coefficient_path = []
for alpha in alphas:
    fitted = log_pipeline(penalised("lasso", alpha=alpha, max_iter=20000),
                          CANDIDATE_TERMS).fit(selection, y_log)
    coefficient_path.append(pd.Series(fitted.named_steps["model"].coef_,
                                      index=CANDIDATE_TERMS, name=alpha))
coefficient_path = pd.DataFrame(coefficient_path)
save(coefficient_path, "lasso_path.csv")
plt.figure(figsize=(8, 5))
for term in CANDIDATE_TERMS:
    plt.plot(alphas, coefficient_path[term], label=term)
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("standardised coefficient")
plt.title("lasso coefficient path")
plt.legend(fontsize=6, ncol=2)
figure("lasso_path.png")

nonzero = (coefficient_path != 0).sum(axis=1)
print(f"\nlasso terms surviving: {nonzero.iloc[0]} at alpha={alphas[0]:.0e}, "
      f"{nonzero.iloc[-1]} at alpha={alphas[-1]:.2f}")

nested = {}
for name, (estimator, grid) in GRIDS.items():
    nested[name] = nested_cv(estimator, grid, selection, CANDIDATE_TERMS, seed=SEED)
    print(f"{name:>12}: nested outer RMSE ${nested[name]['rmse'].mean():.0f} "
          f"(sd ${nested[name]['rmse'].std():.0f}), alphas chosen "
          f"{sorted(nested[name]['alpha'].round(6).unique())}")
save(pd.concat(nested, names=["model"]), "nested_cv.csv")


heading("5. REPEATED-CV AND BOOTSTRAP STABILITY")

STABILITY_MODELS = {
    "OLS full": log_pipeline(penalised("ridge", alpha=1e-9), CANDIDATE_TERMS),
    "OLS BIC": log_pipeline(penalised("ridge", alpha=1e-9), bic_terms),
    "ridge": log_pipeline(penalised("ridge", **{k.replace("model__", ""): v
                                                for k, v in tuned["ridge"][1].items()}),
                          CANDIDATE_TERMS),
    "lasso": log_pipeline(penalised("lasso", max_iter=20000,
                                    **{k.replace("model__", ""): v
                                       for k, v in tuned["lasso"][1].items()}),
                          CANDIDATE_TERMS),
    "elastic net": log_pipeline(penalised("elastic_net", max_iter=20000,
                                          **{k.replace("model__", ""): v
                                             for k, v in tuned["elastic_net"][1].items()}),
                                CANDIDATE_TERMS),
}

cv_rows, cv_coefficients = {}, {}
for name, pipeline in STABILITY_MODELS.items():
    metrics, coefficients = repeated_cv(pipeline, selection, 5, 5, seed=SEED)
    cv_rows[name] = metrics
    cv_coefficients[name] = coefficients

comparison = pd.DataFrame({
    name: {
        "cv_rmse_mean": m["rmse"].mean(), "cv_rmse_sd": m["rmse"].std(),
        "cv_rmse_p5": m["rmse"].quantile(0.05), "cv_rmse_p95": m["rmse"].quantile(0.95),
        "cv_mae": m["mae"].mean(), "cv_r2": m["r2"].mean(),
        "gap": m["rmse"].mean() - m["train_rmse"].mean(),
        "min_sign_consistency": stability_summary(cv_coefficients[name])["sign_consistency"].min(),
    } for name, m in cv_rows.items()}).T
save(comparison, "repeated_cv_comparison.csv")
print(comparison.round(3).to_string())

plt.figure(figsize=(8, 4.5))
plt.boxplot([cv_rows[n]["rmse"] for n in cv_rows], tick_labels=list(cv_rows))
plt.ylabel("fold RMSE ($)")
plt.title("repeated 5-fold CV, 25 folds per model")
figure("cv_distribution.png")

boot_fixed = bootstrap_coefficients(selection, CANDIDATE_TERMS, draws=500, seed=SEED)
boot_summary = stability_summary(boot_fixed)
boot_summary["ci_low"] = boot_fixed.quantile(0.025)
boot_summary["ci_high"] = boot_fixed.quantile(0.975)
save(boot_summary, "bootstrap_fixed.csv")
print("\nbootstrap on the locked 18-term model, 500 draws, standardised coefficients:")
print(boot_summary.round(3).to_string())

boot_selected = bootstrap_coefficients(selection, CANDIDATE_TERMS, draws=100, seed=SEED + 1,
                                       reselect=True)
selection_frequency = (boot_selected != 0).mean().sort_values(ascending=False)
save(selection_frequency.to_frame("selection_frequency"), "bootstrap_selection.csv")
print("\nAIC re-selection inside 100 bootstrap draws, inclusion frequency:")
print(selection_frequency.round(2).to_string())

plt.figure(figsize=(9, 5))
order = boot_summary["mean"].abs().sort_values().index
plt.barh(range(len(order)), boot_summary.loc[order, "mean"],
         xerr=boot_summary.loc[order, "sd"], color="#4a6fa5")
plt.yticks(range(len(order)), order, fontsize=8)
plt.xlabel("standardised coefficient")
plt.title("bootstrap coefficient distribution, 500 draws")
figure("bootstrap_coefficients.png")


heading("6. DIAGNOSTICS")

locked_terms = bic_terms
locked_fit = fit_ols(X_selection[locked_terms], y_log)
level_target_fit = fit_ols(X_selection[locked_terms], selection["price"].values)

vif_table = save(vif(X_selection[locked_terms]).to_frame(), "vif.csv")
design = sm.add_constant(X_selection[locked_terms].values)
condition_number = np.linalg.cond(design / np.linalg.norm(design, axis=0))
print(f"condition number of the locked design: {condition_number:.1f}")
print(vif_table.round(2).to_string())

bp_log = sm.stats.diagnostic.het_breuschpagan(locked_fit.resid, design)
bp_level = sm.stats.diagnostic.het_breuschpagan(level_target_fit.resid, design)
reset = sm.stats.diagnostic.linear_reset(locked_fit, power=2, use_f=True)
print(f"\nBreusch-Pagan, log target: LM {bp_log[0]:.0f}, p {bp_log[1]:.2e}")
print(f"Breusch-Pagan, level target: LM {bp_level[0]:.0f}, p {bp_level[1]:.2e}")
print(f"RESET (power 2): F {reset.fvalue:.1f}, p {reset.pvalue:.2e}")
print(f"residual skew {pd.Series(locked_fit.resid).skew():.2f}, "
      f"kurtosis {pd.Series(locked_fit.resid).kurtosis():.2f}, "
      f"Jarque-Bera p {sm.stats.stattools.jarque_bera(locked_fit.resid)[1]:.2e}")

influence = locked_fit.get_influence()
cooks = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
studentised = influence.resid_studentized_internal
threshold = 4 / len(selection)
print(f"Cook's D above 4/n: {int((cooks > threshold).sum())} rows "
      f"({100 * (cooks > threshold).mean():.2f}%), max {cooks.max():.4f}")

keep = cooks <= threshold
sensitivity_fit = fit_ols(X_selection.loc[keep, locked_terms], y_log[keep])
shift = pd.Series(sensitivity_fit.params[1:] - locked_fit.params[1:], index=locked_terms)
print("largest coefficient shift after dropping high-influence rows: "
      f"{shift.abs().idxmax()} moves {shift.abs().max():.4f} "
      f"({100 * shift.abs().max() / abs(locked_fit.params[1:][shift.abs().values.argmax()]):.1f}%)")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].scatter(level_target_fit.fittedvalues[::5], level_target_fit.resid[::5], s=2,
                   alpha=0.2, color="#c0392b")
axes[0, 0].axhline(0, color="k", lw=0.7)
axes[0, 0].set_title("level target: residual variance fans out")
axes[0, 0].set_xlabel("fitted price ($)")
axes[0, 1].scatter(locked_fit.fittedvalues[::5], locked_fit.resid[::5], s=2, alpha=0.2,
                   color="#4a6fa5")
axes[0, 1].axhline(0, color="k", lw=0.7)
axes[0, 1].set_title("log target: roughly even band")
axes[0, 1].set_xlabel("fitted log price")
sm.qqplot(locked_fit.resid, line="s", ax=axes[1, 0], markersize=1, alpha=0.3)
axes[1, 0].set_title("log-target residual Q-Q")
axes[1, 1].scatter(leverage[::5], studentised[::5], s=3, alpha=0.2, color="#4a6fa5")
axes[1, 1].set_xlabel("leverage")
axes[1, 1].set_ylabel("studentised residual")
axes[1, 1].set_title(f"influence, max Cook's D {cooks.max():.4f}")
figure("diagnostics.png")


heading("7. INFERENCE ON THE HELD-OUT INFERENCE SPLIT")

X_inference, _ = build_features(inference, centers)
y_inference = np.log(inference["price"].values)
robust_fit, coefficient_table, joint = robust_inference(
    X_inference[locked_terms], y_inference,
    {g: [t for t in members if t in locked_terms] for g, members in GROUPS.items()})
save(coefficient_table, "inference_coefficients.csv")
save(joint, "inference_joint_tests.csv")

print(f"locked specification: {len(locked_terms)} terms, "
      f"{len(locked_terms) + 1} estimated coefficients, n = {len(inference)}")
print(f"overall F = {robust_fit.fvalue:.1f}, p = {robust_fit.f_pvalue:.2e}, "
      f"R2 = {robust_fit.rsquared:.4f}, adj R2 = {robust_fit.rsquared_adj:.4f}, "
      f"residual SE = {np.sqrt(robust_fit.mse_resid):.4f}")
plain = fit_ols(X_inference[locked_terms], y_inference)
se_ratio = robust_fit.bse[1:] / plain.bse[1:]
print(f"HC3 vs classical SE ratio: min {se_ratio.min():.2f}, "
      f"median {np.median(se_ratio):.2f}, max {se_ratio.max():.2f}")
print("\n" + coefficient_table.round(4).to_string())
print("\njoint tests:\n" + joint.round(4).to_string())

quality_effect = []
for grade_gap, label in [(1, "one grade")]:
    for term, base in [("clarity_score", 4), ("color_score", 4), ("cut_score", 4)]:
        if term not in locked_terms:
            continue
        square = f"{term}_sq"
        interaction = {"clarity_score": "carat_x_clarity", "color_score": "carat_x_color",
                       "cut_score": "carat_x_cut"}[term]
        beta = dict(zip(locked_terms, robust_fit.params[1:]))
        effect = beta[term] * grade_gap
        if square in beta:
            effect += beta[square] * ((base + grade_gap) ** 2 - base ** 2)
        if interaction in beta:
            effect += beta[interaction] * grade_gap * np.log(selection["carat"].median())
        quality_effect.append({"term": term, "change": label,
                               "log_price_effect": effect,
                               "pct_price_effect": 100 * (np.exp(effect) - 1)})
quality_effect = save(pd.DataFrame(quality_effect).set_index("term"), "practical_effects.csv")
carat_effect = dict(zip(locked_terms, robust_fit.params[1:]))
elasticity = (carat_effect["log_carat"]
              + 2 * carat_effect.get("log_carat_sq", 0) * np.log(selection["carat"].median()))
print(f"\nprice elasticity to carat at the median stone: {elasticity:.2f} "
      f"(a 10% heavier diamond costs {100 * (1.1 ** elasticity - 1):.1f}% more)")
print(quality_effect.round(2).to_string())


heading("8. FINAL TEST, RUN ONCE")

X_test, _ = build_features(test, centers)
y_test_log = np.log(test["price"].values)
development = pd.concat([selection, inference], ignore_index=True)
final_rows = []
for name, pipeline in STABILITY_MODELS.items():
    fitted = pipeline.fit(development, np.log(development["price"].values))
    factor = smearing(np.log(development["price"].values) - fitted.predict(development))
    predicted = np.exp(fitted.predict(test)) * factor
    final_rows.append({"model": name, "smearing": factor,
                       **dollar_metrics(test["price"].values, predicted)})
final = save(pd.DataFrame(final_rows).set_index("model"), "final_test.csv")
print(final.round(3).to_string())
for name in final.index:
    inside = (cv_rows[name]["rmse"].quantile(0.05) <= final.loc[name, "rmse"]
              <= cv_rows[name]["rmse"].quantile(0.95))
    print(f"  {name}: test RMSE ${final.loc[name, 'rmse']:.0f} "
          f"{'inside' if inside else 'OUTSIDE'} the 5-95% repeated-CV band "
          f"[${cv_rows[name]['rmse'].quantile(0.05):.0f}, "
          f"${cv_rows[name]['rmse'].quantile(0.95):.0f}]")

# The final model was fixed before the test split was touched: the BIC specification,
# the simplest model within one CV standard error of the best. Choosing by test RMSE
# here would turn the test set into a selection set.
best = "OLS BIC"
best_pipeline = STABILITY_MODELS[best].fit(development, np.log(development["price"].values))
factor = smearing(np.log(development["price"].values) - best_pipeline.predict(development))
test_prediction = np.exp(best_pipeline.predict(test)) * factor

subgroups = []
for label, key in [("carat quintile", pd.qcut(test["carat"], 5, labels=False)),
                   ("price decile", pd.qcut(test["price"], 10, labels=False)),
                   ("cut", test["cut"]), ("color", test["color"]), ("clarity", test["clarity"])]:
    for group, idx in test.groupby(key, observed=True).groups.items():
        block = test.loc[idx]
        metrics = dollar_metrics(block["price"].values, test_prediction[block.index])
        subgroups.append({"dimension": label, "group": str(group), "n": len(block),
                          "rmse": metrics["rmse"], "mae": metrics["mae"],
                          "mean_error": float((block["price"].values
                                               - test_prediction[block.index]).mean())})
subgroups = save(pd.DataFrame(subgroups), "subgroup_errors.csv")
print(f"\nsubgroup errors for {best} on the test set:")
print(subgroups[subgroups["dimension"] == "price decile"].round(1).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.scatter(test["price"], test_prediction, s=2, alpha=0.15, color="#4a6fa5")
limit = [test["price"].min(), test["price"].max()]
plt.plot(limit, limit, "k--", lw=0.8)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("actual price ($)")
plt.ylabel("predicted price ($)")
plt.title(f"{best} on the untouched test set, RMSE ${final.loc[best, 'rmse']:.0f}")
figure("test_fit.png")

print(f"\ndone in {time.time() - START:.0f}s, tables and figures in {REPORTS}")
