"""Load the data, run baseline vs LightGBM, cross-validate, write charts."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from data import load
from gb import baseline_matrix, cv_auc, fit_baseline, fit_lgbm, importance, prepare

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS, exist_ok=True)

df, source = load()
print(f"Data source: {source}  ({len(df)} rows, {df.shape[1] - 1} features, "
      f"default rate {df['TARGET'].mean():.3f})")

X, y = prepare(df)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
print(f"\nSplit: {len(X_tr)} train / {len(X_va)} valid (stratified)")

base_model, base_auc = fit_baseline(X_tr, y_tr, X_va, y_va)
print(f"\nBaseline logistic regression AUC: {base_auc:.4f}")

booster, gb_auc = fit_lgbm(X_tr, y_tr, X_va, y_va)
print(f"LightGBM AUC:                     {gb_auc:.4f}  "
      f"({booster.best_iteration} trees after early stopping)")

aucs = cv_auc(X, y, n_splits=5)
print(f"\n5-fold CV AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}  "
      f"folds: {np.round(aucs, 4)}")

Xt, Xv = baseline_matrix(X_tr), baseline_matrix(X_va)
Xv = Xv.reindex(columns=Xt.columns, fill_value=0)
p_base = base_model.predict_proba(Xv)[:, 1]
p_gb = booster.predict(X_va, num_iteration=booster.best_iteration)

fpr_b, tpr_b, _ = roc_curve(y_va, p_base)
fpr_g, tpr_g, _ = roc_curve(y_va, p_gb)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_g, tpr_g, label=f"LightGBM (AUC {gb_auc:.3f})")
ax.plot(fpr_b, tpr_b, label=f"Logistic (AUC {base_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(f"ROC - default prediction ({source} data)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "roc.png"), dpi=120)

imp = importance(booster).head(15)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp.index, imp.values)
ax.set_title("LightGBM feature importance (gain), top 15")
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "importance.png"), dpi=120)

fig, ax = plt.subplots(figsize=(6, 5))
for p, label in ((p_gb, "LightGBM"), (p_base, "Logistic")):
    obs, pred = calibration_curve(y_va, p, n_bins=10, strategy="quantile")
    ax.plot(pred, obs, "o-", label=label)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
ax.set_xlabel("Predicted default probability")
ax.set_ylabel("Observed default rate")
ax.set_title("Calibration (10 quantile bins, validation set)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "calibration.png"), dpi=120)

print("\nTop 10 features by gain:")
for name, v in importance(booster).head(10).items():
    print(f"  {name:28s} {v:12.0f}")

print("\nCharts saved to reports/: roc.png, importance.png, calibration.png")
