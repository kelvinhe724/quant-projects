"""Offline checks for gb.py: synthetic data with planted signal features."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import make_synthetic
from gb import cv_auc, fit_baseline, fit_lgbm, importance, prepare

SIGNAL = ["EXT_SCORE_1", "EXT_SCORE_2", "CREDIT_TO_INCOME"]  # planted in data.py

df = make_synthetic(n=20000, seed=7)
X, y = prepare(df)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

checks = []

def check(name, ok):
    checks.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name)

check(f"target is imbalanced (rate {y.mean():.3f})", 0.02 < y.mean() < 0.20)
check("dataset has missing values", X.isna().any().any())
check("dataset has categorical columns", (X.dtypes == "category").any())

booster, gb_auc = fit_lgbm(X_tr, y_tr, X_va, y_va)
check(f"LightGBM AUC above 0.8 on planted signal (got {gb_auc:.3f})", gb_auc > 0.8)

imp = importance(booster)
top = set(imp.index[:5])
check(f"planted signal features rank in the top 5 by gain (top 5: {list(imp.index[:5])})",
      set(SIGNAL) <= top)

_, base_auc = fit_baseline(X_tr, y_tr, X_va, y_va)
check(f"baseline logistic beats a coin flip (got {base_auc:.3f})", base_auc > 0.6)
check(f"boosting beats the baseline ({gb_auc:.3f} vs {base_auc:.3f})", gb_auc > base_auc)

aucs = cv_auc(X, y, n_splits=5)
check(f"5-fold CV mean AUC above 0.8 (got {np.round(aucs, 3)}, mean {aucs.mean():.3f})",
      aucs.mean() > 0.8)
check(f"CV AUCs are stable across folds (spread {aucs.max() - aucs.min():.3f})",
      aucs.max() - aucs.min() < 0.05)

print()
if checks and all(checks):
    print(f"ALL {len(checks)} CHECKS PASS")
else:
    print(f"{sum(checks)}/{len(checks)} passing")
raise SystemExit(0 if checks and all(checks) else 1)
