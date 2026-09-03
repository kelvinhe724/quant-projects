"""Logistic baseline, LightGBM with early stopping, and k-fold CV."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    verbosity=-1,
)


def prepare(df):
    """Split the target off and cast text columns to category.

    LightGBM takes categories and NaNs natively, so no imputation or one-hot.
    """
    y = df["TARGET"]
    X = df.drop(columns="TARGET")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X, y


def baseline_matrix(X):
    # logistic takes neither NaNs nor categories; imputation is in the pipeline
    return pd.get_dummies(X, dummy_na=False)


def fit_baseline(X_tr, y_tr, X_va, y_va):
    """Returns (model, validation AUC)."""
    Xt, Xv = baseline_matrix(X_tr), baseline_matrix(X_va)
    Xv = Xv.reindex(columns=Xt.columns, fill_value=0)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000))
    model.fit(Xt, y_tr)
    auc = roc_auc_score(y_va, model.predict_proba(Xv)[:, 1])
    return model, auc


def fit_lgbm(X_tr, y_tr, X_va, y_va, num_boost_round=2000):
    """Early stopping on validation AUC. Returns (booster, validation AUC)."""
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=num_boost_round,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    auc = roc_auc_score(y_va, booster.predict(X_va, num_iteration=booster.best_iteration))
    return booster, auc


def importance(booster):
    """Gain-based feature importance, descending."""
    s = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    return s.sort_values(ascending=False)


def cv_auc(X, y, n_splits=5, seed=0):
    """Per-fold LightGBM AUCs from a stratified k-fold.

    Early stopping gets its own split carved out of the training fold; the
    scored fold never touches training, so the AUC is a clean estimate.
    """
    aucs = []
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        X_tr, X_es, y_tr, y_es = train_test_split(
            X.iloc[tr], y.iloc[tr], test_size=0.2, stratify=y.iloc[tr],
            random_state=seed)
        booster, _ = fit_lgbm(X_tr, y_tr, X_es, y_es)
        pred = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        aucs.append(roc_auc_score(y.iloc[va], pred))
    return np.array(aucs)
