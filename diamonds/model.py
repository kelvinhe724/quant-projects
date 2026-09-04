"""Feature construction, the OLS ladder, AIC/BIC selection, regularisation and stability."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import CLARITY_ORDER, COLOR_ORDER, CUT_ORDER

SCORES = {
    "cut": {level: i + 1 for i, level in enumerate(CUT_ORDER)},
    "color": {level: i + 1 for i, level in enumerate(COLOR_ORDER)},
    "clarity": {level: i + 1 for i, level in enumerate(CLARITY_ORDER)},
}

CANDIDATE_TERMS = [
    "log_carat", "log_carat_sq", "log_volume", "aspect_xy",
    "depth_c", "depth_c_sq", "table_c", "table_c_sq",
    "cut_score", "cut_score_sq", "color_score", "color_score_sq",
    "clarity_score", "clarity_score_sq",
    "carat_x_cut", "carat_x_color", "carat_x_clarity", "table_x_cut",
]

# A term may only leave the model while everything that depends on it has already gone.
PARENTS = {
    "log_carat_sq": ["log_carat"],
    "depth_c_sq": ["depth_c"],
    "table_c_sq": ["table_c"],
    "cut_score_sq": ["cut_score"],
    "color_score_sq": ["color_score"],
    "clarity_score_sq": ["clarity_score"],
    "carat_x_cut": ["log_carat", "cut_score"],
    "carat_x_color": ["log_carat", "color_score"],
    "carat_x_clarity": ["log_carat", "clarity_score"],
    "table_x_cut": ["table_c", "cut_score"],
}

GROUPS = {
    "carat": ["log_carat", "log_carat_sq"],
    "size": ["log_volume", "aspect_xy"],
    "cut": ["cut_score", "cut_score_sq"],
    "color": ["color_score", "color_score_sq"],
    "clarity": ["clarity_score", "clarity_score_sq"],
    "depth": ["depth_c", "depth_c_sq"],
    "table": ["table_c", "table_c_sq"],
    "interactions": ["carat_x_cut", "carat_x_color", "carat_x_clarity", "table_x_cut"],
}


def build_features(df, centers=None):
    """Build the 18-term candidate design matrix.

    Centring constants come from `centers` when supplied, so a fold's features are
    built with the training fold's means and nothing else.
    """
    if centers is None:
        centers = {"depth": float(df["depth"].mean()), "table": float(df["table"].mean())}

    cut = df["cut"].map(SCORES["cut"]).astype(float)
    color = df["color"].map(SCORES["color"]).astype(float)
    clarity = df["clarity"].map(SCORES["clarity"]).astype(float)
    log_carat = np.log(df["carat"])
    depth_c = df["depth"] - centers["depth"]
    table_c = df["table"] - centers["table"]

    X = pd.DataFrame(index=df.index)
    X["log_carat"] = log_carat
    X["log_carat_sq"] = log_carat ** 2
    X["log_volume"] = np.log(df["x"] * df["y"] * df["z"])
    X["aspect_xy"] = df["x"] / df["y"]
    X["depth_c"] = depth_c
    X["depth_c_sq"] = depth_c ** 2
    X["table_c"] = table_c
    X["table_c_sq"] = table_c ** 2
    X["cut_score"] = cut
    X["cut_score_sq"] = cut ** 2
    X["color_score"] = color
    X["color_score_sq"] = color ** 2
    X["clarity_score"] = clarity
    X["clarity_score_sq"] = clarity ** 2
    X["carat_x_cut"] = log_carat * cut
    X["carat_x_color"] = log_carat * color
    X["carat_x_clarity"] = log_carat * clarity
    X["table_x_cut"] = table_c * cut
    return X[CANDIDATE_TERMS], centers


def one_hot_features(df, centers=None):
    """Alternative encoding: quality grades as dummies instead of ordinal scores."""
    X, centers = build_features(df, centers)
    numeric = X[["log_carat", "log_carat_sq", "log_volume", "aspect_xy",
                 "depth_c", "depth_c_sq", "table_c", "table_c_sq"]]
    dummies = []
    for column, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER),
                          ("clarity", CLARITY_ORDER)]:
        cat = pd.Categorical(df[column], categories=order, ordered=True)
        frame = pd.get_dummies(cat, prefix=column, drop_first=True).astype(float)
        frame.index = X.index
        dummies.append(frame)
    return pd.concat([numeric] + dummies, axis=1), centers


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Learn the centring constants on the training fold and apply them downstream."""

    def __init__(self, terms=None, encoding="score"):
        self.terms = terms
        self.encoding = encoding

    def fit(self, X, y=None):
        builder = build_features if self.encoding == "score" else one_hot_features
        _, self.centers_ = builder(X)
        return self

    def transform(self, X):
        builder = build_features if self.encoding == "score" else one_hot_features
        out, _ = builder(X, self.centers_)
        return out[self.terms] if self.terms else out


def fit_ols(X, y):
    """Fit OLS with an intercept and return the statsmodels result."""
    return sm.OLS(np.asarray(y, dtype=float),
                  sm.add_constant(np.asarray(X, dtype=float), has_constant="add")).fit()


def information_criteria(X, y):
    """Return AIC, BIC, R2 and adjusted R2 for an OLS fit of y on X."""
    fit = fit_ols(X, y)
    return {"terms": X.shape[1], "aic": fit.aic, "bic": fit.bic,
            "r2": fit.rsquared, "adj_r2": fit.rsquared_adj}


def select_backward(X, y, criterion="aic", parents=PARENTS):
    """Backward elimination on AIC or BIC, respecting the hierarchy rules.

    A term is only a candidate for removal once every term listing it as a parent
    has already been removed, so main effects never leave before their squares or
    interactions.
    """
    terms = list(X.columns)
    best = information_criteria(X[terms], y)[criterion]
    while len(terms) > 1:
        blocked = {p for t in terms for p in parents.get(t, [])}
        removable = [t for t in terms if t not in blocked]
        scored = [(information_criteria(X[[c for c in terms if c != t]], y)[criterion], t)
                  for t in removable]
        if not scored:
            break
        value, drop = min(scored)
        if value >= best:
            break
        best, terms = value, [t for t in terms if t != drop]
    return terms, best


def smearing(residuals):
    """Duan's smearing factor for exp() back-transformation of a log-target model."""
    return float(np.mean(np.exp(np.asarray(residuals, dtype=float))))


def dollar_metrics(price, predicted_price):
    """RMSE, MAE, median absolute error and R2 on the dollar scale."""
    price = np.asarray(price, dtype=float)
    predicted_price = np.asarray(predicted_price, dtype=float)
    error = price - predicted_price
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "medae": float(np.median(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / np.sum((price - price.mean()) ** 2)),
    }


def log_pipeline(estimator, terms=None, encoding="score"):
    """Feature build, standardise, then fit; everything learned inside the fold."""
    return Pipeline([("features", FeatureBuilder(terms, encoding)),
                     ("scale", StandardScaler()),
                     ("model", estimator)])


def cv_predict_dollars(pipeline, df, folds):
    """Cross-validated dollar predictions from a log-price pipeline.

    The smearing factor is estimated on each training fold's residuals, never on
    the fold being scored.
    """
    y = np.log(df["price"].values)
    out = np.full(len(df), np.nan)
    train_scores = []
    for train, test in folds:
        fitted = pipeline.fit(df.iloc[train], y[train])
        factor = smearing(y[train] - fitted.predict(df.iloc[train]))
        out[test] = np.exp(fitted.predict(df.iloc[test])) * factor
        in_sample = np.exp(fitted.predict(df.iloc[train])) * factor
        train_scores.append(dollar_metrics(df["price"].values[train], in_sample)["rmse"])
    return out, float(np.mean(train_scores))


def repeated_folds(n, n_splits=5, n_repeats=5, seed=0):
    """Fold index pairs for repeated K-fold, one list per repeat."""
    return [list(KFold(n_splits, shuffle=True, random_state=seed + r).split(np.arange(n)))
            for r in range(n_repeats)]


def repeated_cv(pipeline, df, n_splits=5, n_repeats=5, seed=0):
    """Per-fold dollar metrics and standardised coefficients across repeated K-fold."""
    y = np.log(df["price"].values)
    price = df["price"].values
    records, coefficients = [], []
    for repeat, folds in enumerate(repeated_folds(len(df), n_splits, n_repeats, seed)):
        for fold, (train, test) in enumerate(folds):
            fitted = pipeline.fit(df.iloc[train], y[train])
            factor = smearing(y[train] - fitted.predict(df.iloc[train]))
            predicted = np.exp(fitted.predict(df.iloc[test])) * factor
            in_sample = np.exp(fitted.predict(df.iloc[train])) * factor
            row = dollar_metrics(price[test], predicted)
            row.update(repeat=repeat, fold=fold,
                       train_rmse=dollar_metrics(price[train], in_sample)["rmse"])
            records.append(row)
            names = fitted.named_steps["features"].transform(df.iloc[:1]).columns
            coefficients.append(pd.Series(fitted.named_steps["model"].coef_, index=names))
    return pd.DataFrame(records), pd.DataFrame(coefficients)


def tune(estimator, grid, df, terms=None, n_splits=5, seed=0, one_se=True):
    """Grid-search a penalised model on log price and return the fitted search.

    With `one_se` the reported choice is the most regularised setting whose mean CV
    error is within one standard error of the best, which is the spec's rule.
    """
    y = np.log(df["price"].values)
    search = GridSearchCV(log_pipeline(estimator, terms), grid,
                          scoring="neg_root_mean_squared_error",
                          cv=KFold(n_splits, shuffle=True, random_state=seed),
                          return_train_score=True)
    search.fit(df, y)
    table = pd.DataFrame(search.cv_results_)
    if not one_se:
        return search, search.best_params_, table
    fold_columns = [c for c in table.columns if c.startswith("split") and c.endswith("test_score")]
    standard_error = table[fold_columns].std(axis=1, ddof=1) / np.sqrt(len(fold_columns))
    best = table["mean_test_score"].idxmax()
    threshold = table.loc[best, "mean_test_score"] - standard_error[best]
    eligible = table[table["mean_test_score"] >= threshold]
    choice = eligible.loc[eligible["param_model__alpha"].astype(float).idxmax()]
    return search, dict(choice["params"]), table


def nested_cv(estimator, grid, df, terms=None, outer=5, inner=3, seed=0):
    """Outer-fold dollar RMSE where alpha is chosen inside each outer training fold."""
    y = np.log(df["price"].values)
    price = df["price"].values
    rows = []
    for fold, (train, test) in enumerate(
            KFold(outer, shuffle=True, random_state=seed).split(np.arange(len(df)))):
        search = GridSearchCV(log_pipeline(estimator, terms), grid,
                              scoring="neg_root_mean_squared_error",
                              cv=KFold(inner, shuffle=True, random_state=seed + 1))
        search.fit(df.iloc[train], y[train])
        factor = smearing(y[train] - search.predict(df.iloc[train]))
        predicted = np.exp(search.predict(df.iloc[test])) * factor
        row = dollar_metrics(price[test], predicted)
        row.update(fold=fold, **{k.replace("model__", ""): v
                                 for k, v in search.best_params_.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_coefficients(df, terms, draws=500, seed=0, reselect=False):
    """Bootstrap standardised OLS coefficients, optionally rerunning selection.

    With `reselect` the AIC search runs inside every draw, so the inclusion
    frequencies measure selection stability rather than coefficient noise alone.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    rows = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        X, _ = build_features(sample)
        y = np.log(sample["price"].values)
        chosen = select_backward(X[terms], y)[0] if reselect else terms
        standardised = (X[chosen] - X[chosen].mean()) / X[chosen].std(ddof=0)
        fit = fit_ols(standardised, (y - y.mean()) / y.std(ddof=0))
        rows.append(pd.Series(fit.params[1:], index=chosen))
    return pd.DataFrame(rows).reindex(columns=terms).fillna(0.0)


def stability_summary(coefficients):
    """Mean, dispersion, sign consistency and selection frequency per coefficient."""
    median = coefficients.median()
    return pd.DataFrame({
        "mean": coefficients.mean(),
        "sd": coefficients.std(ddof=1),
        "sign_consistency": (np.sign(coefficients) == np.sign(median)).mean(),
        "selection_frequency": (coefficients != 0).mean(),
    }).assign(relative_dispersion=lambda d: d["sd"] / d["mean"].abs())


def robust_inference(X, y, groups=GROUPS):
    """HC3 robust OLS on a locked specification, with joint tests and adjusted p-values."""
    names = list(X.columns)
    fit = sm.OLS(np.asarray(y, dtype=float),
                 sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
                 ).fit(cov_type="HC3")
    intervals = fit.conf_int()
    table = pd.DataFrame({
        "estimate": fit.params[1:],
        "hc3_se": fit.bse[1:],
        "t": fit.tvalues[1:],
        "p": fit.pvalues[1:],
        "ci_low": intervals[1:, 0],
        "ci_high": intervals[1:, 1],
    }, index=names)
    table["p_holm"] = holm(table["p"].values)
    table["p_bh"] = benjamini_hochberg(table["p"].values)

    joint = []
    for group, members in groups.items():
        present = [t for t in members if t in names]
        if not present:
            continue
        restriction = np.zeros((len(present), len(names) + 1))
        for row, term in enumerate(present):
            restriction[row, names.index(term) + 1] = 1
        test = fit.f_test(restriction)
        joint.append({"group": group, "terms": len(present),
                      "f": float(np.squeeze(test.fvalue)), "p": float(test.pvalue)})
    return fit, table, pd.DataFrame(joint, columns=["group", "terms", "f", "p"]).set_index("group")


def holm(p):
    """Holm step-down family-wise adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order]).clip(max=1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def benjamini_hochberg(p):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def vif(X):
    """Variance inflation factor per column, from the R2 of each column on the rest."""
    out = {}
    for column in X.columns:
        others = X.drop(columns=[column])
        r2 = fit_ols(others, X[column]).rsquared
        out[column] = 1 / max(1 - r2, 1e-12)
    return pd.Series(out, name="vif")


def ridge_grid(low=-4, high=4, n=25):
    """Log-spaced alpha grid for the penalised models."""
    return {"model__alpha": np.logspace(low, high, n)}


def elastic_grid(low=-4, high=2, n=15, ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95)):
    """Alpha and l1_ratio grid for elastic net."""
    return {"model__alpha": np.logspace(low, high, n), "model__l1_ratio": list(ratios)}


def penalised(name, **kwargs):
    """Return an unfitted penalised estimator by name."""
    return {"ridge": Ridge, "lasso": Lasso, "elastic_net": ElasticNet}[name](**kwargs)
