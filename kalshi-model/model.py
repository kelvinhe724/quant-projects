"""Calibration models: features -> P(YES), optional isotonic or Platt post-calibration, the trade rule, and the desk hook.

Three estimators (logistic regression, histogram gradient boosting, a
two-layer MLP on MPS or CPU) times three post-calibrations (none, isotonic, Platt) make the grid.
Each is a research Alpha: fit(X, y) on the feature table, signal(rows) a
signed Kelly weight per market. The bin-based longshot curve from
../prediction-markets is the same interface, so the walk-forward scores it
next to the models.

The trade rule is stricter than the research project's: a market is
traded only when the model's edge at the mid exceeds the whole quoted
spread, so after crossing to the touch at least half a spread of edge is
left. Costs are the quoted touch plus Kalshi's taker fee, 0.07 x P x (1 - P)
a contract.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from research.alpha import Alpha  # noqa: E402
from research.models import ModelRegistry  # noqa: E402

import candles  # noqa: E402

calibration = candles.research_module("calibration")
kelly = candles.research_module("kelly")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FEE = 0.07
MODELS = ("logit", "hgb", "mlp")
CALIBRATIONS = ("none", "isotonic", "platt")
FEATURES = list(candles.FEATURES)
REPORTS = os.path.join(HERE, "reports")


class MLP(ClassifierMixin, BaseEstimator):
    """Two hidden layers, full-batch Adam on binary cross-entropy, standardised inputs."""

    def __init__(self, hidden=16, epochs=300, lr=0.003, weight_decay=1e-2, seed=0):
        self.hidden, self.epochs, self.lr, self.weight_decay, self.seed = hidden, epochs, lr, weight_decay, seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        torch.manual_seed(self.seed)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(self.hidden, 1)).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.tensor((X - self.mu_) / self.sd_).to(DEVICE)
        yt = torch.tensor(y).to(DEVICE)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(net(xt).squeeze(1), yt)
            loss.backward()
            opt.step()
        self.net_ = net.cpu().eval()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(X))).squeeze(1).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def estimator(model):
    if model == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if model == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                              min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    if model == "mlp":
        return MLP()
    raise ValueError(model)


class Calibrator(Alpha):
    """features -> P(YES); signal() is the signed Kelly weight under the edge-over-spread rule."""

    def __init__(self, model="logit", calibration="none", features=None):
        self.model, self.calibration = model, calibration
        self.features = list(features or FEATURES)
        self.name = f"Calibrator[{model}+{calibration}]"
        self.est = None
        self.categories = {}

    def fit(self, X, y):
        est = estimator(self.model)
        if self.calibration != "none":
            method = "sigmoid" if self.calibration == "platt" else "isotonic"
            est = CalibratedClassifierCV(est, method=method, cv=5)
        self.est = est.fit(X[self.features].to_numpy(dtype=float), np.asarray(y, dtype=int))
        return self

    def predict(self, X):
        p = self.est.predict_proba(X[self.features].to_numpy(dtype=float))[:, 1]
        return np.clip(p, 1e-4, 1 - 1e-4)

    def signal(self, rows):
        b = bets(rows, self.predict(rows))
        w = pd.Series(0.0, index=rows.index)
        w[b.index] = np.where(b["side"] == "yes", b["f"], -b["f"])
        return w


class BiasCurve(Alpha):
    """The published longshot fade: bin the mid, interpolate realised frequency. Same interface, no features."""

    name = "BiasCurve"
    categories = {}

    def fit(self, X, y):
        self.table = calibration.bin_calibration(X["price"].to_numpy(), np.asarray(y))
        return self

    def predict(self, X):
        return np.asarray(calibration.bias_curve(self.table)(X["price"].to_numpy()), dtype=float)

    def signal(self, rows):
        return Calibrator.signal(self, rows)


class Market(Alpha):
    """The mid itself: the calibration to beat, never traded."""

    name = "Market"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.clip(X["price"].to_numpy(dtype=float), 1e-4, 1 - 1e-4)


def make(name):
    if name == "bias":
        return BiasCurve()
    if name == "market":
        return Market()
    model, cal = name.split("+")
    return Calibrator(model, cal)


def fee(cost):
    """Kalshi taker fee a contract at price `cost`, as dollars."""
    return FEE * cost * (1 - cost)


def bets(df, p, spread=None, charge_fee=True, rule="edge_over_spread"):
    """Side, cost, Kelly fraction and settled payoff for the markets the rule trades.

    spread: what is crossed; None means the quoted spread (the touch), 0 the
    mid. rule "edge_over_spread" trades only when the model's edge at the
    mid exceeds the whole quoted spread; "kelly" trades whenever the Kelly
    fraction at the cost is positive, the research project's rule. Index
    is df's, so the frame lines up with the rows it came from.
    """
    spread = df["spread"].to_numpy() if spread is None else np.broadcast_to(np.asarray(spread, float), len(df))
    b = kelly.build_bets(df, p, spread=spread, min_edge=-np.inf)
    b.index = df.index
    b["spread"] = df["spread"].to_numpy()
    b["ticker"] = df["ticker"].to_numpy() if "ticker" in df else df.index
    yes = b["side"] == "yes"
    b["edge_mid"] = np.where(yes, b["p_model"] - b["price"], (1 - b["p_model"]) - (1 - b["price"]))
    if charge_fee:
        c = b["cost"] + fee(b["cost"])
        p_side = np.where(yes, b["p_model"], 1 - b["p_model"])
        b["cost"] = c
        b["f"] = kelly.kelly_fraction(p_side, c)
        b["payoff"] = np.where(b["payoff"] > 0, (1 - c) / c, -1.0)
    keep = b["f"] > 0
    if rule == "edge_over_spread":
        keep &= b["edge_mid"] > b["spread"]
    return b[keep]


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(df, p, **kw):
    """Log-loss, Brier decomposition and cost-honest P&L of predictions p on the rows of df."""
    b = bets(df, p, **kw)
    edge = kelly.per_bet_edge(b)
    r, e = kelly.simulate(b, k=0.25)
    s = kelly.stats(r, e)
    br = calibration.brier(p, df["outcome"].to_numpy())
    return {"n": int(len(df)), "log_loss": log_loss(df["outcome"], p), "brier": float(br["brier"]),
            "reliability": float(br["reliability"]), "resolution": float(br["resolution"]),
            "n_bets": int(edge["n"]), "mean_payoff": float(edge["mean_payoff"]), "t": float(edge["t"]),
            "sharpe": float(s["sharpe"]), "log_growth": float(s["log_growth"]),
            "total_return": float(s["total_return"]), "max_drawdown": float(s["max_drawdown"]),
            "days": int(s["days"])}, r


# desk hook

def registered(role="chosen", reports=REPORTS):
    """The latest registry entry with meta role `role` ("chosen" or "baseline"), as (alpha, entry)."""
    reg = ModelRegistry(os.path.join(reports, "models"))
    e = reg.entries()
    e = e[e["meta"].apply(lambda m: m.get("role") == role)] if len(e) else e
    if e.empty:
        raise FileNotFoundError(f"no {role} model in {reg.index}; run kalshi-model/run.py first")
    return reg.load(e.iloc[-1]["id"])


def live_features(row, model, now=None):
    """The training feature row for one live scanner row: candles up to now, the listing's clock and category."""
    now = int(now or time.time())
    series = row["ticker"].split("-")[0]
    bars = candles.fetch_candles(series, row["ticker"], now) or []
    path = candles.path_features(pd.DataFrame(bars, columns=["ticker", "end_ts", "bid", "ask", "last", "volume",
                                                           "oi"]), now)
    if not np.isfinite(path["quote_age_h"]):
        path["quote_age_h"] = 0.0
    hours = float(row["hours"])
    opened = row.get("open_time")
    duration = ((pd.Timestamp(row["close_time"]) - pd.Timestamp(opened)).total_seconds() / 3600
                if opened else hours)
    mid = (row["bid"] + row["ask"]) / 2
    f = candles.market_features(mid, row["ask"] - row["bid"], hours, duration, model.categories.get(series, ""),
                                path)
    f["price"] = mid
    f["spread"] = row["ask"] - row["bid"]
    return f


def curve(role="chosen", reports=REPORTS):
    """mid, row -> P(YES) from the registered model, for kalshi-desk's scanner (SCANNER_MODEL=kalshi-model)."""
    model, entry = registered(role, reports)

    def f(mid, row):
        X = pd.DataFrame([live_features(row, model)])
        return float(model.predict(X)[0])

    f.needs_row = True
    f.entry = entry
    return f
