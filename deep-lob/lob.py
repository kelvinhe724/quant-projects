"""Labels, three classifiers behind one interface, the purged walk-forward split and the cost-honest P&L.

Timing convention, applied in exactly one place (`pnl`): a bar dated t holds
what printed in [t, t+1), a prediction dated t reads bars <= t, and the
position it implies fills at the price of bar t+1 and earns from there. The
label dated t is the move from bar t to bar t+h.
"""
import copy
import os
import sys

# lightgbm has to load before torch: with torch first, the two OpenMP runtimes crash lightgbm's fit on macOS
import lightgbm
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from purgedcv import WalkForwardSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn

# one CPU thread for torch: its OpenMP pool deadlocks against lightgbm's once both have run (the real training is on MPS)
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "orderbook-imbalance"))
from imbalance import imbalance  # noqa: E402
from research.alpha.base import backtest  # noqa: E402

SYMBOL = "BTCUSDT"
HORIZONS = (5, 10, 30)
# Dead band in bps, fixed before any data was seen: half a bp scaled by the square root of the horizon.
BAND = {h: 0.5 * np.sqrt(h) for h in HORIZONS}
WINDOW = 100
LAGS = (0, 1, 2, 5, 10, 30)
TAUS = (0.1, 0.3)
FEE_BPS = 1.0
CLASSES = (-1, 0, 1)
BPS = 1e4


def labels(mid, h, band=None):
    """Class dated t: sign of the move from bar t to bar t+h outside the dead band, 0 inside, nan past the end."""
    band = BAND[h] if band is None else band
    move = (mid.shift(-h) / mid - 1) * BPS
    y = pd.Series(np.where(move > band, 1.0, np.where(move < -band, -1.0, 0.0)), index=mid.index)
    return y.where(move.notna()), move


def label_frame(mid):
    return pd.DataFrame({h: labels(mid, h)[0] for h in HORIZONS})


def imbalance_feature(frame, depth=5):
    """The imbalance baseline's one input: book imbalance over `depth` levels where there is a book, else the flow imbalance of the last `depth` seconds."""
    if "bid_sz_1" in frame:
        b = frame[[f"bid_sz_{i}" for i in range(1, depth + 1)]].to_numpy()
        a = frame[[f"ask_sz_{i}" for i in range(1, depth + 1)]].to_numpy()
        return pd.Series(imbalance(b, a, depth), index=frame.index)
    b = frame["buy_vol"].rolling(depth, min_periods=1).sum().to_numpy()[:, None]
    a = frame["sell_vol"].rolling(depth, min_periods=1).sum().to_numpy()[:, None]
    return pd.Series(imbalance(b, a, 1), index=frame.index)


def lag_stack(X, lags=LAGS):
    """Every feature at each lag, one wide row per second, for the tabular model."""
    return pd.concat({f"l{k}": X.shift(k) for k in lags}, axis=1).pipe(
        lambda d: d.set_axis([f"{c}_{lag}" for lag, c in d.columns], axis=1))


def features(frame):
    return frame.drop(columns=["mid"])


def _ordered(clf, X):
    """predict_proba with columns in CLASSES order whatever the classifier saw."""
    p = clf.predict_proba(X)
    out = np.zeros((len(X), 3))
    for j, c in enumerate(clf.classes_):
        out[:, CLASSES.index(int(c))] = p[:, j]
    return out


class ImbalanceLogit:
    """The orderbook-imbalance project's regressor as a three-class logistic: one feature, one model per horizon."""
    name = "logit-imbalance"

    def fit(self, frame, Y, val=None):
        x = imbalance_feature(frame).to_numpy()[:, None]
        self.models = {}
        for h in HORIZONS:
            ok = Y[h].notna().to_numpy()
            self.models[h] = LogisticRegression(max_iter=500).fit(x[ok], Y[h].to_numpy()[ok])
        return self

    def proba(self, frame):
        x = imbalance_feature(frame).to_numpy()[:, None]
        return {h: _ordered(m, x) for h, m in self.models.items()}


class LightGBM:
    """Plain LightGBM on the feature block at six lags, one model per horizon, early stopping on the validation slice."""
    name = "lightgbm"

    def __init__(self, n_estimators=300, learning_rate=0.05, num_leaves=31):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

    def fit(self, frame, Y, val=None):
        X = lag_stack(features(frame))
        Xv = lag_stack(features(val[0])) if val is not None else None
        self.models = {}
        for h in HORIZONS:
            ok = Y[h].notna().to_numpy()
            clf = LGBMClassifier(**self.params)
            kw = {}
            if val is not None:
                okv = val[1][h].notna().to_numpy()
                kw = dict(eval_X=Xv[okv], eval_y=val[1][h].to_numpy()[okv],
                          callbacks=[lightgbm.early_stopping(20, verbose=False)])
            self.models[h] = clf.fit(X[ok], Y[h].to_numpy()[ok], **kw)
        return self

    def proba(self, frame):
        X = lag_stack(features(frame))
        return {h: _ordered(m, X) for h, m in self.models.items()}


class Net(nn.Module):
    """DeepLOB's block structure on a window x features image: a conv stack, an inception module, an LSTM, one head per horizon.

    The paper's first two convolutions walk the (price, size) pairs and the
    levels of a 40-wide book; here the first layer spans the whole feature
    row, because the archive rows have no level structure to walk.
    """

    def __init__(self, n_feat, n_heads=len(HORIZONS)):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.01)
        self.conv = nn.Sequential(nn.Conv2d(1, 32, (1, n_feat)), act(),
                                  nn.Conv2d(32, 32, (4, 1), padding=(2, 0)), act(),
                                  nn.Conv2d(32, 32, (4, 1), padding=(1, 0)), act())
        self.inc3 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), act(), nn.Conv2d(64, 64, (3, 1), padding=(1, 0)), act())
        self.inc5 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), act(), nn.Conv2d(64, 64, (5, 1), padding=(2, 0)), act())
        self.incp = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), nn.Conv2d(32, 64, (1, 1)), act())
        self.lstm = nn.LSTM(192, 64, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(n_heads)])

    def forward(self, x):
        z = self.conv(x[:, None])
        z = torch.cat([self.inc3(z), self.inc5(z), self.incp(z)], dim=1).squeeze(-1).transpose(1, 2)
        o, _ = self.lstm(z)
        return [head(o[:, -1]) for head in self.heads]


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DeepLOB:
    """The Net trained with Adam on WINDOW-second windows, z-scored with the training rows' moments."""
    name = "deeplob"

    def __init__(self, window=WINDOW, epochs=8, batch=512, lr=1e-3, patience=2, seed=0, dev=None):
        self.window, self.epochs, self.batch, self.lr, self.patience, self.seed = window, epochs, batch, lr, patience, seed
        self.dev = torch.device(dev) if dev else device()
        self.history = []

    def _tensor(self, frame):
        X = ((features(frame) - self.mean) / self.std).clip(-10, 10).fillna(0.0)
        return torch.tensor(X.to_numpy(np.float32), device=self.dev)

    def _targets(self, Y):
        return torch.tensor((Y.fillna(0).to_numpy() + 1).astype(np.int64), device=self.dev)

    def _batch(self, X, idx):
        return X[idx[:, None] + self.offsets]

    def _loss(self, out, T, idx):
        return sum(nn.functional.cross_entropy(o, T[idx, j]) for j, o in enumerate(out))

    def fit(self, frame, Y, val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        F = features(frame)
        self.mean, self.std = F.mean(), F.std().replace(0, 1.0)
        self.offsets = torch.arange(-self.window + 1, 1, device=self.dev)
        X, T = self._tensor(frame), self._targets(Y)
        ok = Y.notna().all(axis=1).to_numpy().copy()
        ok[:self.window - 1] = False
        train_idx = torch.tensor(np.flatnonzero(ok), device=self.dev)
        if val is not None:
            Xv, Tv = self._tensor(val[0]), self._targets(val[1])
            okv = val[1].notna().all(axis=1).to_numpy().copy()
            okv[:self.window - 1] = False
            val_idx = torch.tensor(np.flatnonzero(okv), device=self.dev)
        self.net = Net(F.shape[1]).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = train_idx[torch.randperm(len(train_idx), device=self.dev)]
            total = 0.0
            for i in range(0, len(perm), self.batch):
                idx = perm[i:i + self.batch]
                loss = self._loss(self.net(self._batch(X, idx)), T, idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            row = {"epoch": epoch + 1, "train_loss": total / len(perm)}
            if val is not None:
                row["val_loss"] = self._eval_loss(Xv, Tv, val_idx)
                if row["val_loss"] < best - 1e-4:
                    best, best_state, bad = row["val_loss"], copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
            self.history.append(row)
            if val is not None and bad > self.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _eval_loss(self, X, T, idx):
        self.net.eval()
        total = 0.0
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            total += self._loss(self.net(self._batch(X, b)), T, b).item() * len(b)
        return total / max(len(idx), 1)

    @torch.no_grad()
    def proba(self, frame):
        self.net.eval()
        X = self._tensor(frame)
        out = {h: np.full((len(frame), 3), np.nan) for h in HORIZONS}
        idx_all = torch.arange(self.window - 1, len(frame), device=self.dev)
        for i in range(0, len(idx_all), 4096):
            idx = idx_all[i:i + 4096]
            for j, o in enumerate(self.net(self._batch(X, idx))):
                out[HORIZONS[j]][idx.cpu().numpy()] = torch.softmax(o, dim=1).cpu().numpy()
        return out

    def cpu(self):
        """For pickling: the module on the CPU, no device tensors."""
        m = copy.copy(self)
        m.net = copy.deepcopy(self.net).cpu()
        m.dev = torch.device("cpu")
        m.offsets = self.offsets.cpu()
        return m


MODELS = {ImbalanceLogit.name: ImbalanceLogit, LightGBM.name: LightGBM, DeepLOB.name: DeepLOB}


def proba_on(model, frame, start, end):
    """Class probabilities dated start..end, the WINDOW rows before start visible to the model but not scored."""
    i0 = frame.index.get_loc(start)
    i1 = frame.index.get_loc(end)
    sub = frame.iloc[max(0, i0 - WINDOW):i1 + 1]
    p = model.proba(sub)
    return {h: pd.DataFrame(v, index=sub.index, columns=CLASSES).loc[start:end] for h, v in p.items()}


def positions(p, tau):
    """+1 when P(up) - P(down) > tau, -1 below -tau, else flat; nan probabilities are flat."""
    edge = (p[1] - p[-1]).fillna(0.0)
    return pd.Series(np.where(edge > tau, 1.0, np.where(edge < -tau, -1.0, 0.0)), index=p.index)


def pnl(pos, mid, fee_bps=FEE_BPS):
    """Net return per second: a position decided at t fills at bar t+1's price; the fee is charged on traded weight."""
    P = pos.shift(1).to_frame(SYMBOL)
    return backtest(P, mid.reindex(pos.index).to_frame(SYMBOL), cost_bps=fee_bps)


def summarise(r, pos, fee_bps=FEE_BPS):
    """Totals net and gross of the fee, per-day and per-trade bps, a daily-unit Sharpe, and time in the market."""
    r = r.dropna()
    traded = pos.diff().fillna(pos).abs()
    trades = int((traded > 0).sum())
    sd = float(r.std())
    return {"total_bps": float(r.sum() * BPS), "gross_bps": float(r.sum() * BPS + fee_bps * traded.sum()),
            "hours": len(r) / 3600,
            "bps_per_day": float(r.sum() * BPS / (len(r) / 86400)),
            "sharpe_daily": float(r.mean() / sd * np.sqrt(86400)) if sd > 0 else np.nan,
            "trades": trades, "bps_per_trade": float(r.sum() * BPS / trades) if trades else np.nan,
            "in_market": float((pos != 0).mean()), "n_obs": int(len(r))}


def classification(p, y):
    """Accuracy and macro F1 of the argmax class against y, plus the majority-class accuracy on the same rows."""
    ok = y.notna() & p.notna().all(axis=1)
    pred = p.loc[ok].to_numpy().argmax(axis=1) - 1
    truth = y.loc[ok].to_numpy().astype(int)
    return {"accuracy": float((pred == truth).mean()), "f1_macro": float(f1_score(truth, pred, average="macro")),
            "majority": float(pd.Series(truth).value_counts(normalize=True).iloc[0]), "n": int(ok.sum())}


def folds(index, n_splits, test_size, horizon=max(HORIZONS)):
    """purgedcv walk-forward over seconds: a training row whose label touches a test-window price is purged."""
    split = WalkForwardSplit(n_splits=n_splits, test_size=test_size, prediction_times=pd.Series(index),
                             evaluation_times=pd.Series(index + pd.Timedelta(seconds=horizon + 1)))
    return list(split.split(np.zeros((len(index), 1))))
