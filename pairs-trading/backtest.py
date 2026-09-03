"""Daily backtest engine: per-pair P&L, transaction costs, portfolio metrics."""
import numpy as np
import pandas as pd

from pairs import positions, spread, zscore

TRADING_DAYS = 252
DEFAULT_COST_BPS = 10.0


def pair_backtest(log_px, pair, lookback, entry_z, exit_z, stop_z, max_hold,
                  cost_bps=DEFAULT_COST_BPS):
    """Backtest one pair. Returns a frame of position, gross and net daily return."""
    a, b, beta, intercept = pair.a, pair.b, pair.beta, pair.intercept
    sp = spread(log_px[a], log_px[b], beta, intercept)
    z = zscore(sp, lookback)
    target = positions(z, entry_z, exit_z, stop_z, max_hold)

    # Signal at t, fill at t+1. Everything downstream reads `held`, never `target`.
    held = target.shift(1).fillna(0.0)

    # simple returns, so the (1+r).cumprod() compounding in metrics() is exact
    ret = np.expm1(log_px.diff())
    gross_exposure = 1.0 + abs(beta)
    leg = (ret[a] - beta * ret[b]) / gross_exposure
    gross = held * leg

    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * (cost_bps / 2.0) / 1e4
    return pd.DataFrame({"z": z, "position": held, "gross": gross.fillna(0.0),
                         "net": (gross - cost).fillna(0.0), "turnover": turnover})


def aggregate(legs, capital_slots=None):
    """Average per-pair returns into an equal-capital portfolio."""
    slots = capital_slots or len(legs)
    cols = ("gross", "net", "turnover")
    return pd.DataFrame({c: sum(v[c] for v in legs.values()) / slots for c in cols})


def trades(leg):
    """Extract closed round trips from one pair's backtest frame."""
    pos = leg["position"].to_numpy()
    ret = leg["net"].to_numpy()
    out, start = [], None
    for i in range(len(pos)):
        if pos[i] != 0 and start is None:
            start = i
        elif pos[i] == 0 and start is not None:
            out.append((i - start, ret[start:i].sum()))
            start = None
    if start is not None:
        out.append((len(pos) - start, ret[start:].sum()))
    return pd.DataFrame(out, columns=["days", "return"])


def metrics(returns, all_trades=None):
    """Summarise a daily return series into the usual performance table."""
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return {k: np.nan for k in ("total_return", "annual_return", "annual_vol",
                                    "sharpe", "max_drawdown", "n_trades",
                                    "hit_rate", "avg_hold_days")}
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    out = {
        "total_return": equity.iloc[-1] - 1,
        "annual_return": equity.iloc[-1] ** (1 / years) - 1,
        "annual_vol": r.std() * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }
    if all_trades is not None and len(all_trades):
        out["n_trades"] = len(all_trades)
        out["hit_rate"] = (all_trades["return"] > 0).mean()
        out["avg_hold_days"] = all_trades["days"].mean()
    else:
        out["n_trades"] = 0
        out["hit_rate"] = np.nan
        out["avg_hold_days"] = np.nan
    return out


def evaluate(log_px, selected, window=None, capital_slots=None, **rules):
    """Backtest the selected pairs and score them over `window`.

    Signals are always built on the full price history so the trailing z-score is
    warm at the start of `window`; only the scoring is restricted. The window is
    in the past relative to every point it is scored on, so this is not look-ahead.
    """
    if not len(selected):
        idx = log_px.loc[window[0]:window[1]].index if window else log_px.index
        blank = pd.DataFrame(0.0, index=idx, columns=["gross", "net", "turnover"])
        return metrics(blank["net"]), blank, {}

    legs = {(p.a, p.b): pair_backtest(log_px, p, **rules) for p in selected.itertuples()}
    if window:
        legs = {k: v.loc[window[0]:window[1]] for k, v in legs.items()}

    port = aggregate(legs, capital_slots)
    all_trades = pd.concat([trades(v) for v in legs.values()], ignore_index=True)
    m = metrics(port["net"], all_trades)
    m["gross_sharpe"] = metrics(port["gross"])["sharpe"]
    m["gross_annual_return"] = metrics(port["gross"])["annual_return"]
    m["annual_turnover"] = port["turnover"].sum() / (len(port) / TRADING_DAYS)
    m["n_pairs"] = len(legs)
    return m, port, legs
