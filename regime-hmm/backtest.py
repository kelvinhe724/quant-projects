"""Exposure rules, rolling out-of-sample HMM refits, and the return engine."""
import numpy as np
import pandas as pd

import regime

COST_BPS = 5.0          # one-way, on traded notional
TARGET_VOL = 0.15       # annual, for the vol-targeting benchmark
VOL_SPAN = 20           # days, EWM realised vol for vol-targeting
FIRST_TEST_YEAR = 2010


def hmm_exposure_oos(ret_pct, n_states=2, first_test_year=FIRST_TEST_YEAR, seed=0):
    """Expanding-window out-of-sample exposure: 1 - P(bear) from filtered probabilities.

    Each January the model is refit on every return before that year. The
    forward filter is then run over the whole history up to the end of that
    year with the frozen model, and only that year's rows are kept. The filter
    is causal, so a row dated t depends on returns up to t and parameters
    estimated before the year began. Also returns the parameter table per refit.
    """
    ret_pct = ret_pct.dropna()
    years = sorted(y for y in ret_pct.index.year.unique() if y >= first_test_year)
    exposure = pd.Series(index=ret_pct.index, dtype=float)
    fits = []
    for y in years:
        train = ret_pct[ret_pct.index.year < y]
        model = regime.fit(train, n_states, seed)
        through = ret_pct[ret_pct.index.year <= y]
        p = regime.filtered(model, through)[:, regime.bear_state(model)]
        mask = through.index.year == y
        exposure[through.index[mask]] = 1 - p[mask]
        desc = regime.describe(model)
        desc.insert(0, "fit_year", y)
        fits.append(desc)
    return exposure.dropna(), pd.concat(fits)


def hmm_exposure_insample(ret_pct, n_states=2, seed=0, use_smoothed=True):
    """The trap: fit once on the full sample and use smoothed probabilities. For comparison only."""
    model = regime.fit(ret_pct, n_states, seed)
    f = regime.smoothed if use_smoothed else regime.filtered
    p = f(model, ret_pct)[:, regime.bear_state(model)]
    return pd.Series(1 - p, index=ret_pct.index)


def vol_target_exposure(ret, target=TARGET_VOL, span=VOL_SPAN, cap=1.0):
    """Exposure = target vol / trailing realised vol, capped at 1 so it is unlevered like the HMM rule."""
    realised = ret.ewm(span=span).std() * np.sqrt(regime.TRADING_DAYS)
    return (target / realised).clip(upper=cap)


def run(ret, exposure, cost_bps=COST_BPS):
    """Daily strategy returns. Exposure decided at close t is held over day t+1."""
    position = exposure.reindex(ret.index).shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    gross = position * ret
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"position": position, "turnover": turnover,
                         "gross": gross, "net": net})


def metrics(r, turnover=None):
    """Annualised return, vol, Sharpe, max drawdown, turnover for one daily return series."""
    days = regime.TRADING_DAYS
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    out = {
        "annual_return": equity.iloc[-1] ** (days / len(r)) - 1,
        "annual_vol": r.std() * np.sqrt(days),
        "sharpe": r.mean() / r.std() * np.sqrt(days) if r.std() > 0 else np.nan,
        "max_drawdown": dd.min(),
        "dd_trough": dd.idxmin(),
    }
    if turnover is not None:
        out["annual_turnover"] = turnover.sum() * days / len(r)
    return out


def summary(ret, exposures, cost_bps=COST_BPS):
    """Gross and net metrics for a dict of name -> exposure series on a common window."""
    rows = {}
    for name, exp in exposures.items():
        bt = run(ret, exp, cost_bps)
        g, n = metrics(bt["gross"]), metrics(bt["net"], bt["turnover"])
        rows[name] = {
            "annual_return_gross": g["annual_return"], "annual_return_net": n["annual_return"],
            "annual_vol": n["annual_vol"], "sharpe_gross": g["sharpe"], "sharpe_net": n["sharpe"],
            "max_drawdown_net": n["max_drawdown"], "dd_trough": n["dd_trough"],
            "annual_turnover": n["annual_turnover"], "avg_exposure": bt["position"].mean(),
        }
    return pd.DataFrame(rows).T


def by_year(ret, exposures, cost_bps=COST_BPS):
    """Net compounded return and Sharpe per calendar year for each exposure rule.

    The return is the plain compounded return over the year's rows, not
    annualised, so a partial final year reads as year-to-date.
    """
    rows = []
    for name, exp in exposures.items():
        net = run(ret, exp, cost_bps)["net"]
        for y, r in net.groupby(net.index.year):
            rows.append({"rule": name, "year": y, "return": (1 + r).prod() - 1,
                         "sharpe": metrics(r)["sharpe"]})
    return pd.DataFrame(rows).pivot(index="year", columns="rule", values=["return", "sharpe"])
