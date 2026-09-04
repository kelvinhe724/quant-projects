"""Offline checks of the engine's guarantees on synthetic data. Exits nonzero on any failure.

Run: ../.venv/bin/python3 check.py
"""
import os
import sys

import numpy as np
import pandas as pd
from purgedcv import WalkForwardSplit, deflated_sharpe_ratio, probabilistic_sharpe_ratio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.engine import (Bars, Config, CostModel, LookAheadError, RiskConfig, Strategy, run,
                              sharpe, synthetic)

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


class Monthly(Strategy):
    """Equal weight, rebalanced at month end; the workhorse for most checks."""

    def __init__(self, gross=1.0):
        self.gross = gross

    def on_bar(self, asof, bars):
        if bars.is_month_end(asof):
            n = len(bars.instruments)
            return {k: self.gross / n for k in bars.instruments}


class Momentum(Strategy):
    """Sign of the trailing 20-day return, daily. Depends on every close it can see."""

    def on_bar(self, asof, bars):
        px = bars.close
        if len(px) < 21:
            return None
        sig = np.sign(px.iloc[-1] / px.iloc[-21] - 1).fillna(0.0)
        return (sig / len(sig)).to_dict()


class Daily(Strategy):
    def __init__(self, weights):
        self.w = weights

    def on_bar(self, asof, bars):
        return dict(self.w)


NO_COST = Config(costs=CostModel())
bars = synthetic(n_days=750, seed=1)
cal = bars.calendar

print("AS-OF DATA ACCESS\n")
cut = cal[400]
view = bars.upto(cut)
check("view ends exactly at the as-of date", view.close.index[-1] == cut and len(view) == 401)
check("view still knows the full exchange calendar", len(view.calendar) == len(cal))
try:
    view.upto(cal[401])
    check("asking a view for a later date raises", False)
except LookAheadError:
    check("asking a view for a later date raises", True)
try:
    view.close.loc[cal[401]]
    check("indexing a view past its date fails", False)
except KeyError:
    check("indexing a view past its date fails", True)


class Peeker(Strategy):
    def on_bar(self, asof, bars):
        nxt = bars.calendar[bars.calendar.get_loc(asof) + 1]
        return {n: 1.0 for n in bars.upto(nxt).instruments}


try:
    run(Peeker(), bars, end=cal[-2], config=NO_COST)
    check("a strategy that peeks one bar ahead aborts the backtest", False)
except LookAheadError:
    check("a strategy that peeks one bar ahead aborts the backtest", True)

base = run(Momentum(), bars, config=NO_COST)
rng = np.random.default_rng(7)
frames = {f: bars.field(f).copy() for f in ("open", "high", "low", "close", "volume")}
for f in ("open", "high", "low", "close"):
    later = frames[f].index > cut
    frames[f].loc[later] *= np.exp(rng.normal(0, 0.05, (later.sum(), frames[f].shape[1])))
mutated = run(Momentum(), Bars(frames), config=NO_COST)
t_base, t_mut = base.trades[base.trades.date <= cut], mutated.trades[mutated.trades.date <= cut]
check("mutation test: rewriting every price after the cut leaves every trade up to the cut identical",
      len(t_base) == len(t_mut) and np.allclose(t_base.quantity, t_mut.quantity)
      and np.allclose(t_base.fill, t_mut.fill), f"{len(t_base)} trades compared")
check("mutation test: trades after the cut did change",
      not np.allclose(base.trades[base.trades.date > cut].quantity.head(50),
                      mutated.trades[mutated.trades.date > cut].quantity.head(50)))
check("mutation test: equity up to the cut identical",
      np.allclose(base.equity.loc[:cut], mutated.equity.loc[:cut]))

print("\nFILL TIMING\n")
t0 = cal[300]
t1 = cal[301]


class OneShot(Strategy):
    def on_bar(self, asof, bars):
        return {"A": 1.0} if asof == t0 else None


res = run(OneShot(), bars, config=NO_COST)
tr = res.trades
check("signal at t produces exactly one trade, dated t+1", len(tr) == 1 and tr.date.iloc[0] == t1)
check("fill price is the t+1 open", np.isclose(tr.fill.iloc[0], bars.open.loc[t1, "A"]))
check("shares are sized off equity at t, not t+1",
      np.isclose(tr.quantity.iloc[0] * bars.open.loc[t1, "A"], res.equity.loc[t0]))
res_c = run(OneShot(), bars, config=Config(fill="close"))
check("fill='close' fills at the t+1 close",
      np.isclose(res_c.trades.fill.iloc[0], bars.close.loc[t1, "A"]))
for f, d, moves in (("open", t1, True), ("open", t0, False), ("close", t0, False), ("close", t1, False)):
    frames = {k: bars.field(k).copy() for k in ("open", "high", "low", "close", "volume")}
    frames[f].loc[d, "A"] *= 1.01
    fill = run(OneShot(), Bars(frames), config=NO_COST).trades.fill.iloc[0]
    changed = not np.isclose(fill, tr.fill.iloc[0])
    check(f"mutating {f}[{'t+1' if d == t1 else 't'}] {'moves' if moves else 'does not move'} the fill",
          changed == moves)
IMPACT = Config(costs=CostModel(impact_coef=1.0, half_spread_bps=3))
ref = run(OneShot(), bars, config=IMPACT).trades.fill.iloc[0]
frames = {k: bars.field(k).copy() for k in ("open", "high", "low", "close", "volume")}
frames["close"].loc[t1, "A"] *= 1.05
frames["volume"].loc[t1, "A"] *= 0.1
check("with impact on, the t+1 close and volume still do not move the fill",
      np.isclose(run(OneShot(), Bars(frames), config=IMPACT).trades.fill.iloc[0], ref))

print("\nCOSTS\n")
finals = []
for bps in (0, 5, 10, 20, 40):
    r = run(Momentum(), bars, config=Config(costs=CostModel(commission_bps=bps)))
    finals.append(r.equity.iloc[-1])
check("commission: final equity falls strictly with every bps step", all(np.diff(finals) < 0),
      " > ".join(f"{v:,.0f}" for v in finals))
finals = [run(Momentum(), bars, config=Config(costs=CostModel(half_spread_bps=b))).equity.iloc[-1]
          for b in (0, 5, 20)]
check("half spread: same", all(np.diff(finals) < 0))
finals = [run(Momentum(), bars, config=Config(costs=CostModel(impact_coef=k))).equity.iloc[-1]
          for k in (0, 0.5, 2.0)]
check("impact: same", all(np.diff(finals) < 0))
tr_i = run(Momentum(), bars, config=IMPACT).trades.sort_values("participation")
check("every trade pays the half spread plus k * sigma * sqrt(participation)",
      (tr_i.slippage_bps > 3).all()
      and np.allclose(tr_i.slippage_bps, 3 + 1e4 * tr_i.sigma * np.sqrt(tr_i.participation)))
check("impact grows with participation once the instrument's vol is held fixed",
      ((tr_i.slippage_bps - 3) / tr_i.sigma).is_monotonic_increasing)
big = run(Daily({"A": 30.0}), bars, config=Config(capital=1e9, costs=CostModel(participation_cap=0.05)))
check("participation cap: trades above the cap are cut to it and flagged",
      big.trades.capped.any() and np.isclose(big.trades[big.trades.capped].participation, 0.05).all())
check("no fill ever exceeds the cap", (big.trades.participation <= 0.05 + 1e-12).all())

print("\nPORTFOLIO\n")
book = res.books["OneShot"]
check("per-instrument P&L plus carry reconciles to the equity change",
      np.allclose(book["equity"].diff().dropna(), (book["pnl"].sum(axis=1) + book["carry"]).iloc[1:]))
short = run(Daily({"A": -0.5}), bars, config=Config(borrow_bps=0))
short_b = run(Daily({"A": -0.5}), bars, config=Config(borrow_bps=200))
check("borrow cost on shorts lowers equity", short_b.equity.iloc[-1] < short.equity.iloc[-1])
rate = pd.Series(5.0, index=cal, name="R")
lev_bars = Bars({f: bars.field(f) for f in ("open", "high", "low", "close", "volume")}, {"R": rate})
lev = run(Daily({"A": 2.0}), lev_bars, config=Config(financing=None))
lev_f = run(Daily({"A": 2.0}), lev_bars, config=Config(financing="R"))
check("financing a 2x book at 5% costs about 5% a year",
      -0.04 > (lev_f.carry.sum() / lev_f.equity.mean()) / (750 / 252) > -0.06,
      f"{(lev_f.carry.sum() / lev_f.equity.mean()) / (750 / 252):+.2%}/yr")
sat = pd.Series({pd.Timestamp("2010-01-01"): 1.0, pd.Timestamp("2011-01-01"): 2.0})  # 2011-01-01 is a Saturday
sat_bars = Bars({f: bars.field(f) for f in ("open", "high", "low", "close", "volume")}, {"R": sat})
check("a series dated on a weekend is visible from the next session, not lost",
      sat_bars.series("R").loc["2011-01-03"] == 2.0 and sat_bars.series("R").loc["2010-12-31"] == 1.0)
check("multi-strategy: total equity is the sum of the books and each book keeps its own attribution",
      (lambda m: np.allclose(m.equity, m.equity_by_strategy.sum(axis=1))
       and set(m.attribution_by_strategy) == {"Momentum", "Monthly"})(
          run([Momentum(), Monthly()], bars, config=NO_COST)))

print("\nRISK\n")
vol_bars = synthetic(n_days=252 * 6, vol=0.30, seed=5)
targeted = run(Daily({n: 0.25 for n in vol_bars.instruments}), vol_bars,
               config=Config(risk=RiskConfig(target_vol=0.10)))
realised = targeted.returns.std() * np.sqrt(252)
check("vol targeting lands within 20% of a 10% target on 30%-vol assets", 0.08 < realised < 0.12,
      f"realised {realised:.2%}")
capped = run(Daily({"A": 3.0, "B": -3.0}), bars, config=Config(risk=RiskConfig(max_gross=2.0)))
gross = capped.weights.abs().sum(axis=1)
fill_gross = (capped.trades.groupby("date").apply(
    lambda t: (t.quantity.abs() * t.fill).sum(), include_groups=False)
    / capped.equity.shift(1).reindex(capped.trades.date.unique()))
check("gross limit holds at every fill; close-of-day weights only drift with the day's move",
      gross.median() < 2.01 and gross.max() < 2.2 and fill_gross.max() < 2.0 + 1e-9,
      f"median {gross.median():.3f}, max at close {gross.max():.3f}")
capped = run(Daily({"A": 3.0}), bars, config=Config(risk=RiskConfig(max_weight=0.5)))
check("per-instrument limit holds", capped.weights["A"].max() < 0.55)


class MonthlyLS(Strategy):
    def on_bar(self, asof, bars):
        return {"A": 1.0, "B": -1.0} if bars.is_month_end(asof) else None


loose = run(MonthlyLS(), bars, config=NO_COST)
tight = run(MonthlyLS(), bars, config=Config(risk=RiskConfig(max_gross=2.0, buffer=0.10)))
over = tight.weights.abs().sum(axis=1)
over = over[over > 2.0].index
nxt = pd.DatetimeIndex([cal[cal.get_loc(d) + 1] for d in over if cal.get_loc(d) + 1 < len(cal)])
check("a monthly book that drifts through its gross cap is corrected on the next bar",
      len(over) > 0 and nxt.isin(tight.trades.date).all() and len(tight.trades) > len(loose.trades),
      f"{len(over)} breaches, {len(tight.trades)} trades vs {len(loose.trades)} uncapped")

crash = synthetic(n_days=800, instruments=("X",), vol=0.15, seed=9)
frames = {f: crash.field(f).copy() for f in ("open", "high", "low", "close", "volume")}
path = np.ones(800)
path[500:541] = np.linspace(1, 0.6, 41)
path[541:641] = np.linspace(0.6, 1.3, 100)
path[641:] = 1.3
for f in ("open", "high", "low", "close"):
    frames[f]["X"] *= path
crash = Bars(frames)
raw = run(Daily({"X": 1.0}), crash, config=NO_COST)
ctl = run(Daily({"X": 1.0}), crash, config=Config(risk=RiskConfig(dd_threshold=0.10, dd_scale=0.25)))
check("drawdown control caps a planted 40% crash well below the uncontrolled fall",
      ctl.metrics["max_drawdown"] > raw.metrics["max_drawdown"] + 0.10,
      f"controlled {ctl.metrics['max_drawdown']:.1%} vs raw {raw.metrics['max_drawdown']:.1%}")
w = ctl.weights["X"]
check("exposure is cut past the threshold and restored on recovery",
      w.loc[cal[0]:].iloc[540] < 0.3 and w.iloc[-1] > 0.9, f"during {w.iloc[540]:.2f}, after {w.iloc[-1]:.2f}")
killed = run(Daily({"X": 1.0}), crash, config=Config(risk=RiskConfig(kill_dd=0.15)))
kill_day = killed.equity.index[(killed.equity / killed.equity.cummax() - 1 <= -0.15).argmax()]
after = killed.returns.loc[kill_day:].iloc[2:]
check("kill switch flattens the book and it stays flat",
      killed.books["Daily"]["risk"].killed and (after == 0).all(), f"killed {kill_day.date()}")


class MonthlyLong(Strategy):
    def on_bar(self, asof, bars):
        return {"X": 1.0} if bars.is_month_end(asof) else None


killed_m = run(MonthlyLong(), crash, config=Config(risk=RiskConfig(kill_dd=0.15)))
kill_day = killed_m.equity.index[(killed_m.equity / killed_m.equity.cummax() - 1 <= -0.15).argmax()]
flatten = killed_m.trades.iloc[-1]
check("a monthly strategy is killed the bar after the drawdown breach, not at month end",
      not crash.is_month_end(kill_day) and flatten.date == cal[cal.get_loc(kill_day) + 1]
      and flatten.quantity < 0 and (killed_m.returns.loc[kill_day:].iloc[2:] == 0).all(),
      f"killed {kill_day.date()}, flat {flatten.date.date()}")

print("\nBUFFER\n")
calm = synthetic(n_days=400, instruments=("A", "B"), vol=0.02, seed=11)
BUF = Config(risk=RiskConfig(buffer=0.10))


class Wobble(Strategy):
    """Base weights that wobble 3% a day, stepped up 15% from `jump` on."""

    def __init__(self, base, jump=None):
        self.base, self.jump, self.i = base, jump, 0

    def on_bar(self, asof, bars):
        self.i += 1
        up = 1.15 if self.jump is not None and asof >= self.jump else 1.0
        return {n: w * up * (1 + 0.03 * (-1) ** self.i) for n, w in self.base.items()}


wob = run(Wobble({"A": 0.4, "B": -0.4}), calm, config=BUF)
check("buffer: targets that wobble 3% a day trade once per instrument, on the entry bar, and never again",
      len(wob.trades) == 2 and set(wob.trades.date) == {calm.calendar[1]}, f"{len(wob.trades)} trades")
jump = run(Wobble({"A": 0.4, "B": -0.4}, jump=calm.calendar[200]), calm, config=BUF)
later = jump.trades[jump.trades.date > calm.calendar[1]]
check("buffer: a 15% step in the target trades every instrument on the next bar, back to the target, and nothing else",
      len(later) == 2 and set(later.date) == {calm.calendar[201]}
      and np.allclose(jump.weights.loc[calm.calendar[201]], [0.4 * 1.15 * 0.97, -0.4 * 1.15 * 0.97], atol=0.005),
      f"{jump.weights.loc[calm.calendar[201]].round(4).to_dict()}")
core = run(Daily({"A": 0.5, "B": 0.5}), calm, config=Config(risk=RiskConfig(target_vol=0.10, buffer=0.10)))
loose = run(Daily({"A": 0.5, "B": 0.5}), calm, config=Config(risk=RiskConfig(target_vol=0.10)))
rel = core.weights.diff().abs() / core.weights.abs()
moves = pd.Series([rel.loc[t.date, t.instrument] for t in core.trades.itertuples()])
check("buffer: a constant target under the vol overlay trades only when the overlay has moved it more than 10%",
      loose.trades.date.nunique() > 0.95 * len(calm.calendar)
      and core.trades.date.nunique() < 0.2 * loose.trades.date.nunique() and (moves > 0.09).all(),
      f"{core.trades.date.nunique()} trade days vs {loose.trades.date.nunique()} unbuffered, "
      f"smallest move {moves.min():.1%}")
ls = run(Daily({"A": 1.0, "B": -1.0}), bars, config=Config(risk=RiskConfig(max_gross=2.0, buffer=0.10)))
gross = ls.weights.abs().sum(axis=1)
over = gross[gross > 2.0].index
over = over[over < cal[-1]]
traded = ls.trades.groupby("date").instrument.agg(set)
check("buffer: a gross breach still sends every instrument on the next bar, in band or not",
      len(over) > 0 and all(traded.get(cal[cal.get_loc(d) + 1], set()) == {"A", "B"} for d in over)
      and ls.trades.date.nunique() < 0.5 * len(cal),
      f"{len(over)} breaches, {ls.trades.date.nunique()} trade days of {len(cal)}")
mild = run(Daily({"X": 1.0}), crash, config=Config(risk=RiskConfig(dd_threshold=0.10, dd_scale=0.92, buffer=0.10)))
dd_day = mild.equity.index[(mild.equity / mild.equity.cummax() - 1 <= -0.10).argmax()]
nxt = crash.calendar[crash.calendar.get_loc(dd_day) + 1]
check("buffer: a drawdown cut inside the band still trades the bar after the breach, and the recovery trades back",
      nxt in set(mild.trades.date) and abs(mild.weights.loc[nxt, "X"] - 0.92) < 0.01
      and mild.weights["X"].iloc[-1] > 0.99, f"cut {dd_day.date()}, held {mild.weights.loc[nxt, 'X']:.3f} next bar")

print("\nHONESTY (purgedcv)\n")
noise = synthetic(n_days=750, vol=0.20, seed=21)
rng = np.random.default_rng(3)
N = 100
trial_returns = []
for i in range(N):
    w = rng.normal(0, 0.25, len(noise.instruments))
    trial_returns.append(run(Daily(dict(zip(noise.instruments, w))), noise, config=NO_COST).returns)
srs = np.array([r.mean() / r.std() for r in trial_returns])
best = trial_returns[int(srs.argmax())].to_numpy()
psr = probabilistic_sharpe_ratio(best, 0.0)
dsr = deflated_sharpe_ratio(best, N, var_sharpe=float(srs.var(ddof=1)))
check(f"best of {N} random strategies looks real: Sharpe {sharpe(best):.2f}, PSR {psr:.2f}",
      sharpe(best) > 1.0 and psr > 0.95)
check(f"purgedcv deflated Sharpe says it is not: DSR {dsr:.2f}", dsr < 0.9)
good = rng.normal(0.001, 0.01, 2000)
dsr_good = deflated_sharpe_ratio(good, N, var_sharpe=1 / (len(good) - 1))
check("a genuine edge survives deflation", dsr_good > 0.95, f"DSR {dsr_good:.3f}")

idx = pd.bdate_range("2010-01-01", periods=2500)
X = np.zeros((len(idx), 1))
folds = list(WalkForwardSplit(n_splits=4, test_size=300, prediction_times=idx,
                              evaluation_times=idx).split(X))
check("purgedcv walk-forward: test windows follow training and do not overlap",
      all(tr[-1] < te[0] for tr, te in folds)
      and all(folds[k][1][-1] < folds[k + 1][1][0] for k in range(len(folds) - 1)))

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
