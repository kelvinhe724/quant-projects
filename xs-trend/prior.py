"""The prior layer of PREREG.md: 50/200 long-only on the point-in-time S&P 500, top 100 by dollar volume.

Everything the pre-registration freezes is a constant here. The untouched window
(2021-01-01 to the end of the membership panel) is locked by the price panel's
hash before any in-sample run, and the default run never reads a return inside
it. The reviewer opens it once with --open; a second open raises.

Run:  ../.venv/bin/python3 prior.py            lock the window, walk-forward, in-sample table
      ../.venv/bin/python3 prior.py --open     open the untouched window once (the reviewer's call)
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data-lake"))
import lake  # noqa: E402
from framework.book.strategies import CAPITAL, book_config  # noqa: E402
from framework.engine import by_year, sharpe, summary  # noqa: E402
from research.alpha.base import (Alpha, Untouched, UntouchedWindowUsed, month_ends, positions,  # noqa: E402
                                 walk_forward)
from research.features import Feature, FeatureStore, Raw  # noqa: E402
from research.features.store import panel_hash  # noqa: E402
from research.registry import Registry  # noqa: E402


def _import(project, module):
    """Load a sibling project's module by path (the projects are not packages)."""
    spec = importlib.util.spec_from_file_location(f"{project}_{module}", os.path.join(ROOT, project, module + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mom = _import("momentum", "momentum")        # ma_signal, long_only_weights, MIN_NAMES: the published signal code
mom_data = _import("momentum", "data")       # UNIVERSE (the published 190 names) and the eligibility constants

REPORTS = os.path.join(HERE, "reports")
LOCK = os.path.join(REPORTS, "untouched.json")
REGISTRY = os.path.join(REPORTS, "registry")

# PREREG.md, frozen. Dates follow momentum/: warm-up from 2003-06 so the 200-day
# average and the 252-print rule are complete on the first scored day.
PANEL_START = "2003-06-01"
PANEL_END = "2026-08-31"                     # last day of pit-universe's membership panel; the lake runs on past it
IS_START, IS_END = "2005-01-03", "2020-12-31"
UNTOUCHED_START = "2021-01-01"
TOP_N = 100                                  # §Instrument: N frozen at 100
MA_SHORT, MA_LONG = mom.MA_SHORT, mom.MA_LONG  # 50 / 200
LAG = 1                                      # §Rebalance: signals at t-1 close, the store's shift is the only lag
MIN_PRINTS = mom_data.MIN_HISTORY            # 252, momentum's eligibility
MIN_DOLLAR_VOLUME = mom_data.MIN_DOLLAR_VOLUME  # $10m trailing median
DV_WINDOW = mom_data.ADV_WINDOW              # 63 sessions, window ends t-1
MIN_NAMES = mom.MIN_NAMES                    # 8: fewer names in an uptrend than this is cash, as published
COST_BPS = 10.0                              # §Costs: 10 bps one-way, on top of the book's model below
COSTS = book_config().costs                  # the object the book uses: 5 bps commission, 2 bps half spread, sqrt impact
BORROW_BPS = 0.0                             # long only
TARGET_VOL = 0.15                            # §Sizing: 15% on the sleeve, gross <= 1, no leverage
VOL_DAYS = 63
MAX_GROSS = 1.0
KILL_DD = 0.20                               # §Kills: reported as a diagnostic, not applied to the scored path (README)
FOLDS, TEST_SIZE = 4, 504                    # research/run.py's convention; nothing is tuned, so folds are a stability check
WF_COST_BPS = COST_BPS + COSTS.commission_bps + COSTS.half_spread_bps  # base.backtest takes one scalar; impact (<1 bp) is in the scored path


# --- data -------------------------------------------------------------------------------------

def load_panel(start=PANEL_START, end=PANEL_END):
    """Wide frames from the lake: adjusted close/open for signal and returns, raw close x volume for liquidity."""
    eq = lake.load("equities_daily", start, end)
    wide = lambda col: eq.pivot(index="date", columns="ticker", values=col).sort_index()  # noqa: E731
    close, adj, opn, vol = wide("close"), wide("adj_close"), wide("open"), wide("volume")
    adj, close = adj.where(adj > 0), close.where(close > 0)
    adj_open = (opn * adj / close).where(opn > 0)
    # bridge a gap inside a live series, never before the first print or after the last (pit-universe's rule)
    fill = lambda f: f.ffill().where(f.bfill().notna())  # noqa: E731
    adj, adj_open, close = fill(adj), fill(adj_open), fill(close)
    names = [c for c in adj.columns if c != "SPY"]
    frames = {"close": adj[names], "open": adj_open[names], "raw_close": close[names], "volume": vol[names]}
    return frames, adj["SPY"].pct_change().rename("SPY")


def membership(intervals, calendar, names):
    """dates x names bool panel from pit-universe's membership spells."""
    m = pd.DataFrame(False, index=calendar, columns=names)
    for t, s, e in intervals[["ticker", "start", "end"]].itertuples(index=False):
        if t in m.columns:
            m.loc[s:e, t] = True
    return m


def features(member_pit, member_ref):
    """The store: signal and eligibility at lag 1 (the close of t-1), membership as of t."""
    def mask(member):
        return lambda r: member.reindex(index=r.calendar, columns=r.instruments).fillna(False).astype(float)
    return FeatureStore([
        Feature("ma", lambda r: mom.ma_signal(r.frames["close"], MA_SHORT, MA_LONG), lag=LAG),
        Feature("dv63", lambda r: (r.frames["raw_close"] * r.frames["volume"])
                .rolling(DV_WINDOW, min_periods=DV_WINDOW).median(), lag=LAG),
        Feature("prints", lambda r: r.frames["close"].notna().cumsum().astype(float), lag=LAG),
        Feature("member_pit", mask(member_pit), lag=0),
        Feature("member_ref", mask(member_ref), lag=0),
    ])


# --- the rule ---------------------------------------------------------------------------------

class XSTrend(Alpha):
    """1/N long the names in an uptrend among the top_n most liquid eligible members; cash otherwise.

    trend=False drops the signal and holds every name in that universe at 1/N:
    the buy-and-hold comparator of PREREG.md §Test on the same universe.
    """

    def __init__(self, member="member_pit", top_n=TOP_N, trend=True):
        self.member, self.top_n, self.trend = member, top_n, trend
        self.name = ("XSTrend" if trend else "BuyHold") + f"[{member},N={top_n or 'all'}]"

    def universe(self, xs):
        ok = (xs[self.member] > 0) & (xs["prints"] >= MIN_PRINTS) & (xs["dv63"] >= MIN_DOLLAR_VOLUME)
        if self.top_n:
            ok &= xs["dv63"].where(ok).rank(ascending=False, method="first") <= self.top_n
        return ok

    def signal(self, xs):
        on = self.universe(xs) & ((xs["ma"] > 0) if self.trend else True)
        n = int(on.sum())
        return on.astype(float) / n if n >= MIN_NAMES else pd.Series(0.0, index=xs.index)


# --- the engine: next-open fills, the book's cost model, the vol target --------------------------

def cost_bps(dw, adv20, sigma20):
    """One-way cost in bps on each traded weight: PREREG's 10 bps plus the book's commission, half spread and impact."""
    part = (dw.abs() * CAPITAL / adv20).where(adv20 > 0).fillna(0.0)
    return COST_BPS + COSTS.commission_bps + COSTS.slippage_bps(part, sigma20.fillna(0.0))


def path(pos, adj, adj_open, adv20=None, sigma20=None, bps=None):
    """Net daily returns of a target-weight panel: a target set at t fills at t+1's open.

    Weights already held earn close-to-close; the traded increment earns
    open-to-close on the fill day and pays `bps` (a frame, or a scalar) on its
    absolute size. A held name with no print that day earns zero, and the count
    of such stock-days is returned, not hidden.
    """
    after = pos.reindex(adj.index).ffill().shift(1).reindex(columns=adj.columns).fillna(0.0)  # held after the fill
    before = after.shift(1).fillna(0.0)
    dw = after - before
    r_cc, r_oc = adj.pct_change(), adj / adj_open - 1
    if bps is None:
        bps = cost_bps(dw, adv20, sigma20)
    pnl = (before * r_cc).sum(axis=1) + (dw * r_oc).sum(axis=1)
    cost = (dw.abs() * bps).sum(axis=1) / 1e4
    return {"ret": (pnl - cost).rename("return"), "cost": cost, "traded": dw.abs().sum(axis=1), "after": after,
            "missing_held": int(((before != 0) & r_cc.isna()).sum().sum()),
            "missing_fill": int(((dw != 0) & r_oc.isna()).sum().sum())}


def vol_scale(r_unlevered, dates, target=TARGET_VOL, max_gross=MAX_GROSS):
    """Scalar on the unlevered weights at each rebalance: target over trailing realised vol, never above max_gross.

    The vol window ends the session before the rebalance; with no estimate yet
    there is no position (VRP's rule), not the cap.
    """
    vol = r_unlevered.rolling(VOL_DAYS, min_periods=VOL_DAYS).std().shift(1) * np.sqrt(252)
    return (target / vol).clip(upper=max_gross).reindex(dates).fillna(0.0)


def run_alpha(alpha, panel, frames, liquidity, end):
    """Unlevered and vol-targeted paths of one Alpha from the panel start to `end`; nothing after `end` is read."""
    cal = panel.index[panel.index <= end]
    dates = month_ends(cal)
    adj, adj_open = frames["close"].loc[:end], frames["open"].loc[:end]
    adv20, sigma20 = liquidity[0].loc[:end], liquidity[1].loc[:end]
    pos = positions(alpha, panel.loc[:end], dates)
    unlev = path(pos, adj, adj_open, adv20, sigma20)
    scale = vol_scale(unlev["ret"], dates)
    scored = path(pos.mul(scale, axis=0), adj, adj_open, adv20, sigma20)
    return {"pos": pos, "scale": scale, "unlevered": unlev, "scored": scored}


def liquidity(frames):
    """Trailing ADV and daily vol as the engine's Executor measures them: 20 sessions strictly before the fill."""
    adv20 = (frames["raw_close"] * frames["volume"]).rolling(COSTS.adv_window).mean().shift(1)
    sigma20 = frames["close"].pct_change().rolling(COSTS.adv_window).std().shift(1)
    return adv20, sigma20


def row(res, start, end):
    """The reported statistics of one path over [start, end]."""
    r = res["ret"].loc[start:end]
    m = summary(r)
    held = res["after"].loc[start:end]
    live = held.sum(axis=1) > 0
    years = len(r) / 252
    return {"sharpe": m["sharpe"], "annual_return": m["annual_return"], "annual_vol": m["annual_vol"],
            "max_drawdown": m["max_drawdown"], "turnover_oneway": float(res["traded"].loc[start:end].sum() / years),
            "avg_names": float((held > 0).sum(axis=1)[live].mean()) if live.any() else 0.0,
            "avg_gross": float(held.sum(axis=1).mean()), "cost_drag": float(res["cost"].loc[start:end].sum() / years),
            "pct_days_invested": float(live.mean()), "n_days": int(len(r))}


def kill_dates(r, dd=KILL_DD):
    """Sessions on which PREREG §Kills' drawdown line is first crossed in each episode (diagnostic only)."""
    eq = (1 + r).cumprod()
    under = eq / eq.cummax() - 1
    hit = under <= -dd
    return list(under.index[hit & ~hit.shift(1, fill_value=False)].date)


# --- main -------------------------------------------------------------------------------------

def build():
    """Everything a run needs, built once: frames, feature panel (audited), liquidity, membership."""
    frames, spy = load_panel()
    cal, names = frames["close"].index, list(frames["close"].columns)
    member_pit = membership(lake.sp500_intervals(), cal, names)
    member_ref = pd.DataFrame({t: t in mom_data.UNIVERSE for t in names}, index=cal)
    raw = Raw(frames)
    store = features(member_pit, member_ref)
    panel = store.build(raw, audit=True)     # PeekError here means a feature reads its own future; nothing runs
    return {"frames": frames, "spy": spy, "cal": cal, "names": names, "member_pit": member_pit,
            "member_ref": member_ref, "store": store, "panel": panel, "liquidity": liquidity(frames),
            "n_ref": int(member_ref.iloc[-1].sum())}


ALPHAS = {"pit": XSTrend("member_pit", TOP_N), "ref": XSTrend("member_ref", None), "bh": XSTrend("member_pit", TOP_N, trend=False)}
LABELS = {"pit": f"PIT N={TOP_N}", "ref": "reference 190", "bh": "EW basket (B&H)"}


def config(key, alpha, pathname, window=None):
    c = {"universe": key, "member": alpha.member, "top_n": alpha.top_n, "trend": alpha.trend,
         "ma": [MA_SHORT, MA_LONG], "lag": LAG, "cost_bps": COST_BPS, "book_costs": vars(COSTS),
         "fill": "next_open", "path": pathname, "target_vol": TARGET_VOL if pathname == "scored" else None}
    if window:
        c["window"] = window
    return c


def main(open_window=False):
    os.makedirs(REPORTS, exist_ok=True)
    b = build()
    cal, names, panel, frames = b["cal"], b["names"], b["panel"], b["frames"]
    digest = panel_hash(frames["close"])

    # (a) the lock, before any run. Untouched.lock takes a fraction; the fraction is whatever puts the
    # first session of UNTOUCHED_START at the boundary, and the boundary is asserted, not trusted.
    lock = Untouched(LOCK)
    n_after = int((cal >= UNTOUCHED_START).sum())
    body = lock.lock(cal, n_after / len(cal), data_hash=digest)
    first = str(cal[cal >= UNTOUCHED_START][0].date())
    if body["start"] != first:
        raise SystemExit(f"lock boundary {body['start']} is not the first session on/after {UNTOUCHED_START} ({first})")
    hold = pd.Timestamp(body["start"])
    print(f"panel {cal[0].date()} -> {cal[-1].date()}, {len(cal)} sessions, {len(names)} names, close hash {digest}")
    pit_priced = (b["member_pit"] & frames["close"].notna()).sum(axis=1)
    print(f"PIT members with a lake price: {int(pit_priced.loc[IS_START:IS_END].median())} median in-sample "
          f"(index holds ~500); reference list: {b['n_ref']} of {len(mom_data.UNIVERSE)} names in the lake")
    print(f"untouched window {body['start']} -> {body['end']} ({body['n_sessions']} sessions), "
          f"{'OPENED ' + str(body['opened_at']) if body['opened_at'] else 'locked, not opened'}")
    print(f"costs: {COST_BPS:.0f} bps + book model {COSTS}; capital for participation ${CAPITAL:,.0f}; "
          f"vol target {TARGET_VOL:.0%} on {VOL_DAYS}d, gross <= {MAX_GROSS:.0f}")

    reg = Registry(REGISTRY)
    pre_end = cal[cal < hold][-1]
    if str(pre_end.date()) != IS_END:
        raise SystemExit(f"in-sample end {pre_end.date()} != {IS_END}")

    # (b) walk-forward on 2005-2020 through research.alpha.base: one variant, so the folds are a stability check.
    ins = panel.index[(panel.index >= IS_START) & (panel.index <= IS_END)]
    wf_table, wf_oos, wf_full = walk_forward(lambda **k: XSTrend("member_pit", TOP_N, **k), [{}], panel.loc[ins],
                                             frames["close"].loc[ins], FOLDS, TEST_SIZE, cost_bps=WF_COST_BPS,
                                             registry=reg, universe=names)
    reg.record("XSTrend walk-forward", {"folds": FOLDS, "test_size": TEST_SIZE, "cost_bps": WF_COST_BPS}, names,
               (ins[0].date(), ins[-1].date()), wf_oos, tags={"stage": "walk_forward_oos"})

    # (c) the three configurations on this engine, in-sample only.
    runs = {k: run_alpha(a, panel, frames, b["liquidity"], pre_end) for k, a in ALPHAS.items()}
    stats = {}
    for k, res in runs.items():
        for p in ("unlevered", "scored"):
            r = res[p]["ret"].loc[IS_START:IS_END]
            stats[(k, p)] = row(res[p], IS_START, IS_END)
            reg.record(str(ALPHAS[k]), config(k, ALPHAS[k], p), names, (IS_START, IS_END), r,
                       tags={"stage": "in_sample"})

    # per-fold Sharpe of the scored PIT path on the walk-forward's own test windows
    r_pit = runs["pit"]["scored"]["ret"]
    wf_table["scored_test_sharpe"] = [sharpe(r_pit.loc[str(a):str(z)]) for a, z in
                                      zip(wf_table["test_start"], wf_table["test_end"])]

    # --- the table
    cols = list(ALPHAS)
    tab = pd.DataFrame({LABELS[k]: {
        "net Sharpe, scored": stats[(k, "scored")]["sharpe"],
        "net Sharpe, unlevered": stats[(k, "unlevered")]["sharpe"],
        "annual return, scored": stats[(k, "scored")]["annual_return"],
        "annual return, unlevered": stats[(k, "unlevered")]["annual_return"],
        "annual vol, scored": stats[(k, "scored")]["annual_vol"],
        "annual vol, unlevered": stats[(k, "unlevered")]["annual_vol"],
        "max drawdown, scored": stats[(k, "scored")]["max_drawdown"],
        "max drawdown, unlevered": stats[(k, "unlevered")]["max_drawdown"],
        "turnover one-way /yr, scored": stats[(k, "scored")]["turnover_oneway"],
        "avg names held": stats[(k, "scored")]["avg_names"],
        "avg gross, scored": stats[(k, "scored")]["avg_gross"],
        "cost drag /yr, scored": stats[(k, "scored")]["cost_drag"],
        "days invested": stats[(k, "scored")]["pct_days_invested"],
    } for k in cols})
    pd.set_option("display.width", 160)
    print(f"\nIN-SAMPLE {IS_START} -> {IS_END}, net of {COST_BPS:.0f} bps + book cost model, fills at next open\n")
    print(tab.round(3).to_string())
    gap = {p: {"sharpe": stats[("pit", p)]["sharpe"] - stats[("ref", p)]["sharpe"],
               "annual_return": stats[("pit", p)]["annual_return"] - stats[("ref", p)]["annual_return"]}
           for p in ("scored", "unlevered")}
    print(f"\nPIT minus reference: scored Sharpe {gap['scored']['sharpe']:+.3f}, annual return "
          f"{gap['scored']['annual_return']:+.2%}; unlevered Sharpe {gap['unlevered']['sharpe']:+.3f}, "
          f"annual return {gap['unlevered']['annual_return']:+.2%}")
    print(f"PIT minus EW basket (the comparator the window must beat): scored Sharpe "
          f"{stats[('pit', 'scored')]['sharpe'] - stats[('bh', 'scored')]['sharpe']:+.3f}")
    spy = b["spy"].loc[IS_START:IS_END]
    both = pd.concat([r_pit.loc[IS_START:IS_END], spy], axis=1).dropna()
    beta = float(both.cov().iloc[0, 1] / both.iloc[:, 1].var())
    print(f"scored PIT beta to SPY {beta:.2f}; stock-days held with no print (PIT, unlevered): "
          f"{runs['pit']['unlevered']['missing_held']}, fills with no open: {runs['pit']['unlevered']['missing_fill']}")
    kills = kill_dates(r_pit.loc[IS_START:IS_END])
    print(f"PREREG §Kills 20% drawdown line crossed {len(kills)}x in-sample on the scored path: "
          f"{', '.join(map(str, kills)) or 'never'} (diagnostic; not applied, see README)")

    print(f"\nWALK-FORWARD, {FOLDS} folds x {TEST_SIZE} sessions, base engine ({WF_COST_BPS:.0f} bps flat, close "
          f"fills, unlevered) with the scored path's Sharpe on the same test windows:\n")
    print(wf_table.drop(columns=["picked"]).round(3).to_string())
    print(f"stitched OOS Sharpe (base engine) {sharpe(wf_oos[wf_oos != 0]):.3f}; full in-sample base-engine Sharpe "
          f"{sharpe(wf_full['{}'][wf_full['{}'] != 0]):.3f} vs this engine's unlevered {stats[('pit', 'unlevered')]['sharpe']:.3f}")
    d = reg.dsr(r_pit.loc[IS_START:IS_END])
    print(f"in-sample scored PIT: PSR {d['psr']:.3f}, DSR {d['dsr']:.3f} over {d['n_trials']} logged trials")

    # --- files
    years = {}
    for k in cols:
        for p in ("unlevered", "scored"):
            y = by_year(runs[k][p]["ret"].loc[IS_START:IS_END])
            years[f"{k}_{p}_sharpe"], years[f"{k}_{p}_return"], years[f"{k}_{p}_maxdd"] = y["sharpe"], y["return"], y["max_drawdown"]
    years = pd.DataFrame(years)
    years.index.name = "period"
    years.insert(0, "kind", "year")
    folds = pd.DataFrame({"kind": "fold", "period": [f"fold {i}" for i in wf_table.index],
                          "start": list(wf_table["test_start"]), "end": list(wf_table["test_end"]),
                          "pit_scored_sharpe": list(wf_table["scored_test_sharpe"]),
                          "base_engine_test_sharpe": list(wf_table["picked_test_sharpe"])}).set_index("period")
    pd.concat([years.reset_index(), folds.reset_index()], ignore_index=True).to_csv(
        os.path.join(REPORTS, "prior_insample.csv"), index=False)
    out = {"panel": {"start": str(cal[0].date()), "end": str(cal[-1].date()), "sessions": len(cal), "names": len(names),
                     "close_hash": digest, "n_ref_in_lake": b["n_ref"]},
           "untouched": {k: body[k] for k in ("start", "end", "n_sessions", "data_hash", "opened_at")},
           "in_sample": {f"{k}_{p}": v for (k, p), v in stats.items()}, "gap_pit_minus_ref": gap,
           "beta_spy_scored_pit": beta, "kill_dates_scored_pit": [str(x) for x in kills],
           "walk_forward": wf_table.reset_index().to_dict("records"),
           "walk_forward_oos_sharpe": sharpe(wf_oos[wf_oos != 0]), "dsr_in_sample": d,
           "n_trials": reg.trials(), "missing": {k: {"held": runs[k]["unlevered"]["missing_held"],
                                                     "fill": runs[k]["unlevered"]["missing_fill"]} for k in cols}}
    with open(os.path.join(REPORTS, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    pd.DataFrame({f"{k}_{p}": runs[k][p]["ret"].loc[IS_START:IS_END] for k in cols for p in ("unlevered", "scored")}
                 ).to_csv(os.path.join(REPORTS, "returns_insample.csv"))

    # (d) the window stays shut. This is what opens it, once.
    if not open_window:
        print(f"\nuntouched window {body['start']} -> {body['end']} NOT opened (hash {digest}). The reviewer opens it with:\n"
              f"    cd {HERE} && ../.venv/bin/python3 prior.py --open")
        return

    def fn(start, end):
        """The one read: scored PIT sleeve vs the scored EW basket of the same universe; clears if > 0 and > basket."""
        res = {}
        for k in ("pit", "bh"):
            full = run_alpha(ALPHAS[k], panel, frames, b["liquidity"], end)
            r = full["scored"]["ret"].loc[start:end]
            res[k] = {**row(full["scored"], start, end), "unlevered_sharpe": row(full["unlevered"], start, end)["sharpe"]}
            reg.record(str(ALPHAS[k]), config(k, ALPHAS[k], "scored", "untouched"), names, (start.date(), end.date()),
                       r, tags={"stage": "untouched"})
            if k == "pit":
                res["dsr"] = reg.dsr(r)
                res["kill_dates"] = [str(x) for x in kill_dates(r)]
        res["clears"] = bool(res["pit"]["sharpe"] > 0 and res["pit"]["sharpe"] > res["bh"]["sharpe"])
        return res

    try:
        result = lock.open(fn)
        print("\nUNTOUCHED WINDOW, opened now:")
    except UntouchedWindowUsed as e:
        result = lock.result
        print(f"\n{e}; the stored result:")
    print(json.dumps(result, indent=2, default=str))
    print("\nCLEARS the prior layer (scored net Sharpe > 0 and > the EW basket at the same vol): "
          + ("YES -> gate layer, 6-month paper track" if result["clears"] else "NO -> REFUSALS.md §1, candidate stops"))


if __name__ == "__main__":
    main(open_window="--open" in sys.argv)
