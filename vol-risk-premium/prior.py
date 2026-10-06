"""The prior layer of PREREG.md: the backtest that decides whether VRP earns a forward track.

Everything the pre-registration freezes is a constant here. The untouched window
is locked on first run and opened once, and only after the XSP spread has been
measured and frozen (reports/spread-frozen.json), so the window is never read
against a provisional cost.

Run:  ../.venv/bin/python3 prior.py            lock the window, run the pre-window luck table
      ../.venv/bin/python3 prior.py --open     open the untouched window once (refuses until the spread is frozen)
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
from research.alpha.base import Untouched  # noqa: E402
from research.registry.experiments import Registry  # noqa: E402
from vrp import TRADING_DAYS, metrics, simulate  # noqa: E402

REPORTS = data.REPORTS
LOCK = REPORTS / "untouched.json"
FROZEN = REPORTS / "spread-frozen.json"
REGISTRY = REPORTS / "registry"

# PREREG.md §4, frozen. The spread is provisional until FROZEN exists (§4: the larger
# of the measured XSP number and 1.5%); the window cannot be opened before that.
BASIS = 0.024
SPREAD_PROVISIONAL = 0.015
COST_BPS = 1.0
HORIZON = 21
OFFSETS = range(HORIZON)           # §5: the 21-offset luck test, every one a logged trial
HOLDOUT = 0.2                      # §5: last fifth, same rule as the book
# §3 sizing. ACCOUNT is the ceiling Kelvin set; CAP and PER_UNIT_ACCOUNT are the signed numbers.
# One XSP straddle is 100 x the index, about 100 x SPY: the notional per unit is MULTIPLIER x spot.
ACCOUNT = 35_000.0
CAP = 4
PER_UNIT_ACCOUNT = 7_500.0
TARGET_VOL = 0.15
VOL_DAYS = 63
MULTIPLIER = 100
# §6 kill rules
VIX_ENTRY_MAX = 30.0
FLATTEN = -0.10                    # one day's P&L below this fraction of notional flattens the cycle


def panel():
    df = data.get_daily()
    digest = hashlib.sha256(open(data.DAILY_CACHE, "rb").read()).hexdigest()[:16]
    return df, digest


def spread():
    """The option spread the backtest pays: frozen if measured, provisional otherwise."""
    if FROZEN.exists():
        return float(json.load(open(FROZEN))["spread"]), True
    return SPREAD_PROVISIONAL, False


def kills(df, sim):
    """Apply §6 to a constant-notional simulation: no entry above the VIX ceiling, flatten past the day limit."""
    out = sim.copy()
    for start, g in out.groupby("cycle"):
        if df["vix"].loc[g.index[0]] > VIX_ENTRY_MAX:
            out.loc[g.index, ["pnl", "hedge", "premium"]] = 0.0
            out.loc[g.index, "cycle"] = -1
            continue
        hit = g.index[g["pnl"] <= FLATTEN]
        if len(hit):
            after = g.index[g.index > hit[0]]
            out.loc[after, ["pnl", "hedge"]] = 0.0
    return out


def capped(df, sim, account=ACCOUNT, cap=CAP, per_unit_account=PER_UNIT_ACCOUNT, target=TARGET_VOL):
    """Dollar P&L of the sized path: integer units fixed at each cycle start, vol-targeted, capped.

    Units at a cycle start = min(cap, account / per_unit_account, units that bring the
    trailing VOL_DAYS realised P&L vol of one unit to `target` x account), floored to a
    whole contract. Returns daily dollar P&L, units held and the notional per unit.
    """
    spot = df["spy"].reindex(sim.index)
    per_unit = sim["pnl"] * spot * MULTIPLIER                       # $ per unit, per day
    trailing = per_unit.rolling(VOL_DAYS, min_periods=VOL_DAYS // 2).std() * np.sqrt(TRADING_DAYS)
    units = pd.Series(0.0, index=sim.index)
    for start, g in sim.groupby("cycle"):
        if start < 0:
            continue
        vol = trailing.shift(1).loc[g.index[0]]
        if not np.isfinite(vol) or vol <= 0:
            continue                                   # no vol estimate yet: no position, not the cap
        by_vol = target * account / vol
        units.loc[g.index] = float(np.floor(min(cap, account / per_unit_account, by_vol)))
    return pd.DataFrame({"pnl_usd": per_unit * units, "units": units, "notional_unit": spot * MULTIPLIER})


def score(df, start, end, offset, option_spread, **sizing):
    """One configuration over [start, end]: the §6-killed, §3-sized path and its statistics."""
    sub = df.loc[start:end]
    sim = kills(sub, simulate(sub, horizon=HORIZON, cost_bps=COST_BPS, option_spread=option_spread,
                              iv_offset=BASIS, offset=offset))
    sized = capped(sub, sim, **sizing)
    account = sizing.get("account", ACCOUNT)
    r_acct = sized["pnl_usd"] / account
    live = r_acct.abs().sum() > 0
    m = metrics(r_acct) if live else {"sharpe": 0.0, "max_drawdown": 0.0, "annual_pnl": 0.0}
    held = sized["units"] > 0
    return {"offset": offset, "const_sharpe": float(metrics(sim["pnl"])["sharpe"]),
            "sized_sharpe": float(m["sharpe"]), "sized_annual_pct": float(m["annual_pnl"]),
            "sized_maxdd_pct": float(m["max_drawdown"]),
            "worst_day_usd": float(sized["pnl_usd"].min()), "worst_day_pct_acct": float(sized["pnl_usd"].min() / account),
            "units_mean": float(sized["units"][held].mean()) if held.any() else 0.0,
            "units_max": float(sized["units"].max()),
            "cycles_skipped_vix": int((sub["vix"].iloc[offset::HORIZON] > VIX_ENTRY_MAX).sum()),
            "n_days": int(len(sim))}, r_acct


def main(open_window=False):
    df, digest = panel()
    lock = Untouched(str(LOCK))
    body = lock.lock(df.index, HOLDOUT, data_hash=digest)
    hold = pd.Timestamp(body["start"])
    sp, frozen = spread()
    print(f"panel {df.index[0].date()} -> {df.index[-1].date()}, {len(df)} days, hash {digest}")
    print(f"untouched window {body['start']} -> {body['end']} ({body['n_sessions']} sessions), "
          f"{'opened ' + str(body['opened_at']) if body['opened_at'] else 'not opened'}")
    print(f"basis {BASIS:.3f} frozen; spread {sp:.3%} {'FROZEN' if frozen else 'PROVISIONAL (window stays shut)'}; "
          f"cap {CAP}, ${PER_UNIT_ACCOUNT:,.0f}/unit, {TARGET_VOL:.0%} vol target on ${ACCOUNT:,.0f}")

    reg = Registry(str(REGISTRY))
    pre_end = df.index[df.index < hold][-1]
    rows = []
    for off in OFFSETS:
        row, r = score(df, df.index[0], pre_end, off, sp)
        rows.append(row)
        reg.record("VRP", {"basis": BASIS, "spread": sp, "cap": CAP, "offset": off, "path": "sized",
                           "spread_frozen": frozen}, ["XSP"], (df.index[0], pre_end), r)
    tab = pd.DataFrame(rows).set_index("offset")
    pd.set_option("display.width", 160)
    print(f"\nPRE-WINDOW {df.index[0].date()} -> {pre_end.date()}, {len(OFFSETS)} offsets, each a logged trial "
          f"(registry now holds {reg.trials()} trials)\n")
    print(tab.round(3).to_string())
    print(f"\nsized Sharpe across offsets: median {tab.sized_sharpe.median():.2f}, "
          f"min {tab.sized_sharpe.min():.2f}, max {tab.sized_sharpe.max():.2f}; "
          f"constant-notional median {tab.const_sharpe.median():.2f}")
    print(f"worst single day across offsets: ${tab.worst_day_usd.min():,.0f} = "
          f"{tab.worst_day_pct_acct.min():.1%} of the ${ACCOUNT:,.0f} account")
    tab.to_csv(REPORTS / "prior_prewindow.csv")

    if not open_window:
        return
    if not frozen:
        raise SystemExit("refusing to open the untouched window: the XSP spread is not frozen "
                         "(PREREG.md §4; write reports/spread-frozen.json from measure_spread.py first)")

    def fn(start, end):
        row, r = score(df, start, end, 0, sp)
        reg.record("VRP", {"basis": BASIS, "spread": sp, "cap": CAP, "offset": 0, "path": "sized",
                           "window": "untouched"}, ["XSP"], (start, end), r)
        row["dsr"] = reg.dsr(r)
        row["clears"] = bool(row["sized_sharpe"] > 0)
        return row

    res = lock.open(fn)
    print("\nUNTOUCHED WINDOW, opened once:")
    print(json.dumps(res, indent=2, default=str))
    print("\nCLEARS the prior layer (sized Sharpe > 0): " + ("YES -> forward track" if res["clears"]
                                                             else "NO -> REFUSALS.md §2, candidate stops"))


if __name__ == "__main__":
    main(open_window="--open" in sys.argv)
