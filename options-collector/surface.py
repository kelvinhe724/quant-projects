"""Turn one day's raw chains into a compact surface summary.

Clean mids and the Black-76 inversion come from the svi project, the horizon
interpolation and the implied-correlation solve from the dispersion project,
both imported by file path so this folder stays a thin consumer of them.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import data

PROJECTS = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


svi = _load("svi_model", PROJECTS / "svi" / "svi.py")
svi_data = _load("svi_data", PROJECTS / "svi" / "data.py")
dispersion = _load("dispersion_model", PROJECTS / "dispersion" / "dispersion.py")

implied_vol = svi.implied_vol
bs_price = svi.bs_price
implied_correlation = dispersion.implied_correlation
vol_at_horizon = dispersion.vol_at_horizon

HORIZONS = (30, 60, 90)
MIN_QUOTES = 6
STALE_ZERO_FRAC = 0.5

SUMMARY_COLS = ["date", "ticker", "snapshot", "session", "expiry", "T", "spot", "F", "r",
                "n_raw", "n_clean", "k_min", "k_max", "atm_iv", "iv_25d_put",
                "iv_25d_call", "skew_25d", "stale"]
HORIZON_COLS = ["date", "ticker", "n_expiries", "iv30", "iv60", "iv90",
                "rho30", "rho60", "n_names", "weights_from", "stale"]


def stale_stats(q):
    """Share of raw quotes with no market at all, and with no trade in three days."""
    zero = float(((q["bid"] <= 0) & (q["ask"] <= 0)).mean())
    last = pd.to_datetime(q["lastTradeDate"], utc=True).dt.tz_localize(None)
    snap = pd.to_datetime(q["snapshot"])
    old = float(((snap - last) > pd.Timedelta(days=3)).mean())
    return {"zero_quote_frac": zero, "old_trade_frac": old, "stale": zero > STALE_ZERO_FRAC}


def clean_with_vols(q, r):
    """svi's cleaning pipeline plus its Black-76 inversion, one row per OTM quote."""
    try:
        clean, fwd, counts = svi_data.clean(q, r=r, min_quotes=MIN_QUOTES)
    except KeyError:
        # svi's forward table has no columns when no expiry has a parity pair,
        # which is what a dead chain looks like
        return pd.DataFrame(), pd.DataFrame(), {"raw": len(q)}
    if clean.empty:
        return clean, fwd, counts
    return svi.add_implied_vols(clean), fwd, counts


def forward_delta(k, iv, T, cp):
    """Black-76 delta without the discount factor, from log-moneyness against F."""
    sd = iv * np.sqrt(T)
    d1 = (-k + 0.5 * sd ** 2) / sd
    return np.where(cp == "C", norm.cdf(d1), norm.cdf(d1) - 1)


def _interp(x, y, target):
    """Linear interpolation that refuses to extrapolate."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    if len(x) < 2 or not (x[0] <= target <= x[-1]):
        return np.nan
    return float(np.interp(target, x, y))


def slice_metrics(g):
    """ATM vol, 25-delta wing vols and skew for one expiry's clean OTM quotes."""
    k, iv, T = g["k"].to_numpy(float), g["iv_mid"].to_numpy(float), float(g["T"].iloc[0])
    delta = forward_delta(k, iv, T, g["cp"].to_numpy())
    puts, calls = g["cp"].to_numpy() == "P", g["cp"].to_numpy() == "C"
    put25 = _interp(delta[puts], iv[puts], -0.25)
    call25 = _interp(delta[calls], iv[calls], 0.25)
    return {"n_clean": len(g), "k_min": float(k.min()), "k_max": float(k.max()),
            "atm_iv": _interp(k, iv, 0.0), "iv_25d_put": put25, "iv_25d_call": call25,
            "skew_25d": put25 - call25}


def summarise_chain(q, ticker, date, session):
    """One row per expiry for one ticker. Returns (rows, stale dict)."""
    st = stale_stats(q)
    r = float(q["r"].iloc[0]) if q["r"].notna().any() else np.nan
    snapshot = str(q["snapshot"].iloc[0])
    spot = float(q["spot"].iloc[0])
    raw_n = q.groupby("expiry").size()
    rows = []
    if np.isnan(r):
        return pd.DataFrame(columns=SUMMARY_COLS), st
    clean, fwd, _ = clean_with_vols(q, r)
    if clean.empty:
        return pd.DataFrame(columns=SUMMARY_COLS), st
    for expiry, g in clean.groupby("expiry"):
        f = fwd[fwd["expiry"] == expiry].iloc[0]
        row = {"date": date, "ticker": ticker, "snapshot": snapshot, "session": session,
               "expiry": pd.Timestamp(expiry).strftime("%Y-%m-%d"), "T": float(g["T"].iloc[0]),
               "spot": spot, "F": float(f["F"]), "r": r, "n_raw": int(raw_n[expiry])}
        row.update(slice_metrics(g))
        row["stale"] = st["stale"]
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLS), st


def horizon_rows(summary, weights, weights_from):
    """Fixed-horizon ATM vols per ticker and the basket implied correlation on the SPY row."""
    rows = {}
    for t, g in summary.groupby("ticker"):
        atm = g.dropna(subset=["atm_iv"])
        row = {"date": g["date"].iloc[0], "ticker": t, "n_expiries": len(atm),
               "stale": bool(g["stale"].iloc[0])}
        for h in HORIZONS:
            row[f"iv{h}"] = vol_at_horizon(atm, h) if len(atm) else np.nan
        rows[t] = row
    out = pd.DataFrame(list(rows.values()))
    out["rho30"] = out["rho60"] = np.nan
    out["n_names"] = 0
    out["weights_from"] = ""
    if data.INDEX in rows and weights is not None:
        names = [n for n in data.NAMES if n in rows and not np.isnan(rows[n]["iv30"])]
        if len(names) >= 2:
            w = weights.reindex(names)
            w = w / w.sum()
            at = out.set_index("ticker")
            idx = at.index == data.INDEX
            for h in (30, 60):
                out.loc[idx, f"rho{h}"] = implied_correlation(
                    at.loc[data.INDEX, f"iv{h}"], at.loc[names, f"iv{h}"], w)
            out.loc[idx, "n_names"] = len(names)
            out.loc[idx, "weights_from"] = weights_from
    return out[HORIZON_COLS]


def summarise_day(date, force=False):
    """Write summary/<date>.csv and summary/<date>_horizons.csv unless they exist."""
    data.SUMMARY.mkdir(parents=True, exist_ok=True)
    path, hpath = data.SUMMARY / f"{date}.csv", data.SUMMARY / f"{date}_horizons.csv"
    if path.exists() and hpath.exists() and not force:
        return pd.read_csv(path), pd.read_csv(hpath)
    chains, manifest = data.load_day(date)
    parts = []
    for t, q in chains.items():
        s, st = summarise_chain(q, t, date, manifest["session"])
        if st["stale"]:
            print(f"{date} {t}: stale chain, {st['zero_quote_frac']:.0%} of quotes have no "
                  f"bid or ask", file=sys.stderr)
        if len(s):
            parts.append(s)
    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SUMMARY_COLS)
    weights, weights_from = data.load_weights(date)
    horizons = horizon_rows(summary, weights, weights_from) if len(summary) \
        else pd.DataFrame(columns=HORIZON_COLS)
    summary.to_csv(path, index=False)
    horizons.to_csv(hpath, index=False)
    return summary, horizons


def load_summary():
    """Every day's per-expiry summary in one frame, oldest first."""
    files = sorted(p for p in data.SUMMARY.glob("20*.csv") if "_horizons" not in p.name)
    if not files:
        return pd.DataFrame(columns=SUMMARY_COLS)
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def load_horizons():
    """Every day's horizon table in one frame, oldest first."""
    files = sorted(data.SUMMARY.glob("20*_horizons.csv"))
    if not files:
        return pd.DataFrame(columns=HORIZON_COLS)
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


if __name__ == "__main__":
    days = data.list_days()
    if not days:
        raise SystemExit("nothing collected yet, run collect.py first")
    s, h = summarise_day(days[-1], force=True)
    print(s.to_string())
    print(h.to_string())
