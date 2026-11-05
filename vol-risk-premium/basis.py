"""Read-only clock for the PREREG.md §4 basis re-measure: how many usable sessions, why the rest were lost,
when session 60 lands, the running mean, and the XSP-vs-SPY clause.

A basis row is an NYSE session with a non-NaN SPY iv30 in the collector's horizons summary and a FRED VIXCLS
close; basis = VIX - 100*iv30. No close-only filter (the stale flag is printed, not applied): this is the
definition that reproduces the signed 10 sessions, mean 2.90.

Run:  ../.venv/bin/python3 basis.py          (writes nothing)
"""
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from framework.book.universe import sessions as nyse  # noqa: E402

SRC = Path(os.environ.get("C2_COLLECTOR_SRC", ROOT / "source-material" / "options-collector"))
VIX = Path(os.environ.get("C2_VIXCLS", ROOT / "data-lake" / "store" / "fred" / "VIXCLS.parquet"))
FIRST, TARGET, XSP_N, XSP_MAX = "2026-09-04", 60, 10, 0.3   # PREREG.md §4: first signed session, re-measure, clause
REASONS = ("no-folder", "empty", "no-SPY", "no-rate")


def table(src=SRC, vix=VIX, first=FIRST):
    """One row per NYSE session from `first` to the last day folder, with iv30 (vol pts), vix, stale, reason."""
    folders = sorted(p.name for p in Path(src).iterdir() if p.is_dir() and p.name[:2] == "20")
    v = pd.read_parquet(vix).sort_values("collected_at").drop_duplicates("date", keep="last")  # last collected_at per date
    v = pd.Series(v.value.values, index=v.date.dt.strftime("%Y-%m-%d"))
    rows = []
    for d in nyse(first, folders[-1]).strftime("%Y-%m-%d"):
        f = Path(src) / "summary" / f"{d}_horizons.csv"
        h = pd.read_csv(f) if f.exists() else pd.DataFrame()
        spy = h[h.ticker == "SPY"] if len(h) else h
        iv = 100 * spy.iv30.iloc[0] if len(spy) else float("nan")
        r = ("no-folder" if d not in folders else "empty" if not len(h) else "no-SPY" if iv != iv
             else "no-rate" if d not in v.index or v[d] != v[d] else "ok")
        rows.append((d, iv, v.get(d, float("nan")), bool(spy.stale.iloc[0]) if len(spy) else False, r))
    t = pd.DataFrame(rows, columns=["date", "iv30", "vix", "stale", "reason"]).set_index("date")
    t["basis"] = t.vix - t.iv30
    return t, folders


def project(t, last_folder, target=TARGET):
    """Date of usable session `target`, assuming every later session is usable and trailing no-rate days get their VIX."""
    ok = t[t.reason == "ok"]
    pending = len(t[(t.reason == "no-rate") & (t.index > (ok.index[-1] if len(ok) else ""))])
    need = target - len(ok) - pending
    if len(ok) >= target:
        return ok.index[target - 1]
    return nyse(pd.Timestamp(last_folder) + pd.Timedelta(days=1), pd.Timestamp(last_folder) + pd.Timedelta(days=400))[need - 1].strftime("%Y-%m-%d")


def xsp(t, src=SRC, n=XSP_N):
    """Mean and max-abs daily XSP-minus-SPY iv30 gap (vol pts) over the first n dates both exist; flag on |mean| > 0.3."""
    h = pd.concat([pd.read_csv(f) for f in sorted((Path(src) / "summary").glob("20*_horizons.csv"))])
    iv = h[h.ticker.isin(["SPY", "^XSP"])].pivot(index="date", columns="ticker", values="iv30").dropna()
    iv = (100 * iv).loc[[d for d in iv.index if d in t.index]].head(n)
    gap = iv["^XSP"] - iv["SPY"]
    mean = gap.mean() if len(gap) else float("nan")
    return dict(n=len(gap), mean=mean, max_abs=gap.abs().max() if len(gap) else float("nan"), flag=bool(abs(mean) > XSP_MAX),
                vix_minus_xsp=(t.vix.loc[iv.index] - iv["^XSP"]).mean() if len(gap) else float("nan"))


if __name__ == "__main__":
    t, folders = table()
    ok = t[t.reason == "ok"]
    print(f"n usable: {len(ok)} of {TARGET} (basis = VIX - 100*iv30, NYSE sessions {t.index[0]} -> {t.index[-1]})")
    print(f"day folders: {len(folders)} (info only; includes holidays and empty days)")
    print("lost sessions: " + ", ".join(f"{r} {(t.reason == r).sum()}" for r in REASONS))
    print(f"stale-flagged among usable: {int(ok.stale.sum())} (printed, not filtered)")
    print(f"projected session {TARGET}: {project(t, folders[-1])} (assumes no further misses; pending VIX days count)")
    print(f"running mean: {ok.basis.mean():+.3f} over {len(ok)}; first {XSP_N}: {ok.basis.head(XSP_N).mean():.3f}; "
          f"retire line 3.0 -> {'ABOVE' if ok.basis.mean() > 3.0 else 'below'}")
    x = xsp(t)
    print(f"XSP clause, first {x['n']} dual sessions: mean gap XSP-SPY {x['mean']:+.3f}, max abs daily gap {x['max_abs']:.3f}; "
          f"flag (|mean| > {XSP_MAX}): {x['flag']}")
    print(f"  other sign reading SPY-XSP: {-x['mean']:+.3f} (same flag); if flagged, XSP basis VIX-XSP iv30 = {x['vix_minus_xsp']:.3f} replaces 2.90")
    print("  note: PREREG §4 says 'ATM IV minus VIX'; the signed 2.90 is VIX minus ATM IV, so that wording is sign-reversed.")
