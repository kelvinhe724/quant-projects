"""Offline checks on a planted vol surface: idempotent storage, the IV solver
round-trips known prices, the summary schema is fixed, a dead chain is flagged,
and three sessions a day coexist with close as the default.

Run: python3 check.py
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import data
import surface
from collect import collect_day

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


NOW = pd.Timestamp("2026-09-04 21:05:00", tz="UTC")
DATE = "2026-09-04"
R = 0.04


def planted_iv(k):
    """Skewed smile: 20% at the money, puts richer than calls."""
    return 0.20 - 0.15 * k + 0.10 * k ** 2


def fake_chain(ticker, now, r, spot=100.0, dead_frac=0.0, seed=0):
    """Build a chain priced off planted_iv with a 4 cent spread, as fetch_chain would return it."""
    rng = np.random.default_rng(seed)
    naive = now.tz_convert("UTC").tz_localize(None)
    rows = []
    for days in (30, 90, 180):
        T = days / 365
        F, D = spot * np.exp(r * T), np.exp(-r * T)
        for K in np.arange(70, 132.5, 2.5):
            iv = planted_iv(np.log(K / F))
            for cp in ("C", "P"):
                mid = float(surface.bs_price(F, K, T, iv, D, cp))
                rows.append({"snapshot": naive, "expiry": naive.normalize() + pd.Timedelta(days=days),
                             "T": T, "cp": cp, "strike": K, "bid": mid - 0.02, "ask": mid + 0.02,
                             "lastPrice": mid, "volume": 10, "openInterest": 100,
                             "impliedVolatility": iv, "lastTradeDate": now, "spot": spot, "r": r})
    q = pd.DataFrame(rows)[data.CHAIN_COLS]
    dead = rng.random(len(q)) < dead_frac
    q.loc[dead, ["bid", "ask", "volume"]] = 0.0
    return q, spot, DATE


def fake_rate(date):
    return R, date, "DTB3"


def fake_caps(names):
    return pd.Series({n: 1e12 * (i + 1) for i, n in enumerate(names)}, name="market_cap")


def digest(folder):
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in Path(folder).iterdir() if p.is_file() and p.name != "manifest.json"}


tmp = Path(tempfile.mkdtemp())
data.ROOT, data.SUMMARY = tmp / "raw", tmp / "summary"

print("IDEMPOTENT STORAGE\n")

m1 = collect_day(DATE, NOW, fetch=fake_chain, rate=fake_rate, caps=fake_caps)
check(f"first run writes every ticker ({len(m1['written_this_run'])} of {len(data.UNIVERSE)})",
      m1["written_this_run"] == data.UNIVERSE and not m1["missing"])
check("the day folder has one CSV per ticker plus rates, underlying, weights, manifest",
      sorted(p.name for p in data.day_dir(DATE).iterdir())
      == sorted([f"{t}.csv" for t in data.UNIVERSE]
                + ["rates.csv", "underlying.csv", "weights.csv", "manifest.json"]))
before = digest(data.day_dir(DATE))

calls = []


def counting_fetch(ticker, now, r):
    calls.append(ticker)
    return fake_chain(ticker, now, r, spot=999.0)


m2 = collect_day(DATE, NOW + pd.Timedelta(hours=1), fetch=counting_fetch, rate=fake_rate,
                 caps=fake_caps)
check("a second run on the same date fetches nothing", calls == [])
check("a second run writes nothing and skips everything",
      m2["written_this_run"] == [] and m2["skipped_existing"] == data.UNIVERSE)
check("every file is byte-identical after the rerun", digest(data.day_dir(DATE)) == before)
check("the rate is read back from disk, not refetched", m2["rate"] == R)


def flaky_fetch(ticker, now, r):
    if ticker == "XOM":
        raise RuntimeError("simulated timeout")
    return fake_chain(ticker, now, r)


DATE2 = "2026-09-08"
m3 = collect_day(DATE2, NOW + pd.Timedelta(days=4), fetch=flaky_fetch, rate=fake_rate,
                 caps=fake_caps)
check("a ticker whose download fails is recorded as missing, not filled in",
      m3["missing"] == ["XOM"] and not (data.day_dir(DATE2) / "XOM.csv").exists())
m4 = collect_day(DATE2, NOW + pd.Timedelta(days=4), fetch=fake_chain, rate=fake_rate,
                 caps=fake_caps)
check("a rerun fills only the gap", m4["written_this_run"] == ["XOM"]
      and len(m4["skipped_existing"]) == len(data.UNIVERSE) - 1)
check("the manifest lists every ticker on disk after the gap is filled",
      m4["collected"] == sorted(data.UNIVERSE))
check("session is labelled from the New York clock",
      data.session(pd.Timestamp("2026-09-04 14:00", tz="UTC")) == "intraday"
      and data.session(pd.Timestamp("2026-09-04 21:00", tz="UTC")) == "close")
check("the trading date is the New York date, not UTC",
      data.trading_date(pd.Timestamp("2026-09-05 02:00", tz="UTC")) == "2026-09-04")

print("\nIMPLIED VOL SOLVER (imported from svi)\n")

F, D = 100.0, np.exp(-R * 0.5)
grid = [(K, vol, cp) for K in (80, 95, 100, 105, 120) for vol in (0.1, 0.25, 0.6)
        for cp in ("C", "P")]
err = max(abs(surface.implied_vol(float(surface.bs_price(F, K, 0.5, vol, D, cp)),
                                  F, K, 0.5, D, cp) - vol) for K, vol, cp in grid)
check(f"implied_vol inverts bs_price on a 30-point grid (max error {err:.1e})", err < 1e-7)
check("a price below intrinsic returns NaN, not a vol",
      np.isnan(surface.implied_vol(D * 19.0, F, 80.0, 0.5, D, "C")))
check("a price above the forward returns NaN",
      np.isnan(surface.implied_vol(D * 101.0, F, 80.0, 0.5, D, "C")))
one = float(surface.bs_price(100.0, 100.0, 1.0, 0.2, 1.0, "C"))
check(f"undiscounted ATM call, F=K=100, T=1, vol 20% prices at 7.9656 (got {one:.4f})",
      abs(one - 7.9656) < 1e-3)

chains, manifest = data.load_day(DATE)
q = chains["SPY"]
clean, fwd, counts = surface.clean_with_vols(q, R)
recovered = np.abs(clean["iv_mid"] - planted_iv(clean["k"])).max()
check(f"the cleaned panel recovers the planted smile from mids (max error {recovered:.1e})",
      recovered < 2e-3)
check("parity forward matches the planted forward within 5bp",
      (fwd["F"] / (100 * np.exp(R * fwd["T"])) - 1).abs().max() < 5e-4)
check("only out-of-the-money quotes survive cleaning",
      (((clean["cp"] == "P") & (clean["k"] < 0)) | ((clean["cp"] == "C") & (clean["k"] >= 0))).all())

print("\nSUMMARY SCHEMA\n")

s, st = surface.summarise_chain(q, "SPY", DATE, manifest["session"])
check("summary columns are exactly SUMMARY_COLS, in order", list(s.columns) == surface.SUMMARY_COLS)
check("one row per expiry", len(s) == 3 and s["expiry"].is_unique)
check(f"ATM vol reads the planted 20% (got {s['atm_iv'].iloc[0]:.4f})",
      (s["atm_iv"] - 0.20).abs().max() < 2e-3)
k25 = {}
for _, row in s.iterrows():
    # planted surface: solve for the k where the forward delta is +-0.25, then read the smile
    T = row["T"]
    ks = np.linspace(-0.4, 0.4, 4001)
    d = surface.forward_delta(ks, planted_iv(ks), T, np.where(ks < 0, "P", "C"))
    k25[row["expiry"]] = (ks[np.argmin(abs(d + 0.25))], ks[np.argmin(abs(d - 0.25))])
skew_err = max(abs(row["skew_25d"] - (planted_iv(k25[row["expiry"]][0]) - planted_iv(k25[row["expiry"]][1])))
               for _, row in s.iterrows())
check(f"25-delta skew matches the planted smile at the 25-delta strikes (max error {skew_err:.1e})",
      skew_err < 3e-3)
check("puts are richer than calls, as planted", (s["skew_25d"] > 0).all())
check("no row is flagged stale on a live chain", not s["stale"].any())

summary, horizons = surface.summarise_day(DATE)
check("summarise_day covers every ticker", set(summary["ticker"]) == set(data.UNIVERSE))
check("horizon columns are exactly HORIZON_COLS, in order",
      list(horizons.columns) == surface.HORIZON_COLS)
check("a flat-in-T planted surface interpolates to 20% at 30, 60 and 90 days",
      (horizons[["iv30", "iv60", "iv90"]] - 0.20).abs().max().max() < 2e-3)
spy = horizons.set_index("ticker").loc["SPY"]
# the planted surface is flat in T, so any horizon rule would pass above; this
# one has a term structure, and only total-variance interpolation gives 18.03%
two = pd.DataFrame({"date": DATE, "ticker": "SPY", "T": [30 / 365, 90 / 365],
                    "atm_iv": [0.10, 0.20], "stale": False})
tv = np.sqrt((0.01 * 30 / 365 + 0.04 * 90 / 365) / 2 / (60 / 365))
got60 = surface.horizon_rows(two, None, "").set_index("ticker").loc["SPY", "iv60"]
check(f"a 10%/20% term structure interpolates to {tv:.2%} at 60 days in total variance (got {got60:.2%})",
      abs(got60 - tv) < 1e-12)
check(f"implied correlation is 1 when index and names share one vol (got {spy['rho30']:.4f})",
      abs(spy["rho30"] - 1.0) < 1e-6 and spy["n_names"] == len(data.NAMES))
check("implied correlation only appears on the SPY row",
      horizons.loc[horizons["ticker"] != "SPY", "rho30"].isna().all())
check("weights came from the day's own weights.csv", spy["weights_from"] == DATE)
first = (data.SUMMARY / f"{DATE}.csv").stat().st_mtime_ns
surface.summarise_day(DATE)
check("summarising an already summarised day rewrites nothing",
      (data.SUMMARY / f"{DATE}.csv").stat().st_mtime_ns == first)

vols = np.array([0.3, 0.25, 0.4])
w = np.array([0.5, 0.3, 0.2])
for rho in (0.2, 0.7):
    back = surface.implied_correlation(surface.dispersion.basket_vol(vols, w, rho), vols, w)
    check(f"implied_correlation inverts basket_vol at rho={rho} (got {back:.4f})", abs(back - rho) < 1e-10)

print("\nSTALE QUOTES\n")

dead, _, _ = fake_chain("SPY", NOW, R, dead_frac=0.8, seed=1)
st = surface.stale_stats(dead)
check(f"a chain with 80% empty quotes is flagged stale ({st['zero_quote_frac']:.0%} empty)", st["stale"])
check("a chain that is mostly dead produces no summary rows at all", surface.summarise_chain(dead, "SPY", DATE, "close")[0].empty)
# kill the wings only, so the near-the-money slices survive cleaning and carry the flag
wings = q.copy()
wings.loc[(wings["strike"] - 100).abs() > 10, ["bid", "ask", "volume"]] = 0.0
s_wings, st_w = surface.summarise_chain(wings, "SPY", DATE, "close")
check(f"a chain dead in the wings ({st_w['zero_quote_frac']:.0%} empty) still summarises and every row carries the stale flag",
      st_w["stale"] and len(s_wings) == 3 and s_wings["stale"].all())
check("the live chain is not flagged", not surface.stale_stats(q)["stale"])
old = q.copy()
old["lastTradeDate"] = NOW - pd.Timedelta(days=5)
check("quotes last traded five days ago are all dropped by the cleaning pass",
      surface.clean_with_vols(old, R)[0].empty)
check("old-trade fraction is reported separately from the empty-quote fraction",
      surface.stale_stats(old)["old_trade_frac"] == 1.0 and not surface.stale_stats(old)["stale"])

# ------------------------------------------------ vendor flakiness: retry and exit code
import collect

calls = []


def flaky_once(ticker, now, r, max_days=data.MAX_DAYS):
    calls.append(ticker)
    if calls.count(ticker) == 1:
        raise RuntimeError("curl (28) timed out")
    return fake_chain(ticker, now, r)


real_once = data._fetch_chain_once
try:
    data._fetch_chain_once = flaky_once
    q = data.fetch_chain("SPY", NOW, R, tries=3, wait=0)
    check("a transient chain fetch is retried, not recorded as a miss",
          calls == ["SPY", "SPY"] and not q[0].empty)

    calls.clear()

    def always_fail(ticker, now, r, max_days=data.MAX_DAYS):
        calls.append(ticker)
        raise RuntimeError("curl (28) timed out")

    data._fetch_chain_once = always_fail
    try:
        data.fetch_chain("SPY", NOW, R, tries=3, wait=0)
        raised = False
    except RuntimeError:
        raised = True
    check("a chain that never comes back still raises after the retries",
          raised and len(calls) == 3)
finally:
    data._fetch_chain_once = real_once

def fetch_missing(bad):
    def _f(ticker, now, r, **kw):
        if ticker in bad:
            raise RuntimeError("vendor down")
        return fake_chain(ticker, now, r)
    return _f


real_root, real_fetch, real_rate, real_caps = data.ROOT, data.fetch_chain, data.bill_rate, data.market_caps
try:
    data.bill_rate, data.market_caps = fake_rate, lambda n: pd.Series(dtype=float)
    with tempfile.TemporaryDirectory() as tmp:
        data.ROOT = Path(tmp) / "few"
        data.fetch_chain = fetch_missing({"AAPL", "MSFT"})
        check("two names missing out of thirteen does not fail the run",
              collect.main(["run", "--date", DATE]) == 0)

        data.ROOT = Path(tmp) / "many"
        data.fetch_chain = fetch_missing(set(data.UNIVERSE[:10]))
        check("ten names missing out of thirteen does fail the run",
              collect.main(["run", "--date", DATE]) == 1)
finally:
    data.ROOT, data.fetch_chain, data.bill_rate, data.market_caps = real_root, real_fetch, real_rate, real_caps

# ------------------------------------------------ sessions: open, midday and close share a day
print("\nSESSIONS\n")

OPEN_AT = pd.Timestamp("2026-09-04 15:00:00", tz="UTC")     # 11:00 New York
MIDDAY_AT = pd.Timestamp("2026-09-04 17:30:00", tz="UTC")   # 13:30 New York
AT = {"open": OPEN_AT, "midday": MIDDAY_AT, "close": NOW}
SPOT = {"open": 101.0, "midday": 102.0, "close": 103.0}


def spot_fetch(spot):
    return lambda ticker, now, r: fake_chain(ticker, now, r, spot=spot)


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


real_root, real_summary = data.ROOT, data.SUMMARY
try:
    sroot = Path(tempfile.mkdtemp())
    data.ROOT, data.SUMMARY = sroot / "raw", sroot / "summary"
    ms = {k: collect_day(DATE, AT[k], fetch=spot_fetch(SPOT[k]), rate=fake_rate, caps=fake_caps, session=k)
          for k in ("open", "midday", "close")}
    check("three sessions in one day each write a full set",
          all(m["written_this_run"] == data.UNIVERSE and not m["missing"] for m in ms.values()))
    check("each session loads its own chain (spots 101, 102, 103)",
          all(data.load_day(DATE, k)[0]["SPY"]["spot"].iloc[0] == SPOT[k] for k in SPOT))
    check("manifests say open, midday, and close for the closing run",
          [data.load_day(DATE, k)[1]["session"] for k in ("open", "midday", "close")] == ["open", "midday", "close"])
    check("close is the default reader",
          data.load_day(DATE)[0]["SPY"]["spot"].iloc[0] == 103.0 and data.load_day(DATE)[1]["session"] == "close")
    check("close keeps the original layout: the day folder holds its files, open and midday sit in subfolders",
          sorted(p.name for p in data.day_dir(DATE).iterdir() if p.is_file())
          == sorted([f"{t}.csv" for t in data.UNIVERSE] + ["rates.csv", "underlying.csv", "weights.csv", "manifest.json"])
          and sorted(p.name for p in data.day_dir(DATE).iterdir() if p.is_dir()) == ["midday", "open"])
    before = {k: digest(data.session_dir(DATE, k)) for k in SPOT}
    calls = []

    def counting_spot(ticker, now, r):
        calls.append(ticker)
        return fake_chain(ticker, now, r, spot=999.0)

    again = {k: collect_day(DATE, AT[k], fetch=counting_spot, rate=fake_rate, caps=fake_caps, session=k) for k in SPOT}
    check("a rerun of any session fetches nothing and writes nothing (write-once per session)",
          calls == [] and all(m["written_this_run"] == [] for m in again.values()))
    check("every session's files are byte-identical after the rerun",
          {k: digest(data.session_dir(DATE, k)) for k in SPOT} == before)
    check("an unknown session is refused", all(_raises(lambda: data.session_dir(DATE, bad), ValueError)
                                                  for bad in ("noon", "intraday", "")))

    # a day with only an open snapshot is not a close day: readers that loop over list_days() must not crash on it
    ODAY = "2026-09-10"
    collect_day(ODAY, OPEN_AT + pd.Timedelta(days=6), fetch=fake_chain, rate=fake_rate, caps=fake_caps, session="open")
    check("an open-only day is listed for open and not for close",
          ODAY in data.list_days("open") and ODAY not in data.list_days() and DATE in data.list_days())
    check("every default-listed day loads (no half-written close day to trip over)",
          all(data.load_day(d)[1]["session"] == "close" for d in data.list_days()))

    # the old single-file layout: what the collector wrote before sessions existed
    OLD = "2026-08-31"
    data.day_dir(OLD).mkdir()
    for t in ("SPY", "QQQ"):
        fake_chain(t, NOW, R)[0].to_csv(data.day_dir(OLD) / f"{t}.csv", index=False)
    pd.DataFrame([{"date": OLD, "series": "DTB3", "rate": R, "observation_date": OLD}]).to_csv(
        data.day_dir(OLD) / "rates.csv", index=False)
    pd.DataFrame({"market_cap": [1e12 * (i + 1) for i in range(len(data.NAMES))]}, index=data.NAMES).to_csv(
        data.day_dir(OLD) / "weights.csv")
    (data.day_dir(OLD) / "manifest.json").write_text(json.dumps(
        {"date": OLD, "snapshot_utc": "2026-08-31T21:05:00Z", "session": "close", "rate": R,
         "collected": ["QQQ", "SPY"], "written_this_run": ["SPY", "QQQ"], "skipped_existing": [],
         "missing": [], "rows": 2}))
    ochains, omanifest = data.load_day(OLD)
    check("the old single-file layout still loads by default", OLD in data.list_days() and sorted(ochains) == ["QQQ", "SPY"]
          and omanifest["session"] == "close")
    check("and still summarises (the close path never looked in a subfolder)",
          set(surface.summarise_day(OLD)[0]["ticker"]) == {"QQQ", "SPY"})
    check("an old-layout day has no open or midday",
          OLD not in data.list_days("open") and OLD not in data.list_days("midday"))

    # the command line: --session picks the folder, a late open is refused
    def fake_fetch_all(ticker, now, r, **kw):
        return fake_chain(ticker, now, r)

    real_fetch, real_rate, real_caps = data.fetch_chain, data.bill_rate, data.market_caps
    try:
        data.fetch_chain, data.bill_rate, data.market_caps = fake_fetch_all, fake_rate, lambda n: pd.Series(dtype=float)
        data.ROOT = sroot / "cli"
        rc = collect.main(["run", "--date", DATE, "--session", "midday"], now=MIDDAY_AT)
        check("run --session midday writes the midday folder and exits 0",
              rc == 0 and data.list_days("midday") == [DATE] and data.list_days() == [])
        rc = collect.main(["run", "--date", DATE], now=NOW)
        check("run without --session is the close session, as the launchd job runs it",
              rc == 0 and data.list_days() == [DATE])
        data.ROOT = sroot / "late"
        rc = collect.main(["run", "--date", DATE, "--session", "open"], now=NOW)
        check("an open run after the close is refused (a missed launchd job must not file a 5pm chain as the open)",
              rc == 1 and data.list_days("open") == [])
    finally:
        data.fetch_chain, data.bill_rate, data.market_caps = real_fetch, real_rate, real_caps
finally:
    data.ROOT, data.SUMMARY = real_root, real_summary
    shutil.rmtree(sroot, ignore_errors=True)


# FRED answers maintenance with HTTP 200 and an HTML page; the close snapshot of 2026-10-21 crashed on it before
# collecting anything. The body check must send it down the ^IRX path instead.
class _Maint:
    text = "<!doctype html>\n<html><title>Site Maintenance | St. Louis Fed</title></html>"
    def raise_for_status(self): pass
class _IRX:
    def history(self, period): return pd.DataFrame({"Close": [3.93]}, index=pd.to_datetime(["2026-10-20"]))
_get, _tk = data.requests.get, data.yf.Ticker
try:
    data.requests.get, data.yf.Ticker = (lambda *a, **k: _Maint()), (lambda t: _IRX())
    check("a FRED maintenance page falls through to ^IRX instead of crashing the collector",
          data.bill_rate("2026-10-21") == (0.0393, "2026-10-20", "IRX"))
finally:
    data.requests.get, data.yf.Ticker = _get, _tk

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
