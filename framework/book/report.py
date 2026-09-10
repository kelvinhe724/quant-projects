"""Daily desk report: the book, its risk, the Kalshi shadow desk, the data lake, the daemons.

Read-only. Reads ledger.csv, the cached bars the daemon already downloaded,
kalshi-desk/ledger.sqlite (opened read-only), data-lake/reports/status.json
and `launchctl list`. It never opens framework/.env or any key file and never
writes to a ledger. Output is reports/daily/<date>.md, reports/daily/<date>.json,
reports/latest.json (the same compact JSON at a fixed path for the morning
brief) and reports/alerts.json.

Run: ../../.venv/bin/python3 -m framework.book.report
"""
import glob
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

from framework.book import universe
from framework.book.allocate import LIVE_BOOK, REPORTS, load_allocations
from framework.book.broker import (KILL, RECON_TOL, Limits, ShadowBroker, discrepancies, expected_weights,
                                   filled_value, unfilled)
from framework.book.daemon import ET, HISTORY_YEARS, LEDGER, LIVE_START, last_session, previous_row, read_ledger
from framework.book.strategies import CAPITAL, book_config, sleeves
from framework.engine.data import CACHE, SUBSTITUTED

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CT = "America/Chicago"
DAILY = os.path.join(REPORTS, "daily")
LATEST = os.path.join(REPORTS, "latest.json")
ALERTS = os.path.join(REPORTS, "alerts.json")
KALSHI_LEDGER = os.path.join(ROOT, "kalshi-desk", "ledger.sqlite")
LAKE_STATUS = os.path.join(ROOT, "data-lake", "reports", "status.json")
DAEMON_LOG = os.path.join(REPORTS, "daemon.log")
RISK = book_config().risk  # the overlay's own limits: 3x gross, half at 15% dd, kill at 25%
LIMITS = Limits()          # the broker layer's hard limits: gross, per name, daily loss
JOBS = ("com.kelvinhe.premia-book", "com.kelvinhe.desk-report", "com.kelvinhe.kalshi-shadow",
        "com.kelvinhe.lake-daily", "com.kelvinhe.lake-hourly", "com.kelvinhe.lake-crypto",
        "com.kelvinhe.options-collector")
LAKE_STATUS_MAX_AGE_H = 3   # quality.py runs hourly
VAR_DAYS = 250
Z = {"95": 1.6449, "99": 2.3263}


def f(v, spec="+.2%"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


# ---------------------------------------------------------------- the book

def replay(ledger):
    """Re-run the shadow book from LIVE_START on the bars the daemon cached for its last session.

    Returns (Results, None) or (None, reason). Only cached files are touched;
    if any is missing the replay is skipped rather than downloaded.
    """
    shadow = ledger[ledger.book == "shadow"]
    if shadow.empty:
        return None, "no shadow rows in the ledger"
    session = pd.Timestamp(shadow.date.max())
    start = str((session - pd.DateOffset(years=HISTORY_YEARS)).date())
    end = str((session + pd.Timedelta(days=1)).date())
    missing = [t for t in universe.TICKERS if not os.path.exists(os.path.join(CACHE, "yf", f"{t}_{start}_{end}.csv"))]
    missing += [c for c in universe.CRYPTO
                if not os.path.exists(os.path.join(CACHE, "alpaca", f"{c.replace('/', '-')}_daily_{end}.csv"))]
    if missing:
        return None, f"bars not cached for {session.date()}: {', '.join(missing)}"
    try:
        bars = universe.load_bars(start, end)
        live = {k: v for k, v in load_allocations()["weights"].items() if v > 0}
        broker = ShadowBroker([s for s in sleeves(LIVE_BOOK) if str(s) in live], live, LIVE_START)
        return broker.run(bars), None
    except Exception as e:  # a report must come out even if the engine cannot
        return None, f"replay failed: {type(e).__name__}: {e}"


def drawdown(equity):
    return float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1) if len(equity) else 0.0


def book_summary(ledger, results=None, note=None):
    """Positions, P&L and cap usage. Per-sleeve numbers need the replay; the rest come from the ledger."""
    shadow = ledger[ledger.book == "shadow"].sort_values("date")
    out = {"book": LIVE_BOOK, "capital": CAPITAL, "session": None, "equity": None, "sleeves": {}, "note": note}
    if shadow.empty:
        return out
    last = shadow.iloc[-1]
    out["session"] = pd.Timestamp(last.date).strftime("%Y-%m-%d")
    out["equity"] = float(last.equity)
    out["positions"] = json.loads(last.positions)
    out["targets"] = json.loads(last.targets)
    out["gross"] = float(last.gross)
    out["run_at"] = str(last.run_at)
    out["panel_hash"] = str(last.panel_hash)
    equity = shadow.set_index("date").equity.astype(float)
    if results is not None:
        equity = results.equity
        costs = results.trades.groupby(["date", "strategy"])[["commission", "slippage"]].sum().sum(axis=1) \
            if len(results.trades) else pd.Series(dtype=float)
        for k, b in results.books.items():
            eq = b["equity"]
            net = float(eq.iloc[-1] - eq.iloc[-2]) if len(eq) > 1 else 0.0
            cost = float(costs.get((eq.index[-1], k), 0.0))
            gross_w = float(b["weights"].iloc[-1].abs().sum())
            dd = drawdown(eq)
            out["sleeves"][k] = {"equity": float(eq.iloc[-1]), "pnl_net": net, "pnl_gross": net + cost, "cost": cost,
                                 "drawdown": dd, "gross": gross_w, "gross_cap_used": gross_w / RISK.max_gross,
                                 "dd_half_used": -dd / RISK.dd_threshold, "killed": dd <= -RISK.kill_dd,
                                 "halved": dd <= -RISK.dd_threshold}
    out["equity_path"] = {d.strftime("%Y-%m-%d"): round(float(v), 2) for d, v in equity.items()}
    out["pnl_day"] = float(equity.iloc[-1] - equity.iloc[-2]) if len(equity) > 1 else 0.0
    out["pnl_day_pct"] = out["pnl_day"] / float(equity.iloc[-2]) if len(equity) > 1 else 0.0
    out["pnl_since_start"] = float(equity.iloc[-1] - CAPITAL)
    out["pnl_since_start_pct"] = out["pnl_since_start"] / CAPITAL
    out["drawdown"] = drawdown(equity)
    out["gross_cap_used"] = out["gross"] / RISK.max_gross
    out["daily_loss_used"] = max(-out["pnl_day_pct"], 0.0) / LIMITS.max_daily_loss
    out["max_name"] = max((abs(v) for v in out["positions"].values()), default=0.0)
    out["name_cap_used"] = out["max_name"] / LIMITS.max_per_name
    alpaca = ledger[(ledger.book == "alpaca") & (ledger.date == last.date)]
    if len(alpaca):
        a = alpaca.iloc[-1]
        out["alpaca"] = {"equity": float(a.equity), "positions": json.loads(a.positions), "note": str(a.note)}
    return out


# ---------------------------------------------------------------- risk

def cached_returns(instruments, days=VAR_DAYS):
    """Daily close-to-close returns from the newest cached bar file per instrument. No download."""
    cols = {}
    for name in instruments:
        if name in universe.CRYPTO:
            files = glob.glob(os.path.join(CACHE, "alpaca", f"{name.replace('/', '-')}_daily_*.csv"))
        else:
            files = glob.glob(os.path.join(CACHE, "yf", f"{name}_*.csv"))
        if not files:
            continue
        newest = max(files, key=lambda p: os.path.basename(p).rsplit("_", 1)[-1])
        px = pd.read_csv(newest, index_col=0, parse_dates=True)["close"].astype(float)
        cols[name] = px.pct_change()
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="all").tail(days).fillna(0.0)


def ledoit_wolf_var(positions, returns, equity):
    """One-day parametric VaR of the held weights under a Ledoit-Wolf covariance.

    Gaussian, so it understates tails; the honest use is as a scale, not a
    bound. `positions` are weights of equity; VaR comes back as a fraction of
    equity and in dollars, with the realised 20-day vol of the same weights
    beside it.
    """
    names = [n for n in positions if n in returns.columns and abs(positions[n]) > 1e-9]
    if not names or len(returns) < 20:
        return {"method": "ledoit-wolf", "n_days": int(len(returns)), "sigma_1d": 0.0,
                "var": {k: 0.0 for k in Z}, "var_usd": {k: 0.0 for k in Z}, "realised_vol_20d": 0.0, "instruments": 0}
    from sklearn.covariance import LedoitWolf
    R = returns[names]
    w = np.array([positions[n] for n in names])
    S = LedoitWolf().fit(R.to_numpy()).covariance_
    sigma = float(np.sqrt(w @ S @ w))
    port = R.to_numpy() @ w
    return {"method": "ledoit-wolf", "n_days": int(len(R)), "instruments": len(names), "sigma_1d": sigma,
            "var": {k: z * sigma for k, z in Z.items()}, "var_usd": {k: z * sigma * equity for k, z in Z.items()},
            "realised_vol_20d": float(np.std(port[-20:], ddof=1) * np.sqrt(252)),
            "annual_vol": sigma * np.sqrt(252), "target_vol": RISK.target_vol}


# ---------------------------------------------------------------- kalshi

def kalshi_summary(path=KALSHI_LEDGER):
    """Per-book stats from kalshi-desk's own paper.stats, on a read-only connection."""
    if not os.path.exists(path):
        return {"error": f"no ledger at {path}"}
    try:
        sys.path.insert(0, os.path.join(ROOT, "kalshi-desk"))
        import paper
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        out = {"books": {}}
        for b in paper.BOOKS:
            s = paper.stats(db, b)
            n_fills = db.execute("select count(*) from fills where book = ?", (b,)).fetchone()[0]
            if not n_fills:
                continue
            open_n = db.execute("select count(*) from fills f left join settlements s on s.ticker = f.ticker "
                                "where s.ticker is null and f.book = ?", (b,)).fetchone()[0]
            out["books"][b] = {**{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in s.items()},
                               "fills": int(n_fills), "open": int(open_n), "at_risk": paper.exposure(db, b),
                               "last_fill": db.execute("select max(ts) from fills where book = ?", (b,)).fetchone()[0]}
        out["last_mark"] = db.execute("select max(ts) from marks").fetchone()[0]
        out["backtest"] = paper.BACKTEST
        db.close()
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- data lake and daemons

def lake_summary(path=LAKE_STATUS, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    if not os.path.exists(path):
        return {"ok": False, "summary": "status.json missing", "stale": [], "age_hours": None}
    with open(path) as fh:
        st = json.load(fh)
    age = (now - pd.Timestamp(st["generated_at"]).tz_convert("UTC")).total_seconds() / 3600
    stale = [k for k, v in st.get("datasets", {}).items() if v.get("stale")]
    dupes = [k for k, v in st.get("datasets", {}).items() if v.get("dupes")]
    return {"ok": bool(st.get("ok")), "summary": st.get("summary", ""), "stale": stale, "dupes": dupes,
            "age_hours": age, "generated_at": st["generated_at"],
            "last": {k: v.get("last") for k, v in st.get("datasets", {}).items()}}


def launchd_jobs(text=None):
    """{label: {pid, status}} for our jobs from `launchctl list` (or the given text)."""
    if text is None:
        try:
            text = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=10).stdout
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    jobs = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2] in JOBS:
            pid, status, label = parts
            jobs[label] = {"pid": None if pid == "-" else int(pid), "status": int(status)}
    return {j: jobs.get(j, {"pid": None, "status": None, "loaded": False}) for j in JOBS}


def daemon_health(ledger, now=None, launchctl_text=None, log_path=DAEMON_LOG):
    now = pd.Timestamp.now(tz=ET) if now is None else pd.Timestamp(now).tz_convert(ET)
    shadow = ledger[ledger.book == "shadow"]
    expected = last_session(now)
    have = set(shadow.date.astype(str).str[:10])
    # the daemon fires 16:45 ET; give it until 19:00 ET before calling the session missed
    due = now >= expected.tz_localize(ET) + pd.Timedelta(hours=19)
    return {"jobs": launchd_jobs(launchctl_text),
            "last_run_at": str(ledger.run_at.max()) if len(ledger) else None,
            "last_session_logged": max(have) if have else None,
            "expected_session": expected.strftime("%Y-%m-%d"),
            "missed": due and expected.strftime("%Y-%m-%d") not in have,
            "log_mtime": pd.Timestamp(os.path.getmtime(log_path), unit="s", tz="UTC").tz_convert(ET).isoformat(timespec="seconds")
            if os.path.exists(log_path) else None,
            "substituted_bars": substituted_bars(log_path)}


def substituted_bars(log_path=DAEMON_LOG, lines=400):
    """Cached-bar substitutions the daemon logged in its most recent output."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, errors="replace") as fh:
        tail = fh.readlines()[-lines:]
    return [l.split(SUBSTITUTED + ": ", 1)[1].strip() for l in tail if SUBSTITUTED + ": " in l]


def kill_status(path=KILL):
    """Whether the KILL file exists and the last reason written into it."""
    if not os.path.exists(path):
        return {"present": False, "reason": None}
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    return {"present": True, "reason": lines[-1] if lines else "(empty file)", "path": path}


def reconciliation(ledger):
    """The broker layer's own check, rerun on the ledger: the account's last row against what the
    row before it said it should hold, plus what the broker's own order records say actually
    filled (broker.expected_weights, RECON_TOL), the unfilled orders, and the panel hash."""
    shadow = ledger[ledger.book == "shadow"].sort_values("date")
    out = {"mismatches": [], "unfilled": [], "alpaca_rows": False, "panel_hash_changed": False, "tol": RECON_TOL}
    if len(shadow) > 1:
        h = shadow.panel_hash.astype(str).tolist()
        out["panel_hash_changed"] = h[-1] != h[-2]
        out["panel_hash"] = (h[-2], h[-1])
    alp = ledger[ledger.book == "alpaca"].sort_values("date")
    if alp.empty:
        return out
    last = alp.iloc[-1]
    out["alpaca_rows"] = True
    out["mode"] = str(last.get("mode") or "")
    status = last.get("order_status")
    outcomes = json.loads(status) if isinstance(status, str) and status.strip() else []
    out["unfilled"] = unfilled(outcomes)
    prev = previous_row(ledger, pd.Timestamp(last.date))
    if prev is None:
        return out
    out["against"] = str(prev["date"])[:10]
    was = str(prev.get("mode") or "")
    if was != out["mode"]:
        out["mode_changed"] = (was, out["mode"])  # two accounts: their positions are not comparable
        return out
    off = discrepancies(json.loads(last.positions),
                        expected_weights(prev, filled_value(outcomes) if outcomes else None, last.equity))
    out["mismatches"] = [{"instrument": n, "held": h, "expected": e} for n, (h, e) in off.items()]
    return out


# ---------------------------------------------------------------- alerts and output

def build_alerts(book, lake, health, recon, kalshi, now=None, kill=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    a = []
    if kill and kill["present"]:
        a.append({"kind": "kill", "msg": f"KILL file present, trading halted: {kill['reason']}"})
    if book.get("pnl_day_pct") is not None and book["pnl_day_pct"] <= -LIMITS.max_daily_loss:
        a.append({"kind": "drawdown", "msg": f"shadow book lost {book['pnl_day_pct']:.2%} on the day, past the {LIMITS.max_daily_loss:.0%} daily loss limit"})
    if book.get("max_name", 0.0) > LIMITS.max_per_name + 1e-9:
        a.append({"kind": "drawdown", "msg": f"a single name at {book['max_name']:.0%} of equity, over the {LIMITS.max_per_name:.0%} per-name limit"})
    dd = book.get("drawdown")
    if dd is not None and dd <= -RISK.dd_threshold:
        a.append({"kind": "drawdown", "msg": f"book drawdown {dd:.1%} past the {RISK.dd_threshold:.0%} half-size line"})
    for k, s in book.get("sleeves", {}).items():
        if s["killed"]:
            a.append({"kind": "drawdown", "msg": f"{k} drawdown {s['drawdown']:.1%}: past the {RISK.kill_dd:.0%} kill, sleeve is flat"})
        elif s["halved"]:
            a.append({"kind": "drawdown", "msg": f"{k} drawdown {s['drawdown']:.1%}: past {RISK.dd_threshold:.0%}, running at half size"})
        if s["gross"] > RISK.max_gross * 1.001:
            a.append({"kind": "drawdown", "msg": f"{k} gross {s['gross']:.2f}x over the {RISK.max_gross:.0f}x cap"})
    for msg in health.get("substituted_bars", []):
        a.append({"kind": "feed", "msg": "premia-book traded on cached bars: " + msg})
    if health.get("missed"):
        a.append({"kind": "daemon", "msg": f"premia-book: session {health['expected_session']} not in the ledger "
                                          f"(last {health['last_session_logged']})"})
    for label, j in health.get("jobs", {}).items():
        if j.get("loaded") is False:
            a.append({"kind": "daemon", "msg": f"{label} is not loaded in launchd"})
        elif j.get("status"):
            a.append({"kind": "daemon", "msg": f"{label} last exit status {j['status']}"})
    if kalshi.get("last_mark"):
        age = (now - pd.Timestamp(kalshi["last_mark"]).tz_convert("UTC")).total_seconds() / 3600
        if age > 6:
            a.append({"kind": "daemon", "msg": f"kalshi shadow: no mark for {age:.0f}h"})
    elif "error" in kalshi:
        a.append({"kind": "daemon", "msg": f"kalshi ledger unreadable: {kalshi['error']}"})
    if not lake.get("ok"):
        a.append({"kind": "feed", "msg": "data lake: " + (", ".join(lake["stale"]) + " stale" if lake.get("stale")
                                                     else lake.get("summary", "not ok"))})
    if lake.get("age_hours") is None or lake["age_hours"] > LAKE_STATUS_MAX_AGE_H:
        a.append({"kind": "feed", "msg": f"data lake: status.json is {f(lake.get('age_hours'), '.0f')}h old, quality job not running"})
    for o in recon.get("unfilled", []):
        asked = f"{o['asked_qty']}" if o.get("asked_qty") else f"${o.get('asked_notional')}"
        a.append({"kind": "reconcile", "msg": f"order {o['status']}: {o['side']} {o['instrument']} filled "
                                              f"{o['filled_qty']} of {asked} (broker id {o['id']})"})
    if recon.get("mode_changed"):
        a.append({"kind": "reconcile", "msg": "broker mode changed {} -> {}: positions were not reconciled "
                                              "across the two accounts".format(*recon["mode_changed"])})
    if recon.get("mismatches"):
        names = ", ".join(m["instrument"] for m in recon["mismatches"])
        a.append({"kind": "reconcile", "msg": f"account off the ledger by more than {RECON_TOL:.0%} of equity: {names}"})
    if recon.get("panel_hash_changed"):
        a.append({"kind": "reconcile", "msg": f"panel hash moved {recon['panel_hash'][0]} -> {recon['panel_hash'][1]}: history rewritten, or the hash window changed"})
    return a


def headline(book, risk, kalshi, lake, alerts):
    """Three lines for the morning brief."""
    if book.get("equity") is None:
        pnl = "book: no ledger rows yet"
    else:
        pnl = (f"Book {book['session']}: {book['equity']:,.0f} ({book['pnl_day']:+,.0f} day, "
               f"{book['pnl_since_start_pct']:+.2%} since {LIVE_START}) · DD {book['drawdown']:.2%} · gross {book['gross']:.2f}x of {RISK.max_gross:.0f}x")
        sl = book.get("sleeves")
        if sl:
            pnl += " · " + " / ".join(f"{k} {v['pnl_net']:+,.0f}" for k, v in sl.items())
    rk = f"VaR95 1d {risk['var']['95']:.2%} ({risk['var_usd']['95']:,.0f}), vol {f(risk.get('annual_vol'), '.1%')} vs {RISK.target_vol:.0%} target"
    kb = kalshi.get("books", {}).get("shadow")
    if kb:
        rk += (f" · Kalshi shadow {kb['n']} settled, P&L {kb['pnl_net']:+.2f} on {kb['staked']:.0f} staked, "
               f"{kb['open']} open ({kb['at_risk']:.0f} at risk)")
    rk += f" · lake {'ok' if lake.get('ok') else 'NOT ok'}"
    al = "ALERTS: " + " | ".join(x["msg"] for x in alerts) if alerts else "no alerts, kill switch armed"
    return [pnl, rk, al]


def render_md(r):
    b, risk, k, lake, h, recon = r["book"], r["risk"], r["kalshi"], r["lake"], r["daemons"], r["reconciliation"]
    L = [f"# Desk report {r['date']}", "", f"Generated {r['generated_at']}. Live book `{b['book']}` from {LIVE_START}, "
         f"{b['capital']:,.0f} starting capital, shadow fills at the shadow cost model.", ""]
    L += ["## Headline", ""] + [f"- {x}" for x in r["headline"]] + [""]
    L += ["## Book", ""]
    if b.get("equity") is None:
        L += ["No ledger rows yet.", ""]
    else:
        L += [f"- session {b['session']}, run {b['run_at']}, panel {b['panel_hash']}",
              f"- equity {b['equity']:,.2f}; day {b['pnl_day']:+,.2f} ({b['pnl_day_pct']:+.2%}); since start {b['pnl_since_start']:+,.2f} ({b['pnl_since_start_pct']:+.2%})",
              f"- drawdown {b['drawdown']:.2%} (half size at {RISK.dd_threshold:.0%}, kill at {RISK.kill_dd:.0%})",
              f"- gross {b['gross']:.2f}x, {b['gross_cap_used']:.0%} of the {RISK.max_gross:.0f}x cap; largest name {b['max_name']:.1%}, "
              f"{b['name_cap_used']:.0%} of the {LIMITS.max_per_name:.0%} per-name limit; day loss {b['daily_loss_used']:.0%} of the {LIMITS.max_daily_loss:.0%} daily limit", ""]
        if b.get("note"):
            L += [f"Sleeve attribution unavailable: {b['note']}.", ""]
        if b["sleeves"]:
            L += ["| sleeve | equity | P&L gross | P&L net | cost | drawdown | gross | of 3x cap | of 15% dd |", "|---|---|---|---|---|---|---|---|---|"]
            for n, s in b["sleeves"].items():
                L.append(f"| {n} | {s['equity']:,.0f} | {s['pnl_gross']:+,.2f} | {s['pnl_net']:+,.2f} | {s['cost']:.2f} | "
                         f"{s['drawdown']:.2%} | {s['gross']:.2f}x | {s['gross_cap_used']:.0%} | {s['dd_half_used']:.0%} |")
            L.append("")
        L += ["| instrument | held | target | alpaca |", "|---|---|---|---|"]
        alp = b.get("alpaca", {}).get("positions", {})
        for n in sorted(set(b["positions"]) | set(b["targets"]) | set(alp)):
            L.append(f"| {n} | {b['positions'].get(n, 0.0):+.2%} | {b['targets'].get(n, 0.0):+.2%} | "
                     f"{f(alp[n]) if n in alp else ('n/a' if not alp else '0.00%')} |")
        L.append("")
        L.append(f"Alpaca: {b['alpaca']['equity']:,.2f}, {b['alpaca']['note']}" if b.get("alpaca") else "Alpaca: no rows (no paper keys).")
        L.append("")
    L += ["## Risk", "",
          f"- one-day VaR, {risk['method']} covariance on {risk['n_days']} days, {risk.get('instruments', 0)} instruments: "
          f"95% {risk['var']['95']:.2%} ({risk['var_usd']['95']:,.0f}), 99% {risk['var']['99']:.2%} ({risk['var_usd']['99']:,.0f})",
          f"- model vol {f(risk.get('annual_vol'), '.1%')} annualised, realised 20d {risk['realised_vol_20d']:.1%}, target {RISK.target_vol:.0%} per sleeve",
          "- Gaussian, so a scale not a bound; the overlay's own limits are the drawdown lines above.", ""]
    L += ["## Kalshi shadow desk", ""]
    if "error" in k:
        L += [f"unavailable: {k['error']}", ""]
    else:
        for n, s in k["books"].items():
            L.append(f"- {n}: {s['fills']} fills, {s['n']} settled, {s['open']} open ({s['at_risk']:.2f} at risk); "
                     f"P&L {s['pnl']:+.2f} gross / {s['pnl_net']:+.2f} net on {s['staked']:.2f} staked; "
                     f"mean payoff {f(s['mean_payoff'], '+.4f')} (t {f(s['t'], '+.2f')}), Brier {f(s['brier'], '.4f')}; last fill {s['last_fill']}")
        bt = k["backtest"]
        L += [f"- backtest, {bt.get('scenario')}: mean payoff {bt['mean_payoff']:+.4f} (t {bt.get('t', float('nan')):+.2f}) on {bt['n']} bets; last mark {k['last_mark']}", ""]
    L += ["## Data lake", "", f"- {lake['summary']}; status.json {f(lake.get('age_hours'), '.1f')}h old"
          + (f"; stale: {', '.join(lake['stale'])}" if lake.get("stale") else "") + (f"; dupes: {', '.join(lake['dupes'])}" if lake.get("dupes") else "")]
    for n, last in (lake.get("last") or {}).items():
        L.append(f"  - {n}: last {last}")
    L += ["", "## Daemons", "", f"- ledger last run {h['last_run_at']}, last session logged {h['last_session_logged']}, expected {h['expected_session']}"
          + (" (MISSED)" if h["missed"] else ""), f"- daemon.log last written {h['log_mtime']}"]
    for label, j in h["jobs"].items():
        st = "not loaded" if j.get("loaded") is False else f"pid {j['pid']}, last exit {j['status']}" if j.get("pid") else f"idle, last exit {j['status']}"
        L.append(f"  - {label}: {st}")
    k_ = r["kill"]
    L += ["", f"- kill switch: " + (f"PRESENT at {k_['path']}: {k_['reason']}" if k_["present"] else f"armed, no file at {KILL}")]
    L += [f"- reconciliation: " + (f"{len(recon['mismatches'])} name(s) off the ledger by more than {RECON_TOL:.0%} against {recon['against']}" if recon["mismatches"]
                                   else f"{recon.get('mode', '')} account matches the ledger row of {recon['against']}" if recon.get("against")
                                   else "first account row, nothing to reconcile against" if recon["alpaca_rows"] else "no account rows to reconcile")
          + (f"; panel hash moved {recon['panel_hash'][0]} -> {recon['panel_hash'][1]}" if recon.get("panel_hash_changed") else "")]
    L += ["", "## Alerts", ""] + ([f"- [{x['kind']}] {x['msg']}" for x in r["alerts"]] or ["none"]) + [""]
    return "\n".join(L)


def build(now=None, ledger_path=LEDGER, kalshi_path=KALSHI_LEDGER, lake_path=LAKE_STATUS, launchctl_text=None,
          do_replay=True, kill_path=KILL):
    now = pd.Timestamp.now(tz=CT) if now is None else pd.Timestamp(now).tz_convert(CT)
    ledger = read_ledger(ledger_path)
    results, note = replay(ledger) if do_replay else (None, "replay off")
    book = book_summary(ledger, results, note)
    positions = book.get("positions", {})
    risk = ledoit_wolf_var(positions, cached_returns(list(positions)), book.get("equity") or CAPITAL)
    kalshi = kalshi_summary(kalshi_path)
    lake = lake_summary(lake_path, now)
    health = daemon_health(ledger, now, launchctl_text)
    recon = reconciliation(ledger)
    kill = kill_status(kill_path)
    alerts = build_alerts(book, lake, health, recon, kalshi, now, kill)
    r = {"date": now.strftime("%Y-%m-%d"), "generated_at": now.isoformat(timespec="seconds"),
         "headline": headline(book, risk, kalshi, lake, alerts), "alerts": alerts, "book": book, "risk": risk,
         "kalshi": kalshi, "lake": lake, "daemons": health, "reconciliation": recon, "kill": kill}
    return r


def compact(r):
    """What the brief reads: headline, alerts, and the few numbers behind them."""
    b = r["book"]
    return {"date": r["date"], "generated_at": r["generated_at"], "headline": r["headline"], "alerts": r["alerts"],
            "session": b.get("session"), "equity": b.get("equity"), "pnl_day": b.get("pnl_day"),
            "pnl_since_start_pct": b.get("pnl_since_start_pct"), "drawdown": b.get("drawdown"), "gross": b.get("gross"),
            "sleeves": {k: {"pnl_net": v["pnl_net"], "pnl_gross": v["pnl_gross"], "drawdown": v["drawdown"]} for k, v in b.get("sleeves", {}).items()},
            "var95_1d": r["risk"]["var"]["95"], "kalshi_shadow": r["kalshi"].get("books", {}).get("shadow"),
            "lake_ok": r["lake"].get("ok"), "daemon_missed": r["daemons"].get("missed"), "kill_present": r["kill"]["present"]}


def write(r, reports=REPORTS):
    daily = os.path.join(reports, "daily")
    os.makedirs(daily, exist_ok=True)
    md = os.path.join(daily, f"{r['date']}.md")
    with open(md, "w") as fh:
        fh.write(render_md(r))
    c = compact(r)
    for p in (os.path.join(daily, f"{r['date']}.json"), os.path.join(reports, "latest.json")):
        with open(p, "w") as fh:
            json.dump(c, fh, indent=1, default=str)
    with open(os.path.join(reports, "alerts.json"), "w") as fh:
        json.dump({"date": r["date"], "generated_at": r["generated_at"], "alerts": r["alerts"]}, fh, indent=1)
    return md


if __name__ == "__main__":
    report = build()
    path = write(report)
    print("\n".join(report["headline"]))
    print(f"-> {path}")
