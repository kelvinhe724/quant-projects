"""Offline checks on a planted panel: 40 names, 900 sessions, an earnings event every quarter.

Each event has a true sentiment s + 0.8u. The reaction-day return carries s
only; the ten sessions after it carry s + u in their intraday legs, so the
text knows something the price reaction does not. The timing rule, the
causal ranks, the feature audit, the alpha, the event study and the
research path must all see that, and a shuffled score must not. Exits 1
on any failure. No model download: the scorer here is a word lexicon.

Run: ../.venv/bin/python3 check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.engine import sharpe
from research.alpha import backtest, positions
from research.features import Feature, FeatureStore, PeekError, Raw
from research.features.store import cross_section

from events import (FEATURES, EventDrift, causal_ranks, chunks, event_features, event_table, forward_returns,
                    ic_table, reaction_session)

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


rng = np.random.default_rng(3)
N, T, H = 40, 900, 10
CAL = pd.bdate_range("2019-01-01", periods=T)
NAMES = [f"S{i:02d}" for i in range(N)]

# events: one a quarter per name, accepted 07:00 or 17:00 ET on a session
rows = []
for j, name in enumerate(NAMES):
    for k in range(40, T - 30, 63):
        day = CAL[k + rng.integers(0, 10)]
        hour = rng.choice([7, 17])
        s, u = rng.normal(), rng.normal()
        rows.append({"ticker": name, "accepted": day.tz_localize("America/New_York") + pd.Timedelta(hours=hour),
                     "s": s, "u": u, "true": s + 0.8 * u})
EV = pd.DataFrame(rows)
EV["date"] = reaction_session(EV["accepted"], CAL)

over = rng.normal(0, 0.006, (T, N))
intra = rng.normal(0, 0.010, (T, N))
for _, e in EV.iterrows():
    i, c = CAL.get_loc(e["date"]), NAMES.index(e["ticker"])
    over[i, c] += 0.02 * e["s"]
    intra[i + 1:i + 1 + H, c] += 0.0015 * (e["s"] + e["u"])
close = np.zeros((T, N))
opn = np.zeros((T, N))
prev = np.full(N, 100.0)
for t in range(T):
    opn[t] = prev * np.exp(over[t])
    close[t] = opn[t] * np.exp(intra[t])
    prev = close[t]
OPEN = pd.DataFrame(opn, index=CAL, columns=NAMES)
CLOSE = pd.DataFrame(close, index=CAL, columns=NAMES)
SPY = pd.Series(100.0, index=CAL)


def lexicon_score(texts):
    out = []
    for t in texts:
        w = t.split()
        g, b = w.count("good"), w.count("bad")
        out.append((g - b) / max(g + b, 1))
    return np.array(out)


def text_for(v):
    g = int(np.clip(50 + 20 * v, 0, 100))
    return " ".join(["good"] * g + ["bad"] * (100 - g))


EV["text"] = EV["true"].map(text_for)
EV["sent"] = lexicon_score(EV["text"])
# a second filing on the same reaction session for one name: the first in filing order is the event
dup = EV.iloc[[5]].assign(sent=-0.99, text="bad")
EV = pd.concat([EV.iloc[:6], dup, EV.iloc[6:]], ignore_index=True)

print("TIMING\n")
cal = CAL
sat = cal[100] + pd.Timedelta(days=5 - cal[100].weekday())
holiday = cal[200]
cal_h = cal.drop(holiday)
et = lambda d, hm: (d.tz_localize("America/New_York") + pd.Timedelta(hm)).tz_convert("UTC")
got = reaction_session([et(cal[50], "8h59m"), et(cal[50], "9h"), et(cal[50], "20h30m"), et(sat, "8h"),
                        et(holiday, "8h"), et(cal[-1], "17h")], cal_h)
check("accepted 08:59 ET reacts the same session", got[0] == cal[50])
check("accepted 09:00 ET reacts the next session", got[1] == cal[51])
check("accepted 20:30 ET reacts the next session", got[2] == cal[51])
check("accepted Saturday morning reacts Monday", got[3] == cal[100] + pd.Timedelta(days=7 - cal[100].weekday()))
check("accepted on a holiday morning reacts the next session", got[4] == cal[201])
check("accepted after the last close is NaT", pd.isna(got[5]))

tok = lambda text, **k: {"input_ids": list(range(len(text.split()))), }
check("chunks caps a long document at 4 windows of 510 tokens",
      [len(c) for c in chunks(" ".join(["w"] * 3000), tok)] == [510] * 4)
check("chunks keeps a short document whole", [len(c) for c in chunks(" ".join(["w"] * 100), tok)] == [100])
check("lexicon scorer orders a planted positive above a planted negative",
      lexicon_score([text_for(2.0)])[0] > lexicon_score([text_for(-2.0)])[0])

print("\nCAUSAL RANKS\n")
raw = Raw({"open": OPEN, "close": CLOSE}, filings=EV[["date", "ticker", "text", "sent"]])
ev = event_table(raw)
rk = causal_ranks(ev, CAL)
check("two filings on one reaction session keep the first in filing order",
      len(ev) == len(EV) - 1 and (ev["sent"] != -0.99).all())
cut = CAL[500]
later = ev.copy()
later.loc[later["date"] > cut, "sent"] = 99.0
later.loc[later["date"] > cut, "react"] = -99.0
rk2 = causal_ranks(later, CAL)
early = ev["date"] <= cut
check("ranks of events up to a date are byte-identical when every later event is changed",
      rk[early].equals(rk2[early]))
flat = rk.stack().dropna()
check("percentiles live in [0, 1] and centre near 0.5",
      flat.between(0, 1).all() and abs(flat.mean() - 0.5) < 0.05, f"min {flat.min():.3f} max {flat.max():.3f} mean {flat.mean():.3f}")
c_sr = rk[["sent_pct", "react_pct"]].dropna().corr().iloc[0, 1]
c_rr = rk[["resid_pct", "react_pct"]].dropna().corr().iloc[0, 1]
check("sentiment rank is correlated with the reaction rank on the planted panel", c_sr > 0.3, f"{c_sr:.2f}")
check("residual rank is orthogonal to the reaction rank", abs(c_rr) < 0.1, f"{c_rr:.2f}")
check("ranks are NaN until 50 events have accrued", rk["sent_pct"].iloc[:20].isna().all() and rk["sent_pct"].notna().sum() > 400)

print("\nFEATURE STORE\n")
store = FeatureStore(FEATURES)
try:
    panel = store.build(raw, audit=True)
    check("event features pass the peek audit at lag 0", True)
except PeekError as e:
    panel = store.build(raw)
    check("event features pass the peek audit at lag 0", False, str(e))
peek = Feature("react_peek", lambda r: event_features(Raw({"close": r.frames["close"].shift(-1)},
                                                         filings=r.filings))["react_pct"], 0)
try:
    FeatureStore([peek]).build(raw, audit=True)
    check("a reaction computed on tomorrow's close raises PeekError", False)
except PeekError:
    check("a reaction computed on tomorrow's close raises PeekError", True)
e0 = ev.iloc[100]
i0 = CAL.get_loc(e0["date"])
age = panel["age"][e0["ticker"]]
check("age is 0 on the reaction session and 3 three sessions later",
      age.iloc[i0] == 0 and age.iloc[i0 + 3] == 3)
first = ev[ev["ticker"] == e0["ticker"]]["date"].min()
check("age is NaN before a name's first event", panel["age"][e0["ticker"]].loc[:first].iloc[:-1].isna().all())
feats = event_features(raw)
check("reaction feature is close(R) / close(R - 1) - 1 ranked, carried to R + 3",
      feats["react_pct"][e0["ticker"]].iloc[i0 + 3] == rk.loc[100, "react_pct"])

print("\nALPHA\n")
xs = cross_section(panel, CAL[612])
a = EventDrift("sent", H)
w = a.signal(xs)
active = xs["sent_pct"].where(xs["age"] < H).dropna()
lq, sq = (active >= 0.8).sum(), (active <= 0.2).sum()
check("each leg sums to 1 when five names qualify and is empty otherwise",
      (abs(w[w > 0].sum() - 1) < 1e-12 if lq >= 5 else (w > 0).sum() == 0)
      and (abs(w[w < 0].sum() + 1) < 1e-12 if sq >= 5 else (w < 0).sum() == 0),
      f"{(w > 0).sum()} long of {lq} qualifying, {(w < 0).sum()} short of {sq}, {len(active)} active")
check("every long is in the top fifth and every short in the bottom fifth of active names",
      (active[w[w > 0].index] >= 0.8).all() and (active[w[w < 0].index] <= 0.2).all())
stale = xs.copy()
stale["age"] = H
check("nothing is held once every event is past the horizon", (a.signal(stale) == 0).all())
few = xs.copy()
few.loc[few.index[10:], "age"] = np.nan
check("a leg with fewer than five names is not traded", (EventDrift("sent", H).signal(few) == 0).all())

print("\nHARNESS PRICE MAPPING\n")
p = OPEN.shift(-1)
pos = pd.DataFrame(0.0, index=CAL[:5], columns=NAMES)
pos.loc[CAL[2], "S00"] = 1.0
r = backtest(pos, p.iloc[:6], cost_bps=0.0)
check("a weight set on row t earns open(t + 2) / open(t + 1) - 1 on row t + 1",
      abs(r.loc[CAL[3]] - (OPEN.loc[CAL[4], "S00"] / OPEN.loc[CAL[3], "S00"] - 1)) < 1e-12 and r.loc[CAL[2]] == 0)

print("\nEVENT STUDY\n")
fwd = forward_returns(ev, OPEN, SPY, range(1, 21))
ic = ic_table(ev, fwd).set_index(["signal", "horizon"])
s10, r10, x10 = ic.loc[("sent", 10)], ic.loc[("react", 10)], ic.loc[("resid", 10)]
check("sentiment IC at 10 sessions is large and significant", s10["ic"] > 0.15 and s10["t"] > 3,
      f"IC {s10['ic']:.3f} t {s10['t']:.1f}")
check("reaction IC at 10 sessions is positive", r10["ic"] > 0.05 and r10["t"] > 2, f"IC {r10['ic']:.3f} t {r10['t']:.1f}")
check("residual sentiment IC at 10 sessions is positive: the text knows what the price did not",
      x10["ic"] > 0.05 and x10["t"] > 2, f"IC {x10['ic']:.3f} t {x10['t']:.1f}")
check("sentiment IC peaks at the planted horizon",
      ic.loc[("sent", 20), "ic"] < s10["ic"] and ic.loc[("sent", 5), "ic"] < s10["ic"],
      f"h5 {ic.loc[('sent', 5), 'ic']:.3f} h10 {s10['ic']:.3f} h20 {ic.loc[('sent', 20), 'ic']:.3f}")
sh = ev.copy()
sh["sent_pct"] = rng.permutation(sh["sent_pct"].to_numpy())
ic_sh = ic_table(sh, fwd, signals=("sent",)).set_index("horizon")
check("shuffled scores carry no IC", abs(ic_sh.loc[10, "ic"]) < 0.05 and abs(ic_sh.loc[10, "t"]) < 2.5,
      f"IC {ic_sh.loc[10, 'ic']:.3f} t {ic_sh.loc[10, 't']:.1f}")
early_fwd = forward_returns(ev, OPEN, SPY, [1], entry=0)
o = OPEN[e0["ticker"]]
check("entry=0 is open(R) to open(R + 1) and entry=1 is open(R + 1) to open(R + 2)",
      abs(early_fwd.loc[100, 1] - (o.iloc[i0 + 1] / o.iloc[i0] - 1)) < 1e-12
      and abs(fwd.loc[100, 1] - (o.iloc[i0 + 2] / o.iloc[i0 + 1] - 1)) < 1e-12)

print("\nRESEARCH PATH\n")
trim = CAL[:-1]
res = {}
for sig in ("sent", "react", "resid"):
    pos = positions(EventDrift(sig, H), panel.loc[trim], trim)
    res[sig] = sharpe(backtest(pos, p.loc[trim]))
shuffled = panel.copy()
col = shuffled["sent_pct"]
shuffled["sent_pct"] = pd.DataFrame(rng.permutation(col.to_numpy().ravel()).reshape(col.shape), index=col.index,
                                    columns=col.columns)
res["placebo"] = sharpe(backtest(positions(EventDrift("sent", H), shuffled.loc[trim], trim), p.loc[trim]))
check("sentiment drift scores well over 1 on the planted panel", res["sent"] > 1.0, f"{res['sent']:.2f}")
check("residual sentiment scores positive net of 10 bps", res["resid"] > 0.3, f"{res['resid']:.2f}")
check("reaction drift scores positive", res["react"] > 0.3, f"{res['react']:.2f}")
check("shuffled sentiment scores below zero after costs", res["placebo"] < 0.0, f"{res['placebo']:.2f}")

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
