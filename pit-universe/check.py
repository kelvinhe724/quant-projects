"""Offline checks: the membership walk on a hand-built change list, symbol
reconciliation, and the comparison harness on synthetic prices with a planted
"winners survive" filter whose bias is known in advance.

Run: python3 check.py
"""
import numpy as np
import pandas as pd

import data
import survivorship as sv

checks = []


def check(name, ok):
    checks.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


print("MEMBERSHIP WALK\n")

days = pd.bdate_range("2020-01-01", "2020-12-31")
chg = pd.DataFrame([
    ("2020-03-02", "NEW1", "OLD1"),
    ("2020-06-01", "NEW2", None),
    ("2020-09-01", None, "OLD2"),
    ("2020-12-15", "NEW3", "NEW1"),
], columns=["date", "add", "rem"]).assign(date=lambda d: pd.to_datetime(d["date"]))
anchors = {pd.Timestamp("2020-01-02"): {"OLD1", "OLD2", "OLD3"}}
panel, snaps = data.membership_panel(chg, anchors, days)

check("a name added on d is absent on d-1 and present on d",
      not panel.loc["2020-02-28", "NEW1"] and panel.loc["2020-03-02", "NEW1"])
check("a removed name is present the day before removal and absent on the day",
      panel.loc["2020-02-28", "OLD1"] and not panel.loc["2020-03-02", "OLD1"])
check("a name never touched stays in for the whole year",
      panel["OLD3"].all())
check("an add-only row and a remove-only row both apply",
      not panel.loc["2020-05-29", "NEW2"] and panel.loc["2020-06-01", "NEW2"]
      and panel.loc["2020-08-31", "OLD2"] and not panel.loc["2020-09-01", "OLD2"])
check("a name added and later removed has exactly one spell",
      len(data.intervals(panel).query("ticker == 'NEW1'")) == 1
      and data.intervals(panel).query("ticker == 'NEW1'").iloc[0]["end"] == pd.Timestamp("2020-12-14"))
check("membership count moves by exactly the net of each change",
      panel.sum(axis=1).loc[["2020-02-28", "2020-03-02", "2020-06-01", "2020-09-01", "2020-12-15"]]
      .tolist() == [3, 3, 4, 3, 3])
check("no anchor corrections when the change list is complete", len(snaps) == 0)

# walking backward from a later anchor must give the same panel
back, _ = data.membership_panel(chg, {pd.Timestamp("2020-12-31"): set(panel.columns[panel.iloc[-1]])}, days)
check("backward walk from the last day reproduces the forward panel",
      back.reindex(columns=panel.columns, fill_value=False).equals(panel))

# an anchor that disagrees with the walk snaps the panel and logs the correction
late = {pd.Timestamp("2020-01-02"): {"OLD1", "OLD2", "OLD3"},
        pd.Timestamp("2020-07-01"): {"OLD2", "OLD3", "NEW1", "NEW2", "MISSED"}}
panel2, snaps2 = data.membership_panel(chg, late, days)
check("a name the change list missed enters at the anchor that shows it",
      not panel2.loc["2020-06-30", "MISSED"] and panel2.loc["2020-07-01", "MISSED"]
      and snaps2["kind"].tolist() == ["missed addition"])

# a change row just before an anchor that the snapshot has not caught up with is trusted
lagged = {pd.Timestamp("2020-01-02"): {"OLD1", "OLD2", "OLD3"},
          pd.Timestamp("2020-06-15"): {"OLD2", "OLD3", "NEW1"}}
panel3, snaps3 = data.membership_panel(chg, lagged, days)
check("a snapshot that lags a change row by less than the tolerance does not undo it",
      panel3.loc["2020-06-15", "NEW2"] and len(snaps3) == 0)

print("\nSYMBOL RECONCILIATION\n")

renames = {"OLD": ("NEW", "2015-01-01"), "NEW": ("NEWER", "2018-01-01")}
check("a mention before the rename maps to the current symbol",
      data.canonical("OLD", "2010-06-01", renames) == "NEWER")
check("a mention after the rename is a reused symbol and stays as written",
      data.canonical("OLD", "2016-01-01", renames) == "OLD")
check("a chain stops at the symbol in force on the date",
      data.canonical("OLD", "2014-12-31", renames) == "NEWER"
      and data.canonical("NEW", "2019-01-01", renames) == "NEW")
check("ticker cleaning strips footnotes and uses the yfinance class-share form",
      data.clean_ticker("BRK.B[1]") == "BRK-B" and data.clean_ticker("NYSE: MMM") == "MMM"
      and data.clean_ticker(float("nan")) is None)

s0 = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "name": ["Alpha Corp", "Beta Inc.", "Gamma"], "sector": "X"})
s1 = pd.DataFrame({"ticker": ["AAA", "BBQ", "DDD"], "name": ["Alpha Corp", "Beta, Inc", "Delta"], "sector": "X"})
snaps_toy = {pd.Timestamp("2020-01-01"): s0, pd.Timestamp("2020-07-01"): s1}
chg_toy = pd.DataFrame({"date": [pd.Timestamp("2020-03-01")], "add": ["DDD"], "rem": ["CCC"]})
auto, unresolved = data.reconcile(snaps_toy, chg_toy)
check("a symbol that vanishes while its security name reappears is a rename",
      auto == {"BBB": ("BBQ", "2020-07-01")})
check("a change the table explains is not reported as unresolved", len(unresolved) == 0)

print("\nCOMPARISON HARNESS ON A PLANTED FILTER\n")

rng = np.random.default_rng(11)
N, T = 240, 252 * 7
idx = pd.bdate_range("2012-01-02", periods=T)
names = [f"S{i:03d}" for i in range(N)]
drift = rng.normal(0.0004, 0.0006, N)
rets = drift + rng.normal(0, 0.015, (T, N))
px = pd.DataFrame(40 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=names)
vol = pd.DataFrame(2e6, index=idx, columns=names)
everyone = pd.DataFrame(True, index=idx, columns=names)

# "winners survive": whoever finished in the top 60% is what a list pulled today would hold
final = px.iloc[-1] / px.iloc[0]
survivors = set(final[final >= final.quantile(0.4)].index)
today = everyone.copy()
today[[c for c in names if c not in survivors]] = False

one = pd.DataFrame(False, index=idx, columns=names)
one.loc["2014-01-02":, "S000"] = True
r1 = sv.equal_weight(px, one)
check("a name that joins on d earns nothing on d and its own return on d+1",
      pd.isna(r1.loc["2014-01-02"]) and np.isclose(r1.loc["2014-01-03"], px["S000"].pct_change().loc["2014-01-03"]))
gap = sv.fill(pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, np.nan, np.nan]}))["x"]
check("fill bridges a gap inside a series and invents nothing after the last print",
      gap.tolist()[:4] == [1.0, 2.0, 2.0, 4.0] and gap.iloc[4:].isna().all())

ew_today = sv.annualised(sv.equal_weight(px, today))
ew_all = sv.annualised(sv.equal_weight(px, everyone))
check(f"survivor basket beats the full basket ({ew_today - ew_all:+.2%}/yr)", ew_today > ew_all)

# selecting on the true drift instead of the realised path plants a gap that can
# be written down before the harness runs: the basket's expected daily simple
# return is mean(exp(mu + sigma^2/2)) - 1 over the names it holds
good = drift >= np.quantile(drift, 0.4)
daily = np.exp(drift + 0.015 ** 2 / 2) - 1
planted = (1 + daily[good].mean()) ** 252 - (1 + daily.mean()) ** 252
known = everyone.copy()
known[[n for n, g in zip(names, good) if not g]] = False
measured = sv.annualised(sv.equal_weight(px, known)) - ew_all
check(f"a filter on true drift reproduces the planted gap ({planted:+.2%}/yr planted, "
      f"{measured:+.2%}/yr measured) within 1%/yr", abs(measured - planted) < 0.01)

sv.mom_data.IS_START, sv.mom_data.TEST_END = "2013-01-02", "2018-12-31"
book_today = sv.momentum_book(px, vol, today)
book_all = sv.momentum_book(px, vol, everyone)
win = ("2013-06-01", "2018-12-31")
row_today = sv.momentum_row(book_today, win, survivors)
row_all = sv.momentum_row(book_all, win, survivors)
table = sv.compare(row_today, row_all)
check("the difference column is today minus point-in-time",
      np.allclose(table["difference (a - b)"], table["today's members"] - table["point-in-time"]))
check("the survivor book never holds a name outside the survivor list, the full book does",
      row_today["share of short book outside today's list"] == 0
      and row_all["share of short book outside today's list"] > 0.2)
check(f"filtering out the losers makes the short leg worse ({table.loc['short leg', 'difference (a - b)']:+.2%}/yr)",
      table.loc["short leg", "difference (a - b)"] < 0)
check("the two books use the same rebalance dates and costs",
      book_today.index.equals(book_all.index)
      and (book_today["traded"] > 0).sum() == (book_all["traded"] > 0).sum())

# a filter that drops names at random carries no bias
random_keep = set(rng.choice(names, int(0.6 * N), replace=False))
random_mask = everyone.copy()
random_mask[[c for c in names if c not in random_keep]] = False
ew_random = sv.annualised(sv.equal_weight(px, random_mask))
check(f"a random filter of the same size shows no level bias ({ew_random - ew_all:+.2%}/yr)",
      abs(ew_random - ew_all) < abs(ew_today - ew_all) / 3)

print(f"\n{sum(checks)}/{len(checks)} passed")
raise SystemExit(0 if all(checks) else 1)
